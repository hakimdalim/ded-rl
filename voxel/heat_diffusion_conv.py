"""
Heat Diffusion Convolution Engine

Performs single-step 3D heat diffusion using separable 1D convolutions.
Handles complex boundary conditions with proper layering:
1. Substrate boundary (zero-temperature BC via negation)
2. Air boundaries (zero-flux BC via reflection)
3. Internal gaps (masked convolution)

Physics:
- Substrate at z=0: Zero-temperature BC (heat sink)
- Air boundaries (x, y, z_top): Zero-flux BC (insulated)
- Internal gaps: Handled via masked convolution

Example:
    >>> engine = HeatDiffusionConvolution(
    ...     substrate_thickness=10,
    ...     ambient_temperature=293.15,
    ...     convolution_backend='numba'
    ... )
    >>>
    >>> T_new = engine.diffuse_step(
    ...     temperature=T_field,
    ...     mask=activated_mask,
    ...     kernel=gaussian_kernel,
    ...     pad_width=kernel_size//2
    ... )
"""

import numpy as np
from typing import Union, Tuple, Optional
import numpy.typing as npt

from engines.convolution_engine.masked_convolution_engine import MaskedConvolutionEngine
from engines.convolution_engine.padding_engine.padding_engine import PaddingEngine


class SubstrateBoundaryPadding:
    """Zero-temperature BC at substrate base (bottom of last axis)."""

    def __call__(
        self,
        temperature: npt.NDArray[np.float64],
        substrate_thickness: int,
        pad_width: int
    ) -> npt.NDArray[np.float64]:
        """Pad with substrate boundary conditions.

        Substrate is always at bottom of last axis (z-axis for shape (nx,ny,nz)).

        Args:
            temperature: Temperature field (nx, ny, nz)
            substrate_thickness: Substrate thickness in voxels
            pad_width: Padding width in voxels

        Returns:
            Padded temperature with BCs applied
        """
        # Single pad: bottom of z-axis, both sides of x and y
        pad_config = [(pad_width, pad_width), (pad_width, pad_width), (pad_width, 0)]

        T_padded = np.pad(temperature, pad_config, mode='reflect')

        # Negate substrate regions (enforce T=0 at boundaries)
        # Bottom z (entire bottom padding)
        T_padded[:, :, :pad_width] *= -1

        # Left/right x-sides (only substrate region)
        T_padded[:pad_width, :, pad_width:pad_width+substrate_thickness] *= -1
        T_padded[-pad_width:, :, pad_width:pad_width+substrate_thickness] *= -1

        # Left/right y-sides (only substrate region)
        T_padded[:, :pad_width, pad_width:pad_width+substrate_thickness] *= -1
        T_padded[:, -pad_width:, pad_width:pad_width+substrate_thickness] *= -1

        return T_padded


class HeatDiffusionConvolution:
    """
    3D heat diffusion via separable 1D convolutions.

    Performs a single time step of heat diffusion using three sequential
    1D convolutions (x, y, z) with proper boundary conditions.

    Boundary Conditions (Layered):
    1. Substrate (z=0): Zero-temperature BC (heat sink)
       - Implemented via reflection + negation in SubstrateBoundaryPadding
    2. Air boundaries: Zero-flux BC (insulated)
       - Implemented via symmetric reflection
    3. Internal gaps: Masked convolution handles missing voxels

    Key Design:
    - Separable convolution: 3D → 3× 1D (much faster)
    - Layered BCs: Substrate BC first, then masked convolution respects it
    - Mask extension: Substrate-padded regions marked as valid
    - Configurable backends for performance tuning

    Example:
        >>> # Setup
        >>> engine = HeatDiffusionConvolution(
        ...     substrate_thickness=10,
        ...     ambient_temperature=293.15
        ... )
        >>>
        >>> # Single diffusion step
        >>> T_diffused = engine.diffuse_step(
        ...     temperature=T,
        ...     mask=activated_voxels,
        ...     kernel=gaussian_kernel_1d,
        ...     pad_width=3
        ... )
        >>>
        >>> # Time stepping (external)
        >>> for step in range(n_steps):
        ...     T = engine.diffuse_step(T, mask, kernel, pad_width)
    """

    def __init__(
        self,
        substrate_thickness: int,
        ambient_temperature: float = 293.15,
        convolution_backend: str = 'numba',
        padding_backend: str = 'numba_numpy',
        region_backend: str = 'numba'
    ):
        """
        Initialize heat diffusion engine.

        Args:
            substrate_thickness: Substrate thickness in voxels
            ambient_temperature: Ambient temperature in Kelvin (default: 293.15K = 20°C)
            convolution_backend: Backend for convolution
                               'numba' (recommended), 'numpy', 'scipy', 'pytorch_cpu', 'pytorch_gpu'
            padding_backend: Backend for padding
                           'numba_numpy' (recommended), 'numpy_native', 'numba_pure',
                           'pytorch_cpu', 'pytorch_gpu', 'custom_vectorized'
            region_backend: Backend for region finding
                          'numba' (recommended), 'numpy', 'torch', 'python', 'scipy'

        Example:
            >>> # Fast CPU configuration
            >>> engine = HeatDiffusionConvolution(
            ...     substrate_thickness=10,
            ...     convolution_backend='numba',
            ...     padding_backend='numba_numpy',
            ...     region_backend='numba'
            ... )
            >>>
            >>> # Reliable (always works)
            >>> engine = HeatDiffusionConvolution(
            ...     substrate_thickness=10,
            ...     convolution_backend='numpy',
            ...     padding_backend='numpy_native',
            ...     region_backend='numpy'
            ... )
            >>>
            >>> # GPU configuration
            >>> engine = HeatDiffusionConvolution(
            ...     substrate_thickness=10,
            ...     convolution_backend='pytorch_gpu',
            ...     padding_backend='pytorch_gpu',
            ...     region_backend='torch'
            ... )
        """
        self.substrate_thickness = substrate_thickness
        self.T_ambient = ambient_temperature

        # Create engines
        self.masked_conv_engine = MaskedConvolutionEngine(
            region_backend=region_backend,
            convolution_backend=convolution_backend
        )
        self.padding_engine = PaddingEngine(backend=padding_backend)
        self.substrate_padder = SubstrateBoundaryPadding()

    def _extend_mask_for_substrate_bc(
        self,
        mask: np.ndarray,
        pad_width: int
    ) -> np.ndarray:
        """
        Extend mask to include substrate-padded regions.

        Critical: Only mark regions where SubstrateBoundaryPadding applied BC.
        This prevents masked convolution from overwriting substrate BC.

        Regions marked as valid:
        - Bottom z: Full bottom padding (substrate base)
        - x-sides: Only substrate thickness (not full height = air!)
        - y-sides: Only substrate thickness (not full height = air!)

        Args:
            mask: Original mask (nx, ny, nz)
            pad_width: Padding width

        Returns:
            Extended mask with substrate BC regions marked as valid
        """
        # Pad mask with False (air)
        pad_config = [(pad_width, pad_width), (pad_width, pad_width), (pad_width, 0)]
        mask_extended = np.pad(mask, pad_config, mode='constant', constant_values=False)

        # Mark substrate BC regions as valid (matches SubstrateBoundaryPadding)
        # 1. Bottom z (entire bottom padding) - substrate everywhere
        mask_extended[:, :, :pad_width] = True

        # 2. Left/right x-sides (ONLY substrate thickness, not full height!)
        mask_extended[:pad_width, :, pad_width:pad_width+self.substrate_thickness] = True
        mask_extended[-pad_width:, :, pad_width:pad_width+self.substrate_thickness] = True

        # 3. Left/right y-sides (ONLY substrate thickness, not full height!)
        mask_extended[:, :pad_width, pad_width:pad_width+self.substrate_thickness] = True
        mask_extended[:, -pad_width:, pad_width:pad_width+self.substrate_thickness] = True

        # Above substrate = air = leave as False
        # Will get zero-flux BC from masked convolution

        return mask_extended

    def _convolve_axis(
        self,
        field: np.ndarray,
        mask: np.ndarray,
        kernel: np.ndarray,
        pad_width: int,
        axis: int
    ) -> np.ndarray:
        """
        Convolve along a single axis with masked convolution.

        Handles internal gaps via masked convolution.
        Zero-flux BC (symmetric reflection) for air boundaries.

        Args:
            field: Temperature field (3D)
            mask: Validity mask (3D)
            kernel: 1D convolution kernel
            pad_width: Padding width
            axis: Axis to convolve along (0=x, 1=y, 2=z)

        Returns:
            Convolved field along specified axis
        """
        # Create padding function for air boundaries (zero-flux BC)
        pad_func = lambda x, pw: self.padding_engine.pad(x, pw, mode='symmetric')

        # Move target axis to last position
        field_moved = np.moveaxis(field, axis, -1)
        mask_moved = np.moveaxis(mask, axis, -1)

        # Get shape
        shape = field_moved.shape
        n_slices = shape[0] * shape[1]  # Total number of 1D signals
        signal_length = shape[2]

        # Reshape to 2D: (n_slices, signal_length)
        field_2d = field_moved.reshape(n_slices, signal_length)
        mask_2d = mask_moved.reshape(n_slices, signal_length)

        # Convolve each slice with masked convolution
        result_2d = np.zeros_like(field_2d)
        for i in range(n_slices):
            result_2d[i] = self.masked_conv_engine.convolve(
                signal=field_2d[i],
                kernel=kernel,
                mask=mask_2d[i],
                pad_width=pad_width,
                pad_func=pad_func
            )

        # Reshape back to 3D
        result_moved = result_2d.reshape(shape)

        # Move axis back to original position
        result = np.moveaxis(result_moved, -1, axis)

        return result

    def diffuse_step(
        self,
        temperature: np.ndarray,
        mask: np.ndarray,
        kernel: np.ndarray,
        pad_width: int
    ) -> np.ndarray:
        """
        Perform single 3D heat diffusion step.

        Workflow:
        1. Subtract ambient → work with ΔT
        2. Apply substrate BC padding (zero-temp at z=0)
        3. Extend mask to include substrate-padded regions
        4. Convolve along x-axis (with zero-flux BC for air)
        5. Convolve along y-axis (with zero-flux BC for air)
        6. Convolve along z-axis (with zero-flux BC for air)
        7. Extract valid region (remove padding)
        8. Add ambient back → final temperature
        9. Restore masked region values (gaps don't conduct heat)

        Args:
            temperature: Temperature field (nx, ny, nz) in Kelvin
            mask: Valid voxels (nx, ny, nz), True=valid, False=gap/air
            kernel: 1D diffusion kernel (typically Gaussian)
                   Same kernel used for all three axes
            pad_width: Padding width (typically kernel_size//2)

        Returns:
            Diffused temperature field (nx, ny, nz) in Kelvin
            Masked regions (mask=False) are restored to original values

        Example:
            >>> # Create Gaussian kernel
            >>> sigma = 1.0
            >>> kernel_size = 7
            >>> x = np.arange(kernel_size) - kernel_size//2
            >>> kernel = np.exp(-0.5 * (x/sigma)**2)
            >>> kernel /= kernel.sum()
            >>>
            >>> # Single diffusion step
            >>> T_new = engine.diffuse_step(
            ...     temperature=T,
            ...     mask=activated_voxels,
            ...     kernel=kernel,
            ...     pad_width=kernel_size//2
            ... )
        """
        # Validate inputs
        if temperature.ndim != 3:
            raise ValueError(f"Temperature must be 3D, got shape {temperature.shape}")
        if mask.shape != temperature.shape:
            raise ValueError(f"Mask shape {mask.shape} must match temperature shape {temperature.shape}")
        if kernel.ndim != 1:
            raise ValueError(f"Kernel must be 1D, got shape {kernel.shape}")

        # Step 1: Work with ΔT = T - T_ambient
        dT = temperature - self.T_ambient

        # Step 2: Apply substrate BC (zero-temp at z=0 via negation)
        dT_padded = self.substrate_padder(
            dT,
            substrate_thickness=self.substrate_thickness,
            pad_width=pad_width
        )

        # Step 3: Extend mask to include substrate-padded regions
        mask_extended = self._extend_mask_for_substrate_bc(mask, pad_width)

        # Step 4-6: Convolve along each axis (separable 3D convolution)
        # Each convolution respects:
        # - Substrate BC (via extended mask)
        # - Air BC (via zero-flux reflection in pad_func)
        # - Internal gaps (via masked convolution)

        dT_x = self._convolve_axis(dT_padded, mask_extended, kernel, pad_width, axis=0)
        dT_xy = self._convolve_axis(dT_x, mask_extended, kernel, pad_width, axis=1)
        dT_xyz = self._convolve_axis(dT_xy, mask_extended, kernel, pad_width, axis=2)

        # Step 7: Extract valid region (remove padding)
        dT_diffused = dT_xyz[
            pad_width:-pad_width,
            pad_width:-pad_width,
            pad_width:  # No padding on top of z
        ]

        # Step 8: Add ambient back
        T_diffused = dT_diffused + self.T_ambient

        # Step 9: Restore masked region values (gaps don't conduct heat)
        # This ensures mask=False regions keep their original temperature
        T_diffused[~mask] = temperature[~mask]

        return T_diffused

    def get_info(self) -> dict:
        """
        Get information about engine configuration.

        Returns:
            Dictionary with engine info

        Example:
            >>> info = engine.get_info()
            >>> print(info['convolution']['backend'])
            'numba'
        """
        return {
            'substrate_thickness': self.substrate_thickness,
            'ambient_temperature': self.T_ambient,
            'masked_convolution': self.masked_conv_engine.get_info(),
            'padding': {
                'backend': self.padding_engine.backend
            }
        }

    @staticmethod
    def create_fast(substrate_thickness: int, ambient_temperature: float = 293.15) -> 'HeatDiffusionConvolution':
        """
        Create engine with fast backends (Numba).

        Args:
            substrate_thickness: Substrate thickness in voxels
            ambient_temperature: Ambient temperature in Kelvin

        Returns:
            HeatDiffusionConvolution with Numba backends

        Example:
            >>> engine = HeatDiffusionConvolution.create_fast(substrate_thickness=10)
        """
        return HeatDiffusionConvolution(
            substrate_thickness=substrate_thickness,
            ambient_temperature=ambient_temperature,
            convolution_backend='numba',
            padding_backend='numba_numpy',
            region_backend='numba'
        )

    @staticmethod
    def create_reliable(substrate_thickness: int, ambient_temperature: float = 293.15) -> 'HeatDiffusionConvolution':
        """
        Create engine with reliable backends (NumPy).

        Always works, no additional dependencies.

        Args:
            substrate_thickness: Substrate thickness in voxels
            ambient_temperature: Ambient temperature in Kelvin

        Returns:
            HeatDiffusionConvolution with NumPy backends

        Example:
            >>> engine = HeatDiffusionConvolution.create_reliable(substrate_thickness=10)
        """
        return HeatDiffusionConvolution(
            substrate_thickness=substrate_thickness,
            ambient_temperature=ambient_temperature,
            convolution_backend='numpy',
            padding_backend='numpy_native',
            region_backend='numpy'
        )

    @staticmethod
    def create_gpu(substrate_thickness: int, ambient_temperature: float = 293.15) -> 'HeatDiffusionConvolution':
        """
        Create engine with GPU backends (PyTorch).

        Requires PyTorch with CUDA.

        Args:
            substrate_thickness: Substrate thickness in voxels
            ambient_temperature: Ambient temperature in Kelvin

        Returns:
            HeatDiffusionConvolution with GPU backends

        Example:
            >>> engine = HeatDiffusionConvolution.create_gpu(substrate_thickness=10)
        """
        return HeatDiffusionConvolution(
            substrate_thickness=substrate_thickness,
            ambient_temperature=ambient_temperature,
            convolution_backend='pytorch_gpu',
            padding_backend='pytorch_gpu',
            region_backend='torch'
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HeatDiffusionConvolution(\n"
            f"  substrate_thickness={self.substrate_thickness},\n"
            f"  ambient_temperature={self.T_ambient}K,\n"
            f"  convolution_backend='{self.masked_conv_engine.convolution_backend}',\n"
            f"  padding_backend='{self.padding_engine.backend}',\n"
            f"  region_backend='{self.masked_conv_engine.region_backend}'\n"
            f")"
        )