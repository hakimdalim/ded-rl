"""
Masked Convolution Engine

Handles convolution of signals with masked (gappy) data by:
1. Finding valid regions in the mask
2. Grouping regions by length for efficient batch processing
3. Batch padding and convolving each group
4. Stitching results back together

Example:
    >>> from engines.convolution_engine.padding_engine.padding_engine import PaddingEngine
    >>>
    >>> engine = MaskedConvolutionEngine(
    ...     region_backend='numba',
    ...     convolution_backend='numba'
    ... )
    >>>
    >>> # Setup padding function
    >>> padding_engine = PaddingEngine(backend='numba_numpy')
    >>> pad_func = lambda x, pw: padding_engine.pad(x, pw, mode='reflect')
    >>>
    >>> # Signal with gaps
    >>> signal = np.array([1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan, 11, 12])
    >>> mask = np.isfinite(signal)
    >>> kernel = np.array([0.25, 0.5, 0.25])
    >>>
    >>> # Convolve
    >>> result = engine.convolve(
    ...     signal, kernel, mask,
    ...     pad_width=len(kernel)//2,
    ...     pad_func=pad_func
    ... )
"""

import numpy as np
from typing import Callable, Union, Tuple, Dict, List, Optional
import numpy.typing as npt
from collections import defaultdict

from engines.convolution_engine.region_finding.region_finding_engine import RegionFindingEngine
from engines.convolution_engine.engine import ConvolutionEngine

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class MaskedConvolutionEngine:
    """
    Masked convolution engine with batch-by-length processing.

    Efficiently handles signals with masked/missing data by:
    1. Finding valid contiguous regions
    2. Grouping regions by length
    3. Batch processing each length group (vectorized via backends)
    4. Stitching results back

    Design Philosophy:
        - Single responsibility: only handles masked convolution
        - User controls padding via provided function
        - Leverages existing batch processing backends (Numba/PyTorch)
        - Minimizes Python loops by grouping regions

    Components:
        - RegionFindingEngine: Identifies valid data regions
        - ConvolutionEngine: Performs batch convolution
        - User-provided pad_func: Handles padding strategy

    Performance:
        - Groups regions by length → minimal batch operations
        - Each batch processed via vectorized backend (Numba/PyTorch)
        - Typically 5-10 length groups → 5-10 batch operations total
        - Much faster than per-region loops

    Example:
        >>> engine = MaskedConvolutionEngine()
        >>>
        >>> # User handles padding strategy
        >>> padding_engine = PaddingEngine(backend='numba_numpy')
        >>> pad_func = lambda x, pw: padding_engine.pad(x, pw, mode='reflect')
        >>>
        >>> # Convolve masked signal
        >>> result = engine.convolve(
        ...     signal=signal,
        ...     kernel=kernel,
        ...     mask=mask,
        ...     pad_width=kernel_size//2,
        ...     pad_func=pad_func
        ... )
    """

    def __init__(
        self,
        region_backend: str = 'numba',
        convolution_backend: str = 'numba',
        min_region_length: Optional[int] = None,
    ):
        """
        Initialize masked convolution engine.

        Args:
            region_backend: Backend for region finding
                          'numpy', 'numba' (recommended), 'torch', 'python', 'scipy'
            convolution_backend: Backend for convolution
                               'numpy', 'scipy', 'numba' (recommended), 'pytorch_cpu', 'pytorch_gpu'
            min_region_length: Minimum region length to process (default: kernel_size)
                             If None, will be set to kernel size on first call

        Example:
            >>> # Fast CPU configuration
            >>> engine = MaskedConvolutionEngine(
            ...     region_backend='numba',
            ...     convolution_backend='numba'
            ... )
            >>>
            >>> # Pure NumPy (always works)
            >>> engine = MaskedConvolutionEngine(
            ...     region_backend='numpy',
            ...     convolution_backend='numpy'
            ... )
            >>>
            >>> # GPU configuration
            >>> engine = MaskedConvolutionEngine(
            ...     region_backend='torch',
            ...     convolution_backend='pytorch_gpu'
            ... )
        """
        self.region_backend = region_backend
        self.convolution_backend = convolution_backend
        self.min_region_length = min_region_length

        # Initialize engines
        self.region_engine = RegionFindingEngine(backend=region_backend)
        self.convolution_engine = ConvolutionEngine(backend=convolution_backend)

    def _group_regions_by_length(
        self,
        regions: np.ndarray,
        signal: np.ndarray
    ) -> Dict[int, Tuple[List[int], np.ndarray]]:
        """
        Group regions by their length for batch processing.

        Args:
            regions: Array of shape (n_regions, 2) with [start, end] pairs
            signal: Original signal data

        Returns:
            Dictionary mapping length -> (region_indices, stacked_data)
            where stacked_data is shape (n_regions_with_length, length)

        Example:
            >>> regions = np.array([[0, 3], [5, 8], [10, 13]])
            >>> # Lengths: [3, 3, 3] -> all same length
            >>> groups = _group_regions_by_length(regions, signal)
            >>> # groups[3] = ([0, 1, 2], array([[...], [...], [...]]))
        """
        groups = defaultdict(lambda: ([], []))

        for i in range(len(regions)):
            start, end = regions[i, 0], regions[i, 1]
            length = end - start
            region_data = signal[start:end]

            indices, data_list = groups[length]
            indices.append(i)
            data_list.append(region_data)

        # Stack data for each length group
        result = {}
        for length, (indices, data_list) in groups.items():
            result[length] = (indices, np.stack(data_list))

        return result

    def convolve(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        mask: np.ndarray,
        pad_width: Union[int, Tuple[int, int]],
        pad_func: Callable[[np.ndarray, Union[int, Tuple[int, int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Convolve signal with mask using batch processing.

        Args:
            signal: Input signal (1D array)
            kernel: Convolution kernel (1D array)
            mask: Boolean mask (True = valid, False = masked)
                  Same shape as signal
            pad_width: Padding width for each region
                      int: symmetric padding
                      (left, right): asymmetric padding
            pad_func: Function to pad each batch of regions
                     Signature: pad_func(data_batch, pad_width) -> padded_batch
                     data_batch is 2D: (n_regions, region_length)

        Returns:
            Convolved signal, same shape as input
            Masked regions (mask=False) preserve their original values

        Notes:
            - Output length matches input (like 'same' mode)
            - Regions shorter than kernel are skipped
            - Pad function should handle batched 2D input

        Example:
            >>> from convolution_engine.padding_engine.padding_engine import PaddingEngine
            >>>
            >>> # Setup
            >>> engine = MaskedConvolutionEngine()
            >>> padding_engine = PaddingEngine(backend='numba_numpy')
            >>> pad_func = lambda x, pw: padding_engine.pad(x, pw, mode='reflect')
            >>>
            >>> # Data with gaps
            >>> signal = np.array([1, 2, 3, np.nan, np.nan, 6, 7, 8])
            >>> mask = np.isfinite(signal)
            >>> kernel = np.array([0.25, 0.5, 0.25])
            >>>
            >>> # Convolve
            >>> result = engine.convolve(
            ...     signal, kernel, mask,
            ...     pad_width=1,
            ...     pad_func=pad_func
            ... )
            >>> # result: [?, ?, ?, nan, nan, ?, ?, ?]
        """
        # Validate inputs
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")
        if kernel.ndim != 1:
            raise ValueError(f"Kernel must be 1D, got shape {kernel.shape}")
        if mask.shape != signal.shape:
            raise ValueError(f"Mask shape {mask.shape} must match signal shape {signal.shape}")

        kernel_size = len(kernel)

        # Determine minimum region length
        if self.min_region_length is not None:
            min_length = self.min_region_length
        else:
            min_length = kernel_size

        # Find valid regions
        regions = self.region_engine.find_regions(mask, min_length=min_length)

        # Initialize output by copying input (preserve masked region values)
        # This ensures masked regions keep their original temperature
        output = signal.copy()

        # If no regions found, return original signal unchanged
        if len(regions) == 0:
            return output

        # Group regions by length
        groups = self._group_regions_by_length(regions, signal)

        # Process each length group as a batch
        for length, (region_indices, data_batch) in groups.items():
            # Skip if regions too short
            if length < kernel_size:
                continue

            # Pad batch (all regions at once)
            padded_batch = pad_func(data_batch, pad_width)

            # Convolve batch
            # Check if backend supports 2D (batched) convolution
            supported_dims = self.convolution_engine.supported_dims

            if supported_dims is None or 2 in supported_dims:
                # Backend supports batched convolution (numba, pytorch)
                convolved_batch = self.convolution_engine.convolve(padded_batch, kernel)
            else:
                # Backend only supports 1D (numpy, scipy)
                # Process each padded region individually
                convolved_list = []
                for i in range(padded_batch.shape[0]):
                    convolved_1d = self.convolution_engine.convolve(padded_batch[i], kernel)
                    convolved_list.append(convolved_1d)
                convolved_batch = np.stack(convolved_list)

            # Handle padding effects on output size
            # After padding and valid convolution, we should get back original length
            expected_length = length
            actual_length = convolved_batch.shape[1]

            # Trim or pad result to match expected length
            if actual_length > expected_length:
                # Trim excess (center the result)
                trim_start = (actual_length - expected_length) // 2
                convolved_batch = convolved_batch[:, trim_start:trim_start + expected_length]
            elif actual_length < expected_length:
                # This shouldn't happen with proper padding, but handle it
                raise ValueError(
                    f"Convolved output too short: expected {expected_length}, got {actual_length}. "
                    f"Check pad_width and kernel size."
                )

            # Place results back into output
            for j, region_idx in enumerate(region_indices):
                start, end = regions[region_idx, 0], regions[region_idx, 1]
                output[start:end] = convolved_batch[j]

        return output

    def __call__(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        mask: np.ndarray,
        pad_width: Union[int, Tuple[int, int]],
        pad_func: Callable,
    ) -> np.ndarray:
        """
        Callable interface (shorthand for convolve).

        Args:
            signal: Input signal
            kernel: Convolution kernel
            mask: Boolean mask
            pad_width: Padding width
            pad_func: Padding function

        Returns:
            Convolved signal (masked regions preserve original values)

        Example:
            >>> result = engine(signal, kernel, mask, pad_width=1, pad_func=pad_func)
        """
        return self.convolve(signal, kernel, mask, pad_width, pad_func)

    def get_info(self) -> dict:
        """
        Get information about the engine configuration.

        Returns:
            Dictionary with engine information

        Example:
            >>> engine = MaskedConvolutionEngine()
            >>> info = engine.get_info()
            >>> print(info['region_finding']['backend'])
            'numba'
        """
        return {
            'region_finding': self.region_engine.get_info(),
            'convolution': {
                'backend': self.convolution_engine.backend_name,
            },
            'configuration': {
                'min_region_length': self.min_region_length,
            }
        }

    @staticmethod
    def create_fast() -> 'MaskedConvolutionEngine':
        """
        Create engine with fast backends (Numba).

        Returns:
            MaskedConvolutionEngine with Numba backends

        Example:
            >>> engine = MaskedConvolutionEngine.create_fast()
            >>> # Uses numba for region finding and convolution
        """
        return MaskedConvolutionEngine(
            region_backend='numba',
            convolution_backend='numba'
        )

    @staticmethod
    def create_reliable() -> 'MaskedConvolutionEngine':
        """
        Create engine with reliable backends (NumPy).

        Always works, no additional dependencies.

        Returns:
            MaskedConvolutionEngine with NumPy backends

        Example:
            >>> engine = MaskedConvolutionEngine.create_reliable()
            >>> # Uses numpy for all operations
        """
        return MaskedConvolutionEngine(
            region_backend='numpy',
            convolution_backend='numpy'
        )

    @staticmethod
    def create_gpu() -> 'MaskedConvolutionEngine':
        """
        Create engine with GPU backends (PyTorch).

        Requires PyTorch with CUDA.

        Returns:
            MaskedConvolutionEngine with GPU backends

        Example:
            >>> engine = MaskedConvolutionEngine.create_gpu()
            >>> # Uses pytorch_gpu for convolution
        """
        return MaskedConvolutionEngine(
            region_backend='torch',
            convolution_backend='pytorch_gpu'
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MaskedConvolutionEngine(\n"
            f"  region_backend='{self.region_backend}',\n"
            f"  convolution_backend='{self.convolution_backend}'\n"
            f")"
        )