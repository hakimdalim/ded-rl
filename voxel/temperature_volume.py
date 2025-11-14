import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import Callable, Dict, Union, Tuple

from voxel.activated_volume import extract_volume_coordinates
from utils.masked_gaussian_filter_3D import apply_masked_gaussian_3d
from utils.field_utils import grid_slice
from utils.field_visualization import ThermalPlotter
from laser.temp_field_func_wrapper import TemperatureFieldWrapper
from utils.field_boundary_dimension_search import find_symmetric_boundary_dimensions
from utils.visualization_utils import find_surface, sample_at_surface, ensure_ax

from voxel.heat_diffusion_conv import HeatDiffusionConvolution

class MeltPoolBoundaryError(ValueError):
    """Exception raised when melt pool extends beyond simulation domain boundaries."""
    pass

class TrackTemperature:

    """Tracks temperature evolution in a voxel volume."""

    def __init__(
            self,
            shape: Tuple[int, int, int],
            voxel_size: Union[float, Tuple[float, float, float]],
            ambient_temp: float = 300.0,
            substrate_height: float = 0.0
    ):
        """Initialize temperature tracking volume.

        Args:
            shape: Grid dimensions (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)
            ambient_temp: Ambient temperature in Celsius
            substrate_height: Physical substrate height in meters
        """
        # Standardize voxel_size
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        self.voxel_size = np.array(voxel_size, dtype=float)

        self.shape = shape
        self.ambient_temp = ambient_temp

        # Calculate substrate thickness in voxels
        substrate_nz = int(np.ceil(substrate_height / self.voxel_size[2])) if substrate_height > 0 else 0

        # Initialize temperature array
        self.temperature = np.full(shape=shape, fill_value=ambient_temp, dtype=float)

        # Initialize heat diffusion engine
        self.diffusion_engine = HeatDiffusionConvolution.create_fast(
            substrate_thickness=substrate_nz,
            ambient_temperature=ambient_temp
        )

    def reset(self):
        """Reset temperature field to ambient temperature."""
        self.temperature.fill(self.ambient_temp)

    def apply_heat_source(
            self,
            get_temp: Callable,
            t: float,
            params: Dict,
            start_position: Tuple[float, float, float],
            movement_angle: float,
    ) -> None:
        """
        Apply temperature changes from a heat source to a region around the current position.

        Args:
            get_temp: Callable: Function that returns temperature for given points, time, and params.
            t: float: Current time in seconds.
            params: Dict: Dictionary containing simulation parameters.
            start_position: Tuple[float, float, float]: Initial (x, y, z) position of heat source in physical units.
            movement_angle: float: Angle of movement direction around z-axis in radians.
        """

        # Create temperature field wrapper to handle moving heat source
        temp_field = TemperatureFieldWrapper(
            get_temp=get_temp,
            t=t,
            params=params,
            start_position=start_position,
            movement_angle=movement_angle
        )

        # Get current heat source position from wrapper
        center = temp_field.heat_source_position

        # Calculate cutoff radii using Wolfer eq. 13
        # Standard factor 'a' (paper uses a=4 for 99.99% of infinite integral)
        a = 4.0

        # Get parameters needed for calculations
        D = params['thermal_diffusivity']  # m²/s
        sigma = params.get('laser_beam_radius', params.get('beam_waist_radius'))  # m
        V = params['scan_speed']  # m/s

        # Calculate radii (from Wolfer eq. 13)
        r_xf = a * np.sqrt(sigma ** 2 + 2 * D * t)  # Forward x radius
        r_xr = a * np.sqrt(sigma ** 2 + 2 * D * t) + V * t  # Rear x radius (includes travel distance)
        r_y = a * np.sqrt(sigma ** 2 + 2 * D * t)  # y radius
        r_z = a * np.sqrt(2 * D * t)  # z radius

        # Use the larger x radius for the extraction
        half_width = max(r_xf, r_xr)
        half_length = r_y
        half_depth = r_z

        # Extract coordinates using helper function
        vol_coords = extract_volume_coordinates(
            center=center,
            half_width=half_width,
            half_length=half_length,
            half_depth=half_depth,
            volume_shape=self.shape,
            voxel_size=self.voxel_size
        )

        # Get the points grid and calculate temperatures using wrapper
        points = vol_coords['points_grid']
        temperatures = temp_field(points)  # Use wrapper to get temperatures

        # Update temperature array using the extracted indices
        x_min, x_max, y_min, y_max, z_min, z_max = vol_coords['voxel_indices']
        self.temperature[x_min:x_max, y_min:y_max, z_min:z_max] += temperatures

    def reset_deactivated(self, activation_mask: NDArray[np.bool_]) -> None:
        """Reset temperatures of deactivated voxels to ambient temperature.

        Args:
            activation_mask: Boolean array matching temperature array shape
        """
        self.temperature[~activation_mask] = self.ambient_temp

    def apply_diffusion(self, activation_mask: NDArray[np.bool_], sigma: float) -> None:
        """Apply thermal diffusion with substrate boundary conditions.

        Uses heat diffusion convolution engine with proper substrate BC.
        Implements: ∂T/∂t = D∇²T via separable 3D convolution.

        Args:
            activation_mask: Boolean array indicating active voxels, must match temperature array shape
            sigma: Gaussian sigma for diffusion in PHYSICAL units (meters)
                   Calculated as σ = √(2D∆t) where D is thermal diffusivity

        Raises:
            ValueError: If activation mask shape doesn't match temperature shape
            ValueError: If voxel sizes are not uniform (anisotropic grid not supported)
            ValueError: If kernel size is too small

        Note:
            - Uses substrate boundary conditions (heat sink at z=0)
            - Only activated voxels participate in diffusion
            - Temperature field is modified in-place
            - Non-activated voxels preserve original temperature values
        """
        # Input validation
        if activation_mask.shape != self.temperature.shape:
            raise ValueError("Activation mask shape must match temperature array shape")

        if not np.any(activation_mask):
            return  # Nothing to do if no voxels are activated

        # Check for uniform voxel size (required for isotropic diffusion)
        if not np.allclose(self.voxel_size, self.voxel_size[0]):
            raise ValueError(
                f"Non-uniform voxel sizes detected: {self.voxel_size}. "
                f"Heat diffusion requires uniform voxel grid (isotropic discretization)."
            )

        # Convert sigma to voxel units
        sigma_voxels = sigma / self.voxel_size[0]

        # Create Gaussian kernel (±3σ coverage)
        kernel_size = int(6 * sigma_voxels) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Validate kernel size
        if kernel_size < 3:
            raise ValueError(
                f"Kernel size too small ({kernel_size}). "
                f"Increase sigma or use smaller voxels. "
                f"sigma={sigma:.6f}m, voxel_size={self.voxel_size[0]:.6f}m, "
                f"sigma_voxels={sigma_voxels:.3f}"
            )

        # Build normalized Gaussian kernel
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma_voxels) ** 2)
        kernel /= kernel.sum()

        # Apply diffusion with substrate boundary conditions
        self.temperature = self.diffusion_engine.diffuse_step(
            temperature=self.temperature,
            mask=activation_mask,
            kernel=kernel,
            pad_width=kernel_size // 2
        )

    def get_melt_pool_dimensions(self, params) -> Dict[str, float]:
        """Calculate melt pool dimensions using vectorized interpolation for sub-voxel accuracy."""
        # Check if any cells are above melting temperature
        melt_mask = self.temperature >= params['melting_temp']
        max_temp_idx = np.unravel_index(np.argmax(self.temperature), self.temperature.shape)
        x_center, y_center, z_center = max_temp_idx

        if not np.any(melt_mask):
            return {
                'width': 0.0,
                'length': 0.0,
                'depth': 0.0,
                'max_temp': np.max(self.temperature),
                'voxel_center': (x_center, y_center, z_center)
            }

        # Check if melt pool extends to domain boundaries
        molten_voxels = np.argwhere(melt_mask)
        if len(molten_voxels) > 0:
            x_min, y_min, z_min = molten_voxels.min(axis=0)
            x_max, y_max, z_max = molten_voxels.max(axis=0)

            boundaries_hit = []
            if x_min == 0: boundaries_hit.append("X_min")
            if x_max == self.temperature.shape[0] - 1: boundaries_hit.append("X_max")
            if y_min == 0: boundaries_hit.append("Y_min")
            if y_max == self.temperature.shape[1] - 1: boundaries_hit.append("Y_max")
            if z_min == 0: boundaries_hit.append("Z_min")
            if z_max == self.temperature.shape[2] - 1: boundaries_hit.append("Z_max")

            if boundaries_hit:
                raise MeltPoolBoundaryError(
                    f"Melt pool extends to domain boundary: {boundaries_hit}. "
                    f"Max temp: {np.max(self.temperature):.1f}K at voxel ({x_center},{y_center},{z_center}). "
                    f"Domain too small - increase size."
                )

        # Find coordinates of maximum temperature

        melting_temp = params['melting_temp']

        # Create coordinate arrays and get temperature profiles in one go
        profiles = {
            'width': (
                self.temperature[:, y_center, z_center],
                np.arange(self.temperature.shape[0]) * self.voxel_size[0]
            ),
            'length': (
                self.temperature[x_center, :, z_center],
                np.arange(self.temperature.shape[1]) * self.voxel_size[1]
            ),
            'depth': (
                self.temperature[x_center, y_center, :],
                np.arange(self.temperature.shape[2]) * self.voxel_size[2]
            )
        }

        dimensions = {
            'max_temp': np.max(self.temperature),
            'voxel_center':  (x_center, y_center, z_center)
        }

        # Vectorized calculation for all dimensions
        for dim, (profile, coords) in profiles.items():
            # Find transitions
            above_melt = profile >= melting_temp
            transitions = np.where(above_melt[1:] != above_melt[:-1])[0]

            if len(transitions) >= 2:
                # Get interpolation points for both boundaries
                left_idx, right_idx = transitions[0], transitions[-1] + 1

                # Separate interpolation for each boundary
                left_boundary = np.interp(
                    melting_temp,
                    [profile[left_idx], profile[left_idx + 1]],
                    [coords[left_idx], coords[left_idx + 1]]
                )

                right_boundary = np.interp(
                    melting_temp,
                    [profile[right_idx - 1], profile[right_idx]],
                    [coords[right_idx - 1], coords[right_idx]]
                )

                dimensions[dim] = abs(right_boundary - left_boundary)
            else:
                dimensions[dim] = 0.0

        return dimensions

    @ensure_ax
    def plot_temperature_slice(
            self,
            ax: plt.Axes = None,
            plane: str = 'xy',
            slice_idx: int = None,
            temp_range: Tuple[float, float] = None,
            interpolation: str = 'nearest',
            show_melt_pool: bool = True,
            show_grid: bool = True,
            legend_loc: str = 'lower right',
            title: str = None,
            melting_temp: float = None,
            maybe_plot_direction: bool = True,
            cmap: str = 'viridis',
    ) -> plt.colorbar:
        """
        Create a visualization of the temperature field in a specified plane.
        """

        # Extract the appropriate slice based on plane orientation
        axis_idx = {'xy': 2, 'xz': 1, 'yz': 0}[plane]

        if slice_idx is None:
            slice_idx = self.shape[axis_idx] // 2
        if slice_idx < 0 or slice_idx >= self.shape[axis_idx]:
            raise ValueError(f"slice_idx must be within the range of the {plane.upper()} plane.")

        if plane == 'xy':
            temp_slice = self.temperature[:, :, slice_idx]
            extent = [0, self.shape[0] * self.voxel_size[0] * 1000,
                      0, self.shape[1] * self.voxel_size[1] * 1000]
            xlabel, ylabel = 'X (mm)', 'Y (mm)'

        elif plane == 'xz':
            temp_slice = self.temperature[:, slice_idx, :]
            extent = [0, self.shape[0] * self.voxel_size[0] * 1000,
                      0, self.shape[2] * self.voxel_size[2] * 1000]
            xlabel, ylabel = 'X (mm)', 'Z (mm)'

        elif plane == 'yz':
            temp_slice = self.temperature[slice_idx, :, :]
            extent = [0, self.shape[1] * self.voxel_size[1] * 1000,
                      0, self.shape[2] * self.voxel_size[2] * 1000]
            xlabel, ylabel = 'Y (mm)', 'Z (mm)'

        # Use imshow for visualization
        img = ax.imshow(
            temp_slice.T,
            origin='lower',
            extent=extent,
            interpolation=interpolation,
            aspect='equal',
            vmin=temp_range[0] if temp_range else None,
            vmax=temp_range[1] if temp_range else None,
            cmap=cmap,
        )
        colorbar = plt.colorbar(img, ax=ax, label='Temperature (K)')

        # Add melt pool boundary if requested
        if show_melt_pool and melting_temp is not None and np.min(temp_slice) <= melting_temp <= np.max(temp_slice):
            ax.contour(
                np.linspace(extent[0], extent[1], temp_slice.shape[0]),
                np.linspace(extent[2], extent[3], temp_slice.shape[1]),
                temp_slice.T,
                levels=[melting_temp],
                colors='r',
                linewidths=1,
                label='Melt Pool Boundary'
            )

        # Configure plot aesthetics
        ax.set_title(title or f'{plane.upper()} Plane Temperature')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.3)

        if show_melt_pool and melting_temp is not None:
            ax.legend(loc=legend_loc)

        return colorbar


'''if __name__ == "__main__":
    from process_parameters import set_params
    from thermal.temperature_change import EagarTsaiTemperature
    from thermal.temp_field_func_wrapper import TemperatureFieldWrapper
    import matplotlib.pyplot as plt

    # Initialize with typical simulation parameters
    params = set_params()

    # Common visualization parameters
    temp_range = (300, 2500)  # K
    volume_shape = (500, 500, 250)  # 25mm x 25mm x 12.5mm
    voxel_size = 0.0001  # 100 microns

    times = [0.05, 0.1, 0.5, 1.0]  # seconds
    #times = [1.0]  # seconds

    plt.suptitle('Temperature Field Evolution Over Time')

    for i, t in enumerate(times):
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reset volume for each time point
        volume = TrackTemperature(
            shape=volume_shape,
            voxel_size=voxel_size,
            ambient_temp=300.0
        )

        temperature_field = EagarTsaiTemperature()

        # Apply initial temperature field
        volume.apply_heat_source(
            get_temp=temperature_field.delta_temperature,
            t=t,
            params=params,
            start_position=(0.002, 0.005, 0.001),
            movement_angle=0
        )

        # Visualize initial temperature field
        volume.plot_temperature_slice(
            ax=ax1,
            plane='xy',
            temp_range=temp_range,
            interpolation='bilinear',
            show_melt_pool=True,
            show_grid=True,
            title=f'Initial Temperature Field\nTime: {t * 1000:.0f}ms',
            melting_temp=params['melting_temp'],
            slice_idx=int(0.001 / volume.voxel_size[2]),
        )

        # Create activation mask based on temperature threshold
        activation_mask = volume.temperature > 300
        print(activation_mask.sum())

        # Calculate diffusion sigma based on thermal diffusivity and time step
        dt = 0.01  # 10ms diffusion time step
        diffusion_sigma = np.sqrt(2 * params['thermal_diffusivity'] * dt)
        print(diffusion_sigma)

        print(volume.temperature.max())

        # Apply diffusion
        volume.apply_diffusion(activation_mask, diffusion_sigma)

        print(volume.temperature.max())

        # Visualize temperature field after diffusion
        volume.plot_temperature_slice(
            ax=ax2,
            plane='xy',
            temp_range=temp_range,
            interpolation='bilinear',
            show_melt_pool=True,
            show_grid=True,
            title=f'After Diffusion (dt={dt * 1000:.1f}ms)\nTime: {t * 1000:.0f}ms',
            melting_temp=params['melting_temp'],
            slice_idx=int(0.001 / volume.voxel_size[2]),
        )

        plt.tight_layout()
        plt.show()

        volume.plot_temperature_slice(slice_idx=int(0.001 / volume.voxel_size[2]))'''