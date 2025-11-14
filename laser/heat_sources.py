from typing import Dict, Union, Protocol, Callable, Type
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from powder.powder_stream import YuzeHuangPowderStream
from utils.vectorize_inputs import prepare_points


class PowderStreamProtocol(Protocol):
    """Protocol defining required powder stream calculation methods"""

    @staticmethod
    def number_concentration(points: np.ndarray, params: Dict) -> np.ndarray: ...

    @staticmethod
    def powder_concentration(points: np.ndarray, params: Dict) -> np.ndarray: ...

    @staticmethod
    def is_within_powder_stream_radius(points: np.ndarray, params: Dict) -> np.ndarray: ...

    @staticmethod
    def powder_stream_boundary_at_z(points: np.ndarray, params: Dict) -> np.ndarray: ...


@dataclass
class PowderStreamFunctions:
    """Container for powder stream calculation functions"""
    number_concentration: Callable[[np.ndarray, Dict], np.ndarray]
    powder_concentration: Callable[[np.ndarray, Dict], np.ndarray]
    is_within_powder_stream_radius: Callable[[np.ndarray, Dict], np.ndarray]
    powder_stream_boundary_at_z: Callable[[np.ndarray, Dict], np.ndarray]


class YuzeHuangHeatSources:
    """
    Calculates various intensity and energy-related quantities for laser directed energy
    deposition processes according to Huang et al. (2016).

    The class accepts either a class with static methods or a PowderStreamFunctions instance
    containing the required calculation functions.
    """

    def __init__(
            self,
            powder_stream_calculator: Union[Type[PowderStreamProtocol], PowderStreamFunctions] = YuzeHuangPowderStream
    ):
        """
        Initialize intensity calculator with configurable powder stream model.

        Args:
            powder_stream_calculator: Either a class implementing the required static methods
                                   or a PowderStreamFunctions instance containing the
                                   calculation functions.
                                   Defaults to YuzeHuangPowderStream implementation.
        """
        self.powder_stream = powder_stream_calculator

    def laser_intensity(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate the Gaussian laser beam intensity at given points using equation (6)
        from Huang et al. (2016).

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Process and material parameters including:
                   - laser_power: Laser power (W)
                   - beam_waist_radius: Radius at beam waist (m)
                   - beam_divergence_angle: Far-field divergence angle (rad)
                   - beam_waist_position: Z position of beam waist (m)

        Returns:
            Array of laser beam intensities (W/m²) with shape (...) matching input points
        """
        points = prepare_points(points)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        # Extract parameters
        P_L = params['laser_power']
        R_0L = params['beam_waist_radius']
        theta_L = params['beam_divergence_angle']
        z_0 = params['beam_waist_position']

        # Calculate effective radius at height z (eq. 7)
        R_L = np.sqrt(R_0L ** 2 + 4 * theta_L ** 2 * (z_0 - z) ** 2)

        # Calculate Gaussian intensity distribution (eq. 6)
        numerator = 2 * P_L
        denominator = np.pi * R_L ** 2
        exponent = -2 * (x ** 2 + y ** 2) / R_L ** 2

        return (numerator / denominator) * np.exp(exponent)

    def attenuated_laser_intensity(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate the attenuated laser intensity considering powder stream attenuation
        using equation (10) from Huang et al. (2016).

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Process and material parameters including:
                   - particle_radius: Average powder particle radius (m)
                   All parameters required by powder_stream calculator

        Returns:
            Array of attenuated laser intensities (W/m²) with shape (...) matching input points
        """
        points = prepare_points(points)

        # Get mask for points inside powder stream
        inside_stream = self.powder_stream.is_within_powder_stream_radius(points, params)

        # Initialize with regular intensity
        result = self.laser_intensity(points, params)

        if not np.any(inside_stream):
            return result

        # Extract parameters
        r_p = params.get('particle_radius', 42e-6)  # Default to 42μm as in paper
        sigma = np.pi * r_p ** 2  # Extinction cross-section with Q_ext = 1

        # For points inside stream, calculate attenuation
        points_inside = points[inside_stream]
        z_p = self.powder_stream.powder_stream_boundary_at_z(points_inside, params)

        # Vectorized integration along z for each point
        def integrate_points(pt, z_boundary):
            z = pt[2]
            x, y = pt[0], pt[1]

            def integrand(_z):
                point = np.array([x, y, _z])
                return self.powder_stream.number_concentration(point, params)

            optical_depth, _ = quad(integrand, z, z_boundary)
            return optical_depth

        optical_depths = np.array([integrate_points(pt, zb)
                                   for pt, zb in zip(points_inside, z_p)])

        # Apply Beer-Lambert law
        result[inside_stream] *= np.exp(-sigma * optical_depths)
        return result

    def powder_temperature_increase(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate the powder temperature increment using equation (13) from Huang et al. (2016).

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Process and material parameters including:
                   - laser_absorptivity: Powder absorption coefficient
                   - specific_heat: Powder specific heat capacity (J/kg·K)
                   - density: Powder material density (kg/m³)
                   - particle_velocity: Average particle velocity (m/s)
                   - nozzle_angle: Nozzle inclination angle (rad)
                   All parameters required by powder_stream calculator

        Returns:
            Array of temperature increases (K) with shape (...) matching input points
        """
        points = prepare_points(points)

        # Get mask for points inside powder stream
        inside_stream = self.powder_stream.is_within_powder_stream_radius(points, params)

        # Initialize results array with zeros
        result = np.zeros(points.shape[:-1])

        if not np.any(inside_stream):
            return result

        # Extract parameters
        beta = params['laser_absorptivity']
        c_p = params['specific_heat']
        rho_p = params['density']
        r_p = params.get('particle_radius', 42e-6)
        v_p = params['particle_velocity']
        phi = params['nozzle_angle']

        # For points inside stream, calculate temperature increase
        points_inside = points[inside_stream]
        z_p = self.powder_stream.powder_stream_boundary_at_z(points_inside, params)

        def integrate_points(pt, z_boundary):
            z = pt[2]
            x, y = pt[0], pt[1]

            def integrand(z_prime):
                point = np.array([x, y, z_prime])
                I_A = self.attenuated_laser_intensity(point, params)
                dz = z_prime - z
                dt = dz / (v_p * np.sin(phi))
                return (3 * beta * I_A) / (4 * r_p * rho_p * c_p) * dt

            temp_increase, _ = quad(integrand, z, z_boundary)
            return temp_increase

        temp_increases = np.array([integrate_points(pt, zb)
                                   for pt, zb in zip(points_inside, z_p)])

        result[inside_stream] = temp_increases
        return result

    def powder_heat_source_intensity(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate the powder heat source intensity using equation (14) from Huang et al. (2016).

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Process and material parameters including:
                   - specific_heat: Powder specific heat capacity (J/kg·K)
                   - melting_temp: Material melting temperature (K)
                   - initial_temp: Initial powder temperature (K)
                   All parameters required by powder_stream calculator

        Returns:
            Array of powder heat source intensities (W/m²) with shape (...) matching input points
        """
        points = prepare_points(points)

        # Skip calculation if outside powder stream
        inside_stream = self.powder_stream.is_within_powder_stream_radius(points, params)
        if not np.any(inside_stream):
            return np.zeros(points.shape[:-1])

        # Extract parameters
        c_p = params['specific_heat']
        T_m = params['melting_temp']
        T_0 = params['initial_temp']

        # Calculate based on temperature increase and concentration
        rho = self.powder_stream.powder_concentration(points, params)
        T_p = T_0 + self.powder_temperature_increase(points, params)

        result = np.zeros(points.shape[:-1])
        result[inside_stream] = c_p * rho[inside_stream] * (T_p[inside_stream] - T_m)
        return result

    def net_energy_intensity(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate the net energy intensity using equation (15) from Huang et al. (2016).

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Process and material parameters required by other methods

        Returns:
            Array of net energy intensities (W/m²) with shape (...) matching input points
        """
        points = prepare_points(points)
        I_A = self.attenuated_laser_intensity(points, params)
        I_p = self.powder_heat_source_intensity(points, params)

        return I_A + I_p


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from field_visualization import ThermalPlotter
    from process_parameters import set_params

    # Set up parameters and calculators
    params = set_params()
    heat_sources = YuzeHuangHeatSources()
    plotter = ThermalPlotter(melting_temp=params['melting_temp'])

    # Grid setup
    x_range = (-0.003, 0.003)  # ±3mm
    y_range = (-0.003, 0.003)  # ±3mm
    z_range = (0, 0.012)  # 0-12mm
    n_points = 50

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    z = np.linspace(z_range[0], z_range[1], n_points)

    # Create meshgrids
    X_xy, Y_xy = np.meshgrid(x, y)
    X_xz, Z_xz = np.meshgrid(x, z)
    Y_yz, Z_yz = np.meshgrid(y, z)

    # Stack coordinates for vectorized calculation
    xy_points = np.stack([X_xy, Y_xy, np.zeros_like(X_xy)], axis=-1)
    xz_points = np.stack([X_xz, np.zeros_like(X_xz), Z_xz], axis=-1)
    yz_points = np.stack([np.zeros_like(Y_yz), Y_yz, Z_yz], axis=-1)

    # Fields to visualize
    fields = ['laser_intensity', 'attenuated_laser_intensity',
              'powder_heat_source_intensity', 'net_energy_intensity']

    for field in fields:
        # Create figure with custom layout
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{field.replace("_", " ").title()} Distribution', fontsize=14, y=1.05)

        # Get calculation method
        calc_method = getattr(heat_sources, field)

        # Calculate intensities for each plane and convert from W/m² to W/mm²
        intensities_xy = calc_method(xy_points, params) / 1e6
        intensities_xz = calc_method(xz_points, params) / 1e6
        intensities_yz = calc_method(yz_points, params) / 1e6

        # Data for each plane
        plane_data = [
            (intensities_xy, X_xy * 1000, Y_xy * 1000, 'XY Plane', 'X (mm)', 'Y (mm)'),
            (intensities_xz, X_xz * 1000, Z_xz * 1000, 'XZ Plane', 'X (mm)', 'Z (mm)'),
            (intensities_yz, Y_yz * 1000, Z_yz * 1000, 'YZ Plane', 'Y (mm)', 'Z (mm)')
        ]

        for ax, (intensities, grid1, grid2, title, xlabel, ylabel) in zip(axes, plane_data):
            # Create base intensity visualization using contours
            contours = plotter.create_temperature_pixels(
                ax=ax,
                grid1=grid1,
                grid2=grid2,
                temperatures=intensities,
                label=f'Intensity (W/mm²)'
            )

            # Add powder stream boundary if XY plane
            if 'XY' in title:
                plotter.add_powder_stream_boundary(
                    ax=ax,
                    grid1=grid1 / 1000,  # Convert back to meters for calculation
                    grid2=grid2 / 1000,
                    params=params,
                    label='Powder Stream'
                )

            # Customize appearance
            ax.set_title(title, pad=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.3)

            # Add legend if we have multiple elements
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='lower right')

            # Add intensity value at heat source point
            if 'XY' in title:
                max_intensity = np.max(intensities)
                ax.text(0.02, 0.98,
                       f'Max Intensity: {max_intensity:.2f} W/mm²',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.show()