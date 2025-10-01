from functools import wraps
from typing import Dict
import numpy as np
from scipy.integrate import quad, fixed_quad
from utils.vectorize_inputs import prepare_points

from dataclasses import dataclass
from typing import Literal, Callable


@dataclass
class MovingHeatSourceMetadata:
    """Metadata for moving heat source solutions.

    Attributes:
        movement_axis: Primary axis of heat source movement ('x' or 'y')
    """
    movement_axis: Literal['x', 'y']


def moving_heat_source(movement_axis: Literal['x', 'y']) -> Callable:
    """Decorator for moving heat source solutions.

    Args:
        movement_axis: Primary axis of heat source movement

    Example:
        @moving_heat_source('x')
        def rosenthal_solution(points, t, params):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.metadata = MovingHeatSourceMetadata(movement_axis)
        return wrapper

    return decorator


class YuzaHuangTemperature:
    """
    Calculates temperature distributions for laser directed energy deposition processes
    using various analytical solutions from Huang et al. (2019a).
    """

    @staticmethod
    def validate_time(t: float) -> None:
        """
        Validate that time is positive.

        Args:
            t: Time to validate (s)

        Raises:
            ValueError: If time is not positive
        """
        if t < 0:
            raise ValueError(f"Time must be positive: {t}")

    @moving_heat_source('y')
    def delta_temperature(self, points: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """
        Calculate total temperature rise using equation (9) from Huang et al. (2019a).
        Combines both laser and powder contributions.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            t: Time (s)
            params: Process and material parameters including:
                   - density: Material density (kg/m³)
                   - specific_heat: Specific heat capacity (J/kg·K)
                   - thermal_diffusivity: Thermal diffusivity (m²/s)
                   - laser_power: Laser power (W)
                   - laser_absorptivity: Laser absorption coefficient
                   - powder_feed_rate: Powder mass flow rate (kg/s)
                   - scan_speed: Process velocity (m/s)
                   - nozzle_angle: Nozzle inclination angle (rad)
                   - initial_temp: Initial temperature (K)
                   - melting_temp: Melting temperature (K)
                   - average_powder_stream_radius: Average powder stream radius (m)
                   - laser_beam_radius: Laser beam radius (m)

        Returns:
            Array of temperature rises (K) with shape (...) matching input points
        """
        self.validate_time(t)
        points = prepare_points(points)

        # Extract parameters
        rho = params['density']
        cp = params['specific_heat']
        alpha = params['thermal_diffusivity']

        prefactor = 2 / (rho * cp * np.pi * np.sqrt(np.pi * alpha))

        # Vectorize the integration function
        def integrate_point(point):
            return quad(
                lambda tau: (self._laser_integrand(point[0], point[1], point[2], t, tau, params) +
                             self._powder_integrand(point[0], point[1], point[2], t, tau, params)),
                1e-10,
                t - 1e-10
            )[0]

        vectorized_integral = np.vectorize(integrate_point, signature='(3)->()')(points)
        return prefactor * vectorized_integral

    @moving_heat_source('y')
    def laser_contribution(self, points: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """
        Calculate temperature rise due to laser heating only.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            t: Time (s)
            params: Process parameters dictionary

        Returns:
            Array of temperature rises (K) with shape (...) matching input points
        """
        self.validate_time(t)
        points = prepare_points(points)

        # Extract parameters
        rho = params['density']
        cp = params['specific_heat']
        alpha = params['thermal_diffusivity']

        prefactor = 2 / (rho * cp * np.pi * np.sqrt(np.pi * alpha))

        # Vectorize the integration function
        def integrate_point(point):
            return quad(
                lambda tau: self._laser_integrand(point[0], point[1], point[2], t, tau, params),
                1e-10,
                t
            )[0]

        vectorized_integral = np.vectorize(integrate_point, signature='(3)->()')(points)
        return prefactor * vectorized_integral

    @moving_heat_source('y')
    def powder_contribution(self, points: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """
        Calculate temperature rise due to powder heating only.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            t: Time (s)
            params: Process parameters dictionary

        Returns:
            Array of temperature rises (K) with shape (...) matching input points
        """
        self.validate_time(t)
        points = prepare_points(points)

        # Extract parameters
        rho = params['density']
        cp = params['specific_heat']
        alpha = params['thermal_diffusivity']

        prefactor = 2 / (rho * cp * np.pi * np.sqrt(np.pi * alpha))

        # Vectorize the integration function
        def integrate_point(point):
            return quad(
                lambda tau: self._powder_integrand(point[0], point[1], point[2], t, tau, params),
                1e-10,
                t
            )[0]

        vectorized_integral = np.vectorize(integrate_point, signature='(3)->()')(points)
        return prefactor * vectorized_integral

    @staticmethod
    def _laser_integrand(x: float, y: float, z: float, t: float,
                         tau: float, params: Dict) -> float:
        """Calculate the laser contribution integrand for Huang's solution."""
        dt = t - tau
        alpha = params['thermal_diffusivity']
        PL = params['laser_power']
        beta_w = params['laser_absorptivity']
        v = params['scan_speed']
        RL = params['laser_beam_radius']

        numerator = beta_w * PL / np.sqrt(dt)
        denominator = RL ** 2 + 8 * alpha * dt
        exponent = -2 * (x ** 2 + (y - v * tau) ** 2) / denominator - z ** 2 / (4 * alpha * dt)

        return (numerator / denominator) * np.exp(exponent)

    @staticmethod
    def _powder_integrand(x: float, y: float, z: float, t: float,
                          tau: float, params: Dict) -> float:
        """Calculate the powder contribution integrand for Huang's solution."""
        dt = t - tau
        alpha = params['thermal_diffusivity']
        cp = params['specific_heat']
        mdot = params['powder_feed_rate']
        v = params['scan_speed']
        phi = params['nozzle_angle']
        T0 = params['initial_temp']
        Tm = params['melting_temp']
        r_avg = params['average_powder_stream_radius']

        numerator = cp * mdot * (T0 - Tm) / np.sqrt(dt)
        denominator = r_avg ** 2 + 8 * alpha * dt
        exponent = -2 * (x ** 2 + ((y - v * tau) / np.sin(phi)) ** 2) / denominator - z ** 2 / (4 * alpha * dt)

        return (numerator / denominator) * np.exp(exponent)

    @moving_heat_source('y')
    def adaptive_temperature(self, points: np.ndarray, t: float, params: Dict,
                             t_dwell: float = 0.1) -> np.ndarray:
        """
        Calculate temperature considering cooling effects over a dwell time.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            t: Time (s)
            params: Process parameters dictionary
            t_dwell: Dwell time for cooling consideration (s)

        Returns:
            Array of net temperature changes (K) with shape (...) matching input points
        """
        points = prepare_points(points)
        dT = self.delta_temperature(points, t, params)

        # Calculate cooling effect by shifting y-positions
        points_shifted = points.copy()
        points_shifted[..., 1] -= t_dwell * params['scan_speed']
        dT -= self.delta_temperature(points_shifted, t - t_dwell, params)

        return dT


class EagarTsaiTemperatureV0:
    """
    Calculates temperature distributions for laser directed energy deposition processes
    using the Eagar-Tsai analytical solution for a moving Gaussian heat source.

    Implementation based on Eagar & Tsai (1983) and Wolfer et al. (2019).
    """

    @staticmethod
    def validate_time(t: float) -> None:
        """
        Validate that time is positive.

        Args:
            t: Time to validate (s)

        Raises:
            ValueError: If time is not positive
        """
        if t < 0:
            raise ValueError(f"Time must be positive: {t}")

    @moving_heat_source('x')
    def delta_temperature(self, points: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """
        Calculate temperature rise using the Eagar-Tsai solution for a moving Gaussian heat source.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            t: Time (s)
            params: Process and material parameters including:
                   - density: Material density (kg/m³)
                   - specific_heat: Specific heat capacity (J/kg·K)
                   - thermal_diffusivity: Thermal diffusivity (m²/s)
                   - laser_power: Laser power (W)
                   - laser_absorptivity: Laser absorption coefficient
                   - scan_speed: Process velocity (m/s)
                   - laser_beam_radius: Laser beam radius (m)

        Returns:
            Array of temperature rises (K) with shape (...) matching input points
        """
        self.validate_time(t)
        points = prepare_points(points)

        # Extract parameters
        P = params['laser_power']
        A = params['laser_absorptivity']
        V = params['scan_speed']
        rho = params['density']
        cp = params['specific_heat']
        D = params['thermal_diffusivity']
        sigma = params['laser_beam_radius']

        def integrate_point(point):
            x, y, z = point

            def integrand(tau_bar):
                sigma_sq_plus_2Dt = sigma ** 2 + 2 * D * tau_bar
                exp_xy = np.exp(-((x - V * t + V * tau_bar) ** 2 + y ** 2) / (2 * sigma_sq_plus_2Dt))
                exp_z = np.exp(-z ** 2 / (4 * D * tau_bar))
                return (tau_bar ** (-0.5) / (sigma_sq_plus_2Dt)) * exp_xy * exp_z

            #result, _ = quad(integrand, 1e-10, t, epsabs=1e-6, epsrel=1e-6)
            result, _ = fixed_quad(integrand, 1e-10, t, n=20)
            return result

        prefactor = (A * P) / (rho * cp * np.sqrt(4 * np.pi ** 3 * D))
        vectorized_integral = np.vectorize(integrate_point, signature='(3)->()')(points)
        return prefactor * vectorized_integral


from numpy.polynomial.legendre import leggauss
import numpy as np


# Gaussian quadrature integration function
def gaussian_quadrature(integrand, a, b, num_points=100):
    points, weights = leggauss(num_points)
    # Map points to the interval [a, b]
    transformed_points = 0.5 * (b - a) * points + 0.5 * (b + a)
    transformed_weights = 0.5 * (b - a) * weights
    return np.sum(integrand(transformed_points) * transformed_weights)


class EagarTsaiTemperatureGaussianQuadrature:
    """
    Calculates temperature distributions for laser directed energy deposition processes
    using the Eagar-Tsai analytical solution for a moving Gaussian heat source.
    """

    @staticmethod
    def validate_time(t: float) -> None:
        if t < 0:
            raise ValueError(f"Time must be positive: {t}")

    @moving_heat_source('x')
    def delta_temperature(self, points: np.ndarray, t: float, params: dict) -> np.ndarray:
        self.validate_time(t)
        points = prepare_points(points)

        # Extract parameters
        P = params['laser_power']
        A = params['laser_absorptivity']
        V = params['scan_speed']
        rho = params['density']
        cp = params['specific_heat']
        D = params['thermal_diffusivity']
        sigma = params['laser_beam_radius']

        # Precompute the prefactor
        prefactor = (A * P) / (rho * cp * np.sqrt(4 * np.pi ** 3 * D))

        def integrate_point(point):
            x, y, z = point

            def integrand(tau_bar):
                sigma_sq_plus_2Dt = sigma ** 2 + 2 * D * tau_bar
                exp_xy = np.exp(-((x + V * tau_bar) ** 2 + y ** 2) / (2 * sigma_sq_plus_2Dt))
                exp_z = np.exp(-z ** 2 / (4 * D * tau_bar))
                return (tau_bar ** (-0.5) / (sigma_sq_plus_2Dt)) * exp_xy * exp_z

            # Perform Gaussian Quadrature integration
            return gaussian_quadrature(integrand, 1e-10, t, num_points=10)

        # Compute temperature rise at each point
        vectorized_integral = np.vectorize(integrate_point, signature='(3)->()')(points)
        return prefactor * vectorized_integral


import numpy as np
from numba import njit, prange


@njit
def integrand_numba(tau_bar: float, x: float, y: float, z: float, V: float, D: float, sigma: float) -> float:
    """Calculate the integrand value for a single point."""
    sigma_sq_plus_2Dt = sigma * sigma + 2.0 * D * tau_bar
    exp_xy = np.exp(-((x + V * tau_bar) * (x + V * tau_bar) + y * y) / (2.0 * sigma_sq_plus_2Dt))
    exp_z = np.exp(-(z * z) / (4.0 * D * tau_bar))
    return tau_bar ** (-0.5) / sigma_sq_plus_2Dt * exp_xy * exp_z


@njit
def gaussian_quadrature_single_point(a: float, b: float, num_points: int, x: float, y: float, z: float,
                                     V: float, D: float, sigma: float) -> float:
    """Perform Gaussian quadrature for a single spatial point."""
    # Calculate integration points and weights
    dt = (b - a) / float(num_points)
    integral = 0.0

    # Manual integration loop to avoid array operations
    for i in range(num_points):
        point = a + (float(i) + 0.5) * dt  # Midpoint rule
        integral += integrand_numba(point, x, y, z, V, D, sigma) * dt

    return integral


@njit(parallel=True)
def parallel_integrate(points: np.ndarray, t: float, num_points: int, V: float, D: float, sigma: float) -> np.ndarray:
    """Parallel integration over multiple spatial points."""
    num_spatial_points = points.shape[0]
    results = np.zeros(num_spatial_points, dtype=np.float64)

    for i in prange(num_spatial_points):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        results[i] = gaussian_quadrature_single_point(1e-10, t, num_points, x - V*t, y, z, V, D, sigma)

    return results


class EagarTsaiTemperature:
    """
    Calculates temperature distributions for laser directed energy deposition processes
    using the Eagar-Tsai analytical solution for a moving Gaussian heat source.
    """

    @staticmethod
    def validate_time(t: float) -> None:
        """Validate that the time value is positive."""
        if t <= 0:
            raise ValueError(f"Time must be positive: {t}")

    @moving_heat_source('x')
    def delta_temperature(self, points: np.ndarray, t: float, params: dict) -> np.ndarray:
        """
        Calculate temperature change at specified points and time.

        Args:
            points: Array of shape (N, 3) containing (x, y, z) coordinates
            t: Time at which to calculate temperature
            params: Dictionary containing process parameters

        Returns:
            Array of temperature changes at each point
        """
        self.validate_time(t)

        # Ensure points are in the correct format
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)
        if points.dtype != np.float64:
            points = points.astype(np.float64)
        if len(points.shape) != 2 or points.shape[1] != 3:
            points_shape = points.shape
            points = points.reshape(-1, 3)

        # Extract parameters
        P = float(params['laser_power'])
        A = float(params['laser_absorptivity'])
        V = float(params['scan_speed'])
        rho = float(params['density'])
        cp = float(params['specific_heat'])
        D = float(params['thermal_diffusivity'])
        sigma = float(params['laser_beam_radius'])

        # Precompute the prefactor
        prefactor = (A * P) / (rho * cp * np.sqrt(4 * np.pi ** 3 * D))

        # Parallelized integration
        num_points_quad = 50  # Number of points for Gaussian quadrature
        integral_results = parallel_integrate(points, t, num_points_quad, V, D, sigma)

        return (prefactor * integral_results).reshape(points_shape[:-1])


@moving_heat_source('x')
def rosenthal_temperature(points: np.ndarray, t: float, params: Dict,
                          eps: float = 1e-10) -> np.ndarray:
    """
    Calculate instantaneous temperature using Rosenthal's moving point heat source solution.

    T - T_0 = (Q/(4πk)) * (exp(-v(ξ-R)/(2α))/R)
    where R = sqrt(ξ² + y² + z²) and ξ = x - vt

    Args:
        points: Array of points with shape (..., 3) where last dimension is (x,y,z)
        t: Time (s)
        params: Process parameters including:
               - laser_power: Heat source power (W)
               - scan_speed: Process velocity (m/s)
               - thermal_diffusivity: Thermal diffusivity (m²/s)
               - thermal_conductivity: Thermal conductivity (W/(m·K))
        eps: Small value to prevent division by zero

    Returns:
        Array of temperature rises (K) with shape (...) matching input points
    """
    points = prepare_points(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    Q = params['laser_power']
    v = params['scan_speed']
    alpha = params['thermal_diffusivity']
    k = params['thermal_conductivity']

    epsilon = x - v * t
    R = np.sqrt(epsilon ** 2 + y ** 2 + z ** 2 + eps)

    numerator = Q * np.exp(-v * (epsilon - R) / (2 * alpha))
    denominator = 4 * np.pi * k * R

    return numerator / denominator


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from thermal.field_visualization import ThermalPlotter
    from thermal.field_utils import grid_slice
    from process_parameters import set_params

    # Initialize components
    params = set_params()
    temp_calculator = YuzaHuangTemperature()
    plotter = ThermalPlotter(melting_temp=params['melting_temp'])

    # Define visualization cases
    cases = [
        {
            'name': 'Total Temperature Distribution',
            'method': temp_calculator.delta_temperature,
            'times': [0.1, 1, 4],  # Different time points
        },
        {
            'name': 'Heat Source Contributions',
            'methods': [
                (temp_calculator.laser_contribution, 'Laser Only'),
                (temp_calculator.powder_contribution, 'Powder Only'),
                (temp_calculator.delta_temperature, 'Combined')
            ],
            'time': 2  # Fixed time point for comparison
        },
        {
            'name': 'Adaptive Temperature',
            'method': temp_calculator.adaptive_temperature,
            'times': np.linspace(1, 2, 6),  # Multiple time steps
        }
    ]

    # Define spatial ranges (in meters)
    x_range = (-0.003, 0.003)  # ±3mm
    y_range = (-0.003, 0.003)  # ±3mm
    z_range = (0, 0.012)  # 0-12mm

    # Process each case
    for case in cases:
        print(f"\nProcessing {case['name']}...")

        if case['name'] == 'Total Temperature Distribution':
            # Show temperature evolution at different times
            for time in case['times']:
                fig = plt.figure(figsize=(15, 5))
                fig.suptitle(f"{case['name']}: t = {time * 1000:.1f} ms")

                # Configure subplots
                gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
                axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

                # Define plane configurations
                plane_configs = [
                    ('xy', x_range, y_range, 'XY', 'X', 'Y'),
                    ('xz', x_range, z_range, 'XZ', 'X', 'Z'),
                    ('yz', y_range, z_range, 'YZ', 'Y', 'Z')
                ]

                # Create visualizations for each plane
                for ax, (plane, range1, range2, title, xlabel, ylabel) in zip(axes, plane_configs):
                    # Use grid_slice to calculate temperatures
                    grid1, grid2, temps = grid_slice(
                        t=time,
                        scan_speed=params['scan_speed'],
                        range1=range1,
                        range2=range2,
                        get_temp=case['method'],
                        get_temp_args=(params,),
                        plane=plane
                    )

                    # Use plot_thermal_plane for visualization
                    plotter.plot_thermal_plane(
                        ax=ax,
                        grid1=grid1,
                        grid2=grid2,
                        temperatures=temps,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        time=time,
                        speed=params['scan_speed'],
                        show_melt_pool=True,
                        show_heat_source=True,
                        show_grid=True,
                        params=params,
                        show_powder_stream_boundary=plane=='xy',
                    )

                plt.plot()

        elif case['name'] == 'Heat Source Contributions':
            fig = plt.figure(figsize=(15, 5))
            fig.suptitle(f"{case['name']}: t = {case['time'] * 1000:.1f} ms")

            axes = [plt.subplot(1, 3, i + 1) for i in range(3)]

            for ax, (method, label) in zip(axes, case['methods']):
                # Use grid_slice for XY plane temperatures
                grid1, grid2, temps = grid_slice(
                    t=case['time'],
                    scan_speed=params['scan_speed'],
                    range1=x_range,
                    range2=y_range,
                    get_temp=method,
                    get_temp_args=(params,),
                    plane='xy'
                )

                # Use plot_thermal_plane for visualization
                plotter.plot_thermal_plane(
                    ax=ax,
                    grid1=grid1,
                    grid2=grid2,
                    temperatures=temps,
                    title=label,
                    xlabel='X',
                    ylabel='Y',
                    time=case['time'],
                    speed=params['scan_speed'],
                    show_melt_pool=True,
                    show_heat_source=True,
                    show_grid=True,
                    params=params,
                    show_powder_stream_boundary=plane == 'xy',
                )

            plt.plot()

        elif case['name'] == 'Adaptive Temperature':
            n_times = len(case['times'])
            n_cols = 3
            n_rows = (n_times + n_cols - 1) // n_cols

            fig = plt.figure(figsize=(15, 5 * n_rows))
            fig.suptitle(f"{case['name']}")

            for idx, time in enumerate(case['times']):
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)

                # Use grid_slice for XY plane temperatures
                grid1, grid2, temps = grid_slice(
                    t=time,
                    scan_speed=params['scan_speed'],
                    range1=x_range,
                    range2=y_range,
                    get_temp=case['method'],
                    get_temp_args=(params,),
                    plane='xy'
                )

                # Use plot_thermal_plane for visualization
                plotter.plot_thermal_plane(
                    ax=ax,
                    grid1=grid1,
                    grid2=grid2,
                    temperatures=temps,
                    title=f't = {time * 1000:.1f} ms',
                    xlabel='X',
                    ylabel='Y',
                    time=time,
                    speed=params['scan_speed'],
                    show_melt_pool=True,
                    show_heat_source=True,
                    show_grid=True,
                    params=params,
                    show_powder_stream_boundary=plane == 'xy',
                )

            plt.show()