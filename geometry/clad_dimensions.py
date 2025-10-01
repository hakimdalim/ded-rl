from typing import Dict, Tuple, Callable, Union, Any
import numpy as np


class YuzeHuangCladDimensions:
    """
    Implements clad geometry calculations from:
    'A new physics-based model for laser directed energy deposition
    (powder-fed additive manufacturing): From single-track to multi-track and multi-layer'
    - Huang et al. (2019)

    Calculates final clad dimensions by considering:
    1. Initial geometry from powder catchment
    2. Surface tension effects
    3. Wetting behavior of molten material
    """

    def __init__(self, powder_concentration_func: Callable, resolution: int = 200):
        """
        Initialize YuzeHuangCladDimensions with configurable parameters.

        Args:
            powder_concentration_func: Function to calculate powder concentration
            resolution: Grid resolution for vectorized integration
        """
        self.powder_concentration = powder_concentration_func
        self.resolution = resolution

    def _prepare_concentration_points(self, x, y) -> np.ndarray:
        """
        Prepare points array for powder concentration calculation.

        Args:
            x: X coordinates (scalar or array)
            y: Y coordinates (scalar or array)

        Returns:
            Array of points with shape (..., 3)
        """
        # Stack coordinates with z=0 for substrate level
        return np.stack([x, y, np.zeros_like(x)], axis=-1)

    def compute_initial_height(
            self,
            pool_width: float,
            pool_length: float,
            params: Dict
    ) -> float:
        """
        Calculates initial clad height h₀ based on powder mass conservation.

        Integrates powder concentration over melt pool area to determine
        material accumulation. Uses simplified rectangular integration for
        computational efficiency.

        Args:
            pool_width: Melt pool width from thermal solution
            pool_length: Melt pool length_between from thermal solution
            params: Process and material parameters

        Returns:
            Initial clad height before wetting effects
        """
        # Extract parameters with paper symbols
        v_p = params['particle_velocity']  # Particle velocity
        phi = params['nozzle_angle']  # Nozzle inclination angle
        v = params['scan_speed']  # Process speed
        ρ_p = params['density']  # Powder density

        # Grid based on configured resolution
        dx = pool_width / self.resolution
        dy = pool_length / self.resolution
        #print('pool_width', pool_width, 'pool_length', pool_length)
        #print('dx', dx, 'dy', dy)

        if pool_width == 0 or pool_length == 0:
            return 0

        x_range = np.arange(-pool_width / 2, pool_width / 2, dx)
        y_range = np.arange(-pool_length / 2, pool_length / 2, dy)

        # Create grid and prepare points for powder concentration
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        points = self._prepare_concentration_points(x_grid, y_grid)

        # Calculate mask for integration region
        mask = x_grid ** 2 <= (pool_width / 2) ** 2 * (1 - (y_grid / (pool_length / 2)) ** 2)

        # Calculate concentrations using prepared points
        concentrations = self.powder_concentration(points, params)
        integral = np.sum(concentrations[mask]) * dx * dy

        # Calculate height based on mass conservation (Equation 14 in paper)
        return (3 * v_p * np.sin(phi)) / (2 * pool_width * v * ρ_p) * integral

    def compute_wetting_angle(
            self,
            initial_height: float,
            initial_width: float,
            params: Dict
    ) -> float:
        """
        Calculates final wetting angle α_w based on Hoffman-Voinov-Tanner law.

        Models the spreading behavior of the molten material as it wets
        the substrate surface. Considers surface tension, viscosity, and
        time evolution effects.

        Args:
            initial_height: Initial clad height before wetting (h₀)
            initial_width: Initial clad width (w₀)
            params: Process and material parameters

        Returns:
            Final wetting angle after spreading
        """
        # Extract parameters with paper symbols
        γ_LV = params['surface_tension']  # Liquid-vapor surface tension
        μ = params['viscosity']  # Dynamic viscosity
        ε = params['epsilon']  # Universal constant from HVT law
        ρ_p = params['density']  # Powder density

        if initial_height == 0 or initial_width == 0:
            return 0

        # Initial contact angle α₀ from geometry (Section 2.2)
        α_0 = np.arctan(4 * initial_height / initial_width)

        # Characteristic oscillation time t_osc (Section 2.2)
        t_osc = np.sqrt(ρ_p * (initial_width / 2) ** 3 / γ_LV)

        # Compute final angle at t = 3*t_osc (Equation 18)
        t = 3 * t_osc
        term1 = np.sqrt(3) * μ * t / (9 * ε * ρ_p * initial_width ** 2)
        term2 = np.arctanh(np.sqrt(3) * α_0 / 6)

        return -2 * np.sqrt(3) * np.tanh(term1 - term2)

    def compute_final_geometry(
            self,
            pool_width: float,
            pool_length: float,
            params: Dict
    ) -> Tuple[Union[int, Any], Union[float, Any], Union[float, Any]]:
        """
        Computes final clad width w and height h considering wetting effects.

        Combines initial geometry calculation with wetting angle effects
        to determine final stable clad dimensions.

        Args:
            pool_width: Melt pool width w_p from thermal solution
            pool_length: Melt pool length_between l_p from thermal solution
            params: Process and material parameters

        Returns:
            Tuple of (final_width, final_height)
        """
        # Calculate initial height h₀
        h_0 = self.compute_initial_height(pool_width, pool_length, params)

        if h_0 == 0:
            return 0.0, 0.0, None

        # Calculate wetting angle α_w
        α_w = self.compute_wetting_angle(h_0, pool_width, params)

        # Calculate final dimensions based on volume conservation (Equation 20)
        h = np.sqrt(pool_width * h_0 * np.tan(α_w)) / 2
        w = 2 * np.sqrt(pool_width * h_0 / np.tan(α_w))

        return w, h, α_w


if __name__ == "__main__":
    import time
    from scipy import integrate
    from process_parameters import set_params
    from thermal.powder_stream import YuzeHuangPowderStream
    from thermal.powder_stream_new import VoxelPowderStream

    # Setup test parameters from Huang et al. 2019
    params = set_params()
    params.update({
        'particle_velocity': (2.5 / 60000) / (np.pi * (0.7e-3) ** 2),  # v_p [m/s]
        'nozzle_angle': np.radians(90),  # φ = 90°, co-axial nozzle
        'scan_speed': 0.003,  # v = 3 mm/s
        'density': 7452.5,  # ρ_p = 7452.5 kg/m³ (avg between 25°C-1538°C)
        'nozzle_radius': 0.7e-3,  # r₀ = 0.7 mm
        'powder_divergence_angle': np.radians(5.8),  # θ = 5.8°
        'powder_feed_rate': 2 * (1/1000) * (1/60),  # Convert 2 g/min to kg/s
        'nozzle_height': 0.015,  #15 mm
    })

    powder_stream = VoxelPowderStream(
        r"C:\Users\schuermm\PycharmProjects\faim-jms-sim\thermal\_arrays\250422_10%3A36%3A17\target_stack.npz",
        outlet_offset=1.1635e-3,
        nozzle_height=params['nozzle_height'],
        visualize=False  # Turn off individual visualizations
    )

    # Typical melt pool dimensions
    wp = 0.001  # 1 mm width
    lp = 0.0015  # 1.5 mm length_between

    clad_geometry = YuzeHuangCladDimensions(powder_stream.powder_concentration)
    width, height, α_w = clad_geometry.compute_final_geometry(wp, lp, params)

    print(f"Ponticon clad width: {width * 1000:.3f} mm")
    print(f"Ponticon clad height: {height * 1000:.3f} mm")


    # Create powder stream wrapper function
    def powder_concentration_wrapper(points, params):
        """Wrapper to maintain compatibility with original interface"""
        return YuzeHuangPowderStream.powder_concentration(points, params)

    # Iterate through nozzle angles
    angles_deg = np.arange(15, 91, 5)
    results = []

    for angle_deg in angles_deg:
        params['nozzle_angle'] = np.radians(angle_deg)

        clad_geometry = YuzeHuangCladDimensions(powder_concentration_wrapper)
        width, height, α_w = clad_geometry.compute_final_geometry(wp, lp, params)

        results.append({'angle': angle_deg, 'width': width * 1000, 'height': height * 1000})
        print(f"Angle: {angle_deg:3.0f}° | Width: {width * 1000:6.3f} mm | Height: {height * 1000:6.3f} mm")

    exit()

    class IntegrationBenchmark:
        """
        Benchmarks different integration methods for clad height calculation from:
        'A new physics-based model for laser directed energy deposition' - Huang et al. (2019)
        """

        def __init__(self, pool_width: float, pool_length: float, params: Dict, grid_size: int):
            self.w_p = pool_width
            self.l_p = pool_length
            self.params = params
            self.grid_size = grid_size

        def _prepare_concentration_points(self, x, y) -> np.ndarray:
            """
            Prepare points array for powder concentration calculation.
            """
            return np.stack([x, y, np.zeros_like(x)], axis=-1)

        def loop_integration(self) -> float:
            """Original nested loop implementation with adjustable grid"""
            dx = self.w_p / self.grid_size
            dy = self.l_p / self.grid_size
            x_range = np.arange(-self.w_p / 2, self.w_p / 2, dx)
            y_range = np.arange(-self.l_p / 2, self.l_p / 2, dy)

            integral = 0
            for x in x_range:
                for y in y_range:
                    if x ** 2 <= (self.w_p / 2) ** 2 * (1 - (y / (self.l_p / 2)) ** 2):
                        points = self._prepare_concentration_points(x, y)
                        integral += powder_concentration_wrapper(points, self.params) * dx * dy
            return integral

        def vectorized_integration(self) -> float:
            """NumPy vectorized implementation with adjustable grid"""
            dx = self.w_p / self.grid_size
            dy = self.l_p / self.grid_size
            x_range = np.arange(-self.w_p / 2, self.w_p / 2, dx)
            y_range = np.arange(-self.l_p / 2, self.l_p / 2, dy)

            x_grid, y_grid = np.meshgrid(x_range, y_range)
            points = self._prepare_concentration_points(x_grid, y_grid)
            mask = x_grid ** 2 <= (self.w_p / 2) ** 2 * (1 - (y_grid / (self.l_p / 2)) ** 2)
            concentrations = powder_concentration_wrapper(points, self.params)
            return np.sum(concentrations[mask]) * dx * dy

        def scipy_integration(self, epsrel: float = 1e-3) -> float:
            """
            SciPy dblquad implementation with adjustable tolerance
            """
            def integrand(x: float, y: float) -> float:
                if x ** 2 <= (self.w_p / 2) ** 2 * (1 - (y / (self.l_p / 2)) ** 2):
                    points = self._prepare_concentration_points(np.array(x), np.array(y))
                    return float(powder_concentration_wrapper(points, self.params))
                return 0

            return integrate.dblquad(
                integrand,
                -self.l_p / 2, self.l_p / 2,
                lambda y: -self.w_p / 2 * np.sqrt(1 - (y / (self.l_p / 2)) ** 2),
                lambda y: self.w_p / 2 * np.sqrt(1 - (y / (self.l_p / 2)) ** 2),
                epsrel=1.0 / self.grid_size
            )[0]

        def monte_carlo_integration(self) -> float:
            """Monte Carlo integration with number of points scaling with grid size"""
            n_points = self.grid_size * self.grid_size
            x = np.random.uniform(-self.w_p / 2, self.w_p / 2, n_points)
            y = np.random.uniform(-self.l_p / 2, self.l_p / 2, n_points)
            mask = x ** 2 <= (self.w_p / 2) ** 2 * (1 - (y / (self.l_p / 2)) ** 2)
            points = self._prepare_concentration_points(x[mask], y[mask])
            area = self.w_p * self.l_p
            return area * np.mean(powder_concentration_wrapper(points, self.params))


    def run_benchmark():
        # Setup test parameters from Huang et al. 2019
        params = set_params()
        params.update({
            'particle_velocity': (2.5 / 60000) / (np.pi * (0.7e-3) ** 2),  # v_p [m/s]
            'nozzle_angle': np.radians(50),  # φ = 50°
            'scan_speed': 0.003,  # v = 3 mm/s
            'density': 7452.5,  # ρ_p = 7452.5 kg/m³
            'nozzle_radius': 0.7e-3,  # r₀ = 0.7 mm
            'powder_divergence_angle': np.radians(5.8),  # θ = 5.8°
            'powder_feed_rate': 2 * (1 / 1000) * (1 / 60),  # Convert 2 g/min to kg/s
        })

        # Typical melt pool dimensions
        wp = 0.001  # 1 mm width
        lp = 0.002  # 2 mm length_between

        # Define grid sizes to test
        grid_sizes = [50, 100, 200, 400, 800, 1600]
        n_runs = 5  # Reduced number of runs for higher grid sizes

        print("\nBenchmark Results:")
        print("-" * 80)
        print(f"{'Grid Size':<10} {'Method':<12} {'Time (ms)':<12} {'Points':<12} {'Error':<12} {'Speedup':<12}")
        print("-" * 80)

        for grid_size in grid_sizes:
            benchmark = IntegrationBenchmark(wp, lp, params, grid_size)

            # Compute high-accuracy reference using scipy with fine tolerance
            reference_benchmark = IntegrationBenchmark(wp, lp, params, 1600)
            reference = reference_benchmark.scipy_integration(epsrel=1e-6)

            method_times = {}

            methods = {
                'Loop': benchmark.loop_integration,
                'Vectorized': benchmark.vectorized_integration,
                'SciPy': benchmark.scipy_integration,
                'Monte Carlo': benchmark.monte_carlo_integration
            }

            for name, method in methods.items():
                # Get number of points used
                if name in ['Loop', 'Vectorized']:
                    n_points = grid_size * grid_size
                elif name == 'Monte Carlo':
                    n_points = grid_size * grid_size
                else:
                    n_points = 'Adaptive'

                # Time the method
                start_time = time.time()
                for _ in range(n_runs):
                    result = method()
                avg_time = (time.time() - start_time) / n_runs * 1000

                method_times[name] = avg_time
                rel_error = abs(result - reference) / reference
                speedup = max(method_times.values()) / (avg_time + 1e-10)

                avg_time = float(avg_time)
                rel_error = float(rel_error)
                speedup = float(speedup)

                print(
                    f"{grid_size:<10} {name:<12} {avg_time:>10.2f}ms {n_points:<12} {rel_error:>10.2%} {speedup:>10.2f}x")

            print("-" * 80)

    run_benchmark()