import numpy as np
from typing import Dict
from utils.vectorize_inputs import prepare_points


class YuzeHuangPowderStream:
    """
    Implements powder stream model calculations for laser directed energy deposition.
    Provides methods to analyze powder concentration, distribution, and geometry
    for both coaxial and lateral nozzle configurations.

    This class provides static methods to:
    1. Transform coordinates between laser beam and powder stream coordinate systems
    2. Calculate powder mass and number concentrations at any point
    3. Determine powder stream boundaries and containment
    4. Handle both coaxial (φ = 90°) and lateral (0° < φ < 90°) nozzle configurations

    All calculations follow models commonly used in directed energy deposition literature,
    considering:
    - Nozzle geometry (height, radius, angle)
    - Powder stream characteristics (divergence angle, feed rate)
    - Particle properties (mass, velocity)
    - Spatial distribution of powder concentration
    """

    @staticmethod
    def transform_coordinates(points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Transform coordinates from laser beam center system to powder stream system.

        Performs two transformations:
        1. Shifts origin from substrate to nozzle outlet
        2. Rotates around x-axis by nozzle angle

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Dictionary containing nozzle_height and nozzle_angle

        Returns:
            Array of transformed points with same shape as input
        """
        points = prepare_points(points)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        nozzle_height = params['nozzle_height']  # H
        phi = params['nozzle_angle']

        # 1. Shift origin to nozzle outlet
        if phi != 0:
            y_shifted = y - nozzle_height / np.tan(phi)  # shift in y direction to nozzle
        else:
            y_shifted = y
        z_shifted = z - nozzle_height  # shift up to nozzle height

        # 2. Rotate around x-axis by angle phi
        x_prime = x  # x remains same
        y_prime = y_shifted * np.sin(phi) - z_shifted * np.cos(phi)  # rotate y
        z_prime = y_shifted * np.cos(phi) + z_shifted * np.sin(phi)  # rotate z

        return np.stack([x_prime, y_prime, z_prime], axis=-1)

    @staticmethod
    def powder_concentration(points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate powder mass concentration at given points.

        Uses Gaussian distribution model in transformed coordinate system.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Dictionary containing nozzle_radius, powder_divergence_angle,
                   particle_velocity, and powder_feed_rate

        Returns:
            Array of powder mass concentrations with shape (...) matching input points
        """
        points = prepare_points(points)

        r_0 = params['nozzle_radius']
        theta = params['powder_divergence_angle']
        v_p = params['particle_velocity']
        m_dot = params['powder_feed_rate']

        # Transform coordinates
        transformed = YuzeHuangPowderStream.transform_coordinates(points, params)
        x_prime, y_prime, z_prime = transformed[..., 0], transformed[..., 1], transformed[..., 2]

        # Calculate effective radius using z_prime
        r_z = r_0 - z_prime * np.tan(theta)

        # Calculate concentration using transformed coordinates
        exp_term = -2 * (x_prime ** 2 + y_prime ** 2) / r_z ** 2

        return (2 * m_dot) / (v_p * np.pi * r_z ** 2) * np.exp(exp_term)

    @staticmethod
    def number_concentration(points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate powder number concentration at given points.

        Uses mass concentration divided by individual particle mass.

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Dictionary containing particle_mass and other required parameters

        Returns:
            Array of particle number concentrations with shape (...) matching input points
        """
        points = prepare_points(points)
        m_p = params['particle_mass']

        mass_concentration = YuzeHuangPowderStream.powder_concentration(points, params)
        return mass_concentration / m_p

    @staticmethod
    def is_within_powder_stream_radius(points: np.ndarray, params: dict, t: float = 0.0) -> np.ndarray:
        """
        Check if points lie within the powder stream radius at their respective heights.
        Takes into account the movement of the powder stream over time.

        For lateral nozzle (0° < φ < 90°):
            First transforms to powder stream coordinates (x',y',z') then checks if
            point lies within the effective radius r(z') = r₀ - z'*tan(θ)

        For coaxial nozzle (φ = 90°):
            Uses radial distance r = √(x² + y²) to check if point lies within
            the conical powder stream r(z) = r₀ - (z-H)*tan(θ)

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
            params: Process and material parameters including:
                - nozzle_angle (φ): angle in radians
                - nozzle_height (H): height in m
                - nozzle_radius (r₀): radius in m
                - powder_divergence_angle (θ): angle in radians
                - scan_speed: Process velocity (m/s)
            t: Time point to evaluate (s), determines y-position of powder stream. Default is 0.0.

        Returns:
            Boolean array with shape (...) matching input points
        """
        points = prepare_points(points)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        # Extract parameters
        phi = params['nozzle_angle']  # φ
        H = params['nozzle_height']  # H
        r0 = params['nozzle_radius']  # r₀
        theta = params['powder_divergence_angle']  # θ
        v = params['scan_speed']

        # Calculate current y position
        y_pos = v * t
        # Adjust y coordinates relative to current position
        y_rel = y - y_pos

        # Check if nozzle is coaxial (φ = 90°)
        is_coaxial = np.isclose(phi, np.pi / 2, rtol=1e-10)

        if is_coaxial:
            # Calculate radial distance for coaxial case
            r = np.sqrt(x ** 2 + y_rel ** 2)
            # Check if points are within the conical powder stream
            return r <= r0 - (z - H) * np.tan(theta)
        else:
            # Transform to powder stream coordinates (x', y', z')
            y_prime = (y_rel - H / np.tan(phi)) * np.sin(phi) - (z - H) * np.cos(phi)
            z_prime = (y_rel - H / np.tan(phi)) * np.cos(phi) + (z - H) * np.sin(phi)

            # Calculate effective radius at z'
            r_at_z = r0 - z_prime * np.tan(theta)

            # Check if points' local coordinates fall within the powder stream radius
            return np.sqrt(x ** 2 + y_prime ** 2) <= r_at_z

    @staticmethod
    def powder_stream_boundary_at_z(points: np.ndarray, params: dict) -> np.ndarray:
        """
        Calculate the powder stream boundary height (z) for given (x,y) points.

        For lateral nozzle (0° < φ < 90°):
            z_p(y) = K + (y-K/tan(φ))*tan(φ-θ)
            where K = H + r₀*sin(φ)/tan(θ)

        For coaxial nozzle (φ = 90°):
            z_p(r) = H + (r₀ - r)/tan(θ)
            where r = √(x² + y²)

        Args:
            points: Array of points with shape (..., 3) where last dimension is (x,y,z)
                   Note: z-coordinates are ignored for this calculation
            params: Process and material parameters including:
                - nozzle_angle (φ): angle in radians
                - nozzle_height (H): height in m
                - nozzle_radius (r₀): radius in m
                - powder_divergence_angle (θ): angle in radians

        Returns:
            Array of boundary heights with shape (...) matching input points
        """
        points = prepare_points(points)
        x, y = points[..., 0], points[..., 1]

        # Extract parameters
        phi = params['nozzle_angle']  # φ
        H = params['nozzle_height']  # H
        r0 = params['nozzle_radius']  # r₀
        theta = params['powder_divergence_angle']  # θ

        # Check if nozzle is coaxial (φ = 90°)
        is_coaxial = np.isclose(phi, np.pi / 2, rtol=1e-10)

        if is_coaxial:
            # Calculate radial distance for coaxial case
            r = np.sqrt(x ** 2 + y ** 2)
            # Calculate boundary height
            return H + (r0 - r) / np.tan(theta)
        else:
            # Calculate K constant for lateral nozzle
            K = H + r0 * np.sin(phi) / np.tan(theta)
            # Calculate boundary height for lateral nozzle
            return K + (y - K / np.tan(phi)) * np.tan(phi - theta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from field_visualization import ThermalPlotter
    from process_parameters import set_params

    params = set_params()

    # Create visualization grid
    x = np.linspace(-0.005, 0.005, 100)  # ±5mm
    y = np.linspace(-0.005, 0.010, 200)  # -5mm to 10mm
    X, Y = np.meshgrid(x, y)

    # Stack coordinates for vectorized calculation
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    # Calculate powder concentration and boundary across the grid
    concentration = YuzeHuangPowderStream.powder_concentration(points, params)
    boundary = YuzeHuangPowderStream.is_within_powder_stream_radius(points, params)

    # Create single plot with adjusted figure size for equal aspect ratio
    # Since y range is 1.5x the x range, make height 1.5x the width
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot powder concentration
    plotter = ThermalPlotter(melting_temp=None)
    colorbar = plotter.create_temperature_pixels(
        ax, X * 1000, Y * 1000, concentration,
        label='Powder Concentration (kg/m³)',
        interpolation='bilinear'
    )

    # Add powder stream boundary
    ax.contour(X * 1000, Y * 1000, boundary, levels=[0.5],
               colors='purple', linestyles='--', linewidths=2,
               label='Stream Boundary')

    # Add nozzle position indicator
    nozzle_x = 0
    nozzle_y = params['nozzle_height'] * 1000  # Convert to mm
    ax.plot(nozzle_x, nozzle_y, 'rv', markersize=10, label='Nozzle Position')

    # Customize plot
    ax.set_title('Powder Stream Analysis')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend()

    # Set equal aspect ratio to ensure square grid cells
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()