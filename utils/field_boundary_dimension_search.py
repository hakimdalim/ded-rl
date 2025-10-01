from typing import Callable, Tuple, Dict, Union
import numpy as np
from scipy.optimize import brentq


def find_symmetric_boundary_dimensions(
        scalar_field: Callable[[Tuple[float, float, float]], float],
        center_point: Tuple[float, float, float],
        threshold: float,
        search_radius: float = 0.1,
        tolerance: float = 1e-6,
        axes: Tuple[str, ...] = ('x', 'y', 'z')
) -> Dict[str, float]:
    """Find symmetric boundary dimensions from a center point where scalar field equals threshold.

    Args:
        scalar_field: Function that takes a point (x,y,z) and returns a scalar value
        center_point: (x,y,z) coordinates of the center point to search from
        threshold: Value that defines the boundary
        search_radius: Maximum distance to search in each direction
        tolerance: Convergence tolerance for root finding
        axes: Which axes to search along ('x', 'y', 'z')

    Returns:
        Dictionary mapping axis names to full dimensions (2 * distance from center)

    Example:
        def temperature_field(point):
            return some_temp_calculation(point)

        dims = find_symmetric_boundary_dimensions(
            temperature_field,
            center_point=(0,0,0),
            threshold=1500,  # e.g., melting temperature
            search_radius=0.01
        )
        width, length_between, depth = dims['x'], dims['y'], dims['z']
    """
    # Mapping of axis names to direction vectors
    direction_vectors = {
        'x': (1, 0, 0),
        'y': (0, 1, 0),
        'z': (0, 0, 1)
    }

    def find_boundary_in_direction(
            start_point: Tuple[float, float, float],
            direction: Tuple[float, float, float],
            _axis: str
    ) -> float:
        """Find distance to boundary along given direction vector."""

        def value_difference(dist: float) -> float:
            # Calculate scalar field value at distance and subtract threshold
            point = tuple(p + d * dist for p, d in zip(start_point, direction))
            return scalar_field(point) - threshold

        try:
            # Use Brent's method to find the root (where value = threshold)
            return brentq(value_difference, 0, search_radius, rtol=tolerance)
        except Exception as e:
            # Calculate values at center and search radius for diagnostics
            center_value = float(scalar_field(start_point))
            edge_point = tuple(p + d * search_radius for p, d in zip(start_point, direction))
            edge_value = float(scalar_field(edge_point))

            raise Exception(
                f"No boundary found for axis '{_axis}' within search radius {search_radius}.\n"
                f"Values:\n"
                f"  - Center ({', '.join(f'{x:.3e}' for x in start_point)}): {center_value:.3e}\n"
                f"  - Edge ({', '.join(f'{x:.3e}' for x in edge_point)}): {edge_value:.3e}\n"
                f"  - Target threshold: {threshold:.3e}\n"
                f"Original error: {str(e)}"
            ) from e

    # Calculate dimensions along requested axes
    dimensions = {}
    for axis in axes:
        if axis not in direction_vectors:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of: {tuple(direction_vectors.keys())}")

        # Search in positive direction and double (assuming symmetry)
        direction = direction_vectors[axis]
        positive_dist = find_boundary_in_direction(center_point, direction, axis)
        dimensions[axis] = 2 * positive_dist

    return dimensions


if __name__ == '__main__':

    # Example usage in fast_melt_pool_dims:
    def fast_melt_pool_dims(
            t: float,
            params: dict,
            get_temp: Callable,
            search_radius: float = 0.01,
            tolerance: float = 1e-6
    ) -> Tuple[float, float, float]:
        """Calculate melt pool dimensions assuming symmetry around center point."""

        # Center is at laser position
        center = (0.0, params['scan_speed'] * t, 0.0)

        # Create temperature field function with fixed time and params
        def temp_field(point: Tuple[float, float, float]) -> float:
            return get_temp(point, t, params)

        # Find dimensions where temperature equals melting point
        dims = find_symmetric_boundary_dimensions(
            scalar_field=temp_field,
            center_point=center,
            threshold=params['melting_temp'],
            search_radius=search_radius,
            tolerance=tolerance
        )

        return dims['x'], dims['y'], dims['z']

    # Example test with a simple Gaussian field
    def gaussian_field(point: Tuple[float, float, float]) -> float:
        """Simple 3D Gaussian centered at origin."""
        x, y, z = point
        r2 = x * x + y * y + z * z
        return np.exp(-r2 / 0.01)


    # Find boundary where field equals 0.5
    dimensions = find_symmetric_boundary_dimensions(
        scalar_field=gaussian_field,
        center_point=(0, 0, 0),
        threshold=0.5,
        search_radius=1.0
    )

    print("Gaussian field dimensions at value 0.5:")
    for axis, size in dimensions.items():
        print(f"{axis}: {size * 1000:.3f} mm")