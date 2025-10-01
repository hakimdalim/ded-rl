import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Tuple, Optional

from utils.field_boundary_dimension_search import find_symmetric_boundary_dimensions


def fast_melt_pool_dims(
    t: float,
    params: dict,
    grid_spacing: float = 0.00001,
    search_radius: float = 0.01,
    get_temp: Optional[Callable] = None,
    search_center: bool = False,
    all_directions: bool = False
) -> Tuple[float, float, float]:
    """Calculate melt pool dimensions using gradient ascent to find center.

    Args:
        t: Time point to evaluate dimensions (s)
        params: Process parameters including 'melting_temp' and 'scan_speed'
        grid_spacing: Spacing between points for center search (m)
        search_radius: Maximum radius to search from initial guess (m)
        get_temp: Temperature calculation function that takes (point, time, params)
        search_center: If True, search for maximum temperature point
        all_directions: If True, search in both positive and negative directions

    Returns:
        tuple: (width, length_between, depth)
            width: Melt pool width (x-direction) in m
            length_between: Melt pool length_between (y-direction) in m
            depth: Melt pool depth (z-direction) in m

    Raises:
        ValueError: If get_temp function is not provided
    """
    if get_temp is None:
        raise ValueError('get_temp function must be provided')

    # Initial guess for melt pool center (at laser center)
    current_center = (0.0, params['scan_speed'] * t, 0.0)

    if search_center:
        def get_neighbors(center):
            """Generate neighboring points (3x3x3 cube without center)."""
            neighbors = []
            for dx in [-grid_spacing, 0, grid_spacing]:
                for dy in [-grid_spacing, 0, grid_spacing]:
                    for dz in [-grid_spacing, 0, grid_spacing]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        point = (center[0] + dx, center[1] + dy, center[2] + dz)
                        neighbors.append(point)
            return neighbors

        # Find melt pool center by ascending to temperature maximum
        current_temp = get_temp(current_center, t, params)
        while True:
            neighbors = get_neighbors(current_center)
            neighbor_temps = [(get_temp(n, t, params), n) for n in neighbors]
            max_neighbor_temp, max_neighbor = max(neighbor_temps)

            if max_neighbor_temp <= current_temp:
                break

            current_center = max_neighbor
            current_temp = max_neighbor_temp

    # Create temperature field function with fixed time and params
    def temp_field(point: Tuple[float, float, float]) -> float:
        return get_temp(point, t, params)

    # Find dimensions where temperature equals melting point
    dims = find_symmetric_boundary_dimensions(
        scalar_field=temp_field,
        center_point=current_center,
        threshold=params['melting_temp'],
        search_radius=search_radius,
        all_directions=all_directions
    )

    return (
        dims['x'],  # width
        dims['y'],  # length_between
        dims['z']   # depth
    )


def find_melt_pool_dimension_from_grid(axis_values, temperatures_2d, T_melt):
    """
    Find the maximum dimension of the melt pool along a given axis.

    Parameters:
    axis_values: 1D array - Values along the axis we're measuring (X, Y, or Z coordinates)
    temperatures_2d: 2D array - Temperature values with measuring axis along second dimension
    T_melt: float - Melting temperature threshold

    Returns:
    float: Maximum dimension of melt pool in m (width/length_between/depth)
    """
    max_dimension = 0

    # Loop through each profile
    for temp_profile in temperatures_2d:
        # Find where temperature crosses melting point
        if np.any(temp_profile >= T_melt):
            # Use interpolation to find more precise crossing points
            f = interp1d(axis_values, temp_profile - T_melt, kind='linear')

            # Fine positions for more precise crossing detection
            pos_fine = np.linspace(axis_values[0], axis_values[-1], 1000)
            temp_fine = f(pos_fine)

            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(temp_fine)))[0]

            if len(zero_crossings) >= 2:
                # Calculate dimension between first and last crossing
                current_dim = abs(pos_fine[zero_crossings[-1]] - pos_fine[zero_crossings[0]])
                max_dimension = max(max_dimension, current_dim)

    return max_dimension
