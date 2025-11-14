import numpy as np


def grid_slice(t: float, scan_speed: float, range1: tuple, range2: tuple, get_temp: callable,
               get_temp_args: tuple = None, pixels_per_m: float = 10_000, plane: str = 'xy',
               center_on_source: bool = True) -> tuple:
    """
    Creates a temperature map by sampling temperatures across a 2D plane in 3D space.
    Uses vectorized operations for efficient computation with array-based temperature functions.

    Args:
        t (float): Time point to evaluate (s)
        scan_speed (float): Speed of the heat source movement (m/s)
        range1 (tuple): Physical bounds (min, max) for first coordinate (m)
        range2 (tuple): Physical bounds (min, max) for second coordinate (m)
        get_temp (callable): Temperature calculation function that accepts (points, time, *args)
                           where points is a numpy array of shape (..., 3)
        get_temp_args (tuple, optional): Additional arguments passed to get_temp function
        pixels_per_m (float, optional): Spatial resolution of the temperature map. Higher values
            provide more detailed visualization but increase computation time. Defaults to 10000.
        plane (str, optional): Orientation of the sampling plane. Options:
            'xy': Surface plane (z=0)
            'xz': Vertical plane following heat source (y=v*t)
            'yz': Vertical symmetry plane (x=0)
        center_on_source (bool, optional): If True, centers the grid around the current heat source
            position by shifting the coordinate ranges. Default is False.

    Returns:
        tuple: (grid1, grid2, temperatures)
            grid1, grid2: Coordinate meshgrids
            temperatures: 2D array of calculated temperatures, shaped to match the grids

    Example:
        >>> x_range = (-0.001, 0.001)  # 2mm wide
        >>> z_range = (0, 0.001)       # 1mm deep
        >>> grid1, grid2, temperatures = grid_slice(
        ...     0.001, 0.1, x_range, z_range, get_temp, plane='xz', center_on_source=True
        ... )

    Notes:
        - The function expects the temperature calculation function to handle vectorized inputs
        - When center_on_source is True, the y-coordinate ranges are shifted to center around
          the current heat source position (y = v*t)
    """
    # Calculate current source position
    y_pos = scan_speed * t

    # Adjust ranges if centering on source
    if center_on_source and plane in ['xy', 'yz']:
        if plane == 'xy':
            range2 = (range2[0] + y_pos, range2[1] + y_pos)
        elif plane == 'yz':
            range1 = (range1[0] + y_pos, range1[1] + y_pos)

    # Resolution is determined by physical size to maintain consistent pixel density
    n_points1 = int((range1[1] - range1[0]) * pixels_per_m)
    n_points2 = int((range2[1] - range2[0]) * pixels_per_m)

    # Linear spacing ensures uniform sampling density across the plane
    coord_range1 = np.linspace(range1[0], range1[1], n_points1)
    coord_range2 = np.linspace(range2[0], range2[1], n_points2)

    # Create grid points as 3D array of shape (n_points2, n_points1, 3)
    if plane == 'xy':
        x_grid, y_grid = np.meshgrid(coord_range1, coord_range2)
        points = np.stack([x_grid, y_grid, np.zeros_like(x_grid)], axis=-1)
    elif plane == 'xz':
        x_grid, z_grid = np.meshgrid(coord_range1, coord_range2)
        y_grid = np.full_like(x_grid, y_pos)
        points = np.stack([x_grid, y_grid, z_grid], axis=-1)
    elif plane == 'yz':
        y_grid, z_grid = np.meshgrid(coord_range1, coord_range2)
        points = np.stack([np.zeros_like(y_grid), y_grid, z_grid], axis=-1)
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

    # Vectorized temperature calculation
    get_temp_args = get_temp_args or ()
    temperatures = get_temp(points, t, *get_temp_args)

    # Return the appropriate coordinate grids and temperatures
    if plane == 'xy':
        return x_grid, y_grid, temperatures
    elif plane == 'xz':
        return x_grid, z_grid, temperatures
    elif plane == 'yz':
        return y_grid, z_grid, temperatures
