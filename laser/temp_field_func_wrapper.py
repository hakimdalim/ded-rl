import numpy as np
from typing import Callable, Dict, Tuple, Literal
from utils.coordinate_transform import ToLocalCoordinateSystem


class TemperatureFieldWrapper:
    """Wrapper for moving heat source solutions with fixed coordinate transformation.

    This class wraps analytical solutions for moving heat sources (like Rosenthal or
    Yuza-Huang) and handles all necessary coordinate transformations. It maintains a fixed
    coordinate system transformation based on the initial parameters and movement axis
    of the heat source solution.

    The coordinate systems are defined as follows:

    Global coordinates:
    - Origin and orientation as defined by the process setup
    - Heat source moves at arbitrary angle defined by movement_angle
    - Start position can be anywhere in space

    Local coordinates (specific to each heat source solution):
    - Origin at the process start position
    - Heat source moves along x or y axis (defined by the solution's metadata)
    - Movement is always in positive direction along that axis
    - z-axis remains vertical (unchanged)

    The transformation between these coordinate systems is fixed at initialization
    and includes:
    1. Translation to align origins
    2. Rotation to align movement direction with the solution's expected axis

    Example:
        >>> from process_parameters import set_params
        >>> from thermal.temperature_change import rosenthal_temperature
        >>> import numpy as np
        >>>
        >>> # Set up process parameters
        >>> params = set_params(
        ...     laser_power=650,  # W
        ...     scan_speed=0.003,  # m/s
        ...     thermal_diffusivity=1.172e-5,  # m²/s
        ... )
        >>>
        >>> # Create wrapper for Rosenthal solution at t=1s moving at 45 degrees
        >>> field = TemperatureFieldWrapper(
        ...     get_temp=rosenthal_temperature,  # Must have @moving_heat_source decorator
        ...     t=1.0,
        ...     params=params,
        ...     start_position=(0.002, 0.002, 0),
        ...     movement_angle=np.pi/4
        ... )
        >>>
        >>> # Create points for temperature calculation
        >>> x = np.linspace(-0.005, 0.005, 100)  # ±5mm
        >>> y = np.linspace(-0.005, 0.005, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = np.zeros_like(X)  # z=0 plane
        >>> points = np.stack([X, Y, Z], axis=-1)
        >>>
        >>> # Calculate temperatures
        >>> temperatures = field(points)
        >>>
        >>> # Get current heat source position in global coordinates
        >>> pos = field.heat_source_position
    """

    def __init__(self, get_temp: Callable, t: float, params: Dict,
                 start_position: Tuple[float, float, float],
                 movement_angle: float):
        """Initialize wrapper with heat source solution and fixed process parameters.

        Args:
            get_temp: Moving heat source solution function. Must be decorated with
                @moving_heat_source to specify its movement axis.
            t: Fixed time point for temperature calculation (s)
            params: Process parameters dictionary containing at minimum 'scan_speed'
                and any other parameters required by the heat source solution
            start_position: Process starting position in global coordinates (x, y, z)
            movement_angle: Angle of movement direction in global coordinates (rad),
                measured counter-clockwise from x-axis in x-y plane

        Raises:
            AttributeError: If get_temp lacks the @moving_heat_source decorator and
                associated metadata specifying its movement axis
        """
        if not hasattr(get_temp, 'metadata'):
            raise AttributeError(
                "Temperature function must be decorated with @moving_heat_source "
                "to specify its movement axis"
            )

        self.get_temp = get_temp
        self.t = t
        self.params = params

        # Get movement axis from function metadata
        self.movement_axis = get_temp.metadata.movement_axis
        local_rotation = 0.0 if self.movement_axis == 'x' else np.pi / 2

        # Create and store the fixed coordinate transformation
        self.transform = ToLocalCoordinateSystem(
            position=start_position,
            rotation=movement_angle + local_rotation
        )

    @property
    def heat_source_position(self) -> np.ndarray:
        """Calculate current heat source position in global coordinates.

        The position is calculated by:
        1. Computing displacement from origin in local coordinates based on:
           - Current time
           - Process speed
           - Movement axis of the heat source solution
        2. Transforming this local position back to global coordinates

        Returns:
            Array of shape (3,) containing the heat source position (x, y, z)
            in global coordinates
        """
        # In local coordinates, the heat source moves along the specified axis
        displacement = self.t * self.params['scan_speed']

        if self.movement_axis == 'x':
            local_position = np.array([displacement, 0.0, 0.0])
        else:  # y-axis
            local_position = np.array([0.0, displacement, 0.0])

        # Transform back to global coordinates
        return self.transform.inverse(local_position)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Calculate temperatures for points in global coordinate system.

        The calculation process:
        1. Transform input points from global to local coordinates
        2. Calculate temperatures using the wrapped heat source solution
        3. Return temperatures (no transform needed for scalar values)

        Args:
            points: Array of shape (..., 3) containing points in global coordinates
                   where temperatures should be calculated

        Returns:
            Array of temperatures with same shape as input points (minus last dimension)
        """
        local_points = self.transform(points)
        temperatures = self.get_temp(local_points, self.t, self.params)
        return temperatures


def wrap_temp_field(temp_func: Callable, movement_axis: Literal['x', 'y'] = 'x') -> TemperatureFieldWrapper:
    """Convenience function to create a wrapped temperature field function.

    Args:
        temp_func: Temperature field function to wrap
        movement_axis: Axis along which the function expects movement ('x' or 'y')

    Returns:
        Wrapped function that handles coordinate transformations
    """
    return TemperatureFieldWrapper(temp_func, movement_axis)


if __name__ == '__main__':
    import numpy as np
    from thermal.temperature_change import (
        rosenthal_temperature, YuzaHuangTemperature,
        EagarTsaiTemperature
    )
    from process_parameters import set_params
    import matplotlib.pyplot as plt

    # Create test parameters
    params = set_params(
        laser_power=650,  # W
        scan_speed=0.003,  # m/s
        thermal_diffusivity=1.172e-5,  # m²/s
    )

    # Create a grid of points in global coordinates
    x = np.linspace(-0.005, 0.005, 100)  # ±5mm
    y = np.linspace(-0.005, 0.005, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # z=0 plane
    global_points = np.stack([X, Y, Z], axis=-1)

    # Define plot limits in mm
    xlim = ylim = (-5, 5)  # ±5mm

    # Test cases with different angles and positions
    test_cases = [
        {'angle': 0, 'position': (0, 0, 0), 'title': 'Moving along X-axis'},
        {'angle': np.pi / 4, 'position': (0, 0, 0), 'title': '45° angle'},
        {'angle': np.pi / 2, 'position': (0, 0, 0), 'title': 'Moving along Y-axis'},
        {'angle': np.pi / 4, 'position': (0.002, 0.002, 0), 'title': '45° angle, track_center position'}
    ]

    '''# First plot: Rosenthal solution test cases
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
    axes1 = axes1.flatten()
    fig1.suptitle('Rosenthal Temperature Field')

    for ax, case in zip(axes1, test_cases):
        # Create wrapped temperature field for this case
        rosenthal_field = TemperatureFieldWrapper(
            get_temp=rosenthal_temperature,
            t=0,  # t=0 for initial position
            params=params,
            start_position=case['position'],
            movement_angle=case['angle']
        )

        # Calculate temperatures
        temperatures = rosenthal_field(global_points)

        # Plot temperature field
        im = ax.contourf(X * 1000, Y * 1000, temperatures,
                         levels=np.linspace(0, 3000, 20),
                         cmap='hot')
        ax.set_title(case['title'])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

        # Set equal aspect ratio and consistent limits
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Add arrow to show movement direction
        arrow_length = 0.002
        dx = arrow_length * np.cos(case['angle'])
        dy = arrow_length * -np.sin(case['angle'])
        ax.arrow(case['position'][0] * 1000, case['position'][1] * 1000,
                 dx * 1000, dy * 1000,
                 head_width=0.2, head_length=0.3, fc='white', ec='white')

        # Add marker for current heat source position
        heat_pos = rosenthal_field.heat_source_position
        ax.scatter(heat_pos[0] * 1000, heat_pos[1] * 1000,
                   color='cyan', s=100, marker='x', linewidth=2,
                   label='Heat Source Position')

    plt.colorbar(im, ax=axes1, label='Temperature (K)')
    axes1[0].legend()  # Add legend to first subplot
    plt.show()'''

    # Second plot: EagarTsai solution test cases
    eager_tsai = EagarTsaiTemperature()
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    axes2 = axes2.flatten()
    fig2.suptitle('EagarTsai Temperature Field')

    for ax, case in zip(axes2, test_cases):
        # Create wrapped temperature field for this case
        eager_tsai_field = TemperatureFieldWrapper(
            get_temp=eager_tsai.delta_temperature,
            t=0.01,
            params=params,
            start_position=case['position'],
            movement_angle=case['angle']
        )

        # Calculate temperatures
        temperatures = eager_tsai_field(global_points)

        # Plot temperature field
        im = ax.contourf(X * 1000, Y * 1000, temperatures,
                         levels=np.linspace(0, 3000, 20),
                         cmap='hot')
        ax.set_title(case['title'])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

        # Set equal aspect ratio and consistent limits
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Add arrow to show movement direction
        arrow_length = 0.002
        dx = arrow_length * np.cos(case['angle'])
        dy = arrow_length * -np.sin(case['angle'])
        ax.arrow(case['position'][0] * 1000, case['position'][1] * 1000,
                 dx * 1000, dy * 1000,
                 head_width=0.2, head_length=0.3, fc='white', ec='white')

        # Add marker for current heat source position
        heat_pos = eager_tsai_field.heat_source_position
        ax.scatter(heat_pos[0] * 1000, heat_pos[1] * 1000,
                   color='cyan', s=100, marker='x', linewidth=2,
                   label='Heat Source Position')

    plt.colorbar(im, ax=axes2, label='Temperature (K)')
    axes2[0].legend()  # Add legend to first subplot
    plt.show()

    # Test temperature values at specific points
    print("\nTesting temperature values at specific points:")
    test_point = np.array([[0.001, 0, 0]])  # 1mm in front of source

    '''# Second plot: Yuza-Huang solution test cases
    yuza_huang = YuzaHuangTemperature()
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    axes2 = axes2.flatten()
    fig2.suptitle('Yuza-Huang Temperature Field')

    for ax, case in zip(axes2, test_cases):
        # Create wrapped temperature field for this case
        yuza_huang_field = TemperatureFieldWrapper(
            get_temp=yuza_huang.delta_temperature,
            t=1,
            params=params,
            start_position=case['position'],
            movement_angle=case['angle']
        )

        # Calculate temperatures
        temperatures = yuza_huang_field(global_points)

        # Plot temperature field
        im = ax.contourf(X * 1000, Y * 1000, temperatures,
                         levels=np.linspace(0, 3000, 20),
                         cmap='hot')
        ax.set_title(case['title'])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

        # Set equal aspect ratio and consistent limits
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Add arrow to show movement direction
        arrow_length = 0.002
        dx = arrow_length * np.cos(case['angle'])
        dy = arrow_length * -np.sin(case['angle'])
        ax.arrow(case['position'][0] * 1000, case['position'][1] * 1000,
                 dx * 1000, dy * 1000,
                 head_width=0.2, head_length=0.3, fc='white', ec='white')

        # Add marker for current heat source position
        heat_pos = yuza_huang_field.heat_source_position
        ax.scatter(heat_pos[0] * 1000, heat_pos[1] * 1000,
                   color='cyan', s=100, marker='x', linewidth=2,
                   label='Heat Source Position')

    plt.colorbar(im, ax=axes2, label='Temperature (K)')
    axes2[0].legend()  # Add legend to first subplot
    plt.show()

    # Test temperature values at specific points
    print("\nTesting temperature values at specific points:")
    test_point = np.array([[0.001, 0, 0]])  # 1mm in front of source'''

    print("\nRosenthal Solution:")
    for case in test_cases:
        rosenthal_field = TemperatureFieldWrapper(
            get_temp=rosenthal_temperature,
            t=0,
            params=params,
            start_position=case['position'],
            movement_angle=case['angle']
        )
        temp = rosenthal_field(test_point)
        print(f"{case['title']}:")
        print(f"Temperature at test point: {temp[0]:.1f} K")

    '''print("\nYuza-Huang Solution:")
    for case in test_cases:
        yuza_huang_field = TemperatureFieldWrapper(
            get_temp=yuza_huang.delta_temperature,
            t=0.1,
            params=params,
            start_position=case['position'],
            movement_angle=case['angle']
        )
        temp = yuza_huang_field(test_point)
        print(f"{case['title']}:")
        print(f"Temperature at test point: {temp[0]:.1f} K")'''

    print("\nEag Solution:")
    eagar_tsai = EagarTsaiTemperature()
    for case in test_cases:
        eagar_tsai_field = TemperatureFieldWrapper(
            get_temp=eagar_tsai.delta_temperature,
            t=0.1,
            params=params,
            start_position=case['position'],
            movement_angle=case['angle']
        )
        temp = eagar_tsai_field(test_point)
        print(f"{case['title']}:")
        print(f"Temperature at test point: {temp[0]:.1f} K")