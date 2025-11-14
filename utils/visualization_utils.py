import functools
import inspect
import time

import matplotlib.pyplot as plt
from typing import Callable, Any, Dict, Tuple, List

import numpy as np

class AxisMapping:
    """Utility class for handling axis mappings and labels."""

    DEFAULT_MAPPING = {'x': 0, 'y': 1, 'z': 2}

    @classmethod
    def get_mapping(cls, custom_mapping: Dict[str, int] = None) -> Dict[str, int]:
        """Get the axis mapping, using default if none provided."""
        return custom_mapping if custom_mapping is not None else cls.DEFAULT_MAPPING

    @classmethod
    def get_reverse_mapping(cls, custom_mapping: Dict[str, int] = None) -> Dict[int, str]:
        """Get mapping from indices to uppercase axis names."""
        mapping = cls.get_mapping(custom_mapping)
        return {v: k.upper() for k, v in mapping.items()}

    @classmethod
    def get_slice_axes(cls, main_axis: int, custom_mapping: Dict[str, int] = None) -> List[int]:
        """Get the remaining axes when slicing along main_axis."""
        mapping = cls.get_mapping(custom_mapping)
        all_axes = set(mapping.values())
        return sorted(list(all_axes - {main_axis}))

    @classmethod
    def get_axis_labels(cls, main_axis: int, custom_mapping: Dict[str, int] = None) -> Tuple[str, str]:
        """Get appropriate axis labels for plotting a slice."""
        slice_axes = cls.get_slice_axes(main_axis, custom_mapping)
        reverse_mapping = cls.get_reverse_mapping(custom_mapping)
        return f'{reverse_mapping[slice_axes[0]]}', f'{reverse_mapping[slice_axes[1]]}'


def ensure_ax(func: Callable) -> Callable:
    """Decorator that ensures a matplotlib axis is available for plotting.

    This decorator checks if the decorated function has an 'ax' parameter. If 'ax' is None:
    1. If subplot_kwargs is provided in the function arguments, uses these for creating the subplot
    2. Otherwise creates a default subplot

    If maybe_plot_direct=True and ax=None, the plot will be displayed immediately after creation.

    Args:
        func: The function to decorate. Must have an 'ax' parameter that accepts a matplotlib Axes object.

    Returns:
        Callable: The wrapped function with guaranteed non-None axes.

    Example usage:
    >>> @ensure_ax
    ... def plot_data(data, ax=None, subplot_kwargs=None, maybe_plot_direct=False):
    ...     ax.plot(data)
    ...     return ax
    ...
    >>> # Creates default subplot
    >>> plot_data(data)
    ...
    >>> # Creates and shows plot immediately
    >>> plot_data(data, maybe_plot_direct=True)
    ...
    >>> # Creates customized subplot
    >>> plot_data(data, subplot_kwargs={'figsize': (10, 5)})
    ...
    >>> # Use existing axis
    >>> fig, ax = plt.subplots()
    >>> plot_data(data, ax=ax)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        if bound.arguments['ax'] is None:
            subplot_kwargs = bound.arguments.get('subplot_kwargs', {}) or {}
            _, bound.arguments['ax'] = plt.subplots(**subplot_kwargs)

            # Check if maybe_plot_direct is True and show plot immediately if it is
            maybe_plot_direct = bound.arguments.get('maybe_plot_direct', False)
            result = func(*bound.args, **bound.kwargs)

            if maybe_plot_direct:
                plt.show()
            return result

        return func(*bound.args, **bound.kwargs)

    # Add information about decorator behavior to function signature
    if wrapper.__doc__ is None:
        wrapper.__doc__ = ""
    wrapper.__doc__ += "\n\nNote: This function has been decorated with @ensure_axis which guarantees a matplotlib axis will be available."

    return wrapper


def standardize_voxel_size(func: Callable) -> Callable:
    """Decorator that standardizes voxel size input to a numpy array.

    Converts input voxel_size from various formats to a consistent (3,) numpy array:
    1. If scalar (int/float) -> uniform (float, float, float)
    2. If tuple/list/array -> ensures 3 elements and float type

    Args:
        func: The function to decorate. Must have a 'voxel_size' parameter.

    Returns:
        Callable: The wrapped function with standardized voxel_size parameter.

    Example usage:
    >>> @standardize_voxel_size
    ... def process_volume(data, voxel_size):
    ...     # voxel_size is guaranteed to be (3,) float array
    ...     return data * voxel_size
    ...
    >>> # All these calls work:
    >>> process_volume(data, 0.1)  # -> array([0.1, 0.1, 0.1])
    >>> process_volume(data, (0.1, 0.2, 0.3))  # -> array([0.1, 0.2, 0.3])
    >>> process_volume(data, [0.1, 0.2, 0.3])  # -> array([0.1, 0.2, 0.3])
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        voxel_size = bound.arguments['voxel_size']
        if isinstance(voxel_size, (int, float)):
            bound.arguments['voxel_size'] = np.array([float(voxel_size)] * 3)
        else:
            bound.arguments['voxel_size'] = np.asarray(voxel_size, dtype=float)

        if bound.arguments['voxel_size'].size != 3:
            raise ValueError("voxel_size must have exactly 3 elements")

        return func(*bound.args, **bound.kwargs)

    if wrapper.__doc__ is None:
        wrapper.__doc__ = ""
    wrapper.__doc__ += "\n\nNote: This function has been decorated with @standardize_voxel_size which converts voxel_size to a (3,) numpy array."

    return wrapper

def get_plane_info(plane: str) -> Dict[str, Any]:
    """Helper function that returns standardized information for a given plane.

    Provides a consistent interface for handling different plane orientations
    in volume visualizations and processing.

    Args:
        plane: String identifier for the plane ('xy', 'xz', or 'yz')

    Returns:
        dict with plane information containing:
            - 'axes': tuple of axis indices (e.g., (0,1) for xy-plane)
            - 'default_idx': index of normal axis
            - 'labels': tuple of axis labels
            - 'name': formatted plane name

    Raises:
        ValueError: If plane identifier is not recognized

    Example usage:
    >>> info = get_plane_info('xy')
    >>> plot_indices = info['axes']  # (0, 1)
    >>> normal_idx = info['default_idx']  # 2
    >>> xlabel, ylabel = info['labels']  # ('X', 'Y')
    >>> title = f"{info['name']} slice"  # "XY-plane slice"
    """
    planes = {
        'xy': {
            'axes': (0, 1),
            'default_idx': 2,
            'labels': ('X', 'Y'),
            'name': 'XY-plane'
        },
        'xz': {
            'axes': (0, 2),
            'default_idx': 1,
            'labels': ('X', 'Z'),
            'name': 'XZ-plane'
        },
        'yz': {
            'axes': (1, 2),
            'default_idx': 0,
            'labels': ('Y', 'Z'),
            'name': 'YZ-plane'
        }
    }

    if plane.lower() not in planes:
        raise ValueError(f"Invalid plane '{plane}'. Must be one of {list(planes.keys())}")

    return planes[plane.lower()]


def validate_axis(func: Callable) -> Callable:
    """Decorator that validates and converts axis inputs.

    This decorator:
    1. Checks if the decorated function has an 'axis' parameter
    2. If axis is a string, converts it using axis_mapping
    3. If no axis_mapping provided, uses default (x=0, y=1, z=2)

    Args:
        func: The function to decorate. Must have an 'axis' parameter.

    Returns:
        Callable: The wrapped function with validated axis parameter.

    Example usage:
    >>> @validate_axis
    ... def get_slice(volume, index, axis=2, axis_mapping=None):
    ...     return np.take(volume, index, axis=axis)
    ...
    >>> # Use with numeric axis
    >>> get_slice(volume, 0, axis=2)
    >>> # Use with string axis
    >>> get_slice(volume, 0, axis='z')
    >>> # Use with custom mapping
    >>> get_slice(volume, 0, axis='depth', axis_mapping={'depth': 2})
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        # Get axis and mapping from arguments
        axis = bound.arguments.get('axis')
        axis_mapping = AxisMapping.get_mapping(bound.arguments.get('axis_mapping'))

        # Convert string axis to integer if needed
        if isinstance(axis, str):
            axis = axis.lower()  # Make case-insensitive
            if axis not in axis_mapping:
                raise ValueError(f"Invalid axis string '{axis}'. Must be one of {list(axis_mapping.keys())}")
            bound.arguments['axis'] = axis_mapping[axis]
        elif not isinstance(axis, int):
            raise TypeError(f"Axis must be string or integer, got {type(axis)}")

        return func(*bound.args, **bound.kwargs)

    # Add information about decorator behavior
    if wrapper.__doc__ is None:
        wrapper.__doc__ = ""
    wrapper.__doc__ += "\n\nNote: This function has been decorated with @validate_axis which handles string-based axis inputs."

    return wrapper


@validate_axis
def find_surface(volume: np.ndarray, axis: int = 2, from_top: bool = True,
                 axis_mapping: Dict[str, int] = None) -> np.ndarray:
    """Find the surface indices of a boolean volume along specified axis."""
    # Create a mask where there are no True values along the axis
    no_activation = ~np.any(volume, axis=axis)

    if from_top:
        # For top surface, flip the axis and adjust the index
        surface = np.argmax(np.flip(volume, axis=axis), axis=axis)
        surface = volume.shape[axis] - 1 - surface
    else:
        # For bottom surface, just find first True
        surface = np.argmax(volume, axis=axis)

    # Set invalid positions to 0
    surface[no_activation] = 0

    return surface


@validate_axis
def get_slice(volume: np.ndarray, index: int, axis: int = 2,
              axis_mapping: Dict[str, int] = None) -> np.ndarray:
    """Extract a slice from the volume at a specific index along the given axis."""
    return np.take(volume, index, axis=axis)

@ensure_ax
def plot_im(plt_data: np.ndarray, xlabel: str, ylabel: str, ax=None,
            subplot_kwargs=None, colorbar_kwargs=None) -> plt.Axes:

    im = ax.imshow(plt_data, origin='lower')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, **(colorbar_kwargs or {}))
    return ax

@ensure_ax
@validate_axis
def plot_slice(volume: np.ndarray, index: int, axis: int = 2,
               ax=None, axis_mapping: Dict[str, int] = None,
               subplot_kwargs=None, colorbar_kwargs=None) -> plt.Axes:
    """Plot a slice from the volume with proper orientation."""
    return plot_im(
        get_slice(volume, index, axis, axis_mapping).T,
        *AxisMapping.get_axis_labels(axis, axis_mapping),
        ax=ax,
        subplot_kwargs=subplot_kwargs,
        colorbar_kwargs=colorbar_kwargs
    )

@ensure_ax
@validate_axis
def plot_surface(volume: np.ndarray, axis: int = 2, from_top: bool = True,
                 ax=None, axis_mapping: Dict[str, int] = None,
                 subplot_kwargs=None, colorbar_kwargs=None) -> plt.Axes:
    """Plot the surface indices along the specified axis."""
    return plot_im(
        find_surface(volume, axis, from_top, axis_mapping).T,
        *AxisMapping.get_axis_labels(axis, axis_mapping),
        ax=ax,
        subplot_kwargs=subplot_kwargs,
        colorbar_kwargs=colorbar_kwargs
    )


@validate_axis
def sample_at_surface(volume: np.ndarray, surface_indices: np.ndarray, axis: int = 2,
                      axis_mapping: Dict[str, int] = None) -> np.ndarray:
    """
    Sample values from a volume at specified surface indices along an axis.
    """
    print('started')
    print('surface_indices.shape', surface_indices.shape)
    print('volume.shape', volume.shape)
    print('axis', axis)
    start = time.time()

    # Get indices for each axis position
    i, j = np.indices(surface_indices.shape)

    # Prepare the index arrays based on which axis we're sampling along
    if axis == 0:
        idx = (surface_indices, i, j)
    elif axis == 1:
        idx = (i, surface_indices, j)
    else:  # axis == 2
        idx = (i, j, surface_indices)

    # Sample the values
    v = volume[idx]
    print(time.time() - start)
    print(v.shape)
    return v


if __name__ == '__main__':

    import numpy as np
    from dataclasses import dataclass


    @dataclass
    class DummyTemperatureVolume:
        """Minimal implementation of temperature volume for testing."""
        shape: Tuple[int, int, int]
        temperature: np.ndarray

        @classmethod
        def create_test_volume(cls, shape=(50, 50, 25), ambient_temp=300.0):
            """Create a test volume with a simple heat source pattern."""
            temperature = np.full(shape=shape, fill_value=ambient_temp, dtype=float)

            # Create a simple gaussian heat pattern
            center = np.array(shape) // 2
            x, y, z = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )

            # Calculate distance from center
            r = np.sqrt(
                ((x - center[0]) / 10) ** 2 +
                ((y - center[1]) / 10) ** 2 +
                ((z - center[2]) / 5) ** 2
            )

            # Create temperature pattern
            temperature += 2000 * np.exp(-r ** 2)

            return cls(shape=shape, temperature=temperature)


    @dataclass
    class DummyActivationVolume:
        """Minimal implementation of activation volume for testing."""
        shape: Tuple[int, int, int]
        activated: np.ndarray

        @classmethod
        def create_test_volume(cls, shape=(50, 50, 25), substrate_height=5):
            """Create a test volume with substrate and a simple track pattern."""
            activated = np.zeros(shape=shape, dtype=bool)

            # Add substrate
            activated[:, :, :substrate_height] = True

            # Add a simple track pattern
            x = np.arange(shape[0])
            y = np.arange(shape[1])
            z = np.arange(shape[2])

            # Create track centerline
            center_x = shape[0] // 2
            center_y = shape[1] // 2
            track_y = center_y + 10 * np.sin((x - center_x) / 10)

            # Create track pattern
            for i, ty in enumerate(track_y):
                y_min = int(max(0, ty - 3))
                y_max = int(min(shape[1], ty + 3))
                z_max = min(shape[2], substrate_height + 10)
                activated[i, y_min:y_max, substrate_height:z_max] = True

            return cls(shape=shape, activated=activated)


    def create_test_setup():
        """Create test volumes with a simple track pattern."""
        # Create volumes
        shape = (50, 50, 25)  # 5x5x2.5 mm at 100μm resolution
        voxel_size = 0.0001  # 100 μm

        temp_volume = DummyTemperatureVolume.create_test_volume(
            shape=shape,
            ambient_temp=300.0
        )

        activation_volume = DummyActivationVolume.create_test_volume(
            shape=shape,
            substrate_height=5
        )

        return temp_volume, activation_volume, voxel_size



    # Create test data using our previous setup
    temp_volume, activation_volume, voxel_size = create_test_setup()

    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot different slices
    plot_slice(activation_volume.activated, index=12, axis='z', ax=ax1)
    ax1.set_title('XY Slice at z=12')

    plot_slice(activation_volume.activated, index=25, axis='y', ax=ax2)
    ax2.set_title('XZ Slice at y=25')

    # Plot surfaces
    plot_surface(activation_volume.activated, axis='z', from_top=True, ax=ax3)
    ax3.set_title('Top Surface (looking down Z)')

    plot_surface(activation_volume.activated, axis='z', from_top=False, ax=ax4)
    ax4.set_title('Bottom Surface (looking down Z)')

    plt.tight_layout()
    plt.show()

    # Let's also try with a custom mapping
    custom_mapping = {'width': 0, 'height': 1, 'depth': 2}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Same plots but with custom mapping
    plot_slice(activation_volume.activated, index=12, axis='depth',
               axis_mapping=custom_mapping, ax=ax1)
    ax1.set_title('Width-Height Slice at depth=12')

    plot_slice(activation_volume.activated, index=25, axis='height',
               axis_mapping=custom_mapping, ax=ax2)
    ax2.set_title('Width-Depth Slice at height=25')

    plot_surface(activation_volume.activated, axis='depth', from_top=True,
                 axis_mapping=custom_mapping, ax=ax3)
    ax3.set_title('Top Surface (looking down depth)')

    plot_surface(activation_volume.activated, axis='depth', from_top=False,
                 axis_mapping=custom_mapping, ax=ax4)
    ax4.set_title('Bottom Surface (looking down depth)')

    plt.tight_layout()
    plt.show()