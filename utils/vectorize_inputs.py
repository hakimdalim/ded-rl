import numpy as np
from typing import Any, Union, Tuple


def prepare_points(points: Any) -> np.ndarray:
    """
    Validates and prepares points array for calculations. Handles various input formats
    and ensures output is a properly shaped numpy array with last dimension being 3 (x,y,z).

    Args:
        points: Points in various formats:
            - Single point as tuple/list: (x,y,z) or [x,y,z]
            - Single point as 1D array: array([x,y,z])
            - Multiple points as 2D array: array([[x1,y1,z1], [x2,y2,z2], ...])
            - N-dimensional array with last dim 3: array with shape (..., 3)

    Returns:
        np.ndarray with shape (..., 3). Single points are converted to shape (1, 3)

    Raises:
        ValueError: If input cannot be converted to valid points array

    Examples:
        >>> prepare_points([1, 2, 3])
        array([[1., 2., 3.]])

        >>> prepare_points([[1, 2, 3], [4, 5, 6]])
        array([[1., 2., 3.],
               [4., 5., 6.]])

        >>> prepare_points(np.zeros((2, 3, 4, 3)))  # Complex shape
        array([[[[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]],
               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]],
               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]]])
    """
    # Convert input to numpy array
    points_array = np.asarray(points, dtype=float)

    # Handle 1D array (single point)
    if points_array.ndim == 1:
        if points_array.shape[0] != 3:
            raise ValueError(f"Single point must have exactly 3 coordinates, got {points_array.shape[0]}")
        return points_array.reshape(1, 3)

    # Handle N-dimensional arrays
    if points_array.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3 (x,y,z), got shape {points_array.shape}")

    return points_array


def test_prepare_points():
    """Test the prepare_points function with various inputs"""
    # Test single point inputs
    assert np.array_equal(prepare_points([1, 2, 3]), np.array([[1., 2., 3.]]))
    assert np.array_equal(prepare_points((1, 2, 3)), np.array([[1., 2., 3.]]))
    assert np.array_equal(prepare_points(np.array([1, 2, 3])), np.array([[1., 2., 3.]]))

    # Test 2D array inputs
    points_2d = [[1, 2, 3], [4, 5, 6]]
    expected_2d = np.array([[1., 2., 3.], [4., 5., 6.]])
    assert np.array_equal(prepare_points(points_2d), expected_2d)

    # Test higher dimensional arrays
    shape_3d = (2, 2, 3)
    points_3d = np.zeros(shape_3d)
    assert prepare_points(points_3d).shape == shape_3d

    shape_4d = (2, 3, 4, 3)
    points_4d = np.zeros(shape_4d)
    assert prepare_points(points_4d).shape == shape_4d

    # Test error cases
    try:
        prepare_points([1, 2])  # Not enough coordinates
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    try:
        prepare_points([[1, 2], [3, 4]])  # Wrong last dimension
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    try:
        prepare_points("invalid")  # Invalid input type
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("All tests passed!")


if __name__ == "__main__":
    test_prepare_points()