from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ToLocalCoordinateSystem:
    """Handles transformations between world coordinates and a local coordinate system.

    The local coordinate system is defined by:
    1. An origin point in world coordinates (the position)
    2. A rotation around the z-axis at that point
    """

    def __init__(self, position: Tuple[float, float, float], rotation: float):
        """Initialize local coordinate system.

        Args:
            position: (x, y, z) position of local origin in world coordinates
            rotation: Rotation angle around z-axis in radians
        """
        self.position = np.array(position, dtype=float)
        self.rotation = float(rotation)
        self.transform_matrix = self._compute_transform_matrix()
        self.inverse_matrix = self._compute_inverse_matrix()

    def _compute_transform_matrix(self) -> NDArray[np.float64]:
        """
        Compute the 4x4 homogeneous transformation matrix that converts world coordinates
        to local coordinates.

        For a point p in world coordinates, this:
        1. Translates p relative to the local origin
        2. Rotates around this local origin

        Returns:
            4x4 numpy array representing the transformation to local coordinates
        """
        cos_rot = np.cos(self.rotation)
        sin_rot = np.sin(self.rotation)

        # Matrix that translates to local origin and rotates around it
        transform = np.array([
            [cos_rot, -sin_rot, 0, -(cos_rot * self.position[0] - sin_rot * self.position[1])],
            [sin_rot, cos_rot, 0, -(sin_rot * self.position[0] + cos_rot * self.position[1])],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0, 1]
        ])
        return transform

    def _compute_inverse_matrix(self) -> NDArray[np.float64]:
        """
        Compute the 4x4 homogeneous transformation matrix that converts local coordinates
        back to world coordinates.

        For a point p in local coordinates, this:
        1. Rotates back (negative angle around local origin)
        2. Translates back to world coordinates

        Returns:
            4x4 numpy array representing the transformation to world coordinates
        """
        cos_rot = np.cos(-self.rotation)  # Negative rotation for inverse
        sin_rot = np.sin(-self.rotation)

        # Matrix that rotates back and translates to world coordinates
        inverse = np.array([
            [cos_rot, -sin_rot, 0, self.position[0]],
            [sin_rot, cos_rot, 0, self.position[1]],
            [0, 0, 1, self.position[2]],
            [0, 0, 0, 1]
        ])
        return inverse

    def __call__(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points from world to local coordinates.

        Args:
            points: Array of shape (..., 3) containing world-space coordinates

        Returns:
            Array of shape (..., 3) containing local-space coordinates
        """
        points = np.asarray(points)
        orig_shape = points.shape
        points = np.atleast_2d(points)

        # Convert points to homogeneous coordinates by adding a 1 as the fourth component
        # This allows us to apply both rotation and translation in a single matrix multiply
        points_h = np.pad(
            points,
            ((0, 0),) * (points.ndim - 1) + ((0, 1),),
            constant_values=1
        )

        # Transform points to local space using einsum for efficient batch matrix multiplication
        # The resulting points are in the local coordinate system.
        local_points = np.einsum('...ij,...j->...i', self.transform_matrix, points_h)[..., :3]

        # Restore original shape
        return local_points.reshape(orig_shape)

    def inverse(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points from local coordinates back to world coordinates.

        Args:
            points: Array of shape (..., 3) containing local-space coordinates

        Returns:
            Array of shape (..., 3) containing world-space coordinates
        """
        points = np.asarray(points)
        orig_shape = points.shape
        points = np.atleast_2d(points)

        # Convert points to homogeneous coordinates
        points_h = np.pad(
            points,
            ((0, 0),) * (points.ndim - 1) + ((0, 1),),
            constant_values=1
        )

        # Transform points back to world space
        world_points = np.einsum('...ij,...j->...i', self.inverse_matrix, points_h)[..., :3]

        # Restore original shape
        return world_points.reshape(orig_shape)


class ZRotation:
    """Handles coordinate transformations with rotation around z-axis only."""

    def __init__(self, angle: float):
        """Initialize transform with rotation angle.

        Args:
            angle: Rotation angle around z-axis in radians
        """
        self.angle = float(angle)
        self.cos = np.cos(angle)  # Cache these values since they're used repeatedly
        self.sin = np.sin(angle)

    def __call__(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points from world to rotated coordinates.

        Args:
            points: Array of shape (..., 3) containing world-space coordinates

        Returns:
            Array of shape (..., 3) containing rotated coordinates
        """
        points = np.asarray(points)
        x, y = points[..., 0], points[..., 1]

        # Rotate x and y coordinates, z remains unchanged
        x_rot = x * self.cos - y * self.sin
        y_rot = x * self.sin + y * self.cos

        # Efficiently create output array with same shape as input
        result = np.empty_like(points)
        result[..., 0] = x_rot
        result[..., 1] = y_rot
        result[..., 2] = points[..., 2]

        return result

    def inverse(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points from rotated back to world coordinates.

        Args:
            points: Array of shape (..., 3) containing rotated coordinates

        Returns:
            Array of shape (..., 3) containing world-space coordinates
        """
        points = np.asarray(points)
        x, y = points[..., 0], points[..., 1]

        # Apply inverse rotation (negative angle)
        x_world = x * self.cos + y * self.sin  # Note: cos(-θ) = cos(θ), sin(-θ) = -sin(θ)
        y_world = -x * self.sin + y * self.cos

        result = np.empty_like(points)
        result[..., 0] = x_world
        result[..., 1] = y_world
        result[..., 2] = points[..., 2]

        return result


class Translation:
    """Handles pure translation of coordinates."""

    def __init__(self, offset: Tuple[float, float, float]):
        """Initialize transform with translation vector.

        Args:
            offset: (x, y, z) translation vector
        """
        self.offset = np.array(offset, dtype=float)

    def __call__(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points from world to local coordinates.

        Args:
            points: Array of shape (..., 3) containing world-space coordinates

        Returns:
            Array of shape (..., 3) containing translated coordinates
        """
        points = np.asarray(points)
        # Subtract track_center to move to local coordinates
        return points - self.offset

    def inverse(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points from local back to world coordinates.

        Args:
            points: Array of shape (..., 3) containing translated coordinates

        Returns:
            Array of shape (..., 3) containing world-space coordinates
        """
        points = np.asarray(points)
        # Add track_center to move back to world coordinates
        return points + self.offset


if __name__ == "__main__":
    import numpy as np

    # Test data: Create some 3D points to transform
    test_points = np.array([
        [1.0, 0.0, 0.0],  # Point on x-axis
        [0.0, 1.0, 0.0],  # Point on y-axis
        [1.0, 1.0, 1.0],  # Point in octant 1
        [-1.0, -1.0, 2.0]  # Point with negative coordinates
    ])

    print("Testing coordinate transformations...")
    print("\nOriginal points:")
    print(test_points)

    # Test 1: Pure rotation
    print("\n=== Testing ZRotation ===")
    angle = np.pi / 4  # 45 degrees
    rotation = ZRotation(angle)

    # Transform points
    rotated_points = rotation(test_points)
    print(f"\nPoints after {angle:.2f} radians rotation:")
    print(rotated_points)

    # Verify round-trip
    restored_points = rotation.inverse(rotated_points)
    print("\nPoints after round-trip (should match original):")
    print(restored_points)
    print(f"Maximum round-trip error: {np.max(np.abs(restored_points - test_points)):.2e}")

    # Test 2: Pure translation
    print("\n=== Testing Translation ===")
    offset = (1.0, 2.0, 3.0)
    translation = Translation(offset)

    # Transform points
    translated_points = translation(test_points)
    print(f"\nPoints after translation by {offset}:")
    print(translated_points)

    # Verify round-trip
    restored_points = translation.inverse(translated_points)
    print("\nPoints after round-trip (should match original):")
    print(restored_points)
    print(f"Maximum round-trip error: {np.max(np.abs(restored_points - test_points)):.2e}")

    # Test 3: Combined rotation and translation
    print("\n=== Testing ToLocalCoordinateSystem ===")
    position = (2.0, 1.0, 0.5)
    rotation = np.pi / 6  # 30 degrees
    local_coords = ToLocalCoordinateSystem(position, rotation)

    # Transform points
    local_points = local_coords(test_points)
    print(f"\nPoints in local coordinate system (pos={position}, rot={rotation:.2f}):")
    print(local_points)

    # Verify round-trip
    restored_points = local_coords.inverse(local_points)
    print("\nPoints after round-trip (should match original):")
    print(restored_points)
    print(f"Maximum round-trip error: {np.max(np.abs(restored_points - test_points)):.2e}")

    # Test 4: Batch processing with different shapes
    print("\n=== Testing batch processing ===")
    # Create a 2x2x3 array of points (2 rows, 2 cols of 3D points)
    batch_points = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 2.0]]
    ])
    print("\nOriginal batch shape:", batch_points.shape)

    # Transform batch
    local_batch = local_coords(batch_points)
    print("Transformed batch shape:", local_batch.shape)
    restored_batch = local_coords.inverse(local_batch)
    print("Maximum batch round-trip error:", np.max(np.abs(restored_batch - batch_points)))

    # Verify shape preservation
    assert batch_points.shape == local_batch.shape == restored_batch.shape, \
        "Shape was not preserved during transformations!"

    print("\nAll tests completed successfully!")