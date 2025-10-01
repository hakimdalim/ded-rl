import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Protocol, Callable, Union, Tuple

from scipy.optimize import minimize_scalar

from geometry.activation_boundary import ActivationBoundary
from utils.coordinate_transform import ToLocalCoordinateSystem, ZRotation, Translation
from geometry.surface_boundary import TopSurfaceBoundary


class TrackProfileProtocol(Protocol):
    """
    Protocol defining interface for track cross-section profiles.

    Implementations must provide x_min and x_max bounds that define the valid domain
    for the profile function. The width of the track at this cross-section is implicitly
    defined as (x_max - x_min).
    """
    x_min: float
    x_max: float

    def get_z(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate height values for given x coordinates.

        Args:
            x: Array of x coordinates where x_min <= x <= x_max

        Returns:
            Array of corresponding z (height) values

        Raises:
            ValueError: If any x values lie outside [x_min, x_max]
        """
        ...


class TrackProfile:
    """
    Basic implementation of TrackProfileProtocol that uses a callable to define the z-height.
    """

    def __init__(self, height_func: Callable[[np.ndarray], np.ndarray],
                 x_min: float, x_max: float):
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")

        self.height_func = height_func
        self.x_min = x_min
        self.x_max = x_max

    def get_z(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate height values, returning NaN for out-of-bounds coordinates.

        Args:
            x: Array of x coordinates

        Returns:
            Array of z (height) values with NaN for invalid x coordinates
        """
        x = np.asarray(x)
        z = np.full_like(x, np.nan, dtype=float)
        valid_mask = (x >= self.x_min) & (x <= self.x_max)
        z[valid_mask] = self.height_func(x[valid_mask])
        return z

class TrackSection(ActivationBoundary):
    """
    Track section defined by two profiles in local coordinates with separated transformations.

    The transformation is split into two steps:
    1. A rotation around the Z axis
    2. A translation in world coordinates

    The local coordinate system origin is at 'position' in global coordinates after rotation.
    The start profile is centered at the local origin, while the end profile is at y=length_between.

    Note: This implementation assumes that both start and end profiles are concave functions
    within their defined domains [x_min, x_max]. This assumption is necessary for efficient
    maximum height calculation using gradient-based optimization methods.
    """

    def __init__(
            self,
            start_profile: TrackProfile,
            end_profile: TrackProfile,
            length: float,
            position: Tuple[float, float, float],
            rotation: float
    ):
        """Initialize track section with separated transformations.

        Args:
            start_profile: ProfileGenerator at start of track section (must be concave)
            end_profile: ProfileGenerator at end of track section (must be concave)
            length: Length of track section in local y direction
            position: (x, y, z) world position of start profile
            rotation: Rotation angle around z-axis in radians

        Raises:
            ValueError: If length_between is not positive or profiles have incompatible bounds
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        self.start_profile = start_profile
        self.end_profile = end_profile
        self.length = length
        self.y_axis_max = length

        # Create separated transforms
        self.rotation = ZRotation(rotation)
        self.translation = Translation(position)
        # Create combined transform
        self.transform = ToLocalCoordinateSystem(position, rotation)

        # Cache profile bounds
        self.local_x_min = min(start_profile.x_min, end_profile.x_min)
        self.local_x_max = max(start_profile.x_max, end_profile.x_max)
        self.x_axis_max = self.local_x_max - self.local_x_min

        # Calculate and cache maximum heights
        self.start_max_height = self._find_profile_maximum(start_profile)
        self.end_max_height = self._find_profile_maximum(end_profile)
        self.z_axis_max = max(self.start_max_height, self.end_max_height)

        self.max_axis = max(self.x_axis_max, self.y_axis_max)

        # Create surface boundary
        self.boundary = TopSurfaceBoundary(self._interpolate_surface)

    def _find_profile_maximum(self, profile: TrackProfile) -> float:
        """Find maximum height of a profile using efficient optimization.

        Uses Brent's method for bounded optimization, leveraging the concavity
        assumption to find the global maximum efficiently.

        Args:
            profile: Track profile to analyze (must be concave)

        Returns:
            Maximum height value within profile's valid domain
        """

        # Define the negative height function for minimization
        def neg_height(x):
            return -profile.get_z(np.array([x]))[0]

        # Use Brent's method for bounded optimization
        result = minimize_scalar( # TODO: method benchmarking to determine computationally fastest
            neg_height,
            bounds=(profile.x_min, profile.x_max),
            method='bounded'
        )

        if not result.success:
            raise RuntimeError("Failed to find profile maximum")

        return -result.fun  # Return positive maximum height

    def _interpolate_surface(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate surface z-coordinates between start and end profiles.

        Args:
            points: Array of shape (..., 3) containing coordinates in local space

        Returns:
            Array of shape (...,) containing interpolated z-coordinates
        """
        # Calculate how far along the track each point is (0 at start, 1 at end)
        # This serves as our interpolation parameter between the two profiles
        y_param = points[..., 1] / self.length

        # Ensure that the interpolation parameter y_param is only valid within the range [0, 1].
        # If y_param is outside this range, it is set to NaN. This is done to handle points
        # that lie outside the track section's length_between, ensuring that only valid interpolation
        # parameters are used for further calculations.
        y_param = np.where((y_param >= 0) & (y_param <= 1), y_param, np.nan)

        # Get heights from both profiles
        x_coords = points[..., 0]
        x_valid = (x_coords >= self.local_x_min) & (x_coords <= self.local_x_max)

        # Get heights from both profiles and replace NaNs with zeros within valid x range
        start_z = self.start_profile.height_func(x_coords)
        end_z = self.end_profile.height_func(x_coords)

        # Interpolate between profiles and mask invalid x coordinates with NaN
        interpolated_z = (1 - y_param) * start_z + y_param * end_z
        return np.where(x_valid, interpolated_z, np.nan)


    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Check if points lie below track surface.

        Args:
            points: Array of shape (..., 3) containing world coordinates

        Returns:
            Boolean array of shape (...,) where True indicates points below surface
        """
        # Transform to local coordinates
        local_points = self.transform(points)
        # Use boundary to check containment
        return self.boundary.contains_points(local_points)

    def _create_mask(
            self,
            shape: Tuple[int, int, int],
            voxel_size: Union[float, Tuple[float, float, float]]
    ) -> NDArray[np.bool_]:
        """Create boolean mask indicating which voxels lie inside the track section.

        Args:
            shape: Grid dimensions (nx, ny, nz) for the world space mask
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)

        Returns:
            Boolean array of shape (nx, ny, nz) marking voxels inside track section

        Raises:
            ValueError: If the track section extends beyond the world grid bounds
        """
        # Standardize voxel_size to handle both scalar and per-axis inputs uniformly
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        voxel_size = np.array(voxel_size, dtype=float)


        local_shape = (
            (int(np.ceil(self.max_axis / voxel_size[0])) + 4) * 2, # max_axis because of potential rotation
            (int(np.ceil(self.max_axis / voxel_size[1])) + 4) * 2, # max_axis because of potential rotation
            (int(np.ceil(self.z_axis_max / voxel_size[2])) + 4) * 2
        )

        #print('local_shape:', local_shape)

        # Start with minimal grid that exactly covers our track section
        local_points = self.create_coordinate_grid(local_shape, voxel_size)

        #print('local_points.min():', local_points[...,0].min(), local_points[...,1].min(), local_points[...,2].min())
        #print('local_points.max():', local_points[...,0].max(), local_points[...,1].max(), local_points[...,2].max())

        # Prepare offsets for world positioning
        local_offset = np.array([
            -np.ceil(local_shape[0]/2) * voxel_size[0],
            -np.ceil(local_shape[1]/2) * voxel_size[1],
            -np.ceil(local_shape[2]/2) * voxel_size[2],
        ])
        #print(track_center)
        translation_with_offset = self.translation.offset + local_offset

        # Ensure voxel boundaries align between local and world grids for clean mask combination
        alignment_offset = -(translation_with_offset % voxel_size)
        #print('alignment_offset:', alignment_offset)
        aligned_points = local_points + alignment_offset + local_offset

        #print('aligned_points.min():', aligned_points[...,0].min(), aligned_points[...,1].min(), aligned_points[...,2].min())
        #print('aligned_points.max():', aligned_points[...,0].max(), aligned_points[...,1].max(), aligned_points[...,2].max())

        # Rotate before translation to maintain proper surface boundary calculation
        rotated_points = self.rotation(aligned_points)

        # Compute containment in local space where the geometry is simpler
        local_mask = self.boundary.contains_points(rotated_points)

        # Calculate precise indices for placing local mask in world grid
        start_indices = np.floor(translation_with_offset / voxel_size).astype(int)
        end_indices = start_indices + local_shape

        '''# Prevent silent errors from partial track sections
        if (start_indices < 0).any() or (end_indices > shape).any():
            raise ValueError("Track section extends beyond world grid bounds")'''

        # Store original indices for mask slicing calculation
        start_indices_original = start_indices.copy()
        end_indices_original = end_indices.copy()

        # Clip track section to fit within world grid bounds
        start_indices = np.maximum(start_indices, 0).astype(int)
        end_indices = np.minimum(end_indices, shape).astype(int)

        # Adjust local_mask to match clipped size
        slice_starts = np.maximum(0, -start_indices_original).astype(int)
        slice_ends = (np.array(local_shape) - np.maximum(0, end_indices_original - shape)).astype(int)

        local_mask = local_mask.reshape(tuple(int(x) for x in local_shape))[
                     slice_starts[0]:slice_ends[0],
                     slice_starts[1]:slice_ends[1],
                     slice_starts[2]:slice_ends[2]
                     ]

        # Update local_shape to match clipped size
        local_shape = tuple(int(x) for x in (end_indices - start_indices))

        return start_indices, end_indices, local_mask, local_shape

    def create_mask(
            self,
            shape: Tuple[int, int, int],
            voxel_size: Union[float, Tuple[float, float, float]]
    ) -> NDArray[np.bool_]:
        """Create boolean mask indicating which voxels lie inside the track section.

        Args:
            shape: Grid dimensions (nx, ny, nz) for the world space mask
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)

        Returns:
            Boolean array of shape (nx, ny, nz) marking voxels inside track section

        Raises:
            ValueError: If the track section extends beyond the world grid bounds
        """
        start_indices, end_indices, local_mask, local_shape = self._create_mask(shape, voxel_size)

        # Use efficient slice assignment to place local mask in world space
        world_mask = np.zeros(shape, dtype=bool)
        world_mask[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ] = local_mask.reshape(local_shape)

        return world_mask

    def activate_inplace(self, array: NDArray[np.bool_], voxel_size: Union[float, Tuple[float, float, float]]) -> None:
        """Set array elements inside this boundary instance to True.

        Args:
            array: Boolean array to modify, shape (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)
        """
        start_indices, end_indices, local_mask, local_shape = self._create_mask(array.shape, voxel_size)
        array[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ] |= local_mask.reshape(local_shape)

    def deactivate_inplace(self, array: NDArray[np.bool_],
                           voxel_size: Union[float, Tuple[float, float, float]]) -> None:
        """Set array elements inside this boundary instance to False.

        Args:
            array: Boolean array to modify, shape (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)
        """
        start_indices, end_indices, local_mask, local_shape = self._create_mask(array.shape, voxel_size)
        array[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ] &= ~local_mask.reshape(local_shape)


class TrackSectionDeprecatedVersion:
    """
    Track section defined by two profiles in local coordinates.

    The local coordinate system origin is at 'position' in global coordinates,
    rotated by 'rotation' around the Z axis. The start profile is centered
    at the local origin, while the end profile is centered at y=length_between.
    """

    def __init__(
            self,
            start_profile: TrackProfile,
            end_profile: TrackProfile,
            length: float,
            position: Tuple[float, float, float],
            rotation: float
    ):
        """Initialize track section.

        Args:
            start_profile: ProfileGenerator at start of track section
            end_profile: ProfileGenerator at end of track section
            length: Length of track section in local y direction
            position: (x, y, z) world position of start profile
            rotation: Rotation angle around z-axis in radians

        Raises:
            ValueError: If length_between is not positive
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        self.start_profile = start_profile
        self.end_profile = end_profile
        self.length = length

        # Create coordinate transform
        self.transform = ToLocalCoordinateSystem(position, rotation)

        # Create surface boundary
        self.boundary = TopSurfaceBoundary(self._interpolate_surface)

    def _interpolate_surface(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate surface z-coordinates between start and end profiles.

        Args:
            points: Array of shape (..., 3) containing coordinates in local space

        Returns:
            Array of shape (...,) containing interpolated z-coordinates
        """
        # Calculate how far along the track each point is (0 at start, 1 at end)
        # This serves as our interpolation parameter between the two profiles
        y_param = points[..., 1] / self.length

        # Ensure that the interpolation parameter y_param is only valid within the range [0, 1].
        # If y_param is outside this range, it is set to NaN. This is done to handle points
        # that lie outside the track section's length_between, ensuring that only valid interpolation
        # parameters are used for further calculations.
        y_param = np.where((y_param >= 0) & (y_param <= 1), y_param, np.nan)

        # Get heights from both profiles
        x_coords = points[..., 0]
        start_z = self.start_profile.get_z(x_coords)
        end_z = self.end_profile.get_z(x_coords)

        # Interpolate between profiles
        return (1 - y_param) * start_z + y_param * end_z

    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Check if points lie below track surface.

        Args:
            points: Array of shape (..., 3) containing world coordinates

        Returns:
            Boolean array of shape (...,) where True indicates points below surface
        """
        # Transform to local coordinates
        local_points = self.transform(points)
        # Use boundary to check containment
        return self.boundary.contains_points(local_points)


if __name__ == "__main__":

    def run_test_case(name: str, test_func: Callable) -> bool:
        """Run a single test case with reporting"""
        try:
            test_func()
            print(f"✓ {name}")
            return True
        except AssertionError as e:
            print(f"✗ {name}: {str(e)}")
            return False


    def test_flat_horizontal():
        """Test with flat profiles at z=0, no rotation"""
        flat_profile = TrackProfile(lambda x: np.zeros_like(x), x_min=-1, x_max=1)
        track = TrackSection(
            start_profile=flat_profile,
            end_profile=flat_profile,
            length=2.0,
            position=(0, 0, 0),
            rotation=0
        )

        # Test points exactly on surface
        surface_points = np.array([
            [0, 0, 0],  # Center start
            [0, 1, 0],  # Center middle
            [0, 2, 0],  # Center end
            [-1, 1, 0],  # Left edge
            [1, 1, 0],  # Right edge
        ])
        assert np.all(track.contains_points(surface_points)), "Points on surface should be inside"

        # Test points slightly above/below surface
        epsilon = 1e-10
        above_points = surface_points + np.array([0, 0, epsilon])
        below_points = surface_points - np.array([0, 0, epsilon])
        assert not np.any(track.contains_points(above_points)), "Points slightly above should be outside"
        assert np.all(track.contains_points(below_points)), "Points slightly below should be inside"


    def test_angled_surface():
        """Test with sloped profile and rotation"""
        start_profile = TrackProfile(lambda x: x, x_min=-1, x_max=1)
        end_profile = TrackProfile(lambda x: 2 * x, x_min=-1, x_max=1)

        track = TrackSection(
            start_profile=start_profile,
            end_profile=end_profile,
            length=2.0,
            position=(0, 0, 0),
            rotation=np.pi / 4
        )

        # Local coordinates: (-0.5, 1.0, z) where z varies around -0.75
        rot = np.pi / 4
        x_local = -0.5
        y_local = 1.0
        x_world = x_local * np.cos(rot) + y_local * np.sin(rot)  # counter-clockwise
        y_world = -x_local * np.sin(rot) + y_local * np.cos(rot)  # counter-clockwise

        test_points = np.array([
            [x_world, y_world, -0.85],  # Below surface
            [x_world, y_world, -0.75],  # On surface
            [x_world, y_world, -0.65]  # Above surface
        ])

        results = track.contains_points(test_points)
        assert results[0], "Point below interpolated surface should be inside"
        assert results[1], "Point on interpolated surface should be inside"
        assert not results[2], "Point above interpolated surface should be outside"


    def test_boundary_edges():
        """Test behavior at and beyond profile boundary"""
        profile = TrackProfile(lambda x: np.zeros_like(x), x_min=-1, x_max=1)
        track = TrackSection(
            start_profile=profile,
            end_profile=profile,
            length=2.0,
            position=(0, 0, 0),
            rotation=0
        )

        edge_points = np.array([
            [-1, 1, 0],  # Left boundary
            [1, 1, 0],  # Right boundary
            [-1.1, 1, 0],  # Just beyond left boundary
            [1.1, 1, 0],  # Just beyond right boundary
            [0, -0.1, 0],  # Before start
            [0, 2.1, 0],  # After end
        ])

        results = track.contains_points(edge_points)
        assert results[0] and results[1], "Boundary points should be inside"
        assert not results[2] and not results[3], "Points beyond x bounds should be outside"
        assert not results[4] and not results[5], "Points beyond y bounds should be outside"


    def test_transformed_coordinates():
        """Test with non-zero position and rotation"""
        profile = TrackProfile(lambda x: np.zeros_like(x), x_min=-1, x_max=1)
        track = TrackSection(
            start_profile=profile,
            end_profile=profile,
            length=2.0,
            position=(1, 1, 1),
            rotation=np.pi / 2
        )

        # First define points in local coordinates
        x_local = 0.0  # middle of profile width
        y_local = 1.0  # middle of track length_between

        # Transform to world coordinates
        rot = np.pi / 2
        x_world = x_local * np.cos(rot) + y_local * np.sin(rot)  # counter-clockwise
        y_world = -x_local * np.sin(rot) + y_local * np.cos(rot)  # counter-clockwise

        test_points = np.array([
            [x_world, y_world, 1.0],  # On surface (z=0 + track_center of 1)
            [x_world, y_world, 0.5],  # Below surface
            [x_world, y_world, 1.5]  # Above surface
        ])

        results = track.contains_points(test_points)
        assert results[0], "Point on transformed surface should be inside"
        assert results[1], "Point below transformed surface should be inside"
        assert not results[2], "Point above transformed surface should be outside"


    def test_numerical_stability():
        """Test behavior with very small numbers and potential floating point issues"""
        profile = TrackProfile(lambda x: np.zeros_like(x), x_min=-1, x_max=1)
        track = TrackSection(
            start_profile=profile,
            end_profile=profile,
            length=2.0,
            position=(0, 0, 0),
            rotation=0
        )

        # Test with very small offsets
        epsilon = 1e-15
        test_points = np.array([
            [0, 1, epsilon],  # Slightly above
            [0, 1, -epsilon],  # Slightly below
            [epsilon, 1, 0],  # Slight x track_center
            [0, 1 + epsilon, 0],  # Slight y track_center
        ])

        results = track.contains_points(test_points)
        assert not results[0], "Point slightly above should be outside"
        assert results[1], "Point slightly below should be inside"
        assert results[2], "Point with slight x track_center should be inside"
        assert results[3], "Point with slight y track_center should be inside"


    # Additional test for array broadcasting
    def test_array_broadcasting():
        """Test that the boundary check works with different array shapes"""
        profile = TrackProfile(lambda x: np.zeros_like(x), x_min=-1, x_max=1)
        track = TrackSection(
            start_profile=profile,
            end_profile=profile,
            length=2.0,
            position=(0, 0, 0),
            rotation=0
        )

        # Test with different array shapes
        single_point = np.array([0, 1, 0])
        point_list = np.array([[0, 1, 0], [0, 1, 1]])
        point_grid = np.zeros((2, 3, 3))  # 2x3 grid of 3D points

        # These should all work without raising errors
        result1 = track.contains_points(single_point)
        result2 = track.contains_points(point_list)
        result3 = track.contains_points(point_grid)

        # print(result1)
        # print(result2)
        # print(result3)

        assert isinstance(result1, np.ndarray), "Should return numpy array for single point"
        assert result2.shape == (2,), "Should return 1D array for point list"
        assert result3.shape == (2, 3), "Should preserve leading dimensions"


    # Run all tests
    tests = [
        ("Flat Horizontal Surface", test_flat_horizontal),
        ("Angled Surface", test_angled_surface),
        ("Boundary Edges", test_boundary_edges),
        ("Transformed Coordinates", test_transformed_coordinates),
        ("Numerical Stability", test_numerical_stability),
        ("Array Broadcasting", test_array_broadcasting)
    ]

    passed = sum(1 for name, func in tests if run_test_case(name, func))
    total = len(tests)

    print(f"\nPassed {passed}/{total} tests")
    exit(0 if passed == total else 1)