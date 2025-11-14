from typing import Dict, Tuple, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import open3d as o3d
from scipy import integrate
from geometry.clad_interpolator import interpolate_track
from geometry.clad_profile_function import GenerateParabolicCladProfile as ProfileGenerator
from geometry.clad_profile_function import ParabolicCladProfile
import numpy as np


from scipy.spatial import Delaunay

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError as e:
    O3D_AVAILABLE = False
    o3d = None  # Set to None to avoid NameError
    Warning("Open3D not available. Mesh generation and visualization not possible. Error: ", e)


class ProfileNotFoundError(Exception):
    """Exception raised when a requested profile cannot be found."""
    pass

class ProfileInterpolationError(Exception):
    """Exception raised when profile interpolation cannot be performed."""
    pass


class SubstrateFunction:

    def __init__(self, substrate_height: float):
        self.substrate_height = substrate_height

    def __call__(self, *args, **kwargs):
        return self.substrate_height

    def vectorized(self, x):
        """
        Vectorized evaluation of the substrate function at given x-values.

        This function accepts either:
          - a single float (scalar), or
          - a NumPy array of x-values.
        It returns a single float or an array, respectively.

        Args:
            x: Float (scalar) or np.ndarray (last dimension = x-coordinates)

        Returns:
            Float if x is scalar, or np.ndarray if x is array.
        """
        # 1) Convert scalar input to 1D array, if necessary
        is_scalar = isinstance(x, (int, float))
        if is_scalar:
            x_array = np.array([x], dtype=float)
        else:
            x_array = np.asarray(x, dtype=float)

        # 2) Vectorized evaluation:
        #    The substrate function is constant, so we return the same shape as x_array.
        return np.full_like(x_array, self.substrate_height)


class CladProfileManager:
    """
    Manages clad profile functions for a multi-track, multi-layer build.

    Assumptions and Rules:
    1. Laser movement:
       - Primary movement is back and forth in y-direction
       - Cross-section view is in x-z plane
       - Tracks progress in positive x-direction within each layer

    2. Track and Layer Organization:
       - Multiple tracks form a layer in x-z plane
       - Layers are stacked in z-direction
       - Track indices increase with x-position
       - Layer indices increase with z-position

    3. ProfileGenerator Functions:
       - First track in each layer uses initialization function
       - Subsequent tracks use interpolation between relevant profiles
       - Profiles are stored by their (layer, track, y-position)
    """

    def __init__(self, hatch_distance: float, num_tracks: int, offset: float = 0.0, substrate_height: float = 0.0001):
        """
        Initialize the profile manager.

        Args:
            hatch_distance: Distance between track centers in meters
            num_tracks: Total number of tracks per layer
            offset: Global X-track_center for all tracks in meters (default 0.0)
        """
        self.substrate_height = substrate_height
        self.hatch_distance = hatch_distance
        self.offset = offset
        self.num_tracks = num_tracks
        self._profiles = {}

        self.substrate_function = SubstrateFunction(substrate_height)

    '''def _validate_sequence(self, layer_index: int, track_index: int) -> None:
            """
            Validates that layer and track indices follow the correct building sequence.

            Args:
                layer_index: Layer number in build
                track_index: Track number in current layer

            Raises:
                ValueError: If layer or track index is out of sequence
            """
            if layer_index > self._max_layer + 1:
                raise ValueError(
                    f"Layer index {layer_index} is out of sequence. "
                    f"Current max layer is {self._max_layer}"
                )

            max_track = self._max_tracks.get(layer_index, -1)
            if track_index > max_track + 1:
                raise ValueError(
                    f"Track index {track_index} is out of sequence in layer {layer_index}. "
                    f"Current max track is {max_track}"
                )'''

    def reset(self):
        self._profiles = {}

    def get_x_position(self, track_index: int) -> float:
        """
        Calculate x-position from track index.

        Assumes:
        - Constant hatch distance between track centers
        - Global track_center applied to all tracks
        - All dimensions in meters

        Args:
            track_index: Index of track in layer (starting from 0)

        Returns:
            X-position in meters
        """
        return self.offset + track_index * self.hatch_distance

    def get_z_position(self, layer_index: int, track_index: int, y_pos: float) -> float:
        """
        Calculate z-position for a given position in the build.

        Implementation rationale:
        - Returns baseline height which represents the base z-position for the current layer
        - Will be removed in future versions as this functionality is now handled by track baselines

        Args:
            layer_index: Current layer (starting from 0)
            track_index: Current track in layer (starting from 0)
            y_pos: Position along track length_between in meters

        Returns:
            Z-position in meters (baseline height)
        """
        if layer_index == 0:
            return self.substrate_height
        else:
            track_profile_beneath = self.get_layer_cross_section(layer_index-1, y_pos) #self.get_profile_function(layer_index - 1, track_index, y_pos)
            return track_profile_beneath(self.get_x_position(track_index))

    def get_profile_function(
            self,
            layer_index: int,
            track_index: int,
            y_pos: float,
            threshold: float = 1e-10
    ) -> ParabolicCladProfile:
        """
        Get interpolated profile function at specified position.

        Finds the two closest saved profile functions along the track length_between and
        interpolates between them using the clad_interpolator.

        Implementation steps:
        1. Find all saved y-positions for this layer and track
        2. Get the two closest positions (one before, one after target y_pos)
        3. If exact match exists, return that profile
        4. If only one profile exists or target is outside range, raise ProfileInterpolationError
        5. Otherwise, interpolate between the two closest profiles

        Args:
            layer_index: Layer number in build
            track_index: Track number in current layer
            y_pos: Target y-position along track length_between

        Returns:
            Interpolated profile function

        Raises:
            ProfileNotFoundError: If no profiles exist for the specified layer and track
            ProfileInterpolationError: If interpolation cannot be performed due to insufficient data
        """
        # Get all saved y-positions for this layer and track
        y_positions = sorted([
            y for (layer, track, y) in self._profiles.keys()
            if layer == layer_index and track == track_index
        ])

        # Handle case where no profiles exist
        if not y_positions:
            raise ProfileNotFoundError(
                f"No profiles found for layer={layer_index}, track={track_index}"
            )

        # Check for match within threshold
        for saved_y in y_positions:
            if abs(y_pos - saved_y) <= threshold:
                return self._profiles[(layer_index, track_index, saved_y)]

        # Find closest positions before and after
        y_before = max((y for y in y_positions if y < y_pos), default=None)
        y_after = min((y for y in y_positions if y > y_pos), default=None)

        # Handle edge cases
        if y_before is None or y_after is None:
            #print(self._profiles.keys())
            #print(y_positions)
            raise ProfileInterpolationError(
                f"Cannot interpolate profile at y={y_pos} for layer={layer_index}, "
                f"track={track_index}. "
                f"\ny_before: {y_before}, y_after: {y_after}"
                f"\nAvailable y-positions: {y_positions}"
            )

        # Get the two profiles for interpolation
        profile_before = self._profiles[(layer_index, track_index, y_before)]
        profile_after = self._profiles[(layer_index, track_index, y_after)]

        # Calculate interpolation distances
        distance1 = y_pos - y_before  # Distance from position before
        distance2 = y_pos - y_after  # Distance from position after (will be negative)

        # Create and return interpolated profile
        return interpolate_track(
            profile_before,
            profile_after,
            distance1=distance1,
            distance2=distance2
        )

    @staticmethod
    def _make_cross_section_function(profiles):
        def cross_section(x):
            """
            Evaluate the cross-section for a layer at given x-values.

            This function accepts either:
              - a single float (scalar), or
              - a NumPy array of x-values.
            It returns a single float or an array, respectively.

            Args:
                x: Float (scalar) or np.ndarray (last dimension = x-coordinates)

            Returns:
                Float if x is scalar, or np.ndarray if x is array.
            """

            # 1) Convert scalar input to 1D array, if necessary
            is_scalar = isinstance(x, (int, float))
            if is_scalar:
                x_array = np.array([x], dtype=float)
            else:
                x_array = np.asarray(x, dtype=float)

            # 2) Vectorized evaluation:
            #    Each profile has a .vectorized(...) method that returns the same shape as x_array.
            #    We stack the results of all track profiles along axis=0, then take max across profiles.
            z_stacked = np.stack([p.vectorized(x_array) for p in profiles], axis=0)
            z_max = np.max(z_stacked, axis=0)  # shape = same as x_array

            # 3) If the original input was scalar, return a float; otherwise, return the array
            if is_scalar:
                return z_max[0]  # single float
            else:
                return z_max  # np.ndarray

        return cross_section

    def get_layer_cross_section(
            self,
            layer_index: int,
            y_pos: float
    ) -> Callable[[float], float]:
        """
        Creates a function representing the complete cross-section of a layer at given y-position.

        Implementation rationale:
        1. Each track function naturally handles its own bounds by returning 0 outside
        2. Track functions can be safely added since they don't overlap (to be verified)
        3. Local coordinate system for each layer starts at z=0

        Physical assumptions:
        - Track profile functions have non-overlapping bounds
        - Each track handles its own boundary conditions
        - ProfileGenerator shapes maintain validity through interpolation

        Args:
            layer_index: Layer to evaluate
            y_pos: Position along track length_between in meters

        Returns:
            Function that takes x-position in meters and returns height in meters

        Raises:
            ValueError: If requested layer does not exist
        """
        # Get all track indices for this layer
        track_indices = {
            track for (layer, track, _) in self._profiles.keys()
            if layer == layer_index
        }

        if not track_indices:
            raise ValueError(f"Layer {layer_index} does not exist")

        # Get all valid profile functions for this layer at y_pos
        track_profiles = [self.substrate_function]  # Start with substrate height
        for track_idx in sorted(track_indices):  # Sort for consistent ordering
            profile = self.get_profile_function(layer_index, track_idx, y_pos)
            if profile is not None:
                track_profiles.append(profile)

        return self._make_cross_section_function(track_profiles)

    def get_cross_section(
            self,
            y_pos: float
    ) -> Callable[[float], float]:

        _indices = {(layer, track) for (layer, track, _) in self._profiles.keys()}

        track_profiles = [self.substrate_function]  # Start with substrate height
        for layer_index, track_idx, in _indices:
            try:
                track_profiles.append(self.get_profile_function(layer_index, track_idx, y_pos))
            except ProfileInterpolationError:
                pass

        return self._make_cross_section_function(track_profiles)

    def add_profile(
            self,
            layer_index: int,
            track_index: int,
            y_pos: float,
            width: float,
            height: float,
    ) -> ParabolicCladProfile:
        """
        Add a new profile function at the specified position.

        Args:
            layer_index: Layer number in build
            track_index: Track number in current layer
            y_pos: Position along track length_between in meters
            width: Width of track in meters
            height: Height of track in meters
            material_balance_effect: Whether to apply material balance calculations

        Returns:
            Created profile function

        Raises:
            ValueError: If position is invalid or required profiles are missing
        """
        # Validate layer and track sequence
        #self._validate_sequence(layer_index, track_index)

        track_center = self.get_x_position(track_index)
        cross_section = self.get_cross_section(y_pos)

        z_at_track_center = cross_section(track_center)

        profile = ProfileGenerator.generate_profile_function(
            width=width,
            height=height,
            track_center=track_center,
            cross_section=cross_section,
            metadata={
                'layer': layer_index,
                'track': track_index,
                'y_pos': y_pos,
                'z_at_track_center': z_at_track_center
            },
        )

        actual_height = profile(track_center) - z_at_track_center
        height_ratio = max(actual_height, height) / min(actual_height, height)

        if height_ratio > 50:
            raise ValueError(
                f"Large discrepancy in profile height at track center:\n"
                f"Requested height: {height * 1000:.2f}mm\n"
                f"Actual height at track center (x={track_center * 1000:.2f}mm): {actual_height * 1000:.2f}mm\n"
                f"Ratio: {height_ratio:.2f}"
                f"\n\n ProfileGenerator: {profile.start_x, profile.end_x}"
            )

        # Update tracking information
        self._profiles[(layer_index, track_index, y_pos)] = profile

        return profile

    def plot_cross_section(
            self,
            layer_index: int,
            y_pos: float,
            ax=None,
            padding: float = 0.001
    ) -> None:
        """
        Plot cross-section of a layer showing individual profiles and combined shape.

        Args:
            layer_index: Layer number to plot
            y_pos: Y-position along tracks to plot
            ax: Optional matplotlib axis to plot on
            padding: Padding in meters to add to plot boundaries
        """

        # Get all track profiles for this layer/y-pos
        track_profiles = []
        for track_idx in range(self.num_tracks):
            try:
                profile = self.get_profile_function(layer_index, track_idx, y_pos)
                track_profiles.append(profile)
            except (ProfileNotFoundError, ProfileInterpolationError):
                continue

        if not track_profiles:
            raise ValueError(f"No profiles found for layer {layer_index} at y={y_pos}")

        # Calculate plot boundaries
        x_min = min(p.start_x for p in track_profiles) - padding
        x_max = max(p.end_x for p in track_profiles) + padding
        z_min = min(p.get_baseline() for p in track_profiles) - padding

        # Create x points for evaluation
        x = np.linspace(x_min, x_max, 200)

        # Get combined cross-section
        cross_section = self.get_layer_cross_section(layer_index, y_pos)
        z_cross = np.array([cross_section(x_val) for x_val in x])
        z_max = max(z_cross) + padding

        # Create or use provided axis
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_now = True
        else:
            plot_now = False

        # Get x-positions of track centers for color mapping
        track_centers = np.array([profile.x_center for profile in track_profiles])
        # Normalize positions to [0,1] range for colormap
        normalized_positions = (track_centers - track_centers.min()) / (track_centers.max() - track_centers.min())
        colors = plt.cm.viridis(normalized_positions)

        # Plot individual profiles with position-based colors
        for profile, color in zip(track_profiles, colors):
            z = np.array([profile(x_val) for x_val in x])
            ax.plot(x * 1000, z * 1000, '--', alpha=0.3, color=color)

        # Plot combined cross-section
        ax.plot(x * 1000, z_cross * 1000, 'k-', linewidth=2, label=f'Layer {layer_index}')

        # Set plot properties
        ax.set_xlim(x_min * 1000, x_max * 1000)
        ax.set_ylim(z_min * 1000, z_max * 1000)
        ax.grid(True)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Z Position (mm)')
        ax.set_title(f'Layer {layer_index} Cross Section at y = {y_pos * 1000:.2f}mm')
        ax.legend()
        ax.set_aspect('equal')

        if plot_now:
            plt.show()

    def plot_all_layers(
            self,
            y_pos: float,
            ax=None,
            padding: float = 0.001,
            max_layer_index=None,
            plot_individual_profiles: bool = True,
            plot_legend: bool = True,
    ) -> None:
        """
        Plot cross-sections of all available layers at a given y-position.

        Args:
            y_pos: Y-position along tracks to plot
            ax: Optional matplotlib axis to plot on
            padding: Padding in meters to add to plot boundaries
            max_layer_index: Optional maximum layer index to plot
        """
        # Find all available layers
        available_layers = sorted(set(
            layer for (layer, _, _) in self._profiles.keys()
            if max_layer_index is None or layer <= max_layer_index
        ))

        if not available_layers:
            raise ValueError("No layers found to plot")

        # Create or use provided axis
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_now = True
        else:
            plot_now = False

        # Track global plot boundaries and cross sections
        x_min, x_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        z_values = []

        # Plot each layer and collect z values
        for layer_idx in available_layers:
            try:
                # Get cross section for this layer
                cross_section = self.get_layer_cross_section(layer_idx, y_pos)

                # Get all profiles for this layer
                track_profiles = []
                for track_idx in range(self.num_tracks):
                    try:
                        profile = self.get_profile_function(layer_idx, track_idx, y_pos)
                        track_profiles.append(profile)
                        # Update global boundaries
                        x_min = min(x_min, profile.start_x)
                        x_max = max(x_max, profile.end_x)
                        z_min = min(z_min, profile.get_baseline())
                    except (ProfileNotFoundError, ProfileInterpolationError):
                        continue

                if not track_profiles:
                    continue

                # Create x points for evaluation (only once after we know boundaries)
                if not z_values:  # If this is the first valid layer
                    x = np.linspace(x_min - padding, x_max + padding, 200)

                # Get cross section z values
                z_cross = np.array([cross_section(x_val) for x_val in x])
                z_values.append(z_cross)
                z_max = max(z_max, max(z_cross))

                # Plot individual profiles with very low alpha
                if plot_individual_profiles:
                    for profile in track_profiles:
                        z = np.array([profile(x_val) for x_val in x])
                        ax.plot(x * 1000, z * 1000, '--', alpha=0.1, color='gray')

                # Plot layer cross-section
                ax.plot(x * 1000, z_cross * 1000, '-',
                        linewidth=1.0,
                        color='gray',
                        alpha=0.3,
                        label=f'Layer {layer_idx}')

            except Exception as e:
                print(f"Warning: Could not plot layer {layer_idx}: {e}")
                continue

        # Calculate and plot maximum height profile
        if z_values:
            z_max_combined = np.maximum.reduce(z_values)
            ax.plot(x * 1000, z_max_combined * 1000, '-',
                    linewidth=2.5,
                    color='black',
                    label='Combined Profile')

        # Set plot properties
        ax.set_xlim((x_min - padding) * 1000, (x_max + padding) * 1000)
        ax.set_ylim((z_min - padding) * 1000, (z_max + padding) * 1000)
        ax.grid(True)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Z Position (mm)')
        ax.set_title(f'All Layers Cross Section at y = {y_pos * 1000:.2f}mm')
        if plot_legend:
            ax.legend()
        ax.set_aspect('equal')

        if plot_now:
            plt.show()

    def generate_surface_info(
            self,
            x_res: int = 100,
            y_res: int = 100,
            sampling_factor: float = 0.5,
            interp_method: str = 'linear'
    ):
        """
        Generate surface data as a grid of height values for efficient surface rendering.
        This is an optimized version that skips mesh generation and returns data
        suitable for direct use with Plotly's Surface plot.

        The method first calculates the surface at a lower resolution and then
        upsamples to the requested resolution to improve performance.

        Args:
            x_res, y_res: Number of samples along x and y (final output resolution)
            sampling_factor: Factor to reduce initial calculation resolution (0.5 = half)
            interp_method: Interpolation method to use:
                - 'spline': RectBivariateSpline (default, smooth)
                - 'linear': RegularGridInterpolator with linear method
                - 'cubic': RegularGridInterpolator with cubic method
                - 'nearest': Nearest neighbor, no smoothing
                - 'smooth_spline': SmoothBivariateSpline with light smoothing
                - 'none': No interpolation (uses low-res directly)

        Returns:
            dict: Dictionary containing:
                - 'x': 1D array of x coordinates
                - 'y': 1D array of y coordinates
                - 'z': 2D array of z values (height) with shape (y_res, x_res)
                - 'build_info': Dict with min/max bounds
        """
        # Calculate calculation resolution based on sampling factor
        calc_x_res = max(int(x_res * sampling_factor), 10)  # Ensure at least 10 points
        calc_y_res = max(int(y_res * sampling_factor), 10)

        # Find global bounds across all profiles - same as in compute_build_state_mesh
        if not self._profiles:
            raise ValueError("No profiles found to process")

        # Find global bounds across all profiles
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        # Get all unique y positions
        y_positions = sorted(set(
            y_pos for (_, _, y_pos) in self._profiles.keys()
        ))

        # Find x bounds from all track profiles
        for (_, track_idx, _) in self._profiles.keys():
            try:
                # Try to get profile function for any y position
                profile = self._profiles[(0, track_idx, y_positions[0])]
                x_min = min(x_min, profile.start_x)
                x_max = max(x_max, profile.end_x)
            except (KeyError, AttributeError):
                continue

        # Y bounds from actual y positions
        if y_positions:
            y_min = min(y_positions)
            y_max = max(y_positions)
        else:
            raise ValueError("No y positions found in profiles")

        if x_min == float('inf'):
            raise ValueError("Could not determine bounds from available profiles")

        # Add small padding to bounds
        padding = 0.001  # 1mm padding
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # Create initial grid at lower resolution
        xs_low = np.linspace(x_min, x_max, calc_x_res)
        ys_low = np.linspace(y_min, y_max, calc_y_res)

        # Pre-allocate 2D array for z values
        Z_low = np.zeros((calc_y_res, calc_x_res))

        # Compute z for each (x, y) by calling get_cross_section at low resolution
        for j, yy in enumerate(ys_low):
            # For each y, get maximum height profile function
            cross_section_func = self.get_cross_section(yy)

            # Apply function across all x values in this row
            for i, xx in enumerate(xs_low):
                Z_low[j, i] = cross_section_func(xx)

        # Create final grid at requested resolution
        xs = np.linspace(x_min, x_max, x_res)
        ys = np.linspace(y_min, y_max, y_res)

        # Skip interpolation if requested resolution is the same as calculation resolution
        if x_res == calc_x_res and y_res == calc_y_res or interp_method == 'none':
            Z = Z_low
        else:
            # Interpolate based on the selected method
            if interp_method == 'spline':
                # RectBivariateSpline interpolation (default)
                from scipy.interpolate import RectBivariateSpline
                interpolator = RectBivariateSpline(ys_low, xs_low, Z_low, kx=2, ky=2)
                Z = interpolator(ys, xs, grid=True)

            elif interp_method == 'linear' or interp_method == 'cubic':
                # RegularGridInterpolator with linear or cubic method
                from scipy.interpolate import RegularGridInterpolator

                # Create interpolator
                interpolator = RegularGridInterpolator((ys_low, xs_low), Z_low, method=interp_method)

                # Create grid points for evaluation
                X_high, Y_high = np.meshgrid(xs, ys)
                points = np.column_stack((Y_high.ravel(), X_high.ravel()))

                # Evaluate
                Z_flat = interpolator(points)
                Z = Z_flat.reshape(y_res, x_res)

            elif interp_method == 'nearest':
                # Nearest neighbor interpolation
                from scipy.interpolate import NearestNDInterpolator

                # Create source points and values
                X_low, Y_low = np.meshgrid(xs_low, ys_low)
                points = np.column_stack((Y_low.ravel(), X_low.ravel()))
                values = Z_low.ravel()

                # Create interpolator
                interpolator = NearestNDInterpolator(points, values)

                # Create target grid and evaluate
                X_high, Y_high = np.meshgrid(xs, ys)
                target_points = np.column_stack((Y_high.ravel(), X_high.ravel()))
                Z_flat = interpolator(target_points)
                Z = Z_flat.reshape(y_res, x_res)

            elif interp_method == 'smooth_spline':
                # SmoothBivariateSpline with light smoothing
                from scipy.interpolate import SmoothBivariateSpline

                # Flatten low-res grid for input
                X_low, Y_low = np.meshgrid(xs_low, ys_low)
                x_flat = X_low.ravel()
                y_flat = Y_low.ravel()
                z_flat = Z_low.ravel()

                # Calculate a reasonable smoothing parameter based on data size
                s = len(x_flat) * 0.01  # Light smoothing

                # Create interpolator with smoothing
                try:
                    interpolator = SmoothBivariateSpline(y_flat, x_flat, z_flat, s=s)
                    Z = interpolator(ys, xs, grid=True)
                except Exception as e:
                    print(f"SmoothBivariateSpline failed: {e}, falling back to RectBivariateSpline")
                    # Fallback to RectBivariateSpline if SmoothBivariateSpline fails
                    from scipy.interpolate import RectBivariateSpline
                    interpolator = RectBivariateSpline(ys_low, xs_low, Z_low, kx=2, ky=2)
                    Z = interpolator(ys, xs, grid=True)

            else:
                # Unknown method, use the low-res data directly
                print(f"Warning: Unknown interpolation method '{interp_method}', using low-res data")
                Z = Z_low

        # Return structured data ready for Plotly Surface plot
        return {
            'x': xs,
            'y': ys,
            'z': Z,
            'build_info': {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'z_min': Z.min(),
                'z_max': Z.max(),
            },
            'metadata': {
                'calc_resolution': (calc_x_res, calc_y_res),
                'final_resolution': (x_res, y_res),
                'interpolation': interp_method
            }
        }

    def generate_surface_mesh(
            self,
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            x_res: int = 50,
            y_res: int = 50
    ):
        """
        Generate a 3D mesh representing the current surface of the entire build
        by sampling across a grid in x-y space, using get_cross_section to get
        the maximum height at each point.

        Args:
            x_min, x_max: Bounds in x (meters)
            y_min, y_max: Bounds in y (meters)
            x_res, y_res: Number of samples along x and y

        Returns:
            Open3D TriangleMesh representing the current top surface of the build.
        """
        import open3d as o3d

        # 1) Create a grid of (x, y) points
        xs = np.linspace(x_min, x_max, x_res)
        ys = np.linspace(y_min, y_max, y_res)
        X, Y = np.meshgrid(xs, ys)  # shape: (y_res, x_res)

        # Flatten to get a 2D array of shape (N, 2)
        points_2d = np.column_stack((X.ravel(), Y.ravel()))

        # 2) Compute z for each (x, y) by calling get_cross_section
        Z = []
        for (xx, yy) in points_2d:
            # For each y, get maximum height profile
            cross_section_func = self.get_cross_section(yy)
            z_val = cross_section_func(xx)
            Z.append(z_val)
        Z = np.array(Z)

        # Combine x, y, z into 3D vertex coordinates
        vertices_3d = np.column_stack((points_2d, Z))  # shape (N, 3)

        # 3) Triangulate the 2D (x,y) domain using SciPy's Delaunay
        tri = Delaunay(points_2d)
        faces = tri.simplices  # shape (M, 3) indexing into vertices_3d

        # 4) Create an Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Optional: compute normals for better lighting
        mesh.compute_vertex_normals()

        return mesh

    def compute_build_state_mesh(
            self,
            x_res: int = 50,
            y_res: int = 50
    ):

        """
        Generate a 3D mesh representing the current state of the build.
        Uses get_cross_section to create a single mesh of the current surface.

        Args:
            x_res, y_res: Number of samples along x and y

        Returns:
            Open3D TriangleMesh representing the current top surface of the build.
        """
        if not O3D_AVAILABLE:
            raise ImportError("Open3D is not installed. Cannot generate mesh. Install with: pip install open3d")

        # Find all available profiles to determine bounds
        if not self._profiles:
            raise ValueError("No profiles found to process")

        # Find global bounds across all profiles
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        # Get all unique y positions
        y_positions = sorted(set(
            y_pos for (_, _, y_pos) in self._profiles.keys()
        ))

        # Find x bounds from all track profiles
        for (_, track_idx, _) in self._profiles.keys():
            try:
                # Try to get profile function for any y position
                profile = self._profiles[(0, track_idx, y_positions[0])]
                x_min = min(x_min, profile.start_x)
                x_max = max(x_max, profile.end_x)
            except (KeyError, AttributeError):
                continue

        # Y bounds from actual y positions
        if y_positions:
            y_min = min(y_positions)
            y_max = max(y_positions)
        else:
            raise ValueError("No y positions found in profiles")

        if x_min == float('inf'):
            raise ValueError("Could not determine bounds from available profiles")

        # Add small padding to bounds
        padding = 0.001  # 1mm padding
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # Generate the surface mesh
        surface_mesh = self.generate_surface_mesh(
            x_min,
            x_max,
            y_min,
            y_max,
            x_res,
            y_res
        )

        return surface_mesh

    def save_build_state_mesh(
            self,
            output_path: str,
            x_res: int = 50,
            y_res: int = 50
    ) -> None:
        """
        Generate and save a 3D mesh representing the current state of the build.
        Uses get_cross_section to create a single mesh of the current surface.
        The bounds are automatically determined from the available profiles.

        Args:
            output_path: Path to save the surface mesh (should end in .ply or .stl)
            x_res, y_res: Number of samples along x and y
        """
        surface_mesh = self.compute_build_state_mesh(x_res, y_res)

        # Save the mesh
        file_extension = output_path.lower().split('.')[-1]
        if file_extension == 'ply':
            o3d.io.write_triangle_mesh(output_path, surface_mesh, write_ascii=True)
        elif file_extension == 'stl':
            o3d.io.write_triangle_mesh(output_path, surface_mesh)
        else:
            raise ValueError("Output file must have .ply or .stl extension")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Initialize manager with typical parameters
    hatch_distance = 0.0008  # 1mm between tracks
    manager = CladProfileManager(hatch_distance=hatch_distance, num_tracks=3)

    # Test parameters
    track_width = 0.0012  # 1.2mm track width
    track_height = 0.0004  # 0.4mm track height
    angle = 0.7125633369575821
    y_positions = np.linspace(0, 0.01, 5)  # 5 points along 10mm length_between

    # Build two layers with 3 tracks each
    for layer in range(2):
        for track in range(3):
            for y_pos in y_positions:
                profile = manager.add_profile(layer, track, y_pos, track_width, track_height)
                print()
                print('layer', layer, 'track', track, 'y_pos', y_pos)
                print('start_x:', profile.start_x, 'x_center:', profile.x_center, 'end_x:', profile.end_x)
                print('x_track:', manager.get_x_position(track), 'baseline:', profile.get_baseline())

    manager.plot_cross_section(0, 0.005)
    manager.plot_all_layers(0.005)
    plt.show()

    # Visualization functions
    def plot_layer_cross_section(manager, layer, y_pos, ax, color, label):
        x = np.linspace(-0.001, 0.004, 200)  # Plot range

        # Get cross section function
        cross_section = manager.get_layer_cross_section(layer, y_pos)

        # Calculate heights and plot
        z = np.array([cross_section(x_val) for x_val in x])
        ax.plot(x * 1000, z * 1000, color=color, label=label)


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Define consistent axis ranges
    x_min, x_max = -1, 4  # mm
    z_min, z_max = 0, 1.0  # mm

    # First subplot: Original visualization
    mid_y = y_positions[len(y_positions) // 2]
    plot_layer_cross_section(manager, 0, mid_y, ax1, 'b', 'Layer 0')
    plot_layer_cross_section(manager, 1, mid_y, ax1, 'r', 'Layer 1')

    # Formatting first subplot
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Z Position (mm)')
    ax1.set_title('Multi-Layer Clad ProfileGenerator Cross Section - Combined View')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(z_min, z_max)

    # Make sure both axes have the same scale
    ax1.set_aspect('equal')

    # Second subplot: Individual tracks with baselines
    x = np.linspace(-0.001, 0.004, 200)  # Plot range
    colors = ['blue', 'green', 'red']  # Different color for each track

    for layer in range(2):
        for track in range(3):
            # Get profile function for this track
            profile = manager.get_profile_function(layer, track, mid_y)
            if profile is not None:
                # Calculate heights
                z = np.array([profile(x_val) for x_val in x])
                # Plot track profile
                ax2.plot(x * 1000, z * 1000,
                         color=colors[track],
                         alpha=0.5,  # Make lines semi-transparent
                         linestyle='-' if layer == 0 else '--',  # Solid for layer 0, dashed for layer 1
                         label=f'Layer {layer} Track {track}')

                # Plot baseline
                baseline = profile.get_baseline()
                x_track = manager.get_x_position(track)
                # Plot baseline only in the region where the track exists
                x_baseline = np.array([
                    x_track - track_width / 2,  # Start of track
                    x_track + track_width / 2  # End of track
                ])
                ax2.plot(x_baseline * 1000, [baseline * 1000, baseline * 1000],
                         color=colors[track],
                         alpha=0.7,
                         linestyle=':' if layer == 0 else '-.',
                         linewidth=2,
                         label=f'Baseline L{layer}T{track}')

    # Formatting second subplot
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Z Position (mm)')
    ax2.set_title('Multi-Layer Clad ProfileGenerator Cross Section - Individual Tracks with Baselines')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(z_min, z_max)

    # Make sure both axes have the same scale
    ax2.set_aspect('equal')

    # Adjust layout while maintaining aspect ratio
    plt.tight_layout()


    # Basic validation checks
    def run_validation(manager, y_pos):
        # Check layer 1 height is above layer 0
        x_check = np.linspace(0, hatch_distance * 2, 10)
        layer0 = manager.get_layer_cross_section(0, y_pos)
        layer1 = manager.get_layer_cross_section(1, y_pos)

        for x in x_check:
            z0 = layer0(x)
            z1 = layer1(x)
            assert z1 >= z0, f"Layer violation at x={x}: z1={z1} < z0={z0}"

        print("✓ Layers properly stacked")

        '''# Check material balance
        mid_track = 1
        balance_area, ref_height = manager.calculate_material_balance(
            1, mid_track, y_pos,
            manager.get_x_position(mid_track),
            manager.get_x_position(mid_track) + track_width,
            height=0.0
        )
        print(f"✓ Material balance area for middle track: {balance_area * 1e6:.2f} mm²")
        print(f"✓ Reference height: {ref_height * 1000:.3f} mm")'''


    try:
        run_validation(manager, mid_y)
    except AssertionError as e:
        print(f"Validation failed: {e}")

    plt.show()

    # Create a new figure for the cubic part
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Part dimensions
    part_width = 0.02  # 20 mm
    part_length = 0.04  # 40 mm
    part_height = 0.01  # 10 mm

    # Calculate number of tracks and layers needed
    num_tracks = int(np.ceil(part_width / hatch_distance))
    layer_height = track_height * 0.9  # Approximate effective layer height (90% of track height)
    num_layers = int(np.ceil(part_height / layer_height))
    y_positions = np.linspace(0, part_length, 5)  # 20 points along length_between for smoother visualization

    print(f"Building cubic part with:")
    print(f"Number of tracks per layer: {num_tracks}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of ypos: {len(y_positions)}")
    print(f"Number of profiles: {num_tracks * num_layers * len(y_positions)}")

    # Initialize new manager for the cubic part
    cubic_manager = CladProfileManager(hatch_distance=hatch_distance, num_tracks=num_tracks)


    # Build all layers and tracks
    for layer in range(num_layers):
        for track in range(num_tracks):
            for y_pos in y_positions:
                cubic_manager.add_profile(layer, track, y_pos, track_width, track_height)
    print(f"Cubic part built")
    # Define consistent axis ranges for cubic part
    x_min_cubic = -1  # mm
    x_max_cubic = 22  # mm
    z_min_cubic = 0  # mm
    z_max_cubic = 13  # mm

    mid_y = part_length / 2
    for layer in range(num_layers):
        layer_color = plt.cm.viridis(layer / num_layers)
        cross_section = cubic_manager.get_layer_cross_section(layer, mid_y)
        x = np.linspace(-0.001, part_width + 0.001, 500)
        z = np.array([cross_section(x_val) for x_val in x])
        ax1.plot(x * 1000, z * 1000, color=layer_color)

    # Format first subplot
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Z Position (mm)')
    ax1.set_title('Cubic Part Cross Sections')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(x_min_cubic, x_max_cubic)
    ax1.set_ylim(z_min_cubic, z_max_cubic)

    # Plot individual tracks at middle y-position
    mid_y = part_length / 2
    x = np.linspace(-0.001, part_width + 0.001, 500)

    for layer in range(num_layers):
        layer_color = plt.cm.viridis(layer / num_layers)
        for track in range(num_tracks):
            profile = cubic_manager.get_profile_function(layer, track, mid_y)
            if profile is not None:
                # Plot track profile
                z = np.array([profile(x_val) for x_val in x])
                ax2.plot(x * 1000, z * 1000, color=layer_color, alpha=0.3)

                # Plot baseline
                baseline = profile.get_baseline()
                x_track = cubic_manager.get_x_position(track)
                x_baseline = np.array([
                    x_track - track_width / 2,
                    x_track + track_width / 2
                ])
                ax2.plot(x_baseline * 1000, [baseline * 1000, baseline * 1000],
                         color=layer_color, alpha=0.5, linestyle=':', linewidth=1)

    # Format second subplot
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Z Position (mm)')
    ax2.set_title('Cubic Part Individual Tracks at Mid-Length (y=20mm)')
    ax2.grid(True)
    ax2.set_xlim(x_min_cubic, x_max_cubic)
    ax2.set_ylim(z_min_cubic, z_max_cubic)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Run validation on cubic part
    try:
        print("\nValidating cubic part:")
        run_validation(cubic_manager, mid_y)

        # Additional validations specific to cubic part
        print(f"✓ Total width: {(num_tracks * hatch_distance * 1000):.1f}mm")
        print(f"✓ Total height: {(num_layers * layer_height * 1000):.1f}mm")
    except AssertionError as e:
        print(f"Validation failed: {e}")

    # Create a new figure for the cubic part with alternating directions
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Part dimensions
    part_width = 0.02  # 20 mm
    part_length = 0.04  # 40 mm
    part_height = 0.01  # 10 mm

    # Calculate number of tracks and layers needed
    num_tracks = int(np.ceil(part_width / hatch_distance))
    layer_height = track_height * 0.9  # Approximate effective layer height (90% of track height)
    num_layers = int(np.ceil(part_height / layer_height))
    y_positions = np.linspace(0, part_length, 5)  # 5 points along length_between for smoother visualization

    print(f"Building cubic part with alternating directions:")
    print(f"Number of tracks per layer: {num_tracks}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of y positions: {len(y_positions)}")
    print(f"Number of profiles: {num_tracks * num_layers * len(y_positions)}")

    # Initialize new manager for the cubic part
    cubic_manager = CladProfileManager(hatch_distance=hatch_distance, num_tracks=num_tracks)

    # Build all layers and tracks with alternating directions
    for layer in range(num_layers):
        reverse = (layer % 2) == 1  # Odd layers build in right_to_left

        if not reverse:
            # Forward direction (left to right)
            track_range = range(num_tracks)
        else:
            # Reverse direction (right to left)
            track_range = range(num_tracks - 1, -1, -1)

        print(f"Building layer {layer} - {'right_to_left' if reverse else 'forward'}")
        for track in track_range:
            for y_pos in y_positions:
                cubic_manager.add_profile(layer, track, y_pos, track_width, track_height)

    print(f"Cubic part built")

    # Define consistent axis ranges for cubic part
    x_min_cubic = -1  # mm
    x_max_cubic = 22  # mm
    z_min_cubic = 0  # mm
    z_max_cubic = 14  # mm

    # First subplot: Layer cross sections
    mid_y = part_length / 2
    for layer in range(num_layers):
        layer_color = plt.cm.viridis(layer / num_layers)
        cross_section = cubic_manager.get_layer_cross_section(layer, mid_y)
        x = np.linspace(-0.001, part_width + 0.001, 500)
        z = np.array([cross_section(x_val) for x_val in x])
        ax1.plot(x * 1000, z * 1000, color=layer_color)

    # Format first subplot
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Z Position (mm)')
    ax1.set_title('Cubic Part Cross Sections - Alternating Build Directions')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(x_min_cubic, x_max_cubic)
    ax1.set_ylim(z_min_cubic, z_max_cubic)

    # Second subplot: Individual tracks
    x = np.linspace(-0.001, part_width + 0.001, 500)

    for layer in range(num_layers):
        layer_color = plt.cm.viridis(layer / num_layers)
        reverse = (layer % 2) == 1

        # Use appropriate track range based on build direction
        if not reverse:
            track_range = range(num_tracks)
        else:
            track_range = range(num_tracks - 1, -1, -1)

        for track in track_range:
            profile = cubic_manager.get_profile_function(layer, track, mid_y)
            if profile is not None:
                # Plot track profile
                z = np.array([profile(x_val) for x_val in x])
                linestyle = '--' if reverse else '-'
                ax2.plot(x * 1000, z * 1000, color=layer_color, alpha=0.3,
                         linestyle=linestyle)

                # Plot baseline
                baseline = profile.get_baseline()
                x_track = cubic_manager.get_x_position(track)
                x_baseline = np.array([
                    x_track - track_width / 2,
                    x_track + track_width / 2
                ])
                ax2.plot(x_baseline * 1000, [baseline * 1000, baseline * 1000],
                         color=layer_color, alpha=0.5, linestyle=':', linewidth=1)

    # Format second subplot
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Z Position (mm)')
    ax2.set_title('Individual Tracks at Mid-Length (y=20mm) - Alternating Build Directions')
    ax2.grid(True)
    ax2.set_xlim(x_min_cubic, x_max_cubic)
    ax2.set_ylim(z_min_cubic, z_max_cubic)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Run validation on alternating direction cubic part
    try:
        print("\nValidating cubic part with alternating directions:")
        run_validation(cubic_manager, mid_y)

        # Additional validations specific to cubic part
        print(f"✓ Total width: {(num_tracks * hatch_distance * 1000):.1f}mm")
        print(f"✓ Total height: {(num_layers * layer_height * 1000):.1f}mm")
    except AssertionError as e:
        print(f"Validation failed: {e}")
