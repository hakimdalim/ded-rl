import os
import warnings

import numpy as np
from typing import Callable, Tuple, Optional, Union, Dict

from geometry.clad_interpolator import interpolated_track_surface
from geometry.clad_profile_function import ParabolicCladProfile

class ActivatedVolume:

    def __init__(
            self,
            shape: Tuple[int, int, int],
            voxel_size: Union[float, Tuple[float, float, float], np.ndarray],
            substrate_nz: Optional[int] = None,
            substrate_height: Optional[float] = None
    ):
        """Initialize a build volume with given dimensions.

        Args:
            shape: Grid dimensions (nx, ny, nz)
            voxel_size: Voxel dimensions in meters as either:
                - uniform size (float)
                - per-axis tuple (float, float, float)
                - numpy array of shape (3,)
            substrate_nz: Number of z-voxels to activate from bottom (z=0).
                If both substrate_nz and substrate_height are provided, they must be consistent.
            substrate_height: Physical height of substrate in meters.
                If both substrate_nz and substrate_height are provided, they must be consistent.
        """
        # Standardize voxel_size
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        self.voxel_size = np.asarray(voxel_size, dtype=float)

        # Calculate substrate_nz
        self.substrate_nz = self._resolve_substrate(substrate_nz, substrate_height, self.voxel_size[2])

        self.shape = shape
        self.activated = np.zeros(shape=shape, dtype=bool)
        if self.substrate_nz > 0:
            self.activated[:, :, :self.substrate_nz] = True

    def _resolve_substrate(self, substrate_nz: Optional[int],
                           substrate_height: Optional[float],
                           voxel_z: float) -> int:
        """Resolve substrate specification to z-voxel count.

        Args:
            substrate_nz: Number of z-voxels for substrate (optional)
            substrate_height: Physical height of substrate in meters (optional)
            voxel_z: Size of one voxel in z-direction in meters

        Returns:
            Number of z-voxels for substrate

        Raises:
            ValueError: If both arguments provided but inconsistent
        """
        if substrate_nz is not None and substrate_height is not None:
            # Both provided - verify consistency
            calculated_nz = int(np.ceil(substrate_height / voxel_z))
            if calculated_nz != substrate_nz:
                raise ValueError(f"Inconsistent substrate: height {substrate_height}m "
                                 f"implies {calculated_nz} voxels, but got {substrate_nz}")
            return substrate_nz
        elif substrate_height is not None:
            # Only height provided
            return int(np.ceil(substrate_height / voxel_z))
        elif substrate_nz is not None:
            # Only voxel count provided
            return substrate_nz
        else:
            # Neither provided
            return 0

    @classmethod
    def from_dimensions(cls, dimensions: Tuple[float, float, float],
                        voxel_size: Union[float, Tuple[float, float, float]],
                        substrate_nz: Optional[int] = None,
                        substrate_height: Optional[float] = None) -> 'ActivatedVolume':
        """Create volume from physical dimensions rather than grid shape.

        Args:
            dimensions: Physical dimensions (x, y, z) in meters
            voxel_size: Voxel dimensions in meters, either uniform (float) or per-axis (tuple)
            substrate_nz: Number of z-voxels to activate from bottom (z=0).
                If both substrate_nz and substrate_height are provided, they must be consistent.
            substrate_height: Physical height of substrate in meters.
                If both substrate_nz and substrate_height are provided, they must be consistent.

        Returns:
            ActivatedVolume instance with computed grid shape
        """
        # Standardize voxel_size
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        voxel_size = np.array(voxel_size, dtype=float)

        # Calculate grid shape
        shape = tuple(np.ceil(np.array(dimensions) / voxel_size).astype(int))

        return cls(shape, voxel_size, substrate_nz, substrate_height)

    def reset(self):
        """Reset the activated voxels to the substrate only."""
        self.activated.fill(False)
        if self.substrate_nz > 0:
            self.activated[:, :, :self.substrate_nz] = True

    def add_track_section(
            self,
            start_profile: ParabolicCladProfile,
            end_profile: ParabolicCladProfile,
            length_between: float,
            y_position: float,
    ):
        """
        Adds a track section to the build volume by activating voxels below the track surface
        using direct interpolation between profiles.

        Important Assumptions:
        1. Track profiles are already in the voxel volume's coordinate system
        2. Tracks are strictly along the y-axis, with the start profile to the left of the end profile
        3. No coordinate transformation is needed

        Args:
            start_profile: ProfileGenerator at start of track section, in volume coordinates
            end_profile: ProfileGenerator at end of track section, in volume coordinates
            length_between: Length of track section in meters
            y_position: Starting y-coordinate in volume coordinates
        """
        # Calculate bounds in x direction
        padding = 2 * max(self.voxel_size)
        x_min = min(start_profile.start_x, end_profile.start_x) - padding
        x_max = max(start_profile.end_x, end_profile.end_x) + padding
        y_min = y_position - padding
        y_max = y_position + length_between + padding

        if (np.isnan(x_min) or np.isnan(x_max) or np.isnan(y_min) or np.isnan(y_max)):
            warnings.warn(
                f"Assuming zero volume: NaN values detected at y={y_position * 1000:.3f}mm. Skipping volume activation.",
                RuntimeWarning)
            return

        # Convert to voxel indices
        voxel_x_min = max(0, int(np.floor(x_min / self.voxel_size[0])))
        voxel_x_max = min(self.shape[0], int(np.ceil(x_max / self.voxel_size[0])))
        voxel_y_min = max(0, int(np.floor(y_min / self.voxel_size[1])))
        voxel_y_max = min(self.shape[1], int(np.ceil(y_max / self.voxel_size[1])))

        # Extract coordinates for the relevant volume subset
        x_coords = np.arange(voxel_x_min, voxel_x_max) * self.voxel_size[0]
        y_coords = np.arange(voxel_y_min, voxel_y_max) * self.voxel_size[1]
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Stack coordinates for interpolated_track_surface
        points = np.stack([X, Y], axis=-1)

        # Calculate interpolated surface heights
        surface_heights = interpolated_track_surface(
            points=points,
            start_profile=start_profile,
            end_profile=end_profile,
            length_between=length_between,
            y_position=y_position
        )

        # Convert heights to voxel indices and clip to volume bounds
        height_voxels = np.minimum(
            np.ceil(np.where(np.isnan(surface_heights), 0, surface_heights) / self.voxel_size[2]).astype(int),
            self.shape[2]
        )

        # Create z-indices array for broadcasting
        z_indices = np.arange(self.shape[2])

        # Create the activation mask using broadcasting
        activation_mask = z_indices[None, None, :] < height_voxels[..., None]

        # Apply the mask to the volume using direct indexing
        self.activated[
        voxel_x_min:voxel_x_max,
        voxel_y_min:voxel_y_max,
        :
        ] |= activation_mask

    def visualize_voxels_plotly(self):
        """
        Efficiently visualize voxels by rendering only external faces.

        Returns:
            Plotly figure object
        """
        from voxel_visualize import VoxelVisualizer
        return VoxelVisualizer(self.shape, self.voxel_size).create_figure(
            activated=self.activated,
            substrate_nz=self.substrate_nz,
            title="Voxel Visualization"
        )

def extract_volume_coordinates(
        center: Tuple[float, float, float],
        half_width: float,
        half_length: float,
        half_depth: float,
        volume_shape: Tuple[int, int, int],
        voxel_size: Union[float, Tuple[float, float, float], np.ndarray],
        padding: Optional[float] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, ...]]]:
    """Extract coordinates and indices for a volume of interest within the build volume.

    Assumes rotations only occur around the z-axis. The volume is defined by its center
    and half-dimensions. Returns both physical coordinates and voxel indices.

    Args:
        center: (x, y, z) coordinates of volume center in physical units
        half_width: Half the width (x) of the volume
        half_length: Half the length_between (y) of the volume
        half_depth: Half the depth (z) of the volume
        volume_shape: Shape of the full build volume (nx, ny, nz)
        voxel_size: Voxel dimensions as either:
                - uniform size (float)
                - per-axis tuple (float, float, float)
                - numpy array of shape (3,)
        padding: Optional padding to add around volume (defaults to 2 * max(voxel_size))

    Returns:
        Dictionary containing:
            'physical_coords': (x_coords, y_coords, z_coords) arrays of physical coordinates
            'voxel_indices': (x_min, x_max, y_min, y_max, z_min, z_max) for slicing
            'points_grid': Meshgrid of all points in the volume

    Raises:
        ValueError: If input parameters are invalid or if the volume is completely outside
                   the build volume bounds
    """
    # Input validation
    if any(x <= 0 for x in [half_width, half_length, half_depth]):
        raise ValueError("Half-dimensions must be positive")

    if any(x <= 0 for x in volume_shape):
        raise ValueError("Volume shape dimensions must be positive")

    # Standardize voxel_size to handle both scalar and per-axis inputs
    if isinstance(voxel_size, (int, float)):
        voxel_size = (float(voxel_size),) * 3
    voxel_size = np.asarray(voxel_size, dtype=float)

    if any(x <= 0 for x in voxel_size):
        raise ValueError("Voxel size must be positive")

    # Set default padding if not provided
    if padding is None:
        padding = 2 * max(voxel_size)

    # Calculate max extent for xy-plane due to potential rotation
    max_xy_extent = max(half_width, half_length)

    # Calculate physical ranges with padding
    x_range = (center[0] - max_xy_extent - padding, center[0] + max_xy_extent + padding)
    y_range = (center[1] - max_xy_extent - padding, center[1] + max_xy_extent + padding)
    z_range = (center[2] - half_depth - padding, center[2] + half_depth + padding)

    # Convert to voxel indices - use integer division to align with voxel grid
    x_min = int(x_range[0] // voxel_size[0])
    x_max = int(x_range[1] // voxel_size[0]) + 1
    y_min = int(y_range[0] // voxel_size[1])
    y_max = int(y_range[1] // voxel_size[1]) + 1
    z_min = int(z_range[0] // voxel_size[2])
    z_max = int(z_range[1] // voxel_size[2]) + 1

    # Store original indices for clipping check
    orig_indices = (x_min, x_max, y_min, y_max, z_min, z_max)

    # Clip to volume bounds
    x_min = max(0, min(x_min, volume_shape[0]))
    x_max = max(0, min(x_max, volume_shape[0]))
    y_min = max(0, min(y_min, volume_shape[1]))
    y_max = max(0, min(y_max, volume_shape[1]))
    z_min = max(0, min(z_min, volume_shape[2]))
    z_max = max(0, min(z_max, volume_shape[2]))

    # Check if we have a valid volume after clipping
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        raise ValueError("Volume of interest is completely outside build volume bounds")

    # Check if we clipped significantly and warn
    orig_vol = (orig_indices[1] - orig_indices[0]) * \
               (orig_indices[3] - orig_indices[2]) * \
               (orig_indices[5] - orig_indices[4])
    clipped_vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    if clipped_vol < 0.5 * orig_vol:  # More than 50% volume lost
        warnings.warn(
            "Significant volume clipping occurred. Check if volume of interest is near build volume bounds.",
            RuntimeWarning
        )

    # Extract the physical coordinates
    x_coords = np.arange(x_min, x_max) * voxel_size[0]
    y_coords = np.arange(y_min, y_max) * voxel_size[1]
    z_coords = np.arange(z_min, z_max) * voxel_size[2]

    # Create meshgrid of points
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1)

    return {
        'physical_coords': (x_coords, y_coords, z_coords),
        'voxel_indices': (x_min, x_max, y_min, y_max, z_min, z_max),
        'points_grid': points
    }