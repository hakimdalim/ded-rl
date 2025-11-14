# -----------------------------------------------------------------------------
# _base_camera.py
# -----------------------------------------------------------------------------
# Unified camera module with a shared base class and two projection variants:
# - OrthographicCamera (parallel rays; fixed plane size in meters)
# - PerspectiveCamera  (converging rays; FOV-driven projection)
#
# Optional following behavior is provided via a light-weight mixin that can be
# combined with either camera variant:
# - FollowingOrthographicCamera
# - FollowingPerspectiveCamera
#
# Key Concepts:
# -------------
# - **Sensor Plane**: The image plane at the camera position where pixels are sampled
#   - plane_size = physical dimensions of the sensor (meters)
#   - For perspective cameras, this is NOT the field of view at the target
# - **Field of View (FOV)**: The actual physical area visible at a given distance
#   - For orthographic: FOV = plane_size (constant at all distances)
#   - For perspective: FOV = plane_size * distance / focal_length
# - **Following Cameras**: Track a moving point with fixed relative offset
#   - tilt_angle: Rotates camera position around the X-axis
#   - Camera always looks back at the tracked point
#
# The public API:
#   - Pose: set_position, set_direction, look_at, set_from_spherical
#   - Coverage/Sampling: set_plane_size, set_pixel_size, set_resolution
#   - FOV Control (Perspective only): set_fov_at_distance, get_fov_at_distance
#   - Rendering: render_first_hit(...)
# -----------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional

# Try to import scipy for better resize quality
try:
    from scipy.ndimage import zoom as nd_zoom

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ArrayLike3 = Tuple[float, float, float]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    """Return v normalized; if zero length, return v unchanged."""
    n = np.linalg.norm(v)
    return v if n == 0.0 else (v / n)


# -----------------------------------------------------------------------------
# Base camera (shared pose, sampling, and rendering pipeline)
# -----------------------------------------------------------------------------

class BaseCamera:
    """
    Production-ready **camera base class** that owns common pose/state, sampling,
    and the vectorized **first-hit binner**. Projection is delegated to
    subclasses via :meth:`_project_cam_to_plane`.

    Terminology
    -----------
    - **Sensor Plane**: The image plane at camera position with dimensions plane_size
      - For orthographic: sensor size = field of view at all distances
      - For perspective: sensor size ≠ field of view (FOV depends on distance)
    - **Pose**: Camera position (pos) and orthonormal basis (right, up, forward)
      - Camera looks along the +forward direction
    - **Target**: Point the camera is looking at (used by look_at and following cameras)

    Definition (meters)
    -------------------
    - Pose: ``pos`` (world position of camera/sensor center) and orthonormal basis
      ``(right, up, forward)`` with the camera **looking along** ``forward``.
    - Coverage/Sampling: **sensor plane size** in meters (``plane_size = (width, height)``)
      and either fixed pixel size (meters/pixel) or fixed resolution (pixels).

    Rendering
    ---------
    The renderer casts rays defined by the subclass projection through the
    sensor plane centered at ``pos`` with normal ``forward`` and bins
    first-visible temperatures.

    Fast path
    ---------
    Update only what changes frame-to-frame:
    - Call :meth:`set_position` when the camera moves but keeps orientation.
    - Call :meth:`look_at` or :meth:`set_direction` when orientation changes.
    - Call :meth:`set_plane_size`, :meth:`set_pixel_size`, or
      :meth:`set_resolution` to control the sensor plane coverage / sampling.
    """

    # Pose
    pos: np.ndarray  # Camera position in world coordinates
    target: np.ndarray  # Point the camera is looking at
    right: np.ndarray  # Right vector of camera basis
    up: np.ndarray  # Up vector of camera basis
    forward: np.ndarray  # Forward vector (viewing direction)

    # Sampling/coverage (meters, pixels)
    plane_size: Tuple[float, float]  # Sensor size (width, height) in meters
    pixel_size_xy: Tuple[float, float]  # (px, py) in meters per pixel
    resolution_wh: Tuple[int, int]  # (W, H) in pixels
    voxel_size_xyz: np.ndarray  # (dx, dy, dz) in meters

    def __init__(self,
                 pos: Optional[ArrayLike3] = None,
                 target: Optional[ArrayLike3] = None,
                 up_hint: ArrayLike3 = (0.0, 0.0, 1.0),
                 *,
                 plane_size: Tuple[float, float] = (0.06, 0.04),  # meters (sensor size)
                 pixel_size_xy: Optional[Tuple[float, float]] = None,
                 resolution_wh: Optional[Tuple[int, int]] = None,
                 voxel_size_xyz: Optional[Tuple[float, float, float]] = None,
                 fill_gaps: bool = False,
                 depth_tolerance: Optional[float] = None,
                 max_gap_size: int = 3) -> None:
        """
        up_hint: Which way is "up" in world space (default +Z). Prevents gimbal lock.
        voxel_size_xyz: For converting voxel indices to world coordinates.
        fill_gaps: If True, automatically fill gaps using depth-aware interpolation.
        depth_tolerance: Max depth difference (m) to consider "same surface". If None, auto-set to voxel_size/2.
        max_gap_size: Maximum distance (pixels) from hit to fill gaps.

        Sampling: Specify EITHER resolution_wh OR pixel_size_xy, not both.
        """
        # Pose init
        self.pos = np.zeros(3) if pos is None else np.asarray(pos, float)
        self.target = np.zeros(3) if target is None else np.asarray(target, float)
        self.right = np.array([1.0, 0.0, 0.0], float)
        self.up = np.array([0.0, 1.0, 0.0], float)
        self.forward = np.array([0.0, 0.0, -1.0], float)
        if target is not None:
            self.look_at(self.target, up_hint=up_hint)

        # Voxel size
        if voxel_size_xyz is None:
            self.voxel_size_xyz = np.array([1e-4, 1e-4, 1e-4], dtype=float)
        else:
            self.voxel_size_xyz = np.asarray(voxel_size_xyz, dtype=float)

        # Gap filling
        self.fill_gaps = bool(fill_gaps)
        if depth_tolerance is None and fill_gaps:
            # Auto-set to 3× voxel_size (more practical than 0.5×)
            # This accounts for numerical precision and projection variations
            self.depth_tolerance = float(np.mean(self.voxel_size_xyz) * 3.0)
        else:
            self.depth_tolerance = float(depth_tolerance) if depth_tolerance is not None else 0.0
        self.max_gap_size = int(max_gap_size)

        # Coverage/sampling: plane_size + (resolution OR pixel_size)
        self.plane_size = (float(plane_size[0]), float(plane_size[1]))

        # User specifies EITHER resolution OR pixel_size
        if resolution_wh is not None:
            self.resolution_wh = (int(resolution_wh[0]), int(resolution_wh[1]))
            self._derive_pixel_size_from_plane()
        else:
            # Default to 1mm pixels if not specified
            if pixel_size_xy is None:
                pixel_size_xy = (0.001, 0.001)
            self.pixel_size_xy = (float(pixel_size_xy[0]), float(pixel_size_xy[1]))
            self._derive_resolution_from_plane()

    # ----- internals -----
    @staticmethod
    def _basis_from(direction: np.ndarray, up_hint: ArrayLike3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute an orthonormal basis from a direction vector and an up-hint."""
        d = _norm(np.asarray(direction, float))
        u = _norm(np.asarray(up_hint, float))
        r = np.cross(d, u)
        if np.linalg.norm(r) < 1e-12:
            tmp = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            r = np.cross(d, tmp)
        r = _norm(r)
        u = _norm(np.cross(r, d))
        return r, u, d

    def _world_to_cam_rotation(self) -> np.ndarray:
        """Return the 3x3 rotation matrix from world to camera coordinates."""
        return np.stack([self.right, self.up, -self.forward], axis=0)  # rows

    def _derive_resolution_from_plane(self) -> None:
        """Derive resolution from plane size and pixel size."""
        W = max(1, int(np.ceil(self.plane_size[0] / self.pixel_size_xy[0])))
        H = max(1, int(np.ceil(self.plane_size[1] / self.pixel_size_xy[1])))
        self.resolution_wh = (W, H)

    def _derive_pixel_size_from_plane(self) -> None:
        """Derive pixel size from plane size and resolution."""
        W, H = self.resolution_wh
        self.pixel_size_xy = (self.plane_size[0] / max(1, W),
                              self.plane_size[1] / max(1, H))

    # ----- public API: pose -----
    def set_position(self, pos: ArrayLike3) -> None:
        """Set the camera's position in world coordinates."""
        self.pos = np.asarray(pos, float)

    def set_direction(self, direction: ArrayLike3, up_hint: ArrayLike3 = (0.0, 0.0, 1.0)) -> None:
        """Set the camera's orientation from a direction vector."""
        self.right, self.up, self.forward = self._basis_from(direction, up_hint)

    def look_at(self, target: ArrayLike3, up_hint: ArrayLike3 = (0.0, 0.0, 1.0)) -> None:
        """Orient the camera to look at a target point from its current position."""
        self.target = np.asarray(target, float)
        self.set_direction(self.target - self.pos, up_hint=up_hint)

    def set_from_spherical(self,
                           az_deg: float,
                           el_deg: float,
                           roll_deg: float = 0.0,
                           radius: float = 0.2,
                           target: ArrayLike3 = (0.0, 0.0, 0.0),
                           up_hint: ArrayLike3 = (0.0, 0.0, 1.0)) -> None:
        """
        Place camera on a sphere around ``target`` and face it.

        Parameters
        ----------
        az_deg : float
            Azimuth angle in degrees. 0° = +X axis, 90° = +Y axis
        el_deg : float
            Elevation angle in degrees from horizon. 0° = horizontal, +90° = zenith, -90° = nadir
        roll_deg : float
            Roll angle in degrees (rotation around viewing axis)
        radius : float
            Distance from target to camera (must be positive)
        target : ArrayLike3
            Point to look at and orbit around
        up_hint : ArrayLike3
            World-space up vector for camera orientation

        Raises
        ------
        ValueError
            If radius is not positive
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        roll = np.deg2rad(roll_deg)
        dir_world = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], float)
        dir_world = _norm(dir_world)
        self.pos = np.asarray(target, float) + radius * dir_world
        self.look_at(target, up_hint=up_hint)
        if abs(roll) > 1e-12:
            r, u = self.right, self.up
            self.right = r * np.cos(roll) + u * np.sin(roll)
            self.up = u * np.cos(roll) - r * np.sin(roll)

    # ----- public API: coverage/sampling -----
    def set_plane_size(self, width: float, height: float) -> None:
        """
        Set the sensor plane size (meters) and keep current pixel size.

        Note: For perspective cameras, this affects the field of view at the target.
        Use set_fov_at_distance() for more intuitive zoom control.

        Parameters
        ----------
        width : float
            Sensor width in meters (must be positive)
        height : float
            Sensor height in meters (must be positive)

        Raises
        ------
        ValueError
            If width or height is not positive
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"plane_size must be positive, got ({width}, {height})")
        self.plane_size = (float(width), float(height))
        self._derive_resolution_from_plane()

    def set_pixel_size(self, pixel_size: float) -> None:
        """
        Set pixel spacing (meters, square pixels); resolution derived from plane size.

        Parameters
        ----------
        pixel_size : float
            Pixel size in meters (must be positive)

        Raises
        ------
        ValueError
            If pixel_size is not positive
        """
        if pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")
        self.pixel_size_xy = (float(pixel_size), float(pixel_size))
        self._derive_resolution_from_plane()

    def set_resolution(self, width_px: int, height_px: int) -> None:
        """
        Set output resolution (pixels); pixel size will be derived from plane.

        Parameters
        ----------
        width_px : int
            Image width in pixels (must be positive)
        height_px : int
            Image height in pixels (must be positive)

        Raises
        ------
        ValueError
            If width_px or height_px is not positive
        """
        if width_px <= 0 or height_px <= 0:
            raise ValueError(f"resolution must be positive integers, got ({width_px}, {height_px})")
        self.resolution_wh = (int(width_px), int(height_px))
        self._derive_pixel_size_from_plane()

    # ----- getters -----
    def get_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the camera's orthonormal basis vectors (right, up, forward)."""
        return self.right, self.up, self.forward

    def world_to_camera_matrix(self) -> np.ndarray:
        """Return the 4x4 world-to-camera transformation matrix."""
        R = self._world_to_cam_rotation()
        t = -R @ self.pos
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    # ----- projection hook (must be implemented by subclasses) -----
    def _project_cam_to_plane(self, pos_cam: np.ndarray):
        """
        Map camera-space points to **sensor plane coordinates**.

        Parameters
        ----------
        pos_cam : (N,3) ndarray
            Points in camera coordinates.

        Returns
        -------
        u : (N,) ndarray
            Sensor plane X coordinates (meters), centered at 0.
        v : (N,) ndarray
            Sensor plane Y coordinates (meters), centered at 0.
        zc : (N,) ndarray
            Depth value for **visibility sorting** (larger = closer).
        valid : (N,) boolean ndarray
            Mask indicating which points are **projectable/visible**.

        Notes
        -----
        - Orthographic: (u, v) = (x, y); valid = True for all.
        - Perspective: (u, v) = (f * x / d, f * y / d), with d = -z (>0 in front).
        """
        raise NotImplementedError

    # ----- rendering: first-hit binner (vectorized) -----
    def render_first_hit(self,
                         volume_activated: np.ndarray,
                         temperature_field: np.ndarray,
                         ambient: float = 300.0,
                         roi_world: Optional[Tuple[float, float, float, float, float, float]] = None):
        """
        Render a **first-visible** thermal image using the camera's sensor plane
        anchored at ``self.pos`` with normal ``self.forward`` (meters).

        If fill_gaps=True was set during camera initialization, gaps are automatically
        filled using depth-aware interpolation.

        Parameters
        ----------
        volume_activated : (Nx,Ny,Nz) bool
            Activation mask.
        temperature_field : (Nx,Ny,Nz) float
            Per-voxel temperature.
        ambient : float
            Temperature used for pixels with no hit.
        roi_world : Optional[Tuple[float, float, float, float, float, float]]
            Region of interest in world coordinates: (x_min, x_max, y_min, y_max, z_min, z_max).
            If provided, only voxels within this box are projected (10-100x speedup for small regions).

        Returns
        -------
        img : (W,H) ndarray
            Temperature image (gap-filled if fill_gaps=True).
        extent : (xmin, xmax, ymin, ymax)
            Sensor plane coordinates (meters), centered at 0.
        """
        A = volume_activated
        T = temperature_field
        vs = self.voxel_size_xyz

        # Rotation world→cam; anchor is camera.pos
        R_wc = self._world_to_cam_rotation()

        # Output grid from plane size + pixel size
        W, H = self.resolution_wh
        px, py = self.pixel_size_xy
        half_w = 0.5 * self.plane_size[0]
        half_h = 0.5 * self.plane_size[1]
        extent = (-half_w, half_w, -half_h, half_h)

        # Early out: nothing active
        idx = np.argwhere(A)
        if idx.size == 0:
            img = np.full((W, H), float(ambient), dtype=T.dtype)
            return img, extent

        # ROI pre-filtering: Only project voxels in the region of interest
        if roi_world is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = roi_world
            pos_world = idx * vs

            # Filter voxels: keep only those inside the ROI box
            roi_mask = (
                    (pos_world[:, 0] >= x_min) & (pos_world[:, 0] <= x_max) &
                    (pos_world[:, 1] >= y_min) & (pos_world[:, 1] <= y_max) &
                    (pos_world[:, 2] >= z_min) & (pos_world[:, 2] <= z_max)
            )

            if not np.any(roi_mask):
                # No voxels in ROI
                img = np.full((W, H), float(ambient), dtype=T.dtype)
                return img, extent

            idx = idx[roi_mask]
            pos_world = pos_world[roi_mask]
        else:
            # No ROI: project all activated voxels
            pos_world = idx * vs

        # Project activated voxel centers into camera space, relative to camera.pos
        pos_cam = (R_wc @ (pos_world - self.pos).T).T  # (N,3)

        # Subclass-specific projection to the sensor plane
        u, v, zc, valid = self._project_cam_to_plane(pos_cam)
        if not np.any(valid):
            img = np.full((W, H), float(ambient), dtype=T.dtype)
            return img, extent

        u = u[valid]
        v = v[valid]
        zc = zc[valid]
        idx = idx[valid]

        # Bin to pixels in plane centered at (0,0) with extents [-half_w, half_w] × [-half_h, half_h]
        ix = np.floor((u + half_w) / px).astype(int)
        iy = np.floor((v + half_h) / py).astype(int)

        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not np.any(inside):
            img = np.full((W, H), float(ambient), dtype=T.dtype)
            return img, extent

        ix = ix[inside]
        iy = iy[inside]
        zc = zc[inside]
        flat = ix + W * iy

        # Per-pixel pick closest voxel: primary key = pixel, secondary = depth
        order = np.lexsort((-zc, flat))
        flat_sorted = flat[order]
        idx_sorted = idx[inside][order]
        zc_sorted = zc[order]  # Keep depth values for gap filling

        unique_flat, first_pos = np.unique(flat_sorted, return_index=True)
        vox_hit = idx_sorted[first_pos]
        depths_hit = zc_sorted[first_pos]
        temps = T[vox_hit[:, 0], vox_hit[:, 1], vox_hit[:, 2]]

        img = np.full((W, H), float(ambient), dtype=T.dtype)
        ix_hit = unique_flat % W
        iy_hit = unique_flat // W
        img[ix_hit, iy_hit] = temps

        # Apply gap filling if enabled
        if self.fill_gaps:
            try:
                from camera.fast_gap_filling import fill_gaps_fast
                depth_map = np.full((W, H), np.inf, dtype=float)
                depth_map[ix_hit, iy_hit] = depths_hit
                img = fill_gaps_fast(
                    img, depth_map, ambient,
                    depth_tolerance=self.depth_tolerance,
                    max_gap_size=self.max_gap_size
                )
            except ImportError as e:
                print(f"WARNING: Could not import depth_aware_gap_filling: {e}")
                print("Gap filling disabled. Place depth_aware_gap_filling.py in the same directory as _base_camera.py")

        return img, extent

    def calculate_roi_for_crop(
            self,
            window_center: ArrayLike3,
            window_size: Tuple[float, float],
            margin: float = 0.001
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate ROI bounding box for efficient rendering of a crop window.

        Base implementation creates a simple box around the window.
        Subclasses may override for camera-specific behavior (e.g., perspective frustum).

        Parameters
        ----------
        window_center : ArrayLike3
            Center of the crop window in world coordinates.
        window_size : Tuple[float, float]
            Crop window size (width, height) in meters.
        margin : float
            Extra margin around crop (default: 1mm). Must be non-negative.

        Returns
        -------
        roi_world : Tuple[float, float, float, float, float, float]
            ROI as (x_min, x_max, y_min, y_max, z_min, z_max).

        Raises
        ------
        ValueError
            If window_size contains non-positive values
            If margin is negative
        """
        # Input validation
        w, h = window_size
        if w <= 0 or h <= 0:
            raise ValueError(
                f"window_size must be positive, got width={w}, height={h}"
            )
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")

        cx, cy, cz = window_center

        x_min = cx - 0.5 * w - margin
        x_max = cx + 0.5 * w + margin
        y_min = cy - 0.5 * h - margin
        y_max = cy + 0.5 * h + margin
        z_min = cz - margin
        z_max = cz + margin

        return (x_min, x_max, y_min, y_max, z_min, z_max)

    def render_crop(
            self,
            volume_activated: np.ndarray,
            temperature_field: np.ndarray,
            window_center: ArrayLike3,
            window_size: Tuple[float, float],
            ambient: float = 300.0,
            roi_margin: float = 0.001
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Efficiently render and crop in one call (10-100x faster for small crops).

        Combines ROI pre-filtering with cropping for maximum performance.
        Gap filling (if enabled) is applied automatically.

        Parameters
        ----------
        volume_activated, temperature_field : Arrays
        window_center : ArrayLike3
            Center of crop window in world coords.
        window_size : Tuple[float, float]
            Crop size (width, height) in meters.
        ambient : float
        roi_margin : float

        Returns
        -------
        img : np.ndarray
            Cropped image (gap-filled if fill_gaps=True)
        extent : Tuple[float, float, float, float]
            Image extent (xmin, xmax, ymin, ymax)
        """
        roi = self.calculate_roi_for_crop(window_center, window_size, margin=roi_margin)
        img, extent = self.render_first_hit(volume_activated, temperature_field, ambient=ambient, roi_world=roi)
        return self.apply_crop_window(img, extent, window_center=window_center, window_size=window_size)

    def _world_to_sensor_plane(self, world_pos: ArrayLike3) -> Tuple[float, float]:
        """
        Transform world position to sensor plane coordinates.

        Projects world point onto camera's sensor plane and returns (u, v) coordinates
        in meters from the sensor center.

        Parameters
        ----------
        world_pos : ArrayLike3
            3D position in world coordinates (x, y, z)

        Returns
        -------
        u : float
            Horizontal position on sensor plane (meters from center, +right)
        v : float
            Vertical position on sensor plane (meters from center, +up)

        Raises
        ------
        ValueError
            If point is not visible to camera (behind camera or outside frustum)

        Notes
        -----
        This transformation is used internally to convert world-space crop coordinates
        to sensor-space coordinates for apply_crop_window().

        Example
        -------
        >>> cam = OrthographicCamera(plane_size=(0.01, 0.01))
        >>> cam.set_position([0, 0, 0.1])
        >>> cam.look_at([0, 0, 0])
        >>> # World point at (0.002, -0.003, 0) projects to sensor at (0.002, -0.003)
        >>> u, v = cam._world_to_sensor_plane([0.002, -0.003, 0])
        """
        pos_world = np.asarray(world_pos, float)

        # Transform to camera space
        R_wc = self._world_to_cam_rotation()
        pos_cam = (R_wc @ (pos_world - self.pos).T).T

        # Project to sensor plane (subclass-specific: orthographic or perspective)
        # Reshape to (1, 3) for batch processing, then extract scalar
        u_arr, v_arr, zc_arr, valid_arr = self._project_cam_to_plane(
            pos_cam.reshape(1, 3)
        )

        if not valid_arr[0]:
            raise ValueError(
                f"Point {world_pos} not visible to camera "
                f"(behind camera or outside frustum)"
            )

        return float(u_arr[0]), float(v_arr[0])

    def _crop_sensor_coords(
            self,
            img: np.ndarray,
            extent: Tuple[float, float, float, float],
            center_u: float,
            center_v: float,
            width_m: float,
            height_m: float
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Internal method: Crop in sensor plane coordinates.

        Parameters
        ----------
        img : np.ndarray
            Rendered image
        extent : Tuple[float, float, float, float]
            Extent in sensor coords (umin, umax, vmin, vmax)
        center_u, center_v : float
            Crop center in sensor plane coordinates (meters from camera center)
        width_m, height_m : float
            Crop size in sensor plane (meters)

        Returns
        -------
        cropped_img : np.ndarray
            Cropped image at native resolution
        cropped_extent : Tuple[float, float, float, float]
            New extent of cropped region in sensor coordinates

        Notes
        -----
        This is the low-level cropping implementation that operates entirely in
        sensor plane coordinates. Public methods should use apply_crop_window()
        which handles world-to-sensor transformation.
        """
        # Input validation
        if width_m <= 0 or height_m <= 0:
            raise ValueError(
                f"Crop size must be positive, got width={width_m}, height={height_m}"
            )

        umin, umax, vmin, vmax = extent
        W, H = img.shape

        # Validate extent
        if umax <= umin or vmax <= vmin:
            raise ValueError(
                f"Invalid extent: umax ({umax}) <= umin ({umin}) or "
                f"vmax ({vmax}) <= vmin ({vmin})"
            )

        # Calculate crop window bounds in sensor coordinates
        u0 = center_u - 0.5 * width_m
        u1 = center_u + 0.5 * width_m
        v0 = center_v - 0.5 * height_m
        v1 = center_v + 0.5 * height_m

        # Clamp to extent
        u0_clamped = max(u0, umin)
        u1_clamped = min(u1, umax)
        v0_clamped = max(v0, vmin)
        v1_clamped = min(v1, vmax)

        # Check if valid crop (window overlaps with image)
        if u1_clamped <= u0_clamped or v1_clamped <= v0_clamped:
            # Crop window completely outside image bounds
            import warnings
            warnings.warn(
                f"Crop window (u=[{u0:.6f}, {u1:.6f}], v=[{v0:.6f}, {v1:.6f}]) "
                f"is outside image extent (u=[{umin:.6f}, {umax:.6f}], "
                f"v=[{vmin:.6f}, {vmax:.6f}]). Returning empty image.",
                UserWarning
            )
            # Return empty image with requested extent (clamped)
            return np.array([[]]).reshape(0, 0), (u0_clamped, u1_clamped, v0_clamped, v1_clamped)

        # Convert sensor coords to pixel indices
        # Use floor to be consistent with rendering binning (see line 435)
        # Pixel i covers range [i*pixel_size, (i+1)*pixel_size)
        iu0 = max(0, int(np.floor((u0_clamped - umin) / (umax - umin) * W)))
        iu1 = min(W, int(np.ceil((u1_clamped - umin) / (umax - umin) * W)))
        iv0 = max(0, int(np.floor((v0_clamped - vmin) / (vmax - vmin) * H)))
        iv1 = min(H, int(np.ceil((v1_clamped - vmin) / (vmax - vmin) * H)))

        # Ensure at least one pixel
        if iu1 <= iu0 or iv1 <= iv0:
            import warnings
            warnings.warn(
                "Crop window smaller than one pixel. Returning empty image.",
                UserWarning
            )
            return np.array([[]]).reshape(0, 0), (u0_clamped, u1_clamped, v0_clamped, v1_clamped)

        # Crop image
        cropped = img[iu0:iu1, iv0:iv1]

        # Calculate actual extent of cropped region (in sensor coords)
        # Convert pixel indices back to sensor coordinates
        actual_u0 = umin + (iu0 / W) * (umax - umin)
        actual_u1 = umin + (iu1 / W) * (umax - umin)
        actual_v0 = vmin + (iv0 / H) * (vmax - vmin)
        actual_v1 = vmin + (iv1 / H) * (vmax - vmin)

        return cropped, (actual_u0, actual_u1, actual_v0, actual_v1)

    def apply_crop_window(
            self,
            img: np.ndarray,
            extent: Tuple[float, float, float, float],
            window_center: ArrayLike3,
            window_size: Tuple[float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Crop rendered image to a physical window in world coordinates.

        This method transforms the world-space window center to sensor plane coordinates,
        then performs the crop. This is pure post-processing applied after rendering.

        Parameters
        ----------
        img : np.ndarray
            Rendered image from render_first_hit() with shape (W, H)
        extent : Tuple[float, float, float, float]
            Physical extent of img in SENSOR PLANE coordinates:
            (umin, umax, vmin, vmax) in meters from camera center
            This is the extent returned by render_first_hit()
        window_center : ArrayLike3
            Center of crop window in WORLD coordinates (x, y, z)
            The Z coordinate is used for projection but only X,Y affect the 2D crop
        window_size : Tuple[float, float]
            Physical size of crop window (width, height) in meters
            For orthographic: size at all distances
            For perspective: size at the distance of window_center

        Returns
        -------
        cropped_img : np.ndarray
            Cropped image at native resolution with shape (W', H')
        cropped_extent : Tuple[float, float, float, float]
            New physical extent of cropped image in sensor plane coordinates

        Raises
        ------
        ValueError
            If window_size contains non-positive values
            If window_center is not visible to camera
            If extent is malformed

        Notes
        -----
        **Coordinate System Transformation:**

        This method internally transforms window_center from world coordinates to
        sensor plane coordinates using the camera's projection. The extent is already
        in sensor plane coordinates (as returned by render_first_hit()).

        For orthographic cameras, the transformation is straightforward:
        - World point (x, y, z) projects to sensor (u, v) where u, v are in meters
          from the camera center

        For perspective cameras, the transformation accounts for distance-dependent
        scaling:
        - Objects closer to the camera project larger on the sensor
        - Objects farther project smaller
        - The window_size represents the physical size at the depth of window_center

        Example
        -------
        >>> cam = OrthographicCamera(plane_size=(0.01, 0.01))
        >>> cam.set_position([0, -0.12, 0.04])
        >>> cam.look_at([0, 0, 0])
        >>> img, extent = cam.render_first_hit(activated, temperature)
        >>> # Crop to 2mm × 2mm window centered at nozzle in world space
        >>> cropped, new_extent = cam.apply_crop_window(
        ...     img, extent,
        ...     window_center=(0.05, 0.05, 0.001),  # World coordinates!
        ...     window_size=(0.002, 0.002)
        ... )
        >>> # cropped now contains the 2mm region around (0.05, 0.05, 0.001)
        """
        # Input validation
        width_m, height_m = window_size
        if width_m <= 0 or height_m <= 0:
            raise ValueError(
                f"window_size must be positive, got width={width_m}, height={height_m}"
            )

        # Transform world center to sensor plane coordinates
        try:
            center_u, center_v = self._world_to_sensor_plane(window_center)
        except ValueError as e:
            # Point not visible - return empty crop with warning
            import warnings
            warnings.warn(
                f"Window center {window_center} not visible to camera: {e}. "
                f"Returning empty crop.",
                UserWarning
            )
            return np.array([[]]).reshape(0, 0), extent

        # For perspective cameras, we may need to scale window_size based on distance
        # This is handled by subclasses via overrides if needed, or the user
        # can call get_crop_size_for_window_at_distance() explicitly
        # For orthographic cameras, window_size maps 1:1 to sensor plane

        # Perform crop in sensor coordinates
        return self._crop_sensor_coords(
            img, extent, center_u, center_v, width_m, height_m
        )


# -----------------------------------------------------------------------------
# Following mixin (can be combined with either camera variant)
# -----------------------------------------------------------------------------

class FollowingCameraMixin:
    """
    Mixin providing **following behavior** with clean separation between configuration and execution.

    **NEW in v2.1:** Redesigned API with separate configuration and update methods!

    Design Philosophy
    -----------------
    - **Configure once:** Define HOW the camera should follow (cartesian or spherical)
    - **Update repeatedly:** Just pass new target position, camera follows automatically

    Following Modes
    ---------------
    1. **Cartesian:** Camera maintains fixed offset in local coordinates relative to motion
    2. **Spherical:** Camera maintains fixed angle and distance from target

    Workflow
    --------
    1. During init: Call configure_cartesian_following() OR configure_spherical_following()
    2. During runtime: Call update_target(new_pos, motion_dir) every frame
    3. That's it! Camera follows automatically according to stored configuration

    Example (Cartesian)
    -------------------
    >>> cam = FollowingOrthographicCamera(plane_size=(0.01, 0.01))
    >>> # Configure: camera stays 12cm behind, 4cm above (local coords)
    >>> cam.configure_cartesian_following(
    ...     offset_local=(0, -0.12, 0.04),
    ...     tilt_angle_deg=0
    ... )
    >>> # Runtime: just update target position
    >>> for pos in scan_path:
    ...     motion = pos - prev_pos
    ...     cam.update_target(pos, motion_direction=motion)
    ...     img, extent = cam.render_first_hit(...)

    Example (Spherical)
    -------------------
    >>> cam = FollowingPerspectiveCamera(fov_y_deg=45)
    >>> # Configure: camera follows from behind at 20° elevation, 12cm away
    >>> cam.configure_spherical_following(
    ...     azimuth_offset_deg=180,  # Behind motion
    ...     elevation_deg=20,
    ...     distance=0.12
    ... )
    >>> # Runtime: just update target (azimuth computed from motion automatically)
    >>> for pos in scan_path:
    ...     motion = pos - prev_pos
    ...     cam.update_target(pos, motion_direction=motion)
    ...     img, extent = cam.render_first_hit(...)
    """

    def configure_cartesian_following(self,
                                      offset: ArrayLike3 = (0.0, -0.12, 0.04),
                                      up_hint: ArrayLike3 = (0.0, 0.0, 1.0)) -> None:
        """
        Configure camera to follow using fixed cartesian offset in WORLD coordinates.

        With this mode, camera maintains a FIXED offset in world space:
        - Camera position is always: target_pos + offset (in world coordinates)
        - Camera does NOT rotate with motion direction
        - Simple fixed offset relationship

        Parameters
        ----------
        offset : ArrayLike3
            Camera offset from target in WORLD coordinates (meters):
            - x: offset in world X direction
            - y: offset in world Y direction
            - z: offset in world Z direction
            Default: (0.0, -0.12, 0.04) = 12cm in -Y, 4cm in +Z
        up_hint : ArrayLike3
            World-space up vector for camera orientation (typically +Z)
            Default: (0.0, 0.0, 1.0)

        Examples
        --------
        >>> # Camera 12cm in -Y direction, 4cm in +Z direction
        >>> cam.configure_cartesian_following(
        ...     offset=(0, -0.12, 0.04)
        ... )

        >>> # Camera 10cm in +X direction, 15cm in +Z direction
        >>> cam.configure_cartesian_following(
        ...     offset=(0.10, 0, 0.15)
        ... )

        Notes
        -----
        - After configuration, use update_target() to follow the target
        - The offset is in WORLD coordinates and does NOT change with motion direction
        - For rotating offset based on motion, use configure_spherical_following()
        """
        self._follow_mode = 'cartesian'
        self._follow_config = {
            'offset': np.asarray(offset, float),
            'up_hint': np.asarray(up_hint, float)
        }

    def configure_spherical_following(self,
                                      azimuth_offset_deg: float = 180.0,
                                      elevation_deg: float = 20.0,
                                      distance: float = 0.12,
                                      up_hint: ArrayLike3 = (0.0, 0.0, 1.0)) -> None:
        """
        Configure camera to follow using spherical coordinates.

        With this mode, camera maintains fixed angle and distance from target:
        - Azimuth is relative to motion direction (0° = motion direction, 180° = behind)
        - Elevation is angle above horizontal plane
        - Distance is constant radius from target

        Parameters
        ----------
        azimuth_offset_deg : float
            Angle offset from motion direction in degrees (default: 180° = behind)
            - 0° = same direction as motion (ahead)
            - 90° = 90° clockwise from motion (right side)
            - 180° = opposite to motion (behind)
            - 270° = 90° counter-clockwise from motion (left side)
        elevation_deg : float
            Elevation angle above horizontal plane in degrees (default: 20°)
            - 0° = horizontal (same height as target)
            - +90° = directly above
            - -90° = directly below
        distance : float
            Distance from target to camera in meters (default: 0.12m = 12cm)
            Must be positive
        up_hint : ArrayLike3
            World-space up vector for orientation (typically +Z)

        Raises
        ------
        ValueError
            If distance is not positive

        Examples
        --------
        >>> # Camera behind target, 20° above, 12cm away
        >>> cam.configure_spherical_following(
        ...     azimuth_offset_deg=180,  # Behind
        ...     elevation_deg=20,
        ...     distance=0.12
        ... )

        >>> # Camera to the right, horizontal, 15cm away
        >>> cam.configure_spherical_following(
        ...     azimuth_offset_deg=90,   # Right
        ...     elevation_deg=0,
        ...     distance=0.15
        ... )

        >>> # Camera directly above, 10cm away
        >>> cam.configure_spherical_following(
        ...     azimuth_offset_deg=0,    # Doesn't matter when elevation=90
        ...     elevation_deg=90,
        ...     distance=0.10
        ... )

        Notes
        -----
        After configuration, use update_target() to follow the target.
        Motion direction is used to compute current azimuth angle.
        """
        if distance <= 0:
            raise ValueError(f"distance must be positive, got {distance}")

        self._follow_mode = 'spherical'
        self._follow_config = {
            'azimuth_offset_deg': float(azimuth_offset_deg),
            'elevation_deg': float(elevation_deg),
            'distance': float(distance),
            'up_hint': np.asarray(up_hint, float)
        }

    def update_target(self,
                      target_pos: ArrayLike3,
                      motion_direction: Optional[ArrayLike3] = None,
                      *,
                      orient: bool = True) -> None:
        """
        Update camera position to follow the target (unified method).

        This method uses the stored configuration from configure_cartesian_following()
        or configure_spherical_following() to update camera position.

        **NEW in v2.1:** Single update method replaces follow_target() and follow_target_spherical()!

        Parameters
        ----------
        target_pos : ArrayLike3
            New position of the tracked point in world coordinates
        motion_direction : Optional[ArrayLike3]
            Direction of motion in world coordinates.
            - For cartesian mode: NOT USED (offset is fixed in world space)
            - For spherical mode: Used to compute azimuth angle (camera rotates with motion)
            - If None in spherical mode: Uses absolute azimuth instead of relative
        orient : bool
            If True, camera looks at target after positioning (default: True)
            If False, only translates camera, keeping orientation

        Raises
        ------
        RuntimeError
            If following not configured (call configure_*_following() first)

        Examples
        --------
        >>> # Cartesian mode: fixed world-space offset
        >>> cam.configure_cartesian_following(offset=(0, -0.12, 0.04))
        >>> for pos in scan_path:
        ...     cam.update_target(pos)  # motion_direction not needed
        ...     img, extent = cam.render_first_hit(...)

        >>> # Spherical mode: rotating offset based on motion
        >>> cam.configure_spherical_following(azimuth_offset_deg=180, distance=0.12)
        >>> for pos in scan_path:
        ...     motion = pos - prev_pos
        ...     cam.update_target(pos, motion_direction=motion)  # motion_direction needed!
        ...     img, extent = cam.render_first_hit(...)

        Notes
        -----
        The behavior depends on which configuration method was called:
        - configure_cartesian_following(): Fixed world-space offset (does NOT rotate)
        - configure_spherical_following(): Rotating offset based on motion direction
        """
        if not hasattr(self, '_follow_mode') or self._follow_mode is None:
            raise RuntimeError(
                "Following not configured. Call configure_cartesian_following() "
                "or configure_spherical_following() first."
            )

        target = np.asarray(target_pos, float)

        if self._follow_mode == 'cartesian':
            self._update_cartesian(target, motion_direction, orient=orient)
        elif self._follow_mode == 'spherical':
            self._update_spherical(target, motion_direction, orient=orient)
        else:
            raise RuntimeError(f"Unknown follow mode: {self._follow_mode}")

    def _update_cartesian(self,
                          target: np.ndarray,
                          motion_direction: Optional[np.ndarray],
                          orient: bool) -> None:
        """
        Internal: Update using cartesian offset configuration.

        Applies a FIXED offset in world coordinates. The offset does NOT
        rotate or transform based on motion direction.
        """
        config = self._follow_config
        offset = config['offset']
        up_hint = config['up_hint']

        # Simple fixed offset in world coordinates
        self.pos = target + offset

        if orient:
            self.look_at(target, up_hint=up_hint)

    def _update_spherical(self,
                          target: np.ndarray,
                          motion_direction: Optional[np.ndarray],
                          orient: bool) -> None:
        """Internal: Update using spherical configuration."""
        config = self._follow_config
        azimuth_offset_deg = config['azimuth_offset_deg']
        elevation_deg = config['elevation_deg']
        distance = config['distance']
        up_hint = config['up_hint']

        # Compute azimuth from motion direction
        if motion_direction is not None:
            motion_dir = np.asarray(motion_direction, float)
            motion_magnitude = np.linalg.norm(motion_dir[:2])  # XY plane only

            if motion_magnitude > 1e-10:
                # Compute motion angle in XY plane
                motion_azimuth_rad = np.arctan2(motion_dir[1], motion_dir[0])
                motion_azimuth_deg = np.rad2deg(motion_azimuth_rad)

                # Add offset
                total_azimuth_deg = motion_azimuth_deg + azimuth_offset_deg
            else:
                # Stationary: just use offset directly
                total_azimuth_deg = azimuth_offset_deg
        else:
            # No motion direction: use offset as absolute azimuth
            total_azimuth_deg = azimuth_offset_deg

        # Convert to radians
        az_rad = np.deg2rad(total_azimuth_deg)
        el_rad = np.deg2rad(elevation_deg)

        # Spherical to cartesian
        cos_el = np.cos(el_rad)
        offset_x = distance * cos_el * np.cos(az_rad)
        offset_y = distance * cos_el * np.sin(az_rad)
        offset_z = distance * np.sin(el_rad)

        self.pos = target + np.array([offset_x, offset_y, offset_z], float)
        if orient:
            self.look_at(target, up_hint=up_hint)