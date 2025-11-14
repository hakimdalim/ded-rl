import numpy as np
from typing import Tuple, Optional

from camera._base_camera import FollowingCameraMixin, ArrayLike3, BaseCamera


# -----------------------------------------------------------------------------
# Perspective camera (projection = converging rays, FOV-driven)
# -----------------------------------------------------------------------------

class PerspectiveCamera(BaseCamera):
    """
    **Perspective camera** with converging rays and configurable field-of-view.

    Key Concepts
    ------------
    - **Sensor Plane**: Physical dimensions at camera position (plane_size)
    - **Focal Length**: Distance from camera to virtual image plane
      Computed as: f = plane_size_height / (2 * tan(fov_y/2))
    - **Field of View (FOV) at Distance**: The actual physical area visible at distance d
      Computed as: fov_height_at_d = d * 2 * tan(fov_y/2)
      Note: FOV at distance is INDEPENDENT of plane_size (sensor size)!

    Zoom Control
    ------------
    Two equivalent ways to control zoom:

    1. **set_fov_y(angle)**: Set vertical FOV angle directly
       - Larger angle = wider view at all distances
       - This determines fov_height_at_d = d * 2 * tan(angle/2)

    2. **set_fov_at_distance(width, height, distance)**: Set physical FOV at target
       - Calculates required FOV angle to see (width × height) area at distance
       - More intuitive for zoom control: "I want to see 2mm × 2mm at 10cm away"
       - Adjusts fov_y to achieve the desired FOV at that specific distance

    The sensor plane_size affects resolution/sampling but NOT the field of view.

    Parameters
    ----------
    fov_y_deg : float
        Vertical field-of-view angle in degrees (default: 45°)
        Combined with distance gives: fov_height = distance * 2 * tan(fov_y/2)
    """

    def __init__(self, fov_y_deg: float = 45.0, **kwargs) -> None:
        if not (0 < fov_y_deg < 180):
            raise ValueError(f"fov_y_deg must be in range (0, 180), got {fov_y_deg}")
        super().__init__(**kwargs)
        self.fov_y_deg = float(fov_y_deg)
        self._update_focal_length()

    # ----- internals -----
    def _update_focal_length(self) -> None:
        """
        Recompute focal length from sensor plane_size and fov_y_deg.

        Note: focal_length = plane_size_height / (2 * tan(fov_y/2))
        This ensures that at focal_length distance, the visible height equals plane_size_height.
        """
        fov_rad = np.deg2rad(self.fov_y_deg)
        half_h = 0.5 * self.plane_size[1]
        self._focal_length = half_h / np.tan(0.5 * fov_rad)  # meters

    # ----- public API -----
    def set_fov_y(self, fov_y_deg: float) -> None:
        """
        Set the vertical field-of-view angle (degrees).

        This determines how much you see at any distance:
        fov_height_at_distance = distance * 2 * tan(fov_y/2)

        Parameters
        ----------
        fov_y_deg : float
            Vertical FOV angle in degrees (must be in range (0, 180)). Larger = wider view.

        Raises
        ------
        ValueError
            If fov_y_deg is not in valid range (0, 180)
        """
        if not (0 < fov_y_deg < 180):
            raise ValueError(f"fov_y_deg must be in range (0, 180), got {fov_y_deg}")
        self.fov_y_deg = float(fov_y_deg)
        self._update_focal_length()

    def set_plane_size(self, width: float, height: float) -> None:
        """
        Set the sensor plane size (meters).

        Note: For perspective cameras, this affects pixel density but NOT the field
        of view at the target. Use set_fov_y() or set_fov_at_distance() to control zoom.
        """
        super().set_plane_size(width, height)
        self._update_focal_length()

    def get_fov_at_distance(self, distance: float) -> Tuple[float, float]:
        """
        Calculate the physical field of view (width, height) in meters at a given distance.

        Parameters
        ----------
        distance : float
            Distance from camera in meters.

        Returns
        -------
        fov_width : float
            Physical width visible at that distance (meters)
        fov_height : float
            Physical height visible at that distance (meters)

        Example
        -------
        >>> cam = PerspectiveCamera(fov_y_deg=45.0)
        >>> cam.set_position([0, -0.12, 0.04])
        >>> cam.look_at([0, 0, 0])
        >>> dist = 0.126  # distance to target
        >>> w, h = cam.get_fov_at_distance(dist)
        >>> print(f"At {dist}m, camera sees {w*1000:.2f}mm × {h*1000:.2f}mm")
        """
        fov_rad = np.deg2rad(self.fov_y_deg)
        fov_height = distance * 2.0 * np.tan(0.5 * fov_rad)
        aspect_ratio = self.plane_size[0] / self.plane_size[1]
        fov_width = fov_height * aspect_ratio
        return fov_width, fov_height

    def get_crop_size_for_window_at_distance(
            self,
            window_size: Tuple[float, float],
            distance: float
    ) -> Tuple[float, float]:
        """
        Calculate the crop size at sensor plane needed to show a specific window at distance.

        For perspective cameras, the visible area grows with distance. If you want to crop
        to a fixed physical size at the focal plane, the crop at the sensor must be smaller.

        Relationship: crop_at_sensor = window_at_distance * (focal_length / distance)

        Parameters
        ----------
        window_size : Tuple[float, float]
            Desired window (width, height) at the target distance in meters
        distance : float
            Distance from camera to target plane in meters

        Returns
        -------
        crop_width : float
            Required crop width at sensor plane (meters)
        crop_height : float
            Required crop height at sensor plane (meters)

        Example
        -------
        >>> cam = PerspectiveCamera(fov_y_deg=45.0, plane_size=(0.01, 0.01))
        >>> cam.set_position([0, -0.12, 0.04])
        >>> cam.look_at([0, 0, 0])
        >>> distance = np.linalg.norm(cam.pos - cam.target)
        >>> # Want to see 2mm × 2mm at target
        >>> crop_w, crop_h = cam.get_crop_size_for_window_at_distance((0.002, 0.002), distance)
        >>> # Then crop rendered image to (crop_w, crop_h)
        """
        # focal_length = plane_size_height / (2 * tan(fov_y/2))
        # At sensor: height_sensor / focal_length = height_distance / distance
        # Therefore: height_sensor = height_distance * (focal_length / distance)

        window_w, window_h = window_size
        scale = self._focal_length / distance

        crop_w = window_w * scale
        crop_h = window_h * scale

        return (crop_w, crop_h)

    def calculate_roi_for_crop(
            self,
            window_center: ArrayLike3,
            window_size: Tuple[float, float],
            margin: float = 0.001
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate ROI for perspective camera accounting for frustum expansion.

        For perspective cameras, objects at different distances project to different
        sizes on the sensor. The ROI must be large enough to capture voxels that
        project into the crop window, which forms a pyramid frustum in 3D space.

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

        # Calculate distance from camera to window center
        distance = np.linalg.norm(np.array([cx, cy, cz]) - self.pos)

        # For perspective: expand ROI to account for frustum
        # Objects closer to camera project larger, objects further project smaller
        # Use 2x expansion to safely capture the frustum
        expansion_factor = 2.0

        effective_w = w * expansion_factor
        effective_h = h * expansion_factor
        effective_margin = margin * expansion_factor

        x_min = cx - 0.5 * effective_w - effective_margin
        x_max = cx + 0.5 * effective_w + effective_margin
        y_min = cy - 0.5 * effective_h - effective_margin
        y_max = cy + 0.5 * effective_h + effective_margin
        z_min = cz - effective_margin * 2  # More depth margin for frustum
        z_max = cz + effective_margin * 2

        return (x_min, x_max, y_min, y_max, z_min, z_max)

    def apply_crop_window_at_distance(
            self,
            img: np.ndarray,
            extent: Tuple[float, float, float, float],
            window_size_at_distance: Tuple[float, float],
            distance: float,
            window_center: Optional[ArrayLike3] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Crop to show a specific physical window size at a given distance.

        Automatically calculates correct sensor-plane crop for perspective projection.

        Parameters
        ----------
        img, extent : Rendered image and extent from render_first_hit()
        window_size_at_distance : (width, height) desired at focal distance (meters)
        distance : Distance from camera to focal plane (meters)
        window_center : Optional[ArrayLike3]
            Center of crop window in world coordinates. If None, uses camera target.

        Returns
        -------
        cropped_img, cropped_extent : Cropped image at native resolution and new extent

        Example
        -------
        >>> cam = PerspectiveCamera(fov_y_deg=45.0)
        >>> cam.set_position([0, -0.12, 0.04])
        >>> cam.look_at([0, 0, 0])
        >>> dist = np.linalg.norm(cam.pos - cam.target)
        >>> img, extent = cam.render_first_hit(activated, temperature)
        >>> # Crop to 2mm × 2mm at target distance
        >>> cropped, new_extent = cam.apply_crop_window_at_distance(
        ...     img, extent,
        ...     window_size_at_distance=(0.002, 0.002),
        ...     distance=dist,
        ...     window_center=cam.target  # Or any other world position
        ... )
        """
        # Use camera target if window_center not specified
        if window_center is None:
            window_center = self.target

        crop_size_at_sensor = self.get_crop_size_for_window_at_distance(
            window_size_at_distance, distance
        )

        return self.apply_crop_window(
            img, extent,
            window_center=window_center,
            window_size=crop_size_at_sensor
        )

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

        For perspective cameras, window_size represents the PHYSICAL SIZE in world
        coordinates at the window_center location. This method automatically scales
        to the correct sensor-plane size based on distance and focal length.

        Parameters
        ----------
        volume_activated, temperature_field : Arrays
        window_center : ArrayLike3
            Center of crop window in world coords.
        window_size : Tuple[float, float]
            Crop size (width, height) in meters IN WORLD SPACE at window_center.
        ambient : float
        roi_margin : float

        Returns
        -------
        img, extent : Cropped image at native resolution and extent

        Notes
        -----
        Unlike the base class, this correctly handles perspective projection by
        scaling window_size from world space to sensor plane space based on
        distance and focal length.

        The scaling formula is:
            sensor_size = world_size * (focal_length / distance)

        This ensures that regardless of FOV, distance, or focal length, the
        crop window captures the requested physical size at the target location.

        Example
        -------
        >>> cam = PerspectiveCamera(fov_y_deg=45)
        >>> cam.set_position([0, -0.12, 0.04])
        >>> cam.look_at([0, 0, 0])
        >>> # Crop to 2mm × 2mm in WORLD SPACE at target
        >>> img, extent = cam.render_crop(
        ...     activated, temperature,
        ...     window_center=[0, 0, 0],
        ...     window_size=(0.002, 0.002),  # World-space size!
        ...     ambient=300.0
        ... )
        >>> # Result captures exactly 2mm × 2mm regardless of FOV or distance
        """
        # Calculate ROI and render (same as base class)
        roi = self.calculate_roi_for_crop(window_center, window_size, margin=roi_margin)
        img, extent = self.render_first_hit(
            volume_activated, temperature_field,
            ambient=ambient, roi_world=roi
        )

        # For perspective: convert world-space window_size to sensor-plane size
        # Calculate distance from camera to window center
        window_center_arr = np.asarray(window_center, float)
        distance = np.linalg.norm(window_center_arr - self.pos)

        # Scale window size from world space to sensor plane
        # sensor_size = world_size * (focal_length / distance)
        scale_factor = self._focal_length / distance
        sensor_window_size = (
            window_size[0] * scale_factor,
            window_size[1] * scale_factor
        )

        # Apply crop using sensor-plane size
        return self.apply_crop_window(
            img, extent,
            window_center=window_center,
            window_size=sensor_window_size  # Now in sensor coordinates!
        )

    def set_fov_at_distance(self,
                            fov_width: float,
                            fov_height: float,
                            distance: float,
                            update_plane_size: bool = True) -> None:
        """
        Set the field of view to capture a specific physical area at a given distance.

        This is the most intuitive zoom control: "I want to see a 2mm × 2mm area
        at 10cm distance from the camera."

        Parameters
        ----------
        fov_width : float
            Desired visible width at distance (meters)
        fov_height : float
            Desired visible height at distance (meters)
        distance : float
            Distance from camera to the target plane (meters)
        update_plane_size : bool
            If True, also updates sensor plane_size to maintain aspect ratio.
            If False, only updates fov_y (may change aspect ratio).

        Example
        -------
        >>> cam = PerspectiveCamera()
        >>> # Position camera 12cm behind and 4cm above target
        >>> cam.set_position([0, -0.12, 0.04])
        >>> cam.look_at([0, 0, 0])
        >>> # Calculate distance to target
        >>> dist = np.linalg.norm(cam.pos - cam.target)
        >>> # Set FOV to see 2mm × 2mm area at target
        >>> cam.set_fov_at_distance(0.002, 0.002, dist)
        >>> # Now camera will capture exactly 2mm × 2mm at the target point

        Notes
        -----
        The required FOV angle is calculated as:
        fov_y_required = 2 * arctan(fov_height / (2 * distance))
        """
        # Calculate required vertical FOV angle
        fov_y_required_rad = 2.0 * np.arctan(fov_height / (2.0 * distance))
        self.fov_y_deg = float(np.rad2deg(fov_y_required_rad))

        # Update plane_size to match desired aspect ratio if requested
        if update_plane_size:
            aspect_ratio = fov_width / fov_height
            # Keep a reasonable sensor size (arbitrary, doesn't affect FOV)
            plane_h = 0.01  # 1cm sensor height
            plane_w = plane_h * aspect_ratio
            self.plane_size = (plane_w, plane_h)
            self._derive_resolution_from_plane()

        self._update_focal_length()

    # ----- projection hook implementation -----
    def _project_cam_to_plane(self, pos_cam: np.ndarray):
        """
        Perspective projection: (u, v) = (f * x / d, f * y / d)

        In camera coordinates:
        - Z is negative for points in front of the camera
        - Use d = -z as the forward distance
        - Projection: sensor coordinates = focal_length * (x/d, y/d)

        Returns
        -------
        u, v : Sensor plane coordinates (meters)
        zc : Depth for sorting (less negative = closer)
        valid : Boolean mask (only points with d > 0 are valid)
        """
        x = pos_cam[:, 0]
        y = pos_cam[:, 1]
        z = pos_cam[:, 2]
        d = -z  # points in front → d > 0
        eps = 1e-8
        valid = d > eps
        # Avoid division by zero warnings
        d_safe = np.where(valid, d, 1.0)
        u = (self._focal_length * x) / d_safe
        v = (self._focal_length * y) / d_safe
        # Camera-space depth: Z is negative for points in front of camera
        # Less negative Z = closer to camera (e.g., z=-1 is closer than z=-2)
        # The base renderer negates this (using -zc) for near-to-far sorting
        zc = z
        return u, v, zc, valid


class FollowingPerspectiveCamera(PerspectiveCamera, FollowingCameraMixin):
    """
    Perspective camera that **follows a tracked point** with fixed relative offset
    and tilt angle.

    The camera maintains a constant offset from the tracked point and always looks
    back at it. Perfect for following a heat source, nozzle, or moving object.

    Parameters
    ----------
    source_pos : ArrayLike3
        Initial position of the tracked point
    offset : ArrayLike3
        Camera offset from target in WORLD coordinates (x, y, z):
        - x: offset in world X direction
        - y: offset in world Y direction
        - z: offset in world Z direction
        Default: (0.0, -0.12, 0.04) = 12cm in -Y, 4cm in +Z
        Note: Offset is FIXED in world space and does NOT rotate with motion
    up_hint : ArrayLike3
        World-space up vector for camera orientation (typically +Z)
        Default: (0, 0, 1)
    fov_y_deg : float
        Vertical field-of-view angle (degrees).
        Default: 45°

    Example
    -------
    >>> # Create camera following a heat source
    >>> # Camera will be at fixed offset: -12cm in Y, +4cm in Z
    >>> cam = FollowingPerspectiveCamera(
    ...     source_pos=(0.004, 0.003, 0.001),  # heat source position
    ...     offset=(0.0, -0.12, 0.04),  # Fixed world-space offset
    ...     fov_y_deg=45.0
    ... )
    >>> # Calculate distance to source for zoom control
    >>> dist = np.linalg.norm([0.0, -0.12, 0.04])
    >>> # Set FOV to see 3mm × 3mm at the source
    >>> cam.set_fov_at_distance(0.003, 0.003, dist)
    >>> # Update source position as it moves
    >>> cam.update_target((0.004, 0.004, 0.001))
    """

    def __init__(self,
                 source_pos: ArrayLike3 = (0.0, 0.0, 0.0),
                 offset: ArrayLike3 = (0.0, -0.12, 0.04),
                 up_hint: ArrayLike3 = (0.0, 0.0, 1.0),
                 fov_y_deg: float = 45.0,
                 **camera_kwargs) -> None:
        super().__init__(fov_y_deg=fov_y_deg, **camera_kwargs)
        # Configure cartesian following
        self.configure_cartesian_following(
            offset=offset,
            up_hint=up_hint
        )
        # Position at initial target
        self.update_target(source_pos, motion_direction=None)