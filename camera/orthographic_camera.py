# -----------------------------------------------------------------------------
# Orthographic camera (projection = parallel rays)
# -----------------------------------------------------------------------------
import numpy as np

from camera._base_camera import BaseCamera, FollowingCameraMixin, ArrayLike3


class OrthographicCamera(BaseCamera):
    """
    **Orthographic camera** with parallel ray projection.

    Key Properties
    --------------
    - **Parallel Rays**: All rays are parallel to the forward direction
    - **No Perspective Distortion**: Objects same size regardless of distance
    - **Field of View = Sensor Size**: The plane_size directly determines visible area
      at ALL distances (unlike perspective cameras)
    - **Sensor Plane**: Anchored at camera.pos with normal = forward

    Use Cases
    ---------
    - Technical drawings and measurements
    - When you need consistent scale across depth
    - Viewing large scenes without perspective convergence

    Definition (meters)
    -------------------
    - Pose: ``pos`` (world center of sensor plane) and orthonormal basis
      ``(right, up, forward)`` with the camera **looking along** ``forward``
    - Field of view: Constant at all distances, equal to ``plane_size = (width, height)``
    - Sampling: Either fixed pixel size (meters/pixel) or fixed resolution (pixels)

    Rendering
    ---------
    Casts **parallel rays** through the sensor plane and bins first-hit temperatures.

    Example
    -------
    >>> cam = OrthographicCamera(plane_size=(0.01, 0.01))  # 10mm × 10mm view
    >>> cam.set_position([0, -0.2, 0.1])  # Position camera
    >>> cam.look_at([0, 0, 0])  # Look at origin
    >>> cam.set_resolution(512, 512)  # 512×512 pixels
    >>> img, extent = cam.render_first_hit(volume.activated, temperature_field)
    >>> # extent will be (-0.005, 0.005, -0.005, 0.005) - constant at any distance
    """

    def _project_cam_to_plane(self, pos_cam: np.ndarray):
        """
        Orthographic projection: (u, v) = (x, y).

        No depth-dependent scaling - all points project with parallel rays.

        Returns
        -------
        u, v : Sensor plane coordinates (meters), same as camera X,Y
        zc : Camera Z for depth sorting (larger = closer, since forward is -Z)
        valid : All True (all points projectable in orthographic)
        """
        u = pos_cam[:, 0]
        v = pos_cam[:, 1]
        # Camera-space depth: Z is negative for points in front of camera
        # Less negative Z = closer to camera (e.g., z=-1 is closer than z=-2)
        # The base renderer negates this (using -zc) for near-to-far sorting
        zc = pos_cam[:, 2]
        valid = np.ones_like(u, dtype=bool)
        return u, v, zc, valid


class FollowingOrthographicCamera(OrthographicCamera, FollowingCameraMixin):
    """
    Orthographic camera that **follows a tracked point** with fixed relative offset.

    The camera maintains constant offset from the tracked point and always looks back
    at it, providing a consistent orthographic view of the moving target.

    Perfect for following a heat source, nozzle, or moving object while maintaining
    parallel projection (no perspective distortion).

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

    Example
    -------
    >>> # Create camera following a heat source
    >>> # Camera will be at fixed offset: -12cm in Y, +4cm in Z
    >>> cam = FollowingOrthographicCamera(
    ...     source_pos=(0.004, 0.003, 0.001),
    ...     offset=(0.0, -0.12, 0.04),
    ...     plane_size=(0.01, 0.01)  # See 10mm × 10mm area
    ... )
    >>> # Update source position as it moves
    >>> cam.update_target((0.004, 0.004, 0.001))
    >>> # Render
    >>> img, extent = cam.render_first_hit(volume.activated, temperature_field)
    """

    def __init__(self,
                 source_pos: ArrayLike3 = (0.0, 0.0, 0.0),
                 offset: ArrayLike3 = (0.0, -0.12, 0.04),
                 up_hint: ArrayLike3 = (0.0, 0.0, 1.0),
                 **camera_kwargs) -> None:
        super().__init__(**camera_kwargs)
        # Configure cartesian following
        self.configure_cartesian_following(
            offset=offset,
            up_hint=up_hint
        )
        # Position at initial target
        self.update_target(source_pos, motion_direction=None)