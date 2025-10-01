import numpy as np

from camera._base_camera import FollowingCameraMixin, ArrayLike3, BaseCamera


# -----------------------------------------------------------------------------
# Perspective camera (projection = converging rays, FOV-driven)
# -----------------------------------------------------------------------------

class PerspectiveCamera(BaseCamera):
    """
    **Perspective (normal) camera** with converging rays and a configurable
    vertical field-of-view. The image plane is centered at ``self.pos`` and is
    conceptually placed at a distance equal to the camera's **focal length**
    along ``forward``. The **plane size** and **FOV** together determine the
    focal length.

    Parameters
    ----------
    fov_y_deg : float
        Vertical field-of-view in degrees. Combined with ``plane_size[1]``
        defines the focal length: ``f = 0.5 * plane_size_y / tan(fov_y/2)``.
    """

    def __init__(self, fov_y_deg: float = 45.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fov_y_deg = float(fov_y_deg)
        self._update_focal_length()

    # ----- internals -----
    def _update_focal_length(self) -> None:
        """Recompute focal length from ``plane_size`` and ``fov_y_deg``."""
        fov_rad = np.deg2rad(self.fov_y_deg)
        half_h = 0.5 * self.plane_size[1]
        self._focal_length = half_h / np.tan(0.5 * fov_rad)  # meters

    # ----- public API -----
    def set_fov(self, fov_y_deg: float) -> None:
        """Set the vertical field-of-view (degrees) and update the focal length."""
        self.fov_y_deg = float(fov_y_deg)
        self._update_focal_length()

    def set_plane_size(self, width: float, height: float) -> None:
        """Set the plane size (meters) and update the focal length accordingly."""
        super().set_plane_size(width, height)
        self._update_focal_length()

    # ----- projection hook implementation -----
    def _project_cam_to_plane(self, pos_cam: np.ndarray):
        """
        Perspective projection:
        - Camera Z is negative **in front** of the camera with the current basis.
        - Use positive forward distance ``d = -z`` to avoid division-by-zero.
        - Image-plane coordinates (meters): (u, v) = (f * x / d, f * y / d).
        """
        x = pos_cam[:, 0]
        y = pos_cam[:, 1]
        z = pos_cam[:, 2]
        d = -z  # points in front → d > 0
        eps = 1e-8
        valid = d > eps
        # Avoid warnings; we'll mask invalid later
        d_safe = np.where(valid, d, 1.0)
        u = (self._focal_length * x) / d_safe
        v = (self._focal_length * y) / d_safe
        # Depth for sorting: larger should mean "closer". Since in-front z is negative,
        # less negative (closer) → larger z. Reuse z for consistency with ortho.
        zc = z
        return u, v, zc, valid


class FollowingPerspectiveCamera(PerspectiveCamera, FollowingCameraMixin):
    """
    Camera that **follows a heat source** (perspective projection) with a fixed
    relative pose and elevation ("floor angle"). The image plane stays anchored
    at ``self.pos`` while rays converge based on the configured FOV.
    """

    def __init__(self,
                 source_pos: ArrayLike3 = (0.0, 0.0, 0.0),
                 rel_offset_local: ArrayLike3 = (0.0, -0.12, 0.04),
                 floor_angle_deg: float = 30.0,
                 up_hint: ArrayLike3 = (0.0, 0.0, 1.0),
                 fov_y_deg: float = 45.0,
                 **camera_kwargs) -> None:
        super().__init__(fov_y_deg=fov_y_deg, **camera_kwargs)
        self.init_following(source_pos=source_pos,
                            rel_offset_local=rel_offset_local,
                            floor_angle_deg=floor_angle_deg,
                            up_hint=up_hint)