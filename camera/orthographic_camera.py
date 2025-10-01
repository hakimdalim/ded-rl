# -----------------------------------------------------------------------------
# Orthographic camera (projection = parallel rays)
# -----------------------------------------------------------------------------
import numpy as np

from camera._base_camera import BaseCamera, FollowingCameraMixin, ArrayLike3


class OrthographicCamera(BaseCamera):
    """
    Production-ready **orthographic camera** whose image plane is anchored at
    ``camera.pos``. No external "look_at" point is needed for rendering.

    Definition (meters)
    -------------------
    - Pose: ``pos`` (world center of the image plane) and orthonormal basis
      ``(right, up, forward)`` with the camera **looking along** ``forward``.
    - Field of view: finite **plane size in meters**, given by ``plane_size =
      (width, height)``.
    - Sampling: either fixed pixel size (meters/pixel) or fixed resolution
      (pixels). You can supply either; the other is derived.

    Rendering
    ---------
    The renderer casts **parallel rays** through the finite plane and bins
    first-hit temperatures.
    """

    def _project_cam_to_plane(self, pos_cam: np.ndarray):
        """
        Orthographic projection: (u, v) = (x, y). All points are projectable.
        Depth for sorting uses camera Z; larger values mean closer (less negative).
        """
        u = pos_cam[:, 0]
        v = pos_cam[:, 1]
        zc = pos_cam[:, 2]            # With our rotation, points in front have negative z
        valid = np.ones_like(u, dtype=bool)
        return u, v, zc, valid


class FollowingOrthographicCamera(OrthographicCamera, FollowingCameraMixin):
    """
    Camera that **follows a heat source** (orthographic projection) with a fixed
    relative pose and elevation ("floor angle"). The image plane stays anchored
    at ``self.pos``.
    """
    def __init__(self,
                 source_pos: ArrayLike3 = (0.0, 0.0, 0.0),
                 rel_offset_local: ArrayLike3 = (0.0, -0.12, 0.04),
                 floor_angle_deg: float = 30.0,
                 up_hint: ArrayLike3 = (0.0, 0.0, 1.0),
                 **camera_kwargs) -> None:
        super().__init__(**camera_kwargs)
        self.init_following(source_pos=source_pos,
                            rel_offset_local=rel_offset_local,
                            floor_angle_deg=floor_angle_deg,
                            up_hint=up_hint)
