# -----------------------------------------------------------------------------
# cameras.py
# -----------------------------------------------------------------------------
# Unified camera module with a shared base class and two projection variants:
# - OrthographicCamera (parallel rays; fixed plane size in meters)
# - PerspectiveCamera  (converging rays; FOV-driven projection) ###FIXME: relevant for Hakim
#
# Optional following behavior is provided via a light-weight mixin that can be
# combined with either camera variant:
# - FollowingOrthographicCamera ###FIXME: FollowingOrthographicCamera for Hakim
# - FollowingPerspectiveCamera
#
# The public API mirrors your previous script:
#   - Pose: set_position, set_direction, look_at, set_from_spherical
#   - Coverage/Sampling: set_plane_size, set_pixel_size, set_resolution
#   - Rendering: render_first_hit(...)
#
# Only the projection step differs between orthographic and perspective and is
# implemented via a single overridable method on the base class.
# -----------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional

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

    Definition (meters)
    -------------------
    - Pose: ``pos`` (world center of the image plane) and orthonormal basis
      ``(right, up, forward)`` with the camera **looking along** ``forward``.
    - Coverage/Sampling: finite **plane size in meters** (``plane_size =
      (width, height)``) and either fixed pixel size (meters/pixel) or fixed
      resolution (pixels). You can supply either; the other is derived.

    Rendering
    ---------
    The renderer casts rays defined by the subclass projection through the
    finite plane centered at ``pos`` with normal ``forward`` and bins
    first-visible temperatures. Orientation and position come solely from the
    camera state; there is no additional world anchor argument.

    Fast path
    ---------
    Update only what changes frame-to-frame:
    - Call :meth:`set_position` when the camera moves but keeps orientation.
    - Call :meth:`look_at` or :meth:`set_direction` when orientation changes.
    - Call :meth:`set_plane_size`, :meth:`set_pixel_size`, or
      :meth:`set_resolution` to control the plane coverage / sampling.
    """

    # Pose
    pos: np.ndarray
    target: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray

    # Sampling/coverage (meters, pixels)
    plane_size: Tuple[float, float]          # (width, height) in meters
    pixel_size_xy: Tuple[float, float]       # (px, py) in meters per pixel
    resolution_wh: Tuple[int, int]           # (W, H) in pixels
    voxel_size_xyz: np.ndarray               # (dx, dy, dz) in meters

    def __init__(self,
                 pos: Optional[ArrayLike3] = None,
                 target: Optional[ArrayLike3] = None,
                 up_hint: ArrayLike3 = (0.0, 0.0, 1.0),
                 *,
                 plane_size: Tuple[float, float] = (0.06, 0.04),  # meters
                 pixel_size_xy: Optional[Tuple[float, float]] = None,
                 resolution_wh: Optional[Tuple[int, int]] = None,
                 voxel_size_xyz: Optional[Tuple[float, float, float]] = None) -> None:
        # Pose init
        self.pos = np.zeros(3) if pos is None else np.asarray(pos, float)
        self.target = np.zeros(3) if target is None else np.asarray(target, float)
        self.right  = np.array([1.0, 0.0, 0.0], float)
        self.up     = np.array([0.0, 1.0, 0.0], float)
        self.forward= np.array([0.0, 0.0, -1.0], float)
        if target is not None:
            self.look_at(self.target, up_hint=up_hint)

        # Voxel size
        if voxel_size_xyz is None:
            self.voxel_size_xyz = np.array([1e-4, 1e-4, 1e-4], dtype=float)
        else:
            self.voxel_size_xyz = np.asarray(voxel_size_xyz, dtype=float)

        # Coverage/sampling init (meters)
        self.plane_size = (float(plane_size[0]), float(plane_size[1]))
        # default pixel size = 1 mm (0.001 m) unless user overrides
        self.pixel_size_xy = (0.001, 0.001)
        # derive initial resolution from plane & pixel size
        self._derive_resolution_from_plane()
        # override with user-provided sampling if given
        if pixel_size_xy is not None and resolution_wh is not None:
            self.pixel_size_xy = (float(pixel_size_xy[0]), float(pixel_size_xy[1]))
            self._derive_resolution_from_plane()
        elif pixel_size_xy is not None:
            self.pixel_size_xy = (float(pixel_size_xy[0]), float(pixel_size_xy[1]))
            self._derive_resolution_from_plane()
        elif resolution_wh is not None:
            self.resolution_wh = (int(resolution_wh[0]), int(resolution_wh[1]))
            self._derive_pixel_size_from_plane()

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

    def set_direction(self, direction: ArrayLike3, up_hint: ArrayLike3=(0.0,0.0,1.0)) -> None:
        """Set the camera's orientation from a direction vector."""
        self.right, self.up, self.forward = self._basis_from(direction, up_hint)

    def look_at(self, target: ArrayLike3, up_hint: ArrayLike3=(0.0,0.0,1.0)) -> None:
        """Orient the camera to look at a target point from its current position."""
        self.target = np.asarray(target, float)
        self.set_direction(self.target - self.pos, up_hint=up_hint)

    def set_from_spherical(self,
                           az_deg: float,
                           el_deg: float,
                           roll_deg: float = 0.0,
                           radius: float = 0.2,
                           target: ArrayLike3 = (0.0, 0.0, 0.0),
                           up_hint: ArrayLike3=(0.0,0.0,1.0)) -> None:
        """Place camera on a sphere around ``target`` and face it."""
        az = np.deg2rad(az_deg); el = np.deg2rad(el_deg); roll = np.deg2rad(roll_deg)
        dir_world = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], float)
        dir_world = _norm(dir_world)
        self.pos = np.asarray(target, float) - radius * dir_world
        self.look_at(target, up_hint=up_hint)
        if abs(roll) > 1e-12:
            r, u = self.right, self.up
            self.right =  r*np.cos(roll) + u*np.sin(roll)
            self.up    =  u*np.cos(roll) - r*np.sin(roll)

    # ----- public API: coverage/sampling -----
    def set_plane_size(self, width: float, height: float) -> None:
        """Set the finite plane size (meters) and keep current pixel size."""
        self.plane_size = (float(width), float(height))
        self._derive_resolution_from_plane()

    def set_pixel_size(self, px: float, py: float) -> None:
        """Set pixel spacing (meters); resolution will be derived from plane size."""
        self.pixel_size_xy = (float(px), float(py))
        self._derive_resolution_from_plane()

    def set_resolution(self, width_px: int, height_px: int) -> None:
        """Set output resolution (pixels); pixel size will be derived from plane."""
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
        M[:3,:3] = R
        M[:3, 3] = t
        return M

    # ----- projection hook (must be implemented by subclasses) -----
    def _project_cam_to_plane(self, pos_cam: np.ndarray):
        """
        Map camera-space points to **image-plane coordinates**.

        Parameters
        ----------
        pos_cam : (N,3) ndarray
            Points in camera coordinates.

        Returns
        -------
        u : (N,) ndarray
            Image-plane X coordinates (meters), centered at 0.
        v : (N,) ndarray
            Image-plane Y coordinates (meters), centered at 0.
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
                         ambient: float = 300.0):
        """
        Render a **first-visible** thermal image using the camera's plane
        anchored at ``self.pos`` with normal ``self.forward`` (meters).

        Parameters
        ----------
        volume_activated : (Nx,Ny,Nz) bool
            Activation mask.
        temperature_field : (Nx,Ny,Nz) float
            Per-voxel temperature.
        ambient : float
            Temperature used for pixels with no hit.

        Returns
        -------
        img : (W,H) ndarray
        extent : (xmin, xmax, ymin, ymax) in camera-plane coordinates (meters), centered at 0.
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

        # Project activated voxel centers into camera space, relative to camera.pos
        pos_world = idx * vs
        pos_cam = (R_wc @ (pos_world - self.pos).T).T  # (N,3)

        # Subclass-specific projection to the image plane
        u, v, zc, valid = self._project_cam_to_plane(pos_cam)
        if not np.any(valid):
            img = np.full((W, H), float(ambient), dtype=T.dtype)
            return img, extent

        u = u[valid]; v = v[valid]; zc = zc[valid]
        idx = idx[valid]

        # Bin to pixels in plane centered at (0,0) with extents [-half_w, half_w] × [-half_h, half_h]
        ix = np.floor((u + half_w) / px).astype(int)
        iy = np.floor((v + half_h) / py).astype(int)

        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not np.any(inside):
            img = np.full((W, H), float(ambient), dtype=T.dtype)
            return img, extent

        ix = ix[inside]; iy = iy[inside]
        zc = zc[inside]
        flat = ix + W * iy

        # Per-pixel pick closest voxel: primary key = pixel, secondary = depth
        order = np.lexsort((-zc, flat))
        flat_sorted = flat[order]
        idx_sorted = idx[inside][order]

        unique_flat, first_pos = np.unique(flat_sorted, return_index=True)
        vox_hit = idx_sorted[first_pos]
        temps = T[vox_hit[:,0], vox_hit[:,1], vox_hit[:,2]]

        img = np.full((W, H), float(ambient), dtype=T.dtype)
        ix_hit = unique_flat % W
        iy_hit = unique_flat // W
        img[ix_hit, iy_hit] = temps

        return img, extent

# -----------------------------------------------------------------------------
# Following mixin (can be combined with either camera variant)
# -----------------------------------------------------------------------------

class FollowingCameraMixin:
    """
    Mixin providing **following behavior** for any camera that implements
    :meth:`look_at` and owns ``pos``. Combine with OrthographicCamera or
    PerspectiveCamera to create a following camera that keeps a fixed relative
    pose to a moving source and optional elevation ("floor angle").
    """

    def init_following(self,
                       *,
                       source_pos: ArrayLike3 = (0.0, 0.0, 0.0),
                       rel_offset_local: ArrayLike3 = (0.0, -0.12, 0.04),  # meters
                       floor_angle_deg: float = 30.0,
                       up_hint: ArrayLike3 = (0.0, 0.0, 1.0)) -> None:
        """
        Initialize following state and orient the camera toward the source.

        Args
        ----
        source_pos : ArrayLike3
            Initial position of the heat source to follow.
        rel_offset_local : ArrayLike3
            Camera offset relative to the source, in the source's local frame
            **before** elevation.
        floor_angle_deg : float
            Elevation angle about local X-axis (degrees), looking "down".
        up_hint : ArrayLike3
            World-space up vector for orientation.
        """
        self.up_hint = np.asarray(up_hint, float)
        self.set_relative_config(rel_offset_local=rel_offset_local,
                                 floor_angle_deg=floor_angle_deg,
                                 up_hint=up_hint)
        self.follow_heat_source(source_pos, orient=True)

    def set_relative_config(self,
                            rel_offset_local: Optional[ArrayLike3] = None,
                            floor_angle_deg: Optional[float] = None,
                            up_hint: Optional[ArrayLike3] = None) -> None:
        """
        Set the following behavior.

        Args
        ----
        rel_offset_local : Optional[ArrayLike3]
            New local offset from the heat source.
        floor_angle_deg : Optional[float]
            New elevation angle (degrees).
        up_hint : Optional[ArrayLike3]
            New world-space up vector for orientation.
        """
        if rel_offset_local is not None:
            self.rel_offset_local = np.asarray(rel_offset_local, float)
        if floor_angle_deg is not None:
            self.floor_angle_deg = float(floor_angle_deg)
            ang = np.deg2rad(self.floor_angle_deg)
            self._cos_ang = float(np.cos(ang))
            self._sin_ang = float(np.sin(ang))
        if up_hint is not None:
            self.up_hint = np.asarray(up_hint, float)

    def follow_heat_source(self, source_pos: ArrayLike3, *, orient: bool = True) -> None:
        """
        Update camera position and orientation to follow the heat source.

        Args
        ----
        source_pos : ArrayLike3
            The new position of the heat source in world coordinates.
        orient : bool
            If True, re-orient the camera to look at the source.
            If False, only translate the camera, keeping its orientation.
        """
        src = np.asarray(source_pos, float)
        # Elevate stored offset about local X by floor angle
        ox, oy, oz = self.rel_offset_local
        oy_e =  self._cos_ang * oy - self._sin_ang * oz
        oz_e =  self._sin_ang * oy + self._cos_ang * oz
        offset_world = np.array([ox, oy_e, oz_e], float)
        self.pos = src + offset_world
        if orient:
            self.look_at(src, up_hint=self.up_hint)

