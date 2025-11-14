"""
Perspective camera callback (clean, from-scratch) that follows the tracked point and renders
using the real camera implementation. Designed for a testing dashboard that must not
re-implement camera math.

Key properties:
- Constructable purely from knobs (no UI-side math)
- Optical FOV independent from sampling knobs
- Optional digital crop to a physical window at the focal point (in mm)
- Uses the old simulation context shape (step_context + trackers)
- Exposes get_latest_image() and get_camera() for the dashboard
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Optional: saving preview via matplotlib
try:
    import matplotlib.pyplot as plt

    _MATP = True
except Exception:
    _MATP = False

# Base callback contract
from callbacks._base_callbacks import IntervalCallback, SimulationEvent

# Real camera (do not touch camera API)
try:
    from camera.perspective_camera import FollowingPerspectiveCamera

    _CAM_AVAILABLE = True
except Exception as e:
    _CAM_AVAILABLE = False
    _CAM_IMPORT_ERROR = e


class PerspectiveCameraCallback(IntervalCallback):
    """Follows the nozzle/heat-source and renders a thermal view from a perspective camera."""

    def __init__(
            self,
            # Camera configuration (pure knobs)
            rel_offset_local: Tuple[float, float, float] = (0.0, -0.12, 0.04),
            tilt_angle_deg: float = 30.0,
            fov_y_deg: float = 45.0,
            up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),

            # Sampling knobs
            plane_size: Tuple[float, float] = (0.06, 0.04),
            resolution_wh: Optional[Tuple[int, int]] = None,
            pixel_size_xy: Optional[Tuple[float, float]] = None,

            # Rendering
            ambient_temp: float = 300.0,
            cmap: str = "hot",

            # Optional saving (dashboard should disable)
            save_images: bool = False,
            save_dir: str = "camera_images",
            image_format: str = "png",
            dpi: int = 150,

            # Callback scheduling
            interval: int = 1,

            # Digital zoom window (mm at focal point)
            target_window_mm: Optional[Tuple[float, float]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(events=SimulationEvent.STEP_COMPLETE, interval=interval, **kwargs)

        if not _CAM_AVAILABLE:
            raise ImportError(f"camera.perspective_camera import failed: {_CAM_IMPORT_ERROR}")

        # Camera config
        self.rel_offset_local = tuple(map(float, rel_offset_local))
        self.tilt_angle_deg = float(tilt_angle_deg)
        self.fov_y_deg = float(fov_y_deg)
        self.up_hint = tuple(map(float, up_hint))

        # Sampling
        self.plane_size = tuple(map(float, plane_size))
        self.resolution_wh = tuple(map(int, resolution_wh)) if resolution_wh else None
        self.pixel_size_xy = (
            tuple(map(float, pixel_size_xy)) if pixel_size_xy else (0.001, 0.001)
        )

        # Rendering
        self.ambient_temp = float(ambient_temp)
        self.cmap = str(cmap)

        # Saving
        self.save_images = bool(save_images)
        self.save_dir = str(save_dir)
        self.image_format = str(image_format)
        self.dpi = int(dpi)

        # Target window (meters)
        self.target_window_m = (
            (target_window_mm[0] / 1000.0, target_window_mm[1] / 1000.0)
            if target_window_mm else None
        )

        # Lazy camera
        self.camera: Optional[FollowingPerspectiveCamera] = None
        self.latest_image = None
        self.latest_extent = None

    # ------------------------------------------------------------------
    # Core callback logic
    # ------------------------------------------------------------------
    def _execute(self, context: Dict[str, Any]) -> None:
        sim = context.get("simulation")
        if sim is None:
            warnings.warn("PerspectiveCameraCallback: missing simulation in context")
            return

        sc = getattr(sim, "step_context", None)
        if not sc or "position" not in sc:
            warnings.warn("PerspectiveCameraCallback: step_context.position missing")
            return
        pos = sc["position"]
        nozzle_pos = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)

        # Init camera if needed
        if self.camera is None:
            cfg = getattr(sim, "config", None)
            if not cfg or "voxel_size" not in cfg:
                warnings.warn("PerspectiveCameraCallback: config.voxel_size missing")
                return
            voxel_size = cfg["voxel_size"]
            self.camera = self._create_camera(nozzle_pos, voxel_size)

        # Follow
        self.camera.follow_heat_source(nozzle_pos, orient=True)

        # Data arrays
        activation = getattr(getattr(sim, "volume_tracker", None), "activated", None)
        temperature = getattr(getattr(sim, "temperature_tracker", None), "temperature", None)
        if activation is None or temperature is None:
            warnings.warn("PerspectiveCameraCallback: missing activation/temperature")
            return

        img, extent = self.camera.render_first_hit(activation, temperature, ambient=self.ambient_temp)

        # Apply digital crop if configured
        if self.target_window_m:
            img, extent = self.camera.apply_crop_window(
                img, extent,
                window_center=nozzle_pos,
                window_size=self.target_window_m
            )

        self.latest_image, self.latest_extent = img, extent

        if self.save_images and _MATP:
            self._save_image(img, extent)

    # ------------------------------------------------------------------
    # Camera creation
    # ------------------------------------------------------------------
    def _create_camera(self, nozzle_pos, voxel_size_xyz) -> FollowingPerspectiveCamera:
        return FollowingPerspectiveCamera(
            source_pos=tuple(map(float, nozzle_pos)),
            rel_offset_local=self.rel_offset_local,
            tilt_angle_deg=self.tilt_angle_deg,
            up_hint=self.up_hint,
            fov_y_deg=self.fov_y_deg,
            plane_size=self.plane_size,
            resolution_wh=self.resolution_wh,
            pixel_size_xy=self.pixel_size_xy,
            voxel_size_xyz=tuple(map(float, voxel_size_xyz)),
        )

    # ------------------------------------------------------------------
    # Optional save
    # ------------------------------------------------------------------
    def _save_image(self, img, extent):
        if not _MATP:
            return
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(img.T, origin="lower", extent=extent, cmap=self.cmap)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Perspective Camera (thermal)")
        fig.colorbar(im, ax=ax)
        fig.savefig(save_dir / f"camera_frame.{self.image_format}", dpi=self.dpi)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Dashboard accessors
    # ------------------------------------------------------------------
    def get_latest_image(self):
        return (self.latest_image, self.latest_extent) if self.latest_image is not None else None

    def get_camera(self):
        return self.camera