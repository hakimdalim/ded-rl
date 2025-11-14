"""
Generalized camera callback that supports both orthographic and perspective cameras.

Follows a tracked point and renders using the actual camera implementation.
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Literal
import numpy as np

# Optional: saving preview via matplotlib
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False

# Base callback contract
from callbacks._base_callbacks import IntervalCallback, SimulationEvent

# Import both camera types
try:
    from camera.perspective_camera import FollowingPerspectiveCamera
    from camera.orthographic_camera import FollowingOrthographicCamera
    _CAM_AVAILABLE = True
except Exception as e:
    _CAM_AVAILABLE = False
    _CAM_IMPORT_ERROR = e


class CameraCallback(IntervalCallback):
    """
    Follows the nozzle/heat-source and renders thermal view from either camera type.
    
    Parameters
    ----------
    camera_type : Literal["perspective", "orthographic"]
        Type of camera to use
    fov_y_deg : float
        Vertical field of view for perspective camera (ignored for orthographic)
    """

    def __init__(
            self,
            # Camera type selection
            camera_type: Literal["perspective", "orthographic"] = "perspective",
            
            # Camera configuration
            offset: Tuple[float, float, float] = (0.0, -0.12, 0.04),
            fov_y_deg: float = 45.0,
            up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),

            # Sampling
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

            fill_gaps=True,
            **kwargs: Any,
    ) -> None:
        super().__init__(events=SimulationEvent.STEP_COMPLETE, interval=interval, **kwargs)

        if not _CAM_AVAILABLE:
            raise ImportError(f"camera import failed: {_CAM_IMPORT_ERROR}")

        # Camera type
        self.camera_type = camera_type
        if camera_type not in ["perspective", "orthographic"]:
            raise ValueError(f"camera_type must be 'perspective' or 'orthographic', got {camera_type}")

        # Camera config
        self.offset = tuple(map(float, offset))
        self.fov_y_deg = float(fov_y_deg)
        self.up_hint = tuple(map(float, up_hint))

        # Sampling
        self.plane_size = tuple(map(float, plane_size))
        self.resolution_wh = tuple(map(int, resolution_wh)) if resolution_wh else None
        self.pixel_size_xy = tuple(map(float, pixel_size_xy)) if pixel_size_xy else (0.001, 0.001)

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
        self.camera = None
        self.latest_image = None
        self.latest_extent = None

        # Motion tracking for local coordinate transformation
        self._previous_nozzle_pos = None

        self.fill_gaps=fill_gaps

    # ------------------------------------------------------------------
    # Core callback logic
    # ------------------------------------------------------------------
    def _execute(self, context: Dict[str, Any]) -> None:
        sim = context.get("simulation")
        if sim is None:
            warnings.warn("CameraCallback: missing simulation in context")
            return

        sc = getattr(sim, "step_context", None)
        if not sc or "position" not in sc:
            warnings.warn("CameraCallback: step_context.position missing")
            return
        pos = sc["position"]
        nozzle_pos = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)

        # Init camera if needed
        if self.camera is None:
            cfg = getattr(sim, "config", None)
            if not cfg or "voxel_size" not in cfg:
                warnings.warn("CameraCallback: config.voxel_size missing")
                return
            voxel_size = cfg["voxel_size"]
            self.camera = self._create_camera(nozzle_pos, voxel_size)

        # Compute motion direction from position history
        motion_direction = None
        if self._previous_nozzle_pos is not None:
            motion_direction = nozzle_pos - self._previous_nozzle_pos
            motion_magnitude = np.linalg.norm(motion_direction)
            if motion_magnitude < 1e-10:
                # Stationary: no motion
                motion_direction = None

        # Follow with proper local coordinate transformation (v2.1 API)
        self.camera.update_target(nozzle_pos, motion_direction=motion_direction, orient=True)

        # Update history
        self._previous_nozzle_pos = nozzle_pos.copy()

        # Data arrays
        activation = getattr(getattr(sim, "volume_tracker", None), "activated", None)
        temperature = getattr(getattr(sim, "temperature_tracker", None), "temperature", None)

        # Render using public method
        self.render(activation=activation, temperature=temperature, focus_pos=nozzle_pos)

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------
    def render(self,
              activation: np.ndarray,
              temperature: np.ndarray,
              focus_pos: Optional[np.ndarray] = None,
              ambient: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """
        Public rendering method that can be called externally.

        This method renders the thermal view and optionally saves the image.
        It can be called either from the internal _execute() method or
        from external code for manual rendering.

        Parameters
        ----------
        activation : np.ndarray
            Boolean array indicating activated voxels
        temperature : np.ndarray
            Temperature field array
        focus_pos : Optional[np.ndarray]
            Position to focus on for cropping (if target_window_m is set).
            If None, uses camera target.
        ambient : Optional[float]
            Ambient temperature for non-activated voxels.
            If None, uses self.ambient_temp.

        Returns
        -------
        img : Optional[np.ndarray]
            Rendered image, or None if rendering failed
        extent : Optional[Tuple]
            Image extent (xmin, xmax, ymin, ymax), or None if rendering failed

        Examples
        --------
        >>> callback = CameraCallback(...)
        >>> # Manual rendering at specific position
        >>> img, extent = callback.render(
        ...     activation=volume.activated,
        ...     temperature=volume.temperature,
        ...     focus_pos=nozzle_position
        ... )

        Notes
        -----
        The rendered image is stored in self.latest_image and self.latest_extent
        for access by dashboards/visualizations.
        """
        if activation is None or temperature is None:
            warnings.warn("CameraCallback.render(): missing activation or temperature arrays")
            return None, None

        if ambient is None:
            ambient = self.ambient_temp

        if focus_pos is None:
            focus_pos = self.camera.target if hasattr(self.camera, 'target') else np.array([0, 0, 0])

        # Render with optional crop
        try:
            if self.target_window_m:
                # Use efficient render_crop (10-100x faster)
                img, extent = self.camera.render_crop(
                    activation, temperature,
                    window_center=focus_pos,
                    window_size=self.target_window_m,
                    ambient=ambient
                )
            else:
                # No crop: render full view
                img, extent = self.camera.render_first_hit(activation, temperature, ambient=ambient)

            self.latest_image, self.latest_extent = img, extent

            if self.save_images and _MATPLOTLIB_AVAILABLE:
                self._save_image(img, extent)

            return img, extent

        except Exception as e:
            warnings.warn(f"CameraCallback.render() failed: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Camera creation
    # ------------------------------------------------------------------
    def _create_camera(self, nozzle_pos, voxel_size_xyz):
        """Create the appropriate camera type based on configuration."""
        base_kwargs = {
            'source_pos': tuple(map(float, nozzle_pos)),
            'offset': self.offset,
            'up_hint': self.up_hint,
            'plane_size': self.plane_size,
            'resolution_wh': self.resolution_wh,
            'pixel_size_xy': self.pixel_size_xy,
            'voxel_size_xyz': tuple(map(float, voxel_size_xyz)),
            'fill_gaps': self.fill_gaps,
        }
        
        if self.camera_type == "perspective":
            return FollowingPerspectiveCamera(
                fov_y_deg=self.fov_y_deg,
                **base_kwargs
            )
        else:  # orthographic
            return FollowingOrthographicCamera(**base_kwargs)

    # ------------------------------------------------------------------
    # Optional save
    # ------------------------------------------------------------------
    def _save_image(self, img, extent):
        if not _MATPLOTLIB_AVAILABLE:
            return
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(img.T, origin="lower", extent=extent, cmap=self.cmap)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{self.camera_type.capitalize()} Camera (thermal)")
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