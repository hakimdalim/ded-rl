"""
Perspective camera callback that follows the nozzle/heat source during simulation.

This callback creates a perspective camera that tracks the laser/nozzle position
and optionally saves thermal images after each step.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "matplotlib not installed. PerspectiveCameraCallback will not work. "
        "Install with: pip install matplotlib"
    )

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn(
        "PIL not installed. Overlay rendering will not work. "
        "Install with: pip install pillow"
    )

from callbacks._base_callbacks import IntervalCallback, SimulationEvent


# Import camera classes
try:
    from camera.perspective_camera import FollowingPerspectiveCamera
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    warnings.warn(
        "Camera module not available. PerspectiveCameraCallback will not work."
    )


class PerspectiveCameraCallback(IntervalCallback):
    """
    Perspective camera that follows the nozzle/heat source and optionally saves images.

    This callback creates a perspective camera with configurable position relative to
    the nozzle. The camera automatically follows the nozzle as it moves during the build.

    Features:
    - Perspective projection (realistic view with depth)
    - Follows nozzle/heat source automatically
    - Configurable camera position and viewing angle
    - Optional image saving after each step
    - Thermal colormap visualization
    -  Realistic overlay with nozzle body and powder stream

    Overlay Features:###TODO: refactor to 2D Hakim
    - Coaxial nozzle body geometry (frustum cone)
    - V-shaped powder stream with 2000 particles
    - Gaussian radial distribution (denser in center)
    - Perspective-correct projection
    - Configurable appearance and density
    - Can be toggled on/off independently

    Usage:
        # Basic usage - camera follows nozzle, saves every 10 steps
        PerspectiveCameraCallback(
            save_images=True,
            interval=10
        )

        # Custom camera position - look from behind and above
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.15, 0.05),  # x, y, z in meters
            floor_angle_deg=35.0,                  # look down at 35 degrees
            fov_y_deg=50.0,                        # field of view
            save_images=True,
            interval=5
        )

        # With realistic overlay (nozzle + powder stream)
        PerspectiveCameraCallback(
            enable_overlay=True,
            overlay_config={
                'stream_height_mm': 15.0,        # V-cone height (13-16mm typical)
                'v_angle_deg': 15.0,             # V-opening angle
                'num_particles': 2000,           # Particle count
                'nozzle_outlet_radius_mm': 4.0,  # Nozzle tip radius
                'nozzle_top_radius_mm': 10.0,    # Nozzle top radius
                'nozzle_height_mm': 40.0,        # Nozzle body height
            },
            save_images=True,
            interval=1
        )

        # Disable overlay (thermal only)
        PerspectiveCameraCallback(
            enable_overlay=False,
            save_images=True,
            interval=1
        )

    Output:
        camera_images/
            thermal_step_0001.png
            thermal_step_0002.png
            ...

    Camera Position:
        The camera position is specified relative to the nozzle in the nozzle's
        local coordinate frame:
        - X: perpendicular to scan direction (right when looking forward)
        - Y: along scan direction (negative = behind nozzle)
        - Z: vertical (up)

        Default: (0.0, -0.12, 0.04) = 12cm behind, 4cm above
    """

    def __init__(
        self,
        # Camera configuration
        rel_offset_local: Tuple[float, float, float] = (0.0, -0.12, 0.04),
        floor_angle_deg: float = 30.0,
        fov_y_deg: float = 45.0,
        up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),

        # Plane/resolution configuration
        plane_size: Tuple[float, float] = (0.06, 0.04),  # meters (width, height)
        resolution_wh: Optional[Tuple[int, int]] = None,  # pixels (width, height)
        pixel_size_xy: Optional[Tuple[float, float]] = None,  # meters per pixel

        # Rendering configuration
        ambient_temp: float = 300.0,
        cmap: str = 'hot',

        # Overlay configuration
        enable_overlay: bool = True,
        overlay_config: Optional[Dict[str, Any]] = None,

        # Saving configuration
        save_images: bool = True,
        save_dir: str = "cam",  # Short name to avoid Windows path length issues
        image_format: str = "png",
        dpi: int = 150,

        # Callback configuration
        interval: int = 1,
        **kwargs
    ):
        """
        Initialize perspective camera callback.

        Args:
            rel_offset_local: Camera offset relative to nozzle (x, y, z) in meters
                            Default: (0.0, -0.12, 0.04) = 12cm behind, 4cm above
            floor_angle_deg: Viewing angle downward in degrees (0=horizontal, 90=straight down)
                           Default: 30 degrees
            fov_y_deg: Vertical field of view in degrees
                      Default: 45 degrees
            up_hint: World-space up vector for camera orientation
                    Default: (0, 0, 1) = Z-up

            plane_size: Image plane size in meters (width, height)
                       Default: (0.06, 0.04) = 6cm x 4cm
            resolution_wh: Output resolution in pixels (width, height)
                         If not specified, derived from plane_size and pixel_size
            pixel_size_xy: Pixel spacing in meters (px, py)
                          Default: 0.001m = 1mm per pixel

            ambient_temp: Temperature for pixels with no hit (K)
                         Default: 300K
            cmap: Matplotlib colormap for thermal visualization
                 Default: 'hot'

            save_images: If True, save images to disk after each step
                        Default: True
            save_dir: Directory name for saved images (relative to output_dir)
                     Default: "camera_images"
            image_format: Image file format ('png', 'jpg', etc.)
                        Default: 'png'
            dpi: Image resolution (dots per inch)
                Default: 150

            interval: Save/render every N steps
                     Default: 1 (every step)
            **kwargs: Additional arguments for IntervalCallback
        """
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )

        if not CAMERA_AVAILABLE:
            raise ImportError(
                "Camera module is required for PerspectiveCameraCallback. "
                "Check camera/ directory is available."
            )

        if save_images and not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for saving images. "
                "Install with: pip install matplotlib"
            )

        # Camera configuration
        self.rel_offset_local = rel_offset_local
        self.floor_angle_deg = floor_angle_deg
        self.fov_y_deg = fov_y_deg
        self.up_hint = up_hint
        self.plane_size = plane_size
        self.resolution_wh = resolution_wh
        self.pixel_size_xy = pixel_size_xy if pixel_size_xy is not None else (0.001, 0.001)

        # Rendering configuration
        self.ambient_temp = ambient_temp
        self.cmap = cmap

        # Saving configuration
        self.save_images = save_images
        self.save_dir = save_dir
        self.image_format = image_format
        self.dpi = dpi

        # Overlay configuration
        self.enable_overlay = enable_overlay
        if enable_overlay and not PIL_AVAILABLE:
            warnings.warn(
                "PIL not available. Overlay will be disabled. "
                "Install with: pip install pillow"
            )
            self.enable_overlay = False

        # Set overlay parameters with smart defaults
        default_overlay_config = {
            'stream_height_mm': 15.0,
            'v_angle_deg': 15.0,
            'num_particles': 2000,
            'gaussian_sigma_ratio': 0.3,
            'nozzle_outlet_radius_mm': 4.0,
            'nozzle_top_radius_mm': 10.0,
            'nozzle_height_mm': 40.0,
            'nozzle_segments': 32,
            'particle_color': (255, 255, 255),
            'particle_alpha': 0.7,
            'particle_size_px': 2,
            'nozzle_color': (180, 180, 200),
            'nozzle_alpha': 0.8,
            'nozzle_outline_color': (100, 100, 120),
            'nozzle_line_width': 2,
        }
        if overlay_config is not None:
            default_overlay_config.update(overlay_config)
        self.overlay_config = default_overlay_config

        # Convert overlay parameters to internal units (meters, radians)
        self._stream_height = self.overlay_config['stream_height_mm'] / 1000.0
        self._v_angle_rad = np.deg2rad(self.overlay_config['v_angle_deg'])
        self._nozzle_outlet_radius = self.overlay_config['nozzle_outlet_radius_mm'] / 1000.0
        self._nozzle_top_radius = self.overlay_config['nozzle_top_radius_mm'] / 1000.0
        self._nozzle_height = self.overlay_config['nozzle_height_mm'] / 1000.0

        # Camera instance (created on first execute)
        self.camera = None

    def _execute(self, context: dict) -> None:
        """Render thermal image from perspective camera."""
        sim = context['simulation']

        # Get current nozzle/heat source position
        if not hasattr(sim, 'step_context') or sim.step_context is None:
            warnings.warn("No step context available for camera tracking")
            return

        step_ctx = sim.step_context
        if 'position' not in step_ctx:
            warnings.warn("No position in step context for camera tracking")
            return

        # Get nozzle position
        nozzle_pos = np.array([
            step_ctx['position']['x'],
            step_ctx['position']['y'],
            step_ctx['position']['z']
        ])

        # Create camera on first call
        if self.camera is None:
            # Get voxel size from simulation
            voxel_size = sim.config.get('voxel_size', (1e-4, 1e-4, 1e-4))

            # Create following perspective camera
            self.camera = FollowingPerspectiveCamera(
                source_pos=nozzle_pos,
                rel_offset_local=self.rel_offset_local,
                floor_angle_deg=self.floor_angle_deg,
                up_hint=self.up_hint,
                fov_y_deg=self.fov_y_deg,
                plane_size=self.plane_size,
                resolution_wh=self.resolution_wh,
                pixel_size_xy=self.pixel_size_xy,
                voxel_size_xyz=voxel_size
            )

            print(f"Perspective camera initialized:")
            print(f"  Offset: {self.rel_offset_local} m")
            print(f"  Angle: {self.floor_angle_deg}°")
            print(f"  FOV: {self.fov_y_deg}°")
            print(f"  Resolution: {self.camera.resolution_wh} px")

        # Update camera to follow nozzle
        self.camera.follow_heat_source(nozzle_pos, orient=True)

        # Get activation volume and temperature field
        activation = sim.volume_tracker.activated
        temperature = sim.temperature_tracker.temperature

        if activation is None or temperature is None:
            warnings.warn("No activation or temperature data available")
            return

        # Render thermal image
        img, extent = self.camera.render_first_hit(
            activation,
            temperature,
            ambient=self.ambient_temp
        )

        # Store latest image for access
        self.latest_image = img
        self.latest_extent = extent

        # Save image if requested
        if self.save_images:
            step = sim.progress_tracker.step_count
            self._save_image(img, extent, step, context, nozzle_pos)

    # ========================================================================
    # Overlay Rendering Methods
    # ========================================================================

    def _generate_nozzle_geometry(self, nozzle_pos: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate 3D geometry for conical nozzle body (frustum).

        Args:
            nozzle_pos: Nozzle tip position in world coordinates (x, y, z)

        Returns:
            Dictionary with 'top_circle' and 'bottom_circle' arrays
        """
        # Generate angles for circular cross-sections
        n_segments = self.overlay_config['nozzle_segments']
        theta = np.linspace(0, 2*np.pi, n_segments, endpoint=True)

        # Top circle (larger radius, above nozzle tip)
        top_z = nozzle_pos[2] + self._nozzle_height
        top_circle = np.column_stack([
            nozzle_pos[0] + self._nozzle_top_radius * np.cos(theta),
            nozzle_pos[1] + self._nozzle_top_radius * np.sin(theta),
            np.full(len(theta), top_z)
        ])

        # Bottom circle (smaller radius, at nozzle tip/outlet)
        bottom_circle = np.column_stack([
            nozzle_pos[0] + self._nozzle_outlet_radius * np.cos(theta),
            nozzle_pos[1] + self._nozzle_outlet_radius * np.sin(theta),
            np.full(len(theta), nozzle_pos[2])
        ])

        return {
            'top_circle': top_circle,
            'bottom_circle': bottom_circle
        }

    def _generate_powder_particles(self, nozzle_pos: np.ndarray) -> np.ndarray:
        """
        Generate random powder particle positions in V-shaped cone.

        Distribution:
        - Vertical: Uniform from nozzle to stream_height below
        - Radial: Gaussian (denser in center)
        - Angular: Uniform around axis

        Args:
            nozzle_pos: Nozzle tip position (x, y, z)

        Returns:
            (num_particles, 3) array of particle positions
        """
        num_particles = self.overlay_config['num_particles']
        sigma_ratio = self.overlay_config['gaussian_sigma_ratio']

        particles = np.zeros((num_particles, 3))

        for i in range(num_particles):
            # Sample vertical position (uniform from nozzle down to stream_height)
            z_offset = -np.random.uniform(0, self._stream_height)
            z = nozzle_pos[2] + z_offset

            # Calculate cone radius at this height
            cone_radius = self._nozzle_outlet_radius + abs(z_offset) * np.tan(self._v_angle_rad)

            # Sample radial distance (truncated Gaussian)
            sigma = cone_radius * sigma_ratio
            valid = False
            while not valid:
                r = abs(np.random.normal(0, sigma))
                if r <= cone_radius:
                    valid = True

            # Sample angular position (uniform)
            theta = np.random.uniform(0, 2*np.pi)

            # Convert to Cartesian
            x = nozzle_pos[0] + r * np.cos(theta)
            y = nozzle_pos[1] + r * np.sin(theta)

            particles[i] = [x, y, z]

        return particles

    def _project_to_screen(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D screen coordinates using camera projection.

        Args:
            points_3d: (N, 3) array of 3D points in world coordinates

        Returns:
            Tuple of (screen_coords, visible):
                - screen_coords: (N, 2) array of pixel coordinates
                - visible: (N,) boolean array for visibility
        """
        if len(points_3d) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=bool)

        # Transform to camera space
        R_wc = self.camera._world_to_cam_rotation()
        pos_cam = (R_wc @ (points_3d - self.camera.pos).T).T  # (N, 3)

        # Use camera's projection method
        u, v, zc, valid = self.camera._project_cam_to_plane(pos_cam)

        # Convert plane coordinates (u, v) in meters to pixel coordinates
        W, H = self.camera.resolution_wh
        px, py = self.camera.pixel_size_xy
        half_w = 0.5 * self.camera.plane_size[0]
        half_h = 0.5 * self.camera.plane_size[1]

        # Map from plane coordinates (centered at 0) to pixel coordinates
        ix = (u + half_w) / px
        iy = (v + half_h) / py

        screen_coords = np.column_stack([ix, iy])

        return screen_coords, valid

    def _render_overlay_on_image(self, img: np.ndarray, nozzle_pos: np.ndarray) -> np.ndarray:
        """
        Render overlay (nozzle + powder) on thermal image.

        Args:
            img: (W, H) thermal image array
            nozzle_pos: Nozzle position in world coordinates

        Returns:
            (H, W, 3) RGB image with overlay
        """
        # Convert thermal image to RGB for compositing
        # Normalize to 0-255
        vmin = self.ambient_temp
        vmax = img.max() if img.max() > vmin else vmin + 100
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)

        # Apply colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap(self.cmap)
        img_rgb = (cmap(img_norm)[:, :, :3] * 255).astype(np.uint8)

        # Convert to PIL Image - PIL expects (H, W, 3), img is (W, H) -> img_rgb is (W, H, 3)
        # Transpose only the first two dimensions to get (H, W, 3)
        img_rgb_transposed = np.transpose(img_rgb, (1, 0, 2))
        img_pil = Image.fromarray(img_rgb_transposed)

        # Create overlay layer
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Generate geometries
        nozzle_geom = self._generate_nozzle_geometry(nozzle_pos)
        particles = self._generate_powder_particles(nozzle_pos)

        # Project to screen
        top_screen, top_visible = self._project_to_screen(nozzle_geom['top_circle'])
        bottom_screen, bottom_visible = self._project_to_screen(nozzle_geom['bottom_circle'])
        particles_screen, particles_visible = self._project_to_screen(particles)

        # Draw nozzle body
        nozzle_color = self.overlay_config['nozzle_outline_color']
        nozzle_alpha = int(self.overlay_config['nozzle_alpha'] * 255)
        line_width = self.overlay_config['nozzle_line_width']

        # Draw top and bottom circles
        for i in range(len(top_screen) - 1):
            if top_visible[i] and top_visible[i + 1]:
                draw.line(
                    [tuple(top_screen[i]), tuple(top_screen[i + 1])],
                    fill=(*nozzle_color, nozzle_alpha),
                    width=line_width
                )

            if bottom_visible[i] and bottom_visible[i + 1]:
                draw.line(
                    [tuple(bottom_screen[i]), tuple(bottom_screen[i + 1])],
                    fill=(*nozzle_color, nozzle_alpha),
                    width=line_width
                )

        # Draw connecting lines between top and bottom
        for i in range(0, len(top_screen), 4):  # Every 4th to avoid clutter
            if top_visible[i] and bottom_visible[i]:
                draw.line(
                    [tuple(top_screen[i]), tuple(bottom_screen[i])],
                    fill=(*nozzle_color, nozzle_alpha // 2),
                    width=1
                )

        # Draw powder particles
        particle_color = self.overlay_config['particle_color']
        particle_alpha = int(self.overlay_config['particle_alpha'] * 255)
        particle_size = self.overlay_config['particle_size_px']

        for pos, vis in zip(particles_screen, particles_visible):
            if vis:
                x, y = pos
                r = particle_size
                bbox = [x - r, y - r, x + r, y + r]
                draw.ellipse(
                    bbox,
                    fill=(*particle_color, particle_alpha),
                    outline=None
                )

        # Composite overlay onto base image
        img_pil_rgba = img_pil.convert('RGBA')
        result = Image.alpha_composite(img_pil_rgba, overlay)

        # Convert back to numpy RGB
        result_rgb = result.convert('RGB')
        result_array = np.array(result_rgb)

        return result_array

    def _save_image(self, img: np.ndarray, extent: tuple, step: int, context: dict, nozzle_pos: np.ndarray = None):
        """Save thermal image to file."""
        # Resolve save directory (following ThermalPlotSaver pattern)
        save_path = self.resolve_path(context, self.save_dir)
        self.ensure_dir(save_path)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Apply overlay if enabled
        if self.enable_overlay and nozzle_pos is not None:
            # Render overlay on thermal image
            img_with_overlay = self._render_overlay_on_image(img, nozzle_pos)

            # Display RGB image with overlay
            im = ax.imshow(
                img_with_overlay,
                origin='lower',
                extent=extent,
                aspect='auto'
            )

            # Note: Colorbar doesn't make sense for RGB overlay image
            # So we skip it when overlay is enabled

        else:
            # Display thermal image (original behavior)
            im = ax.imshow(
                img.T,
                origin='lower',
                extent=extent,
                cmap=self.cmap,
                vmin=self.ambient_temp,
                vmax=img.max() if img.max() > self.ambient_temp else self.ambient_temp + 100
            )

            # Add colorbar (only for thermal-only view)
            cbar = plt.colorbar(im, ax=ax, label='Temperature (K)')

        # Add title with simulation info
        sim = context['simulation']
        if hasattr(sim, 'step_context') and sim.step_context:
            ctx = sim.step_context
            layer = ctx.get('build', {}).get('layer', 'N/A')
            track = ctx.get('build', {}).get('track', 'N/A')
            title = f'Perspective Camera View - Step {step}\nLayer {layer}, Track {track}'
        else:
            title = f'Perspective Camera View - Step {step}'

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Save figure
        filename = f"thermal_step_{step:04d}.{self.image_format}"
        filepath = save_path / filename

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def get_latest_image(self) -> Optional[Tuple[np.ndarray, tuple]]:
        """
        Get the most recently rendered image.

        Returns:
            Tuple of (image, extent) or None if no image has been rendered yet.
            - image: (W, H) temperature array
            - extent: (xmin, xmax, ymin, ymax) in meters
        """
        if hasattr(self, 'latest_image'):
            return self.latest_image, self.latest_extent
        return None

    def get_camera(self) -> Optional[FollowingPerspectiveCamera]:
        """
        Get the camera instance for manual control.

        Returns:
            FollowingPerspectiveCamera instance or None if not yet created.
        """
        return self.camera


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
PerspectiveCameraCallback - Following camera with thermal visualization

Usage in simulation:
    from callbacks.perspective_camera_callback import PerspectiveCameraCallback

    # Basic usage - default camera position
    callback = PerspectiveCameraCallback(
        save_images=True,
        interval=10
    )

    # Custom camera angle - look from side and above
    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.05, -0.10, 0.06),  # 5cm right, 10cm behind, 6cm up
        floor_angle_deg=40.0,                   # look down at 40 degrees
        fov_y_deg=60.0,                         # wider field of view
        save_images=True,
        interval=5
    )

    # High-resolution capture
    callback = PerspectiveCameraCallback(
        resolution_wh=(1920, 1080),  # HD resolution
        save_images=True,
        dpi=300,                      # high DPI for publication
        interval=20
    )

Camera position guide:###TODO:Hakim refers to ONSHAPE 
          https://cad.onshape.com/documents/6134144a908f15066ef9ae27/w/792d8ed1e99f0aab1992f8b1/e/3dc108567c4a0c0e8c757863?renderMode=0&uiState=68fb6ecd25013ae0d3fce007
          
    rel_offset_local = (x, y, z) in meters, relative to nozzle:

    X: perpendicular to scan direction
       - Positive = right (when looking along scan direction)
       - Negative = left

    Y: along scan direction
       - Positive = in front of nozzle
       - Negative = behind nozzle (typical for following view)

    Z: vertical
       - Positive = above nozzle
       - Negative = below nozzle

    Examples:
        (0.0, -0.12, 0.04)   - Default: behind and above
        (0.08, -0.08, 0.05)  - Right, behind, and above (angled view)
        (0.0, -0.20, 0.10)   - Far behind and high above (overview)
        (0.0, 0.05, 0.02)    - In front of nozzle (leading view)

Viewing angle:
    floor_angle_deg: angle looking down from horizontal

    - 0°: horizontal (side view)
    - 30°: moderate downward angle (default)
    - 45°: diagonal view
    - 90°: straight down (top view)
""")
