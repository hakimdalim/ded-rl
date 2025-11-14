# PerspectiveCameraCallback - Usage Guide

## Overview

`PerspectiveCameraCallback` creates a perspective camera that automatically follows the nozzle/laser during simulation and optionally saves thermal images. The camera provides a realistic view with depth perception, similar to a real camera mounted on the print head.

## Key Features

- **Automatic nozzle tracking**: Camera follows the laser/nozzle as it moves
- **Perspective projection**: Realistic view with depth (closer objects appear larger)
- **Configurable position**: Set camera location relative to nozzle
- **Adjustable viewing angle**: Control downward tilt angle
- **Thermal visualization**: Renders temperature field with hot colormap
- **Optional image saving**: Save images at specified intervals

## Quick Start

### 1. Basic Usage in `simulate.py`

```python
from simulate import SimulationRunner
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.completion_callbacks import HeightCompletionCallback

# Create callbacks
callbacks = [
    HeightCompletionCallback(target_height_mm=10.0),

    # Add perspective camera (default: behind and above nozzle)
    PerspectiveCameraCallback(
        save_images=True,
        interval=10  # Save every 10 steps
    )
]

# Run simulation
runner = SimulationRunner.from_human_units(
    build_volume_mm=(50.0, 50.0, 30.0),
    part_volume_mm=(20.0, 20.0, 10.0),
    voxel_size_um=200.0,
    delta_t_ms=100.0,
    scan_speed_mm_s=5.0,
    laser_power_W=800.0,
    powder_feed_g_min=3.0,
    experiment_label="my_experiment",
    callbacks=callbacks
)

runner.run()
```

### 2. Output Structure

Camera images are saved to a subdirectory in your simulation output:

```
_experiments/
  └── my_experiment/
      └── job123.../
          ├── cam/                     ← Camera images directory
          │   ├── thermal_step_0010.png
          │   ├── thermal_step_0020.png
          │   └── ...
          ├── simulation_params.csv
          └── ... (other outputs)
```

## Camera Configuration

### Position and Orientation

The camera position is specified **relative to the nozzle** in the nozzle's local coordinate frame:

- **X**: Perpendicular to scan direction (right when looking forward)
- **Y**: Along scan direction (negative = behind nozzle)
- **Z**: Vertical (up)

### Common Camera Positions

```python
# Default: Behind and above (typical following view)
PerspectiveCameraCallback(
    rel_offset_local=(0.0, -0.12, 0.04),  # 12cm behind, 4cm above
    floor_angle_deg=30.0                   # Look down at 30 degrees
)

# High overview: Far behind and high up
PerspectiveCameraCallback(
    rel_offset_local=(0.0, -0.20, 0.10),  # 20cm behind, 10cm above
    floor_angle_deg=45.0                   # Look down at 45 degrees
)

# Side angle view: Right side, behind, and above
PerspectiveCameraCallback(
    rel_offset_local=(0.08, -0.08, 0.05),  # 8cm right, 8cm behind, 5cm above
    floor_angle_deg=20.0                    # Slight downward tilt
)

# Leading view: In front of nozzle (see where it's going)
PerspectiveCameraCallback(
    rel_offset_local=(0.0, 0.05, 0.02),  # 5cm ahead, 2cm above
    floor_angle_deg=15.0                  # Slight downward view
)

# Top-down view: Directly above
PerspectiveCameraCallback(
    rel_offset_local=(0.0, 0.0, 0.15),  # 15cm directly above
    floor_angle_deg=90.0                 # Look straight down
)
```

### All Configuration Parameters

```python
PerspectiveCameraCallback(
    # Camera position and orientation
    rel_offset_local=(0.0, -0.12, 0.04),  # Position relative to nozzle (m)
    floor_angle_deg=30.0,                  # Downward viewing angle (degrees)
    fov_y_deg=45.0,                        # Vertical field of view (degrees)
    up_hint=(0.0, 0.0, 1.0),              # World up vector

    # Image plane and resolution
    plane_size=(0.06, 0.04),               # Image plane size in meters (w, h)
    resolution_wh=(800, 600),              # Output resolution in pixels (w, h)
    pixel_size_xy=(0.001, 0.001),         # Pixel spacing in meters (alternative)

    # Rendering settings
    ambient_temp=300.0,                    # Background temperature (K)
    cmap='hot',                            # Colormap ('hot', 'inferno', 'viridis', etc.)

    # Saving settings
    save_images=True,                      # Enable/disable image saving
    save_dir="cam",                        # Output subdirectory name
    image_format="png",                    # Image format (png, jpg, etc.)
    dpi=150,                              # Image resolution (dots per inch)

    # Callback settings
    interval=10                            # Save every N steps
)
```

## Parameter Guide

### Camera Position (`rel_offset_local`)

**Format**: `(x, y, z)` in meters, relative to nozzle

- **X** (left/right): Positive = right, Negative = left
- **Y** (forward/back): Positive = in front, Negative = behind
- **Z** (up/down): Positive = above, Negative = below

**Typical ranges**:
- X: -0.10 to +0.10 m (±10cm from centerline)
- Y: -0.20 to +0.05 m (20cm behind to 5cm ahead)
- Z: 0.02 to 0.15 m (2cm to 15cm above)

### Viewing Angle (`floor_angle_deg`)

Controls how much the camera tilts downward:

- **0°**: Horizontal (side view)
- **30°**: Moderate downward angle (default, good for following)
- **45°**: Diagonal view
- **60°**: Steep downward view
- **90°**: Straight down (top view)

### Field of View (`fov_y_deg`)

Vertical field of view in degrees:

- **30°**: Narrow (telephoto lens, zoomed in)
- **45°**: Standard (default, balanced view)
- **60°**: Wide (wide-angle lens, more context)
- **90°**: Very wide (fish-eye effect)

### Resolution (`resolution_wh`)

Output image size in pixels:

| Setting | Resolution | Use Case |
|---------|------------|----------|
| `(640, 480)` | VGA | Fast rendering, small files |
| `(800, 600)` | SVGA | Good balance (default) |
| `(1280, 720)` | HD | High quality |
| `(1920, 1080)` | Full HD | Publication quality |

**Note**: Higher resolution = slower rendering and larger files

### Save Interval (`interval`)

How often to save images:

```python
interval=1     # Every step (many images, slow)
interval=10    # Every 10 steps (recommended)
interval=50    # Every 50 steps (fewer images, faster)
interval=100   # Every 100 steps (minimal storage)
```

**Storage estimate** (800×600 resolution):
- File size per image: ~50-200 KB (depending on content)
- 1000 steps with `interval=10`: ~100 images = 5-20 MB
- 1000 steps with `interval=1`: ~1000 images = 50-200 MB

## Advanced Usage

### Multiple Camera Angles

You can add multiple cameras with different viewpoints:

```python
callbacks = [
    HeightCompletionCallback(target_height_mm=10.0),

    # Camera 1: Following view (behind and above)
    PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        save_dir="cam_follow",
        interval=20
    ),

    # Camera 2: Side view (right side)
    PerspectiveCameraCallback(
        rel_offset_local=(0.10, -0.05, 0.05),
        floor_angle_deg=25.0,
        save_dir="cam_side",
        interval=20
    ),

    # Camera 3: Top-down overview
    PerspectiveCameraCallback(
        rel_offset_local=(0.0, 0.0, 0.15),
        floor_angle_deg=90.0,
        fov_y_deg=60.0,  # Wider FOV for overview
        save_dir="cam_top",
        interval=50  # Less frequent for overview
    )
]
```

### Access Camera During Simulation

You can access the camera object for manual control or to get the latest image:

```python
# After simulation setup
camera_callback = callbacks[1]  # Assuming camera is second callback

# Get camera instance
camera = camera_callback.get_camera()

# Manually update camera position (advanced use)
if camera:
    camera.set_position([0.05, 0.05, 0.08])
    camera.look_at([0.0, 0.0, 0.0])

# Get latest rendered image
latest = camera_callback.get_latest_image()
if latest:
    img, extent = latest
    print(f"Image shape: {img.shape}")
    print(f"Temp range: {img.min():.1f} - {img.max():.1f} K")
```

### Camera Without Saving (Live Monitoring)

If you only want to create the camera for live monitoring without saving images:

```python
PerspectiveCameraCallback(
    save_images=False,  # Don't save to disk
    interval=1           # Update every step
)
```

Then access the latest image programmatically:

```python
img, extent = camera_callback.get_latest_image()
# Use img for live visualization, analysis, etc.
```

### Custom Colormaps

Use different colormaps for visualization:

```python
# Hot (default - good for thermal)
PerspectiveCameraCallback(cmap='hot')

# Inferno (perceptually uniform)
PerspectiveCameraCallback(cmap='inferno')

# Viridis (colorblind-friendly)
PerspectiveCameraCallback(cmap='viridis')

# Jet (classic rainbow - not recommended)
PerspectiveCameraCallback(cmap='jet')

# Custom range
PerspectiveCameraCallback(
    cmap='hot',
    ambient_temp=273.15  # 0°C background
)
```

## Comparison: Perspective vs Orthographic Camera

| Feature | Perspective (this callback) | Orthographic |
|---------|----------------------------|--------------|
| **Projection** | Converging rays (realistic) | Parallel rays (no depth) |
| **Depth perception** | Yes (closer = larger) | No (size constant) |
| **Use case** | Realistic view, following | Technical drawings, measurements |
| **Field of view** | Adjustable (FOV) | Fixed plane size |
| **Appearance** | Like a real camera | Like CAD software |

**Use perspective camera when:**
- You want realistic visualization
- Depth perception is important
- Creating videos or presentations
- Simulating real camera systems

**Use orthographic camera when:**
- You need precise measurements
- Size comparison across depth
- Technical documentation
- CAD-like views

## Troubleshooting

### Issue: Images are too dark

**Solution**: Increase the temperature range or change colormap:

```python
PerspectiveCameraCallback(
    cmap='inferno',  # Brighter colormap
    ambient_temp=273.15  # Lower background temp (0°C)
)
```

### Issue: Camera is too close/far

**Solution**: Adjust the offset distance:

```python
# Too close - move camera back
PerspectiveCameraCallback(
    rel_offset_local=(0.0, -0.20, 0.06)  # Farther back and higher
)

# Too far - move camera closer
PerspectiveCameraCallback(
    rel_offset_local=(0.0, -0.08, 0.03)  # Closer
)
```

### Issue: Can't see the melt pool

**Solution**: Adjust viewing angle or position:

```python
PerspectiveCameraCallback(
    rel_offset_local=(0.02, -0.10, 0.05),  # Slightly to the side
    floor_angle_deg=35.0,  # Steeper angle
    fov_y_deg=50.0  # Wider view
)
```

### Issue: Images are saving too slowly

**Solution**: Reduce resolution or increase interval:

```python
PerspectiveCameraCallback(
    resolution_wh=(640, 480),  # Lower resolution
    interval=20,  # Save less frequently
    dpi=100  # Lower DPI
)
```

### Issue: Path too long error on Windows

**Solution**: Use shorter directory name or experiment label:

```python
# Use shorter save directory
PerspectiveCameraCallback(
    save_dir="cam"  # Instead of "camera_images"
)

# Or use shorter experiment label
runner = SimulationRunner.from_human_units(
    experiment_label="exp1",  # Instead of long descriptive name
    ...
)
```

### Issue: Camera not following nozzle

**Check**: Make sure simulation has step context with position data. This callback requires:
- `sim.step_context['position']` to be available
- Simulation is running (not just initialized)

## Performance Tips

1. **Optimize interval**: Don't save every step unless necessary
   ```python
   interval=20  # Good balance for most cases
   ```

2. **Choose appropriate resolution**: Higher isn't always better
   ```python
   resolution_wh=(800, 600)  # Usually sufficient
   ```

3. **Use efficient format**: PNG is good balance of quality/size
   ```python
   image_format="png"  # Good compression, lossless
   ```

4. **Limit active cameras**: Each camera adds overhead
   ```python
   # Use 1-2 cameras max for real-time performance
   ```

## Example Workflows

### 1. Process Monitoring

Monitor the build process from a following camera:

```python
PerspectiveCameraCallback(
    rel_offset_local=(0.0, -0.15, 0.05),
    floor_angle_deg=30.0,
    interval=10,
    resolution_wh=(1280, 720),
    save_dir="monitoring"
)
```

### 2. Video Creation

High-quality images for video production:

```python
PerspectiveCameraCallback(
    rel_offset_local=(0.0, -0.12, 0.04),
    floor_angle_deg=30.0,
    interval=1,  # Every frame for smooth video
    resolution_wh=(1920, 1080),
    dpi=300,
    save_dir="video_frames"
)
```

### 3. Quality Inspection

Angled view for defect detection:

```python
PerspectiveCameraCallback(
    rel_offset_local=(0.05, -0.10, 0.06),
    floor_angle_deg=35.0,
    fov_y_deg=55.0,
    interval=5,
    cmap='inferno',  # Better contrast
    save_dir="inspection"
)
```

### 4. Multi-Angle Documentation

Three camera setup for comprehensive documentation:

```python
callbacks = [
    # Camera 1: Standard following view
    PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        save_dir="cam_follow",
        interval=20
    ),

    # Camera 2: Side diagnostic view
    PerspectiveCameraCallback(
        rel_offset_local=(0.08, -0.08, 0.05),
        floor_angle_deg=25.0,
        save_dir="cam_side",
        interval=20
    ),

    # Camera 3: Overview for context
    PerspectiveCameraCallback(
        rel_offset_local=(0.0, 0.0, 0.15),
        floor_angle_deg=90.0,
        fov_y_deg=60.0,
        save_dir="cam_overview",
        interval=50
    )
]
```

## Related Callbacks

- **OrthographicCameraCallback**: Parallel projection (no depth)
- **ThermalPlotSaver**: 2D slice visualizations
- **HDF5ThermalSaver**: Save complete 3D temperature volumes

## Next Steps

After using the perspective camera callback:
1. Create videos from saved images using ffmpeg
2. Analyze thermal patterns across multiple angles
3. Compare different process parameters visually
4. Generate documentation with representative frames

---

**Questions or issues?** Check `test_callbacks/test_perspective_camera.py` for working examples.
