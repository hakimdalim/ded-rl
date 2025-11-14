# What Does PerspectiveCameraCallback Capture?

## TL;DR

**YES, the perspective camera captures the DED process in real-time!**

It creates **realistic thermal camera views** that follow the nozzle as it deposits material, showing:
- ✅ Temperature field (hot melt pool, cooling tracks)
- ✅ Build geometry (deposited layers, substrate)
- ✅ Process dynamics (laser movement, layer-by-layer growth)
- ✅ Optional: Nozzle outline + powder stream overlay

---

## What It Captures

### 1. **Thermal Field Visualization**

```
┌─────────────────────────────────────────────────┐
│  Perspective Camera View - Step 450             │
│  Following Nozzle: Layer 3, Track 5             │
│                                                  │
│         [Hot melt pool moving →]                │
│              ███████                             │
│            ██████████████                        │
│          ████████████████████                    │
│        ████  MELT POOL  █████                   │
│          █████████████████████                   │
│            ██████████████                        │
│         ─────█████──────  (cooling track)       │
│      ─────────────────────────                  │
│   ────────────────────────────────── (previous) │
│  ───────────────────────────────────── (layer)  │
│ ──────────────────────────[Substrate]────────── │
│                                                  │
│  Color = Temperature (300K → 2500K)              │
│  Hot = White/Yellow, Cool = Dark Red/Black      │
└─────────────────────────────────────────────────┘
```

**What you see**:
- Bright white/yellow spot = **Active melt pool** (>1800K)
- Orange/red streaks = **Cooling tracks** (1500-1800K)
- Dark red = **Solidified material** (500-1000K)
- Black = **Substrate and ambient** (~300K)

### 2. **Camera Following the Nozzle**

The camera moves with the laser/nozzle:

```
Step 1: Camera behind nozzle     Step 100: Track 1 complete
   ↓                                 ↓
   [CAM]                            [CAM]
     ↓                                ↓
   START ──→ ███                    ───────────███ END


Step 250: Layer 2                 Step 500: Multiple layers
   ↓                                 ↓
   [CAM]                            [CAM]
     ↓                                ↓
   ═══════════                      ║ Layer 3
   ───────────                      ║ Layer 2
   ───────────                      ║ Layer 1
                                    ╚═════════════
```

**Camera automatically**:
- Tracks nozzle position
- Maintains fixed distance/angle
- Shows build progress over time

### 3. **Perspective View (Realistic Depth)**

Unlike flat 2D plots, the perspective camera shows **realistic 3D depth**:

```
   Close objects (nozzle/melt pool) = LARGE
              ████████
            ██████████████

   Medium distance (previous track) = MEDIUM
         ──────────────

   Far objects (early tracks) = SMALL
      ────────────
```

This makes it look like a **real camera mounted on the print head**.

### 4. **Optional Nozzle + Powder Overlay**

With `enable_overlay=True`, it also shows:

```
┌─────────────────────────────────────────┐
│                                         │
│          ┌─────────┐                    │
│          │ NOZZLE  │ (dark blue cone)   │
│          │   ▼     │                    │
│          └────┬────┘                    │
│              ·│·                        │
│             · │ ·                       │
│            ·  │  ·                      │
│           ·   │   · (white powder       │
│          ·    │    ·  particles)        │
│         ·     │     ·                   │
│        ·      ▼      ·                  │
│  ─────────[MELT POOL]───────            │
│  ────────────────────────────           │
│                                         │
└─────────────────────────────────────────┘
```

Shows:
- **Nozzle geometry** (frustum cone, semi-transparent blue)
- **Powder stream** (white particles in V-shape)
- **Melt pool** (thermal field underneath)

---

## What It Does vs What It Doesn't

### ✅ What PerspectiveCameraCallback DOES

1. **Captures thermal field** at every step (or interval)
   - Shows temperature distribution in 3D view
   - Uses colormap (hot = bright, cold = dark)
   - Projects 3D voxel data to 2D image

2. **Follows the laser/nozzle automatically**
   - Camera moves with nozzle
   - Maintains configurable offset (e.g., 12cm behind, 4cm above)
   - Viewing angle adjustable (e.g., 30° downward tilt)

3. **Creates realistic perspective images**
   - Like a real camera (depth perception)
   - Closer objects appear larger
   - FOV adjustable (wide angle vs telephoto)

4. **Saves images at intervals**
   - Every step, every 10 steps, etc.
   - PNG format by default
   - Configurable resolution (640×480 to 1920×1080+)

5. **Optional schematic overlay** (OUR ADDITION)
   - Nozzle outline
   - Powder stream visualization
   - Helps understand process geometry

### ❌ What It DOESN'T Do

1. **Doesn't modify the simulation**
   - Pure visualization/monitoring
   - No effect on physics calculations
   - Can be disabled with no impact

2. **Doesn't capture actual photos**
   - Renders from simulation data
   - Not a webcam/real camera
   - Shows computed temperature field

3. **Doesn't require real camera hardware**
   - Software-only rendering
   - Uses simulation's temperature data
   - Virtual camera system

4. **Not used by default**
   - Must be explicitly added to callbacks
   - Standard simulate.py doesn't include it
   - OUR enhancement to the original repo

---

## How It Works (Technical)

```
┌──────────────────────────────────────────────────────────┐
│  SIMULATION STEP                                         │
│  ├─ Calculate laser position                             │
│  ├─ Update temperature field                             │
│  ├─ Update activated volume                              │
│  └─ Trigger STEP_COMPLETE event                          │
│                                                           │
│         ↓                                                 │
│                                                           │
│  PERSPECTIVE CAMERA CALLBACK                              │
│  ├─ 1. Get nozzle position from simulation               │
│  ├─ 2. Calculate camera position                         │
│  │     (offset from nozzle in local frame)               │
│  ├─ 3. Get temperature field (3D voxels)                 │
│  ├─ 4. Project 3D voxels to 2D image plane               │
│  │     (perspective transformation)                       │
│  ├─ 5. Render temperature with colormap                  │
│  ├─ 6. [Optional] Overlay nozzle + powder                │
│  └─ 7. Save image to disk                                │
│                                                           │
│         ↓                                                 │
│                                                           │
│  OUTPUT: thermal_step_0450.png                           │
└──────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **Simulation computes physics**:
   - Laser heats material → temperature field updates
   - Powder deposited → clad geometry grows
   - Time advances → simulation state changes

2. **Camera callback triggered** (every N steps):
   - Reads current nozzle position
   - Calculates camera position relative to nozzle
   - Extracts temperature data from 3D field

3. **3D → 2D projection**:
   - Uses perspective projection matrix
   - Transforms 3D voxel positions to 2D pixels
   - Applies depth-based scaling (closer = larger)

4. **Rendering**:
   - Maps temperature values to colors (hot colormap)
   - Composites voxels into image
   - Optionally overlays nozzle/powder graphics

5. **Image saved**:
   - PNG file written to `cam/` directory
   - Filename includes step number
   - Can be later compiled into video

---

## Example Output Sequence

**Step 10**: Start of first track
```
thermal_step_0010.png
  - Small bright spot (melt pool just started)
  - Mostly dark (substrate, no heat yet)
  - Camera behind nozzle looking forward
```

**Step 100**: First track complete
```
thermal_step_0100.png
  - Bright spot at end of track (current melt pool)
  - Red/orange line (cooling track behind)
  - Substrate visible around edges
```

**Step 500**: Multiple tracks, Layer 1
```
thermal_step_0500.png
  - Bright spot moving along track 5
  - Orange/red parallel lines (previous tracks)
  - Darker tracks from earlier (cooled down)
  - 3D depth visible (earlier tracks farther away)
```

**Step 2000**: Multiple layers
```
thermal_step_2000.png
  - Current bright melt pool on Layer 4
  - Previous layers visible below
  - Layering effect shows build height
  - Perspective shows depth clearly
```

---

## Comparison with Standard Outputs

| Output | What It Shows | PerspectiveCamera Equivalent |
|--------|---------------|------------------------------|
| **ThermalPlotSaver** | 2D slice (XY, XZ, YZ planes) | 3D perspective view |
| **CSV data** | Numbers (temp, position) | Visual thermal field |
| **Cross-sections** | Final layer profiles | Real-time build progress |
| **VoxelTemperatureSaver** | Raw 3D data (.npy) | Rendered 2D image |

**Key difference**: PerspectiveCamera creates **realistic visual representations** for:
- Understanding process dynamics visually
- Creating videos/animations
- Presentations and demonstrations
- Qualitative analysis

---

## Usage Examples

### Basic: Add to Simulation

```python
from callbacks.perspective_camera_callback import PerspectiveCameraCallback

callbacks = [
    # ... other callbacks ...

    # Add perspective camera
    PerspectiveCameraCallback(
        save_images=True,
        interval=10  # Save every 10 steps
    )
]
```

### With Nozzle + Powder Overlay

```python
PerspectiveCameraCallback(
    enable_overlay=True,           # Show nozzle + powder
    save_images=True,
    interval=5,
    overlay_config={
        'num_particles': 600,       # Powder particle count
        'particle_color': (255, 255, 255),  # White
        'particle_size_px': 8,
        'nozzle_fill_color': (30, 60, 100),  # Dark blue
    }
)
```

### Multiple Camera Angles

```python
callbacks = [
    # Camera 1: Following behind
    PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        save_dir="cam_follow",
        interval=10
    ),

    # Camera 2: Side view
    PerspectiveCameraCallback(
        rel_offset_local=(0.08, -0.08, 0.05),
        floor_angle_deg=25.0,
        save_dir="cam_side",
        interval=10
    ),

    # Camera 3: Top-down
    PerspectiveCameraCallback(
        rel_offset_local=(0.0, 0.0, 0.15),
        floor_angle_deg=90.0,
        save_dir="cam_top",
        interval=50
    )
]
```

---

## Real-World Analogy

Think of PerspectiveCameraCallback as:

**"A thermal camera mounted on a robot arm that follows the print head"**

- 📹 **Thermal camera**: Shows temperature (not regular RGB)
- 🤖 **Following mount**: Moves with nozzle automatically
- 🎥 **Perspective lens**: Realistic depth (like real camera)
- 💾 **Recording**: Saves images at intervals
- 🎨 **Optional overlay**: Adds schematic graphics (nozzle/powder)

It lets you **watch the DED process** as if you had a camera there, but showing temperature instead of visible light!

---

## Summary

**Q: Does the perspective camera capture the DED process?**

**A: YES! It captures:**
1. ✅ Real-time thermal field (melt pool, cooling tracks, layers)
2. ✅ 3D build geometry (perspective view with depth)
3. ✅ Process dynamics (laser movement, layer-by-layer)
4. ✅ Optional nozzle + powder visualization

**Output**: Sequence of PNG images showing the build process from a realistic camera perspective, saved at configurable intervals throughout the simulation.

**Not included in default run** - must be explicitly added as a callback!
