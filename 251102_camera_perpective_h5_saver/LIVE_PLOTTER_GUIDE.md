# Live Plotter Guide

Real-time visualization during DED simulation runs.

## Overview

There are **two live plotters** available:

1. **`LivePlotter`** - Simple, lightweight build height tracking
2. **`AdvancedLivePlotter`** - Full thermal visualization with temperature fields and cross-sections

---

## 1. Simple LivePlotter (Lightweight)

### What It Shows
- **Single plot**: Build height vs time step
- **Updates**: Every N steps (configurable)
- **Performance**: Very fast, minimal overhead

### Basic Usage

```python
from simulate import SimulationRunner
from callbacks.callback_collection import LivePlotter, get_default_callbacks

# Get default callbacks
callbacks = get_default_callbacks()

# Add LivePlotter
callbacks.append(LivePlotter(interval=10))  # Update every 10 steps

# Create and run simulation
runner = SimulationRunner.from_human_units(
    part_volume_mm=(5.0, 5.0, 2.0),
    voxel_size_um=200.0,
    experiment_label="live_demo",
    callbacks=callbacks
)

runner.run()
```

### Command Line Usage

Unfortunately, `LivePlotter` is not enabled by default in `simulate.py`. You need to use a Python script:

**Create `run_with_live_plot.py`:**
```python
from simulate import SimulationRunner
from callbacks.callback_collection import (
    HeightCompletionCallback,
    StepDataCollector,
    ProgressPrinter,
    LivePlotter  # Import the simple live plotter
)

# Define callbacks with live plotter
callbacks = [
    HeightCompletionCallback(),
    StepDataCollector(save_path="simulation_data.csv"),
    ProgressPrinter(),
    LivePlotter(interval=5),  # Update every 5 steps
]

# Run simulation
runner = SimulationRunner.from_human_units(
    part_volume_mm=(5.0, 5.0, 2.0),
    voxel_size_um=200.0,
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    experiment_label="live_plot_demo",
    callbacks=callbacks
)

runner.run()
```

Then run:
```bash
python run_with_live_plot.py
```

### Configuration Options

```python
LivePlotter(
    interval=10  # Update every 10 steps (default: 10)
)
```

**`interval`**: How often to update the plot
- Lower = more frequent updates, slower simulation
- Higher = less frequent updates, faster simulation
- Recommended: 5-20 for quick runs, 50-100 for long runs

### Output Example

```
┌─────────────────────────────────────┐
│  Build Progress                     │
│  Max Height (mm)                    │
│    2.0 ┤                          ╭─┤
│        │                      ╭───╯  │
│    1.5 ┤                  ╭───╯      │
│        │              ╭───╯          │
│    1.0 ┤          ╭───╯              │
│        │      ╭───╯                  │
│    0.5 ┤  ╭───╯                      │
│        │──╯                          │
│    0.0 └──────────────────────────────
│         0   500  1000  1500  2000    │
│              Step                    │
└─────────────────────────────────────┘
```

### When to Use Simple LivePlotter
- ✓ Quick monitoring of simulation progress
- ✓ Checking if build height is increasing correctly
- ✓ Long simulations where you want minimal overhead
- ✓ Debugging layer completion logic
- ✗ Need to see thermal fields (use Advanced instead)
- ✗ Need detailed visualization (use Advanced instead)

---

## 2. Advanced LivePlotter (Full Visualization)

### What It Shows
- **4 real-time plots**:
  1. **Top view (XY)**: Temperature field at current Z height
  2. **Front view (XZ)**: Temperature field at current Y position
  3. **Side view (YZ)**: Temperature field at current X position
  4. **Cross-section**: All layer profiles at current Y position

- **Overlays**:
  - White dashed lines: Activated volume (deposited material)
  - Cyan lines: Melt pool boundary (melting temperature contour)

- **Updates**: Every N steps with efficient in-place data updates

### Basic Usage

```python
from simulate import SimulationRunner
from callbacks.live_plotter_callback import AdvancedLivePlotter
from callbacks.callback_collection import (
    HeightCompletionCallback,
    StepDataCollector,
    ProgressPrinter
)

# Define callbacks with advanced live plotter
callbacks = [
    HeightCompletionCallback(),
    StepDataCollector(save_path="simulation_data.csv"),
    ProgressPrinter(),
    AdvancedLivePlotter(
        interval=5,              # Update every 5 steps
        temp_range=(300, 2500),  # Temperature range (K)
        figsize=(20, 12),        # Large window
        enabled=True
    )
]

# Run simulation
runner = SimulationRunner.from_human_units(
    part_volume_mm=(5.0, 5.0, 2.0),
    voxel_size_um=200.0,
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    experiment_label="advanced_live_demo",
    callbacks=callbacks
)

runner.run()
```

### Full Example Script

**Create `run_with_advanced_live_plot.py`:**

```python
"""
DED Simulation with Advanced Live Visualization
Shows real-time temperature fields and build progress.
"""

from simulate import SimulationRunner
from callbacks.live_plotter_callback import AdvancedLivePlotter
from callbacks.callback_collection import (
    HeightCompletionCallback,
    StepDataCollector,
    ProgressPrinter
)

def main():
    print("Starting DED simulation with advanced live visualization...")

    # Configure callbacks
    callbacks = [
        # Completion condition
        HeightCompletionCallback(),

        # Data collection
        StepDataCollector(save_path="simulation_data.csv"),

        # Console progress
        ProgressPrinter(),

        # ADVANCED LIVE PLOTTER
        AdvancedLivePlotter(
            interval=1,              # Update every step (can be slow!)
            temp_range=(300, 2500),  # Temperature colormap range (K)
            figsize=(20, 12),        # Window size (inches)
            enabled=True             # Enable/disable easily
        )
    ]

    # Create simulation runner
    runner = SimulationRunner.from_human_units(
        # Part dimensions
        part_volume_mm=(5.0, 5.0, 2.0),

        # Resolution (use coarser for faster live updates)
        voxel_size_um=200.0,
        delta_t_ms=200.0,

        # Process parameters
        laser_power_W=600.0,
        scan_speed_mm_s=3.0,
        powder_feed_g_min=2.0,

        # Output
        experiment_label="advanced_live_visualization",

        # Attach callbacks
        callbacks=callbacks
    )

    print("\nLive plot window will appear...")
    print("Close the plot window to stop simulation (or let it complete)")
    print("\nRunning simulation...\n")

    # Run simulation
    runner.run()

    print("\nSimulation complete!")
    print(f"Results saved to: {runner.simulation.output_dir}")

if __name__ == "__main__":
    main()
```

Then run:
```bash
python run_with_advanced_live_plot.py
```

### Configuration Options

```python
AdvancedLivePlotter(
    interval=1,                  # Update frequency (steps)
    temp_range=(300, 2500),      # Temperature colormap range (K)
    figsize=(20, 12),            # Figure size (width, height) in inches
    enabled=True                 # Enable/disable toggle
)
```

#### Parameter Details

**`interval`** (int): Update frequency
- `1`: Update every step (very slow, but real-time)
- `5`: Update every 5 steps (good balance)
- `10-20`: Update periodically (faster, still informative)
- Recommendation:
  - Quick tests (5-10 min): `interval=1`
  - Standard runs (1-2 hours): `interval=5`
  - Long runs (10+ hours): `interval=20`

**`temp_range`** (tuple): Temperature colormap limits
- `(300, 2500)`: Default, shows ambient to well above melting
- `(1700, 2500)`: Focus on melt pool (melting temp ~1800K for 316L)
- `(300, 1000)`: Focus on heat-affected zone
- Adjust based on your material properties

**`figsize`** (tuple): Window size in inches
- `(20, 12)`: Large display (recommended for dual monitors)
- `(16, 10)`: Medium display
- `(12, 8)`: Laptop screen
- Larger = easier to see details, but may not fit screen

**`enabled`** (bool): Quick toggle
- `True`: Live plotting enabled
- `False`: Disabled (no overhead, useful for production runs)

### Visual Layout

```
┌────────────────────────────────────────────────────────────────┐
│  Top View (z=1.2mm) - Step 234    │ Front View (y=2.5mm)      │
│  ┌─────────────────┐               │ ┌──────────────────┐      │
│  │     [Hot]       │               │ │                  │      │
│  │      ███        │  XY plane     │ │       ██         │  XZ  │
│  │     █████       │  (current Z)  │ │      ████        │      │
│  │    ███████      │               │ │     ██████       │      │
│  │   ─────────     │ ← Activation  │ │    ────────      │      │
│  │   - - - - -     │ ← Melt pool   │ │   - - - - -      │      │
│  └─────────────────┘               │ └──────────────────┘      │
│                                    │                           │
│  Side View (x=1.8mm)               │                           │
│  ┌──────────────────┐              │                           │
│  │                  │  YZ plane    │                           │
│  │       ██         │              │                           │
│  │      ████        │              │                           │
│  │     ██████       │              │                           │
│  │    ────────      │              │                           │
│  └──────────────────┘              │                           │
├────────────────────────────────────────────────────────────────┤
│  Layer Cross Sections (y=2.5mm)                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Z                                                         │  │
│  │ 2 ┤                           ╭─────╮                     │  │
│  │   │                       ╭───╯Layer3╰───╮               │  │
│  │ 1 ┤                   ╭───╯Layer 2    ╰───╮             │  │
│  │   │               ╭───╯Layer 1            ╰───╮         │  │
│  │ 0 ┤───────────────╯Layer 0 (substrate)────────╰────────  │  │
│  │   └────────────────────────────────────────────────────  │  │
│  │         X (mm)                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### What Each View Shows

1. **Top View (XY plane)**
   - Temperature field at current laser position's Z height
   - Shows melt pool moving along scan path
   - White dashed contour = activated volume boundary
   - Cyan contour = melt pool boundary (melting temp)

2. **Front View (XZ plane)**
   - Temperature field vertical slice at current Y position
   - Shows depth of melt pool
   - Shows heat penetration into substrate/previous layers
   - Layer stacking visible

3. **Side View (YZ plane)**
   - Temperature field vertical slice at current X position
   - Shows longitudinal melt pool shape
   - Useful for seeing layer height evolution

4. **Cross-Section Plot**
   - All layer profiles at current Y position
   - Each layer in different color
   - Shows build geometry evolution
   - Legend indicates layer numbers

### When to Use Advanced LivePlotter

✓ **Good for:**
- Development and debugging
- Understanding thermal behavior
- Monitoring melt pool dynamics
- Checking layer stacking
- Teaching/demonstrations
- Short to medium runs (<2 hours)
- Publications (screenshot the live window)

✗ **Avoid for:**
- Very long production runs (>10 hours)
- High-resolution simulations (slow updates)
- Batch processing multiple cases
- Cluster/HPC jobs without display

---

## Performance Comparison

| Plotter | Update Time/Step | Memory Overhead | CPU Overhead | Best For |
|---------|------------------|-----------------|--------------|----------|
| **Simple** | ~10 ms | ~1 MB | <1% | Long runs, monitoring only |
| **Advanced** | ~100-500 ms | ~50 MB | 5-20% | Development, visualization |
| **None** | 0 ms | 0 MB | 0% | Production, batch runs |

### Impact on Simulation Runtime

**Example**: 5mm × 5mm × 2mm part, 200μm voxels, ~1000 steps

| Configuration | Total Runtime | Overhead |
|---------------|---------------|----------|
| No live plotting | 60 min | - |
| Simple LivePlotter (interval=10) | 61 min | +1.7% |
| Advanced LivePlotter (interval=10) | 65 min | +8.3% |
| Advanced LivePlotter (interval=1) | 90 min | +50% |

**Recommendation**: Use `interval=5-10` for Advanced LivePlotter to balance visualization and performance.

---

## Tips & Best Practices

### 1. Choosing Update Interval

```python
# For quick tests (5-15 min runtime)
AdvancedLivePlotter(interval=1)  # Update every step

# For standard runs (1-2 hours)
AdvancedLivePlotter(interval=5)  # Update every 5 steps

# For long runs (>5 hours)
AdvancedLivePlotter(interval=20)  # Update every 20 steps
```

### 2. Reduce Overhead for Large Simulations

```python
# High-resolution simulation + Advanced plotter = SLOW
# Solution: Update less frequently

AdvancedLivePlotter(
    interval=50,  # Update every 50 steps instead of every step
    enabled=True
)
```

### 3. Disable for Production Runs

```python
# Toggle on/off easily
USE_LIVE_PLOT = False  # Set to False for production

callbacks = [
    # ... other callbacks
    AdvancedLivePlotter(
        interval=5,
        enabled=USE_LIVE_PLOT  # Controlled by flag
    )
]
```

### 4. Interactive Mode Requirements

Both plotters require an **interactive matplotlib backend**. If you get errors:

```python
# Add to top of your script
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### 5. Headless Systems (No Display)

If running on HPC cluster or server without display:

```bash
# Don't use live plotters!
# Or use X11 forwarding (slow)
ssh -X user@server
```

### 6. Save Live Plot Screenshots

The Advanced LivePlotter shows real-time data. To save specific frames:

```python
# Manually save current view (during run, via interactive console)
# Or add to callback:

import matplotlib.pyplot as plt

class AdvancedLivePlotterWithSave(AdvancedLivePlotter):
    def _execute(self, context):
        super()._execute(context)

        # Save every 100 steps
        if context['simulation'].progress_tracker.step_count % 100 == 0:
            step = context['simulation'].progress_tracker.step_count
            self.fig.savefig(f'live_plot_step{step:04d}.png', dpi=150)
```

---

## Troubleshooting

### "Matplotlib is currently using agg, which is a non-GUI backend"

**Fix**: Set interactive backend before importing pyplot

```python
import matplotlib
matplotlib.use('TkAgg')  # Try: TkAgg, Qt5Agg, Qt4Agg
import matplotlib.pyplot as plt

# Then import simulation modules
from simulate import SimulationRunner
from callbacks.live_plotter_callback import AdvancedLivePlotter
```

### Plot Window Not Updating

**Cause**: Event loop blocked

**Fix**: Ensure you're not blocking the main thread. The plotter uses `plt.pause()` and `canvas.flush_events()` internally.

### "Figure is too large to fit screen"

**Fix**: Reduce `figsize`

```python
AdvancedLivePlotter(figsize=(12, 8))  # Smaller window
```

### Simulation Very Slow with Live Plotter

**Fix 1**: Increase `interval`
```python
AdvancedLivePlotter(interval=20)  # Update less often
```

**Fix 2**: Use Simple LivePlotter instead
```python
LivePlotter(interval=10)  # Much faster
```

**Fix 3**: Disable for this run
```python
AdvancedLivePlotter(enabled=False)  # No overhead
```

### Memory Issues

Advanced LivePlotter stores contour collections. For very long runs:

**Fix**: Clear old contours more aggressively (already implemented in code)

Or use Simple LivePlotter which has minimal memory footprint.

---

## Complete Example: Development Workflow

**Scenario**: Developing new process parameters, want to see what's happening

```python
"""
Development workflow with full visualization.
"""

from simulate import SimulationRunner
from callbacks.live_plotter_callback import AdvancedLivePlotter
from callbacks.callback_collection import (
    HeightCompletionCallback,
    StepDataCollector,
    ProgressPrinter,
    ThermalPlotSaver
)

# Development flags
LIVE_VISUALIZATION = True
SAVE_THERMAL_PLOTS = True

# Callbacks
callbacks = [
    HeightCompletionCallback(),
    StepDataCollector(save_path="data.csv"),
    ProgressPrinter(),
]

# Add live visualization for development
if LIVE_VISUALIZATION:
    callbacks.append(
        AdvancedLivePlotter(
            interval=5,
            temp_range=(300, 2500),
            figsize=(20, 12),
            enabled=True
        )
    )

# Also save static thermal plots periodically
if SAVE_THERMAL_PLOTS:
    callbacks.append(
        ThermalPlotSaver(interval=10)
    )

# Run simulation (quick test parameters)
runner = SimulationRunner.from_human_units(
    part_volume_mm=(3.0, 3.0, 1.5),  # Small part
    voxel_size_um=200.0,              # Medium resolution
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    experiment_label="dev_visualization",
    callbacks=callbacks
)

print("Starting simulation with live visualization...")
print("Watch the plot window for real-time thermal fields!\n")

runner.run()
```

---

## Summary

| Feature | Simple LivePlotter | Advanced LivePlotter |
|---------|-------------------|---------------------|
| **What it shows** | Build height vs time | 4 plots: thermal fields + cross-sections |
| **Update speed** | ~10 ms/update | ~100-500 ms/update |
| **Overhead** | <1% | 5-20% |
| **Memory** | ~1 MB | ~50 MB |
| **Best for** | Long runs, progress monitoring | Development, understanding physics |
| **Difficulty** | Easy | Easy |
| **Configuration** | `interval` only | `interval`, `temp_range`, `figsize` |

**Recommendation**:
- **Development**: Use `AdvancedLivePlotter` with `interval=5`
- **Production**: Disable live plotting, use saved thermal plots instead
- **Teaching**: Use `AdvancedLivePlotter` with `interval=1` for real-time demo

Enjoy real-time visualization of your DED simulations! 🔥
