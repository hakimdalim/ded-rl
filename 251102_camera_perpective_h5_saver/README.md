# DED-LB Semi-Analytical Simulation

This repository contains a semi-analytical simulation framework for **Directed Energy Deposition with Laser Beam (DED-LB)** additive manufacturing processes. It simulates the thermal field, melt pool dynamics, powder capture, and build geometry evolution during multi-track, multi-layer deposition.

## Table of Contents
- [Quick Start](#quick-start)
- [Standard Parameters](#standard-parameters)
- [Expected Runtime](#expected-runtime)
- [Output Files](#output-files)
- [Post-Processing](#post-processing)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Prerequisites
```bash
# Install required packages
pip install numpy pandas matplotlib scipy pillow h5py
```

### Basic Run
```bash
# Run with default parameters (5mm x 5mm x 5mm part)
python simulate.py

# Run with custom parameters
python simulate.py --part-x 10.0 --part-y 10.0 --part-z 3.0 --exp-label my_experiment
```

### Minimal Example
```bash
# Small test run (2mm x 2mm x 2mm part, coarse resolution)
python simulate.py --part-x 2.0 --part-y 2.0 --part-z 2.0 --voxel-size 300 --exp-label test_run
```

---

## Standard Parameters

### Default Configuration (from `simulate.py --help`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `--build-x` | 20.0 | mm | Build volume X dimension |
| `--build-y` | 20.0 | mm | Build volume Y dimension |
| `--build-z` | 15.0 | mm | Build volume Z dimension |
| `--part-x` | 5.0 | mm | Part width (X direction) |
| `--part-y` | 5.0 | mm | Part length (Y direction) |
| `--part-z` | 5.0 | mm | Part height (Z direction) |
| `--voxel-size` | 200.0 | μm | Voxel discretization size |
| `--delta-t` | 200.0 | ms | Time step duration |
| `--scan-speed` | 3.0 | mm/s | Laser scan speed |
| `--laser-power` | 600.0 | W | Laser power |
| `--powder-feed` | 2.0 | g/min | Powder feed rate |
| `--exp-label` | "unlabeled" | - | Output directory label |

### Process Strategy (hardcoded defaults)
- **Hatch spacing**: 700 μm (0.7 mm)
- **Layer spacing**: 350 μm (0.35 mm)
- **Substrate height**: 5.0 mm
- **Bidirectional tracks**: Yes
- **Bidirectional layers**: Yes
- **Switch scan direction between layers**: Yes
- **Turnaround time**: 0.1 s

### Material Properties (316L Stainless Steel by default)
- **Melting temperature**: 1811 K
- **Density**: 7850 kg/m³
- **Specific heat**: 460 J/(kg·K)
- **Thermal diffusivity**: 1.172×10⁻⁵ m²/s
- **Absorptivity**: 0.6

---

## Expected Runtime

### Computational Complexity
Runtime depends on:
- **Number of voxels**: `build_volume / voxel_size³`
- **Number of tracks**: `part_width / hatch_spacing`
- **Number of layers**: `part_height / layer_spacing`
- **Number of time steps**: `track_length / (scan_speed × delta_t)`

### Typical Runtimes (Single CPU Core)

| Configuration | Part Size | Voxel Size | Tracks | Steps/Track | Estimated Time* |
|---------------|-----------|------------|--------|-------------|-----------------|
| **Default** | 5×5×5 mm | 200 μm | 7 | 25 | **2-4 hours** |
| **Coarse** | 2×2×2 mm | 300 μm | 3 | 10 | **15-30 min** |
| **Fine** | 5×5×5 mm | 100 μm | 7 | 50 | **10-20 hours** |
| **Large** | 10×10×10 mm | 200 μm | 14 | 50 | **15-30 hours** |

*Estimates based on typical hardware (Intel i7, 16GB RAM). Actual runtime varies by CPU speed and memory.

### Runtime Scaling
- **Linear** with number of tracks and layers
- **~Quadratic** with inverse voxel size (finer voxels = longer runtime)
- Memory usage: ~8 bytes per voxel (for temperature field)

**Example**: Default configuration (5mm³ part, 200μm voxels)
- Voxels: 100×100×75 ≈ **750K voxels**
- Memory: ~6 MB per field (temperature, activation)
- Number of tracks: **7**
- Number of layers: **14** (at 350μm spacing for 5mm height)
- Total steps: **~2500-3500**
- Expected runtime: **2-4 hours**

### Progress Monitoring
The simulation prints progress to console:
```
Layer (count):   1  |  Track (count):   1  |  Height (max): 0.3500 mm
Layer (count):   1  |  Track (count):   2  |  Height (max): 0.3500 mm
...
```

---

## Output Files

The simulation creates a timestamped directory with the following structure:

```
_experiments/unlabeled/job<timestamp>_build20.0x20.0x15.0mm_part5.0x5.0x5.0mm.../
├── simulation_data.csv              # Step-by-step data (melt pool, clad, etc.)
├── parameter_history.csv            # Process parameter history
│
├── temperatures/                    # Temperature field slices
│   ├── xy_slice_step0001.npy       # Horizontal plane (2D array)
│   ├── xz_slice_step0001.npy       # Front view (2D array)
│   └── yz_slice_step0001.npy       # Side view (2D array)
│
├── voxel_temps/                     # Full 3D temperature volumes
│   └── voxel_temps_step0001.npy    # Complete temperature field (3D array)
│
├── build_mesh/                      # 3D geometry meshes
│   └── build_state_step0001.stl    # STL mesh of deposited material
│
├── thermal_plots/                   # Thermal visualization images
│   ├── thermal0001_top_view.png    # XY temperature plot
│   ├── thermal0001_front_view.png  # XZ temperature plot
│   └── thermal0001_side_view.png   # YZ temperature plot
│
├── cross_sections/                  # Final cross-section plots
│   ├── cross_section_y0.5mm.png
│   ├── cross_section_y1.5mm.png
│   └── ...
│
├── final_activated_vol.npy          # Final activated volume (3D boolean)
├── final_temperature_vol.npy        # Final temperature field (3D array)
├── simulation_params.csv            # Summary statistics
│
└── clad_manager.pkl                 # Pickled clad profile manager
```

### Key Output Files

#### 1. `simulation_data.csv` (Primary Data File)
Contains step-by-step simulation data:
- Position (x, y, z)
- Voxel indices
- Layer/track numbers
- Melt pool dimensions (width, length, depth)
- Clad dimensions (width, height, wetting angle)
- Profile parameters (baseline, max_z, etc.)

**Columns**:
```
step, position.x, position.y, position.z, voxel.x, voxel.y, voxel.z,
build.layer, build.track, melt_pool.width, melt_pool.length, melt_pool.depth,
clad.width, clad.height, clad.wetting_angle, ...
```

#### 2. Temperature Fields
- **2D Slices** (`temperatures/*.npy`): Temperature at XY, XZ, YZ planes
- **3D Volumes** (`voxel_temps/*.npy`): Complete temperature field (all voxels)
- Units: Kelvin (K)
- Format: NumPy arrays

#### 3. Build Geometry
- **STL Meshes** (`build_mesh/*.stl`): 3D surface meshes viewable in MeshLab, ParaView
- **Activated Volume** (`final_activated_vol.npy`): Boolean 3D array of deposited material

#### 4. Visualizations
- **Thermal Plots** (`thermal_plots/*.png`): Temperature contours with melt pool boundaries
- **Cross-Sections** (`cross_sections/*.png`): Layer-by-layer geometry at different Y positions

---

## Post-Processing

### 1. Load Simulation Data
```python
import pandas as pd
import numpy as np

# Load step data
data = pd.read_csv('_experiments/.../simulation_data.csv')

# Plot melt pool width over time
import matplotlib.pyplot as plt
plt.plot(data['step'], data['melt_pool.width'] * 1000)  # Convert to mm
plt.xlabel('Step')
plt.ylabel('Melt Pool Width (mm)')
plt.show()
```

### 2. Load Temperature Fields
```python
# Load 2D temperature slice
temp_xy = np.load('temperatures/xy_slice_step0100.npy')

# Load 3D temperature volume
temp_3d = np.load('voxel_temps/voxel_temps_step0100.npy')

# Visualize
plt.imshow(temp_xy.T, origin='lower', cmap='hot', vmin=300, vmax=2500)
plt.colorbar(label='Temperature (K)')
plt.show()
```

### 3. Load Geometry
```python
# Load final activated volume
activated = np.load('final_activated_vol.npy')

# Count deposited voxels
num_voxels = activated.sum()
voxel_size = 200e-6  # 200 μm in meters
volume_m3 = num_voxels * voxel_size**3
print(f"Deposited volume: {volume_m3 * 1e9:.2f} mm³")

# Visualize with matplotlib
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(activated, edgecolor='k')
plt.show()
```

### 4. Analyze Melt Pool Statistics
```python
import pandas as pd

data = pd.read_csv('simulation_data.csv')

# Calculate average melt pool dimensions
avg_width = data['melt_pool.width'].mean() * 1000  # mm
avg_depth = data['melt_pool.depth'].mean() * 1000  # mm

# Calculate aspect ratio
data['aspect_ratio'] = data['melt_pool.width'] / data['melt_pool.depth']

print(f"Average melt pool width: {avg_width:.3f} mm")
print(f"Average melt pool depth: {avg_depth:.3f} mm")
print(f"Average aspect ratio: {data['aspect_ratio'].mean():.2f}")
```

### 5. Extract Clad Profile Data
```python
import pickle

# Load clad manager
with open('clad_manager.pkl', 'rb') as f:
    clad_manager = pickle.load(f)

# Access stored profiles (dict keyed by (layer, track, y_pos))
profiles = clad_manager._profiles

# Example: Get profile for layer 0, track 0, at y=0.0025m
profile = profiles.get((0, 0, 0.0025))
if profile:
    print(f"Profile width: {profile.width * 1000:.3f} mm")
    print(f"Profile height: {profile.height * 1000:.3f} mm")
```

### 6. View STL Meshes
```bash
# Install MeshLab (free open-source tool)
# Open STL files in MeshLab for 3D visualization

# Or use Python
import trimesh
mesh = trimesh.load('build_mesh/build_state_step0100.stl')
mesh.show()
```

---

## Advanced Usage

### Custom Callbacks
You can customize what data is saved by modifying the callback system:

```python
from simulate import SimulationRunner
from callbacks.callback_collection import (
    StepDataCollector,
    ProgressPrinter,
    ThermalPlotSaver
)
from callbacks.completion_callbacks import HeightCompletionCallback

# Define custom callback set (saves only essential data)
callbacks = [
    HeightCompletionCallback(),
    StepDataCollector(save_path="data.csv"),
    ProgressPrinter(),
    ThermalPlotSaver(interval=10),  # Save plots every 10 steps
]

# Create runner with custom callbacks
runner = SimulationRunner.from_human_units(
    part_volume_mm=(10.0, 10.0, 5.0),
    voxel_size_um=200.0,
    laser_power_W=800.0,
    scan_speed_mm_s=4.0,
    experiment_label="my_custom_run",
    callbacks=callbacks
)

# Run simulation
runner.run()
```

### Programmatic Access
```python
from simulate import SimulationRunner

# Create runner
runner = SimulationRunner.from_human_units(
    part_volume_mm=(5.0, 5.0, 2.0),
    voxel_size_um=200.0,
    delta_t_ms=200.0,
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    powder_feed_g_min=2.0,
    experiment_label="programmatic_run"
)

# Access configuration
print(runner.config.print_summary())

# Run simulation
runner.run()

# Access results after completion
final_temp = runner.simulation.temperature_tracker.temperature
final_activated = runner.simulation.volume_tracker.activated
```

### Parameter Sweeps
```python
# Run parameter sweep (laser power)
for power in [400, 600, 800, 1000]:
    runner = SimulationRunner.from_human_units(
        part_volume_mm=(3.0, 3.0, 2.0),
        laser_power_W=power,
        experiment_label=f"power_sweep/power_{power}W"
    )
    runner.run()
```

---

## Common Use Cases

### 1. Quick Test (5-10 minutes)
```bash
python simulate.py --part-x 2.0 --part-y 2.0 --part-z 1.0 --voxel-size 300 --exp-label quick_test
```

### 2. Standard Quality Run (2-4 hours)
```bash
python simulate.py --part-x 5.0 --part-y 5.0 --part-z 5.0 --voxel-size 200 --exp-label standard_run
```

### 3. High Resolution (12-24 hours)
```bash
python simulate.py --part-x 5.0 --part-y 5.0 --part-z 5.0 --voxel-size 100 --exp-label high_res
```

### 4. Process Optimization Study
```bash
python simulate.py --laser-power 500 --scan-speed 2.5 --exp-label low_energy
python simulate.py --laser-power 700 --scan-speed 3.5 --exp-label high_energy
```

---

## Troubleshooting

### Memory Issues
- **Reduce voxel size**: Use larger voxels (e.g., 300 μm instead of 200 μm)
- **Reduce build volume**: Smaller build volume = fewer voxels
- **Reduce save frequency**: Use `interval=10` in callbacks to save less frequently

### Slow Runtime
- **Increase delta_t**: Larger time steps (e.g., 300 ms instead of 200 ms)
- **Reduce part size**: Smaller parts = fewer tracks/layers
- **Disable heavy callbacks**: Turn off `VoxelTemperatureSaver` or `MeshSaver`

### Disk Space Issues
- Output size for default run: **~500 MB - 2 GB**
- Large runs can generate **10+ GB** of data
- Solution: Enable compression callbacks or save only essential data

---

## Citation

If you use this simulation framework, please cite:

```
[Insert appropriate citation here based on your research paper]
```

---

## Contact

For questions or issues, please contact [your contact information].
