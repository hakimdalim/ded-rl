# HDF5 Visualization Notebook - User Guide

## Overview

`visualize_hdf5_results.ipynb` is a comprehensive Jupyter notebook for visualizing thermal and activation data saved in HDF5 format from DED simulations. It leverages existing visualization utilities from the repository.

## Features

✓ **Load and Inspect HDF5 Data**
  - Thermal fields from `hdf5_thermal_saver`
  - Activation volumes from `hdf5_activation_saver`

✓ **Thermal Field Visualization** (using `ThermalPlotter`)
  - Multi-plane temperature views (XY, XZ, YZ)
  - Melt pool boundary detection
  - Temperature evolution over time

✓ **Activation Volume Visualization** (using existing utilities)
  - 2D slices using `plot_slice` and `plot_surface`
  - 3D interactive views using `VoxelVisualizer`
  - Activation growth tracking

✓ **Combined Analysis**
  - Overlay thermal and activation data
  - Track solidification patterns

✓ **Export Capabilities**
  - High-resolution figures for publications
  - Batch export all visualizations

## Quick Start

### 1. Prerequisites

Ensure you have the required packages installed:

```bash
pip install h5py numpy matplotlib plotly jupyter
```

### 2. Run a Simulation with HDF5 Callbacks

First, generate HDF5 data using the HDF5 savers:

```python
from simulate import SimulationRunner
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.hdf5_activation_saver import HDF5ActivationSaver
from callbacks.completion_callbacks import HeightCompletionCallback

callbacks = [
    HeightCompletionCallback(target_height_mm=10.0),

    # Save thermal fields
    HDF5ThermalSaver(
        filename="thermal_fields.h5",
        interval=10,  # Save every 10 steps
        compression='gzip',
        compression_opts=4
    ),

    # Save activation volumes
    HDF5ActivationSaver(
        filename="activation_volumes.h5",
        interval=10,
        compression='gzip',
        compression_opts=9
    )
]

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

This will create:
```
_experiments/
  └── my_experiment/
      └── job_xxxxx/
          ├── thermal_fields.h5        ← Your thermal data
          ├── activation_volumes.h5    ← Your activation data
          └── ... (other outputs)
```

### 3. Launch Jupyter Notebook

```bash
cd path/to/hypo-simulations
jupyter notebook visualize_hdf5_results.ipynb
```

### 4. Configure the Notebook

In **Section 2** of the notebook, update the file paths:

```python
# Update these paths to your actual HDF5 files
THERMAL_HDF5 = "_experiments/my_experiment/job_xxxxx/thermal_fields.h5"
ACTIVATION_HDF5 = "_experiments/my_experiment/job_xxxxx/activation_volumes.h5"
```

### 5. Run the Cells

Execute the cells in order:
1. **Section 1**: Imports and setup
2. **Section 2**: Configure paths
3. **Section 3**: Inspect files (verify data loaded correctly)
4. **Sections 4-8**: Generate visualizations

## Notebook Sections

### Section 1: Setup and Imports
- Loads all required libraries
- Imports HDF5 utilities and visualization tools
- Configures matplotlib settings

### Section 2: Configuration
- **IMPORTANT**: Update your file paths here
- Set material properties (melting temperature, etc.)
- Choose visualization settings (colormap, temperature range)

### Section 3: Inspect HDF5 Files
- View file contents and metadata
- Check available timesteps
- Verify data shapes and ranges

### Section 4: Thermal Field Visualization
- **4.1**: Multi-plane views of temperature fields
  - Uses `ThermalPlotter` from `utils/field_visualization.py`
  - Shows XY, XZ, YZ slices
  - Highlights melt pool boundary

- **4.2**: Temperature evolution over time
  - Max/mean temperature plots
  - Layer progression tracking

### Section 5: Activation Volume Visualization
- **5.1**: 2D slices using `plot_slice` and `plot_surface`
  - From `utils/visualization_utils.py`
  - XY, XZ, YZ activation slices
  - Top surface height map

- **5.2**: 3D interactive visualization
  - Uses `VoxelVisualizer` from `voxel/voxel_visualize.py`
  - Interactive Plotly visualization
  - Rotate and zoom in browser

- **5.3**: Activation growth over time
  - Uses `get_activation_statistics`
  - Tracks build progression

### Section 6: Combined Thermal + Activation
- Overlay temperature and activation
- See relationship between thermal field and solidification

### Section 7: Export Visualizations
- Save high-resolution figures (300 DPI)
- Batch export all visualizations
- Creates `visualizations/` directory with PNG files

### Section 8: Summary and Statistics
- Comprehensive summary of simulation results
- Final statistics and metadata

## Visualization Examples

### Thermal Field (Section 4.1)
```
XY Plane          XZ Plane          YZ Plane
┌────────┐       ┌────────┐       ┌────────┐
│        │       │        │       │        │
│  🔥    │       │  🔥    │       │  🔥    │
│        │       │        │       │        │
└────────┘       └────────┘       └────────┘
```
Shows temperature distribution with melt pool boundary (red contour)

### Activation Volume (Section 5.1)
```
XY Slice         XZ Slice         YZ Slice         Surface
┌────────┐       ┌────────┐       ┌────────┐       ┌────────┐
│███     │       │███     │       │███     │       │▲▲▲▲    │
│███     │       │███     │       │███     │       │▲▲▲▲    │
│████████│       │████████│       │████████│       │████████│
└────────┘       └────────┘       └────────┘       └────────┘
substrate        substrate        substrate        height map
```

### 3D Visualization (Section 5.2)
Interactive 3D view with:
- Substrate shown in blue
- Deposited material in red
- Rotate, zoom, pan
- Hover for coordinates

## Customization Options

### Change Colormap
In Section 2:
```python
COLORMAP = 'hot'     # For thermal: 'hot', 'inferno', 'plasma', 'viridis'
```

### Adjust Temperature Range
```python
TEMP_RANGE = (300.0, 2500.0)  # (min, max) in Kelvin
```

### Sample Fewer Timesteps
In Section 4.2:
```python
fig = plot_temperature_evolution(thermal_path, steps_to_sample=20)  # Sample only 20 steps
```

### Choose Specific Timestep
```python
# Instead of middle step
step_to_plot = thermal_steps[0]      # First step
step_to_plot = thermal_steps[-1]     # Last step
step_to_plot = 100                    # Specific step number
```

## Using Existing Code

The notebook leverages these existing utilities:

### From `utils/field_visualization.py`
```python
from utils.field_visualization import ThermalPlotter, MeltPoolDimensions

plotter = ThermalPlotter(melting_temp=1700, cmap='hot')
plotter.plot_thermal_plane(ax, grid1, grid2, temperatures, ...)
```

### From `utils/visualization_utils.py`
```python
from utils.visualization_utils import plot_slice, plot_surface

plot_slice(activation, index=10, axis='z', ax=ax)
plot_surface(activation, axis='z', from_top=True, ax=ax)
```

### From `voxel/voxel_visualize.py`
```python
from voxel.voxel_visualize import VoxelVisualizer

viz = VoxelVisualizer(shape=activation.shape, voxel_size=voxel_size)
fig = viz.create_figure(activated=activation, substrate_nz=5)
fig.show()
```

### From HDF5 Savers
```python
from callbacks.hdf5_thermal_saver import load_thermal_field, load_thermal_metadata
from callbacks.hdf5_activation_saver import load_activation_volume, get_activation_statistics

temp_field = load_thermal_field("thermal_fields.h5", step=10)
activation = load_activation_volume("activation_volumes.h5", step=10)
```

## Troubleshooting

### Issue: "File not found"
**Solution**: Update paths in Section 2 to point to your actual HDF5 files.
```python
# Check your actual path - it should look like this:
THERMAL_HDF5 = "_experiments/test_exp/job1234567890_build20.0x20.0x15.0mm_part5.0x5.0x2.0mm_vox200.0um_dt200.0ms_v3.0mms_p600.0W_f2.0gmin/thermal_fields.h5"
```

### Issue: "h5py not installed"
**Solution**: Install required packages
```bash
pip install h5py matplotlib plotly jupyter
```

### Issue: "Module not found: utils.field_visualization"
**Solution**: Ensure you're running the notebook from the repository root directory
```bash
cd /path/to/hypo-simulations
jupyter notebook
```

### Issue: Plotly visualization not showing
**Solution**: Make sure you're using Jupyter Notebook (not JupyterLab) or install extensions:
```bash
# For JupyterLab
jupyter labextension install jupyterlab-plotly
```

### Issue: Out of memory when loading large files
**Solution**: Sample fewer timesteps or visualize specific steps instead of all
```python
# Sample only 10 timesteps instead of all
fig = plot_temperature_evolution(thermal_path, steps_to_sample=10)
```

## Tips for Best Results

1. **Start with inspection** (Section 3) to verify data before visualization
2. **Use appropriate intervals** when saving HDF5 (interval=10 is usually good)
3. **Export figures** (Section 7) for high-quality publication-ready images
4. **Try different colormaps** - 'hot' for thermal, 'viridis' for general
5. **Interactive 3D** works best in Jupyter Notebook (not JupyterLab without extensions)

## Creating Animations

To create animations from multiple timesteps, you can extend the notebook:

```python
import matplotlib.animation as animation

def create_thermal_animation(filepath, output_file='thermal_animation.mp4'):
    steps = list_thermal_steps(str(filepath))

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(step):
        ax.clear()
        temp_field = load_thermal_field(str(filepath), step=step)
        mid_z = temp_field.shape[2] // 2
        im = ax.imshow(temp_field[:, :, mid_z].T, origin='lower',
                      cmap='hot', vmin=300, vmax=2500)
        ax.set_title(f'Step {step}')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=steps,
                                   interval=200, blit=True)
    anim.save(output_file, writer='ffmpeg', fps=5)

# Usage
create_thermal_animation(thermal_path)
```

## Next Steps

After using this notebook, you can:

1. **Compare multiple runs** - Load HDF5 files from different experiments
2. **Export to ParaView** - Convert to VTK format for advanced 3D visualization
3. **Machine Learning** - Use HDF5 data as training datasets
4. **Quantitative Analysis** - Extract cooling rates, thermal gradients, etc.

## Related Documentation

- `callbacks/README_HDF5_THERMAL_SAVER.md` - Thermal saver usage guide
- `callbacks/README_HDF5_ACTIVATION_SAVER.md` - Activation saver usage guide
- `test_callbacks/test_hdf5_saver.py` - Test examples for thermal data
- `test_callbacks/test_hdf5_activation_saver.py` - Test examples for activation data

## Questions?

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the related documentation
3. Look at the test scripts in `test_callbacks/`
4. Verify your HDF5 files were created correctly by running the test scripts

---

**Happy Visualizing! 🎨📊**
