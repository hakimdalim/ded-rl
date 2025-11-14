# HDF5ThermalSaver Callback - Usage Readme



## Quick Start

### 1. Basic Usage in `simulate.py`

```python
from simulate import SimulationRunner
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.completion_callbacks import HeightCompletionCallback

# Create your callbacks
callbacks = [
    HeightCompletionCallback(target_height_mm=10.0),

    # Add HDF5 thermal saver
    HDF5ThermalSaver(
        filename="thermal_fields.h5",  # Output filename
        interval=10,                    # Save every 10 steps
        compression='gzip',             # Compression algorithm
        compression_opts=4              # Compression level (0-9)
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
    experiment_label="_experiment",
    callbacks=callbacks
)

runner.run()
```

### 2. Output Structure

The HDF5 file will be saved to your simulation output directory:

```
_experiments/
  └── my_experiment/
      └── job123.../
          ├── thermal_fields.h5        ← Your HDF5 file
          ├── simulation_params.csv
          └── ... (other outputs)
```

**HDF5 File Structure:**

```
thermal_fields.h5
├── /step_0010/
│   ├── temperature  [nx, ny, nz] array (3D temperature field)
│   └── metadata     (attributes: step, time, position, melt_pool, etc.)
├── /step_0020/
│   ├── temperature
│   └── metadata
└── ... (one group per saved step)
```

## Configuration Options

### Parameters

```python
HDF5ThermalSaver(
    filename="thermal_fields.h5",  # Filename (saved in output_dir)
    interval=1,                     # Save every N steps (default: 1 = every step)
    compression='gzip',             # Compression: 'gzip', 'lzf', or None
    compression_opts=4,             # Compression level 0-9 (for gzip only)
    save_metadata=True              # Save simulation metadata with each step
)
```


## Reading Data

### Load Temperature Field

```python
from callbacks.hdf5_thermal_saver import load_thermal_field, load_thermal_metadata

# Load temperature field from specific step
temp_field = load_thermal_field("thermal_fields.h5", step=10)
print(temp_field.shape)  # (nx, ny, nz)

# Temperature field is a numpy array
max_temp = temp_field.max()
mean_temp = temp_field.mean()
```


## Troubleshooting

### Issue: HDF5 file is too large

**Solution**: Increase `interval` or reduce `compression_opts`:

```python
# Save less frequently
HDF5ThermalSaver(interval=50)  # Instead of interval=1

# Use higher compression (slower but smaller)
HDF5ThermalSaver(compression='gzip', compression_opts=9)
```

### Issue: Simulation is slow when saving

**Solution**: Use faster compression or save less frequently:

```python
# Faster compression
HDF5ThermalSaver(compression='lzf')  # Instead of 'gzip'

# Or save less often
HDF5ThermalSaver(interval=20)
```


```python
# The file will be here:
# _experiments/{experiment_label}/job{timestamp}_{params}/thermal_fields.h5

# You can find the path from simulation output:
print(runner.simulation.output_dir)
```

### Issue: h5py not installed

**Solution**: Install h5py:

```bash
pip install h5py
```




## Related Callbacks

- `FinalStateSaver`: Saves only final temperature field (NPY format)
- `TemperatureSliceSaver`: Saves 2D slices of temperature field (PNG images)
- `ThermalPlotSaver`: Saves temperature plots (PNG images)
- `StepDataCollector`: Saves scalar metrics (CSV format)

