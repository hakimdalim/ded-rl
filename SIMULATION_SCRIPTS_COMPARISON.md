# Simulation Scripts Comparison

This document explains the differences between the simulation scripts in this repository and when to use each one.

## Overview

We have **three main simulation scripts**:

| Script | Purpose | Callbacks | Use Case |
|--------|---------|-----------|----------|
| `simulate.py` | **ORIGINAL** | Default callbacks only | Standard simulations, testing |
| `_simulate.py` | **ENHANCED** | Default + NEW callbacks | Local runs with full features |
| `testing/simulate_with_all_callbacks.py` | **SLURM** | Default + NEW callbacks | HPC batch jobs |

---

## 1. `simulate.py` - Original Script

### Location
```
git/hypo-simulations/simulate.py
```

### What it does
- Uses **ONLY** the original/default callbacks
- Provides the baseline functionality
- Kept unchanged for backward compatibility

### Callbacks included
- ✓ HeightCompletionCallback
- ✓ ProgressPrinter
- ✓ StepDataCollector
- ✓ FinalStateSaver
- ✓ ThermalPlotSaver
- ✓ CrossSectionPlotter
- ✓ PickleSaver
- ✓ And other default callbacks from `get_default_callbacks()`

### Outputs
```
_experiments/unlabeled/job.../
├── simulation_data.csv
├── thermal_plots/
├── cross_sections/
├── clad_manager.pkl
└── final_*.npy
```

### Usage
```bash
# Basic run with defaults
python simulate.py

# With custom parameters
python simulate.py --part-x 10.0 --part-z 8.0 --laser-power 800 --exp-label my_test
```

### When to use
- ✓ Quick tests
- ✓ When you only need basic outputs
- ✓ Validating against original behavior
- ✓ When HDF5 and camera outputs are not needed

---

## 2. `_simulate.py` - Enhanced Script

### Location
```
git/hypo-simulations/_simulate.py
```

### What it does
- Uses **ALL** callbacks: original + new custom callbacks
- Provides complete data capture (thermal history, camera views)
- Identical structure to `simulate.py` with one function added

### Difference from `simulate.py`
```python
# simulate.py line 423
callbacks = get_default_callbacks()  # Original callbacks only

# _simulate.py line 423 (equivalent)
callbacks = get_enhanced_callbacks()  # Original + NEW callbacks
```

### NEW function added
```python
def get_enhanced_callbacks():
    """Combines original callbacks + HDF5 + camera callbacks."""
    callbacks = get_default_callbacks()
    callbacks.append(HDF5ThermalSaver(...))
    callbacks.append(HDF5ActivationSaver(...))
    callbacks.append(PerspectiveCameraCallback(...))
    return callbacks
```

### Additional callbacks
- ✨ **HDF5ThermalSaver** - Compressed thermal field history
- ✨ **HDF5ActivationSaver** - Compressed activation volume history
- ✨ **PerspectiveCameraCallback** - Following camera with nozzle overlay

### Outputs (includes all original + new)
```
_experiments/unlabeled/job.../
├── simulation_data.csv          ✓ Original
├── thermal_plots/               ✓ Original
├── cross_sections/              ✓ Original
├── clad_manager.pkl             ✓ Original
├── final_*.npy                  ✓ Original
├── thermal_fields.h5            ✨ NEW
├── activation_volumes.h5        ✨ NEW
└── cam/                         ✨ NEW
    ├── thermal_step_0005.png
    ├── thermal_step_0010.png
    └── ...
```

### Usage
```bash
# Basic run with defaults (includes all callbacks)
python _simulate.py

# With custom parameters
python _simulate.py --part-x 10.0 --part-z 8.0 --laser-power 800 --exp-label enhanced_test
```

### When to use
- ✓ Local development and testing
- ✓ When you need complete thermal history
- ✓ When you want camera visualizations
- ✓ Machine learning data collection
- ✓ Publication-quality outputs

---

## 3. `testing/simulate_with_all_callbacks.py` - SLURM Script

### Location
```
git/hypo-simulations/testing/simulate_with_all_callbacks.py
```

### What it does
- **Same callbacks as `_simulate.py`** (original + new)
- **Optimized for SLURM batch job execution**
- Called by SLURM submission scripts
- Includes detailed logging for job monitoring

### Difference from `_simulate.py`
- Designed for automated batch execution
- Better error reporting for remote jobs
- Configurable save intervals via `--save-interval`
- More verbose output for SLURM logs

### Callbacks included
- Same as `_simulate.py`: Original + HDF5 + Camera

### Usage
```bash
# Called by SLURM submission scripts
python testing/simulate_with_all_callbacks.py \
    --part-x 5.0 --part-y 5.0 --part-z 2.0 \
    --laser-power 800 --scan-speed 5.0 \
    --save-interval 5 \
    --exp-label ded_doe_v6
```

### When to use
- ✓ SLURM job submissions
- ✓ Parameter sweeps (DoE experiments)
- ✓ Automated batch processing
- ✓ HPC cluster runs

### SLURM integration
This script is referenced in your SLURM submission script:
```bash
# In your SLURM job script
python -u testing/simulate_with_all_callbacks.py ${args}
```

---

## Quick Reference

### I want to...

**...run a quick test locally**
```bash
python simulate.py --part-z 2.0 --exp-label quick_test
```

**...get complete outputs (HDF5 + camera) locally**
```bash
python _simulate.py --part-z 2.0 --exp-label full_test
```

**...submit SLURM batch jobs**
```bash
# Edit your SLURM script to use:
python -u testing/simulate_with_all_callbacks.py ${args}
```

**...compare original vs enhanced**
```bash
# Run both and compare outputs
python simulate.py --exp-label original_run
python _simulate.py --exp-label enhanced_run
```

---

## File Size Comparison

| Script | Typical Output Size | Time Overhead |
|--------|--------------------:|--------------|
| `simulate.py` | ~15-25 MB | Baseline |
| `_simulate.py` | ~30-40 MB | +10-20% |
| `testing/simulate_with_all_callbacks.py` | ~30-40 MB | +10-20% |

The additional ~15-20 MB comes from:
- HDF5 thermal fields: ~15 MB (compressed)
- HDF5 activation volumes: ~0.03 MB (highly compressed)
- Camera images: ~0.4 MB

---

## Code Comparison

### simulate.py (line 423)
```python
callbacks = get_default_callbacks()
```

### _simulate.py (line 423 equivalent)
```python
callbacks = get_enhanced_callbacks()  # Defined in this file
```

### testing/simulate_with_all_callbacks.py
```python
callbacks = create_all_callbacks(save_interval=args.save_interval)
```

---

## Summary

| Feature | `simulate.py` | `_simulate.py` | `testing/...` |
|---------|:-------------:|:--------------:|:-------------:|
| Original callbacks | ✓ | ✓ | ✓ |
| HDF5 thermal saver | ✗ | ✓ | ✓ |
| HDF5 activation saver | ✗ | ✓ | ✓ |
| Camera callback | ✗ | ✓ | ✓ |
| Command-line args | ✓ | ✓ | ✓ |
| SLURM optimized | - | - | ✓ |
| Local development | ✓ | ✓ | ✓ |
| Batch jobs | ✓ | ✓ | ✓✓ |

---

## Recommendation

- **Keep `simulate.py` unchanged** (backward compatibility)
- **Use `_simulate.py`** for local development with full features
- **Use `testing/simulate_with_all_callbacks.py`** for SLURM batch jobs
- All three scripts accept the same command-line arguments
- All three scripts produce compatible outputs (enhanced versions have additional files)

---

## Troubleshooting

### Camera callback fails with "FileNotFoundError"

**Problem**: Directory doesn't exist before matplotlib tries to save
```
FileNotFoundError: '...\\cam_imgs\\thermal_step_0004.png'
```

**Solution**: The callback already has `self.ensure_dir(save_path)` on line 564. Make sure:
1. You're using `save_dir="cam"` not `save_dir="cam_imgs"`
2. The callback is imported correctly
3. The output directory has write permissions

### HDF5 import error

**Problem**: `ImportError: No module named 'h5py'`

**Solution**: Install h5py
```bash
pip install h5py
# or
conda install h5py
```

### Comparison between scripts

To verify all three scripts produce identical outputs (except for new files):
```bash
# Run all three with same parameters
python simulate.py --exp-label test_original --part-z 2.0
python _simulate.py --exp-label test_enhanced --part-z 2.0
python testing/simulate_with_all_callbacks.py --exp-label test_slurm --part-z 2.0

# Compare CSV files (should be identical)
diff _experiments/test_original/job.../simulation_data.csv \
     _experiments/test_enhanced/job.../simulation_data.csv
```

Expected: CSV files identical, enhanced versions have additional HDF5 and cam/ outputs.

---

## Next Steps

1. Test `_simulate.py` locally first
2. Verify all outputs are created correctly
3. Update SLURM submission scripts to use `testing/simulate_with_all_callbacks.py`
4. Run parameter sweep using SLURM

---

**Created**: 2025-11-05
**Purpose**: Document the three simulation script variants and usage guidelines
