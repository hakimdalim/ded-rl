# How to Run Video Production Simulation

## Prerequisites

Make sure you're in the correct conda environment where the simulation works:

```bash
# Activate your simulation environment
conda activate ded_featreps_env

# OR if it's a different environment, activate that one
# conda activate <your_env_name>
```

## Quick Run

From the root directory of the project:

```bash
python run_video_production.py
```

This will:
1. Run a simulation with all three new callbacks
2. Generate camera frames for video (every 2 steps)
3. Save HDF5 thermal and activation data (every 10 steps)
4. Export STL of final geometry
5. Create a VIDEO_PRODUCTION_GUIDE.md with FFmpeg commands

## Expected Output

The simulation will create a directory like:
```
_experiments/video_prod/job.../
├── cam/                          # Camera frames for video
│   ├── thermal_step_0002.png
│   ├── thermal_step_0004.png
│   └── ... (30-50 frames)
├── thermal_plots/                # Original thermal plots
│   ├── thermal0010_top_view.png
│   ├── thermal0010_front_view.png
│   └── ...
├── thermal_fields.h5             # Complete thermal history
├── activation_volumes.h5         # Complete activation history
├── build_geometry.stl            # Final geometry (for Blender/ParaView)
├── simulation_data.csv           # Time series data
├── parameter_history.csv         # Process parameters
└── VIDEO_PRODUCTION_GUIDE.md     # Guide with FFmpeg commands
```

## Estimated Time

- Simulation: 2-3 minutes
- Post-processing: 10-20 seconds
- **Total**: ~3 minutes

## Creating Video

After the simulation completes, follow the instructions in:
```
_experiments/video_prod/job.../VIDEO_PRODUCTION_GUIDE.md
```

Quick video creation:
```bash
cd _experiments/video_prod/job.../cam
ffmpeg -framerate 10 -i thermal_step_%04d.png -c:v libx264 -pix_fmt yuv420p build_process.mp4
```

## Troubleshooting

### Error: "No module named 'numba'"

Install numba in your environment:
```bash
conda install numba
# OR
pip install numba
```

### Error: "No module named 'h5py'"

Install h5py:
```bash
pip install h5py
```

### Error: STL export fails

Install required packages:
```bash
pip install scikit-image numpy-stl
```

## Alternative: Run Tests Instead

If the full simulation has issues, you can use the test files which are shorter:

```bash
# Test HDF5 thermal saver (creates thermal_fields.h5)
python test_callbacks/test_hdf5_saver.py

# Test HDF5 activation saver (creates activation_volumes.h5)
python test_callbacks/test_hdf5_activation_saver.py

# Test perspective camera (creates cam/ images)
python test_callbacks/test_perspective_camera.py
```

These tests run faster (20-30 seconds each) and produce similar outputs in smaller directories.

## Manual Verification

After running, check the following:

1. **Camera frames exist**:
   ```bash
   ls _experiments/video_prod/job.../cam/
   ```
   Should see: `thermal_step_0002.png`, `thermal_step_0004.png`, etc.

2. **HDF5 files exist**:
   ```bash
   ls _experiments/video_prod/job.../*.h5
   ```
   Should see: `thermal_fields.h5`, `activation_volumes.h5`

3. **STL file exists**:
   ```bash
   ls _experiments/video_prod/job.../*.stl
   ```
   Should see: `build_geometry.stl`

4. **Check file sizes**:
   ```bash
   du -h _experiments/video_prod/job.../
   ```

## Next Steps

1. Review outputs in the experiment directory
2. Read `VIDEO_PRODUCTION_GUIDE.md` for video creation commands
3. Use HDF5 files for custom visualization
4. Import STL into Blender for 3D animation

## Questions?

- Check the README files in `callbacks/`:
  - `README_HDF5_THERMAL_SAVER.md`
  - `README_HDF5_ACTIVATION_SAVER.md`
  - `README_PERSPECTIVE_CAMERA.md`
- Check `COMPARISON_BEFORE_AFTER.md` for details on outputs
