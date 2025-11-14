"""
Video Production Simulation

This script runs a complete simulation with all three new callbacks
optimized for video production and visualization.

Outputs:
- Thermal field evolution (HDF5)
- Activation volume evolution (HDF5)
- Perspective camera images for video frames
- Original thermal plots for comparison
- STL export of final geometry
- Complete metadata

Run: python run_video_production.py
"""

import sys
from pathlib import Path
import numpy as np

# Import simulation
from simulate import SimulationRunner

# Import ORIGINAL callbacks
from callbacks.completion_callbacks import HeightCompletionCallback, StepCountCompletionCallback
from callbacks.callback_collection import (
    ProgressPrinter,
    ParameterLogger,
    FinalStateSaver,
    ThermalPlotSaver
)
from callbacks.step_data_collector import StepDataCollector

# Import NEW custom callbacks
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver, load_thermal_field, load_thermal_metadata
from callbacks.hdf5_activation_saver import HDF5ActivationSaver, load_activation_volume, get_activation_statistics
from callbacks.perspective_camera_callback import PerspectiveCameraCallback


def export_to_stl(activation_volume, output_path, voxel_size):
    """
    Export activation volume to STL file.

    Args:
        activation_volume: Boolean array of activated voxels
        output_path: Path to save STL file
        voxel_size: Size of each voxel (meters)
    """
    try:
        from skimage import measure
        from stl import mesh

        print(f"\nExporting to STL...")

        # Use marching cubes to create mesh
        verts, faces, normals, values = measure.marching_cubes(
            activation_volume.astype(float),
            level=0.5
        )

        # Scale vertices to real dimensions
        verts = verts * voxel_size

        # Create STL mesh
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts[face[j]]

        # Save to file
        stl_mesh.save(str(output_path))

        # Get mesh statistics
        num_vertices = len(verts)
        num_faces = len(faces)
        volume_mm3 = stl_mesh.get_mass_properties()[0] * 1e9  # Convert m³ to mm³

        print(f"✓ STL export successful:")
        print(f"  File: {output_path}")
        print(f"  Vertices: {num_vertices:,}")
        print(f"  Faces: {num_faces:,}")
        print(f"  Volume: {volume_mm3:.2f} mm³")

        return True

    except ImportError as e:
        print(f"✗ STL export requires scikit-image and numpy-stl")
        print(f"  Install with: pip install scikit-image numpy-stl")
        return False
    except Exception as e:
        print(f"✗ STL export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_video_frames_summary(output_dir):
    """Create summary document for video production."""
    summary_file = output_dir / "VIDEO_PRODUCTION_GUIDE.md"

    # Find all camera images
    cam_dir = output_dir / "cam"
    if cam_dir.exists():
        cam_images = sorted(cam_dir.glob("thermal_step_*.png"))
        num_frames = len(cam_images)
    else:
        cam_images = []
        num_frames = 0

    # Find thermal plots
    thermal_dir = output_dir / "thermal_plots"
    if thermal_dir.exists():
        thermal_images = sorted(thermal_dir.glob("thermal*_top_view.png"))
        num_thermal = len(thermal_images)
    else:
        thermal_images = []
        num_thermal = 0

    content = f"""# Video Production Guide

## Output Directory
`{output_dir}`

## Available Assets

### 1. Camera Frames for Video
**Location**: `cam/`
**Count**: {num_frames} frames
**Format**: PNG (800x600 px)
**Naming**: `thermal_step_XXXX.png`

**FFmpeg command for video**:
```bash
cd cam
ffmpeg -framerate 10 -pattern_type glob -i 'thermal_step_*.png' \\
       -c:v libx264 -pix_fmt yuv420p -crf 18 \\
       build_process.mp4
```

### 2. Thermal Evolution Plots
**Location**: `thermal_plots/`
**Count**: {num_thermal} timesteps (3 views each)
**Views**: top_view, front_view, side_view

**FFmpeg command for thermal video**:
```bash
cd thermal_plots
ffmpeg -framerate 5 -pattern_type glob -i 'thermal*_top_view.png' \\
       -c:v libx264 -pix_fmt yuv420p -crf 18 \\
       thermal_evolution.mp4
```

### 3. HDF5 Data Files
**Thermal fields**: `thermal_fields.h5`
- Complete temperature evolution
- 4D data: (time, x, y, z)
- Compressed format

**Activation volumes**: `activation_volumes.h5`
- Build geometry evolution
- 4D data: (time, x, y, z)
- Highly compressed

**Usage**:
```python
from callbacks.hdf5_thermal_saver import load_thermal_field
from callbacks.hdf5_activation_saver import load_activation_volume

# Load specific timestep
temp = load_thermal_field("thermal_fields.h5", step=10)
activation = load_activation_volume("activation_volumes.h5", step=10)
```

### 4. STL Export (if available)
**Location**: `build_geometry.stl`
- Final build geometry
- Can be imported into:
  - Blender (3D animation)
  - MeshLab (mesh processing)
  - ParaView (scientific visualization)
  - Fusion 360 (CAD)

### 5. Metadata
**Simulation data**: `simulation_data.csv`
- Position, temperature, melt pool data
- Time series for plots

**Parameters**: `parameter_history.csv`
- Process parameters over time

**Summary**: `simulation_params.csv`
- Final simulation statistics

## Video Production Workflows

### Workflow 1: Simple Build Process Video
```bash
# Use camera frames directly
cd cam
ffmpeg -framerate 10 -i thermal_step_%04d.png \\
       -c:v libx264 -pix_fmt yuv420p build.mp4
```

### Workflow 2: Side-by-Side Comparison
```bash
# Combine camera + thermal plot
ffmpeg -framerate 10 \\
       -i cam/thermal_step_%04d.png \\
       -i thermal_plots/thermal%04d_top_view.png \\
       -filter_complex hstack \\
       comparison.mp4
```

### Workflow 3: Multi-Angle View
Requires editing software (Blender, Premiere, DaVinci Resolve)
1. Import camera images as image sequence
2. Import thermal plots as separate sequence
3. Create 2x2 or 1x3 grid layout
4. Add titles and annotations

### Workflow 4: 3D Animation with STL
Using Blender:
1. Import `build_geometry.stl`
2. Load camera images as background/texture
3. Animate camera path
4. Render with thermal overlay

## Recommended Settings

### For Presentations
- Resolution: 1920x1080 (Full HD)
- Framerate: 10-15 fps
- Codec: H.264
- Quality: CRF 18 (high quality)

### For Publications
- Resolution: As captured (800x600 or higher)
- Format: Individual PNG frames
- Include scale bars and annotations

### For Web/Social Media
- Resolution: 1280x720 (HD)
- Framerate: 15-30 fps
- Codec: H.264
- Quality: CRF 23 (balanced)

## Frame Numbering

Camera frames: {num_frames} total
"""

    if cam_images:
        first_frame = cam_images[0].name
        last_frame = cam_images[-1].name
        content += f"""
First frame: {first_frame}
Last frame: {last_frame}
"""

    content += f"""
## Next Steps

1. **Preview frames**: Check `cam/` directory for image quality
2. **Create video**: Use FFmpeg commands above
3. **Post-process**: Add titles, annotations, music
4. **Export STL**: Use for 3D visualization if needed

## Tools Needed

- **FFmpeg**: Video creation from frames
  - Download: https://ffmpeg.org/download.html
  - Windows: `choco install ffmpeg` or `winget install ffmpeg`

- **Blender** (optional): 3D animation
  - Download: https://www.blender.org/

- **ParaView** (optional): Scientific visualization
  - Download: https://www.paraview.org/

- **Python libraries** (optional): Custom processing
  ```bash
  pip install opencv-python matplotlib numpy h5py
  ```

## Contact

For questions about data format or processing, refer to:
- `README_HDF5_THERMAL_SAVER.md`
- `README_HDF5_ACTIVATION_SAVER.md`
- `README_PERSPECTIVE_CAMERA.md`
"""

    with open(summary_file, 'w') as f:
        f.write(content)

    print(f"\n✓ Video production guide saved: {summary_file}")


def main():
    """Run video production simulation."""
    print("\n" + "="*70)
    print("VIDEO PRODUCTION SIMULATION")
    print("="*70)
    print("\nThis simulation generates all assets needed for video production:")
    print("  1. Camera frames for build process video")
    print("  2. Thermal evolution plots")
    print("  3. HDF5 data for custom visualization")
    print("  4. STL export of final geometry")
    print("  5. Complete metadata")
    print("\nEstimated duration: 2-3 minutes")
    print("="*70)

    # Configuration
    print("\n" + "-"*70)
    print("SIMULATION PARAMETERS")
    print("-"*70)
    print("  Build volume: 30mm x 30mm x 20mm")
    print("  Part size: 10mm x 10mm x 5mm")
    print("  Voxel size: 200µm")
    print("  Time step: 100ms")
    print("  Target height: 8mm (will run until reached)")
    print("  Camera interval: Every 2 steps (for smooth video)")
    print("  Data save interval: Every 10 steps")

    # Create callbacks
    callbacks = [
        # Completion
        HeightCompletionCallback(target_height_mm=8.0),
        StepCountCompletionCallback(max_steps=100),  # Safety limit

        # Progress monitoring
        ProgressPrinter(),

        # Original callbacks (for comparison)
        ParameterLogger(save_file="parameter_history.csv"),
        FinalStateSaver(),
        ThermalPlotSaver(save_dir="thermal_plots", interval=10),
        StepDataCollector(
            tracked_fields=['position', 'melt_pool', 'build', 'clad'],
            save_path="simulation_data.csv"
        ),

        # NEW: HDF5 data storage (compressed)
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=10,  # Every 10 steps
            compression='gzip',
            compression_opts=4
        ),

        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=10,  # Every 10 steps
            compression='gzip',
            compression_opts=9
        ),

        # NEW: Perspective camera (for video frames)
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.015, 0.008),  # 15mm behind, 8mm above
            floor_angle_deg=35.0,  # Nice downward view
            fov_y_deg=50.0,  # Wide enough to see build
            save_images=True,
            save_dir="cam",
            interval=2,  # Every 2 steps = smooth video
            resolution_wh=(800, 600),
            cmap='hot',
            dpi=150
        )
    ]

    try:
        import time
        start_time = time.time()

        # Create simulation
        print("\n" + "-"*70)
        print("CREATING SIMULATION")
        print("-"*70)

        runner = SimulationRunner.from_human_units(
            build_volume_mm=(30.0, 30.0, 20.0),
            part_volume_mm=(10.0, 10.0, 5.0),
            voxel_size_um=200.0,
            delta_t_ms=100.0,
            scan_speed_mm_s=5.0,
            laser_power_W=800.0,
            powder_feed_g_min=3.0,
            hatch_spacing_um=700.0,
            layer_spacing_um=300.0,
            substrate_height_mm=5.0,
            experiment_label="video_prod",
            callbacks=callbacks
        )

        output_dir = Path(runner.simulation.output_dir)
        print(f"✓ Simulation created")
        print(f"✓ Output directory: {output_dir}")

        # Run simulation
        print("\n" + "-"*70)
        print("RUNNING SIMULATION")
        print("-"*70)
        print("Progress will be shown below...\n")

        runner.run()

        elapsed = time.time() - start_time
        print(f"\n✓ Simulation completed in {elapsed:.1f} seconds")

        # Post-processing
        print("\n" + "="*70)
        print("POST-PROCESSING")
        print("="*70)

        # 1. Export STL
        print("\n1. STL EXPORT")
        print("-"*70)

        # Load final activation volume
        final_activation = runner.simulation.volume_tracker.activated
        voxel_size = runner.simulation.config['voxel_size'][0]

        stl_path = output_dir / "build_geometry.stl"
        export_to_stl(final_activation, stl_path, voxel_size)

        # 2. Analyze HDF5 data
        print("\n2. HDF5 DATA ANALYSIS")
        print("-"*70)

        h5_thermal = output_dir / "thermal_fields.h5"
        h5_activation = output_dir / "activation_volumes.h5"

        if h5_thermal.exists():
            from callbacks.hdf5_thermal_saver import list_steps_in_file
            steps = list_steps_in_file(str(h5_thermal))
            size_mb = h5_thermal.stat().st_size / (1024**2)
            print(f"✓ Thermal fields HDF5:")
            print(f"  - Timesteps: {len(steps)}")
            print(f"  - File size: {size_mb:.2f} MB")
            print(f"  - Steps saved: {steps[:5]}{'...' if len(steps) > 5 else ''}")

        if h5_activation.exists():
            stats = get_activation_statistics(str(h5_activation))
            size_mb = h5_activation.stat().st_size / (1024**2)
            print(f"✓ Activation volumes HDF5:")
            print(f"  - Timesteps: {stats['num_steps']}")
            print(f"  - File size: {size_mb:.2f} MB")
            print(f"  - Final activation: {stats.get('final_activation_fraction', 0)*100:.2f}%")

        # 3. Count output files
        print("\n3. OUTPUT SUMMARY")
        print("-"*70)

        cam_images = list(output_dir.glob("cam/thermal_step_*.png"))
        thermal_plots = list(output_dir.glob("thermal_plots/*.png"))
        csv_files = list(output_dir.glob("*.csv"))
        npy_files = list(output_dir.glob("*.npy"))
        h5_files = list(output_dir.glob("*.h5"))
        stl_files = list(output_dir.glob("*.stl"))

        print(f"Camera frames:  {len(cam_images):3d} images")
        print(f"Thermal plots:  {len(thermal_plots):3d} images")
        print(f"CSV files:      {len(csv_files):3d} files")
        print(f"NPY files:      {len(npy_files):3d} files")
        print(f"HDF5 files:     {len(h5_files):3d} files")
        print(f"STL files:      {len(stl_files):3d} files")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
        print(f"\nTotal output:   {total_size/(1024**2):.2f} MB")

        # 4. Create video production guide
        create_video_frames_summary(output_dir)

        # Success!
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print("\nAll assets generated for video production!")
        print(f"\nOutput location:")
        print(f"  {output_dir}")
        print(f"\nKey files:")
        print(f"  - Camera frames: cam/thermal_step_*.png ({len(cam_images)} frames)")
        print(f"  - Thermal data: thermal_fields.h5")
        print(f"  - Geometry: build_geometry.stl")
        print(f"  - Guide: VIDEO_PRODUCTION_GUIDE.md")
        print(f"\nNext steps:")
        print(f"  1. Read VIDEO_PRODUCTION_GUIDE.md for FFmpeg commands")
        print(f"  2. Create video: cd cam && ffmpeg -framerate 10 -i thermal_step_%04d.png build.mp4")
        print(f"  3. Open STL in Blender/ParaView for 3D visualization")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if not success:
        print("\nWARNING: Simulation failed. Check errors above.")
        sys.exit(1)
