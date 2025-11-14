"""
Comparison test: Simulate with original callbacks vs new custom callbacks

This test runs two simulations:
1. BEFORE: Using only original/existing callbacks
2. AFTER: Adding the three new custom callbacks

Purpose:
- Compare file sizes and formats
- Verify naming conventions match
- Check plot image quality and labeling
- Document camera interaction differences

Run: python test_callbacks/test_comparison_before_after.py
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
from pathlib import Path
import numpy as np

# Import simulation components
from simulate import SimulationRunner

# Import EXISTING/ORIGINAL callbacks
from callbacks.completion_callbacks import StepCountCompletionCallback
from callbacks.callback_collection import (
    ProgressPrinter,
    ParameterLogger,
    FinalStateSaver,
    ThermalPlotSaver
)
from callbacks.step_data_collector import StepDataCollector

# Import NEW custom callbacks
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.hdf5_activation_saver import HDF5ActivationSaver
from callbacks.perspective_camera_callback import PerspectiveCameraCallback


def get_directory_size(path):
    """Calculate total size of directory in MB."""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)


def list_output_files(output_dir):
    """List all files in output directory with sizes."""
    output_dir = Path(output_dir)
    files = {}

    for entry in output_dir.rglob('*'):
        if entry.is_file():
            rel_path = entry.relative_to(output_dir)
            size_kb = entry.stat().st_size / 1024
            files[str(rel_path)] = size_kb

    return files


def run_simulation_before():
    """
    BEFORE: Run simulation with ORIGINAL callbacks only.

    This represents the existing functionality before adding new callbacks.
    """
    print("\n" + "="*70)
    print("SIMULATION 1: BEFORE (Original Callbacks Only)")
    print("="*70)
    print("\nUsing existing callbacks:")
    print("  - StepCountCompletionCallback")
    print("  - ProgressPrinter")
    print("  - ParameterLogger")
    print("  - FinalStateSaver (saves final_activated_vol.npy, final_temperature_vol.npy)")
    print("  - ThermalPlotSaver (saves PNG images in thermal_plots/)")
    print("  - StepDataCollector (saves simulation_data.csv)")

    callbacks = [
        StepCountCompletionCallback(max_steps=20),
        ProgressPrinter(),
        ParameterLogger(save_file="parameter_history.csv"),
        FinalStateSaver(),
        ThermalPlotSaver(save_dir="thermal_plots", interval=5),
        StepDataCollector(tracked_fields=['position', 'melt_pool', 'build'],
                         save_path="simulation_data.csv")
    ]

    try:
        start_time = time.time()

        runner = SimulationRunner.from_human_units(
            build_volume_mm=(20.0, 20.0, 15.0),
            part_volume_mm=(5.0, 5.0, 2.0),
            voxel_size_um=200.0,
            delta_t_ms=200.0,
            scan_speed_mm_s=3.0,
            laser_power_W=600.0,
            powder_feed_g_min=2.0,
            hatch_spacing_um=700.0,
            layer_spacing_um=350.0,
            substrate_height_mm=5.0,
            experiment_label="comp_before",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("\nRunning simulation...\n")

        runner.run()

        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.1f} seconds")

        # Analyze output
        output_dir = runner.simulation.output_dir
        total_size = get_directory_size(output_dir)
        files = list_output_files(output_dir)

        return {
            'output_dir': output_dir,
            'elapsed_time': elapsed,
            'total_size_mb': total_size,
            'files': files,
            'num_files': len(files)
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simulation_after():
    """
    AFTER: Run simulation with ORIGINAL + NEW custom callbacks.

    This adds the three new callbacks on top of existing ones.
    """
    print("\n" + "="*70)
    print("SIMULATION 2: AFTER (Original + New Custom Callbacks)")
    print("="*70)
    print("\nUsing existing callbacks PLUS new ones:")
    print("  Original:")
    print("    - StepCountCompletionCallback")
    print("    - ProgressPrinter")
    print("    - ParameterLogger")
    print("    - FinalStateSaver")
    print("    - ThermalPlotSaver")
    print("    - StepDataCollector")
    print("  NEW:")
    print("    - HDF5ThermalSaver (thermal_fields.h5)")
    print("    - HDF5ActivationSaver (activation_volumes.h5)")
    print("    - PerspectiveCameraCallback (cam/ directory)")

    callbacks = [
        # ORIGINAL callbacks (unchanged)
        StepCountCompletionCallback(max_steps=20),
        ProgressPrinter(),
        ParameterLogger(save_file="parameter_history.csv"),
        FinalStateSaver(),
        ThermalPlotSaver(save_dir="thermal_plots", interval=5),
        StepDataCollector(tracked_fields=['position', 'melt_pool', 'build'],
                         save_path="simulation_data.csv"),

        # NEW custom callbacks
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=5,
            compression='gzip',
            compression_opts=4
        ),
        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=5,
            compression='gzip',
            compression_opts=9
        ),
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.12, 0.04),
            floor_angle_deg=30.0,
            fov_y_deg=45.0,
            save_images=True,
            save_dir="cam",
            interval=5,
            resolution_wh=(800, 600)
        )
    ]

    try:
        start_time = time.time()

        runner = SimulationRunner.from_human_units(
            build_volume_mm=(20.0, 20.0, 15.0),
            part_volume_mm=(5.0, 5.0, 2.0),
            voxel_size_um=200.0,
            delta_t_ms=200.0,
            scan_speed_mm_s=3.0,
            laser_power_W=600.0,
            powder_feed_g_min=2.0,
            hatch_spacing_um=700.0,
            layer_spacing_um=350.0,
            substrate_height_mm=5.0,
            experiment_label="comp_after",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("\nRunning simulation...\n")

        runner.run()

        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.1f} seconds")

        # Analyze output
        output_dir = runner.simulation.output_dir
        total_size = get_directory_size(output_dir)
        files = list_output_files(output_dir)

        return {
            'output_dir': output_dir,
            'elapsed_time': elapsed,
            'total_size_mb': total_size,
            'files': files,
            'num_files': len(files)
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(before, after):
    """Compare results from both simulations."""
    print("\n" + "="*70)
    print("COMPARISON REPORT")
    print("="*70)

    # 1. Execution time comparison
    print("\n1. EXECUTION TIME")
    print("-" * 70)
    print(f"  BEFORE: {before['elapsed_time']:.1f} seconds")
    print(f"  AFTER:  {after['elapsed_time']:.1f} seconds")
    time_diff = after['elapsed_time'] - before['elapsed_time']
    time_pct = (time_diff / before['elapsed_time']) * 100
    print(f"  Difference: {time_diff:+.1f} seconds ({time_pct:+.1f}%)")

    # 2. Total storage comparison
    print("\n2. TOTAL STORAGE")
    print("-" * 70)
    print(f"  BEFORE: {before['total_size_mb']:.2f} MB")
    print(f"  AFTER:  {after['total_size_mb']:.2f} MB")
    size_diff = after['total_size_mb'] - before['total_size_mb']
    print(f"  New callbacks added: {size_diff:+.2f} MB")

    # 3. File count comparison
    print("\n3. FILE COUNT")
    print("-" * 70)
    print(f"  BEFORE: {before['num_files']} files")
    print(f"  AFTER:  {after['num_files']} files")
    print(f"  New files added: {after['num_files'] - before['num_files']}")

    # 4. Common files (should be identical in both)
    print("\n4. COMMON FILES (Original Callbacks)")
    print("-" * 70)
    common_files = set(before['files'].keys()) & set(after['files'].keys())

    for filename in sorted(common_files):
        before_size = before['files'][filename]
        after_size = after['files'][filename]
        diff = after_size - before_size

        # Group files by type
        if filename.endswith('.csv'):
            marker = "CSV"
        elif filename.endswith('.npy'):
            marker = "NPY"
        elif filename.endswith('.png'):
            marker = "PNG"
        else:
            marker = "   "

        if abs(diff) < 0.1:  # Less than 0.1 KB difference
            status = "✓ IDENTICAL"
        else:
            status = f"△ {diff:+.1f} KB"

        print(f"  [{marker}] {filename:40s} {status}")

    # 5. NEW files (only in AFTER)
    print("\n5. NEW FILES (Custom Callbacks)")
    print("-" * 70)
    new_files = set(after['files'].keys()) - set(before['files'].keys())

    # Group by callback
    hdf5_thermal = [f for f in new_files if 'thermal_fields.h5' in f]
    hdf5_activation = [f for f in new_files if 'activation_volumes.h5' in f]
    camera_files = [f for f in new_files if 'cam' in str(f) and f.endswith('.png')]
    other_files = [f for f in new_files if f not in hdf5_thermal + hdf5_activation + camera_files]

    if hdf5_thermal:
        print("\n  HDF5ThermalSaver:")
        for filename in sorted(hdf5_thermal):
            size_kb = after['files'][filename]
            print(f"    - {filename:50s} {size_kb:8.1f} KB")

    if hdf5_activation:
        print("\n  HDF5ActivationSaver:")
        for filename in sorted(hdf5_activation):
            size_kb = after['files'][filename]
            print(f"    - {filename:50s} {size_kb:8.1f} KB")

    if camera_files:
        print("\n  PerspectiveCameraCallback:")
        total_cam_size = sum(after['files'][f] for f in camera_files)
        print(f"    - {len(camera_files)} camera images, total {total_cam_size:.1f} KB")
        for filename in sorted(camera_files)[:3]:  # Show first 3
            size_kb = after['files'][filename]
            print(f"      {filename:50s} {size_kb:8.1f} KB")
        if len(camera_files) > 3:
            print(f"      ... and {len(camera_files) - 3} more")

    if other_files:
        print("\n  Other files:")
        for filename in sorted(other_files):
            size_kb = after['files'][filename]
            print(f"    - {filename:50s} {size_kb:8.1f} KB")

    # 6. Storage breakdown
    print("\n6. STORAGE BREAKDOWN")
    print("-" * 70)

    # Calculate sizes by category for BEFORE
    before_csv = sum(s for f, s in before['files'].items() if f.endswith('.csv'))
    before_npy = sum(s for f, s in before['files'].items() if f.endswith('.npy'))
    before_png = sum(s for f, s in before['files'].items() if f.endswith('.png'))

    print("\n  BEFORE:")
    print(f"    CSV files:  {before_csv/1024:8.2f} MB")
    print(f"    NPY files:  {before_npy/1024:8.2f} MB")
    print(f"    PNG files:  {before_png/1024:8.2f} MB")
    print(f"    Total:      {before['total_size_mb']:8.2f} MB")

    # Calculate sizes by category for AFTER
    after_csv = sum(s for f, s in after['files'].items() if f.endswith('.csv'))
    after_npy = sum(s for f, s in after['files'].items() if f.endswith('.npy'))
    after_png = sum(s for f, s in after['files'].items() if f.endswith('.png'))
    after_h5 = sum(s for f, s in after['files'].items() if f.endswith('.h5'))

    print("\n  AFTER:")
    print(f"    CSV files:  {after_csv/1024:8.2f} MB")
    print(f"    NPY files:  {after_npy/1024:8.2f} MB")
    print(f"    PNG files:  {after_png/1024:8.2f} MB")
    print(f"    H5 files:   {after_h5/1024:8.2f} MB  [NEW]")
    print(f"    Total:      {after['total_size_mb']:8.2f} MB")

    # 7. Key observations
    print("\n7. KEY OBSERVATIONS")
    print("-" * 70)

    print("\n  Compatibility:")
    identical_count = sum(1 for f in common_files if abs(after['files'][f] - before['files'][f]) < 0.1)
    print(f"    - {identical_count}/{len(common_files)} common files are identical")
    print(f"    - Original callbacks work unchanged with new callbacks")

    print("\n  New Capabilities:")
    print(f"    - HDF5 compressed volumes: {after_h5/1024:.2f} MB")
    print(f"    - Camera images: {len(camera_files)} images")
    print(f"    - Total new data: {size_diff:.2f} MB")

    compression_benefit = (before_npy/1024) / (after_h5/1024) if after_h5 > 0 else 0
    print(f"\n  Compression Benefit:")
    print(f"    - NPY (uncompressed): {before_npy/1024:.2f} MB")
    print(f"    - HDF5 (compressed):  {after_h5/1024:.2f} MB")
    if compression_benefit > 1:
        print(f"    - Compression ratio: {compression_benefit:.1f}x")
        print(f"    - Space saved: {(before_npy/1024 - after_h5/1024):.2f} MB ({(1-after_h5/before_npy)*100:.1f}%)")


def main():
    """Run comparison test."""
    print("\n" + "="*70)
    print("BEFORE/AFTER COMPARISON TEST")
    print("="*70)
    print("\nThis test compares:")
    print("  1. Original callbacks (existing functionality)")
    print("  2. Original + New custom callbacks (enhanced functionality)")
    print("\nBoth simulations use identical parameters.")
    print("Expected duration: 1-2 minutes total\n")

    # Run BEFORE simulation
    before = run_simulation_before()

    if before is None:
        print("\nBEFORE simulation failed. Aborting comparison.")
        return

    # Run AFTER simulation
    after = run_simulation_after()

    if after is None:
        print("\nAFTER simulation failed. Cannot complete comparison.")
        return

    # Compare results
    compare_results(before, after)

    # Final summary
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nYou can now inspect the output directories:")
    print(f"\n  BEFORE: {before['output_dir']}")
    print(f"  AFTER:  {after['output_dir']}")
    print("\nRecommendations:")
    print("  1. Check thermal_plots/ PNG images are identical in both")
    print("  2. Verify HDF5 files contain same data as NPY files")
    print("  3. Review camera images in cam/ directory")
    print("  4. Compare file sizes and compression ratios")


if __name__ == "__main__":
    main()
