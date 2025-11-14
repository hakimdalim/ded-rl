"""
Dry-run test for simulate.py with REAL simulation (not DummySimulation).

This tests the complete system with actual MultiTrackMultiLayerSimulation
but with VERY small parameters for fast execution.

Run time: ~2-5 minutes depending on hardware
"""

import sys
sys.path.append('..')

import time
from pathlib import Path
import shutil

# Import simulate components
from simulate import SimulationRunner

# Import callbacks

from callbacks.completion_callbacks import (
    HeightCompletionCallback,
    StepCountCompletionCallback
)
from callbacks.step_data_collector import StepDataCollector
from callbacks.callback_collection import (
    ProgressPrinter,
    ParameterLogger,
    TemperatureSliceSaver,
    ThermalPlotSaver,
    CrossSectionPlotter,
    FinalStateSaver,
    get_default_callbacks
)


def print_header(msg):
    """Print test header."""
    print("\n" + "="*70)
    print(msg)
    print("="*70)


def print_success(msg):
    """Print success."""
    print(f"✓ {msg}")


def print_failure(msg):
    """Print failure."""
    print(f"✗ {msg}")


def cleanup_test_outputs():
    """Clean up old test outputs."""
    test_dirs = [
        "_experiments/test_minimal",
        "_experiments/test_data_collection",
        "_experiments/test_visualization",
        "_experiments/test_default_callbacks"
    ]

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)


def test_minimal_simulation():
    """
    Test 1: Minimal simulation with basic callbacks.
    Goal: Verify simulation runs without errors.
    """
    print_header("TEST 1: Minimal Real Simulation")

    print("\nCreating TINY simulation (fast execution):")
    print("  Build volume: 10mm x 10mm x 8mm")
    print("  Part size: 5mm x 5mm x 2mm")
    print("  Voxel size: 200µm")
    print("  Expected: ~20-50 steps")

    callbacks = [
        StepCountCompletionCallback(max_steps=50),  # Stop early for speed
        ProgressPrinter(),
        StepDataCollector(tracked_fields=['position', 'melt_pool'])
    ]

    try:
        start_time = time.time()

        runner = SimulationRunner.from_human_units(
            # VERY small simulation for speed
            build_volume_mm=(10.0, 10.0, 8.0),
            part_volume_mm=(5.0, 5.0, 2.0),
            voxel_size_um=200.0,
            delta_t_ms=50.0,  # Smaller time step to avoid completing track in 1 step
            scan_speed_mm_s=3.0,  # Slower scan speed
            laser_power_W=600.0,
            powder_feed_g_min=2.0,
            hatch_spacing_um=700.0,
            layer_spacing_um=350.0,
            substrate_height_mm=2.0,  # Reduced to fit in build volume
            experiment_label="test_minimal",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("\nRunning simulation...")

        runner.run()

        elapsed = time.time() - start_time

        print_success(f"Simulation completed in {elapsed:.1f} seconds")
        print_success(f"Output saved to: {runner.simulation.output_dir}")

        # Verify output directory exists
        if Path(runner.simulation.output_dir).exists():
            print_success("Output directory created")
        else:
            print_failure("Output directory not found")

        # Check for data file
        csv_files = list(Path(runner.simulation.output_dir).glob("*.csv"))
        if csv_files:
            print_success(f"CSV files saved: {len(csv_files)}")
        else:
            print_failure("No CSV files found")

        return True

    except Exception as e:
        print_failure(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collection():
    """
    Test 2: Verify data collection callbacks work.
    Goal: Check all data files are created correctly.
    """
    print_header("TEST 2: Data Collection Verification")

    callbacks = [
        StepCountCompletionCallback(max_steps=30),
        StepDataCollector(
            tracked_fields=['position', 'melt_pool', 'clad', 'build'],
            save_path="simulation_data.csv"
        ),
        ParameterLogger(save_file="parameter_history.csv"),
        FinalStateSaver(),
        ProgressPrinter()
    ]

    try:
        runner = SimulationRunner.from_human_units(
            build_volume_mm=(3.0, 3.0, 2.0),
            part_volume_mm=(1.0, 1.0, 0.5),
            voxel_size_um=100.0,
            delta_t_ms=200.0,
            scan_speed_mm_s=5.0,
            laser_power_W=400.0,
            powder_feed_g_min=1.5,
            experiment_label="test_data_collection",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")

        runner.run()

        output_dir = Path(runner.simulation.output_dir)

        # Check expected files
        expected_files = [
            "simulation_data.csv",
            "parameter_history.csv",
            "final_activated_vol.npy",
            "final_temperature_vol.npy",
            "simulation_params.csv"
        ]

        found_files = []
        missing_files = []

        for filename in expected_files:
            if (output_dir / filename).exists():
                found_files.append(filename)
                print_success(f"Found: {filename}")
            else:
                missing_files.append(filename)
                print_failure(f"Missing: {filename}")

        if len(found_files) == len(expected_files):
            print_success(f"All {len(expected_files)} files created!")
            return True
        else:
            print_failure(f"Missing {len(missing_files)} files")
            return False

    except Exception as e:
        print_failure(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_callbacks():
    """
    Test 3: Verify visualization callbacks work.
    Goal: Check plots and images are generated.
    """
    print_header("TEST 3: Visualization Callbacks")

    callbacks = [
        StepCountCompletionCallback(max_steps=20),
        ThermalPlotSaver(interval=10),  # Save every 10 steps
        CrossSectionPlotter(num_sections=3),
        ProgressPrinter()
    ]

    try:
        runner = SimulationRunner.from_human_units(
            build_volume_mm=(3.0, 3.0, 2.0),
            part_volume_mm=(1.0, 1.0, 0.5),
            voxel_size_um=100.0,
            delta_t_ms=200.0,
            scan_speed_mm_s=5.0,
            laser_power_W=400.0,
            powder_feed_g_min=1.5,
            experiment_label="test_visualization",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("Note: This may take longer due to plot generation...")

        runner.run()

        output_dir = Path(runner.simulation.output_dir)

        # Check for plot directories
        thermal_dir = output_dir / "thermal_plots"
        cross_dir = output_dir / "cross_sections"

        success_count = 0

        if thermal_dir.exists():
            png_files = list(thermal_dir.glob("*.png"))
            if png_files:
                print_success(f"Thermal plots saved: {len(png_files)} files")
                success_count += 1
            else:
                print_failure("No thermal plots found")
        else:
            print_failure("Thermal plot directory not created")

        if cross_dir.exists():
            png_files = list(cross_dir.glob("*.png"))
            if png_files:
                print_success(f"Cross-section plots saved: {len(png_files)} files")
                success_count += 1
            else:
                print_failure("No cross-section plots found")
        else:
            print_failure("Cross-section directory not created")

        return success_count == 2

    except Exception as e:
        print_failure(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_callbacks():
    """
    Test 4: Test with default callback set (full production setup).
    Goal: Verify get_default_callbacks() works correctly.

    WARNING: This will be SLOW due to all the file saving!
    """
    print_header("TEST 4: Default Callback Set (Production Mode)")

    print("\n⚠️  WARNING: This test runs with ALL default callbacks.")
    print("   It will generate many files and take longer.")
    print("   Skip this test if you want quick results.\n")

    # Uncomment to skip this test
    # print("Skipping test 4 (slow)...")
    # return True

    try:
        # Get full default callback set
        callbacks = get_default_callbacks()

        # But limit steps for speed
        from callbacks.completion_callbacks import SimulationComplete

        # Replace height completion with step count for speed
        for i, cb in enumerate(callbacks):
            if cb.__class__.__name__ == 'HeightCompletionCallback':
                callbacks[i] = StepCountCompletionCallback(max_steps=30)
                break

        print(f"Running with {len(callbacks)} callbacks:")
        for cb in callbacks:
            print(f"  - {cb.__class__.__name__}")

        runner = SimulationRunner.from_human_units(
            build_volume_mm=(3.0, 3.0, 2.0),
            part_volume_mm=(1.0, 1.0, 0.5),
            voxel_size_um=100.0,
            delta_t_ms=200.0,
            scan_speed_mm_s=5.0,
            laser_power_W=400.0,
            powder_feed_g_min=1.5,
            experiment_label="test_default_callbacks",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("Running... (this may take a few minutes)\n")

        start_time = time.time()
        runner.run()
        elapsed = time.time() - start_time

        print_success(f"Completed in {elapsed:.1f} seconds")

        # Check output directory structure
        output_dir = Path(runner.simulation.output_dir)

        # List all subdirectories
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        print(f"\n✓ Created {len(subdirs)} output subdirectories:")
        for subdir in subdirs:
            file_count = len(list(subdir.iterdir()))
            print(f"    {subdir.name}/  ({file_count} files)")

        # List all files in root
        files = [f for f in output_dir.iterdir() if f.is_file()]
        print(f"\n✓ Created {len(files)} output files in root:")
        for file in files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"    {file.name}  ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        print_failure(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all dry-run tests."""
    print("\n" + "█"*70)
    print("SIMULATE.PY DRY-RUN TEST SUITE")
    print("█"*70)
    print("\nTesting with REAL simulation (MultiTrackMultiLayerSimulation)")
    print("Parameters are VERY small for fast execution.")
    print("\nEstimated total time: 3-7 minutes\n")

    # Clean up old outputs
    print("Cleaning up old test outputs...")
    cleanup_test_outputs()

    # Run tests
    results = {}

    print("\n" + "█"*70)
    results['test_1_minimal'] = test_minimal_simulation()

    print("\n" + "█"*70)
    results['test_2_data'] = test_data_collection()

    print("\n" + "█"*70)
    results['test_3_viz'] = test_visualization_callbacks()

    print("\n" + "█"*70)
    results['test_4_default'] = test_default_callbacks()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your callback system is working correctly.")
        print("\nNext steps:")
        print("  1. Check output in _experiments/test_* directories")
        print("  2. Run full simulation with: python simulate.py")
        print("  3. Customize callbacks for your use case")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Check error messages above for details.")
        print("See TESTING_GUIDE.md for debugging tips.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()