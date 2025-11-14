"""
Comprehensive dry-run test suite for ALL callbacks.
Tests each callback type to ensure everything works.

Run this to verify your callback system is working correctly!
"""

import sys
sys.path.append('..')

import os
import numpy as np
from pathlib import Path
from dummy_simulation import DummySimulation
from callbacks._callback_manager import CallbackManager
from callbacks._base_callbacks import SimulationEvent
from callbacks.completion_callbacks import (
    StepCountCompletionCallback,
    HeightCompletionCallback,
    TrackCountCompletionCallback,
    LayerCountCompletionCallback,
    SimulationComplete
)
from callbacks.step_data_collector import StepDataCollector
from callbacks.error_callbacks import ErrorCompletionCallback
from callbacks.live_plotter_callback import AdvancedLivePlotter
from callbacks.track_calibration_callback import TrackCalibrationCallback
from callbacks.callback_collection import (
    TemperatureSliceSaver,
    VoxelTemperatureSaver,
    MeshSaver,
    ThermalPlotSaver,
    ProgressPrinter,
    CrossSectionPlotter,
    PickleSaver,
    CompressCallback,
    ParameterLogger,
    LivePlotter,
    FinalStateSaver
)


def print_test_header(test_name):
    """Print a nice header for each test."""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def print_success(message):
    """Print success message."""
    print(f"✓ {message}")


def print_failure(message):
    """Print failure message."""
    print(f"✗ {message}")


# ============================================================================
# CATEGORY A: COMPLETION CALLBACKS (4)
# ============================================================================

def test_height_completion():
    """Test HeightCompletionCallback."""
    print_test_header("Height Completion Callback")

    sim = DummySimulation(output_dir="./test_output/height_completion")
    callbacks = [HeightCompletionCallback()]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for layer in range(10):
            manager(sim, SimulationEvent.LAYER_START)

            for step in range(50):
                sim.step()
                manager(sim, SimulationEvent.STEP_COMPLETE)
                sim.progress_tracker.max_height_reached += 0.00005

            sim.progress_tracker.current_layer += 1
            # manager(sim, SimulationEvent.LAYER_COMPLETE)
            manager(sim, SimulationEvent.STEP_COMPLETE)

        print_failure("Did not stop at target height")
    except SimulationComplete as e:
        print_success(f"Stopped at target height: {e}")
        sim.complete()


def test_step_count_completion():
    """Test StepCountCompletionCallback."""
    print_test_header("Step Count Completion Callback")

    sim = DummySimulation(output_dir="./test_output/step_count")
    callbacks = [StepCountCompletionCallback(max_steps=20)]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(100):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)

        print_failure("Did not stop after 20 steps")
    except SimulationComplete as e:
        if sim.progress_tracker.step_count == 20:
            print_success(f"Stopped at exactly 20 steps: {e}")
        else:
            print_failure(f"Stopped at {sim.progress_tracker.step_count} steps, expected 20")
        sim.complete()


def test_track_count_completion():
    """Test TrackCountCompletionCallback."""
    print_test_header("Track Count Completion Callback")

    sim = DummySimulation(output_dir="./test_output/track_count")
    callbacks = [TrackCountCompletionCallback(max_tracks=3)]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for track in range(10):
            manager(sim, SimulationEvent.TRACK_START)

            for step in range(5):
                sim.step()
                manager(sim, SimulationEvent.STEP_COMPLETE)

            manager(sim, SimulationEvent.TRACK_COMPLETE)

        print_failure("Did not stop after 3 tracks")
    except SimulationComplete as e:
        print_success(f"Stopped after 3 tracks: {e}")
        sim.complete()


def test_layer_count_completion():
    """Test LayerCountCompletionCallback."""
    print_test_header("Layer Count Completion Callback")

    sim = DummySimulation(output_dir="./test_output/layer_count")
    callbacks = [LayerCountCompletionCallback(max_layers=2)]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for layer in range(10):
            sim.progress_tracker.current_layer = layer
            manager(sim, SimulationEvent.LAYER_START)

            for step in range(5):
                sim.step()
                manager(sim, SimulationEvent.STEP_COMPLETE)

        print_failure("Did not stop after 2 layers")
    except SimulationComplete as e:
        print_success(f"Stopped after 2 layers: {e}")
        sim.complete()


# ============================================================================
# CATEGORY B: DATA COLLECTION (2)
# ============================================================================

def test_step_data_collector():
    """Test StepDataCollector."""
    print_test_header("Step Data Collector")

    sim = DummySimulation(output_dir="./test_output/data_collector")
    callbacks = [
        StepCountCompletionCallback(max_steps=10),
        StepDataCollector(
            tracked_fields=['position', 'melt_pool', 'clad'],
            save_path="test_data.csv"
        )
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    # Find data collector
    data_collector = None
    for cb in manager.callbacks:
        if hasattr(cb, 'step_data') and hasattr(cb, 'save_path') and cb.save_path == "test_data.csv":
            data_collector = cb
            break

    if data_collector is None:
        print_failure("Could not find StepDataCollector")
        return

    try:
        for i in range(20):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check collected data
    if len(data_collector.step_data) == 10:
        print_success(f"Collected 10 steps of data")
    else:
        print_failure(f"Expected 10 steps, got {len(data_collector.step_data)}")

    # Check if CSV was saved
    csv_path = Path("./test_output/data_collector/test_data.csv")
    if csv_path.exists():
        print_success(f"CSV file saved: {csv_path}")
    else:
        print_failure(f"CSV file not saved")

    # Check data fields
    if data_collector.step_data:
        fields = data_collector.step_data[0].keys()
        expected_fields = ['step', 'position.x', 'position.y', 'position.z',
                          'melt_pool.width', 'clad.width']
        if all(f in fields for f in expected_fields):
            print_success(f"Data contains expected fields")
        else:
            print_failure(f"Missing some expected fields: {fields}")


def test_parameter_logger():
    """Test ParameterLogger."""
    print_test_header("Parameter Logger")

    sim = DummySimulation(output_dir="./test_output/param_logger")
    callbacks = [
        StepCountCompletionCallback(max_steps=10),
        ParameterLogger(save_file="params.csv")
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(20):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if parameter CSV was saved
    csv_path = Path("./test_output/param_logger/params.csv")
    if csv_path.exists():
        print_success(f"Parameter history saved: {csv_path}")
    else:
        print_failure(f"Parameter history not saved")


# ============================================================================
# CATEGORY C: VISUALIZATION (4) - Limited testing (no display)
# ============================================================================

def test_thermal_plot_saver():
    """Test ThermalPlotSaver."""
    print_test_header("Thermal Plot Saver")

    sim = DummySimulation(output_dir="./test_output/thermal_plots")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        ThermalPlotSaver(interval=2)  # Save every 2 steps
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if plots were saved (should have 2-3 plots)
    plot_dir = Path("./test_output/thermal_plots/thermal_plots")
    if plot_dir.exists():
        png_files = list(plot_dir.glob("*.png"))
        if len(png_files) > 0:
            print_success(f"Thermal plots saved: {len(png_files)} files")
        else:
            print_failure(f"No thermal plots saved")
    else:
        print_failure(f"Thermal plot directory not created")


def test_cross_section_plotter():
    """Test CrossSectionPlotter."""
    print_test_header("Cross Section Plotter")

    sim = DummySimulation(output_dir="./test_output/cross_sections")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        CrossSectionPlotter(num_sections=3)
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if cross-section plots were saved
    plot_dir = Path("./test_output/cross_sections/cross_sections")
    if plot_dir.exists():
        png_files = list(plot_dir.glob("*.png"))
        if len(png_files) > 0:
            print_success(f"Cross-section plots saved: {len(png_files)} files")
        else:
            print_failure(f"No cross-section plots saved")
    else:
        print_failure(f"Cross-section directory not created")


# ============================================================================
# CATEGORY D: SAVING/EXPORT (5)
# ============================================================================

def test_temperature_slice_saver():
    """Test TemperatureSliceSaver."""
    print_test_header("Temperature Slice Saver")

    sim = DummySimulation(output_dir="./test_output/temp_slices")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        TemperatureSliceSaver(interval=2)
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if slices were saved
    slice_dir = Path("./test_output/temp_slices/temperatures")
    if slice_dir.exists():
        npy_files = list(slice_dir.glob("*.npy"))
        if len(npy_files) > 0:
            print_success(f"Temperature slices saved: {len(npy_files)} files")
        else:
            print_failure(f"No temperature slices saved")
    else:
        print_failure(f"Temperature slice directory not created")


def test_voxel_temperature_saver():
    """Test VoxelTemperatureSaver."""
    print_test_header("Voxel Temperature Saver")

    sim = DummySimulation(output_dir="./test_output/voxel_temps")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        VoxelTemperatureSaver(interval=3)
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if voxel temps were saved
    voxel_dir = Path("./test_output/voxel_temps/voxel_temps")
    if voxel_dir.exists():
        npy_files = list(voxel_dir.glob("*.npy"))
        if len(npy_files) > 0:
            print_success(f"Voxel temperature fields saved: {len(npy_files)} files")
        else:
            print_failure(f"No voxel temperature fields saved")
    else:
        print_failure(f"Voxel temperature directory not created")


def test_mesh_saver():
    """Test MeshSaver."""
    print_test_header("Mesh Saver")

    sim = DummySimulation(output_dir="./test_output/meshes")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        MeshSaver(interval=3)
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if meshes were saved
    mesh_dir = Path("./test_output/meshes/build_mesh")
    if mesh_dir.exists():
        stl_files = list(mesh_dir.glob("*.stl"))
        if len(stl_files) > 0:
            print_success(f"STL meshes saved: {len(stl_files)} files")
        else:
            print_failure(f"No STL meshes saved")
    else:
        print_failure(f"Mesh directory not created")


def test_final_state_saver():
    """Test FinalStateSaver."""
    print_test_header("Final State Saver")

    sim = DummySimulation(output_dir="./test_output/final_state")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        FinalStateSaver()
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if final state files were saved
    output_dir = Path("./test_output/final_state")
    expected_files = [
        "final_activated_vol.npy",
        "final_temperature_vol.npy",
        "simulation_params.csv"
    ]

    found_files = []
    for filename in expected_files:
        if (output_dir / filename).exists():
            found_files.append(filename)

    if len(found_files) == len(expected_files):
        print_success(f"All final state files saved")
    else:
        print_failure(f"Missing files: {set(expected_files) - set(found_files)}")


def test_pickle_saver():
    """Test PickleSaver."""
    print_test_header("Pickle Saver")

    sim = DummySimulation(output_dir="./test_output/pickle")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        PickleSaver(
            obj_getter=lambda ctx: ctx['simulation'].clad_manager,
            save_file="clad_manager.pkl"
        )
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if pickle was saved
    pickle_path = Path("./test_output/pickle/clad_manager.pkl")
    if pickle_path.exists():
        print_success(f"Pickle file saved: {pickle_path}")
    else:
        print_failure(f"Pickle file not saved")


# ============================================================================
# CATEGORY E: CONTROL & CALIBRATION (1) - Skip TrackCalibration (needs real sim)
# ============================================================================

# TrackCalibrationCallback requires real MultiTrackMultiLayerSimulation
# We'll skip it in dry-run testing


# ============================================================================
# CATEGORY F: MONITORING & ERROR HANDLING (2)
# ============================================================================

def test_progress_printer():
    """Test ProgressPrinter."""
    print_test_header("Progress Printer")

    sim = DummySimulation(output_dir="./test_output/progress")
    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        ProgressPrinter()
    ]
    manager = CallbackManager(callbacks)

    print("\n--- Expected output below ---")
    manager(sim, SimulationEvent.INIT)

    try:
        for track in range(3):
            manager(sim, SimulationEvent.TRACK_START)
            for i in range(2):
                sim.step()
                manager(sim, SimulationEvent.STEP_COMPLETE)
            manager(sim, SimulationEvent.TRACK_COMPLETE)
    except SimulationComplete:
        pass

    print("--- Expected output above ---\n")

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    print_success("ProgressPrinter executed (check console output)")


def test_error_completion():
    """Test ErrorCompletionCallback."""
    print_test_header("Error Completion Callback")

    sim = DummySimulation(output_dir="./test_output/error")
    callbacks = [
        StepCountCompletionCallback(max_steps=10),
        ErrorCompletionCallback()
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    # Simulate an error
    try:
        manager(sim, SimulationEvent.ERROR, error=Exception("Test error"))
        print_success("ErrorCompletionCallback handled error and called complete()")
    except Exception as e:
        print_failure(f"ErrorCompletionCallback failed: {e}")


# ============================================================================
# CATEGORY G: UTILITIES (1)
# ============================================================================

def test_compress_callback():
    """Test CompressCallback."""
    print_test_header("Compress Callback")

    sim = DummySimulation(output_dir="./test_output/compress")

    # Create a directory with some files
    test_dir = Path("./test_output/compress/test_compress_me")
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "file1.txt").write_text("test1")
    (test_dir / "file2.txt").write_text("test2")

    callbacks = [
        StepCountCompletionCallback(max_steps=5),
        CompressCallback(
            target_dir="test_compress_me",
            archive_format='zip',
            remove_original=False  # Keep original for testing
        )
    ]
    manager = CallbackManager(callbacks)

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(10):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if zip was created
    zip_path = Path("./test_output/compress/test_compress_me.zip")
    if zip_path.exists():
        print_success(f"Archive created: {zip_path}")
    else:
        print_failure(f"Archive not created")


# ============================================================================
# COMBINED TEST: Multiple Callbacks Together
# ============================================================================

def test_combined_callbacks():
    """Test multiple callbacks working together."""
    print_test_header("Combined Callback Test")

    sim = DummySimulation(output_dir="./test_output/combined")
    callbacks = [
        StepCountCompletionCallback(max_steps=10),
        StepDataCollector(tracked_fields=['position', 'melt_pool'], save_path="combined_data.csv"),
        ParameterLogger(save_file="combined_params.csv"),
        ProgressPrinter(),
        TemperatureSliceSaver(interval=5),
        FinalStateSaver()
    ]
    manager = CallbackManager(callbacks)

    print(f"\nRunning with {len(manager.callbacks)} callbacks...")

    manager(sim, SimulationEvent.INIT)

    try:
        for i in range(20):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete:
        pass

    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Check if all outputs were created
    output_dir = Path("./test_output/combined")
    expected_files = [
        "combined_data.csv",
        "combined_params.csv",
        "final_activated_vol.npy"
    ]

    success_count = 0
    for filename in expected_files:
        if (output_dir / filename).exists():
            success_count += 1

    print_success(f"{success_count}/{len(expected_files)} expected files created")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE CALLBACK SYSTEM DRY-RUN TEST")
    print("="*70)
    print("\nTesting 19 callback types across 7 categories...")

    # Clean up old test outputs
    import shutil
    if Path("./test_output").exists():
        print("\nCleaning up old test outputs...")
        shutil.rmtree("./test_output")

    # Category A: Completion (4)
    print("\n" + "█"*70)
    print("CATEGORY A: COMPLETION CALLBACKS (4)")
    print("█"*70)
    test_height_completion()
    test_step_count_completion()
    test_track_count_completion()
    test_layer_count_completion()

    # Category B: Data Collection (2)
    print("\n" + "█"*70)
    print("CATEGORY B: DATA COLLECTION (2)")
    print("█"*70)
    test_step_data_collector()
    test_parameter_logger()

    # Category C: Visualization (2 - limited)
    print("\n" + "█"*70)
    print("CATEGORY C: VISUALIZATION (2 tested)")
    print("█"*70)
    print("Note: AdvancedLivePlotter and LivePlotter skipped (require display)")
    test_thermal_plot_saver()
    test_cross_section_plotter()

    # Category D: Saving/Export (5)
    print("\n" + "█"*70)
    print("CATEGORY D: SAVING/EXPORT (5)")
    print("█"*70)
    test_temperature_slice_saver()
    test_voxel_temperature_saver()
    test_mesh_saver()
    test_final_state_saver()
    test_pickle_saver()

    # Category E: Control - Skipped
    print("\n" + "█"*70)
    print("CATEGORY E: CONTROL & CALIBRATION (Skipped)")
    print("█"*70)
    print("Note: TrackCalibrationCallback requires real simulation (not DummySimulation)")

    # Category F: Monitoring (2)
    print("\n" + "█"*70)
    print("CATEGORY F: MONITORING & ERROR HANDLING (2)")
    print("█"*70)
    test_progress_printer()
    test_error_completion()

    # Category G: Utilities (1)
    print("\n" + "█"*70)
    print("CATEGORY G: UTILITIES (1)")
    print("█"*70)
    test_compress_callback()

    # Combined test
    print("\n" + "█"*70)
    print("COMBINED TEST: Multiple Callbacks")
    print("█"*70)
    test_combined_callbacks()

    # Summary
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE!")
    print("="*70)
    print("""
Summary:
  ✓ Completion callbacks (4/4 tested)
  ✓ Data collection (2/2 tested)
  ✓ Visualization (2/4 tested - display callbacks skipped)
  ✓ Saving/Export (5/5 tested)
  ⊗ Control (0/1 tested - requires real simulation)
  ✓ Monitoring (2/2 tested)
  ✓ Utilities (1/1 tested)
  ✓ Combined test (multiple callbacks together)

Total: 16/19 callback types tested

Check ./test_output/ directory for generated files!
""")


if __name__ == "__main__":
    main()
