"""
Test script to run simulation with new visual callbacks.
Tests: CameraCallback (perspective/orthographic), AdvancedLivePlotter, HDF5 savers
"""

from simulate import SimulationRunner
from callbacks.completion_callbacks import HeightCompletionCallback
from callbacks.callback_collection import ProgressPrinter
from callbacks.camera_callback import CameraCallback
from callbacks.camera_live_plotter_callback import CameraLivePlotterCallback

import matplotlib

matplotlib.use('TkAgg')  # Force external window

def test_camera_callback_perspective():
    """Test CameraCallback with perspective camera mode."""

    # Create minimal callback set (avoid MeshSaver which needs Open3D)
    callbacks = [
        # Core callbacks (needed for simulation to run)
        HeightCompletionCallback(),  # Stop when target height reached
        ProgressPrinter(),            # Print progress to console

        # HDF5 savers for post-processing
        #HDF5ThermalSaver(
        #    save_interval=5,  # Save every 5 steps
        #    compression='gzip',
        #    compression_opts=4
        #),

        #HDF5ActivationSaver(
        #    save_interval=5,  # Save every 5 steps
        #    compression='gzip',
        #    compression_opts=4
        #),

        # Perspective camera following the nozzle
        CameraCallback(
            camera_type="perspective",  # Use perspective camera
            offset=(-0.0262, 0.0, 0.02198),  # -26.20 mm behind, 21.98 mm above target point
            #offset=(0.0, -0.0262, 0.02198),  # -26.20 mm behind, 21.98 mm above target point
            fov_y_deg=45.0,
            plane_size=(0.06151, 0.04613),  # Sensor plane size (meters)
            resolution_wh=(int(640/4), int(480/4)), # originally 640x480, downsampled
            target_window_mm=(20.0, 20.0),  # Crop to 2.0cm × 2.0cm window around nozzle
            save_images=False,
            save_dir="cam_imgs",  # Short name to avoid path issues
            interval=1,  # Save every 5 steps
            dpi=150,
            cmap="hot",
            ambient_temp=300.0,
            fill_gaps=False,
        ),

        # Camera live plotter - displays camera view in real-time
        CameraLivePlotterCallback(
            interval=1,  # Update every step
            temp_range=(300, 2500),
            figsize=(10, 8),
            enabled=True  # Set to False if no display available
        ),

        # Advanced live plotter (disable if running headless)
        # AdvancedLivePlotter(
        #     interval=10,
        #     temp_range=(300, 2500),
        #     enabled=True  # Set to False if no display available
        # ),
    ]

    # Create small test simulation
    runner = SimulationRunner.from_human_units(
        # Small build for quick testing
        build_volume_mm=(20.0, 20.0, 15.0),
        part_volume_mm=(5.0, 5.0, 2.0),  # Just 2mm tall for quick test

        # Simulation parameters
        voxel_size_um=200.0,
        delta_t_ms=200.0,
        scan_speed_mm_s=3.0,
        laser_power_W=600.0,
        powder_feed_g_min=2.0,

        # Output
        output_base_dir="_experiments",
        experiment_label="test_visual_callbacks",

        # Callbacks
        callbacks=callbacks
    )

    print("\n" + "="*70)
    print("Testing New Visual Callbacks")
    print("="*70)
    print("\nCallbacks being tested:")
    print("  1. HDF5ThermalSaver - Save thermal fields to HDF5 (commented out)")
    print("  2. HDF5ActivationSaver - Save activation volumes to HDF5 (commented out)")
    print("  3. CameraCallback - Following camera with thermal view (perspective mode)")
    print("  4. CameraLivePlotterCallback - Live camera view display")
    print("  5. AdvancedLivePlotter - Live thermal plotting (commented out)")
    print("\nOutput will be saved to:")
    print(f"  {runner.simulation.output_dir}")
    print("="*70 + "\n")

    # Run simulation
    runner.run()

    print("\n" + "="*70)
    print("✓ All callbacks executed without errors")
    print("="*70)
    print("\nCheck the output directory for:")
    print("  - thermal_fields.h5 (HDF5 thermal data) - if enabled")
    print("  - activation_volumes.h5 (HDF5 activation data) - if enabled")
    print("  - cam_imgs/ (perspective camera images)")
    print("\nNote: CameraLivePlotterCallback displayed real-time camera view")
    print(f"\nOutput directory: {runner.simulation.output_dir}")

    # Assertions to make it a proper test
    assert runner.simulation is not None, "Simulation should be initialized"
    assert runner.simulation.output_dir.exists(), "Output directory should exist"


if __name__ == "__main__":

    # Allow direct execution as well
    test_camera_callback_perspective()