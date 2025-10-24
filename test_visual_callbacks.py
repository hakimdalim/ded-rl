"""
Test script to run simulation with new visual callbacks.
Tests: PerspectiveCameraCallback, AdvancedLivePlotter, HDF5 savers
"""

from simulate import SimulationRunner
from callbacks.completion_callbacks import HeightCompletionCallback
from callbacks.callback_collection import ProgressPrinter
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.live_plotter_callback import AdvancedLivePlotter
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.hdf5_activation_saver import HDF5ActivationSaver

if __name__ == "__main__":

    # Create minimal callback set (avoid MeshSaver which needs Open3D)
    callbacks = [
        # Core callbacks (needed for simulation to run)
        HeightCompletionCallback(),  # Stop when target height reached
        ProgressPrinter(),            # Print progress to console

        # HDF5 savers for post-processing
        HDF5ThermalSaver(
            save_interval=5,  # Save every 5 steps
            compression='gzip',
            compression_opts=4
        ),

        HDF5ActivationSaver(
            save_interval=5,  # Save every 5 steps
            compression='gzip',
            compression_opts=4
        ),

        # Perspective camera following the nozzle (with overlay) ### TODO: fix overlay bug , rework to 2D overlay
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.12, 0.04),  # 12cm behind, 4cm above nozzle
            floor_angle_deg=30.0,
            fov_y_deg=45.0,
            save_images=True,
            save_dir="cam_imgs",  # Short name to avoid path issues
            interval=5,  # Save every 5 steps
            dpi=150,
            enable_overlay=True,  # Enable overlay (nozzle + particles)
            resolution_wh=(800, 600)  # Higher resolution for better quality
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
    print("  1. HDF5ThermalSaver - Save thermal fields to HDF5")
    print("  2. HDF5ActivationSaver - Save activation volumes to HDF5")
    print("  3. PerspectiveCameraCallback - Following camera with thermal view")
    print("  4. AdvancedLivePlotter - Live thermal plotting (commented out)")
    print("\nOutput will be saved to:")
    print(f"  {runner.simulation.output_dir}")
    print("="*70 + "\n")

    # Run simulation
    try:
        runner.run()

        print("\n" + "="*70)
        print(" All callbacks executed without errors")
        print("="*70)
        print("\nCheck the output directory for:")
        print("  - thermal_fields.h5 (HDF5 thermal data)")
        print("  - activation_volumes.h5 (HDF5 activation data)")
        print("  - cam_imgs/ (perspective camera images)")
        print(f"\nOutput directory: {runner.simulation.output_dir}")

    except Exception as e:
        print("\n" + "="*70)
        print("ERROR during simulation")
        print("="*70)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
