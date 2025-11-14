"""
Run simulation with NEW callbacks for comparison with original run_default.py

NEW Callbacks being tested:
1. HDF5ThermalSaver - Efficient HDF5 saving for thermal field
2. HDF5ActivationSaver - Efficient HDF5 saving for activation volume
3. PerspectiveCameraCallback - Following camera with V-shaped powder overlay

This runs the SAME simulation parameters as run_default.py but uses new callbacks.
"""

from simulate import SimulationRunner

# Import ORIGINAL callbacks for comparison
from callbacks.callback_collection import (
    ProgressPrinter,
    FinalStateSaver,
)
from callbacks.completion_callbacks import HeightCompletionCallback
from callbacks.step_data_collector import StepDataCollector

# Import NEW callbacks to test
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.hdf5_activation_saver import HDF5ActivationSaver
from callbacks.perspective_camera_callback import PerspectiveCameraCallback

print("\n" + "="*70)
print("TESTING NEW CALLBACKS - Comparison with Original")
print("="*70)
print("\nNEW Callbacks being tested:")
print("  1. HDF5ThermalSaver - Thermal field (complete voxel volume)")
print("  2. HDF5ActivationSaver - Activation volume (complete voxel volume)")
print("  3. PerspectiveCameraCallback - Following camera with overlay")
print("\nOverlay features:")
print("  - Nozzle outlet (frustum cone)")
print("  - V-shaped powder stream (13-16mm height, configurable)")
print("  - Randomly generated particles with Gaussian distribution")
print("  - Camera follows nozzle position")
print("="*70)

# Create NEW callbacks
new_callbacks = [
    # Completion (same as original)
    HeightCompletionCallback(),

    # Progress monitoring (same as original)
    ProgressPrinter(),

    # Data collection (same as original for comparison)
    StepDataCollector(tracked_fields=None, save_path="simulation_data.csv"),

    # Final state (same as original)
    FinalStateSaver(),

    # NEW: HDF5 Thermal Field Saver
    HDF5ThermalSaver(
        filename="thermal_fields.h5",
        interval=1,  # Save every step (like original TemperatureSliceSaver)
        compression='gzip',
        compression_opts=4
    ),

    # NEW: HDF5 Activation Volume Saver
    HDF5ActivationSaver(
        filename="activation_volumes.h5",
        interval=1,  # Save every step (like original VoxelTemperatureSaver)
        compression='gzip',
        compression_opts=9  # High compression for binary data
    ),

    # NEW: Perspective Camera with Overlay
    PerspectiveCameraCallback(
        # Camera follows nozzle with optimized positioning
        # (defaults are now optimized from our work)
        enable_overlay=True,
        save_images=True,
        save_dir="cam",
        interval=1,  # Save every step
        overlay_config={
            # V-shaped powder stream configuration (as per task requirements)
            'stream_height_mm': 15.0,  # 13-16mm typical range, using 15mm
            'v_angle_deg': 15.0,  # V-opening angle
            'num_particles': 600,  # Random particles
            'gaussian_sigma_ratio': 0.25,  # Gaussian distribution

            # Nozzle geometry
            'nozzle_outlet_radius_mm': 4.0,
            'nozzle_top_radius_mm': 10.0,
            'nozzle_height_mm': 40.0,

            # Schematic mode for clear visualization
            'render_mode': 'schematic',
            'schematic_bg_color': (200, 200, 200),
            'show_substrate_line': True,
            'substrate_line_color': (255, 0, 0),

            # Visual settings
            'particle_color': (50, 50, 50),  # Dark particles on light background
            'nozzle_fill_color': (60, 100, 160),  # Blue nozzle
            'show_v_cone': True,  # Show V-cone edge lines
        }
    )
]

print("\n" + "-"*70)
print("SIMULATION PARAMETERS (same as run_default.py)")
print("-"*70)
print("  Build volume: 20mm x 20mm x 15mm")
print("  Part size: 5mm x 5mm x 5mm")
print("  Voxel size: 200um")
print("  Time step: 200ms")
print("  Scan speed: 3.0 mm/s")
print("  Laser power: 600W")
print("  Powder feed: 2.0 g/min")
print("  Completion: Height-based")
print("-"*70)

print("\n" + "-"*70)
print("EXPECTED OUTPUTS")
print("-"*70)
print("  1. thermal_fields.h5 - HDF5 thermal field data")
print("  2. activation_volumes.h5 - HDF5 activation data")
print("  3. cam/ - Perspective camera images with overlay")
print("  4. simulation_data.csv - Step data (for comparison)")
print("  5. final_state.npz - Final state (for comparison)")
print("-"*70)

# Run with same parameters as run_default.py
runner = SimulationRunner.from_human_units(
    build_volume_mm=(20.0, 20.0, 15.0),
    part_volume_mm=(5.0, 5.0, 5.0),
    voxel_size_um=200.0,
    delta_t_ms=200.0,
    scan_speed_mm_s=3.0,
    laser_power_W=600.0,
    powder_feed_g_min=2.0,
    experiment_label="new_callbacks_test",
    callbacks=new_callbacks
)

print("\nRunning simulation with NEW callbacks...")
print("Estimated time: 2-4 hours (same as original)")
print("="*70 + "\n")

import time
start_time = time.time()

runner.run()

elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)

print("\n" + "="*70)
print("SIMULATION COMPLETED")
print("="*70)
print(f"Time elapsed: {hours}h {minutes}m")
print(f"Results: {runner.simulation.output_dir}")
print("\n" + "-"*70)
print("VERIFICATION CHECKLIST")
print("-"*70)
print("  [  ] thermal_fields.h5 created and readable")
print("  [  ] activation_volumes.h5 created and readable")
print("  [  ] Camera images show nozzle and V-shaped powder")
print("  [  ] simulation_data.csv matches original format")
print("  [  ] final_state.npz created")
print("\nCompare these outputs with original run_default.py results.")
print("="*70 + "\n")
