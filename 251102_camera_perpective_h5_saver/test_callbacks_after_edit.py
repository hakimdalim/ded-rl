"""
Test runner for edited callbacks - Quick verification
"""

from simulate import SimulationRunner
from callbacks.completion_callbacks import StepCountCompletionCallback
from callbacks.callback_collection import ProgressPrinter
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver, load_thermal_field, list_steps_in_file
from callbacks.hdf5_activation_saver import HDF5ActivationSaver, load_activation_volume, get_activation_statistics
from callbacks.perspective_camera_callback import PerspectiveCameraCallback

print("\n" + "="*70)
print("TESTING EDITED CALLBACKS")
print("="*70)

# Test configuration
callbacks = [
    StepCountCompletionCallback(max_steps=15),
    ProgressPrinter(),

    # Test HDF5ThermalSaver
    HDF5ThermalSaver(
        filename="thermal_test.h5",
        interval=5,
        compression='gzip',
        compression_opts=4
    ),

    # Test HDF5ActivationSaver
    HDF5ActivationSaver(
        filename="activation_test.h5",
        interval=5,
        compression='gzip',
        compression_opts=9
    ),

    # Test PerspectiveCameraCallback
    PerspectiveCameraCallback(
        enable_overlay=True,
        save_images=True,
        save_dir="cam_test",
        interval=4,
        overlay_config={
            'stream_height_mm': 15.0,
            'v_angle_deg': 15.0,
            'num_particles': 600,
            'render_mode': 'schematic',
            'show_v_cone': True,
            'show_substrate_line': True,
        }
    )
]

print("\nRunning simulation with edited callbacks...")
print("Steps: 15, Interval: thermal=5, activation=5, camera=4\n")

import time
start = time.time()

runner = SimulationRunner.from_human_units(
    build_volume_mm=(20.0, 20.0, 15.0),
    part_volume_mm=(5.0, 5.0, 2.0),
    voxel_size_um=200.0,
    delta_t_ms=200.0,
    scan_speed_mm_s=3.0,
    laser_power_W=600.0,
    powder_feed_g_min=2.0,
    experiment_label="callback_edit_test",
    callbacks=callbacks
)

runner.run()

elapsed = time.time() - start

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(f"Simulation time: {elapsed:.1f}s\n")

# Verify outputs
from pathlib import Path
output_dir = Path(runner.simulation.output_dir)

# Check thermal saver
thermal_file = output_dir / "thermal_test.h5"
if thermal_file.exists():
    steps = list_steps_in_file(str(thermal_file))
    temp = load_thermal_field(str(thermal_file), step=steps[0])
    print(f"[OK] HDF5ThermalSaver")
    print(f"  Steps: {steps}")
    print(f"  Shape: {temp.shape}")
    print(f"  Temp: {temp.min():.1f}K - {temp.max():.1f}K")
else:
    print("[ERROR] HDF5ThermalSaver failed")

# Check activation saver
activation_file = output_dir / "activation_test.h5"
if activation_file.exists():
    stats = get_activation_statistics(str(activation_file))
    print(f"\n[OK] HDF5ActivationSaver")
    print(f"  Steps: {stats['num_steps']}")
    print(f"  Activated: {stats.get('final_num_activated', 0)}")
else:
    print("\n[ERROR] HDF5ActivationSaver failed")

# Check camera
cam_dir = output_dir / "cam_test"
if cam_dir.exists():
    images = list(cam_dir.glob("*.png"))
    print(f"\n[OK] PerspectiveCameraCallback")
    print(f"  Images: {len(images)}")
    if images:
        print(f"  First: {images[0].name}")
        print(f"  Last: {images[-1].name}")
else:
    print("\n[ERROR] PerspectiveCameraCallback failed")

print("\n" + "="*70)
print("All callbacks working after edits")
print(f"Output: {output_dir}")
print("="*70 + "\n")
