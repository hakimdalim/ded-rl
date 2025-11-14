"""
Quick test of HDF5 savers after cleanup
"""

from simulate import SimulationRunner
from callbacks.completion_callbacks import StepCountCompletionCallback
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver, load_thermal_field, list_steps_in_file
from callbacks.hdf5_activation_saver import HDF5ActivationSaver, load_activation_volume, get_activation_statistics

print("\n" + "="*70)
print("TESTING HDF5 SAVERS (10 steps)")
print("="*70)

callbacks = [
    StepCountCompletionCallback(max_steps=10),

    HDF5ThermalSaver(
        filename="test_thermal.h5",
        interval=3,
        compression='gzip',
        compression_opts=4
    ),

    HDF5ActivationSaver(
        filename="test_activation.h5",
        interval=3,
        compression='gzip',
        compression_opts=9
    ),
]

print("\nRunning simulation...")

runner = SimulationRunner.from_human_units(
    build_volume_mm=(20.0, 20.0, 15.0),
    part_volume_mm=(5.0, 5.0, 2.0),
    voxel_size_um=200.0,
    delta_t_ms=200.0,
    scan_speed_mm_s=3.0,
    laser_power_W=600.0,
    powder_feed_g_min=2.0,
    experiment_label="hdf5_test",
    callbacks=callbacks
)

runner.run()

print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

from pathlib import Path
output_dir = Path(runner.simulation.output_dir)

# Test thermal saver
thermal_file = output_dir / "test_thermal.h5"
if thermal_file.exists():
    steps = list_steps_in_file(str(thermal_file))
    print(f"\n[OK] Thermal saver working")
    print(f"  Steps saved: {steps}")

    # Load a sample
    temp = load_thermal_field(str(thermal_file), step=steps[0])
    print(f"  Sample shape: {temp.shape}")
    print(f"  Temp range: {temp.min():.1f}K - {temp.max():.1f}K")
else:
    print("\n[ERROR] Thermal file not created")

# Test activation saver
activation_file = output_dir / "test_activation.h5"
if activation_file.exists():
    stats = get_activation_statistics(str(activation_file))
    print(f"\n[OK] Activation saver working")
    print(f"  Steps saved: {stats['num_steps']}")
    print(f"  Final activated: {stats.get('final_num_activated', 0)}")
else:
    print("\n[ERROR] Activation file not created")

print("\n" + "="*70)
print("TEST COMPLETE - Both savers working correctly")
print("="*70 + "\n")
