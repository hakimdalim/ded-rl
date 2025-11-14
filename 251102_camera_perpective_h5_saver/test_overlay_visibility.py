"""
Quick test: Verify camera overlay is visible with new light gray background
"""

from simulate import SimulationRunner
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.completion_callbacks import HeightCompletionCallback

print("\n" + "="*70)
print("  TESTING: Camera Overlay Background Visibility")
print("="*70)
print("\nChange: Black background → Light gray background")
print("Expected: Dark blue nozzle should now be clearly visible")
print("\nRunning quick test (1mm part, ~1 minute)...")
print("="*70 + "\n")

# Single camera with overlay
camera = PerspectiveCameraCallback(
    rel_offset_local=(0.06, -0.10, 0.05),
    floor_angle_deg=28.0,
    enable_overlay=True,  # WITH overlay (nozzle + powder)
    save_images=True,
    save_dir="test_overlay_visibility",
    interval=5,
    overlay_config={
        'num_particles': 600,
        'particle_color': (255, 255, 255),  # White
        'particle_size_px': 8,
        'particle_alpha': 255,
        'particle_glow': False,  # No glow for cleaner look
        'nozzle_fill_color': (30, 60, 100),  # Dark blue (original)
        'nozzle_fill_alpha': 230,
    }
)

# Quick test
runner = SimulationRunner.from_human_units(
    part_volume_mm=(1.0, 1.0, 1.0),  # Very small
    voxel_size_um=200.0,
    delta_t_ms=200.0,
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    powder_feed_g_min=2.0,
    experiment_label="overlay_test",
    callbacks=[HeightCompletionCallback(), camera]
)

runner.run()

print("\n" + "="*70)
print("  TEST COMPLETE!")
print("="*70)
print(f"\nCheck output: {runner.simulation.output_dir}/test_overlay_visibility/")
print("\nExpected result:")
print("  - Background: Light gray (instead of black)")
print("  - Nozzle: Dark blue (clearly visible)")
print("  - Powder: White particles (visible)")
print("  - Melt pool: Hot thermal colors (red/yellow)")
print("="*70 + "\n")
