"""
Test: Camera Overlay WITH LABELS
Shows what each element is with arrows and annotations
"""

from simulate import SimulationRunner
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.completion_callbacks import HeightCompletionCallback

print("\n" + "="*70)
print("  LABELED CAMERA OVERLAY TEST")
print("="*70)
print("\nThis will create annotated images showing:")
print("  ➤ Nozzle (with label + arrow)")
print("  ➤ Powder Stream (with label + arrow)")
print("  ➤ Thermal Field / Melt Pool (with label + arrow)")
print("  ➤ Background (with label + arrow)")
print("  ➤ Color Legend (in corner)")
print("\nRunning quick test (1mm part, ~1 minute)...")
print("="*70 + "\n")

# Camera with overlay AND labels
camera = PerspectiveCameraCallback(
    rel_offset_local=(0.06, -0.10, 0.05),
    floor_angle_deg=28.0,
    enable_overlay=True,
    save_images=True,
    save_dir="labeled_overlay",
    interval=5,
    overlay_config={
        'num_particles': 600,
        'particle_color': (255, 255, 255),  # White
        'particle_size_px': 8,
        'particle_alpha': 255,
        'particle_glow': False,
        'nozzle_fill_color': (30, 60, 100),  # Dark blue
        'nozzle_fill_alpha': 230,
        'add_labels': True,  # ← ENABLE LABELS!
    }
)

# Quick test
runner = SimulationRunner.from_human_units(
    part_volume_mm=(1.0, 1.0, 1.0),
    voxel_size_um=200.0,
    delta_t_ms=200.0,
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    powder_feed_g_min=2.0,
    experiment_label="labeled_test",
    callbacks=[HeightCompletionCallback(), camera]
)

runner.run()

print("\n" + "="*70)
print("  LABELED IMAGES CREATED!")
print("="*70)
print(f"\nOutput: {runner.simulation.output_dir}/labeled_overlay/")
print("\nEach image now has:")
print("  ✓ Arrows pointing to each element")
print("  ✓ Text labels explaining what you see")
print("  ✓ Color legend in corner")
print("  ✓ Clear identification of all components")
print("="*70 + "\n")
