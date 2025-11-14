"""
Test V3 Improvements: Realistic DED Overlay
- Thermal colors preserved (red/yellow melt pool visible)
- Particle edges (black outlines for visibility)
- V-cone edge lines (shows powder cone structure)
- Better match to real DED process
"""

from simulate import SimulationRunner
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.completion_callbacks import StepCountCompletionCallback

print("\n" + "="*70)
print("  V3 IMPROVED OVERLAY TEST")
print("="*70)
print("\nImprovements:")
print("  [OK] Thermal colors preserved (hot melt pool = red/yellow/white)")
print("  [OK] Particle edges added (black outlines for visibility)")
print("  [OK] V-cone edge lines (shows powder cone structure)")
print("  [OK] Better match to real DED process")
print("\nRunning: 2mm x 2mm x 0.5mm part, 60 steps (~2-3 minutes)")
print("="*70 + "\n")

# Camera with improved overlay
camera = PerspectiveCameraCallback(
    rel_offset_local=(0.06, -0.10, 0.05),
    floor_angle_deg=28.0,
    enable_overlay=True,
    save_images=True,
    save_dir="improved_overlay",
    interval=3,
    overlay_config={
        'num_particles': 600,
        'particle_color': (255, 255, 255),  # White
        'particle_size_px': 8,
        'particle_alpha': 255,
        'particle_glow': False,
        'particle_edges': True,  # NEW: Black outlines
        'show_v_cone': True,  # NEW: V-cone edge lines
        'nozzle_fill_color': (30, 60, 100),  # Dark blue
        'nozzle_fill_alpha': 230,
        'add_labels': True,  # With labels for clarity
    }
)

# Run simulation
runner = SimulationRunner.from_human_units(
    part_volume_mm=(2.0, 2.0, 0.5),
    voxel_size_um=200.0,
    delta_t_ms=150.0,
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    powder_feed_g_min=2.0,
    experiment_label="v3_improved",
    callbacks=[
        StepCountCompletionCallback(60),
        camera
    ]
)

print("Starting simulation...\n")
runner.run()

print("\n" + "="*70)
print("  V3 IMPROVED OVERLAY COMPLETE!")
print("="*70)
print(f"\nOutput: {runner.simulation.output_dir}/improved_overlay/")
print("\nImprovements you should see:")
print("  1. Thermal colors (red/yellow/white for hot melt pool)")
print("  2. Particle outlines (black edges on white particles)")
print("  3. V-cone lines (white lines showing powder cone edges)")
print("  4. Nozzle visible (dark blue on light gray background)")
print("  5. Much more realistic DED visualization!")
print("="*70 + "\n")
