"""
Test: Show ACTIVE MELT POOL with hot thermal colors
Runs longer to capture red/yellow/white thermal field
"""

from simulate import SimulationRunner
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.completion_callbacks import StepCountCompletionCallback

print("\n" + "="*70)
print("  ACTIVE MELT POOL VISUALIZATION TEST")
print("="*70)
print("\nThis will run longer to show:")
print("  [HOT] Active melt pool (RED/YELLOW/WHITE hot colors)")
print("  [BLUE] Nozzle moving along track")
print("  [WHITE] Powder stream feeding material")
print("  [TEMP] Temperature gradients (hot -> cool)")
print("\nRunning: 2mm x 2mm x 0.5mm part (~2-3 minutes)")
print("Saving images every 3 steps for detailed visualization")
print("="*70 + "\n")

# Camera with labeled overlay
camera_labeled = PerspectiveCameraCallback(
    rel_offset_local=(0.06, -0.10, 0.05),
    floor_angle_deg=28.0,
    enable_overlay=True,
    save_images=True,
    save_dir="melt_pool_labeled",
    interval=3,  # Save more frequently
    overlay_config={
        'num_particles': 600,
        'particle_color': (255, 255, 255),
        'particle_size_px': 8,
        'particle_alpha': 255,
        'particle_glow': False,
        'nozzle_fill_color': (30, 60, 100),
        'nozzle_fill_alpha': 230,
        'add_labels': True,  # WITH labels
    }
)

# Clean version without labels (for comparison)
camera_clean = PerspectiveCameraCallback(
    rel_offset_local=(0.06, -0.10, 0.05),
    floor_angle_deg=28.0,
    enable_overlay=True,
    save_images=True,
    save_dir="melt_pool_clean",
    interval=3,
    overlay_config={
        'num_particles': 600,
        'particle_color': (255, 255, 255),
        'particle_size_px': 8,
        'particle_alpha': 255,
        'particle_glow': False,
        'nozzle_fill_color': (30, 60, 100),
        'nozzle_fill_alpha': 230,
        'add_labels': False,  # NO labels (clean)
    }
)

# Medium test - runs longer to build up heat
runner = SimulationRunner.from_human_units(
    part_volume_mm=(2.0, 2.0, 0.5),  # Wider but shorter (faster)
    voxel_size_um=200.0,
    delta_t_ms=150.0,  # Slightly faster timestep
    laser_power_W=600.0,
    scan_speed_mm_s=3.0,
    powder_feed_g_min=2.0,
    experiment_label="melt_pool_demo",
    callbacks=[
        StepCountCompletionCallback(60),  # Run 60 steps (plenty for thermal buildup)
        camera_labeled,
        camera_clean
    ]
)

print("Starting simulation...\n")
runner.run()

print("\n" + "="*70)
print("  MELT POOL IMAGES CREATED!")
print("="*70)
print(f"\nOutput: {runner.simulation.output_dir}/")
print("\nTwo sets of images created:")
print("\n1. melt_pool_labeled/")
print("   - WITH arrows and labels")
print("   - Shows what each element is")
print("   - Best for understanding/teaching")
print("\n2. melt_pool_clean/")
print("   - WITHOUT labels (clean)")
print("   - Full thermal field visible")
print("   - Best for videos/analysis")
print("\nBoth show:")
print("  [HOT] Hot melt pool (white/yellow/red)")
print("  [COOL] Cooling tracks (orange/dark red)")
print("  [BLUE] Nozzle geometry (dark blue)")
print("  [WHITE] Powder stream (white particles)")
print("  [GRAY] Background (light gray)")
print("="*70 + "\n")
