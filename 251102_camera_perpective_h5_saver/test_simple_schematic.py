"""
Test: Simple Schematic Mode (like reference image)
- Light gray background (no thermal complexity)
- Blue nozzle (clearly visible)
- Dark powder particles (V-cone distribution)
- Red substrate line
- V-cone edge lines
- Clean and simple visualization
"""

from simulate import SimulationRunner
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.completion_callbacks import StepCountCompletionCallback

print("\n" + "="*70)
print("  SIMPLE SCHEMATIC MODE TEST")
print("="*70)
print("\nSimple, clear visualization:")
print("  [OK] Light gray background (solid, no thermal)")
print("  [OK] Blue nozzle (clearly visible)")
print("  [OK] Dark powder particles (V-cone)")
print("  [OK] V-cone edge lines (white)")
print("  [OK] Red substrate line")
print("\nRunning: 2mm x 2mm x 0.5mm part, 60 steps (~2-3 minutes)")
print("="*70 + "\n")

# Simple schematic camera with optimized framing
camera = PerspectiveCameraCallback(
    # Use improved camera positioning (closer, wider FOV, auto-zoom enabled)
    # rel_offset_local and fov_y_deg now use optimized defaults
    floor_angle_deg=28.0,
    enable_overlay=True,
    save_images=True,
    save_dir="simple_schematic",
    interval=3,
    overlay_config={
        'render_mode': 'schematic',  # KEY: Use schematic mode
        'schematic_bg_color': (200, 200, 200),  # Light gray background
        'num_particles': 600,
        'particle_size_px': 6,
        'particle_alpha': 255,
        'particle_glow': False,
        'particle_edges': False,  # Clean dots
        'show_v_cone': True,  # V-cone edge lines
        'nozzle_fill_color': (60, 100, 160),  # Blue nozzle
        'nozzle_fill_alpha': 230,
        'show_substrate_line': True,  # Red substrate line
        'substrate_line_color': (255, 0, 0),  # Red
        'add_labels': False,  # Clean, no labels
        'auto_zoom': True,  # Enable auto-zoom/crop to geometry
        'zoom_padding': 0.15,  # 15% padding around geometry
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
    experiment_label="simple_schematic",
    callbacks=[
        StepCountCompletionCallback(60),
        camera
    ]
)

print("Starting simulation...\n")
runner.run()

print("\n" + "="*70)
print("  SIMPLE SCHEMATIC COMPLETE!")
print("="*70)
print(f"\nOutput: {runner.simulation.output_dir}/simple_schematic/")
print("\nWhat you should see:")
print("  1. Light gray background (clean, simple)")
print("  2. Blue nozzle at top (clearly visible)")
print("  3. Dark powder particles in V-cone")
print("  4. White V-cone edge lines")
print("  5. Red horizontal substrate line")
print("  6. Simple, clear schematic - like reference image!")
print("="*70 + "\n")
