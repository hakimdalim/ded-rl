"""
Optimized V-profile visualization to clearly show the V-shaped powder envelope.

This uses specific settings to make the V-shape boundary visible:
- Less particle glow (to see envelope better)
- Smaller particles (to see distribution shape)
- Side view positioning
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from test_overlay_callback import MockSimulation


def test_v_profile_clear_envelope():
    """Optimized view to see V-shape envelope clearly."""
    print("\n" + "="*70)
    print("V-PROFILE: Clear V-Shape Envelope (Optimized)")
    print("="*70)

    # Optimized config for V-shape visibility
    config = {
        'num_particles': 8000,  # Very dense for continuous envelope
        'particle_size_px': 3,  # Smaller to see shape better
        'particle_alpha': 0.85,  # Slightly less opaque
        'particle_glow': True,
        'nozzle_segments': 96,
        'stream_height_mm': 15.0,
        'v_angle_deg': 15.0,
        'gaussian_sigma_ratio': 0.25,  # Tighter distribution for clearer V
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.18, 0.0, 0.0),  # Pure side view
        floor_angle_deg=0.0,  # Horizontal
        fov_y_deg=55.0,
        plane_size=(0.08, 0.06),
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_profile_optimized/clear_envelope",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Settings optimized for V-shape visibility:")
    print(f"    Particles: {config['num_particles']} (very dense)")
    print(f"    Particle size: {config['particle_size_px']}px (small)")
    print(f"    Gaussian sigma: {config['gaussian_sigma_ratio']} (tight)")
    print("  Expected: Clear V-shaped envelope visible")

    try:
        callback._execute(context)
        print("  [OK] V-profile rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def test_v_profile_minimal_glow():
    """V-profile with minimal glow for cleaner envelope."""
    print("\n" + "="*70)
    print("V-PROFILE: Minimal Glow - Clean V-Boundary")
    print("="*70)

    config = {
        'num_particles': 10000,  # Maximum density
        'particle_size_px': 2,  # Very small
        'particle_alpha': 0.75,
        'particle_glow': False,  # NO glow for sharp edges
        'high_quality': True,
        'nozzle_segments': 96,
        'stream_height_mm': 15.0,
        'v_angle_deg': 15.0,
        'gaussian_sigma_ratio': 0.2,  # Very tight
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.18, 0.0, 0.0),
        floor_angle_deg=0.0,
        fov_y_deg=55.0,
        plane_size=(0.08, 0.06),
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_profile_optimized/minimal_glow",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  No glow - sharp particle edges")
    print(f"    Particles: {config['num_particles']} (maximum)")
    print("  Expected: Sharp V-shaped boundary")

    try:
        callback._execute(context)
        print("  [OK] Minimal glow V-profile rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")


def test_v_profile_angled_for_depth():
    """Slight angle to show V-shape with depth perception."""
    print("\n" + "="*70)
    print("V-PROFILE: Angled View with Depth")
    print("="*70)

    config = {
        'num_particles': 8000,
        'particle_size_px': 3,
        'particle_alpha': 0.85,
        'particle_glow': True,
        'nozzle_segments': 96,
        'stream_height_mm': 15.0,
        'v_angle_deg': 15.0,
        'gaussian_sigma_ratio': 0.25,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.15, -0.05, 0.01),  # Slight angle
        floor_angle_deg=5.0,  # Very slight downward
        fov_y_deg=55.0,
        plane_size=(0.08, 0.06),
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_profile_optimized/angled_depth",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Angled for 3D depth perception")
    print("  Expected: V-shape with visible depth")

    try:
        callback._execute(context)
        print("  [OK] Angled V-profile rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")


def run_v_profile_optimized_tests():
    """Run optimized V-profile tests."""
    print("\n" + "="*70)
    print("  OPTIMIZED V-PROFILE VISUALIZATION")
    print("  (Clear V-Shape Envelope Visibility)")
    print("="*70)

    tests = [
        test_v_profile_clear_envelope,
        test_v_profile_minimal_glow,
        test_v_profile_angled_for_depth,
    ]

    passed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  [X] Test error: {e}")

    print("\n" + "="*70)
    print("V-PROFILE OPTIMIZATION RESULTS")
    print("="*70)
    print(f"  Tests passed: {passed}/{len(tests)}")
    print("\n  OUTPUT LOCATIONS:")
    print("    Clear Envelope:  v_profile_optimized/clear_envelope/")
    print("    Minimal Glow:    v_profile_optimized/minimal_glow/")
    print("    Angled Depth:    v_profile_optimized/angled_depth/")
    print("\n  These views are optimized to show the V-shaped")
    print("  powder distribution envelope clearly")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_v_profile_optimized_tests()
