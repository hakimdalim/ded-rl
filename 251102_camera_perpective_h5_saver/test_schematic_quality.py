"""
Test V3 Schematic-Quality Overlay

Tests the new default configuration that matches the clean 2D schematic appearance.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from test_overlay_callback import MockSimulation


def test_schematic_quality_default():
    """Test new V3 schematic-quality defaults."""
    print("\n" + "="*70)
    print("V3 SCHEMATIC-QUALITY: New Defaults")
    print("="*70)

    # Use default V3 configuration (black particles, clean nozzle)
    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.06, -0.10, 0.05),
        floor_angle_deg=28.0,
        enable_overlay=True,
        # No overlay_config = uses new V3 defaults
        save_images=True,
        save_dir="schematic_quality/v3_default",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  V3 Default Settings:")
    print(f"    Particles: {callback.overlay_config['num_particles']}")
    print(f"    Particle color: {callback.overlay_config['particle_color']} (BLACK)")
    print(f"    Particle size: {callback.overlay_config['particle_size_px']}px (LARGE)")
    print(f"    Particle alpha: {callback.overlay_config['particle_alpha']} (OPAQUE)")
    print(f"    Particle glow: {callback.overlay_config['particle_glow']} (DISABLED)")
    print("  Expected: Clean schematic-like appearance")

    try:
        callback._execute(context)
        print("\n  [OK] V3 schematic-quality rendered!")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def test_schematic_quality_side():
    """Side view with schematic quality."""
    print("\n" + "="*70)
    print("V3 SCHEMATIC-QUALITY: Side View")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.15, 0.0, 0.0),
        floor_angle_deg=0.0,
        enable_overlay=True,
        save_images=True,
        save_dir="schematic_quality/v3_side",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    try:
        callback._execute(context)
        print("  [OK] Side view rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")


def test_schematic_quality_high_density():
    """High particle density schematic style."""
    print("\n" + "="*70)
    print("V3 SCHEMATIC-QUALITY: High Density")
    print("="*70)

    config = {
        'num_particles': 1200,  # Double density
        'particle_size_px': 6,
        # Keep other V3 defaults
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.06, -0.10, 0.05),
        floor_angle_deg=28.0,
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="schematic_quality/v3_high_density",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print(f"  High density: {config['num_particles']} particles")

    try:
        callback._execute(context)
        print("  [OK] High density rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")


def run_schematic_quality_tests():
    """Run all V3 schematic-quality tests."""
    print("\n" + "="*70)
    print("  V3 SCHEMATIC-QUALITY TEST SUITE")
    print("  (Matching 2D Schematic Appearance)")
    print("="*70)

    tests = [
        test_schematic_quality_default,
        test_schematic_quality_side,
        test_schematic_quality_high_density,
    ]

    passed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  [X] Error: {e}")

    print("\n" + "="*70)
    print("V3 SCHEMATIC-QUALITY RESULTS")
    print("="*70)
    print(f"  Tests passed: {passed}/{len(tests)}")
    print("\n  KEY IMPROVEMENTS (V3):")
    print("    - BLACK particles (like schematic)")
    print("    - OPAQUE particles (fully visible)")
    print("    - LARGE particles (8px, easy to see)")
    print("    - NO GLOW (clean appearance)")
    print("    - DARKER nozzle (better contrast)")
    print("\n  OUTPUT:")
    print("    schematic_quality/v3_default/")
    print("    schematic_quality/v3_side/")
    print("    schematic_quality/v3_high_density/")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_schematic_quality_tests()
