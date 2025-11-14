"""
Test script for V2 Enhanced Quality Overlay

Compares V1 (basic) vs V2 (enhanced quality) rendering
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from test_overlay_callback import MockSimulation


def test_v1_basic():
    """Test V1 (original) rendering."""
    print("\n" + "="*70)
    print("V1 (Original) - Basic Quality")
    print("="*70)

    # V1 configuration (basic)
    v1_config = {
        'num_particles': 2000,
        'nozzle_segments': 32,
        'particle_alpha': 0.7,
        'particle_size_px': 2,
        'particle_glow': False,
        'high_quality': False,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        enable_overlay=True,
        overlay_config=v1_config,
        save_images=True,
        save_dir="test_output_v2/v1_basic",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print("  Configuration:")
    print(f"    Particles: {v1_config['num_particles']}")
    print(f"    Particle size: {v1_config['particle_size_px']}px")
    print(f"    Particle glow: {v1_config['particle_glow']}")
    print(f"    Nozzle segments: {v1_config['nozzle_segments']}")
    print(f"    High quality: {v1_config['high_quality']}")

    try:
        callback._execute(context)
        print("  [OK] V1 rendered successfully")
    except Exception as e:
        print(f"  [X] V1 failed: {e}")
        import traceback
        traceback.print_exc()


def test_v2_enhanced():
    """Test V2 (enhanced) rendering."""
    print("\n" + "="*70)
    print("V2 (Enhanced) - High Quality")
    print("="*70)

    # V2 uses defaults from callback (already enhanced)
    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        enable_overlay=True,
        # Using V2 defaults (no overlay_config needed)
        save_images=True,
        save_dir="test_output_v2/v2_enhanced",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print("  Configuration:")
    print(f"    Particles: {callback.overlay_config['num_particles']}")
    print(f"    Particle size: {callback.overlay_config['particle_size_px']}px")
    print(f"    Particle glow: {callback.overlay_config['particle_glow']}")
    print(f"    Nozzle segments: {callback.overlay_config['nozzle_segments']}")
    print(f"    High quality: {callback.overlay_config['high_quality']}")

    try:
        callback._execute(context)
        print("  [OK] V2 rendered successfully")
    except Exception as e:
        print(f"  [X] V2 failed: {e}")
        import traceback
        traceback.print_exc()


def test_v2_side_view():
    """Test V2 from side view."""
    print("\n" + "="*70)
    print("V2 Enhanced - Side View")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.15, 0.0, 0.0),  # From the side
        floor_angle_deg=0.0,  # Horizontal
        enable_overlay=True,
        save_images=True,
        save_dir="test_output_v2/v2_side_view",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    try:
        callback._execute(context)
        print("  [OK] Side view rendered")
    except Exception as e:
        print(f"  [X] Side view failed: {e}")


def test_v2_angled_view():
    """Test V2 from angled view (like reference images)."""
    print("\n" + "="*70)
    print("V2 Enhanced - Angled View (45° from side, 35° down)")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.08, -0.08, 0.06),  # Diagonal angle
        floor_angle_deg=35.0,
        enable_overlay=True,
        save_images=True,
        save_dir="test_output_v2/v2_angled",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    try:
        callback._execute(context)
        print("  [OK] Angled view rendered")
    except Exception as e:
        print(f"  [X] Angled view failed: {e}")


def test_v2_ultra_quality():
    """Test V2 with maximum quality settings."""
    print("\n" + "="*70)
    print("V2 Enhanced - ULTRA QUALITY (5000 particles, 6px size)")
    print("="*70)

    ultra_config = {
        'num_particles': 5000,
        'particle_size_px': 6,
        'particle_alpha': 0.95,
        'nozzle_segments': 128,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        enable_overlay=True,
        overlay_config=ultra_config,
        save_images=True,
        save_dir="test_output_v2/v2_ultra",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print("  Configuration:")
    print(f"    Particles: {ultra_config['num_particles']}")
    print(f"    Particle size: {ultra_config['particle_size_px']}px")
    print(f"    Nozzle segments: {ultra_config['nozzle_segments']}")

    try:
        callback._execute(context)
        print("  [OK] Ultra quality rendered")
    except Exception as e:
        print(f"  [X] Ultra quality failed: {e}")


def run_v2_tests():
    """Run all V2 comparison tests."""
    print("\n" + "="*70)
    print("          V2 ENHANCED QUALITY TEST SUITE")
    print("="*70)

    tests = [
        test_v1_basic,
        test_v2_enhanced,
        test_v2_side_view,
        test_v2_angled_view,
        test_v2_ultra_quality,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  [X] Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {passed/len(tests)*100:.1f}%")
    print("\n  COMPARISON:")
    print("    V1 Basic:      test_output_v2/v1_basic/thermal_step_0000.png")
    print("    V2 Enhanced:   test_output_v2/v2_enhanced/thermal_step_0000.png")
    print("    V2 Side:       test_output_v2/v2_side_view/thermal_step_0000.png")
    print("    V2 Angled:     test_output_v2/v2_angled/thermal_step_0000.png")
    print("    V2 Ultra:      test_output_v2/v2_ultra/thermal_step_0000.png")
    print("\n  Compare these with reference images:")
    print("    nozzle_z+1.png, nozzle_z+5.png, nozzle_z+10.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_v2_tests()
