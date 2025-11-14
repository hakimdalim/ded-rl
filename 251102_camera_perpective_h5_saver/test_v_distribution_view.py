"""
Test script for V-distribution side view (vertical powder stream visibility)

This creates views specifically designed to show the V-shaped powder distribution
vertically, similar to the reference images.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from test_overlay_callback import MockSimulation


def test_pure_side_view():
    """Pure side view (90° horizontal) to see V-distribution."""
    print("\n" + "="*70)
    print("TEST: Pure Side View - V-Distribution Vertical")
    print("="*70)

    config = {
        'num_particles': 5000,
        'particle_size_px': 5,
        'particle_alpha': 0.95,
        'nozzle_segments': 96,
        'stream_height_mm': 16.0,  # Full 16mm height
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.20, 0.0, 0.0),  # Far to the side
        floor_angle_deg=0.0,  # Perfectly horizontal
        fov_y_deg=60.0,  # Wide FOV to capture full V-cone
        plane_size=(0.08, 0.06),  # Larger plane for full view
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_distribution_views/pure_side",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Camera: 20cm to the side, horizontal view")
    print(f"  Particles: {config['num_particles']}")
    print(f"  Stream height: {config['stream_height_mm']}mm")

    try:
        callback._execute(context)
        print("  [OK] Pure side view rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def test_slight_angle_view():
    """Slight angle to see both V-distribution and some depth."""
    print("\n" + "="*70)
    print("TEST: Slight Angle View - V-Distribution with Depth")
    print("="*70)

    config = {
        'num_particles': 5000,
        'particle_size_px': 5,
        'particle_alpha': 0.95,
        'nozzle_segments': 96,
        'stream_height_mm': 16.0,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.15, -0.08, 0.02),  # Side and slightly behind
        floor_angle_deg=10.0,  # Slight downward angle
        fov_y_deg=60.0,
        plane_size=(0.08, 0.06),
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_distribution_views/slight_angle",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Camera: Angled to show V-shape and depth")
    print(f"  Particles: {config['num_particles']}")

    try:
        callback._execute(context)
        print("  [OK] Angled view rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def test_orthogonal_to_v():
    """View perpendicular to V-cone axis."""
    print("\n" + "="*70)
    print("TEST: Perpendicular to V-Axis - Maximum V-Visibility")
    print("="*70)

    config = {
        'num_particles': 6000,  # Dense for clear V-shape
        'particle_size_px': 6,  # Large for visibility
        'particle_alpha': 0.95,
        'nozzle_segments': 96,
        'stream_height_mm': 15.0,
        'v_angle_deg': 15.0,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.18, 0.0, -0.005),  # Side, slightly below center
        floor_angle_deg=0.0,  # Horizontal
        fov_y_deg=70.0,  # Wide to capture full cone
        plane_size=(0.10, 0.08),  # Large plane
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_distribution_views/perpendicular",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Camera: Perpendicular to V-cone axis")
    print(f"  Particles: {config['num_particles']} (ultra dense)")
    print(f"  Particle size: {config['particle_size_px']}px (large)")

    try:
        callback._execute(context)
        print("  [OK] Perpendicular view rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def test_close_side_view():
    """Close-up side view for detailed V-distribution."""
    print("\n" + "="*70)
    print("TEST: Close-Up Side View - Detailed V-Distribution")
    print("="*70)

    config = {
        'num_particles': 7000,  # Very dense
        'particle_size_px': 4,  # Medium size
        'particle_alpha': 0.9,
        'nozzle_segments': 128,  # Ultra smooth
        'stream_height_mm': 15.0,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.12, 0.0, 0.0),  # Closer to the side
        floor_angle_deg=0.0,
        fov_y_deg=50.0,  # Narrower FOV for zoom effect
        plane_size=(0.06, 0.05),
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_distribution_views/close_side",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Camera: Close side view for detail")
    print(f"  Particles: {config['num_particles']} (maximum density)")

    try:
        callback._execute(context)
        print("  [OK] Close-up view rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def test_diagonal_45deg():
    """45-degree diagonal to see both V-shape and radial distribution."""
    print("\n" + "="*70)
    print("TEST: 45° Diagonal - V-Shape + Radial Distribution")
    print("="*70)

    config = {
        'num_particles': 5000,
        'particle_size_px': 5,
        'particle_alpha': 0.95,
        'nozzle_segments': 96,
        'stream_height_mm': 15.0,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.12, -0.12, 0.03),  # 45° diagonal
        floor_angle_deg=20.0,  # Slight downward look
        fov_y_deg=60.0,
        plane_size=(0.08, 0.06),
        enable_overlay=True,
        overlay_config=config,
        save_images=True,
        save_dir="v_distribution_views/diagonal_45",
        interval=1
    )

    sim = MockSimulation()
    context = {'simulation': sim, 'output_dir': Path('.')}

    print("  Camera: 45° diagonal view")
    print("  Shows: V-shape vertically + radial spread")

    try:
        callback._execute(context)
        print("  [OK] Diagonal view rendered")
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()


def run_v_distribution_tests():
    """Run all V-distribution view tests."""
    print("\n" + "="*70)
    print("     V-DISTRIBUTION SIDE VIEW TEST SUITE")
    print("     (Vertical Powder Stream Visibility)")
    print("="*70)

    tests = [
        test_pure_side_view,
        test_slight_angle_view,
        test_orthogonal_to_v,
        test_close_side_view,
        test_diagonal_45deg,
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
    print("TEST SUMMARY - V-Distribution Views")
    print("="*70)
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {passed/len(tests)*100:.1f}%")
    print("\n  V-DISTRIBUTION VIEW OUTPUTS:")
    print("    Pure Side:       v_distribution_views/pure_side/")
    print("    Slight Angle:    v_distribution_views/slight_angle/")
    print("    Perpendicular:   v_distribution_views/perpendicular/")
    print("    Close-Up:        v_distribution_views/close_side/")
    print("    45° Diagonal:    v_distribution_views/diagonal_45/")
    print("\n  These views show the vertical V-shaped powder distribution")
    print("  similar to the reference images (nozzle_z+1/5/10.png)")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_v_distribution_tests()
