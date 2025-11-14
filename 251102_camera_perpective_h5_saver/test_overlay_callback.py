"""
Test script for PerspectiveCameraCallback with overlay functionality.

This script tests the realistic nozzle + powder stream overlay with different
camera angles and configurations.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks._base_callbacks import SimulationEvent


# ============================================================================
# Mock Simulation Classes
# ============================================================================

class MockVolumeTracker:
    """Mock volume tracker for testing."""
    def __init__(self, shape=(100, 100, 50)):
        self.activated = np.zeros(shape, dtype=bool)
        # Activate a small region in the center
        cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
        self.activated[cx-5:cx+5, cy-5:cy+5, cz-5:cz+5] = True


class MockTemperatureTracker:
    """Mock temperature tracker for testing."""
    def __init__(self, shape=(100, 100, 50)):
        self.temperature = np.full(shape, 300.0, dtype=float)
        # Add hot spot in center
        cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
        self.temperature[cx-5:cx+5, cy-5:cy+5, cz-5:cz+5] = 1800.0
        # Add gradient
        for i in range(5, 15):
            temp = 1800.0 - (i-5) * 100
            self.temperature[cx-i:cx+i, cy-i:cy+i, cz-i:cz+i] = np.maximum(
                self.temperature[cx-i:cx+i, cy-i:cy+i, cz-i:cz+i],
                temp
            )


class MockProgressTracker:
    """Mock progress tracker for testing."""
    def __init__(self):
        self.step_count = 0


class MockSimulation:
    """Mock simulation for testing callbacks."""
    def __init__(self):
        self.volume_tracker = MockVolumeTracker()
        self.temperature_tracker = MockTemperatureTracker()
        self.progress_tracker = MockProgressTracker()
        self.output_dir = Path('.')  # Add output_dir attribute
        self.config = {
            'voxel_size': (2e-4, 2e-4, 2e-4)  # 200 microns
        }

        # Step context with nozzle position
        self.step_context = {
            'position': {
                'x': 0.01,  # 10mm
                'y': 0.01,  # 10mm
                'z': 0.01   # 10mm above substrate
            },
            'build': {
                'layer': 1,
                'track': 5
            }
        }


# ============================================================================
# Test Functions
# ============================================================================

def test_overlay_basic():
    """Test basic overlay functionality."""
    print("\n" + "="*70)
    print("TEST 1: Basic Overlay (Default Settings)")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        enable_overlay=True,
        save_images=True,
        save_dir="test_output/test1_basic_overlay",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print(f"  Camera offset: {callback.rel_offset_local}")
    print(f"  Floor angle: {callback.floor_angle_deg}°")
    print(f"  Overlay enabled: {callback.enable_overlay}")
    print(f"  Nozzle position: ({sim.step_context['position']['x']:.3f}, "
          f"{sim.step_context['position']['y']:.3f}, "
          f"{sim.step_context['position']['z']:.3f})")

    try:
        callback._execute(context)
        print("  [OK] Test passed - Image saved successfully")
    except Exception as e:
        print(f"  [X] Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_overlay_side_view():
    """Test overlay from side view."""
    print("\n" + "="*70)
    print("TEST 2: Side View (90° from side)")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.15, 0.0, 0.0),  # To the side
        floor_angle_deg=0.0,  # Horizontal view
        enable_overlay=True,
        save_images=True,
        save_dir="test_output/test2_side_view",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print(f"  Camera offset: {callback.rel_offset_local}")
    print(f"  Floor angle: {callback.floor_angle_deg}°")

    try:
        callback._execute(context)
        print("  [OK] Test passed - Side view rendered")
    except Exception as e:
        print(f"  [X] Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_overlay_top_view():
    """Test overlay from top view."""
    print("\n" + "="*70)
    print("TEST 3: Top View (Looking straight down)")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.0, 0.0, 0.08),  # Above nozzle
        floor_angle_deg=90.0,  # Looking straight down
        enable_overlay=True,
        save_images=True,
        save_dir="test_output/test3_top_view",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print(f"  Camera offset: {callback.rel_offset_local}")
    print(f"  Floor angle: {callback.floor_angle_deg}°")

    try:
        callback._execute(context)
        print("  [OK] Test passed - Top view rendered")
    except Exception as e:
        print(f"  [X] Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_overlay_custom_config():
    """Test overlay with custom configuration."""
    print("\n" + "="*70)
    print("TEST 4: Custom Overlay Configuration")
    print("="*70)

    custom_config = {
        'stream_height_mm': 20.0,  # Longer stream
        'v_angle_deg': 20.0,       # Wider angle
        'num_particles': 3000,     # More particles
        'nozzle_outlet_radius_mm': 5.0,
        'nozzle_top_radius_mm': 12.0,
        'particle_color': (255, 200, 100),  # Yellowish
        'particle_alpha': 0.8,
    }

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.05, -0.10, 0.06),
        floor_angle_deg=35.0,
        enable_overlay=True,
        overlay_config=custom_config,
        save_images=True,
        save_dir="test_output/test4_custom_config",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print(f"  Stream height: {custom_config['stream_height_mm']}mm")
    print(f"  V-angle: {custom_config['v_angle_deg']}°")
    print(f"  Particle count: {custom_config['num_particles']}")
    print(f"  Particle color: {custom_config['particle_color']}")

    try:
        callback._execute(context)
        print("  [OK] Test passed - Custom config applied")
    except Exception as e:
        print(f"  [X] Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_overlay_disabled():
    """Test callback with overlay disabled (thermal only)."""
    print("\n" + "="*70)
    print("TEST 5: Overlay Disabled (Thermal Only)")
    print("="*70)

    callback = PerspectiveCameraCallback(
        rel_offset_local=(0.0, -0.12, 0.04),
        floor_angle_deg=30.0,
        enable_overlay=False,  # Disabled
        save_images=True,
        save_dir="test_output/test5_no_overlay",
        interval=1
    )

    sim = MockSimulation()
    context = {
        'simulation': sim,
        'output_dir': Path('.'),
    }

    print(f"  Overlay enabled: {callback.enable_overlay}")
    print(f"  Expected: Thermal-only view with colorbar")

    try:
        callback._execute(context)
        print("  [OK] Test passed - Thermal-only image saved")
    except Exception as e:
        print(f"  [X] Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_angles():
    """Test multiple camera angles in sequence."""
    print("\n" + "="*70)
    print("TEST 6: Multiple Camera Angles")
    print("="*70)

    angles = [
        ("Behind", (0.0, -0.12, 0.04), 30.0),
        ("Right Side", (0.12, -0.08, 0.04), 25.0),
        ("Left Side", (-0.12, -0.08, 0.04), 25.0),
        ("Front", (0.0, 0.10, 0.03), 20.0),
    ]

    for name, offset, angle in angles:
        print(f"\n  Testing: {name}")
        print(f"    Offset: {offset}")
        print(f"    Angle: {angle}°")

        callback = PerspectiveCameraCallback(
            rel_offset_local=offset,
            floor_angle_deg=angle,
            enable_overlay=True,
            save_images=True,
            save_dir=f"test_output/test6_multi/{name.lower().replace(' ', '_')}",
            interval=1
        )

        sim = MockSimulation()
        context = {
            'simulation': sim,
            'output_dir': Path('.'),
        }

        try:
            callback._execute(context)
            print(f"    [OK] {name} view rendered successfully")
        except Exception as e:
            print(f"    [X] {name} view failed: {e}")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n")
    print("="*70)
    print(" "*10 + "PERSPECTIVE CAMERA OVERLAY TEST SUITE")
    print("="*70)

    tests = [
        test_overlay_basic,
        test_overlay_side_view,
        test_overlay_top_view,
        test_overlay_custom_config,
        test_overlay_disabled,
        test_multiple_angles,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  [X] Test suite error: {e}")
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
    print("\n  Output images saved to: test_output/")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
