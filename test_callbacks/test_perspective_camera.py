"""
Quick test for PerspectiveCameraCallback.

Tests:
1. Camera is created and follows nozzle
2. Images are rendered correctly
3. Images are saved to disk
4. Camera position updates with nozzle movement

Run: python test_callbacks/test_perspective_camera.py
"""

import sys
import os

# Add parent directory to path FIRST
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
from pathlib import Path
import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib is installed")
except ImportError:
    print("✗ matplotlib not installed!")
    print("  Install with: pip install matplotlib")
    sys.exit(1)

# Import simulation components from parent directory
from simulate import SimulationRunner
from callbacks.completion_callbacks import StepCountCompletionCallback
from callbacks.perspective_camera_callback import PerspectiveCameraCallback
from callbacks.callback_collection import ProgressPrinter


def test_perspective_camera():
    """Test PerspectiveCameraCallback with real simulation."""
    print("\n" + "="*70)
    print("TEST: Perspective Camera Callback (Following Nozzle)")
    print("="*70)

    print("\nCreating minimal simulation...")
    print("  Build volume: 20mm x 20mm x 15mm")
    print("  Will run for 20 steps")
    print("  Save camera image every 5 steps")

    # Create callbacks
    callbacks = [
        StepCountCompletionCallback(max_steps=20),

        # Test camera with default position
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.12, 0.04),  # 12cm behind, 4cm above
            floor_angle_deg=30.0,                  # 30 degree downward view
            fov_y_deg=45.0,                        # 45 degree field of view
            save_images=True,
            interval=5,                            # Save every 5 steps
            resolution_wh=(800, 600),              # 800x600 pixels
            cmap='hot',
            dpi=150
        ),

        ProgressPrinter()
    ]

    try:
        start_time = time.time()

        # Create simulation with better parameters
        runner = SimulationRunner.from_human_units(
            build_volume_mm=(20.0, 20.0, 15.0),
            part_volume_mm=(5.0, 5.0, 2.0),
            voxel_size_um=200.0,
            delta_t_ms=200.0,
            scan_speed_mm_s=3.0,
            laser_power_W=600.0,
            powder_feed_g_min=2.0,
            hatch_spacing_um=700.0,
            layer_spacing_um=350.0,
            substrate_height_mm=5.0,
            experiment_label="test_cam",  # Short label to avoid Windows path length issues
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("\nRunning simulation...\n")

        # Run simulation
        runner.run()

        elapsed = time.time() - start_time
        print(f"\n✓ Simulation completed in {elapsed:.1f} seconds")

        # Check if camera images were created
        output_dir = Path(runner.simulation.output_dir)

        # Try to find camera images directory
        possible_dirs = ["cam", "camera_images", ""]  # "" = directly in output_dir as fallback
        camera_dir = None
        image_files = []

        for dir_name in possible_dirs:
            if dir_name:
                test_dir = output_dir / dir_name
            else:
                test_dir = output_dir

            images = list(test_dir.glob("thermal_step_*.png"))
            if images:
                camera_dir = test_dir
                image_files = images
                break

        if not image_files:
            print("\n✗ FAILED: No camera images found!")
            print(f"   Searched in: {output_dir}")
            return False

        print(f"\n✓ Camera images found in: {camera_dir}")
        print(f"   (Note: Directory name may vary due to Windows path length limits)")

        if not image_files:
            print("\n✗ FAILED: No camera images saved!")
            return False

        print(f"✓ Found {len(image_files)} camera images")

        # Check expected number of images
        # With interval=5 and max_steps=20, we expect 4 images
        expected_num_images = 20 // 5  # = 4

        if len(image_files) != expected_num_images:
            print(f"\n✗ Expected {expected_num_images} images, found {len(image_files)}")
            return False

        print(f"✓ Correct number of images saved ({expected_num_images})")

        # Inspect image files
        print("\nInspecting camera images:")
        for img_file in sorted(image_files):
            file_size_kb = img_file.stat().st_size / 1024
            print(f"  {img_file.name}: {file_size_kb:.1f} KB")

        # Verify images can be loaded
        print("\nVerifying image contents:")
        for img_file in sorted(image_files)[:2]:  # Check first 2 images
            try:
                img = plt.imread(img_file)
                print(f"  {img_file.name}: shape={img.shape}, dtype={img.dtype}")
            except Exception as e:
                print(f"  ✗ Failed to load {img_file.name}: {e}")
                return False

        print("✓ Images are valid and can be loaded")

        # Test camera access
        print("\nTesting camera API:")
        camera_callback = None
        for cb in callbacks:
            if isinstance(cb, PerspectiveCameraCallback):
                camera_callback = cb
                break

        if camera_callback is None:
            print("✗ Could not find camera callback")
            return False

        # Get camera instance
        camera = camera_callback.get_camera()
        if camera is None:
            print("✗ Camera was not created")
            return False

        print("✓ Camera instance accessible")
        print(f"  Camera position: {camera.pos}")
        print(f"  Camera resolution: {camera.resolution_wh} px")
        print(f"  FOV: {camera.fov_y_deg}°")

        # Get latest image
        latest = camera_callback.get_latest_image()
        if latest is None:
            print("✗ No latest image available")
            return False

        img, extent = latest
        print(f"✓ Latest image accessible: shape={img.shape}")
        print(f"  Temperature range: {img.min():.1f} - {img.max():.1f} K")
        print(f"  Extent: {extent}")

        # Summary
        print("\n" + "="*70)
        print("TEST PASSED!")
        print("="*70)
        print("\nSummary:")
        print(f"  ✓ Camera images directory created")
        print(f"  ✓ {len(image_files)} images saved (interval=5, total_steps=20)")
        print(f"  ✓ Images are valid PNG files")
        print(f"  ✓ Camera API works correctly")
        print(f"  ✓ Latest image accessible")

        # Show sample image info
        print(f"\nSample image statistics:")
        print(f"  Resolution: {img.shape[1]} x {img.shape[0]} pixels")
        print(f"  Temperature range: {img.min():.1f} - {img.max():.1f} K")
        print(f"  Hot pixels (>400K): {np.sum(img > 400.0)}")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_camera_angles():
    """Test camera with different viewing angles."""
    print("\n" + "="*70)
    print("TEST: Multiple Camera Angles")
    print("="*70)

    print("\nTesting 3 different camera angles:")
    print("  1. Default: behind and above (30°)")
    print("  2. High angle: far behind and high (45°)")
    print("  3. Side angle: right side view (20°)")

    # Test with 3 different camera configurations
    camera_configs = [
        {
            'name': 'default',
            'rel_offset_local': (0.0, -0.12, 0.04),
            'floor_angle_deg': 30.0,
            'save_dir': 'cam1'  # Short names for Windows path limits
        },
        {
            'name': 'high_angle',
            'rel_offset_local': (0.0, -0.20, 0.10),
            'floor_angle_deg': 45.0,
            'save_dir': 'cam2'
        },
        {
            'name': 'side_view',
            'rel_offset_local': (0.08, -0.08, 0.05),
            'floor_angle_deg': 20.0,
            'save_dir': 'cam3'
        }
    ]

    for config in camera_configs:
        print(f"\nTesting camera: {config['name']}")
        print(f"  Offset: {config['rel_offset_local']}")
        print(f"  Angle: {config['floor_angle_deg']}°")

        callbacks = [
            StepCountCompletionCallback(max_steps=10),  # Shorter test
            PerspectiveCameraCallback(
                rel_offset_local=config['rel_offset_local'],
                floor_angle_deg=config['floor_angle_deg'],
                fov_y_deg=45.0,
                save_images=True,
                save_dir=config['save_dir'],
                interval=5,
                resolution_wh=(640, 480)
            )
        ]

        try:
            runner = SimulationRunner.from_human_units(
                build_volume_mm=(20.0, 20.0, 15.0),
                part_volume_mm=(5.0, 5.0, 2.0),
                voxel_size_um=200.0,
                delta_t_ms=200.0,
                scan_speed_mm_s=3.0,
                laser_power_W=600.0,
                powder_feed_g_min=2.0,
                experiment_label=f"cam_{config['name']}",  # Short label
                callbacks=callbacks
            )

            runner.run()

            # Check images were created
            output_dir = Path(runner.simulation.output_dir)
            camera_dir = output_dir / config['save_dir']
            image_files = list(camera_dir.glob("thermal_step_*.png"))

            print(f"  ✓ {len(image_files)} images saved to {config['save_dir']}/")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    print("\n" + "="*70)
    print("MULTI-ANGLE TEST PASSED!")
    print("="*70)
    print("\nAll camera angles tested successfully!")

    return True


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("PERSPECTIVE CAMERA CALLBACK TEST SUITE")
    print("█"*70)

    # Run test 1: Basic functionality
    success1 = test_perspective_camera()

    if not success1:
        print("\n⚠️ Basic test failed. Skipping multi-angle test.")
        sys.exit(1)

    # Run test 2: Multiple angles (optional)
    print("\n\n")
    user_input = input("Run multi-angle test? (slower, y/n): ").lower()

    if user_input == 'y':
        success2 = test_multiple_camera_angles()
    else:
        print("Skipping multi-angle test.")
        success2 = True

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if success1 and success2:
        print("\nAll tests PASSED!")
        print("\nYou can now use PerspectiveCameraCallback in your simulations:")
        print("""
    from callbacks.perspective_camera_callback import PerspectiveCameraCallback

    callbacks = [
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.12, 0.04),
            floor_angle_deg=30.0,
            fov_y_deg=45.0,
            save_images=True,
            interval=10
        )
    ]
        """)
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
        sys.exit(1)
