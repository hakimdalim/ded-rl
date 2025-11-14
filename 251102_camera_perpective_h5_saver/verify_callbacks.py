"""
Comprehensive verification script for all new callbacks.
This script tests each callback individually and reports their status.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Test imports
print("="*70)
print("CALLBACK VERIFICATION SCRIPT")
print("="*70)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Track results
results = {}

# ============================================================================
# 1. Test Callback Imports
# ============================================================================
print("\n[1/5] Testing callback imports...")

try:
    from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
    results['HDF5ThermalSaver import'] = '[OK] PASS'
    print("  [OK] PASS HDF5ThermalSaver imported successfully")
except Exception as e:
    results['HDF5ThermalSaver import'] = f'[X] FAIL {e}'
    print(f"  [X] FAIL HDF5ThermalSaver import failed: {e}")

try:
    from callbacks.hdf5_activation_saver import HDF5ActivationSaver
    results['HDF5ActivationSaver import'] = '[OK] PASS'
    print("  [OK] PASS HDF5ActivationSaver imported successfully")
except Exception as e:
    results['HDF5ActivationSaver import'] = f'[X] FAIL {e}'
    print(f"  [X] FAIL HDF5ActivationSaver import failed: {e}")

try:
    from callbacks.perspective_camera_callback import PerspectiveCameraCallback
    results['PerspectiveCameraCallback import'] = '[OK] PASS'
    print("  [OK] PASS PerspectiveCameraCallback imported successfully")
except Exception as e:
    results['PerspectiveCameraCallback import'] = f'[X] FAIL {e}'
    print(f"  [X] FAIL PerspectiveCameraCallback import failed: {e}")

try:
    from callbacks.live_plotter_callback import AdvancedLivePlotter
    results['AdvancedLivePlotter import'] = '[OK] PASS'
    print("  [OK] PASS AdvancedLivePlotter imported successfully")
except Exception as e:
    results['AdvancedLivePlotter import'] = f'[X] FAIL {e}'
    print(f"  [X] FAIL AdvancedLivePlotter import failed: {e}")

# ============================================================================
# 2. Test Callback Instantiation
# ============================================================================
print("\n[2/5] Testing callback instantiation...")

try:
    thermal_saver = HDF5ThermalSaver(save_interval=5)
    results['HDF5ThermalSaver instantiation'] = '[OK] PASS'
    print("  [OK] HDF5ThermalSaver instantiated successfully")
except Exception as e:
    results['HDF5ThermalSaver instantiation'] = f'[X] FAIL: {e}'
    print(f"  [X] HDF5ThermalSaver instantiation failed: {e}")

try:
    activation_saver = HDF5ActivationSaver(save_interval=5)
    results['HDF5ActivationSaver instantiation'] = '[OK] PASS'
    print("  [OK] HDF5ActivationSaver instantiated successfully")
except Exception as e:
    results['HDF5ActivationSaver instantiation'] = f'[X] FAIL: {e}'
    print(f"  [X] HDF5ActivationSaver instantiation failed: {e}")

try:
    camera_callback = PerspectiveCameraCallback(
        enable_overlay=False,
        resolution_wh=(640, 480),
        interval=5
    )
    results['PerspectiveCameraCallback instantiation'] = '[OK] PASS'
    print("  [OK] PerspectiveCameraCallback instantiated successfully")
except Exception as e:
    results['PerspectiveCameraCallback instantiation'] = f'[X] FAIL: {e}'
    print(f"  [X] PerspectiveCameraCallback instantiation failed: {e}")

try:
    # Don't actually create live plotter in headless mode
    # Just verify it can be imported and has correct signature
    import inspect
    sig = inspect.signature(AdvancedLivePlotter.__init__)
    results['AdvancedLivePlotter instantiation'] = '[OK] PASS (signature check)'
    print("  [OK] AdvancedLivePlotter has valid signature")
except Exception as e:
    results['AdvancedLivePlotter instantiation'] = f'[X] FAIL: {e}'
    print(f"  [X] AdvancedLivePlotter check failed: {e}")

# ============================================================================
# 3. Check Required Dependencies
# ============================================================================
print("\n[3/5] Checking required dependencies...")

try:
    import h5py
    results['h5py dependency'] = f'[OK] PASS (version {h5py.__version__})'
    print(f"  [OK] h5py available (version {h5py.__version__})")
except ImportError:
    results['h5py dependency'] = '[X] FAIL: not installed'
    print("  [X] h5py not available - HDF5 callbacks will not work!")

try:
    import matplotlib
    results['matplotlib dependency'] = f'[OK] PASS (version {matplotlib.__version__})'
    print(f"  [OK] matplotlib available (version {matplotlib.__version__})")
except ImportError:
    results['matplotlib dependency'] = '[X] FAIL: not installed'
    print("  [X] matplotlib not available - plotting callbacks will not work!")

try:
    from camera.perspective_camera import FollowingPerspectiveCamera
    results['camera module'] = '[OK] PASS'
    print("  [OK] Camera module available")
except ImportError as e:
    results['camera module'] = f'[X] FAIL: {e}'
    print(f"  [X] Camera module not available: {e}")

# ============================================================================
# 4. Run Mini Simulation Test
# ============================================================================
print("\n[4/5] Running minimal simulation test...")

try:
    from simulate import SimulationRunner
    from callbacks.completion_callbacks import HeightCompletionCallback
    from callbacks.callback_collection import ProgressPrinter

    # Create callbacks list
    test_callbacks = [
        HeightCompletionCallback(),
        ProgressPrinter(),
        HDF5ThermalSaver(save_interval=2, compression='gzip', compression_opts=4),
        HDF5ActivationSaver(save_interval=2, compression='gzip', compression_opts=4),
        PerspectiveCameraCallback(
            enable_overlay=False,
            resolution_wh=(320, 240),  # Small resolution for speed
            interval=2,
            save_dir="cam"
        ),
    ]

    # Create tiny simulation (runs very fast)
    runner = SimulationRunner.from_human_units(
        build_volume_mm=(10.0, 10.0, 10.0),
        part_volume_mm=(3.0, 3.0, 1.0),  # Just 1mm tall
        voxel_size_um=200.0,
        delta_t_ms=200.0,
        scan_speed_mm_s=5.0,  # Fast scan
        laser_power_W=600.0,
        powder_feed_g_min=2.0,
        output_base_dir="_experiments",
        experiment_label="callback_verification_test",
        callbacks=test_callbacks
    )

    print("\n  Running mini simulation (this may take a minute)...")
    runner.run()

    output_dir = Path(runner.simulation.output_dir)
    results['Mini simulation'] = '[OK] PASS'
    print(f"  [OK] Simulation completed successfully")
    print(f"    Output: {output_dir}")

    # ========================================================================
    # 5. Verify Output Files
    # ========================================================================
    print("\n[5/5] Verifying output files...")

    # Check HDF5 thermal file
    thermal_h5 = output_dir / "thermal_fields.h5"
    if thermal_h5.exists():
        import h5py
        with h5py.File(thermal_h5, 'r') as f:
            num_steps = len([k for k in f.keys() if k.startswith('step_')])
            results['HDF5ThermalSaver output'] = f'[OK] PASS ({num_steps} timesteps saved)'
            print(f"  [OK] thermal_fields.h5 created with {num_steps} timesteps")
    else:
        results['HDF5ThermalSaver output'] = '[X] FAIL: file not created'
        print(f"  [X] thermal_fields.h5 not found")

    # Check HDF5 activation file
    activation_h5 = output_dir / "activation_volumes.h5"
    if activation_h5.exists():
        import h5py
        with h5py.File(activation_h5, 'r') as f:
            num_steps = len([k for k in f.keys() if k.startswith('step_')])
            results['HDF5ActivationSaver output'] = f'[OK] PASS ({num_steps} timesteps saved)'
            print(f"  [OK] activation_volumes.h5 created with {num_steps} timesteps")
    else:
        results['HDF5ActivationSaver output'] = '[X] FAIL: file not created'
        print(f"  [X] activation_volumes.h5 not found")

    # Check camera images
    cam_dir = output_dir / "cam"
    if cam_dir.exists():
        images = list(cam_dir.glob("thermal_step_*.png"))
        if images:
            results['PerspectiveCameraCallback output'] = f'[OK] PASS ({len(images)} images saved)'
            print(f"  [OK] Camera images created: {len(images)} images in {cam_dir}")
        else:
            results['PerspectiveCameraCallback output'] = '[X] FAIL: no images found'
            print(f"  [X] No camera images found in {cam_dir}")
    else:
        results['PerspectiveCameraCallback output'] = '[X] FAIL: directory not created'
        print(f"  [X] Camera directory not found")

except Exception as e:
    results['Mini simulation'] = f'[X] FAIL: {e}'
    print(f"  [X] Simulation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

passed = sum(1 for v in results.values() if v.startswith('[OK]'))
failed = sum(1 for v in results.values() if v.startswith('[X]'))

for test, result in results.items():
    print(f"{result:40s} | {test}")

print("\n" + "-"*70)
print(f"Total: {passed} passed, {failed} failed out of {len(results)} tests")
print("="*70)

if failed == 0:
    print("\nSUCCESS All callbacks are functioning properly!")
    sys.exit(0)
else:
    print(f"\nWARNING  {failed} callback(s) need attention")
    sys.exit(1)
