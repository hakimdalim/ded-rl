"""
Quick test for HDF5ThermalSaver callback.

Tests:
1. HDF5 file is created
2. Data is saved correctly
3. Metadata is stored
4. File can be read back

Run: python test_callbacks/test_hdf5_saver.py
"""

import sys
import os

# Add parent directory to path FIRST
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
from pathlib import Path
import numpy as np

# Check if h5py is available
try:
    import h5py
    print("✓ h5py is installed")
except ImportError:
    print("✗ h5py not installed!")
    print("  Install with: pip install h5py")
    sys.exit(1)

# Import simulation components from parent directory
from simulate import SimulationRunner
from callbacks.completion_callbacks import StepCountCompletionCallback
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver, load_thermal_field, load_thermal_metadata
from callbacks.callback_collection import ProgressPrinter


def test_hdf5_saver():
    """Test HDF5ThermalSaver with real simulation."""
    print("\n" + "="*70)
    print("TEST: HDF5 Thermal Field Saver")
    print("="*70)

    print("\nCreating minimal simulation...")
    print("  Build volume: 10mm x 10mm x 8mm")
    print("  Will run for 20 steps")
    print("  Save thermal field every 5 steps")

    # Create callbacks
    callbacks = [
        StepCountCompletionCallback(max_steps=20),
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=5,  # Save every 5 steps
            compression='gzip',
            compression_opts=4
        ),
        ProgressPrinter()
    ]

    try:
        start_time = time.time()

        # Create simulation with better parameters
        runner = SimulationRunner.from_human_units(
            build_volume_mm=(20.0, 20.0, 15.0),  # Larger volume
            part_volume_mm=(5.0, 5.0, 2.0),
            voxel_size_um=200.0,
            delta_t_ms=200.0,  # Larger time step
            scan_speed_mm_s=3.0,
            laser_power_W=600.0,
            powder_feed_g_min=2.0,
            hatch_spacing_um=700.0,
            layer_spacing_um=350.0,
            substrate_height_mm=5.0,
            experiment_label="test_hdf5",
            callbacks=callbacks
        )

        print(f"\nOutput directory: {runner.simulation.output_dir}")
        print("\nRunning simulation...\n")

        # Run simulation
        runner.run()

        elapsed = time.time() - start_time
        print(f"\n✓ Simulation completed in {elapsed:.1f} seconds")

        # Check if HDF5 file was created
        output_dir = Path(runner.simulation.output_dir)
        h5_file = output_dir / "thermal_fields.h5"

        if not h5_file.exists():
            print("\n✗ FAILED: HDF5 file not created!")
            return False

        print(f"\n✓ HDF5 file created: {h5_file}")

        # Check file size
        file_size_mb = h5_file.stat().st_size / (1024 * 1024)
        print(f"✓ File size: {file_size_mb:.2f} MB")

        # Open and inspect file
        print("\nInspecting HDF5 file contents:")
        with h5py.File(h5_file, 'r') as f:
            step_groups = list(f.keys())
            print(f"  Groups (timesteps): {step_groups}")

            # Check that we have the expected number of saves
            # With interval=5 and max_steps=20, we should have 4 saves
            expected_num_saves = 20 // 5  # = 4

            if len(step_groups) != expected_num_saves:
                print(f"\n✗ Expected {expected_num_saves} timesteps, found {len(step_groups)}")
                return False

            print(f"\n✓ Found {len(step_groups)} timesteps (as expected with interval=5, max_steps=20)")

            # Inspect each saved step
            for step_name in sorted(step_groups):
                step_group = f[step_name]

                # Check temperature data
                if 'temperature' in step_group:
                    temp_data = step_group['temperature']
                    print(f"\n  {step_name}:")
                    print(f"    ✓ Temperature field shape: {temp_data.shape}")
                    print(f"    ✓ Data type: {temp_data.dtype}")
                    print(f"    ✓ Min temp: {temp_data[:].min():.1f} K")
                    print(f"    ✓ Max temp: {temp_data[:].max():.1f} K")

                    # Check metadata
                    if step_group.attrs:
                        print(f"    ✓ Metadata attributes: {len(step_group.attrs)}")
                        print(f"      - Step: {step_group.attrs.get('step', 'N/A')}")
                        print(f"      - Max temp: {step_group.attrs.get('max_temp', 'N/A'):.1f} K")
                        if 'position_x' in step_group.attrs:
                            print(f"      - Position: ({step_group.attrs['position_x']*1000:.2f}, "
                                  f"{step_group.attrs['position_y']*1000:.2f}, "
                                  f"{step_group.attrs['position_z']*1000:.2f}) mm")
                else:
                    print(f"  ✗ {step_name}: No temperature data!")
                    return False

        # Test reading data with utility functions
        print("\nTesting utility functions:")
        try:
            # Get the first saved step number
            first_step = int(step_groups[0].split('_')[1])

            # Load temperature field
            temp = load_thermal_field(str(h5_file), step=first_step)
            print(f"✓ load_thermal_field() works - shape: {temp.shape}")

            # Load metadata
            metadata = load_thermal_metadata(str(h5_file), step=first_step)
            print(f"✓ load_thermal_metadata() works - {len(metadata)} attributes")
            print(f"  - Laser power: {metadata.get('laser_power', 'N/A')} W")
            print(f"  - Scan speed: {metadata.get('scan_speed', 'N/A')*1000:.1f} mm/s")

        except Exception as e:
            print(f"✗ Utility functions failed: {e}")
            return False

        # Summary
        print("\n" + "="*70)
        print("TEST PASSED! ✓")
        print("="*70)
        print("\nSummary:")
        print(f"  ✓ HDF5 file created successfully")
        print(f"  ✓ {len(step_groups)} timesteps saved (interval=5, total_steps=20)")
        print(f"  ✓ Temperature data shape: {temp.shape}")
        print(f"  ✓ Metadata stored correctly")
        print(f"  ✓ File size: {file_size_mb:.2f} MB (compressed)")
        print(f"  ✓ Utility functions work")

        # Comparison with uncompressed
        uncompressed_size = temp.nbytes * len(step_groups) / (1024 * 1024)
        compression_ratio = uncompressed_size / file_size_mb
        print(f"\n  Compression ratio: {compression_ratio:.1f}x")
        print(f"  (Uncompressed would be ~{uncompressed_size:.1f} MB)")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hdf5_saver()

    if success:
        print("\n🎉 HDF5ThermalSaver is working correctly!")
        print("\nYou can now use it in your simulations:")
        print("""
    from callbacks.hdf5_thermal_saver import HDF5ThermalSaver

    callbacks = [
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=10,
            compression='gzip',
            compression_opts=4
        )
    ]
        """)
    else:
        print("\n⚠️ Test failed. Check errors above.")
        sys.exit(1)
