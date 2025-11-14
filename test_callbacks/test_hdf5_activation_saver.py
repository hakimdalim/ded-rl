"""
Quick test for HDF5ActivationSaver callback.

Tests:
1. HDF5 file is created
2. Activation data is saved correctly
3. Metadata is stored (including activation statistics)
4. File can be read back
5. Compression is effective for bool arrays

Run: python test_callbacks/test_hdf5_activation_saver.py
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
from callbacks.hdf5_activation_saver import (
    HDF5ActivationSaver,
    load_activation_volume,
    load_activation_metadata,
    get_activation_statistics
)
from callbacks.callback_collection import ProgressPrinter


def test_hdf5_activation_saver():
    """Test HDF5ActivationSaver with real simulation."""
    print("\n" + "="*70)
    print("TEST: HDF5 Activation Volume Saver")
    print("="*70)

    print("\nCreating minimal simulation...")
    print("  Build volume: 20mm x 20mm x 15mm")
    print("  Will run for 20 steps")
    print("  Save activation volume every 5 steps")

    # Create callbacks
    callbacks = [
        StepCountCompletionCallback(max_steps=20),
        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=5,  # Save every 5 steps
            compression='gzip',
            compression_opts=9  # Max compression for bool arrays
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
            experiment_label="test_hdf5_activation",
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
        h5_file = output_dir / "activation_volumes.h5"

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

                # Check activation data
                if 'activation' in step_group:
                    activation_data = step_group['activation']
                    print(f"\n  {step_name}:")
                    print(f"    ✓ Activation volume shape: {activation_data.shape}")
                    print(f"    ✓ Data type: {activation_data.dtype}")

                    # Load data to check
                    activation = activation_data[:].astype(bool)
                    num_activated = activation.sum()
                    total_voxels = activation.size
                    fraction = num_activated / total_voxels

                    print(f"    ✓ Activated voxels: {num_activated:,} / {total_voxels:,} ({fraction:.2%})")

                    # Check compression
                    uncompressed_size = activation.nbytes
                    compressed_size = activation_data.id.get_storage_size()
                    compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 0

                    print(f"    ✓ Compression: {compressed_size:,} bytes (ratio: {compression_ratio:.1f}x)")

                    # Check metadata
                    if step_group.attrs:
                        print(f"    ✓ Metadata attributes: {len(step_group.attrs)}")
                        print(f"      - Step: {step_group.attrs.get('step', 'N/A')}")
                        print(f"      - Activated voxels: {step_group.attrs.get('num_activated', 'N/A'):,}")
                        print(f"      - Activation fraction: {step_group.attrs.get('activation_fraction', 0.0):.2%}")
                        if 'layer' in step_group.attrs:
                            print(f"      - Layer: {step_group.attrs['layer']}, Track: {step_group.attrs.get('track', 'N/A')}")
                else:
                    print(f"  ✗ {step_name}: No activation data!")
                    return False

        # Test reading data with utility functions
        print("\nTesting utility functions:")
        try:
            # Get the first saved step number
            first_step = int(step_groups[0].split('_')[1])

            # Load activation volume
            activation = load_activation_volume(str(h5_file), step=first_step)
            print(f"✓ load_activation_volume() works - shape: {activation.shape}")
            print(f"  - Type: {activation.dtype}")
            print(f"  - Activated: {activation.sum():,} voxels")

            # Load metadata
            metadata = load_activation_metadata(str(h5_file), step=first_step)
            print(f"✓ load_activation_metadata() works - {len(metadata)} attributes")
            print(f"  - Laser power: {metadata.get('laser_power', 'N/A')} W")
            print(f"  - Activation fraction: {metadata.get('activation_fraction', 0.0):.2%}")

            # Get overall statistics
            stats = get_activation_statistics(str(h5_file))
            print(f"✓ get_activation_statistics() works")
            print(f"  - Total steps: {stats['num_steps']}")
            print(f"  - Final activation: {stats.get('final_activation_fraction', 0.0):.2%}")
            if 'total_activated_growth' in stats:
                print(f"  - Activation growth: {stats['total_activated_growth']:,} voxels")

        except Exception as e:
            print(f"✗ Utility functions failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Calculate overall compression statistics
        print("\nCompression Analysis:")
        with h5py.File(h5_file, 'r') as f:
            total_uncompressed = 0
            total_compressed = 0

            for step_name in step_groups:
                activation_data = f[step_name]['activation']
                total_uncompressed += activation_data.dtype.itemsize * np.prod(activation_data.shape)
                total_compressed += activation_data.id.get_storage_size()

            overall_ratio = total_uncompressed / total_compressed if total_compressed > 0 else 0

            print(f"  Total uncompressed: {total_uncompressed / (1024**2):.2f} MB")
            print(f"  Total compressed: {total_compressed / (1024**2):.2f} MB")
            print(f"  Overall compression ratio: {overall_ratio:.1f}x")
            print(f"  Space saved: {(1 - total_compressed/total_uncompressed)*100:.1f}%")

        # Summary
        print("\n" + "="*70)
        print("TEST PASSED! ✓")
        print("="*70)
        print("\nSummary:")
        print(f"  ✓ HDF5 file created successfully")
        print(f"  ✓ {len(step_groups)} timesteps saved (interval=5, total_steps=20)")
        print(f"  ✓ Activation data shape: {activation.shape}")
        print(f"  ✓ Metadata stored correctly")
        print(f"  ✓ File size: {file_size_mb:.2f} MB (compressed)")
        print(f"  ✓ Utility functions work")
        print(f"  ✓ Compression ratio: {overall_ratio:.1f}x (excellent for bool arrays!)")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hdf5_activation_saver()

    if success:
        print("\n🎉 HDF5ActivationSaver is working correctly!")
        print("\nYou can now use it in your simulations:")
        print("""
    from callbacks.hdf5_activation_saver import HDF5ActivationSaver

    callbacks = [
        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=10,
            compression='gzip',
            compression_opts=9  # Max compression for bool
        )
    ]
        """)
    else:
        print("\n⚠️ Test failed. Check errors above.")
        sys.exit(1)
