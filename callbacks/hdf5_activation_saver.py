"""
HDF5-based activation volume saver for  storage of 3D activation states.

HDF5 provides:
- Compression 
- Chunked storage 
- Fast I/O for large arrays
- Metadata support
"""

import warnings
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn(
        "h5py not installed. "
    )

from callbacks._base_callbacks import IntervalCallback, SimulationEvent


class HDF5ActivationSaver(IntervalCallback):
    """


    File structure:
        activation_volumes.h5
        ├── /step_0001/
        │   ├── activation   [nx, ny, nz] bool array (True = activated voxel)
        │   └── metadata     (attributes: step, time, position, num_activated, etc.)
        ├── /step_0002/
        │   ├── activation
        │   └── metadata
        └── ...
    """

    def __init__(
        self,
        filename: str = "activation_volumes.h5",
        interval: int = 1,
        compression: str = 'gzip',
        compression_opts: int = 9,  # Use max compression for bool (fast anyway)
        save_metadata: bool = True,
        **kwargs
    ):
        """
        Initialize HDF5 activation volume saver.

        Args:
            filename: HDF5 filename (saved in simulation output_dir)
            interval: Save every N steps (default: 1 = every step)
            compression: Compression algorithm ('gzip', 'lzf', None)
                        'gzip': Best compression (highly recommended for bool)
                        'lzf': Faster, less compression
                        None: No compression (not recommended - much larger files)
            compression_opts: Compression level (0-9 for gzip, ignored for lzf)
                            Default 9 (max) is recommended for bool arrays
            save_metadata: If True, save simulation metadata with each step
            **kwargs: Additional arguments for IntervalCallback
        """
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )

        if not HDF5_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5ActivationSaver. "
            )

        self.filename = filename
        self.compression = compression
        self.compression_opts = compression_opts if compression == 'gzip' else None
        self.save_metadata = save_metadata

        # For single file mode
        self._h5file = None
        self._file_path = None

    def _execute(self, context: dict) -> None:
        """Save activation volume to HDF5."""
        sim = context['simulation']

        # Get activation volume
        activation_vol = sim.volume_tracker.activated

        if activation_vol is None:
            warnings.warn("No activation volume available")
            return

        # Get current step
        step = sim.progress_tracker.step_count

        # Save to file
        self._save_to_file(activation_vol, step, context)

    def _save_to_file(self, activation_vol: np.ndarray, step: int, context: dict):
        """Save to HDF5 file (all timesteps in one file)."""
        # Open or create file on first call
        if self._h5file is None:
            # File saved directly in simulation output_dir
            save_path = self.resolve_path(context, self.filename)
            self._file_path = save_path

            # Ensure directory exists
            self.ensure_dir(self._file_path.parent)

            # Open file in append mode
            self._h5file = h5py.File(self._file_path, 'a')
            print(f"Saving activation volumes to: {self._file_path}")

        # Create group for this step
        step_group = self._h5file.create_group(f'step_{step:04d}')

        # Save activation data (bool array compresses extremely well!)
        step_group.create_dataset(
            'activation',
            data=activation_vol,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
            dtype='bool'  # Explicitly use bool for best compression
        )

        # Add metadata if requested
        if self.save_metadata:
            self._save_metadata_to_group(step_group, step, context, activation_vol)

        # Flush to disk
        self._h5file.flush()

    def _save_metadata_to_group(self, group, step: int, context: dict, activation_vol: np.ndarray):
        """Save metadata as HDF5 attributes."""
        sim = context['simulation']

        # Basic metadata
        group.attrs['step'] = step
        group.attrs['time'] = step * sim.delta_t if hasattr(sim, 'delta_t') else 0.0

        # Activation statistics
        num_activated = np.sum(activation_vol)
        total_voxels = activation_vol.size
        group.attrs['num_activated'] = int(num_activated)
        group.attrs['total_voxels'] = int(total_voxels)
        group.attrs['activation_fraction'] = float(num_activated / total_voxels)

        # Simulation state
        if sim.step_context:
            ctx = sim.step_context

            # Position
            if 'position' in ctx:
                pos = ctx['position']
                group.attrs['position_x'] = pos['x']
                group.attrs['position_y'] = pos['y']
                group.attrs['position_z'] = pos['z']

            # Voxel indices
            if 'voxel' in ctx:
                vox = ctx['voxel']
                group.attrs['voxel_x'] = vox['x']
                group.attrs['voxel_y'] = vox['y']
                group.attrs['voxel_z'] = vox['z']

            # Build progress
            if 'build' in ctx:
                build = ctx['build']
                group.attrs['layer'] = build['layer']
                group.attrs['track'] = build['track']

            # Process parameters
            if 'params' in ctx:
                params = ctx['params']
                group.attrs['laser_power'] = params.get('laser_power', 0.0)
                group.attrs['scan_speed'] = params.get('scan_speed', 0.0)
                group.attrs['powder_feed_rate'] = params.get('powder_feed_rate', 0.0)

        # Simulation config
        if hasattr(sim, 'config'):
            config = sim.config
            group.attrs['voxel_size_x'] = config.get('voxel_size', [0, 0, 0])[0]
            group.attrs['voxel_size_y'] = config.get('voxel_size', [0, 0, 0])[1]
            group.attrs['voxel_size_z'] = config.get('voxel_size', [0, 0, 0])[2]

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if self._h5file is not None:
            try:
                self._h5file.close()
                print(f"Closed HDF5 file: {self._file_path}")
            except:
                pass


# ============================================================================
# Utility Functions for Reading HDF5 Files
# ============================================================================

def load_activation_volume(filepath: str, step: Optional[int] = None) -> np.ndarray:
    """
    Load activation volume from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        step: Step number to load

    Returns:
        Activation volume as boolean numpy array

    Examples:
        # Load activation volume
        activation = load_activation_volume("activation_volumes.h5", step=42)
        print(f"Activated voxels: {activation.sum()}")
    """
    with h5py.File(filepath, 'r') as f:
        step_group = f[f'step_{step:04d}']
        return step_group['activation'][:].astype(bool)


def load_activation_metadata(filepath: str, step: Optional[int] = None) -> dict:
    """
    Load metadata from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        step: Step number to load

    Returns:
        Dictionary of metadata attributes

  """
    with h5py.File(filepath, 'r') as f:
        group = f[f'step_{step:04d}']
        return dict(group.attrs)


def list_steps_in_file(filepath: str) -> list:

    with h5py.File(filepath, 'r') as f:
        step_groups = [key for key in f.keys() if key.startswith('step_')]
        steps = [int(key.split('_')[1]) for key in step_groups]
        return sorted(steps)


def get_file_info(filepath: str):
    """
    Print information about HDF5 file contents.

    """
    with h5py.File(filepath, 'r') as f:
        print(f"\nHDF5 File: {filepath}")
        print("="*60)

        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")

                # Calculate sizes
                uncompressed = obj.dtype.itemsize * np.prod(obj.shape)
                compressed = obj.nbytes if hasattr(obj, 'nbytes') else obj.id.get_storage_size()

                print(f"    Uncompressed: {uncompressed / (1024**2):.2f} MB")
                print(f"    Compressed: {compressed / (1024**2):.2f} MB")
                if compressed > 0:
                    ratio = uncompressed / compressed
                    print(f"    Compression ratio: {ratio:.1f}x")

                if obj.compression:
                    print(f"    Compression: {obj.compression}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
                if obj.attrs:
                    print(f"    Attributes: {list(obj.attrs.keys())[:5]}...")

        f.visititems(print_item)
        print("="*60)


def get_activation_statistics(filepath: str) -> dict:
    """
    Get statistics across all timesteps in file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary with statistics

     """
    steps = list_steps_in_file(filepath)

    stats = {
        'num_steps': len(steps),
        'steps': steps,
        'activation_per_step': [],
        'num_activated_per_step': []
    }

    with h5py.File(filepath, 'r') as f:
        for step in steps:
            group = f[f'step_{step:04d}']
            if 'num_activated' in group.attrs:
                stats['num_activated_per_step'].append(group.attrs['num_activated'])
            if 'activation_fraction' in group.attrs:
                stats['activation_per_step'].append(group.attrs['activation_fraction'])

    if stats['activation_per_step']:
        stats['final_activation_fraction'] = stats['activation_per_step'][-1]
        stats['max_activation_fraction'] = max(stats['activation_per_step'])

    if stats['num_activated_per_step']:
        stats['final_num_activated'] = stats['num_activated_per_step'][-1]
        stats['total_activated_growth'] = stats['num_activated_per_step'][-1] - stats['num_activated_per_step'][0]

    return stats


if __name__ == "__main__":
    print("""
HDF5ActivationSaver - 

Usage in simulation:
    from callbacks.hdf5_activation_saver import HDF5ActivationSaver

    callback = HDF5ActivationSaver(
        filename="activation_volumes.h5",
        interval=10,  # Save every 10 steps
        compression='gzip',
        compression_opts=9  # Max compression for bool arrays
    )

Loading data:
    from callbacks.hdf5_activation_saver import (
        load_activation_volume,
        load_activation_metadata,
        get_activation_statistics
    )

    # Load activation volume
    activation = load_activation_volume("activation_volumes.h5", step=42)
    print(f"Activated voxels: {activation.sum()}")

    # Load metadata
    metadata = load_activation_metadata("activation_volumes.h5", step=42)
    print(f"Activation fraction: {metadata['activation_fraction']:.2%}")

    # Get overall statistics
    stats = get_activation_statistics("activation_volumes.h5")
    print(f"Build progress: {stats['final_activation_fraction']:.2%}")


""")
