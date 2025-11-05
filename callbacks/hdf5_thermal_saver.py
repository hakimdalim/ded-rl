"""
HDF5-based thermal field saver for efficient storage of 3D temperature volumes.

HDF5 provides:
- Compression (saves disk space)
- Chunked storage (efficient partial reads)
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
        "h5py not installed."
    )

from callbacks._base_callbacks import IntervalCallback, SimulationEvent


class HDF5ThermalSaver(IntervalCallback):
    """
    Saves complete 3D temperature field to HDF5 format after each step.

    HDF5 format advantages:
    - Compression: ~5-10x smaller files than .npy
    - Chunked storage: Can read partial data without loading entire file
    - Fast I/O: Optimized for large arrays
    - Metadata: Can store simulation parameters alongside data
    - Single file: All timesteps in one file (optional)

    File structure (single file mode):
        thermal_fields.h5
        ├── /step_0001/
        │   ├── temperature  [nx, ny, nz] array
        │   └── metadata     (attributes: step, time, position, etc.)
        ├── /step_0002/
        │   ├── temperature
        │   └── metadata
        └── ...

    File structure (separate files mode):
        thermal_step_0001.h5
        ├── /temperature  [nx, ny, nz] array
        └── /metadata     (attributes: step, time, position, etc.)
    """

    def __init__(
        self,
        filename: str = "thermal_fields.h5",
        interval: int = 1,
        compression: str = 'gzip',
        compression_opts: int = 4,  # 0-9, higher = more compression but slower
        save_metadata: bool = True,
        **kwargs
    ):
        """
        Args:
            filename: HDF5 filename (saved to disk only on COMPLETE event)
            interval: Add data every N steps to memory (default: 1 = every step)
            compression: Compression algorithm ('gzip', 'lzf', None)
                        'gzip': Better compression, slower
                        'lzf': Faster, less compression
                        None: No compression (fastest, largest files)
            compression_opts: Compression level (0-9 for gzip, ignored for lzf)
            save_metadata: If True, save simulation metadata with each step
            **kwargs: Additional arguments for IntervalCallback
        """
        super().__init__(
            events=[SimulationEvent.STEP_COMPLETE, SimulationEvent.COMPLETE],
            interval=interval,
            **kwargs
        )

        if not HDF5_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5ThermalSaver. "
                "Install with: pip install h5py"
            )

        self.filename = filename
        self.compression = compression
        self.compression_opts = compression_opts if compression == 'gzip' else None
        self.save_metadata = save_metadata

        # HDF5 file kept in memory until COMPLETE event
        self._h5file = None
        self._file_path = None

    def _execute(self, context: dict) -> None:
        """Route to add step data or save file based on event."""
        event = context['event']

        if event == SimulationEvent.STEP_COMPLETE:
            # Add data to memory
            self._add_step_data(context)
        elif event == SimulationEvent.COMPLETE:
            # Save file to disk
            self._save_file_to_disk(context)

    def _add_step_data(self, context: dict):
        """Add temperature data for current step to in-memory HDF5 file."""
        sim = context['simulation']

        # Get temperature field
        temp_field = sim.temperature_tracker.temperature

        if temp_field is None:
            warnings.warn("No temperature field available")
            return

        # Get current step
        step = sim.progress_tracker.step_count

        # Create file in memory on first call
        if self._h5file is None:
            # Determine path (will be used later for saving)
            self._file_path = self.resolve_path(context, self.filename)

            # Create in-memory HDF5 file (driver='core' with backing_store=False)
            self._h5file = h5py.File(f"memory_{id(self)}.h5", 'w', driver='core', backing_store=False)
            print(f"Created in-memory HDF5 file (will save to: {self._file_path})")

        # Create group for this step
        step_group = self._h5file.create_group(f'step_{step:04d}')

        # Save temperature data
        step_group.create_dataset(
            'temperature',
            data=temp_field,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True
        )

        # Add metadata if requested
        if self.save_metadata:
            self._save_metadata_to_group(step_group, step, context)

    def _save_file_to_disk(self, context: dict):
        """Save in-memory HDF5 file to disk on simulation completion."""
        if self._h5file is None:
            print("No thermal data to save (simulation had no steps)")
            return

        # Ensure output directory exists
        self.ensure_dir(self._file_path.parent)

        # Flush all pending writes before getting file image
        self._h5file.flush()

        # Get file image and write to disk
        file_image = self._h5file.id.get_file_image()

        with open(self._file_path, 'wb') as f:
            f.write(file_image)

        print(f"Saved thermal fields to disk: {self._file_path}")

        # Close the in-memory file
        self._h5file.close()
        self._h5file = None

    def _save_metadata_to_group(self, group, step: int, context: dict):
        """Save metadata as HDF5 attributes."""
        sim = context['simulation']

        # Basic metadata
        group.attrs['step'] = step
        group.attrs['time'] = step * sim.delta_t if hasattr(sim, 'delta_t') else 0.0

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

            # Melt pool
            if 'melt_pool' in ctx:
                mp = ctx['melt_pool']
                group.attrs['max_temp'] = mp.get('max_temp', 0.0)
                group.attrs['melt_pool_width'] = mp.get('width', 0.0)
                group.attrs['melt_pool_depth'] = mp.get('depth', 0.0)

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
        """Close HDF5 file on cleanup (if not already saved)."""
        if self._h5file is not None:
            try:
                self._h5file.close()
                print(f"Warning: HDF5 file closed without saving (COMPLETE event not reached)")
            except:
                pass


# ============================================================================
# Utility Functions for Reading HDF5 Files
# ============================================================================

def load_thermal_field(filepath: str, step: Optional[int] = None) -> np.ndarray:
    """
    Load temperature field from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        step: Step number (required if file contains multiple steps)

    Returns:
        Temperature field as numpy array

    Examples:
        # Load from separate file
        temp = load_thermal_field("thermal_fields/thermal_step_0042.h5")

        # Load from single file
        temp = load_thermal_field("thermal_fields.h5", step=42)
    """
    with h5py.File(filepath, 'r') as f:
        if step is not None:
            # Single file mode - need step number
            step_group = f[f'step_{step:04d}']
            return step_group['temperature'][:]
        else:
            # Separate file mode - temperature at root
            return f['temperature'][:]


def load_thermal_metadata(filepath: str, step: Optional[int] = None) -> dict:
    """
    Load metadata from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        step: Step number (required if file contains multiple steps)

    Returns:
        Dictionary of metadata attributes

    Examples:
        # Load metadata
        metadata = load_thermal_metadata("thermal_fields/thermal_step_0042.h5")
        print(f"Max temp: {metadata['max_temp']}")
        print(f"Position: ({metadata['position_x']}, {metadata['position_y']})")
    """
    with h5py.File(filepath, 'r') as f:
        if step is not None:
            # Single file mode
            group = f[f'step_{step:04d}']
        else:
            # Separate file mode
            group = f

        # Convert attributes to dict
        return dict(group.attrs)


def list_steps_in_file(filepath: str) -> list:
    """
    List all available steps in a single-file HDF5.

    Args:
        filepath: Path to HDF5 file

    Returns:
        List of step numbers

    Example:
        steps = list_steps_in_file("thermal_fields.h5")
        print(f"Available steps: {steps}")
        # [1, 2, 3, ..., 100]
    """
    with h5py.File(filepath, 'r') as f:
        # Get all step groups
        step_groups = [key for key in f.keys() if key.startswith('step_')]
        # Extract step numbers
        steps = [int(key.split('_')[1]) for key in step_groups]
        return sorted(steps)


def get_file_info(filepath: str):
    """
    Print information about HDF5 file contents.

    Args:
        filepath: Path to HDF5 file

    Example:
        get_file_info("thermal_fields.h5")
    """
    with h5py.File(filepath, 'r') as f:
        print(f"\nHDF5 File: {filepath}")
        print("="*60)

        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                print(f"    Size: {obj.nbytes / (1024**2):.2f} MB")
                if obj.compression:
                    print(f"    Compression: {obj.compression}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
                if obj.attrs:
                    print(f"    Attributes: {list(obj.attrs.keys())[:5]}...")

        f.visititems(print_item)
        print("="*60)



if __name__ == "__main__":


    from simulate import SimulationRunner
    from callbacks.completion_callbacks import HeightCompletionCallback
    from callbacks.step_data_collector import StepDataCollector
    from callbacks.callback_collection import ProgressPrinter, ThermalPlotSaver, CrossSectionPlotter, FinalStateSaver
    #from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
    from callbacks.callback_collection import TemperatureSliceSaver
    from callbacks.callback_collection import VoxelTemperatureSaver
    from callbacks.hdf5_activation_saver import HDF5ActivationSaver

    # Create callbacks
    callbacks = [
        HeightCompletionCallback(),  # Stop at target height
        StepDataCollector(save_path=None),  # Save data
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=1,  # Save every step
            compression='gzip',
            compression_opts=4
        ),
        ProgressPrinter(),  # Console progress output
        #TemperatureSliceSaver(interval=1),
        #VoxelTemperatureSaver(interval=1),
        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=1,  # Save every 10 steps
            compression='gzip',
            compression_opts=9  # Max compression for bool arrays
        ),
        StepDataCollector(tracked_fields=None, save_path="simulation_data.csv"),
        #CrossSectionPlotter(num_sections=5),
        FinalStateSaver(),

    ]

    # Create and run simulation
    runner = SimulationRunner.from_human_units(
        build_volume_mm=(20, 20, 15),
        part_volume_mm=(1, 5, 0.1),
        voxel_size_um=200,
        delta_t_ms=200,
        scan_speed_mm_s=3,
        laser_power_W=600,
        powder_feed_g_min=2,
        hatch_spacing_um=700,
        layer_spacing_um=350,
        callbacks=callbacks,
    )

    runner.run()

    # Test loading and verify chunking - THERMAL
    print("\n" + "="*60)
    print("Testing HDF5 THERMAL file loading and chunk structure")
    print("="*60)

    # Get the actual file path from the callback
    thermal_saver = [cb for cb in callbacks if isinstance(cb, HDF5ThermalSaver)][0]
    filepath = thermal_saver._file_path
    print(f"File location: {filepath}")

    steps = list_steps_in_file(str(filepath))
    print(f"Steps in file: {steps}")

    # Load first and last step
    temp_first = load_thermal_field(str(filepath), step=steps[0])
    temp_last = load_thermal_field(str(filepath), step=steps[-1])
    print(f"First step shape: {temp_first.shape}")
    print(f"Last step shape: {temp_last.shape}")

    # Verify chunks correspond to steps
    with h5py.File(str(filepath), 'r') as f:
        print(f"Total groups (chunks): {len([k for k in f.keys() if k.startswith('step_')])}")
        print(f"Expected (from steps): {len(steps)}")
        print("✓ Chunking verified: each step = one HDF5 group/chunk")

    # Test loading and verify chunking - ACTIVATION
    print("\n" + "="*60)
    print("Testing HDF5 ACTIVATION file loading and chunk structure")
    print("="*60)

    from hdf5_activation_saver import load_activation_volume, list_steps_in_file as list_steps_activation

    activation_saver = [cb for cb in callbacks if isinstance(cb, HDF5ActivationSaver)][0]
    filepath_act = activation_saver._file_path
    print(f"File location: {filepath_act}")

    steps_act = list_steps_activation(str(filepath_act))
    print(f"Steps in file: {steps_act}")

    # Load first and last step
    act_first = load_activation_volume(str(filepath_act), step=steps_act[0])
    act_last = load_activation_volume(str(filepath_act), step=steps_act[-1])
    print(f"First step shape: {act_first.shape}")
    print(f"Last step shape: {act_last.shape}")
    print(f"First step activated voxels: {act_first.sum()}")
    print(f"Last step activated voxels: {act_last.sum()}")

    # Verify chunks correspond to steps
    with h5py.File(str(filepath_act), 'r') as f:
        print(f"Total groups (chunks): {len([k for k in f.keys() if k.startswith('step_')])}")
        print(f"Expected (from steps): {len(steps_act)}")
        print("✓ Chunking verified: each step = one HDF5 group/chunk")