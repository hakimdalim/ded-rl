"""
Missing callbacks ported to the new callback system.
These callbacks were in the old saving_callbacks.py but not yet implemented in the new system.
"""

import os
import warnings
import shutil
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from callbacks._base_callbacks import BaseCallback, IntervalCallback, SimulationEvent


class TemperatureSliceSaver(IntervalCallback):
    """
    Saves 2D temperature field slices in three orthogonal planes.

    When:
        - STEP_COMPLETE event (configurable via interval)

    What is saved:
        - xy_slice_stepXXXX.npy: Horizontal slice at current z position
        - xz_slice_stepXXXX.npy: Front view slice at current y position
        - yz_slice_stepXXXX.npy: Side view slice at current x position
        - Each file contains 2D numpy array of temperatures in Kelvin

    Output: 'temperatures/' directory with .npy files for each plane/step
    """

    def __init__(self, save_dir: str = "temperatures", interval: int = 1, **kwargs):
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )
        self.save_dir = save_dir

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        # Get temperature slices from simulation's temp_slices property
        temp_slices = sim.temp_slices

        if not temp_slices:
            warnings.warn("No temperature slices available")
            return

        # Resolve save directory
        save_path = self.resolve_path(context, self.save_dir)
        self.ensure_dir(save_path)

        # Save each slice
        step = sim.progress_tracker.step_count
        for plane, data in temp_slices.items():
            filename = save_path / f"{plane}_slice_step{step:04d}.npy"
            np.save(filename, data)


class VoxelTemperatureSaver(IntervalCallback):
    """
    Saves complete 3D temperature field of entire simulation volume.

    When:
        - STEP_COMPLETE event (default: every 10 steps via interval)

    What is saved:
        - voxel_temps_stepXXXX.npy: Full 3D temperature array
        - Shape: (nx, ny, nz) matching simulation volume discretization
        - Values: Temperature in Kelvin for each voxel

    Output: 'voxel_temps/' directory with .npy files
    Note: Large files - use interval to reduce frequency
    """

    def __init__(self, save_dir: str = "voxel_temps", interval: int = 10, **kwargs):
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )
        self.save_dir = save_dir

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        # Get full temperature field
        temp_field = sim.temperature_tracker.temperature

        # Resolve save directory
        save_path = self.resolve_path(context, self.save_dir)
        self.ensure_dir(save_path)

        # Save temperature field
        step = sim.progress_tracker.step_count
        filename = save_path / f"voxel_temps_step{step:04d}.npy"
        np.save(filename, temp_field)


class MeshSaver(IntervalCallback):
    """
    Saves 3D mesh representation of built geometry as STL files.

    When:
        - STEP_COMPLETE event (default: every 10 steps via interval)

    What is saved:
        - build_state_stepXXXX.stl: STL mesh file of current build state
        - Contains triangulated surface mesh of all deposited material
        - Can be viewed in 3D software (MeshLab, ParaView, etc.)

    Output: 'build_mesh/' directory with .stl files
    Note: STL generation can be slow for complex geometries
    """

    def __init__(self, save_dir: str = "build_mesh", interval: int = 10, **kwargs):
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )
        self.save_dir = save_dir

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        # Resolve save directory
        save_path = self.resolve_path(context, self.save_dir)
        self.ensure_dir(save_path)

        # Save mesh using clad_manager's method
        step = sim.progress_tracker.step_count
        filename = save_path / f"build_state_step{step:04d}.stl"
        sim.clad_manager.save_build_state_mesh(str(filename))


class ThermalPlotSaver(IntervalCallback):
    """
    Creates thermal visualization plots with temperature fields and contours.

    When:
        - STEP_COMPLETE event (default: every 10 steps via interval)

    What is saved:
        - thermalXXXX_top_view.png: XY plane temperature plot
        - thermalXXXX_front_view.png: XZ plane temperature plot
        - thermalXXXX_side_view.png: YZ plane temperature plot
        - Each plot shows temperature field (300-2500K), melt pool boundary,
          and activated volume contour

    Output: 'thermal_plots/' directory with PNG images
    Note: Matplotlib plotting can be slow - use interval
    """

    def __init__(self, save_dir: str = "thermal_plots", interval: int = 10, **kwargs):
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )
        self.save_dir = save_dir

    def _execute(self, context: dict) -> None:
        sim = context['simulation']
        step_ctx = sim.step_context

        if not step_ctx:
            warnings.warn("No step context available for thermal plots")
            return

        # Resolve save directory
        save_path = self.resolve_path(context, self.save_dir)
        self.ensure_dir(save_path)

        # Get positions and parameters
        x_val = step_ctx['position']['x']
        y_val = step_ctx['position']['y']
        z_val = step_ctx['position']['z']
        params = step_ctx['params']

        # Create thermal plots
        step = sim.progress_tracker.step_count
        self._save_thermal_plots(
            sim,
            str(save_path),
            f"thermal{step:04d}",
            x_val, y_val, z_val,
            params
        )

    def _save_thermal_plots(self, simulation, save_dir, id_name, x_val, y_val, z_val, params):
        """
        Save individual thermal plots for XY, XZ, and YZ planes.
        Ported directly from simulate.py to avoid import dependencies.
        """
        # Get activated volume mask for contours
        activated_mask = simulation.volume_tracker.activated

        # Define the three plot configurations
        plot_configs = [
            {
                'plane': 'xy',
                'slice_val': z_val,
                'slice_idx': int(z_val / simulation.config['voxel_size'][2]),
                'view_name': 'top',
                'title': f'Top View (z={z_val * 1000:.3f}mm)',
                'mask_slice': lambda mask: mask[:, :, int(z_val / simulation.config['voxel_size'][2])].T,
                'coords': lambda mask: (
                    np.linspace(0, simulation.config['volume_shape'][0] * simulation.config['voxel_size'][0] * 1000,
                                mask.shape[0]),
                    np.linspace(0, simulation.config['volume_shape'][1] * simulation.config['voxel_size'][1] * 1000,
                                mask.shape[1])
                )
            },
            {
                'plane': 'xz',
                'slice_val': y_val,
                'slice_idx': int(y_val / simulation.config['voxel_size'][1]),
                'view_name': 'front',
                'title': f'Front View (y={y_val * 1000:.3f}mm)',
                'mask_slice': lambda mask: mask[:, int(y_val / simulation.config['voxel_size'][1]), :].T,
                'coords': lambda mask: (
                    np.linspace(0, simulation.config['volume_shape'][0] * simulation.config['voxel_size'][0] * 1000,
                                mask.shape[0]),
                    np.linspace(0, simulation.config['volume_shape'][2] * simulation.config['voxel_size'][2] * 1000,
                                mask.shape[2])
                )
            },
            {
                'plane': 'yz',
                'slice_val': x_val,
                'slice_idx': int(x_val / simulation.config['voxel_size'][0]),
                'view_name': 'side',
                'title': f'Side View (x={x_val * 1000:.3f}mm)',
                'mask_slice': lambda mask: mask[int(x_val / simulation.config['voxel_size'][0]), :, :].T,
                'coords': lambda mask: (
                    np.linspace(0, simulation.config['volume_shape'][1] * simulation.config['voxel_size'][1] * 1000,
                                mask.shape[1]),
                    np.linspace(0, simulation.config['volume_shape'][2] * simulation.config['voxel_size'][2] * 1000,
                                mask.shape[2])
                )
            }
        ]

        # Create and save each plot
        for config in plot_configs:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot temperature slice
            simulation.temperature_tracker.plot_temperature_slice(
                ax=ax,
                plane=config['plane'],
                slice_idx=config['slice_idx'],
                temp_range=(300, 2500),
                show_melt_pool=False,
                melting_temp=params.get('melting_temp', 1700),
                title=config['title'],
                cmap='hot'
            )

            # Add activation contour
            activated_slice = config['mask_slice'](activated_mask)
            x_coords, y_coords = config['coords'](activated_mask)
            ax.contour(x_coords, y_coords, activated_slice,
                       levels=[0.5], colors='white', linestyles='dashed', linewidths=0.5,
                       antialiased=True, corner_mask=True)

            # Save plot
            filename = f"{id_name}_{config['view_name']}_view.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)


class ProgressPrinter(BaseCallback):
    """
    Prints simulation progress and parameters to console.

    When:
        - INIT event: Prints simulation parameters and start time
        - TRACK_COMPLETE event: Prints layer/track number and current build height

    What is printed:
        - At init: Build/part volumes, voxel size, process parameters
        - Per track: Layer number, track number, maximum height reached

    Output: Console (stdout)
    """

    def __init__(self, **kwargs):
        super().__init__(
            events=[SimulationEvent.INIT, SimulationEvent.TRACK_COMPLETE],
            **kwargs
        )
        self.max_height_reached = 0.0

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        if context['event'] == SimulationEvent.INIT:
            config = sim.config

            print(f"\nSimulation initialized with parameters:")
            print(f"Build volume: {[x * 1000 for x in config['voxel_size']]} mm/voxel")
            print(f"Volume shape: {config['volume_shape']} voxels")
            print(f"Track length: {config.get('track_length', 0) * 1000:.1f} mm")
            print(f"Hatch spacing: {config.get('hatch_spacing', 0) * 1000:.1f} mm")
            print(f"Number of tracks: {config.get('num_tracks', 0)}")
            print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Output directory (if needed by callbacks):\n{sim.output_dir}\n")

        elif context['event'] == SimulationEvent.TRACK_COMPLETE:
            summary = sim.progress_tracker.get_transition_summary()

            # Use the cumulative values from progress_tracker
            print(f"Layer (count): {summary['count_layers'] + 1:3d}  |  "
                  f"Track (count): {summary['count_tracks'] + 1:3d}  |  "
                  f"Height (max): {summary['max_height_reached'] * 1000:6.4f} mm")


class CrossSectionPlotter(BaseCallback):
    """
    Saves cross-section plots at simulation completion.

    When:
        - COMPLETE event

    What is saved:
        - cross_section_yXXXmm.png: Cross-section plots at multiple y positions
        - Shows all layers and tracks at each y position

    Output: 'cross_sections/' directory with PNG images
    """

    def __init__(self, save_dir: str = "cross_sections", num_sections: int = 5, **kwargs):
        super().__init__(
            events=SimulationEvent.COMPLETE,
            **kwargs
        )
        self.save_dir = save_dir
        self.num_sections = num_sections

    def _execute(self, context: dict) -> None:
        sim = context['simulation']
        clad_manager = sim.clad_manager

        # Get all y positions from profiles (as shown in simulate.py)
        y_positions = sorted(set(y for (_, _, y) in clad_manager._profiles.keys()))

        if not y_positions:
            warnings.warn("No profiles found in clad_manager")
            return

        # Select y positions for cross-sections
        y_min, y_max = min(y_positions), max(y_positions)
        y_intervals = np.linspace(y_min, y_max, min(self.num_sections, len(y_positions)))

        # Resolve save directory
        save_path = self.resolve_path(context, self.save_dir)
        self.ensure_dir(save_path)

        print(f"\nSaving {len(y_intervals)} cross-section plots...")

        for i, y_pos in enumerate(y_intervals):
            fig, ax = plt.subplots(figsize=(20, 12))

            try:
                # Use clad_manager's plot_all_layers method (as used in simulate.py)
                clad_manager.plot_all_layers(y_pos=y_pos, ax=ax)

                filename = save_path / f"cross_section_y{y_pos * 1000:.1f}mm.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot {i + 1}/{len(y_intervals)}")
            except Exception as e:
                plt.close(fig)
                warnings.warn(f"Could not save cross-section plot at y={y_pos * 1000:.1f}mm: {e}")


class PickleSaver(BaseCallback):
    """
    Saves objects as pickle files.

    Generic callback for pickling any object accessible from the simulation.
    """

    def __init__(
            self,
            obj_getter: Callable[[dict], Any],
            save_file: str,
            event: SimulationEvent = SimulationEvent.COMPLETE,
            **kwargs
    ):
        """
        Args:
            obj_getter: Function(context) -> object to pickle
            save_file: Filename for pickle file
            event: When to save (default: COMPLETE)
        """
        super().__init__(events=event, **kwargs)
        self.obj_getter = obj_getter
        self.save_file = save_file

    def _execute(self, context: dict) -> None:
        # Get object to pickle
        obj = self.obj_getter(context)

        # Resolve save path
        save_path = self.resolve_path(context, self.save_file)

        # Save pickle
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved pickle to {save_path}")


class CompressCallback(BaseCallback):
    """
    Compresses directories at simulation completion.

    When:
        - COMPLETE event

    What happens:
        - Creates archive of specified directory
        - Optionally removes original directory

    Output: Compressed archive (zip, tar, etc.)
    """

    def __init__(
            self,
            target_dir: str,
            archive_format: str = 'zip',
            remove_original: bool = True,
            **kwargs
    ):
        super().__init__(events=SimulationEvent.COMPLETE, **kwargs)
        self.target_dir = target_dir
        self.archive_format = archive_format
        self.remove_original = remove_original

    def _execute(self, context: dict) -> None:
        # Resolve target directory
        target_path = self.resolve_path(context, self.target_dir)

        if not target_path.exists():
            warnings.warn(f"Target directory {target_path} does not exist")
            return

        print(f"\nCompressing {target_path}...")

        # Create archive
        shutil.make_archive(str(target_path), self.archive_format, target_path)

        # Remove original if requested
        if self.remove_original:
            shutil.rmtree(target_path)
            print(f"Removed original directory")


class ParameterLogger(BaseCallback):
    """
    Tracks process parameter changes throughout simulation.

    When:
        - STEP_COMPLETE event: Collects parameter values
        - COMPLETE event: Saves parameter history to CSV

    What is saved:
        - CSV with columns: step, param.laser_power, param.scan_speed, etc.
        - Each row shows parameter values at that step
        - Useful for adaptive control analysis

    Output: 'parameter_history.csv' in output directory
    """

    def __init__(self, save_file: str = "parameter_history.csv", **kwargs):
        super().__init__(
            events=[SimulationEvent.STEP_COMPLETE, SimulationEvent.COMPLETE],
            **kwargs
        )
        self.save_file = save_file
        self.param_history = []

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        if context['event'] == SimulationEvent.STEP_COMPLETE:
            # Get current parameters from step context
            if sim.step_context and 'params' in sim.step_context:
                params = sim.step_context['params']
                step = sim.progress_tracker.step_count

                param_entry = {
                    'step': step,
                    **{f'param.{k}': v for k, v in params.items()}
                }
                self.param_history.append(param_entry)

        elif context['event'] == SimulationEvent.COMPLETE:
            if self.param_history:
                # Save parameter history
                df = pd.DataFrame(self.param_history)
                save_path = self.resolve_path(context, self.save_file)
                df.to_csv(save_path, index=False)
                print(f"Saved parameter history to {save_path}")


class LivePlotter(IntervalCallback):
    """
    Creates real-time plot showing build progress.

    When:
        - STEP_COMPLETE event (default: every 10 steps via interval)

    What is displayed:
        - Interactive matplotlib plot (requires interactive backend)
        - Shows maximum build height over time

    Output: Live matplotlib window
    Note: May slow down simulation if updated too frequently
    """

    def __init__(self, interval: int = 10, **kwargs):
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            **kwargs
        )
        self.fig = None
        self.axes = None
        self.heights = []
        self.steps = []

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        # Collect data
        step = sim.progress_tracker.step_count
        self.steps.append(step)

        # Get max height from profile in step_context
        max_height = 0.0
        if sim.step_context and 'profile' in sim.step_context:
            profile_data = sim.step_context['profile']
            max_height = profile_data.get('max_z', 0.0)

        self.heights.append(max_height * 1000)  # Convert to mm

        # Initialize plot on first call
        if self.fig is None:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(1, 1, figsize=(10, 6))

        # Update plot
        self.axes.clear()
        self.axes.plot(self.steps, self.heights, 'b-')
        self.axes.set_xlabel('Step')
        self.axes.set_ylabel('Max Height (mm)')
        self.axes.set_title('Build Progress')
        self.axes.grid(True)

        plt.draw()
        plt.pause(0.01)


class FinalStateSaver(BaseCallback):
    """
    Saves final simulation state at completion.

    When:
        - COMPLETE event

    What is saved:
        - final_activated_vol.npy: Final activated volume state
        - final_temperature_vol.npy: Final temperature field
        - simulation_params.csv: Simulation parameters and statistics

    Output: Files in output directory root
    """

    def __init__(self, **kwargs):
        super().__init__(events=SimulationEvent.COMPLETE, **kwargs)

    def _execute(self, context: dict) -> None:
        sim = context['simulation']

        # Save final volume states
        output_dir = Path(sim.output_dir)
        np.save(output_dir / "final_activated_vol.npy", sim.volume_tracker.activated)
        np.save(output_dir / "final_temperature_vol.npy", sim.temperature_tracker.temperature)

        # Prepare simulation parameters
        summary = sim.progress_tracker.get_transition_summary()
        config = sim.config

        # Get final height from last profile if available
        max_height = 0.0
        if sim.step_context and 'profile' in sim.step_context:
            max_height = sim.step_context['profile'].get('max_z', 0.0)

        params_dict = {
            'voxel_size': config.get('voxel_size', [0, 0, 0])[0],
            'track_length': config.get('track_length', 0),
            'hatch_spacing': config.get('hatch_spacing', 0),
            'num_tracks': config.get('num_tracks', 0),
            'completed_steps': summary['step_count'],
            'completed_layers': summary['current_layer'],
            'max_height_reached': max_height,
        }

        # Save parameters
        pd.Series(params_dict).to_csv(output_dir / "simulation_params.csv")
        print(f"Saved final simulation state to {output_dir}")


# ============================================================================
# Helper Functions for Creating Callback Sets
# ============================================================================

def get_visualization_callbacks(interval: int = 10) -> List[BaseCallback]:
    """Get callbacks for visualization outputs."""
    return [
        TemperatureSliceSaver(interval=interval),
        VoxelTemperatureSaver(interval=interval),
        MeshSaver(interval=interval),
        ThermalPlotSaver(interval=interval),
        CrossSectionPlotter(),
    ]


def get_data_callbacks() -> List[BaseCallback]:
    """Get callbacks for data collection and saving."""
    from callbacks.step_data_collector import StepDataCollector

    return [
        StepDataCollector(tracked_fields=None, save_path="simulation_data.csv"),
        ParameterLogger(),
        PickleSaver(
            lambda ctx: ctx['simulation'].clad_manager,
            save_file="clad_manager.pkl"
        ),
        FinalStateSaver(),
    ]


def get_monitoring_callbacks(live_plot: bool = False) -> List[BaseCallback]:
    """Get callbacks for monitoring simulation progress."""
    callbacks = [ProgressPrinter()]

    if live_plot:
        callbacks.append(LivePlotter(interval=10))

    return callbacks


def get_compression_callbacks() -> List[BaseCallback]:
    """Get callbacks for post-processing compression."""
    return [
        CompressCallback(target_dir="temperatures"),
        CompressCallback(target_dir="voxel_temps"),
    ]


def get_default_callbacks() -> List[BaseCallback]:
    """Get the default callback set matching the original simulate.py behavior."""
    from callbacks.step_data_collector import StepDataCollector
    from callbacks.completion_callbacks import HeightCompletionCallback

    return [
        # Completion condition
        HeightCompletionCallback(),

        # Data collection
        StepDataCollector(tracked_fields=None, save_path="simulation_data.csv"),

        # Visualization saves (every step like original)
        TemperatureSliceSaver(interval=1),
        VoxelTemperatureSaver(interval=1),
        MeshSaver(interval=1),
        ThermalPlotSaver(interval=1),

        # Progress monitoring
        ProgressPrinter(),

        # Final outputs
        CrossSectionPlotter(num_sections=5),
        PickleSaver(lambda ctx: ctx['simulation'].clad_manager, save_file="clad_manager.pkl"),
        FinalStateSaver(),

        # Compression
        CompressCallback(target_dir="temperatures"),
    ]