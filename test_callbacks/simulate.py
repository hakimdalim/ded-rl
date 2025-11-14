"""
SimulationRunner - Streamlined runner for DED simulations with new callback system.
"""
import os
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import pandas as pd

# Import SimulationComplete from completion_callbacks
from callbacks.completion_callbacks import SimulationComplete


class SimulationRunner:
    """
    Streamlined runner for DED simulations with new callback system.

    The runner now primarily handles:
    - Initialization and output directory setup
    - The main simulation loop
    - Backward compatibility interface
    - Parameter updates during simulation

    All tracking and data collection is now handled by the simulation itself
    and the callback system.
    """

    def __init__(
        self,
        simulation,
        config,
        initial_params: Dict[str, Any],
    ):
        """
        Initialize the simulation runner.

        Args:
            simulation: MultiTrackMultiLayerSimulation instance
            config: SimulationConfig instance
            initial_params: Initial process parameters dictionary
        """
        self.simulation = simulation
        self.config = config
        self.initial_params = initial_params.copy()
        self.params = initial_params.copy()

        # Store for backward compatibility
        self._step_dict = {}
        self._temp_slices = {}
        self._progress_dict = {}

    @classmethod
    def from_si_units(
        cls,
        build_volume: Tuple[float, float, float],  # meters
        part_volume: Tuple[float, float, float],   # meters
        voxel_size: float,                          # meters
        delta_t: float,                             # seconds
        scan_speed: float,                          # m/s
        laser_power: float,                         # W
        powder_feed_rate: float,                    # kg/s
        hatch_spacing: float = 0.0007,             # meters
        layer_spacing: float = 0.00035,
        substrate_height: float = 0.005,           # meters
        output_base_dir: str = "_experiments",
        experiment_label: Optional[str] = None,
        callbacks: Optional[List] = None,
        **kwargs
    ):
        """
        Initialize from SI units (meters, seconds, kg, W).
        """
        from core.multi_track_multi_layer import MultiTrackMultiLayerSimulation
        from configuration.simulation_config import SimulationConfig
        from configuration.process_parameters import set_params

        # Create output directory
        output_dir = cls._name_output_directory(
            build_volume, part_volume, voxel_size, delta_t,
            scan_speed, laser_power, powder_feed_rate,
            output_base_dir, experiment_label
        )

        # Create simulation configuration
        config = SimulationConfig(
            build_volume_size=build_volume,
            voxel_size=voxel_size,
            part_width=part_volume[0],
            part_length=part_volume[1],
            part_height=part_volume[2],
            hatch_spacing=hatch_spacing,
            layer_spacing=layer_spacing,
            substrate_height=substrate_height,
            bidirectional_tracks=kwargs.pop('bidirectional_tracks', True),
            bidirectional_layers=kwargs.pop('bidirectional_layers', True),
            switch_scan_direction_between_layers=kwargs.pop('switch_scan_direction', True),
            turnaround_time=kwargs.pop('turnaround_time', 0.0),
        )

        # Get process parameters
        params = set_params(
            laser_power=laser_power,
            scan_speed=scan_speed,
            powder_feed_rate=powder_feed_rate
        )

        # Create simulation instance with callbacks and output_dir
        simulation = MultiTrackMultiLayerSimulation(
            config=config.get_simulation_config(),
            delta_t=delta_t,
            callbacks=callbacks,
            output_dir=output_dir,
        )
        simulation.reset()

        # Create runner instance
        instance = cls(simulation, config, params)

        return instance

    @classmethod
    def from_human_units(
        cls,
        build_volume_mm: Tuple[float, float, float] = (20.0, 20.0, 15.0),
        part_volume_mm: Tuple[float, float, float] = (5.0, 5.0, 5.0),
        voxel_size_um: float = 200.0,
        delta_t_ms: float = 200.0,
        scan_speed_mm_s: float = 3.0,
        laser_power_W: float = 600.0,
        powder_feed_g_min: float = 2.0,
        hatch_spacing_um: float = 700.0,
        layer_spacing_um: float = 350.0,
        substrate_height_mm: float = 5.0,
        output_base_dir: str = "_experiments",
        experiment_label: str = "unlabeled",
        callbacks: Optional[List] = None,
        **kwargs
    ):
        """
        Initialize from human-readable units.
        """
        # Convert to SI units
        build_volume = tuple(x / 1000 for x in build_volume_mm)
        part_volume = tuple(x / 1000 for x in part_volume_mm)
        voxel_size = voxel_size_um / 1e6
        delta_t = delta_t_ms / 1000
        scan_speed = scan_speed_mm_s / 1000
        powder_feed_rate = powder_feed_g_min * (1/1000) * (1/60)  # g/min to kg/s
        hatch_spacing = hatch_spacing_um / 1e6
        layer_spacing = layer_spacing_um / 1e6
        substrate_height = substrate_height_mm / 1000

        return cls.from_si_units(
            build_volume=build_volume,
            part_volume=part_volume,
            voxel_size=voxel_size,
            delta_t=delta_t,
            scan_speed=scan_speed,
            laser_power=laser_power_W,
            powder_feed_rate=powder_feed_rate,
            hatch_spacing=hatch_spacing,
            layer_spacing=layer_spacing,
            substrate_height=substrate_height,
            output_base_dir=output_base_dir,
            experiment_label=experiment_label,
            callbacks=callbacks,
            **kwargs
        )

    def run(self, params_updater: Optional[Callable] = None) -> None:
        """
        Run the simulation.

        Args:
            params_updater: Optional function that takes (params, runner)
                          and returns updated params. The runner provides
                          access to step_dict, temp_slices, and progress_dict.
        """
        try:
            # Run simulation loop
            while True:
                try:
                    # Perform simulation step
                    self.simulation.step(self.params)

                    # Update parameters if updater provided
                    if params_updater:
                        # Only update compatibility data if param updater needs it
                        self._update_compatibility_data()
                        self.params = params_updater(self.params, self)

                except SimulationComplete as e:
                    print(f"\n{e}")
                    break

        except Exception as e:
            tb = ''.join(traceback.format_tb(e.__traceback__))
            warnings.warn(f"\nTraceback:\n{tb}\n\nBuild interrupted: {e!r}")
            raise

    def _update_compatibility_data(self):
        """
        Update backward compatibility data structures.
        These allow the runner to be used like the old Simulator class.
        """
        sim = self.simulation

        # Build step_dict from simulation's step_context
        if sim.step_context:
            ctx = sim.step_context
            summary = sim.progress_tracker.get_transition_summary()

            self._step_dict = {
                'step': summary['step_count'],

                # Position data
                'position.x': ctx['position']['x'],
                'position.y': ctx['position']['y'],
                'position.z': ctx['position']['z'],

                # Voxel indices
                'voxel.x': ctx['voxel']['x'],
                'voxel.y': ctx['voxel']['y'],
                'voxel.z': ctx['voxel']['z'],

                # Build progress
                'build.layer': ctx['build']['layer'],
                'build.track': ctx['build']['track'],
                'build.reverse_direction': ctx['build']['reverse_direction'],
                'build.reverse_track': ctx['build']['reverse_track'],
                'build.time_step': ctx['build']['time_step'],

                # Melt pool dimensions
                'melt_pool.width': ctx['melt_pool']['width'],
                'melt_pool.length': ctx['melt_pool']['length'],
                'melt_pool.depth': ctx['melt_pool']['depth'],

                # Clad dimensions
                'clad.width': ctx['clad']['width'],
                'clad.height': ctx['clad']['height'],
                'clad.wetting_angle': ctx['clad']['wetting_angle'],

                # Profile data
                'profile.baseline': ctx['profile']['baseline'],
                'profile.max_z': ctx['profile']['max_z'],
                'profile.height_at_center': ctx['profile']['height_at_center'],
                'profile.width': ctx['profile']['width'],
            }

            # Add profile object data if available
            if 'object' in ctx['profile']:
                profile = ctx['profile']['object']
                self._step_dict.update({
                    'profile.start_x': profile.start_x,
                    'profile.end_x': profile.end_x,
                    'profile.a': profile.a,
                    'profile.b': profile.b,
                    'profile.c': profile.c,
                    'profile.track_center': profile.offset,
                    'profile.required_area': profile.required_area,
                })

        # Update temperature slices
        self._temp_slices = sim.temp_slices

        # Update progress dict
        summary = sim.progress_tracker.get_transition_summary()
        self._progress_dict = {
            'count_steps': summary['step_count'],
            'count_layers': summary.get('count_layers', summary['current_layer']),
            'count_tracks': summary.get('count_tracks', 0),
            'max_height_reached': summary.get('max_height_reached', 0.0)
        }

    # ========================================================================
    # Backward Compatibility Properties and Methods
    # ========================================================================

    @property
    def step_dict(self) -> Dict[str, Any]:
        """Access step data dictionary for backward compatibility."""
        return self._step_dict.copy()

    @property
    def temp_slices(self) -> Dict[str, np.ndarray]:
        """Access temperature slices for backward compatibility."""
        return self._temp_slices

    @property
    def progress_dict(self) -> Dict[str, Any]:
        """Access progress dictionary for backward compatibility."""
        return self._progress_dict

    def send(self, new_params: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict, Dict, Dict, Dict]:
        """
        Backward compatible send method that mimics generator behavior.

        Returns:
            Tuple of (step, step_dict, temp_slices, params, progress_dict)
        """
        if new_params is not None:
            self.params = new_params

        # Perform step
        self.simulation.step(self.params)

        # Update compatibility data for send method
        self._update_compatibility_data()

        return (
            self._progress_dict['count_steps'],
            self._step_dict,
            self._temp_slices,
            self.params,
            self._progress_dict
        )

    @staticmethod
    def _name_output_directory(
        build_vol: tuple, part_vol: tuple, voxel_size: float, delta_t: float,
        scan_speed: float, laser_power: float, powder_feed_rate: float,
        base_dir: str, experiment_label: Optional[str]
    ) -> str:
        """Create output directory with descriptive name including job ID."""

        # Handle base directory with SCRATCH environment variable
        exp_dir = os.environ.get('SCRATCH', None)
        if exp_dir is not None:
            exp_dir = os.path.join(exp_dir, 'ded_sim_experiments')
        else:
            exp_dir = base_dir

        if experiment_label:
            exp_dir = os.path.join(exp_dir, experiment_label)

        # Get job ID from SLURM environment or timestamp
        array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
        array_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if array_job_id and array_id:
            job_id = f"{array_job_id}_{array_id}"
        else:
            job_id = os.environ.get('SLURM_JOB_ID', str(time.time_ns()))

        # Create directory name with parameters
        dirname = SimulationRunner._create_safe_dirname(
            build_vol, part_vol, voxel_size, delta_t,
            scan_speed, laser_power, powder_feed_rate
        )

        output_dir = os.path.join(exp_dir, f"job{job_id}_{dirname}")
        #os.makedirs(output_dir, exist_ok=True)

        return output_dir

    @staticmethod
    def _create_safe_dirname(
        build_vol: tuple, part_vol: tuple, voxel_size: float, delta_t: float,
        scan_speed: float, laser_power: float, powder_feed_rate: float
    ) -> str:
        """Create a safe directory name with simulation parameters."""

        # Format dimensions as strings with one decimal point
        build_str = f"{build_vol[0] * 1000:.1f}x{build_vol[1] * 1000:.1f}x{build_vol[2] * 1000:.1f}mm"
        part_str = f"{part_vol[0] * 1000:.1f}x{part_vol[1] * 1000:.1f}x{part_vol[2] * 1000:.1f}mm"

        # Create directory name
        dirname = (f"build{build_str}_part{part_str}"
                  f"_vox{voxel_size * 1e6:.1f}um_dt{delta_t * 1000:.1f}ms"
                  f"_v{scan_speed * 1000:.1f}mms_p{laser_power:.1f}W"
                  f"_f{powder_feed_rate * 60 * 1000:.1f}gmin")

        return dirname


if __name__ == "__main__":
    import argparse

    # Import callback helpers
    from callbacks.callback_collection import get_default_callbacks

    parser = argparse.ArgumentParser(description='Run DED simulation')

    # Build volume parameters
    parser.add_argument('--build-x', type=float, default=20.0,
                       help='Build volume X dimension in mm (default: 20.0)')
    parser.add_argument('--build-y', type=float, default=20.0,
                       help='Build volume Y dimension in mm (default: 20.0)')
    parser.add_argument('--build-z', type=float, default=15.0,
                       help='Build volume Z dimension in mm (default: 15.0)')

    # Part volume parameters
    parser.add_argument('--part-x', type=float, default=5.0,
                       help='Part X dimension in mm (default: 5.0)')
    parser.add_argument('--part-y', type=float, default=5.0,
                       help='Part Y dimension in mm (default: 5.0)')
    parser.add_argument('--part-z', type=float, default=5.0,
                       help='Part Z dimension in mm (default: 5.0)')

    # Process parameters
    parser.add_argument('--voxel-size', type=float, default=200.0,
                       help='Voxel size in micrometers (default: 200.0)')
    parser.add_argument('--delta-t', type=float, default=200.0,
                       help='Time step in milliseconds (default: 200.0)')
    parser.add_argument('--scan-speed', type=float, default=3.0,
                       help='Scan speed in mm/s (default: 3.0)')
    parser.add_argument('--laser-power', type=float, default=600.0,
                       help='Laser power in W (default: 600.0)')
    parser.add_argument('--powder-feed', type=float, default=2.0,
                       help='Powder feed rate in g/min (default: 2.0)')

    # Output control
    parser.add_argument('--exp-label', type=str, default="unlabeled",
                       help='Directory name to save the output (default: unlabeled)')

    args = parser.parse_args()

    # Get default callbacks
    callbacks = get_default_callbacks()

    # Create runner with human units
    runner = SimulationRunner.from_human_units(
        build_volume_mm=(args.build_x, args.build_y, args.build_z),
        part_volume_mm=(args.part_x, args.part_y, args.part_z),
        voxel_size_um=args.voxel_size,
        delta_t_ms=args.delta_t,
        scan_speed_mm_s=args.scan_speed,
        laser_power_W=args.laser_power,
        powder_feed_g_min=args.powder_feed,
        experiment_label=args.exp_label,
        callbacks=callbacks
    )

    # Run simulation
    runner.run()

    print(f"\nSimulation completed successfully!")
    print(f"Results saved to: {runner.simulation.output_dir}")