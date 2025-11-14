"""
Step data collection callback for DED simulation.
Mirrors the Simulator's flexible field tracking system.
"""

import warnings
from pathlib import Path
from typing import Union, List, Set, Optional, Dict, Any
import pandas as pd

from callbacks._base_callbacks import BaseCallback, SimulationEvent


class StepDataCollector(BaseCallback):
    """
    Collects step data during simulation and saves as CSV at completion.

    Flexible field tracking system allows specifying exactly what data to collect.

    Usage:
        # Track everything (default)
        collector = StepDataCollector()

        # Track specific fields only
        collector = StepDataCollector(tracked_fields=['position', 'melt_pool', 'clad'])

        # Minimal tracking (just step number)
        collector = StepDataCollector(tracked_fields=[])
    """

    def __init__(
        self,
        tracked_fields: Optional[List[str]] = None,
        save_path: str = "simulation_data.csv",
        **kwargs
    ):
        """
        Initialize the data collector.

        Args:
            tracked_fields: None (track everything), empty list (minimal tracking),
                          or list of field names to track. Available fields:
                          ['position', 'voxel', 'build', 'melt_pool', 'clad',
                           'profile', 'temperature']
            save_path: Filename for CSV output (relative to simulation output_dir)
            **kwargs: Additional configuration passed to BaseCallback
        """
        super().__init__(
            events=[SimulationEvent.STEP_COMPLETE, SimulationEvent.COMPLETE],
            **kwargs
        )

        self.tracked_fields = tracked_fields
        self.save_path = save_path
        self.step_data = []
        self._current_step_dict = {}

        # Prepare tracking methods
        self._prepare_tracking_methods()

    @property
    def available_fields(self) -> List[str]:
        """Get list of available field names that can be tracked."""
        return [name[9:] for name in dir(self)
                if name.startswith('_collect_') and callable(getattr(self, name))]

    @property
    def current_step_dict(self) -> Dict[str, Any]:
        """Get the most recently collected step dictionary."""
        return self._current_step_dict.copy()

    def _prepare_tracking_methods(self):
        """
        Determine which fields to track based on configuration.
        Raises ValueError if requested fields don't have collection methods.
        """
        if self.tracked_fields is None:
            # Track everything - use available_fields property
            self.fields_to_track = self.available_fields

        elif self.tracked_fields == []:
            # Track nothing except step count
            self.fields_to_track = []

        else:
            # Track only specified fields that have corresponding methods
            self.fields_to_track = []
            missing_methods = []

            for field in self.tracked_fields:
                method_name = f'_collect_{field}'
                if hasattr(self, method_name) and callable(getattr(self, method_name)):
                    self.fields_to_track.append(field)
                else:
                    missing_methods.append(field)

            if missing_methods:
                raise ValueError(
                    f"No collection methods found for fields: {missing_methods}\n"
                    f"Available fields: {self.available_fields}"
                )

    def _execute(self, context: dict) -> None:
        """
        Execute the callback - collect data or save to CSV.

        Args:
            context: Simulation context containing 'event' and 'simulation'
        """
        sim = context['simulation']

        if context['event'] == SimulationEvent.STEP_COMPLETE:
            # Build step dict from simulation's step_context
            step_dict = {'step': sim.progress_tracker.step_count}

            # Collect specified fields
            for field in self.fields_to_track:
                method = getattr(self, f'_collect_{field}')
                # Pass both step_context and sim for methods that need temperature field access
                step_dict.update(method(sim.step_context, sim))

            self._current_step_dict = step_dict
            self.step_data.append(step_dict)

        elif context['event'] == SimulationEvent.COMPLETE and self.save_path is not None:
            # Save collected data to CSV
            if self.step_data:
                df = pd.DataFrame(self.step_data)
                filepath = self.resolve_path(context, self.save_path)

                # Ensure parent directory exists
                filepath.parent.mkdir(parents=True, exist_ok=True)

                # Save CSV
                df.to_csv(filepath, index=False)
                print(f"Saved {len(self.step_data)} steps to {filepath}")
            else:
                print("No step data collected - nothing to save")

    # =========================================================================
    # Collection methods for each field type
    # Each method receives (step_context, simulation) and returns a dict
    # =========================================================================

    def _collect_position(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, float]:
        """Collect position data (x, y, z coordinates)."""
        pos = step_context['position']
        return {
            'position.x': pos['x'],
            'position.y': pos['y'],
            'position.z': pos['z'],
        }

    def _collect_voxel(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, int]:
        """Collect voxel indices (discretized position)."""
        vox = step_context['voxel']
        return {
            'voxel.x': vox['x'],
            'voxel.y': vox['y'],
            'voxel.z': vox['z'],
        }

    def _collect_build(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, Any]:
        """Collect build progress data (layer, track, direction)."""
        #print(step_context)
        build = step_context['build']
        return {
            'build.layer': build['layer'],
            'build.track': build['track'],
            'build.reverse_direction': build['reverse_direction'],
            'build.reverse_track': build['reverse_track'],
            'build.time_step': build['time_step'],
            'build.track_progress': build.get('track_progress', 0),
        }

    def _collect_melt_pool(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, float]:
        """Collect melt pool dimensions (width, length, depth)."""
        mp = step_context['melt_pool']
        return {
            'melt_pool.width': mp['width'],
            'melt_pool.length': mp['length'],
            'melt_pool.depth': mp['depth'],
        }

    def _collect_clad(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, float]:
        """Collect clad dimensions (width, height, wetting angle)."""
        clad = step_context['clad']
        return {
            'clad.width': clad['width'],
            'clad.height': clad['height'],
            'clad.wetting_angle': clad['wetting_angle'],
        }

    def _collect_profile(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, float]:
        """Collect profile parameters (geometry of deposited track)."""
        prof = step_context['profile']
        profile_obj = prof.get('object')

        if profile_obj:
            return {
                'profile.baseline': prof['baseline'],
                'profile.max_z': prof['max_z'],
                'profile.height_at_center': prof['height_at_center'],
                'profile.width': prof['width'],
                'profile.start_x': profile_obj.start_x,
                'profile.end_x': profile_obj.end_x,
                'profile.a': profile_obj.a,
                'profile.b': profile_obj.b,
                'profile.c': profile_obj.c,
                'profile.track_center': profile_obj.offset,
                'profile.required_area': profile_obj.required_area,
            }
        return {}

    def _collect_temperature(self, step_context: Dict[str, Any], sim: Any) -> Dict[str, Any]:
        """
        Collect temperature data including depth profile.

        Samples temperature at current position and at multiple depths below.
        """
        mp = step_context.get('melt_pool', {})
        voxel = step_context.get('voxel', {})

        temp_data = {}

        # Basic temperature data from melt pool
        if 'max_temp' in mp:
            temp_data['temperature.max_temp'] = mp['max_temp']
        if 'voxel_center' in mp:
            temp_data['temperature.voxel_center'] = mp['voxel_center']

            # Temperature at the voxel center
            if hasattr(sim, 'temperature_tracker'):
                temp_field = sim.temperature_tracker.temperature
                center_idx = mp['voxel_center']
                temp_data['temperature.test_center_temp'] = temp_field[center_idx]

        # Temperature depth profile at current position
        if voxel and hasattr(sim, 'temperature_tracker'):
            temp_field = sim.temperature_tracker.temperature
            x_idx = voxel['x']
            y_idx = voxel['y']
            z_idx = voxel['z']

            # Sample temperature at different depths below current position
            # Samples at z, z-5, z-10, z-15, z-20 voxels
            for i in range(5):
                depth_offset = i * 5
                z_sample = z_idx - depth_offset

                if z_sample >= 0:
                    temp_data[f'temperature.voxel.z.-{depth_offset}'] = (
                        temp_field[x_idx, y_idx, z_sample]
                    )
                else:
                    temp_data[f'temperature.voxel.z.-{depth_offset}'] = None

        return temp_data