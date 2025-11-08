from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from callbacks._base_callbacks import SimulationEvent
from callbacks._callback_manager import CallbackManager
from callbacks.completion_callbacks import SimulationComplete
from configuration.process_parameters import set_params
from geometry.clad_profile_manager import CladProfileManager
from geometry.clad_dimensions import YuzeHuangCladDimensions
from scan_path.scan_path_manager import ScanPathManager
from voxel.activated_volume import ActivatedVolume
from voxel.temperature_volume import TrackTemperature
from powder.powder_stream import YuzeHuangPowderStream
from powder.powder_stream_new import VoxelPowderStream
from thermal.temperature_change import EagarTsaiTemperature

import warnings

# Suppress only the specific legend warning
warnings.filterwarnings('ignore', message='No artists with labels found to put in legend.*')


class ProgressTracker:
    """Tracks simulation progress and detects transitions."""

    def __init__(self, track_length: float):
        self.track_length = track_length
        self.reset()

    def reset(self):
        """Reset all tracking state to initial values."""
        self.step_count = 0

        # Add cumulative counters like the old system
        self.count_tracks = 0  # Total tracks completed
        self.count_layers = 0  # Current layer count
        self.max_height_reached = 0.0  # Maximum height reached

        # Current state
        self.current_layer = None
        self.current_track = None
        self._raw_track_progress = 0.0
        self.current_y = None
        self.init_next_track = True

        # Previous state (for detecting transitions)
        self._prev_layer = None
        self._prev_track = None
        self._prev_progress = 0.0

        # Track whether we're in a step
        self._in_step = False

        # Snapshot of properties from last completed step
        self._last_step_state = {
            'track_just_started': False,
            'track_just_completed': False,
            'layer_just_started': False
        }

    @property
    def current_track_progress(self) -> float:
        """Returns clamped track progress (0 to 1)."""
        return min(1.0, self._raw_track_progress)

    @property
    def remaining_track_length(self) -> float:
        """Returns remaining length in current track."""
        return self.track_length * (1.0 - self.current_track_progress)

    @property
    def track_just_started(self) -> bool:
        """Returns True if a new track started."""
        if not self._in_step:
            # Between steps: return saved state
            return self._last_step_state['track_just_started']

        # During step: calculate from state
        return (
                (self._prev_track is None and self.current_track is not None) or
                (self._prev_track is not None and self.current_track != self._prev_track) or
                self._prev_progress >= 1.0 - 1e-9  # Account for floating point errors
        )

    @property
    def track_just_completed(self) -> bool:
        """Returns True if the current track just completed."""
        if not self._in_step:
            # Between steps: return saved state
            return self._last_step_state['track_just_completed']

        # During step: check current progress
        return self._raw_track_progress >= 1.0 - 1e-9  # Account for floating point errors

    @property
    def layer_just_started(self) -> bool:
        """Returns True if a new layer started."""
        if not self._in_step:
            # Between steps: return saved state
            return self._last_step_state['layer_just_started']

        # During step: calculate from state
        return (
                (self._prev_layer is None and self.current_layer is not None) or
                (self._prev_layer is not None and
                 self.current_layer is not None and
                 self.current_layer != self._prev_layer)
        )

    def begin_step(self, scan_state: tuple):
        """
        Call at beginning of step to update state.

        Args:
            scan_state: (layer_idx, track_idx, reverse_direction, reverse_track_idx)
        """
        layer_idx, track_idx, _, _ = scan_state

        # Save previous state before updating
        self._prev_layer = self.current_layer
        self._prev_track = self.current_track
        self._prev_progress = self._raw_track_progress

        # Update current state
        self.current_layer = layer_idx
        self.current_track = track_idx

        # Mark that we're in a step
        self._in_step = True

        # NOTE: init_next_track should NOT be cleared here!
        # It should be cleared in the main simulation after calling scan_manager.next_track()

    def update_track_progress(self, delta_y: float, new_y: float):
        """
        Update track progress during a step.

        Args:
            delta_y: Distance traveled in this step
            new_y: New Y position
        """
        self._raw_track_progress += delta_y / self.track_length
        self.current_y = new_y

    def end_step(self):
        """Call at end of step to save state and reset if needed."""
        # Save current property values before any changes
        self._last_step_state = {
            'track_just_started': self.track_just_started,
            'track_just_completed': self.track_just_completed,
            'layer_just_started': self.layer_just_started
        }

        # If track is complete, reset progress for next track
        if self._raw_track_progress >= 1.0 - 1e-9:  # Account for floating point errors
            self._raw_track_progress = 0.0
            self.current_y = None
            self.init_next_track = True
            self.count_tracks += 1  # Increment track counter

        # Update layer count
        if self.current_layer is not None:
            self.count_layers = self.current_layer  # Track current layer index

        self.step_count += 1

        # Mark that we're no longer in a step
        self._in_step = False

    def update_max_height(self, height: float):
        """Update the maximum height reached."""
        self.max_height_reached = max(self.max_height_reached, height)

    def get_transition_summary(self) -> dict:
        """
        Get a summary of current transitions for debugging.

        Returns:
            Dictionary with all transition states
        """
        return {
            'step_count': self.step_count,
            'count_tracks': self.count_tracks,  # Add cumulative track count
            'count_layers': self.count_layers,  # Add layer count
            'max_height_reached': self.max_height_reached,  # Add max height
            'current_layer': self.current_layer,
            'current_track': self.current_track,
            'current_progress': self.current_track_progress,
            'track_just_started': self.track_just_started,
            'track_just_completed': self.track_just_completed,
            'layer_just_started': self.layer_just_started,
            'init_next_track': self.init_next_track,
            'in_step': self._in_step,
        }


class MultiTrackMultiLayerSimulation:
    """
    Simulates multi-track, multi-layer additive manufacturing of cubes.

    This class handles the core simulation logic by combining:
    - Clad profile management (track geometry)
    - Scan path management (build sequence)
    - Volume activation tracking (solidified material)
    - Temperature tracking (thermal history)
    - Physical track dimensions (based on Huang et al. 2019 model)
    """

    def __init__(
            self,
            config: Dict[str, Any],
            delta_t: float = 0.1,
            powder_concentration_func=None,
            callbacks: Optional[List[Any]] = None,
            output_dir='_simulation_output',
    ):
        """
        Initialize the simulation with configuration parameters.

        Args:
            config: Dictionary containing:
                - Process parameters (laser power, scan speed, etc.)
                - Geometric parameters (track dimensions, hatch spacing)
                - Build volume dimensions
                - Thermal properties
                - Discretization settings (voxel size)
                - Material properties (density, viscosity, surface tension)
                - Powder delivery parameters (feed rate, nozzle geometry)
        """

        self.callback_manager = CallbackManager(callbacks)

        self.config = config
        self.delta_t = delta_t
        self.powder_concentration_func = powder_concentration_func

        self.callback_manager(self, SimulationEvent.CONFIG_LOADED)


        self.y_offset = self.config.get('y_offset', 0.0)
        #self.track_length = self.config['track_length']
        self.substrate_height = self.config.get('substrate_height', 0.0)

        # Core simulation state
        #self.current_track_progress = 0.0
        #self.current_y = None
        #self.init_next_track = True

        self.output_dir = output_dir

        self.step_context = None
        self.init_params = None

        self.progress_tracker = ProgressTracker(self.config['track_length'])

        self.scan_manager = ScanPathManager(
            num_tracks=self.config['num_tracks'],
            track_length=self.config['track_length'],
            hatch_space=self.config['hatch_spacing'],
            turnaround_time=self.config.get('turnaround_time', 0.0),
            bidirectional_tracks=self.config.get('bidirectional_tracks', True),
            bidirectional_layers=self.config.get('bidirectional_layers', True),
            switch_scan_direction_between_layers=self.config.get('switch_scan_direction_between_layers', True),
        )

        self.temperature_field = EagarTsaiTemperature()

        self.temperature_tracker = TrackTemperature(
            shape=self.config['volume_shape'],
            voxel_size=self.config['voxel_size'],
            ambient_temp=self.config.get('ambient_temp', 300.0),
            substrate_height=self.config.get('substrate_height', 0)
        )

        if powder_concentration_func is None:
            powder_concentration_func = YuzeHuangPowderStream.powder_concentration


        self.clad_dimensions = YuzeHuangCladDimensions(
            powder_concentration_func=powder_concentration_func,
            resolution=self.config.get('powder_distribution_integral_resolution', 200)
        )

        self.clad_manager = CladProfileManager(
            hatch_distance=self.config['hatch_spacing'],
            num_tracks=self.config['num_tracks'],
            offset=self.config.get('x_offset', 0.0),
            substrate_height=self.config.get('substrate_height', 0.0),
        )

        self.volume_tracker = ActivatedVolume(
            shape=self.config['volume_shape'],
            voxel_size=self.config['voxel_size'],
            substrate_height=self.config.get('substrate_height', 0)
        )

        self.callback_manager(self, SimulationEvent.INIT)

        # List of instances that may be reset
        self.resettable_instances = [
            self.clad_manager,
            self.scan_manager,
            self.volume_tracker,
            self.temperature_tracker,
            self.temperature_field,
            self.progress_tracker,
        ]

        self.reset()

    # For backwards compatibility:
    @property
    def track_length(self):
        return self.progress_tracker.track_length

    @property
    def current_track_progress(self):
        return self.progress_tracker.current_track_progress

    @property
    def current_y(self):
        return self.progress_tracker.current_y

    @property
    def init_next_track(self):
            return self.progress_tracker.init_next_track

    def reset(self) -> Dict[str, Any]:
        """
        Reset the simulation to initial state.

        Returns:
            Dictionary containing initial simulation state
        """
        #self.current_track_progress = 0.0
        #self.current_y = None
        #self.init_next_track = True

        self.step_context = {}

        for instance in self.resettable_instances:
            if hasattr(instance, 'reset'):
                instance.reset()

        self.init_params = None

    def complete(self):
        """Mark the simulation as complete and trigger callbacks."""
        self.callback_manager(self, SimulationEvent.COMPLETE)

    def step(self, params):
        """
        Execute one simulation step with given process parameters.

        Args:
            params: Dictionary containing process parameters
            initial: Boolean indicating if this is the first step

        Returns:
            Tuple containing:
            - Dictionary of melt pool dimensions
            - Dictionary of clad dimensions
            - Boolean indicating if simulation needs to transition to next track
            - Actual time step used (may be shortened for track completion)
        """
        try:
            return self._step(params)
        except KeyboardInterrupt:
            # Call complete() for graceful shutdown on user interrupt
            print("\nKeyboard interrupt detected - Attempting to shut down gracefully...")
            self.complete()
            raise
        except Exception as e:
            self.callback_manager(self, SimulationEvent.ERROR, error=e)
            raise e

    def _step(self, params):

        if self.init_params is None:
            self.init_params = params.copy()

        # Get current track state
        if self.progress_tracker.init_next_track:
            self.scan_manager.next_track(params=params)
            #self.progress_tracker.init_next_track = False

        layer_idx, track_idx, reverse_direction, reverse_track_idx = self.scan_manager.current_state

        self.progress_tracker.begin_step(self.scan_manager.current_state)

        # Trigger events at START of step/layer/track
        if self.progress_tracker.layer_just_started:
            self.callback_manager(self, SimulationEvent.LAYER_START)
        if self.progress_tracker.track_just_started:
            self.callback_manager(self, SimulationEvent.TRACK_START)
        self.callback_manager(self, SimulationEvent.STEP_START)

        # Check if we'll exceed track length and adjust time step if needed
        delta_t = min(self.delta_t, self.progress_tracker.remaining_track_length / params['scan_speed'])
        delta_y = params['scan_speed'] * delta_t

        if self.progress_tracker.init_next_track and delta_y >= self.progress_tracker.track_length:
            raise Exception(
                "Initial time step is too large: Reaching the end of the track in one step. "
                "\nCheck time step length, scan speed and track length."
            )

        # Calculate positions
        x = self.clad_manager.get_x_position(track_index=track_idx)
        # if the track is reversed, start from the end of the track (when current_y is None; init_next_track)
        y_0 = self.current_y or self.y_offset + float(reverse_direction) * self.track_length
        # if the track is reversed, move in the opposite direction
        y_1 = y_0 + (-delta_y if reverse_direction else delta_y)

        z = self.clad_manager.get_z_position(
            layer_index=layer_idx,
            track_index=track_idx,
            y_pos=y_1
        )
        expected_z = self.substrate_height + layer_idx * self.config['layer_spacing']

        # Recalibrate height-related parameters based on deviation from expected height
        height_deviation = z - expected_z
        params.update({
            'beam_waist_position': self.init_params['beam_waist_position'] - height_deviation,
            'nozzle_height': self.init_params['nozzle_height'] - height_deviation,
        })
        params.update(set_params(**params))

        # Apply diffusion to temperature field
        self.temperature_tracker.apply_diffusion(
            activation_mask=np.ones_like(self.volume_tracker.activated, dtype=bool),
            sigma=np.sqrt(2 * params['thermal_diffusivity'] * delta_t)
        )

        activation_mask_with_melt_pool = np.logical_or(
            self.volume_tracker.activated,
            self.temperature_tracker.temperature >= params['melting_temp']
        )

        # Reset deactivated voxels
        self.temperature_tracker.reset_deactivated(activation_mask_with_melt_pool)

        # Apply heat source at current position
        self.temperature_tracker.apply_heat_source(
            get_temp=self.temperature_field.delta_temperature,
            t=delta_t,
            params=params,
            start_position=(x, y_0, z),
            movement_angle= 1/2 * np.pi if reverse_direction else 3/2 * np.pi,
        )

        # Get melt pool dimensions
        melt_pool_dims = self.temperature_tracker.get_melt_pool_dimensions(params=params)

        if melt_pool_dims['depth'] <= 0.0 or melt_pool_dims['width'] <= 0.0 or melt_pool_dims['length'] <= 0.0:
            raise ValueError(f"Melt pool dimensions are invalid: {melt_pool_dims}")

        # Calculate clad dimensions
        clad_width, clad_height, wetting_angle = self.clad_dimensions.compute_final_geometry(
            pool_width=melt_pool_dims['width'],
            pool_length=melt_pool_dims['length'],
            params=params
        )

        clad_dims = {
            'width': clad_width,
            'height': clad_height,
            'wetting_angle': wetting_angle
        }

        # Check for large dimensional discrepancies
        if not (melt_pool_dims['width'] == 0 or clad_width == 0):
            width_ratio = max(melt_pool_dims['width'], clad_width) / min(melt_pool_dims['width'], clad_width)
        else:
            width_ratio = 0
        if not (melt_pool_dims['depth'] == 0 or clad_height == 0):
            height_ratio = max(melt_pool_dims['depth'], clad_height) / min(melt_pool_dims['depth'], clad_height)
        else:
            height_ratio = 0

        if width_ratio > 50 or height_ratio > 50:
            raise ValueError(
                f"Large discrepancy between melt pool and clad dimensions detected:\n"
                f"Melt pool width: {melt_pool_dims['width'] * 1000:.2f}mm, Clad width: {clad_width * 1000:.2f}mm (ratio: {width_ratio:.2f})\n"
                f"Melt pool height: {melt_pool_dims['depth'] * 1000:.2f}mm, Clad height: {clad_height * 1000:.2f}mm (ratio: {height_ratio:.2f})"
            )

        if self.init_next_track:
            # Add the first profile (with same clad dimensions as the second one)
            # Simplification for numerical stability, as moving heat source must travel
            profile_0 = self.clad_manager.add_profile(
                layer_index=layer_idx,
                track_index=track_idx,
                y_pos=y_0,
                width=clad_width,
                height=clad_height,
            )
        else:
            # Get previous profile at y_0
            profile_0 = self.clad_manager.get_profile_function(
                layer_index=layer_idx,
                track_index=track_idx,
                y_pos=y_0
            )
        #print(profile_0)
        #print('layer_idx:', layer_idx, 'track_idx:', track_idx, 'y_0:', y_0, 'y_1:', y_1)
        #print('reverse_direction:', reverse_direction, 'reverse_track_idx:', reverse_track_idx)
        # Create track profile at new position
        profile_1 = self.clad_manager.add_profile(
            layer_index=layer_idx,
            track_index=track_idx,
            y_pos=y_1,
            width=clad_width,
            height=clad_height,
        )
        #print('profile height:', profile_1(self.clad_manager.get_x_position(track_idx)))
        #print('profile baseline:', profile_1.baseline)

        # Add track section
        self.volume_tracker.add_track_section(
            start_profile=profile_0 if not reverse_direction else profile_1,
            end_profile=profile_1 if not reverse_direction else profile_0,
            length_between=delta_y,
            y_position=y_0 if not reverse_direction else y_1
        )

        # Update progress
        self.progress_tracker.update_track_progress(delta_y, y_1)
        height_at_center = profile_1(x)
        self.progress_tracker.update_max_height(height_at_center)

        self.step_context = {
            'position': {
                'x': x,
                'y': y_1,
                'z': z,
                'expected_z': expected_z,
            },
            'voxel': {
                'x': int(x / self.config['voxel_size'][0]),
                'y': int(y_1 / self.config['voxel_size'][1]),
                'z': int(z / self.config['voxel_size'][2])
            },
            'build': {
                'layer': layer_idx,
                'track': track_idx,
                'reverse_direction': reverse_direction,
                'reverse_track': reverse_track_idx,
                'time_step': delta_t,
                'track_progress': self.progress_tracker.current_track_progress
            },
            'melt_pool': melt_pool_dims,  # Already a dict with width, length, depth, etc.
            'clad': clad_dims,  # Already a dict with width, height, wetting_angle
            'profile': {
                'object': profile_1,  # The actual profile object if needed
                'baseline': profile_1.baseline,
                'max_z': profile_1.get_max_f(),
                'height_at_center': height_at_center,
                'width': clad_width,
            },
            'params': params,  # Store current parameters for this step
        }

        # Trigger events at END of step/layer/track
        self.callback_manager(self, SimulationEvent.STEP_COMPLETE)
        if self.progress_tracker.track_just_completed:
            self.callback_manager(self, SimulationEvent.TRACK_COMPLETE)

        # Clear the flag at the END of the step instead
        if self.progress_tracker.init_next_track:
            self.progress_tracker.init_next_track = False

        self.progress_tracker.end_step()

        return melt_pool_dims, clad_dims, delta_t, y_1, self.scan_manager.current_state, profile_1

    @property
    def temp_slices(self):
        """
        Compute temperature slices on demand at current position.
        """

        if 'temp_slices' in self.step_context:
            return self.step_context['temp_slices']

        # Compute slices and cache in step_context
        voxel = self.step_context['voxel']

        # Ensure indices are within bounds
        x_idx = min(voxel['x'], self.temperature_tracker.temperature.shape[0] - 1)
        y_idx = min(voxel['y'], self.temperature_tracker.temperature.shape[1] - 1)
        z_idx = min(voxel['z'], self.temperature_tracker.temperature.shape[2] - 1)

        self.step_context['temp_slices'] = {
            'xy': self.temperature_tracker.temperature[:, :, z_idx].T,
            'xz': self.temperature_tracker.temperature[:, y_idx, :].T,
            'yz': self.temperature_tracker.temperature[x_idx, :, :].T,
        }

        return self.step_context['temp_slices']


if __name__ == "__main__":

    import matplotlib
    matplotlib.use('TkAgg')  # Force external window

    from simulate import SimulationRunner
    from callbacks.completion_callbacks import HeightCompletionCallback
    from callbacks.step_data_collector import StepDataCollector
    from callbacks.live_plotter_callback import AdvancedLivePlotter
    from callbacks.callback_collection import ProgressPrinter
    from callbacks.track_calibration_callback import TrackCalibrationCallback

    # Create callbacks
    callbacks = [
        HeightCompletionCallback(),  # Stop at target height
        StepDataCollector(save_path=None),  # Save data
        AdvancedLivePlotter(interval=1),  # Live visualization every step
        ProgressPrinter(),  # Console progress output
        #TrackCalibrationCallback(),
    ]

    # Create and run simulation
    runner = SimulationRunner.from_human_units(
        build_volume_mm=(20, 20, 15),
        part_volume_mm=(5, 5, 1.0),
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
