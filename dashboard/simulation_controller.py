import sys
import os
import time
import threading
import numpy as np
from pathlib import Path
import pandas as pd
from collections import deque
import importlib
import traceback

# Add parent directory to path so we can import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simulation modules - updated imports
from core.multi_track_multi_layer import MultiTrackMultiLayerSimulation
from configuration.process_parameters import set_params
from configuration.simulation_config import SimulationConfig


class SimulationController:
    """Controller for the DED simulation for dashboard integration"""

    def __init__(self):
        """Initialize the simulation controller"""
        self.simulation = None
        self.config = None
        self.params = None
        self.simulation_generator = None
        self.simulation_thread = None
        self.running = False
        self.paused = True
        self.simulation_completed = False
        self.simulation_error = None
        self.error_traceback = None

        # Simulation state tracking
        self.current_step = 0
        self.current_step_data = None
        self.current_thermal_data = None
        self.current_build_data = None
        self.current_build_mesh = None
        self.current_mesh_file = None
        self.mesh_files = deque(maxlen=20)  # Store paths to last 20 mesh files
        self.mesh_history = deque(maxlen=20)  # Store last 20 mesh data
        self.temperature_history = deque(maxlen=100)  # Store last 100 max temperatures
        self.parameter_history = deque(maxlen=100)  # Store last 100 parameter sets

        # New tracking for clad metrics and wetting angle
        self.clad_height_history = deque(maxlen=100)  # Store last 100 clad heights
        self.melt_pool_depth_history = deque(maxlen=100)  # Store last 100 melt pool depths
        self.wetting_angle_history = deque(maxlen=100)  # Store last 100 wetting angles

        # Initialize with default parameters
        self.reset_simulation()

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        # Stop any running simulation
        if self.running:
            self.stop_simulation()

        # Clear error and completion flags
        self.simulation_completed = False
        self.simulation_error = None
        self.error_traceback = None

        # Set default parameters
        self.params = set_params(
            laser_power=600.0,
            scan_speed=0.003,  # 3 mm/s
            powder_feed_rate=2.0 / (60 * 1000)  # 2 g/min to kg/s
        )

        # Create simulation configuration
        self.config = SimulationConfig(
            build_volume_size=(0.02, 0.02, 0.015),  # 20x20x15 mm
            voxel_size=200e-6,  # 200 µm
            part_width=0.005,  # 5 mm
            part_length=0.005,  # 5 mm
            part_height=0.005,  # 5 mm
            hatch_spacing=0.0007,  # 700 µm track spacing
            layer_spacing=0.00035,  # 350 µm layer spacing (added missing parameter)
            substrate_height=0.005,  # 5 mm substrate
            bidirectional_tracks=True,
            bidirectional_layers=True,
            switch_scan_direction_between_layers=True,
            turnaround_time=0.0,
        )

        # Create simulation instance - updated to use new API
        self.simulation = MultiTrackMultiLayerSimulation(
            config=self.config.get_simulation_config(),
            delta_t=0.2,  # 200 ms
            callbacks=None,  # No callbacks for dashboard
            output_dir=None,  # No output directory needed
        )
        self.simulation.reset()

        # Reset state tracking
        self.current_step = 0
        self.current_step_data = None
        self.current_thermal_data = None
        self.current_build_data = None
        self.current_build_mesh = None
        self.temperature_history.clear()
        self.parameter_history.clear()
        self.mesh_history.clear()

        # Clear new tracking data
        self.clad_height_history.clear()
        self.melt_pool_depth_history.clear()
        self.wetting_angle_history.clear()

        self.parameter_history.append(self.params.copy())

        # Initialize simulation generator
        self._initialize_generator()

        # Signal that reset is complete
        self.running = False
        self.paused = True

        print("Simulation reset complete")

    def _initialize_generator(self):
        """Initialize the simulation - no longer uses generator"""
        # This method is no longer needed with the new API
        # but kept for backward compatibility
        pass

    def update_parameters(self, new_params):
        """Update simulation parameters"""
        # Update parameters with new values
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value

        # Store in parameter history
        self.parameter_history.append(self.params.copy())

        print(f"Parameters updated: {new_params}")

    def start_simulation(self):
        """Start the simulation in a separate thread"""
        if not self.running:
            # Clear any previous error states
            self.simulation_error = None
            self.error_traceback = None
            self.simulation_completed = False

            self.running = True
            self.paused = False

            # Start simulation in a separate thread
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()

            print("Simulation started")

    def _run_simulation(self):
        """Run the simulation loop"""
        while self.running:
            if not self.paused:
                try:
                    result = self.step_simulation()
                    if result is None:  # This likely means the simulation is done
                        self.simulation_completed = True
                        self.running = False
                        self.paused = True
                        break
                    # Small delay to avoid thread hogging
                    time.sleep(0.1)
                except StopIteration:
                    print("Simulation completed successfully")
                    self.simulation_completed = True
                    self.running = False
                    self.paused = True
                    break
                except Exception as e:
                    print(f"Simulation error: {e}")
                    self.simulation_error = e
                    self.error_traceback = traceback.format_exc()
                    self.running = False
                    self.paused = True
                    break
            else:
                # If paused, just wait but don't terminate the thread
                time.sleep(0.1)

    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        print("Simulation paused")

    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.paused = True

        # Wait for thread to finish if it exists
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)

        print("Simulation stopped")

    def step_simulation(self):
        """Advance the simulation by one step"""
        try:
            # Run one step of the simulation - updated to use new API
            if self.simulation:
                try:
                    # Call step directly - new API
                    self.simulation.step(self.params)

                    # Extract data from step_context
                    if not self.simulation.step_context:
                        print("ERROR: step_context is empty")
                        raise ValueError("step_context is empty after step")

                    ctx = self.simulation.step_context

                    # Update current state from step_context
                    self.current_step = self.simulation.progress_tracker.step_count

                    # Build step_dict from step_context
                    self.current_step_data = {
                        'step': self.current_step,

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

                        # Temperature data
                        'temperature.max_temp': np.max(self.simulation.temperature_tracker.temperature),

                        # Max height
                        'build.max_height': self.simulation.progress_tracker.max_height_reached,
                    }

                    # Get thermal data from simulation
                    self.current_thermal_data = self.simulation.temp_slices

                    # Create build_data dict
                    self.current_build_data = {
                        'x': ctx['position']['x'],
                        'y': ctx['position']['y'],
                        'z': ctx['position']['z'],
                        'layer': ctx['build']['layer'],
                        'track': ctx['build']['track'],
                        'clad_width': ctx['clad']['width'],
                        'clad_height': ctx['clad']['height']
                    }

                    # Update build mesh data directly from clad_manager
                    if hasattr(self, 'simulation') and self.simulation is not None:
                        try:
                            self.current_build_mesh = self.get_current_mesh_data()
                        except Exception as mesh_error:
                            print(f"Error getting mesh data: {mesh_error}")
                            import traceback
                            traceback.print_exc()
                            self.current_build_mesh = None

                    # Update tracking histories
                    if self.current_step_data and 'temperature.max_temp' in self.current_step_data:
                        self.temperature_history.append(self.current_step_data['temperature.max_temp'])

                    # Update new metric histories
                    if self.current_step_data:
                        # Get clad height (convert to mm)
                        clad_height = self.current_step_data.get('clad.height', 0) * 1000  # m to mm
                        self.clad_height_history.append(clad_height)

                        # Get melt pool depth (convert to mm)
                        melt_pool_depth = self.current_step_data.get('melt_pool.depth', 0) * 1000  # m to mm
                        self.melt_pool_depth_history.append(melt_pool_depth)

                        # Get wetting angle (already in radians or degrees)
                        wetting_angle = self.current_step_data.get('clad.wetting_angle', 0)
                        self.wetting_angle_history.append(wetting_angle)

                    # Return updated parameters
                    return self.params

                except Exception as e:
                    print(f"Error during simulation step execution: {e}")
                    self.simulation_error = e
                    import traceback
                    self.error_traceback = traceback.format_exc()
                    traceback.print_exc()
                    raise
            else:
                print("Warning: simulation is None")
                raise ValueError("Simulation is not initialized")
        except Exception as e:
            print(f"Error in step_simulation: {e}")
            self.simulation_error = e
            import traceback
            self.error_traceback = traceback.format_exc()
            traceback.print_exc()
            raise

    def get_current_metrics(self):
        """Get current simulation metrics for display"""
        if not self.current_step_data:
            return {}

        # Extract key metrics
        metrics = {
            'step': self.current_step,
            'layer': self.current_step_data.get('build.layer', 0),
            'track': self.current_step_data.get('build.track', 0),
            'max_height': self.current_step_data.get('build.max_height', 0) * 1000,  # Convert to mm
            'melt_pool_width': self.current_step_data.get('melt_pool.width', 0) * 1000,  # Convert to mm
            'melt_pool_length': self.current_step_data.get('melt_pool.length', 0) * 1000,  # Convert to mm
            'melt_pool_depth': self.current_step_data.get('melt_pool.depth', 0) * 1000,  # Convert to mm
            'clad_width': self.current_step_data.get('clad.width', 0) * 1000,  # Convert to mm
            'clad_height': self.current_step_data.get('clad.height', 0) * 1000,  # Convert to mm
            'wetting_angle': self.current_step_data.get('clad.wetting_angle', 0),  # Radians or degrees
            'max_temp': self.current_step_data.get('temperature.max_temp', 0),  # Kelvin
            'position_x': self.current_step_data.get('position.x', 0) * 1000,  # Convert to mm
            'position_y': self.current_step_data.get('position.y', 0) * 1000,  # Convert to mm
            'position_z': self.current_step_data.get('position.z', 0) * 1000,  # Convert to mm
        }

        return metrics

    def get_current_thermal_data(self):
        """Get current thermal data for visualization"""
        return self.current_thermal_data

    def get_current_build_data(self):
        """Get current build data for visualization"""
        return self.current_build_data

    def get_temperature_history(self):
        """Get temperature history for plotting"""
        return list(self.temperature_history)

    def get_clad_height_history(self):
        """Get clad height history for plotting"""
        return list(self.clad_height_history)

    def get_melt_pool_depth_history(self):
        """Get melt pool depth history for plotting"""
        return list(self.melt_pool_depth_history)

    def get_wetting_angle_history(self):
        """Get wetting angle history for plotting"""
        return list(self.wetting_angle_history)

    def get_parameter_history(self):
        """Get parameter history for plotting"""
        return list(self.parameter_history)

    def get_current_mesh_data(self):
        """
        Get current surface data directly from the clad_manager
        using the new generate_surface_info method for efficient visualization

        Returns:
            Dictionary with surface data ready for Plotly Surface plot
        """
        # Add verbose debugging
        print("Attempting to get surface data...")

        # Check if simulation exists
        if self.simulation is None:
            print("Error: simulation is None")
            return None

        # Check if clad_manager exists
        if not hasattr(self.simulation, 'clad_manager'):
            print("Error: simulation has no clad_manager attribute")
            return None

        if self.simulation.clad_manager is None:
            print("Error: clad_manager is None")
            return None

        try:
            # Use the new generate_surface_info method for direct surface data
            surface_data = self.simulation.clad_manager.generate_surface_info(x_res=100, y_res=100)

            if surface_data is None:
                print("Error: generate_surface_info returned None")
                return None

            # Store timestamp and step info
            surface_data['step'] = self.current_step
            surface_data['timestamp'] = time.time()

            print(f"Successfully generated surface data with grid size {surface_data['z'].shape}")

            return surface_data

        except Exception as e:
            print(f"Error generating surface data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_current_mesh_file(self):
        """Get the path to the current mesh file"""
        return self.current_mesh_file

    def get_current_mesh(self):
        """Get current mesh data"""
        return self.current_build_mesh

    def get_mesh_history(self):
        """Get list of mesh files for animation"""
        return list(self.mesh_files)

