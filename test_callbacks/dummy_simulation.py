#dummy_simulation.py
"""
Dummy Simulation for Testing Callbacks
This is a simplified version that mimics the real simulation structure.
"""

import numpy as np
from pathlib import Path


class DummyProgressTracker:
    """Fake progress tracker"""
    def __init__(self):
        self.step_count = 0
        self.current_layer = 0
        self.current_track = 0
        self.max_height_reached = 0.0
    
    def get_transition_summary(self):
        return {
            'step_count': self.step_count,
            'current_layer': self.current_layer,
            'count_layers': self.current_layer,
            'count_tracks': self.current_track,
            'max_height_reached': self.max_height_reached
        }


class DummyTemperatureTracker:
    """Fake temperature tracker"""
    def __init__(self, shape=(50, 50, 30)):
        self.temperature = np.ones(shape) * 300  # Start at room temp
        self.shape = shape


class DummyVolumeTracker:
    """Fake volume tracker"""
    def __init__(self, shape=(50, 50, 30)):
        self.activated = np.zeros(shape, dtype=bool)
        self.shape = shape


class DummyCladManager:
    """Fake clad manager"""
    def __init__(self):
        self._profiles = {}
    
    def save_build_state_mesh(self, filename):
        """Pretend to save mesh"""
        print(f"[DUMMY] Would save mesh to: {filename}")
    
    def plot_all_layers(self, y_pos, ax):
        """Pretend to plot"""
        print(f"[DUMMY] Would plot layers at y={y_pos}")
    
    def get_layer_cross_section(self, layer_idx, y_pos):
        """Return a dummy function"""
        return lambda x: 0.001 * layer_idx  # Each layer 1mm tall


class DummySimulation:
    """
    Simplified simulation class that mimics the real one.
    This has all the attributes that callbacks expect.
    """
    
    def __init__(self, config=None, output_dir="./test_output"):
        # Configuration
        self.config = config or self._default_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trackers
        self.progress_tracker = DummyProgressTracker()
        self.temperature_tracker = DummyTemperatureTracker()
        self.volume_tracker = DummyVolumeTracker()

        #clad manager
        self.clad_manager = DummyCladManager()
        
        # Current step context (updated each step)
        self.step_context = None
        
        # Time step
        self.delta_t = 1e-4  # 0.1 milliseconds
        
    def _default_config(self):
        """Default configuration"""
        return {
            'voxel_size': (0.0001, 0.0001, 0.0001),  # 0.1mm per voxel
            'volume_shape': (50, 50, 30),  # 5mm x 5mm x 3mm volume
            'track_length': 0.004,  # 4mm
            'hatch_spacing': 0.0008,  # 0.8mm
            'num_tracks': 5,
            'part_height': 0.002,  # 2mm
            'substrate_height': 0.0005,  # 0.5mm
            'part_width': 0.004,
            'layer_spacing': 0.0003,
            'x_offset': 0.0005,
        }
    
    def step(self, params=None):
        """
        Simulate one step.
        Updates all the data that callbacks might read.
        """
        if params is None:
            params = self._default_params()
        
        # Update progress
        self.progress_tracker.step_count += 1
        
        # Simulate laser moving
        # Simple: move along x-axis
        step = self.progress_tracker.step_count
        x = 0.001 + (step * 0.0001) % self.config['track_length']
        y = 0.002
        z = 0.001 + self.progress_tracker.current_layer * self.config['layer_spacing']
        
        # Update height
        self.progress_tracker.max_height_reached = max(
            self.progress_tracker.max_height_reached, 
            z + 0.0003  # Add clad height
        )
        
        # Convert to voxel indices
        voxel_x = int(x / self.config['voxel_size'][0])
        voxel_y = int(y / self.config['voxel_size'][1])
        voxel_z = int(z / self.config['voxel_size'][2])
        
        # Simulate temperature increase at laser position
        if voxel_x < self.temperature_tracker.shape[0] and \
           voxel_y < self.temperature_tracker.shape[1] and \
           voxel_z < self.temperature_tracker.shape[2]:
            self.temperature_tracker.temperature[voxel_x, voxel_y, voxel_z] = \
                300 + np.random.uniform(1200, 1800)  # Melt pool temp
            self.volume_tracker.activated[voxel_x, voxel_y, voxel_z] = True
        
        # Create step context (what callbacks read)
        self.step_context = {
            'position': {'x': x, 'y': y, 'z': z},
            'voxel': {'x': voxel_x, 'y': voxel_y, 'z': voxel_z},
            'build': {
                'layer': self.progress_tracker.current_layer,
                'track': self.progress_tracker.current_track,
                'reverse_direction': False,
                'reverse_track': False,
                'time_step': step * self.delta_t,
                'track_progress': (step % 40) / 40.0,  # 40 steps per track
            },
            'melt_pool': {
                'width': 0.0003 + np.random.uniform(-0.00005, 0.00005),
                'length': 0.0005 + np.random.uniform(-0.00005, 0.00005),
                'depth': 0.0002 + np.random.uniform(-0.00003, 0.00003),
                'max_temp': 1500 + np.random.uniform(-200, 200),
                'voxel_center': (voxel_x, voxel_y, voxel_z),
            },
            'clad': {
                'width': 0.0008 + np.random.uniform(-0.0001, 0.0001),
                'height': 0.0003 + np.random.uniform(-0.00005, 0.00005),
                'wetting_angle': 45 + np.random.uniform(-5, 5),
            },
            'profile': {
                'baseline': z,
                'max_z': z + 0.0003,
                'height_at_center': 0.0003,
                'width': 0.0008,
                'object': None,  # Would be actual profile object
            },
            'params': params,
        }
        
        return self.step_context
    
    def _default_params(self):
        """Default process parameters"""
        return {
            'laser_power': 500,  # Watts
            'scan_speed': 0.8,   # m/s
            'powder_feed_rate': 5.0,  # g/min
            'melting_temp': 1700,  # Kelvin
        }
    
    @property
    def temp_slices(self):
        """Get temperature slices for saving"""
        temp = self.temperature_tracker.temperature
        z_mid = temp.shape[2] // 2
        y_mid = temp.shape[1] // 2
        x_mid = temp.shape[0] // 2
        
        return {
            'xy': temp[:, :, z_mid],
            'xz': temp[:, y_mid, :],
            'yz': temp[x_mid, :, :],
        }
    
    def complete(self):
        """Called when simulation completes"""
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total steps: {self.progress_tracker.step_count}")
        print(f"Final height: {self.progress_tracker.max_height_reached * 1000:.3f}mm")
        print(f"Output saved to: {self.output_dir}")
        print(f"{'='*60}\n")