import warnings
from typing import Dict, Any, Union, Tuple
from collections import UserDict
import numpy as np


class SimulationConfig(UserDict):
    """
    Configuration manager for multi-track, multi-layer additive manufacturing simulation.

    Inherits from UserDict to act as a dictionary while maintaining configuration logic.
    """

    def __init__(
            self,
            # Build volume dimensions
            build_volume_size: Tuple[float, float, float],  # Physical size (x, y, z) in meters
            voxel_size: Union[float, Tuple[float, float, float]],  # meters

            # Part dimensions
            part_width: float,  # Width (X) of the part in meters
            part_length: float,  # Length (Y) of the part in meters
            part_height: float,  # Height (Z) of the part in meters

            # Build parameters
            hatch_spacing: float,  # Distance between track centers in meters
            layer_spacing: float,  # Distance between layers in meters
            substrate_height: float = 0.0,  # meters

            # Process strategy
            bidirectional_tracks: bool = True,
            bidirectional_layers: bool = True,
            switch_scan_direction_between_layers: bool = True,
            turnaround_time: float = 0.1,  # seconds

            # Optional positioning (centered in build volume by default)
            x_offset: float = None,  # meters
            y_offset: float = None,  # meters
    ):
        """Initialize simulation configuration."""
        super().__init__()

        # Store initial arguments as protected (immutable)
        self.__init_config_kwargs = {
            'build_volume_size': build_volume_size,
            'voxel_size': voxel_size,
            'part_width': part_width,
            'part_length': part_length,
            'part_height': part_height,
            'hatch_spacing': hatch_spacing,
            'layer_spacing': layer_spacing,
            'substrate_height': substrate_height,
            'bidirectional_tracks': bidirectional_tracks,
            'bidirectional_layers': bidirectional_layers,
            'switch_scan_direction_between_layers': switch_scan_direction_between_layers,
            'turnaround_time': turnaround_time,
            'x_offset': x_offset,
            'y_offset': y_offset,
        }

        # Create mutable copy for configuration
        self.config_kwargs = self.__init_config_kwargs.copy()

        # Run configuration
        self.configure()

    def reset(self):
        """Reset configuration to initial values."""
        self.config_kwargs = self.__init_config_kwargs.copy()
        self.configure()

    @property
    def _init_config_kwargs(self) -> Dict[str, Any]:
        """Protected property to access initial configuration kwargs."""
        return self.__init_config_kwargs.copy()

    def configure(self):
        """Configure all parameters based on config_kwargs."""
        # Clear existing data
        self.data.clear()

        # Get args for convenience
        args = self.config_kwargs

        # Standardize voxel size
        voxel_size = args['voxel_size']
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        else:
            voxel_size = tuple(float(x) for x in voxel_size)

        # Store basic dimensions
        self.data['build_volume_size'] = args['build_volume_size']
        self.data['voxel_size'] = voxel_size
        self.data['part_width'] = args['part_width']
        self.data['part_length'] = args['part_length']
        self.data['part_height'] = args['part_height']
        self.data['hatch_spacing'] = args['hatch_spacing']
        self.data['layer_spacing'] = args['layer_spacing']
        self.data['substrate_height'] = args['substrate_height']

        # Calculate derived dimensions
        self.data['num_tracks'] = int(np.ceil(args['part_width'] / args['hatch_spacing']))
        self.data['actual_width'] = self.data['num_tracks'] * args['hatch_spacing']
        self.data['track_length'] = args['part_length']

        # Build strategy
        self.data['bidirectional_tracks'] = args['bidirectional_tracks']
        self.data['bidirectional_layers'] = args['bidirectional_layers']
        self.data['switch_scan_direction_between_layers'] = args['switch_scan_direction_between_layers']
        self.data['turnaround_time'] = args['turnaround_time']

        # Calculate offsets (center if not provided)
        x_offset = args['x_offset']
        y_offset = args['y_offset']

        if x_offset is None:
            x_offset = (args['build_volume_size'][0] - self.data['actual_width']) / 2
        if y_offset is None:
            y_offset = (args['build_volume_size'][1] - args['part_length']) / 2

        self.data['x_offset'] = x_offset
        self.data['y_offset'] = y_offset

        # Calculate volume shape
        self.data['volume_shape'] = tuple(
            int(np.ceil(size / vsize))
            for size, vsize in zip(args['build_volume_size'], voxel_size)
        )

        # Run validation
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        # Check build volume dimensions
        if any(x <= 0 for x in self.data['build_volume_size']):
            raise ValueError("All build volume dimensions must be positive")

        if any(x <= 0 for x in self.data['voxel_size']):
            raise ValueError("All voxel dimensions must be positive")

        # Check part dimensions
        if any(x <= 0 for x in (self.data['part_width'], self.data['part_length'], self.data['part_height'])):
            raise ValueError("All part dimensions must be positive")

        if self.data['substrate_height'] < 0:
            raise ValueError("Substrate height cannot be negative")

        if self.data['hatch_spacing'] <= 0:
            raise ValueError("Hatch spacing must be positive")

        if self.data['layer_spacing'] <= 0:
            raise ValueError("Layer spacing must be positive")

        # Check track parameters
        if self.data['num_tracks'] < 1:
            raise ValueError("Must have at least one track")

        if self.data['track_length'] <= 0:
            raise ValueError("Track length must be positive")

        # Check if part fits within build volume
        if self.data['actual_width'] + self.data['x_offset'] > self.data['build_volume_size'][0]:
            raise ValueError(
                f"Part width ({self.data['actual_width'] * 1000:.2f}mm) plus offset ({self.data['x_offset'] * 1000:.2f}mm) "
                f"exceeds build volume width ({self.data['build_volume_size'][0] * 1000:.2f}mm)"
            )
        if self.data['part_length'] + self.data['y_offset'] > self.data['build_volume_size'][1]:
            raise ValueError(
                f"Part length ({self.data['part_length'] * 1000:.2f}mm) plus offset ({self.data['y_offset'] * 1000:.2f}mm) "
                f"exceeds build volume length ({self.data['build_volume_size'][1] * 1000:.2f}mm)"
            )
        if self.data['part_height'] + self.data['substrate_height'] > self.data['build_volume_size'][2]:
            raise ValueError(
                f"Part height ({self.data['part_height'] * 1000:.2f}mm) plus substrate ({self.data['substrate_height'] * 1000:.2f}mm) "
                f"exceeds build volume height ({self.data['build_volume_size'][2] * 1000:.2f}mm)"
            )

        # Check discretization resolution
        POINTS_ALONG_LENGTH = 100  # Desired minimum number of points along track length
        POINTS_PER_HATCH = 5  # Desired minimum number of points per hatch spacing

        # Check track length resolution
        points_along_length = self.data['track_length'] / self.data['voxel_size'][1]
        if points_along_length < POINTS_ALONG_LENGTH:
            warnings.warn(
                f"Low resolution along track length. Current configuration has {points_along_length:.0f} points "
                f"per track (recommended minimum: {POINTS_ALONG_LENGTH}). "
                f"This might affect simulation accuracy. Consider decreasing Y voxel size from "
                f"{self.data['voxel_size'][1] * 1e6:.1f}µm to {(self.data['track_length'] / POINTS_ALONG_LENGTH) * 1e6:.1f}µm or less."
            )

        # Check hatch spacing resolution
        points_per_hatch = self.data['hatch_spacing'] / self.data['voxel_size'][0]
        if points_per_hatch < POINTS_PER_HATCH:
            warnings.warn(
                f"Low resolution across hatch spacing. Current configuration has {points_per_hatch:.1f} points "
                f"per hatch spacing (recommended minimum: {POINTS_PER_HATCH}). "
                f"This might affect simulation accuracy. Consider decreasing X voxel size from "
                f"{self.data['voxel_size'][0] * 1e6:.1f}µm to {(self.data['hatch_spacing'] / POINTS_PER_HATCH) * 1e6:.1f}µm or less."
            )

    def get_simulation_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary (for backward compatibility)."""
        return dict(self.data)

    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("\nSimulation Configuration Summary")
        print("=" * 40)

        print("\nBuild Volume:")
        print(f"Size: {[x * 1000 for x in self['build_volume_size']]} mm")
        print(f"Voxel size: {[x * 1e6 for x in self['voxel_size']]} µm")
        print(f"Volume shape: {self['volume_shape']} voxels")

        # Calculate total voxels
        x_voxels = self['volume_shape'][0]
        xy_voxels = x_voxels * self['volume_shape'][1]
        total_voxels = xy_voxels * self['volume_shape'][2]
        print(f"Total number of voxels: {self._format_large_number(total_voxels)}")

        memory_estimate_mb = (total_voxels * 8) / (1024 * 1024)  # 8 bytes per voxel
        if memory_estimate_mb > 1024:
            memory_estimate_gb = memory_estimate_mb / 1024
            print(f"Estimated memory per field: {memory_estimate_gb:.1f} GB")
        else:
            print(f"Estimated memory per field: {memory_estimate_mb:.1f} MB")

        print(f"Substrate height: {self['substrate_height'] * 1000:.2f} mm")

        print("\nDiscretization Details:")
        points_along_track = self['track_length'] / self['voxel_size'][1]
        points_per_hatch = self['hatch_spacing'] / self['voxel_size'][0]
        points_per_height = self['part_height'] / self['voxel_size'][2]
        print(f"Points along track length: {points_along_track:.1f}")
        print(f"Points per hatch spacing: {points_per_hatch:.1f}")
        print(f"Points along part height: {points_per_height:.1f}")

        print("\nPart Dimensions:")
        print(f"Desired width: {self['part_width'] * 1000:.3f} mm")
        print(f"Actual width: {self['actual_width'] * 1000:.3f} mm")
        print(f"Length: {self['part_length'] * 1000:.3f} mm")
        print(f"Height: {self['part_height'] * 1000:.3f} mm")
        print(f"Position (offset): ({self['x_offset'] * 1000:.2f}, {self['y_offset'] * 1000:.2f}) mm")

        print("\nTrack Configuration:")
        print(f"Number of tracks: {self['num_tracks']}")
        print(f"Track length: {self['track_length'] * 1000:.2f} mm")
        print(f"Hatch spacing: {self['hatch_spacing'] * 1000:.2f} mm")
        points_per_track = np.ceil(self['track_length'] / self['voxel_size'][1])
        total_track_points = points_per_track * self['num_tracks']
        print(f"Points per track: {points_per_track:.0f}")
        print(f"Total track points: {self._format_large_number(int(total_track_points))}")

        print("\nBuild Strategy:")
        print(f"Bidirectional tracks: {'✓' if self['bidirectional_tracks'] else '✗'}")
        print(f"Bidirectional layers: {'✓' if self['bidirectional_layers'] else '✗'}")
        print(f"Switch direction between layers: "
              f"{'✓' if self['switch_scan_direction_between_layers'] else '✗'}")
        print(f"Turnaround time: {self['turnaround_time'] * 1000:.1f} ms")

    def _format_large_number(self, number: int) -> str:
        """Format large numbers with appropriate unit suffixes."""
        suffixes = ['', 'K', 'M', 'B', 'T']
        for suffix in suffixes:
            if number < 1000:
                if suffix == '':
                    return f"{number:d}"
                return f"{number:.2f}{suffix}"
            number /= 1000
        return f"{number:.2f}T+"


if __name__ == "__main__":
    # Example usage
    config = SimulationConfig(
        # Build volume (20mm x 20mm x 10mm)
        build_volume_size=(0.02, 0.02, 0.01),
        voxel_size=0.00005,  # 50µm voxels

        # Part dimensions (10mm x 10mm x 5mm)
        part_width=0.01,
        part_length=0.01,
        part_height=0.005,

        # Build parameters
        hatch_spacing=0.0005,  # 500µm track spacing
        layer_spacing=0.00035,  # 350µm layer spacing
        substrate_height=0.001,  # 1mm substrate
    )

    # Print configuration summary
    config.print_summary()

    # Access as dictionary
    print("\n\nAccessing as dictionary:")
    print(f"Number of tracks: {config['num_tracks']}")
    print(f"Volume shape: {config['volume_shape']}")

    # Get configuration dictionary for simulation (backward compatibility)
    sim_config = config.get_simulation_config()

    # Modify config_kwargs and reconfigure
    print("\n\nModifying configuration:")
    config.config_kwargs['part_width'] = 0.008  # Change to 8mm
    config.config_kwargs['hatch_spacing'] = 0.0004  # Change to 400µm
    config.configure()  # Reconfigure

    print(f"New number of tracks: {config['num_tracks']}")
    print(f"New actual width: {config['actual_width'] * 1000:.3f} mm")

    # Show that init_config_kwargs is protected
    print("\n\nProtected initial config:")
    print(f"Original part width: {config._init_config_kwargs['part_width'] * 1000:.3f} mm")
    print(f"Current part width: {config.config_kwargs['part_width'] * 1000:.3f} mm")

    # Example validation error - part too large for build volume
    try:
        invalid_config = SimulationConfig(
            build_volume_size=(0.01, 0.01, 0.01),  # 10mm cube build volume
            voxel_size=0.0001,
            part_width=0.015,  # 15mm - too large!
            part_length=0.015,  # 15mm - too large!
            part_height=0.005,
            hatch_spacing=0.0005,
            layer_spacing=0.00035,  # 350µm layer spacing
        )
    except ValueError as e:
        print("\nValidation Error Example:")
        print(e)