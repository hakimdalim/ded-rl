import itertools

from typing import Tuple


class ScanPathManager:
    """
    Manages scan path strategies for multi-track, multi-layer builds.

    Handles scan direction and track ordering based on different strategies:
    1. Bidirectional tracks: Alternate scan direction for adjacent tracks within a layer
    2. Bidirectional layers: Alternate starting side for each layer
    3. Switch scan direction between layers: When starting a new layer, switch scan direction
       for first track compared to last track of previous layer

    Default behavior (all True):
    - Tracks alternate direction within layer (zigzag pattern)
    - Each layer starts from alternate side
    - First track of new layer goes in opposite direction of last track in previous layer

    Args:
        num_tracks: Number of tracks per layer
        track_length: Length of each track in meters
        hatch_space: Distance between track centers in meters
        turnaround_time: Time taken for laser to change direction in seconds
        bidirectional_tracks: If True, alternate scan direction between adjacent tracks
        bidirectional_layers: If True, alternate starting side for each layer
        switch_scan_direction_between_layers: If True, switch scan direction when starting new layer
    """

    def __init__(
            self,
            num_tracks: int,
            track_length: float,
            hatch_space: float,
            turnaround_time: float,
            bidirectional_tracks: bool = True,
            bidirectional_layers: bool = True,
            switch_scan_direction_between_layers: bool = True,
    ):

        self.num_tracks = num_tracks
        self.track_length = track_length
        self.hatch_space = hatch_space
        self.turnaround_time = turnaround_time
        self.bidirectional_tracks = bidirectional_tracks
        self.bidirectional_layers = bidirectional_layers
        self.switch_scan_direction_between_layers = switch_scan_direction_between_layers

        self.scan_sequence = None
        self._current_state = None
        self.is_layer_end = False
        self.reset()

    def reset(self):
        """Reset the scan sequence generator to its initial state."""
        self.scan_sequence = self._generate_scan_sequence()
        self._current_state = None
        self.is_layer_end = False

    def next_track(self, params: dict) -> Tuple[Tuple[int, int, bool], dict]:
        """
        Get the next track in the sequence and calculate transition requirements.

        Args:
            params: Dictionary containing process parameters including scan_speed

        Returns:
            tuple: (
                (layer_idx, track_idx, scan_direction, total_time),
                transition_info: dict with transition calculations or None if first track
            )
        """
        try:
            # Get next track information
            next_layer, next_track, next_direction, reverse_track_idx = next(self.scan_sequence)

            # Store values for future transition calculations
            if not hasattr(self, '_current_state') or self._current_state is None:
                # First track, initialize state by copying the current state (will not result in a transition)
                self._current_state = (next_layer, next_track, next_direction, reverse_track_idx)
                '''self.is_layer_end = False
            else:
                # Check if we're at the end of a layer
                current_layer, current_track, current_direction, _ = self._current_state
                self.is_layer_end = (
                        current_track == (0 if self.bidirectional_layers and current_layer % 2 == 1
                                          else self.num_tracks - 1)
                )'''

            # Calculate transition from current to next track
            current_layer, current_track, current_direction, _ = self._current_state

            # Get transition requirements
            transition_info = self._track_transition(
                current_layer, current_track, current_direction,
                next_layer, next_track, next_direction
            )

            # Calculate total travel distance and time
            travel_distance = (
                    transition_info['track_lengths_to_travel'] * self.track_length +
                    transition_info['hatch_spaces_to_travel'] * self.hatch_space
            )

            dwell_time, turnaround_time = self._calc_turn_around_time(
                num_direction_changes=transition_info['num_turnarounds'],
                travel_distance=travel_distance,
                params=params
            )

            # Add time calculations to transition info
            transition_info.update({
                'dwell_time': dwell_time,
                'turnaround_time': turnaround_time,
                'travel_time': dwell_time - turnaround_time,
                'travel_distance': travel_distance
            })

            # Update current state
            self._current_state = (next_layer, next_track, next_direction, reverse_track_idx)

            return (next_layer, next_track, next_direction, reverse_track_idx, dwell_time), transition_info

        except StopIteration:
            # This won't actually happen with our current implementation
            # since _generate_scan_sequence is an infinite generator
            return None, None

    @property
    def current_state(self):
        return self._current_state

    def _calc_turn_around_time(
            self,
            num_direction_changes: int,
            travel_distance: float,
            params: dict
    ) -> Tuple[float, float]:
        """
        Calculate total time including turnarounds and travel.

        Args:
            num_direction_changes: Number of times the direction changes
            travel_distance: Total distance traveled in meters
            params: Dictionary containing process parameters including scan_speed

        Returns:
            tuple: (total_time, turnaround_time)
                total_time: Total time including travel and turnarounds in seconds
                turnaround_time: Time spent in turnarounds only in seconds

        Raises:
            KeyError: If scan_speed not found in params
        """
        travel_time = travel_distance / params['scan_speed']
        total_turnaround_time = num_direction_changes * self.turnaround_time
        total_time = travel_time + total_turnaround_time

        return total_time, total_turnaround_time

    def _generate_scan_sequence(self):
        """
        Generate sequence of tracks in the correct order with their scan directions.
        Yields scan information for each track based on configured scan strategy.

        Yields:
            tuple: (layer_idx, track_idx, scan_direction)
                layer_idx: Index of current layer
                track_idx: Index of current track
                scan_direction: True for negative y direction, False for positive y direction
        """

        def _layers_and_tracks():
            _layer_idx = 0
            while True:
                if self.bidirectional_layers and _layer_idx % 2 == 1:
                    for i, _track_idx in enumerate(range(self.num_tracks - 1, -1, -1)):
                        self.is_layer_end = (i == self.num_tracks - 1)  # True on last track
                        yield _layer_idx, _track_idx, True
                else:
                    for i, _track_idx in enumerate(range(self.num_tracks)):
                        self.is_layer_end = (i == self.num_tracks - 1)  # True on last track
                        yield _layer_idx, _track_idx, False
                _layer_idx += 1

        scan_direction = False
        layer_tracking = 0
        for layer_idx, track_idx, reverse_track_idx in _layers_and_tracks():

            layer_changed = layer_idx != layer_tracking
            layer_tracking = layer_idx

            if (
                    (layer_changed and self.switch_scan_direction_between_layers)
                    or self.bidirectional_tracks
            ):
                scan_direction = not scan_direction

            yield layer_idx, track_idx, scan_direction, reverse_track_idx

    def _track_transition(
            self,
            current_layer: int,
            current_track: int,
            current_direction: bool,
            next_layer: int,
            next_track: int,
            next_direction: bool
    ) -> dict:
        """
        Calculate the movement requirements between two tracks.

        Args:
            current_layer: Current layer index
            current_track: Current track index
            current_direction: Current scan direction (True for negative y, False for positive y)
            next_layer: Next layer index
            next_track: Next track index
            next_direction: Next scan direction (True for negative y, False for positive y)

        Returns:
            dict containing:
                num_turnarounds: Number of direction changes needed
                track_lengths_to_travel: Number of track lengths to travel
                hatch_spaces_to_travel: Number of hatch spaces to travel
                direction_changed: Whether direction changed
                layer_changed: Whether layer changed
        """
        layer_changed = current_layer != next_layer
        direction_changed = current_direction != next_direction
        track_changed = current_track != next_track

        if not (layer_changed or track_changed or direction_changed):
            # No change in any parameter
            return {
                'num_turnarounds': 0,
                'track_lengths_to_travel': 0,
                'hatch_spaces_to_travel': 0,
                'direction_changed': False,
                'layer_changed': False
            }

        # Same layer transitions
        elif not layer_changed:
            if direction_changed:
                # Only need one turnaround for bidirectional scanning
                return {
                    'num_turnarounds': 1,
                    'track_lengths_to_travel': 0,
                    'hatch_spaces_to_travel': abs(next_track - current_track),
                    'direction_changed': True,
                    'layer_changed': False
                }
            else:
                # Need two turnarounds and travel one track length_between
                return {
                    'num_turnarounds': 2,
                    'track_lengths_to_travel': 1,
                    'hatch_spaces_to_travel': abs(next_track - current_track),
                    'direction_changed': False,
                    'layer_changed': False
                }

        # Layer change with same track index
        elif current_track == next_track:
            if direction_changed:
                # One turnaround when changing direction between layers
                return {
                    'num_turnarounds': 1,
                    'track_lengths_to_travel': 0,
                    'hatch_spaces_to_travel': 0,
                    'direction_changed': True,
                    'layer_changed': True
                }
            else:
                # Two turnarounds and travel one track length_between if keeping same direction
                return {
                    'num_turnarounds': 2,
                    'track_lengths_to_travel': 1,
                    'hatch_spaces_to_travel': 0,
                    'direction_changed': False,
                    'layer_changed': True
                }

        # Layer change with track index change
        else:
            if direction_changed:
                # One turnaround plus hatch space travel
                return {
                    'num_turnarounds': 1,
                    'track_lengths_to_travel': 0,
                    'hatch_spaces_to_travel': self.num_tracks,
                    'direction_changed': True,
                    'layer_changed': True
                }
            else:
                # Two turnarounds plus track length_between and hatch space travel
                return {
                    'num_turnarounds': 2,
                    'track_lengths_to_travel': 1,
                    'hatch_spaces_to_travel': self.num_tracks,
                    'direction_changed': False,
                    'layer_changed': True
                }

    def summary(self, num_layers: int, params: dict, print_sequence: bool = True, title="Scan Configuration Analysis") -> dict:
        """
        Prints a detailed analysis of the scan sequence and returns statistics.
        Uses its own sequence generation to avoid interfering with main tracking.

        Args:
            num_layers: Number of layers to analyze
            params: Dictionary containing process parameters including scan_speed
            print_sequence: If True, prints detailed sequence information

        Returns:
            dict: Dictionary containing statistics about the scan sequence
        """

        def _generate_sequence():
            """Local generator function mimicking _generate_scan_sequence"""

            def _layers_and_tracks():
                _layer_idx = 0
                while True:
                    if self.bidirectional_layers and _layer_idx % 2 == 1:
                        for _track_idx in range(self.num_tracks - 1, -1, -1):
                            yield _layer_idx, _track_idx
                    else:
                        for _track_idx in range(self.num_tracks):
                            yield _layer_idx, _track_idx
                    _layer_idx += 1

            scan_direction = False
            layer_tracking = 0
            for layer_idx, track_idx in _layers_and_tracks():
                layer_changed = layer_idx != layer_tracking
                layer_tracking = layer_idx

                if (
                        (layer_changed and self.switch_scan_direction_between_layers)
                        or self.bidirectional_tracks
                ):
                    scan_direction = not scan_direction

                yield layer_idx, track_idx, scan_direction

        # Get sequence for specified number of layers
        sequence = list(itertools.islice(_generate_sequence(), self.num_tracks * num_layers))

        # Print configuration summary
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"Tracks per layer: {self.num_tracks}")
        print(f"Number of layers: {num_layers}")
        print(f"Track length_between: {self.track_length * 1000:.2f} mm")
        print(f"Hatch spacing: {self.hatch_space * 1000:.2f} mm")
        print(f"Scan speed: {params['scan_speed'] * 1000:.2f} mm/s")
        print(f"Turnaround time: {self.turnaround_time * 1000:.2f} ms")
        print("\nStrategy:")
        print(f"├─ Bidirectional tracks: {'✓' if self.bidirectional_tracks else '✗'}")
        print(f"├─ Bidirectional layers: {'✓' if self.bidirectional_layers else '✗'}")
        print(f"└─ Switch direction between layers: {'✓' if self.switch_scan_direction_between_layers else '✗'}")

        if print_sequence:
            print("\nScan Sequence:")
            print("-" * 95)
            print(f"{'Layer':^6} {'Track':^6} {'Direction':^10} {'Track':^12} {'Transit':^12} "
                  f"{'Turns':^10} {'Total':^12} {'Cumulative':^15}")
            print("-" * 95)

        # Initialize statistics
        stats = {
            'total_time': 0,
            'travel_time': 0,
            'turnaround_time': 0,
            'num_direction_changes': 0,
            'num_tracks': len(sequence),
            'transition_distance': 0
        }

        # Calculate times for each track and transition
        track_time = self.track_length / params['scan_speed']
        cumulative_time = 0
        previous_state = None

        for i, (layer_idx, track_idx, direction) in enumerate(sequence):
            cumulative_time += track_time

            # Calculate transition from previous track if not first track
            if previous_state:
                prev_layer, prev_track, prev_direction = previous_state

                # Get transition requirements
                transition = self._track_transition(
                    prev_layer, prev_track, prev_direction,
                    layer_idx, track_idx, direction
                )

                # Calculate transition times and distances
                travel_distance = (
                        transition['track_lengths_to_travel'] * self.track_length +
                        transition['hatch_spaces_to_travel'] * self.hatch_space
                )

                travel_time = travel_distance / params['scan_speed']
                turnaround_time = transition['num_turnarounds'] * self.turnaround_time
                total_transition_time = travel_time + turnaround_time

                # Update statistics
                stats['num_direction_changes'] += transition['num_turnarounds']
                stats['turnaround_time'] += turnaround_time
                stats['travel_time'] += travel_time
                stats['transition_distance'] += travel_distance
                cumulative_time += total_transition_time

                if print_sequence:
                    print(f"{layer_idx:^6d} {track_idx:^6d} "
                          f"{'↓' if direction else '↑':^10} "
                          f"{track_time * 1000:^12.1f} {travel_time * 1000:^12.1f} "
                          f"{transition['num_turnarounds']:^10} {total_transition_time * 1000:^12.1f} "
                          f"{cumulative_time * 1000:^15.1f}")
            else:
                if print_sequence:
                    print(f"{layer_idx:^6d} {track_idx:^6d} "
                          f"{'↓' if direction else '↑':^10} "
                          f"{track_time * 1000:^12.1f} {'-':^12} "
                          f"{'-':^10} {'-':^12} "
                          f"{cumulative_time * 1000:^15.1f}")

            previous_state = (layer_idx, track_idx, direction)

        # Add track travel time to total travel time
        total_track_time = track_time * len(sequence)
        stats['travel_time'] += total_track_time
        stats['total_time'] = cumulative_time
        stats['efficiency'] = (stats['travel_time'] / stats['total_time']) * 100
        stats['average_time_per_track'] = stats['total_time'] / len(sequence)

        # Print summary statistics
        print("\nProcess Statistics:")
        print("-" * 40)
        print(f"Total tracks: {stats['num_tracks']}")
        print(f"Direction changes: {stats['num_direction_changes']}")
        print(f"Track travel time: {total_track_time * 1000:.1f}ms")
        print(f"Transition travel time: {(stats['travel_time'] - total_track_time) * 1000:.1f}ms")
        print(f"Turnaround time: {stats['turnaround_time'] * 1000:.1f}ms")
        print(f"Total time: {stats['total_time'] * 1000:.1f}ms")
        print(f"Process efficiency: {stats['efficiency']:.1f}%")
        print(f"Average time per track: {stats['average_time_per_track'] * 1000:.1f}ms")
        print(f"Total transition distance: {stats['transition_distance'] * 1000:.1f}mm")

        return stats


if __name__ == "__main__":
    # Test parameters
    track_length = 0.01  # 10mm
    hatch_space = 0.001  # 1mm
    turnaround_time = 0.1  # 100ms
    test_params = {'scan_speed': 0.1}  # 100mm/s

    # Create all possible configurations using itertools.product
    bool_options = [True, False]
    num_tracks_options = [4, 3]  # Even and odd number of tracks
    config_names = [
        'bidirectional_tracks',
        'bidirectional_layers',
        'switch_scan_direction_between_layers'
    ]

    # Generate all combinations
    configs = []
    for num_tracks in num_tracks_options:
        for values in itertools.product(bool_options, repeat=3):
            config = {
                'name': (f"Tracks: {num_tracks} ({'odd' if num_tracks % 2 else 'even'}) | " +
                        ' | '.join(f"{name.split('_')[1]}={'✓' if value else '✗'}"
                                 for name, value in zip(config_names, values))),
                'params': dict(zip(config_names, values)),
                'num_tracks': num_tracks
            }
            configs.append(config)

    # Test each configuration
    total_configs = len(configs)
    for config_num, config in enumerate(configs, 1):

        # Create manager with configuration
        manager = ScanPathManager(
            num_tracks=config['num_tracks'],
            track_length=track_length,
            hatch_space=hatch_space,
            turnaround_time=turnaround_time,
            **config['params']
        )

        # Use summary method with configuration name as title
        stats = manager.summary(
            num_layers=3,
            params=test_params,
            title=f"Configuration {config_num}/{total_configs}: {config['name']}"
        )
