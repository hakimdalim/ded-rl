"""
Base callback system for DED simulation with event-driven architecture.
"""

import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, List, Union, Set
import numpy as np


class SimulationEvent(Enum):
    """Events that can trigger callbacks during simulation."""
    # Simulation lifecycle
    CONFIG_LOADED = auto()  # Configuration loaded
    INIT = auto()  # Simulation initialized (constructor)
    RESET = auto()  # Simulation reset
    ERROR = auto()  # Error during step
    COMPLETE = auto()  # All layers/tracks done

    # Step events
    STEP_START = auto()  # Before step calculations
    STEP_COMPLETE = auto()  # After step calculations

    # Track events
    TRACK_START = auto()  # Starting a new track
    TRACK_COMPLETE = auto()  # Track finished

    # Layer events
    LAYER_START = auto()  # Starting a new layer


class BaseCallback(ABC):
    """Minimal base class for callbacks."""

    def __init__(
            self,
            events: Union[SimulationEvent, List[SimulationEvent], 
                          Set[SimulationEvent]],
            enabled: bool = True,
            **kwargs
    ):
        if isinstance(events, SimulationEvent):
            self.events = {events}
        else:
            self.events = set(events)

        self.enabled = enabled
        self.config = kwargs

        # Event counter for tracking
        self._event_counts = {event: 0 for event in self.events}

    def __call__(self, context: dict) -> Any:
        """Main entry point."""
        event = context.get('event')

        if not self.enabled or event not in self.events:
            return None

        self._event_counts[event] += 1

        if self.should_execute(context):
            return self._execute(context)

        return None

    def should_execute(self, context: dict) -> bool:
        """Override for custom logic. Default: always execute."""
        return True

    def resolve_path(self, context: dict, *path_parts) -> Path:
        """
        Helper to resolve paths relative to simulation output directory.

        Args:
            context: Simulation context
            *path_parts: Path components to join

        Returns:
            Full path under simulation.output_dir
        """
        output_dir = context['simulation'].output_dir
        return Path(output_dir) / Path(*path_parts)

    def ensure_dir(self, path: Path) -> Path:
        """Helper to ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def _execute(self, context: dict) -> Any:
        """Implement callback logic."""
        pass


class IntervalCallback(BaseCallback):
    """Callback that executes at intervals."""

    def __init__(self, *args, interval: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def should_execute(self, context: dict) -> bool:
        """Only execute every N occurrences."""
        event = context['event']
        return self._event_counts[event] % self.interval == 0


