"""
Completion callbacks for DED simulation.
Handle simulation termination conditions and completion logic.
"""

from abc import abstractmethod
from typing import Optional
from callbacks._base_callbacks import BaseCallback, SimulationEvent


class SimulationComplete(Exception):
    """Exception raised when simulation should terminate."""
    pass


class BaseCompletionCallback(BaseCallback):
    """
    Base class for all completion callbacks.

    Completion callbacks monitor simulation state and raise SimulationComplete
    when their termination condition is met.
    """

    def _execute(self, context: dict) -> None:
        """
        Check completion condition and raise if met.

        Args:
            context: Simulation context

        Raises:
            SimulationComplete: When completion condition is met
        """
        if self.is_complete(context):
            message = self.get_completion_message(context)
            raise SimulationComplete(message)

    @abstractmethod
    def is_complete(self, context: dict) -> bool:
        """
        Check if completion condition is met.

        Args:
            context: Simulation context

        Returns:
            True if simulation should complete
        """
        pass

    @abstractmethod
    def get_completion_message(self, context: dict) -> str:
        """
        Get the completion message to include in the exception.

        Args:
            context: Simulation context

        Returns:
            Message describing why simulation is completing
        """
        pass


class HeightCompletionCallback(BaseCompletionCallback):
    """
    Completes simulation when build height is reached at layer start.

    Checks if the maximum height reached exceeds the target part height
    when a new layer begins, ensuring clean layer-by-layer completion.
    """

    def __init__(self, target_height: Optional[float] = None, **kwargs):
        """
        Initialize height completion callback.

        Args:
            target_height: Target build height in meters. If None, uses
                          config.part_height + config.substrate_height
            **kwargs: Additional configuration for BaseCallback
        """
        super().__init__(events=SimulationEvent.LAYER_START, **kwargs)
        self.target_height = target_height
        self._calculated_target = None
        self._current_height = None

    def is_complete(self, context: dict) -> bool:
        """Check if build height has been reached."""
        sim = context['simulation']

        # Determine target height
        if self.target_height is None:
            config = sim.config
            self._calculated_target = config.get('part_height', 0) + config.get('substrate_height', 0)
        else:
            self._calculated_target = self.target_height

        # Get current maximum height
        self._current_height = sim.progress_tracker.get_transition_summary().get('max_height_reached', 0)

        return self._current_height >= self._calculated_target

    def get_completion_message(self, context: dict) -> str:
        """Get height completion message."""
        return (f"Build complete: reached target height {self._calculated_target*1000:.2f}mm "
                f"(current: {self._current_height*1000:.2f}mm)")


class TrackCountCompletionCallback(BaseCompletionCallback):
    """
    Completes simulation after a specified number of tracks.

    Useful for debugging or creating partial builds.
    """

    def __init__(self, max_tracks: int, **kwargs):
        """
        Initialize track count completion callback.

        Args:
            max_tracks: Maximum number of tracks to complete
            **kwargs: Additional configuration for BaseCallback
        """
        super().__init__(events=SimulationEvent.TRACK_COMPLETE, **kwargs)
        self.max_tracks = max_tracks
        self.completed_tracks = 0

    def is_complete(self, context: dict) -> bool:
        """Check if max tracks completed."""
        self.completed_tracks += 1
        return self.completed_tracks >= self.max_tracks

    def get_completion_message(self, context: dict) -> str:
        """Get track completion message."""
        return f"Build stopped: completed {self.completed_tracks} tracks (limit: {self.max_tracks})"


class LayerCountCompletionCallback(BaseCompletionCallback):
    """
    Completes simulation after a specified number of layers.

    Ensures exact layer count for controlled builds.
    """

    def __init__(self, max_layers: int, **kwargs):
        """
        Initialize layer count completion callback.

        Args:
            max_layers: Maximum number of layers to build
            **kwargs: Additional configuration for BaseCallback
        """
        super().__init__(events=SimulationEvent.LAYER_START, **kwargs)
        self.max_layers = max_layers
        self._current_layer = None

    def is_complete(self, context: dict) -> bool:
        """Check if max layers reached."""
        sim = context['simulation']
        self._current_layer = sim.progress_tracker.current_layer
        # Layer indices start at 0
        return self._current_layer >= self.max_layers

    def get_completion_message(self, context: dict) -> str:
        """Get layer completion message."""
        return f"Build stopped: reached layer {self._current_layer} (limit: {self.max_layers} layers)"


class StepCountCompletionCallback(BaseCompletionCallback):
    """
    Completes simulation after a specified number of steps.

    Useful for time-limited simulations or debugging.
    """

    def __init__(self, max_steps: int, **kwargs):
        """
        Initialize step count completion callback.

        Args:
            max_steps: Maximum number of simulation steps
            **kwargs: Additional configuration for BaseCallback
        """
        super().__init__(events=SimulationEvent.STEP_COMPLETE, **kwargs)
        self.max_steps = max_steps
        self._current_steps = None

    def is_complete(self, context: dict) -> bool:
        """Check if max steps completed."""
        sim = context['simulation']
        self._current_steps = sim.progress_tracker.step_count
        return self._current_steps >= self.max_steps

    def get_completion_message(self, context: dict) -> str:
        """Get step completion message."""
        return f"Build stopped: completed {self._current_steps} steps (limit: {self.max_steps})"