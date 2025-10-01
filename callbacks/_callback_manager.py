import warnings
from typing import List

from callbacks._base_callbacks import SimulationEvent, BaseCallback
from callbacks.completion_callbacks import BaseCompletionCallback, SimulationComplete
from callbacks.error_callbacks import ErrorCompletionCallback
from callbacks.step_data_collector import StepDataCollector


class CallbackManager:
    """Manages callback execution for simulations."""

    def __init__(self, callbacks=None):
        """
        Initialize callback manager.

        Args:
            callbacks: None, single callback, or list of callbacks
        """
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            # Single callback
            self.callbacks = [callbacks]

        # Validate and auto-configure callbacks
        self._validate_and_configure_callbacks()

    def _validate_and_configure_callbacks(self):
        """
        Validate callback configuration and add defaults if needed.

        Warns if no completion callbacks present.
        Adds StepDataCollector if none present.
        Adds ErrorCompletionCallback if none present.
        """

        # Check for error callbacks
        has_error_callback = any(
            isinstance(cb, ErrorCompletionCallback)
            for cb in self.callbacks
        )

        if not has_error_callback:
            # Add default ErrorCompletionCallback
            error_callback = ErrorCompletionCallback(log_traceback=True)
            self.callbacks.append(error_callback)
            print("Added default ErrorCompletionCallback for proper error handling")

        # Check for completion callbacks
        has_completion = any(
            isinstance(cb, BaseCompletionCallback)
            for cb in self.callbacks
        )

        if not has_completion:
            warnings.warn(
                "No completion callback detected - simulation will run indefinitely!\n"
                "Consider adding a completion callback such as:\n"
                "  - HeightCompletionCallback() for height-based completion\n"
                "  - LayerCountCompletionCallback(n) for fixed layer count\n"
                "  - StepCountCompletionCallback(n) for fixed step count",
                RuntimeWarning
            )

        # Check for step data collectors
        has_step_collector = any(
            isinstance(cb, StepDataCollector)
            for cb in self.callbacks
        )

        if not has_step_collector:
            # Add default StepDataCollector with no saving
            # tracked_fields=None means track everything
            # save_path=None means no CSV output
            default_collector = StepDataCollector(
                tracked_fields=None,  # Track everything
                save_path=None  # Don't save to file
            )
            self.callbacks.append(default_collector)
            print("Added default StepDataCollector (tracking all fields, no file output)")

    def __call__(self, simulation, event: SimulationEvent, **extra_context):
        """
        Execute callbacks for given event.

        Args:
            simulation: The simulation instance (has all the data we need)
            event: The event that triggered this call
            **extra_context: Any additional context data
        """
        # Build context dict
        context = {
            'event': event,
            'simulation': simulation,
            **extra_context
        }

        # Execute callbacks
        for callback in self.callbacks:
            if callback.enabled and event in callback.events:
                try:
                    callback(context)
                except SimulationComplete:
                    # Reraise to allow simulation to handle completion
                    raise
                except Exception as e:
                    # Add callback information to the exception without losing stack trace
                    callback_info = (
                        f"\n[Callback Error] {callback.__class__.__name__} "
                        f"(from {callback.__class__.__module__}) failed during {event.name} event"
                    )
                    # Create new exception with enhanced message
                    new_msg = f"{callback_info}\nOriginal error: {str(e)}"
                    new_exception = type(e)(new_msg)
                    # Preserve the original traceback
                    raise new_exception.with_traceback(e.__traceback__) from None

    def add_callback(self, callback: BaseCallback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)
        # Re-validate after adding
        self._validate_and_configure_callbacks()

    def remove_callback(self, callback: BaseCallback):
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def enable_callback(self, callback_class: type):
        """Enable all callbacks of a given class."""
        for callback in self.callbacks:
            if isinstance(callback, callback_class):
                callback.enabled = True

    def disable_callback(self, callback_class: type):
        """Disable all callbacks of a given class."""
        for callback in self.callbacks:
            if isinstance(callback, callback_class):
                callback.enabled = False

    def clear(self):
        """Remove all callbacks."""
        self.callbacks = []

    def get_callbacks_for_event(self, event: SimulationEvent) -> List[BaseCallback]:
        """Get all callbacks registered for a specific event."""
        return [cb for cb in self.callbacks if event in cb.events and cb.enabled]