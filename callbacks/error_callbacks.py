"""
Error callbacks for DED simulation.
Handle simulation errors and ensure proper cleanup.
"""

from callbacks._base_callbacks import BaseCallback, SimulationEvent


class ErrorCompletionCallback(BaseCallback):
    """
    Ensures simulation completes properly when an error occurs.

    This callback is triggered on ERROR events and calls the simulation's
    complete() method to ensure proper cleanup and final callbacks are executed.
    """

    def __init__(self, **kwargs):
        """
        Initialize error completion callback.

        Args:
            **kwargs: Additional configuration for BaseCallback
        """
        super().__init__(events=SimulationEvent.ERROR, **kwargs)

    def _execute(self, context: dict) -> None:
        """
        Handle error by completing simulation properly.

        Args:
            context: Simulation context with 'error' key containing the exception
        """
        sim = context['simulation']

        # Call simulation's complete method to trigger COMPLETE callbacks
        # This ensures cleanup callbacks run even on error
        sim.complete()