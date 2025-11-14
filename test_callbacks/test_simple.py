#test_simple.py
"""
TEST 1: Simple Callback Test
This is the easiest test to understand - it just prints messages.
"""

import sys
sys.path.append('..')  # Add parent directory to find callback modules

from dummy_simulation import DummySimulation
from callbacks._base_callbacks import BaseCallback, SimulationEvent
from callbacks._callback_manager import CallbackManager


class HelloCallback(BaseCallback):
    """
    A simple callback that just prints hello.
    This runs at the start of simulation.
    """
    def __init__(self):
        super().__init__(events=SimulationEvent.INIT)
        print("🎉 Hello! events=SimulationEvent.INIT is starting!")

    def _execute(self, context):
        print("🎉 Hello! Simulation is starting!")


class StepCounterCallback(BaseCallback):
    """
    Counts and prints every 5 steps.
    """
    def __init__(self):
        super().__init__(events=SimulationEvent.STEP_COMPLETE)
        self.count = 0
    
    def _execute(self, context):
        self.count += 1
        if self.count % 5 == 0:
            print(f"✓ Completed {self.count} steps")


class GoodbyeCallback(BaseCallback):
    """
    Says goodbye when simulation completes.
    """
    def __init__(self):
        super().__init__(events=SimulationEvent.COMPLETE)
    
    def _execute(self, context):
        sim = context['simulation']
        print(f"👋 Goodbye! Ran {sim.progress_tracker.step_count} steps total")


def main():
    print("\n" + "="*60)
    print("TEST 1: Simple Callbacks")
    print("="*60 + "\n")
    
    # Create simulation
    sim = DummySimulation(output_dir="./test_output/simple")
    
    # Create callbacks
    callbacks = [
        HelloCallback(),
        StepCounterCallback(),
        GoodbyeCallback(),
    ]
    
    # Create manager
    manager = CallbackManager(callbacks)
    
    print("\n--- Starting Simulation ---\n")
    
    # Trigger INIT event
    manager(sim, SimulationEvent.INIT)
    
    # Run 20 steps
    for i in range(20):
        sim.step()
        manager(sim, SimulationEvent.STEP_COMPLETE)
    
    # Trigger COMPLETE event
    manager(sim, SimulationEvent.COMPLETE)
    
    print("\n--- Test Complete ---\n")


if __name__ == "__main__":
    main()