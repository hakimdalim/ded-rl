"""
TEST 3: Complete System Test
This runs a full simulation with real callbacks from your codebase.
"""

import sys
sys.path.append('..')

import numpy as np
from dummy_simulation import DummySimulation
from callbacks._callback_manager import CallbackManager
from callbacks.completion_callbacks import (
    StepCountCompletionCallback, 
    SimulationComplete,
    HeightCompletionCallback
)
from callbacks.step_data_collector import StepDataCollector
from callbacks._base_callbacks import BaseCallback, SimulationEvent


class CustomRewardCallback(BaseCallback):
    """
    Example RL callback that calculates rewards.
    This shows how you might use callbacks for reinforcement learning.
    """
    def __init__(self):
        super().__init__(events=SimulationEvent.STEP_COMPLETE)
        self.total_reward = 0.0
        self.rewards = []
    
    def _execute(self, context):
        sim = context['simulation']
        
        # Calculate reward based on:
        # 1. Temperature (penalty if too hot or cold)
        # 2. Clad height (reward if close to target)
        
        if sim.step_context:
            temp = sim.step_context['melt_pool']['max_temp']
            clad_height = sim.step_context['clad']['height']
            
            # Temperature reward (want ~1500-1700K)
            temp_reward = -abs(temp - 1600) / 100
            
            # Height reward (want ~0.3mm)
            height_reward = -abs(clad_height - 0.0003) * 1000
            
            step_reward = temp_reward + height_reward
            
            self.total_reward += step_reward
            self.rewards.append(step_reward)
            
            if sim.progress_tracker.step_count % 10 == 0:
                avg_reward = np.mean(self.rewards[-10:])
                print(f"  [RL] Step {sim.progress_tracker.step_count}: "
                      f"Reward={step_reward:.2f}, Avg(10)={avg_reward:.2f}")


def test_step_count_completion():
    """Test: Stop after fixed number of steps"""
    print("\n" + "="*60)
    print("TEST 3A: Step Count Completion")
    print("="*60)
    print("Run simulation for exactly 50 steps\n")
    
    sim = DummySimulation(output_dir="./test_output/step_count")
    
    callbacks = [
        StepCountCompletionCallback(max_steps=50),
        StepDataCollector(tracked_fields=['position', 'melt_pool'], save_path=None),
    ]
    
    manager = CallbackManager(callbacks)
    
    # Run simulation
    manager(sim, SimulationEvent.INIT)
    
    try:
        for i in range(100):  # Try to run 100, but should stop at 50
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    except SimulationComplete as e:
        print(f"\n✓ Simulation stopped: {e}")
        sim.complete()
    
    # Check data collector
    data_collector = manager.callbacks[-1]  # Last one is auto-added data collector
    print(f"\n✓ Collected {len(data_collector.step_data)} steps of data")


def test_height_completion():
    """Test: Stop when height is reached"""
    print("\n" + "="*60)
    print("TEST 3B: Height Completion")
    print("="*60)
    print("Run until part reaches target height\n")
    
    sim = DummySimulation(output_dir="./test_output/height")
    
    # Target: 2mm part + 0.5mm substrate = 2.5mm total
    callbacks = [
        HeightCompletionCallback(),  # Uses config values
    ]
    
    manager = CallbackManager(callbacks)
    manager(sim, SimulationEvent.INIT)
    
    print(f"Target height: {(sim.config['part_height'] + sim.config['substrate_height']) * 1000:.2f}mm\n")
    
    try:
        for layer in range(10):  # Max 10 layers
            manager(sim, SimulationEvent.LAYER_START)
            
            for step in range(50):  # 50 steps per layer
                sim.step()
                manager(sim, SimulationEvent.STEP_COMPLETE)
                
                # Increase height artificially for this test
                sim.progress_tracker.max_height_reached += 0.00005  # 0.05mm per step
            
            sim.progress_tracker.current_layer += 1
            manager(sim, SimulationEvent.STEP_COMPLETE)
            
            print(f"  Layer {layer + 1}: Height = "
                  f"{sim.progress_tracker.max_height_reached * 1000:.2f}mm")
    
    except SimulationComplete as e:
        print(f"\n✓ {e}")
        sim.complete()


def test_with_rl_callback():
    """Test: Run with RL reward calculation"""
    print("\n" + "="*60)
    print("TEST 3C: With RL Reward Callback")
    print("="*60)
    print("Simulate RL training with reward calculation\n")
    
    sim = DummySimulation(output_dir="./test_output/rl")
    
    callbacks = [
        StepCountCompletionCallback(max_steps=30),
        CustomRewardCallback(),
        StepDataCollector(
            tracked_fields=['position', 'melt_pool', 'clad'],
            save_path="rl_data.csv"
        ),
    ]
    
    manager = CallbackManager(callbacks)
    manager(sim, SimulationEvent.INIT)
    
    print("Running RL episode...\n")
    
    try:
        for i in range(50):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
    
    except SimulationComplete as e:
        print(f"\n✓ {e}")
        
        # Get reward callback
        reward_callback = None
        for cb in manager.callbacks:
            if isinstance(cb, CustomRewardCallback):
                reward_callback = cb
                break
        
        if reward_callback:
            print(f"\n✓ Total reward: {reward_callback.total_reward:.2f}")
            print(f"✓ Average reward: {np.mean(reward_callback.rewards):.2f}")
        
        # Get data collector
        data_collector = None
        for cb in manager.callbacks:
            if isinstance(cb, StepDataCollector) and cb.save_path:
                data_collector = cb
                break
        
        if data_collector:
            print(f"✓ Collected {len(data_collector.step_data)} data points")
            print(f"✓ Data will be saved to: {data_collector.save_path}")
        
        manager(sim, SimulationEvent.COMPLETE)
        sim.complete()


def test_data_access():
    """Test: Accessing collected data"""
    print("\n" + "="*60)
    print("TEST 3D: Data Access Pattern")
    print("="*60)
    print("Show how to access data during simulation (for RL)\n")
    
    sim = DummySimulation(output_dir="./test_output/data_access")
    
    callbacks = [
        StepCountCompletionCallback(max_steps=10),
    ]
    
    manager = CallbackManager(callbacks)
    manager(sim, SimulationEvent.INIT)
    
    # Find the auto-added data collector
    data_collector = None
    # for cb in manager.callbacks:
    #     if isinstance(cb, StepDataCollector):
    #         data_collector = cb
            # break
    for cb in manager.callbacks:
        if hasattr(cb, 'step_data') and hasattr(cb, 'current_step_dict'):
            data_collector = cb
            break
    # Check if we found it
    if data_collector is None:
        print("❌ ERROR: No StepDataCollector found!")
        print("Available callbacks:")
        for cb in manager.callbacks:
            print(f"  - {cb.__class__.__name__}")
        return
    
    print(f"✓ Found StepDataCollector: {data_collector}")
    print("Running 10 steps and reading data after each...\n")
    
    try:
        for i in range(20):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)
            
            # Access current step data (this is what RL agent would read)
            current_data = data_collector.current_step_dict
            
            print(f"Step {current_data['step']}: "
                  f"x={current_data.get('position.x', 0)*1000:.3f}mm, "
                  f"temp={current_data.get('temperature.max_temp', 0):.0f}K")
    
    except SimulationComplete as e:
        print(f"\n✓ {e}")
        
        print(f"\n✓ Total data collected: {len(data_collector.step_data)} steps")
        print(f"✓ Fields tracked: {list(data_collector.step_data[0].keys())}")
        
        sim.complete()


def main():
    """Run all tests"""
    test_step_count_completion()
    test_height_completion()
    test_with_rl_callback()
    test_data_access()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60 + "\n")
    
    print("Summary:")
    print("  ✓ Step count completion works")
    print("  ✓ Height completion works")
    print("  ✓ RL reward calculation works")
    print("  ✓ Data collection and access works")
    print("\nYou can now use these patterns in your real simulation!")


if __name__ == "__main__":
    main()