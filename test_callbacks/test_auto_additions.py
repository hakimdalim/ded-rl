#test_auto_additions.py
"""
TEST 2: Auto-Addition Test
This tests the _validate_and_configure_callbacks function.
It shows what happens when you don't provide certain callbacks.
"""

import sys
sys.path.append('..')

from dummy_simulation import DummySimulation
from callbacks._callback_manager import CallbackManager
from callbacks.completion_callbacks import StepCountCompletionCallback


def test_empty_callbacks():
    """Test 1: Empty callback list"""
    print("\n" + "="*60)
    print("TEST 2A: Empty Callback List")
    print("="*60)
    print("What happens when you provide NO callbacks?\n")
    
    # Create manager with NO callbacks
    manager = CallbackManager(callbacks=[])
    
    print(f"\nResult: Manager has {len(manager.callbacks)} callbacks")
    for i, cb in enumerate(manager.callbacks, 1):
        print(f"  {i}. {cb.__class__.__name__}")
    
    print("\n✓ Expected: ErrorCompletionCallback and StepDataCollector auto-added")
    print("⚠️  Warnipython ng should appear about missing completion callback")


def test_only_completion():
    """Test 2: Only completion callback"""
    print("\n" + "="*60)
    print("TEST 2B: Only Completion Callback")
    print("="*60)
    print("What happens when you only provide a completion callback?\n")
    
    # Create manager with only completion
    callbacks = [
        StepCountCompletionCallback(max_steps=100)
    ]
    manager = CallbackManager(callbacks)
    
    print(f"\nResult: Manager has {len(manager.callbacks)} callbacks")
    for i, cb in enumerate(manager.callbacks, 1):
        print(f"  {i}. {cb.__class__.__name__}")
    
    print("\n✓ Expected: ErrorCompletionCallback and StepDataCollector auto-added")
    print("✓ No warning because completion callback exists")


# def test_complete_callbacks():
#     """Test 3: Complete callback set"""
#     print("\n" + "="*60)
#     print("TEST 2C: Complete Callback Set")
#     print("="*60)
#     print("What happens when you provide all required callbacks?\n")
    
#     from callbacks.completion_callbacks import HeightCompletionCallback
#     from callbacks.step_data_collector import StepDataCollector
#     from callbacks.error_callbacks import ErrorCompletionCallback
    
#     # Provide all required callbacks
#     callbacks = [
#         HeightCompletionCallback(target_height=0.002),
#         StepDataCollector(tracked_fields=None, save_path="data.csv"),
#         ErrorCompletionCallback(),
#     ]
#     manager = CallbackManager(callbacks)
    
#     print(f"\nResult: Manager has {len(manager.callbacks)} callbacks")
#     for i, cb in enumerate(manager.callbacks, 1):
#         print(f"  {i}. {cb.__class__.__name__}")
    
#     print("\n✓ Expected: Nothing auto-added, no warnings")

def test_complete_callbacks():
    from callbacks.completion_callbacks import HeightCompletionCallback
    from callbacks.step_data_collector import StepDataCollector
    from callbacks.error_callbacks import ErrorCompletionCallback
    print("TEST 2C: Detailed Inspection")
    print("="*60)
    print("Let's see exactly what the test_complete_callbacks do:\n")
    # Provide all required callbacks
    error_cb = ErrorCompletionCallback()
    data_cb = StepDataCollector(tracked_fields=None, save_path="data.csv")
    height_cb = HeightCompletionCallback(target_height=0.002)
    
    print(f"Created error callback: {type(error_cb)}")
    print(f"Is ErrorCompletionCallback? {isinstance(error_cb, ErrorCompletionCallback)}")
    
    callbacks = [height_cb, data_cb, error_cb]
    
    # Check BEFORE creating manager
    print(f"Callbacks before manager: {len(callbacks)}")
    
    manager = CallbackManager(callbacks)
    
    # Check AFTER
    print(f"Callbacks after manager: {len(manager.callbacks)}")

def test_callback_list_inspection():
    """Test 4: Inspect what got auto-added"""
    print("\n" + "="*60)
    print("TEST 2D: Detailed Inspection")
    print("="*60)
    print("Let's see exactly what the auto-added callbacks do:\n")
    
    manager = CallbackManager(callbacks=[])
    
    print("Auto-added callbacks:")
    for cb in manager.callbacks:
        print(f"\n  {cb.__class__.__name__}:")
        print(f"    Events: {[e.name for e in cb.events]}")
        print(f"    Enabled: {cb.enabled}")
        
        # Check specific properties
        if hasattr(cb, 'tracked_fields'):
            print(f"    Tracked fields: {cb.tracked_fields}")
        if hasattr(cb, 'save_path'):
            print(f"    Save path: {cb.save_path}")


def main():
    """Run all tests"""
    test_empty_callbacks()
    test_only_completion()
    test_complete_callbacks()
    test_callback_list_inspection()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()