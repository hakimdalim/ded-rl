# Debugging Guide for Simulation System

## 1. Understanding the Test 3D Bug

### The Issue
In `test_data_access()`, the code searches for `StepDataCollector` but `data_collector` stays `None`, even though we can see it's in the callback list.

### Root Cause
The `isinstance()` check is failing because of how the module is imported.

**In test file:**
```python
from callbacks.step_data_collector import StepDataCollector  # Full path import
```

**In CallbackManager:**
```python
from step_data_collector import StepDataCollector  # Relative import
```

These create **different class objects** in Python's module system, so `isinstance()` returns False!

### Fix Options

#### Option 1: Use duck typing (check for attributes)
```python
data_collector = None
for cb in manager.callbacks:
    if hasattr(cb, 'step_data') and hasattr(cb, 'current_step_dict'):
        data_collector = cb
        break
```

#### Option 2: Check by class name (string comparison)
```python
data_collector = None
for cb in manager.callbacks:
    if cb.__class__.__name__ == 'StepDataCollector':
        data_collector = cb
        break
```

#### Option 3: Add a helper method to CallbackManager
```python
def get_callback_by_type(self, callback_class):
    """Get first callback matching the given class."""
    for cb in self.callbacks:
        if isinstance(cb, callback_class):
            return cb
    return None
```

---

## 2. Python Debugging Techniques

### A. Using Print Debugging (Quick & Simple)

```python
# 1. Print variable values
print(f"DEBUG: data_collector = {data_collector}")
print(f"DEBUG: type = {type(data_collector)}")

# 2. Print with context
print(f"DEBUG [{__name__}]: Searching for StepDataCollector...")

# 3. Print all callback types
for i, cb in enumerate(manager.callbacks):
    print(f"  Callback {i}: {type(cb).__module__}.{type(cb).__name__}")
    print(f"    isinstance check: {isinstance(cb, StepDataCollector)}")
    print(f"    Module: {cb.__class__.__module__}")
```

### B. Using Python Debugger (pdb)

#### Basic Usage
```python
import pdb

# Add this line where you want to break
pdb.set_trace()  # Execution stops here

# Or use the new breakpoint() (Python 3.7+)
breakpoint()  # Cleaner syntax
```

#### Debugger Commands
```
(Pdb) h              # Help - show all commands
(Pdb) l              # List - show code around current line
(Pdb) n              # Next - execute next line
(Pdb) s              # Step - step into function calls
(Pdb) c              # Continue - continue execution
(Pdb) p variable     # Print - print variable value
(Pdb) pp variable    # Pretty print - formatted output
(Pdb) w              # Where - show stack trace
(Pdb) up             # Move up one stack frame
(Pdb) down           # Move down one stack frame
(Pdb) q              # Quit debugger
```

#### Example Debug Session for Test 3D
```python
def test_data_access():
    # ... setup code ...

    # Break right before the search
    breakpoint()

    data_collector = None
    for cb in manager.callbacks:
        if isinstance(cb, StepDataCollector):
            data_collector = cb
            break
```

Then in the debugger:
```
(Pdb) p manager.callbacks
[<callbacks.completion_callbacks.StepCountCompletionCallback>, ...]

(Pdb) p [type(cb).__name__ for cb in manager.callbacks]
['StepCountCompletionCallback', 'ErrorCompletionCallback', 'StepDataCollector']

(Pdb) cb = manager.callbacks[-1]  # Get last callback
(Pdb) p cb.__class__.__name__
'StepDataCollector'

(Pdb) p isinstance(cb, StepDataCollector)
False  # ⚠️ This is the bug!

(Pdb) p cb.__class__.__module__
'step_data_collector'

(Pdb) p StepDataCollector.__module__
'callbacks.step_data_collector'  # ⚠️ Different modules!
```

### C. Using VS Code Debugger (Recommended!)

#### Setup
1. Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${fileDirname}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Debug Test System",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/test_callbacks/test_complete_system.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/test_callbacks"
        }
    ]
}
```

#### Usage
1. Open the test file in VS Code
2. Click in the left margin (line number area) to set breakpoints
3. Press F5 or click "Run and Debug"
4. Use the debug toolbar:
   - Continue (F5)
   - Step Over (F10)
   - Step Into (F11)
   - Step Out (Shift+F11)
5. Hover over variables to see values
6. Use Debug Console to evaluate expressions

#### Conditional Breakpoints
- Right-click on a breakpoint
- Select "Edit Breakpoint" → "Conditional Breakpoint"
- Enter condition, e.g., `i == 10` or `data_collector is None`

### D. Logging (Production-Ready Debugging)

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('simulation_debug.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Use in code
logger.debug(f"Searching for StepDataCollector in {len(manager.callbacks)} callbacks")
logger.info(f"Found data collector: {data_collector}")
logger.warning("No StepDataCollector found!")
logger.error(f"Failed to process step: {e}")
```

### E. Assertions for Catching Bugs Early

```python
# Check assumptions
assert data_collector is not None, "StepDataCollector should be auto-added"
assert len(manager.callbacks) > 0, "Manager should have callbacks"

# Type checking
assert isinstance(sim, DummySimulation), f"Expected DummySimulation, got {type(sim)}"
```

---

## 3. Debugging the Callback System Specifically

### Understanding Exception-Based Flow Control

The confusing "[Callback Error]" messages are **intentional**:

```python
# In _callback_manager.py:105-118
try:
    callback(context)
except SimulationComplete:
    # This is GOOD - simulation completed normally
    raise  # Re-raise to let simulation handle it
except Exception as e:
    # This catches REAL errors
    # But it ALSO catches SimulationComplete because it's an Exception!
```

**The Problem**: `SimulationComplete` inherits from `Exception`, so it gets caught by the error handler first.

**Better Solution**: Check exception type order:
```python
try:
    callback(context)
except SimulationComplete:
    # Handle completion FIRST (more specific)
    raise
except Exception as e:
    # Then handle other errors
    # ... error handling code ...
```

Wait... looking at the code again, this IS the correct order! So why the error message?

**Ah!** The issue is in `completion_callbacks.py:36`:
```python
def _execute(self, context: dict) -> None:
    if self.is_complete(context):
        message = self.get_completion_message(context)
        raise SimulationComplete(message)
```

This `SimulationComplete` exception is raised INSIDE `callback(context)`, and it's being caught as a generic `Exception` on line 108, which adds the "[Callback Error]" prefix before re-raising.

### Debugging Strategy for Callbacks

1. **List all callbacks and their events**:
```python
print("\nCallback Configuration:")
for i, cb in enumerate(manager.callbacks):
    print(f"{i}. {cb.__class__.__name__}")
    print(f"   Events: {[e.name for e in cb.events]}")
    print(f"   Enabled: {cb.enabled}")
```

2. **Trace callback execution**:
```python
# Add to CallbackManager.__call__
print(f"\n--- Event: {event.name} ---")
for callback in self.callbacks:
    if callback.enabled and event in callback.events:
        print(f"  ▶ Executing: {callback.__class__.__name__}")
```

3. **Check callback matching**:
```python
# See which callbacks will fire for an event
active_callbacks = manager.get_callbacks_for_event(SimulationEvent.STEP_COMPLETE)
print(f"Active callbacks for STEP_COMPLETE: {[cb.__class__.__name__ for cb in active_callbacks]}")
```

---

## 4. Common Debugging Patterns

### Pattern 1: Bisection (Binary Search for Bugs)
Comment out half your code, see if bug persists, repeat.

### Pattern 2: Minimal Reproduction
Create the smallest possible test that shows the bug:
```python
def test_isinstance_bug():
    """Minimal test showing the isinstance issue"""
    from callbacks.step_data_collector import StepDataCollector as SDC1
    import sys
    sys.path.append('..')
    from step_data_collector import StepDataCollector as SDC2

    collector1 = SDC1()
    collector2 = SDC2()

    print(f"SDC1 module: {SDC1.__module__}")
    print(f"SDC2 module: {SDC2.__module__}")
    print(f"collector1 isinstance SDC1: {isinstance(collector1, SDC1)}")  # True
    print(f"collector1 isinstance SDC2: {isinstance(collector1, SDC2)}")  # False!
```

### Pattern 3: Rubber Duck Debugging
Explain the code line-by-line to someone (or something). Often you'll spot the bug while explaining.

### Pattern 4: Git Bisect (for regressions)
```bash
git bisect start
git bisect bad  # Current commit is broken
git bisect good abc123  # This old commit worked
# Git will check out commits for you to test
python test_complete_system.py
git bisect good  # or bad
# Repeat until git finds the breaking commit
```

---

## 5. Quick Reference: Debugging Checklist

When you hit a bug:

- [ ] Can you reproduce it consistently?
- [ ] What's the error message? (Read it carefully!)
- [ ] What line is failing? (Check the traceback)
- [ ] What are the variable values at that point?
- [ ] What did you expect vs. what actually happened?
- [ ] Does it fail in a simpler test case?
- [ ] Did it ever work? (Check git history)
- [ ] Have you tried turning it off and on again? 😄

---

## 6. Debugging Resources

- Python debugger docs: https://docs.python.org/3/library/pdb.html
- VS Code Python debugging: https://code.visualstudio.com/docs/python/debugging
- Logging tutorial: https://docs.python.org/3/howto/logging.html
- pdb cheatsheet: https://appletree.or.kr/quick_reference_cards/Python/Python%20Debugger%20Cheatsheet.pdf
