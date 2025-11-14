# 🧪 CALLBACK SYSTEM TESTING GUIDE

Complete guide for testing your callback system to ensure everything works correctly.

---

## Quick Start

```bash
# 1. Test all callbacks with DummySimulation (fast, no display needed)
cd test_callbacks
python test_all_callbacks.py

# 2. Test with real simulate.py (slow, full simulation)
python test_simulate_dry_run.py

# 3. Run existing callback tests
python test_complete_system.py
```

---

## Test Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    TESTING HIERARCHY                        │
└─────────────────────────────────────────────────────────────┘

Level 1: Unit Tests (test_callbacks/)
  ├── test_simple.py              - Basic callback mechanics
  ├── test_auto_additions.py      - CallbackManager auto-add
  ├── test_complete_system.py     - Integration tests
  └── test_all_callbacks.py       - ALL 19 callbacks ⭐ NEW

Level 2: Integration Tests
  └── test_simulate_dry_run.py    - Real simulation tests ⭐ NEW

Level 3: Production Run
  └── simulate.py                 - Full production simulation
```

---

## Level 1: Dry-Run Tests (Fast)

### Test 1: All Callbacks (`test_all_callbacks.py`)

**What it tests:** All 19 callback types with DummySimulation

**Run time:** ~30 seconds

**Usage:**
```bash
cd test_callbacks
python test_all_callbacks.py
```

**Expected output:**
```
============================================================
COMPREHENSIVE CALLBACK SYSTEM DRY-RUN TEST
============================================================

Testing 19 callback types across 7 categories...

████████████████████████████████████████████████████████
CATEGORY A: COMPLETION CALLBACKS (4)
████████████████████████████████████████████████████████

============================================================
TEST: Height Completion Callback
============================================================
✓ Stopped at target height: Build complete: reached...

[... more tests ...]

============================================================
TEST SUITE COMPLETE!
============================================================

Summary:
  ✓ Completion callbacks (4/4 tested)
  ✓ Data collection (2/2 tested)
  ✓ Visualization (2/4 tested - display callbacks skipped)
  ✓ Saving/Export (5/5 tested)
  ⊗ Control (0/1 tested - requires real simulation)
  ✓ Monitoring (2/2 tested)
  ✓ Utilities (1/1 tested)
  ✓ Combined test (multiple callbacks together)

Total: 16/19 callback types tested

Check ./test_output/ directory for generated files!
```

**What to check:**
1. All tests show `✓` (success)
2. `test_output/` directory created with subdirectories
3. Files saved correctly (CSV, NPY, PNG, STL, etc.)

---

### Test 2: Complete System (`test_complete_system.py`)

**What it tests:** Real-world usage patterns (RL, height completion, data access)

**Run time:** ~10 seconds

**Usage:**
```bash
cd test_callbacks
python test_complete_system.py
```

**Expected output:**
```
============================================================
TEST 3A: Step Count Completion
============================================================
Run simulation for exactly 50 steps

✓ Simulation stopped:
✓ Collected 49 steps of data

[... more tests ...]

============================================================
ALL TESTS PASSED! ✓
============================================================
```

---

## Level 2: Integration Tests with Real Simulation

### Test 3: Real Simulation Dry-Run (`test_simulate_dry_run.py`)

**What it tests:** Real `simulate.py` with actual physics simulation

**Run time:** ~2-5 minutes (depends on hardware)

**Usage:**
```bash
python test_callbacks/test_simulate_dry_run.py
```

**What it does:**
- Runs VERY small simulation (tiny part, few steps)
- Tests with different callback combinations
- Validates output files
- Tests error handling

**Expected behavior:**
- Creates output directory in `_experiments/test_*`
- Generates all expected files
- Prints progress to console
- Completes without errors

---

## Level 3: Production Testing

### Test 4: Full Production Run

**What it tests:** Complete simulation with all features

**Run time:** 10 minutes to hours (depends on parameters)

**Minimal test (fast):**
```bash
python simulate.py \
  --build-x 5.0 --build-y 5.0 --build-z 3.0 \
  --part-x 2.0 --part-y 2.0 --part-z 1.0 \
  --voxel-size 200.0 \
  --exp-label "test_minimal"
```

**Medium test (thorough):**
```bash
python simulate.py \
  --build-x 10.0 --build-y 10.0 --build-z 5.0 \
  --part-x 5.0 --part-y 5.0 --part-z 2.0 \
  --voxel-size 150.0 \
  --scan-speed 3.0 \
  --laser-power 600.0 \
  --powder-feed 2.0 \
  --exp-label "test_medium"
```

**Expected output structure:**
```
_experiments/
  └── test_minimal/
      └── job<timestamp>_build5.0x5.0x3.0mm_part2.0x2.0x1.0mm.../
          ├── simulation_data.csv
          ├── parameter_history.csv
          ├── final_activated_vol.npy
          ├── final_temperature_vol.npy
          ├── simulation_params.csv
          ├── clad_manager.pkl
          ├── temperatures.zip
          ├── thermal_plots/
          │   ├── thermal0001_top_view.png
          │   ├── thermal0001_front_view.png
          │   ├── ...
          ├── build_mesh/
          │   ├── build_state_step0001.stl
          │   ├── ...
          └── cross_sections/
              ├── cross_section_y0.5mm.png
              ├── ...
```

---

## Debugging Failed Tests

### Problem 1: Import Errors

```
ModuleNotFoundError: No module named 'callbacks'
```

**Solution:**
```bash
# Make sure you're in the right directory
cd test_callbacks  # For test files
# OR
cd ..  # For simulate.py

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%  # Windows CMD
$env:PYTHONPATH += ";$(pwd)"  # Windows PowerShell
```

---

### Problem 2: Display Errors (matplotlib)

```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Solution:**
```python
# For tests without display (AdvancedLivePlotter, LivePlotter)
# These are SKIPPED in test_all_callbacks.py

# To test manually:
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

### Problem 3: File Permission Errors

```
PermissionError: [Errno 13] Permission denied: './test_output/...'
```

**Solution:**
```bash
# Clean up locked files
rm -rf test_output/  # Linux/Mac
rmdir /s test_output  # Windows

# Or run as admin/sudo if needed
```

---

### Problem 4: Callback Not Found

```
❌ Could not find StepDataCollector
```

**Solution:**
Use duck typing instead of isinstance:
```python
# ❌ WRONG (fails due to import paths)
if isinstance(cb, StepDataCollector):
    data_collector = cb

# ✓ RIGHT (duck typing)
if hasattr(cb, 'step_data') and hasattr(cb, 'current_step_dict'):
    data_collector = cb
```

---

### Problem 5: Tests Pass But No Files Created

**Check these:**

1. **Output directory exists?**
   ```python
   print(f"Output dir: {sim.output_dir}")
   assert Path(sim.output_dir).exists()
   ```

2. **Callbacks executed?**
   ```python
   for cb in manager.callbacks:
       print(f"{cb.__class__.__name__}: {cb._event_counts}")
   ```

3. **Event triggered?**
   ```python
   # Make sure you call the COMPLETE event!
   manager(sim, SimulationEvent.COMPLETE)
   ```

---

## Custom Test Template

Create your own test for a specific use case:

```python
"""
test_my_custom_scenario.py - Test my specific use case
"""
import sys
sys.path.append('..')

from dummy_simulation import DummySimulation
from callbacks._callback_manager import CallbackManager
from callbacks._base_callbacks import SimulationEvent
from callbacks.completion_callbacks import StepCountCompletionCallback, SimulationComplete
from callbacks.step_data_collector import StepDataCollector

def test_my_scenario():
    """Test my specific scenario."""
    print("Testing my scenario...")

    # Setup
    sim = DummySimulation(output_dir="./test_output/my_scenario")
    callbacks = [
        StepCountCompletionCallback(max_steps=100),
        StepDataCollector(tracked_fields=['position', 'melt_pool'])
    ]
    manager = CallbackManager(callbacks)

    # Initialize
    manager(sim, SimulationEvent.INIT)

    # Run simulation
    try:
        for i in range(200):
            sim.step()
            manager(sim, SimulationEvent.STEP_COMPLETE)

            # Your custom logic here
            # ...

    except SimulationComplete as e:
        print(f"✓ Completed: {e}")

    # Cleanup
    manager(sim, SimulationEvent.COMPLETE)
    sim.complete()

    # Verify results
    # ... your assertions here ...

    print("✓ Test passed!")

if __name__ == "__main__":
    test_my_scenario()
```

---

## Verification Checklist

After running tests, verify:

### ✅ Basic Functionality
- [ ] All tests pass without errors
- [ ] Output directories created
- [ ] Console shows progress messages

### ✅ Data Collection
- [ ] CSV files exist and have data
- [ ] Data contains expected columns
- [ ] Row counts match step counts

### ✅ File Outputs
- [ ] NPY files (temperature, voxel data)
- [ ] PNG files (plots, visualizations)
- [ ] STL files (meshes)
- [ ] Pickle files (serialized objects)
- [ ] ZIP files (compressed archives)

### ✅ Completion Logic
- [ ] Simulations stop at correct conditions
- [ ] SimulationComplete exception raised
- [ ] Cleanup callbacks execute

### ✅ Error Handling
- [ ] Errors don't crash simulation
- [ ] ErrorCompletionCallback triggers
- [ ] COMPLETE event fires even on error

---

## Performance Benchmarking

Check callback overhead:

```python
import time

# Without callbacks
start = time.time()
sim = DummySimulation()
for i in range(1000):
    sim.step()
baseline = time.time() - start
print(f"Baseline: {baseline:.2f}s")

# With callbacks
start = time.time()
sim = DummySimulation()
callbacks = [StepDataCollector(), ProgressPrinter()]
manager = CallbackManager(callbacks)
for i in range(1000):
    sim.step()
    manager(sim, SimulationEvent.STEP_COMPLETE)
with_callbacks = time.time() - start
print(f"With callbacks: {with_callbacks:.2f}s")
print(f"Overhead: {(with_callbacks/baseline - 1)*100:.1f}%")
```

**Expected overhead:** < 5% for most callbacks

**High overhead callbacks:**
- `AdvancedLivePlotter` (plotting is slow)
- `ThermalPlotSaver` (matplotlib rendering)
- `MeshSaver` (STL generation)

**Tip:** Use `interval` parameter to reduce overhead:
```python
ThermalPlotSaver(interval=10)  # Only save every 10 steps
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Callbacks

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install numpy pandas matplotlib

    - name: Run callback tests
      run: |
        cd test_callbacks
        python test_all_callbacks.py

    - name: Check outputs
      run: |
        ls -R test_output/
```

---

## Summary: Recommended Testing Order

**1. Quick sanity check (30 seconds):**
```bash
cd test_callbacks && python test_complete_system.py
```

**2. Comprehensive callback test (1 minute):**
```bash
cd test_callbacks && python test_all_callbacks.py
```

**3. Real simulation dry-run (5 minutes):**
```bash
python test_callbacks/test_simulate_dry_run.py
```

**4. Small production test (10 minutes):**
```bash
python simulate.py --part-x 2.0 --part-y 2.0 --part-z 1.0 --exp-label "test"
```

**5. Full production run (hours):**
```bash
python simulate.py --exp-label "production_run_v1"
```

---

## Need Help?

**Common Issues:**
1. Import errors → Check PYTHONPATH and current directory
2. Display errors → Use matplotlib 'Agg' backend
3. File not found → Check output_dir and Path construction
4. Callback not executing → Verify event subscribed and enabled
5. Tests fail → Check git status for uncommitted changes

**Debug mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use breakpoint
breakpoint()  # Debugger stops here
```

**Get help:**
- Check `DEBUGGING_GUIDE.md` for debugging techniques
- Check `CALLBACK_SYSTEM_GUIDE.md` for callback reference
- Use `test_all_callbacks.py` as template for custom tests

---

**Happy Testing! 🧪**
