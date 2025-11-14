# How to Monitor Running Simulations

## Quick Answer: Is My Simulation Done?

### Method 1: Check Background Job Status (Fastest)

```bash
# In Python/terminal where you started it
# Look at the output - simulation will print:
"Simulation completed successfully!"
```

### Method 2: Check Process List

```bash
# Windows
tasklist | findstr python

# Check if python process is still running
# If no python.exe related to conda → simulation done
```

### Method 3: Check Output Files

```bash
# Look for completion marker files
ls _experiments/<your_label>/*/clad_manager.pkl

# If this file exists → simulation completed
```

---

## Monitoring Your Current Default Simulation

**Your simulation is currently RUNNING** ⏳

**Started**: ~20 minutes ago (as of this doc creation)
**Expected runtime**: 2-4 hours total
**Current status**: Still initializing/early steps

### Check Progress

#### Option 1: Check CSV File Line Count

The CSV file grows as the simulation runs:

```bash
# Count steps completed
wc -l _experiments/default_run/*/simulation_data.csv

# Or on Windows
find _experiments/default_run -name "simulation_data.csv" -exec wc -l {} \;
```

**Interpretation**:
- 1 line = header only (just started)
- 100 lines = ~99 steps completed
- 1000 lines = ~999 steps completed
- 2500-3500 lines = simulation should be nearly done

#### Option 2: Check Console Output

If you see console output, look for:

```
Layer (count):   1  |  Track (count):   1  |  Height (max): 0.3500 mm
Layer (count):   1  |  Track (count):   2  |  Height (max): 0.3500 mm
...
Layer (count):  14  |  Track (count):   8  |  Height (max): 5.0000 mm  ← Nearly done!
```

**Height progress**:
- Target: 5.0 mm (for 5mm part)
- Current height shown in console
- When current ≥ target → done!

#### Option 3: Check File Timestamps

```bash
# Windows
dir /O-D _experiments\default_run\*\*.png | more

# See if files are being actively created
# Recent timestamps = still running
```

---

## Background Job Monitoring (Python/Bash)

Since you're running in background, you have these options:

### Check if Process is Running

```bash
# Get process ID of your simulation
ps aux | grep "python.*simulate.py"

# Or on Windows
tasklist | findstr python
```

### Monitor Log Output

If you redirected output to a file:

```bash
# Follow log in real-time
tail -f simulation_log.txt

# Or on Windows
Get-Content simulation_log.txt -Wait
```

### Check Background Job Status

```bash
# List background jobs (if using job control)
jobs

# Check specific job
jobs -l
```

---

## Simulation Completion Indicators

### ✅ Signs Simulation is COMPLETE

1. **Console output shows**:
   ```
   Simulation completed successfully!
   Results saved to: _experiments/...
   ```

2. **Final files exist**:
   ```
   _experiments/default_run/.../
   ├─ clad_manager.pkl          ✓ Exists
   ├─ final_activated_vol.npy   ✓ Exists
   ├─ final_temperature_vol.npy ✓ Exists
   └─ cross_sections/           ✓ Contains .png files
   ```

3. **Python process is no longer running**
   ```bash
   # No output when searching for simulate.py
   ps aux | grep simulate.py
   ```

4. **CSV file has stopped growing**
   ```bash
   # Check file size, wait 5 min, check again
   ls -lh simulation_data.csv
   # If size unchanged → done
   ```

### ⏳ Signs Simulation is STILL RUNNING

1. **Console output shows**:
   ```
   Layer (count):   3  |  Track (count):   5  |  Height (max): 1.0500 mm
   # Still printing progress
   ```

2. **Python process exists**:
   ```bash
   tasklist | findstr python
   # Shows python.exe process
   ```

3. **Files are being created**:
   ```bash
   # Recent timestamps (within last few minutes)
   dir /O-D _experiments\default_run\*\thermal_plots\
   ```

4. **CSV file is growing**:
   ```bash
   # Line count increasing
   wc -l simulation_data.csv
   ```

### ❌ Signs Simulation FAILED/CRASHED

1. **Console shows error**:
   ```
   Traceback (most recent call last):
     File "simulate.py", line ...
   ImportError: ...
   ```

2. **Process not running but no completion message**
   - No python process
   - No "completed successfully" message
   - Incomplete output files

3. **CSV file stopped growing but height target not reached**
   ```
   # Last line shows height < target
   tail -1 simulation_data.csv
   ```

---

## Detailed Progress Monitoring

### Estimate Completion Time

**Formula**:
```
Total steps = (part height / layer spacing) × (part width / hatch spacing) × (track length / (scan speed × delta_t))

For default (5×5×5mm):
Layers = 5mm / 0.35mm ≈ 14 layers
Tracks per layer = 5mm / 0.7mm ≈ 7 tracks
Steps per track ≈ 25-30 steps

Total steps ≈ 14 × 7 × 27 ≈ 2646 steps
```

**Time estimate**:
```
If 1 step takes ~3-5 seconds:
2646 steps × 4 sec = 10,584 seconds ≈ 2.9 hours
```

### Calculate Percent Complete

```bash
# Count steps in CSV
CURRENT_STEPS=$(wc -l < simulation_data.csv)
CURRENT_STEPS=$((CURRENT_STEPS - 1))  # Subtract header

# Estimate total
TOTAL_STEPS=2646

# Calculate percentage
PERCENT=$((CURRENT_STEPS * 100 / TOTAL_STEPS))

echo "Progress: $PERCENT%"
```

### Monitor in Real-Time

Create a monitoring script:

```bash
# monitor_sim.sh
while true; do
  STEPS=$(wc -l < _experiments/default_run/*/simulation_data.csv 2>/dev/null || echo "1")
  STEPS=$((STEPS - 1))
  echo "$(date): Steps completed: $STEPS"
  sleep 300  # Check every 5 minutes
done
```

---

## Your Current Simulation Status

**Simulation ID**: `867a78` (background job)
**Label**: `default_run`
**Configuration**: 5×5×5mm part, 200μm voxels

### Expected Timeline

```
Time from start    Status                              Progress
─────────────────────────────────────────────────────────────────
0-5 min            Initialization                      0%
5-30 min           First few tracks                    1-5%
30 min-1 hour      Layer 1-2 complete                  10-20%
1-2 hours          Layers 3-7 complete                 30-50%
2-3 hours          Layers 8-12 complete                60-85%
3-4 hours          Final layers, completion            90-100%
```

**Current status**: Simulation has been running ~20 minutes
**Estimated current progress**: 1-5%
**Estimated time remaining**: 2-3.5 hours

### Check Current Progress Now

```bash
# Quick check
find _experiments/default_run -name "simulation_data.csv" -exec wc -l {} \;

# Shows number of lines (steps + 1 for header)
# If output is:
#   1     → Just started
#   50    → ~49 steps (2-3%)
#   500   → ~499 steps (20%)
#   1000  → ~999 steps (40%)
#   2500  → ~2499 steps (95%)
```

---

## What to Do While Waiting

### Option 1: Let it Run
- Close terminal (simulation continues in background)
- Check back in 2-3 hours
- Look for completion indicators

### Option 2: Monitor Periodically
```bash
# Check every 30 minutes
watch -n 1800 'wc -l _experiments/default_run/*/simulation_data.csv'
```

### Option 3: Set Up Notification

**Windows PowerShell**:
```powershell
# Wait for completion, then show notification
while (Get-Process -Name python -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}
# Simulation done
Write-Host "SIMULATION COMPLETE!" -ForegroundColor Green
```

**Email notification** (if configured):
```python
# Add to end of simulate.py
import smtplib
send_email("Simulation complete!", "Your DED sim finished.")
```

---

## After Simulation Completes

### Verify Successful Completion

1. **Check for completion message**:
   ```
   Simulation completed successfully!
   ```

2. **Verify output files exist**:
   ```bash
   ls _experiments/default_run/*/
   # Should see:
   # - simulation_data.csv
   # - clad_manager.pkl
   # - final_*.npy files
   # - thermal_plots/
   # - cross_sections/
   # - etc.
   ```

3. **Check CSV completeness**:
   ```bash
   tail -5 simulation_data.csv
   # Last rows should show height ≥ 5.0mm
   ```

4. **Verify file sizes are reasonable**:
   ```bash
   du -sh _experiments/default_run/*
   # Should be 500MB - 2GB total
   ```

---

## Troubleshooting

### Simulation Seems Stuck

**Symptoms**: No new files for >30 minutes

**Check**:
```bash
# Is process still running?
tasklist | findstr python

# Check CPU usage
# If 0% CPU for >10 min → might be stuck
```

**Action**: Kill and restart

### Simulation Running Too Long

**Expected**: 2-4 hours for default
**If running >6 hours**: Might be issue

**Check**:
```bash
# How many steps completed?
wc -l simulation_data.csv

# If <100 steps after 6 hours → problem
```

### Out of Memory

**Symptoms**: Process killed, no error message

**Check**: Windows Event Viewer for crash logs

**Solution**:
- Reduce voxel resolution (300μm instead of 200μm)
- Reduce save frequency (interval=10 instead of interval=1)
- Close other applications

---

## Quick Reference Commands

```bash
# ============================================
# QUICK STATUS CHECK
# ============================================

# How many steps completed?
wc -l _experiments/default_run/*/simulation_data.csv

# Is simulation still running?
tasklist | findstr python

# When was last file created?
dir /O-D _experiments\default_run\*\thermal_plots | more

# Check current height (last CSV line)
tail -1 _experiments/default_run/*/simulation_data.csv | cut -d',' -f4

# ============================================
# ESTIMATED TIME REMAINING
# ============================================

CURRENT_STEPS=$(wc -l < simulation_data.csv)
CURRENT_STEPS=$((CURRENT_STEPS - 1))
TOTAL_STEPS=2646
REMAINING=$((TOTAL_STEPS - CURRENT_STEPS))
TIME_PER_STEP=4  # seconds
REMAINING_TIME=$((REMAINING * TIME_PER_STEP / 3600))  # hours

echo "Estimated time remaining: $REMAINING_TIME hours"
```

---

## Summary: How to Know When Done

### The Simplest Way

**Just check if this file exists**:

```bash
ls _experiments/default_run/*/clad_manager.pkl
```

- ✅ **If exists** → Simulation complete!
- ❌ **If not exists** → Still running or failed

### The Most Reliable Way

**Check console for**:
```
Simulation completed successfully!
```

That's the definitive completion indicator!

---

## Your Next Steps

1. **Wait 2-4 hours** (simulation is long!)
2. **Check progress every 30-60 minutes** using:
   ```bash
   wc -l _experiments/default_run/*/simulation_data.csv
   ```
3. **When done**, you'll see:
   - `Simulation completed successfully!`
   - `clad_manager.pkl` file exists
   - All output directories populated

4. **Then explore results** using the guides I created:
   - `README.md` - How to use outputs
   - `CAMERA_VS_STANDARD_COMPARISON.md` - Visual comparison

**Current status**: Your simulation is running and making progress! ⏳
**Check back in**: ~2-3 hours 🕐
