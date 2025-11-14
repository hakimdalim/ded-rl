# Camera Resolution vs Voxel Resolution

## TL;DR Answer to Your Questions

**Q1: What do you lose with quick tests vs high resolution?**
- Quick tests lose: fine geometry, sharp thermal gradients, accurate melt pool shape
- Quick tests retain: overall part shape, layer structure, general behavior
- See detailed comparison below

**Q2: Is camera output affected by voxel resolution?**
- **NO! Camera resolution is INDEPENDENT of voxel resolution**
- Camera resolution = output image quality (pixels in PNG)
- Voxel resolution = simulation accuracy (thermal field, geometry)
- You can have high-res camera with low-res simulation (and vice versa)

---

## Key Concept: Two Independent Resolutions

### 1. **Voxel Resolution** (Simulation Accuracy)
Controls how accurately the physics is computed:

```python
voxel_size_um=200.0  # Simulation uses 200μm voxels
```

- **Affects**: Thermal field accuracy, melt pool geometry, clad profile
- **Trade-off**: Smaller voxels = more accurate BUT slower runtime
- **Examples**:
  - Quick test: 300μm voxels → coarse simulation, fast (minutes)
  - High-res: 100μm voxels → accurate simulation, slow (hours)

### 2. **Camera Resolution** (Image Quality)
Controls the output image pixels:

```python
resolution_wh=(60, 40)    # Camera outputs 60×40 pixel images
resolution_wh=(1920, 1080)  # Camera outputs 1920×1080 HD images
```

- **Affects**: PNG image quality, visual sharpness
- **Trade-off**: More pixels = larger PNG files, slightly slower rendering
- **Examples**:
  - Default: 60×40 pixels → small images, fast
  - HD: 1920×1080 pixels → publication-quality images

---

## How They Work Together

### Camera Renders What Simulation Computed

```
┌─────────────────────────────────────────────┐
│  SIMULATION (Voxel Resolution)              │
│  Computes thermal field on voxel grid       │
│  200μm voxels = blocky thermal boundaries   │
│  100μm voxels = smooth thermal boundaries   │
└──────────────────┬──────────────────────────┘
                   │
                   ↓ (camera views this)
┌─────────────────────────────────────────────┐
│  CAMERA (Image Resolution)                  │
│  Renders what it sees into pixels           │
│  60×40 = low-res image of simulation        │
│  1920×1080 = high-res image of simulation   │
└─────────────────────────────────────────────┘
```

**Example 1: Low Voxel, Low Camera**
```python
voxel_size_um=300,           # Blocky thermal field
resolution_wh=(60, 40)       # Small image
```
→ **Result**: Blocky simulation shown in low-res image (FAST)

**Example 2: Low Voxel, High Camera**
```python
voxel_size_um=300,           # Blocky thermal field
resolution_wh=(1920, 1080)   # HD image
```
→ **Result**: Blocky simulation shown in crisp HD image (you'll see the blockiness clearly!)

**Example 3: High Voxel, Low Camera**
```python
voxel_size_um=100,           # Smooth thermal field
resolution_wh=(60, 40)       # Small image
```
→ **Result**: Smooth simulation shown in low-res image (SLOW simulation, image doesn't show detail)

**Example 4: High Voxel, High Camera**
```python
voxel_size_um=100,           # Smooth thermal field
resolution_wh=(1920, 1080)   # HD image
```
→ **Result**: Smooth simulation shown in crisp HD image (SLOW but beautiful!)

---

## Quick Test vs High Resolution: What You Lose/Gain

### Quick Test Configuration
```python
python simulate.py --part-x 2.0 --part-y 2.0 --part-z 1.0 --voxel-size 300 --exp-label quick_test
```

**Voxel Resolution**: 300 μm
**Runtime**: 5-15 minutes
**Memory**: ~50 MB

#### What You LOSE:
1. **Geometric Accuracy**
   - Track edges are stepped/blocky (±300-600 μm error)
   - Overlap zones poorly defined
   - Surface appears faceted

2. **Thermal Detail**
   - Temperature gradients are blurred
   - Melt pool boundary is ~2-3 voxels wide (fuzzy)
   - Cooling rate accuracy reduced

3. **Melt Pool Morphology**
   - Shape is approximated (looks like rounded rectangle)
   - Depth measurement has ±150 μm error
   - Cannot resolve keyhole vs conduction mode differences

4. **Resolution Warnings**
   ```
   Low resolution across hatch spacing. Current configuration has 2.3 points
   per hatch spacing (recommended minimum: 5).
   ```

#### What You RETAIN:
1. **Overall Behavior**
   - Part dimensions are correct (±1 voxel)
   - Number of tracks/layers is accurate
   - Total build time is correct

2. **Qualitative Physics**
   - Melt pool exists in roughly correct location
   - Heat accumulation trends are visible
   - Layer-to-layer interactions are captured

3. **Parameter Sensitivity**
   - Trends with laser power changes are correct
   - Scan speed effects are qualitatively right
   - Powder feed rate impact is captured

#### Use Cases for Quick Tests:
- ✓ Initial debugging (code works?)
- ✓ Parameter space exploration (which direction to go?)
- ✓ Sanity checks (does it make physical sense?)
- ✓ Teaching demonstrations (show concepts)
- ✗ Quantitative predictions (not accurate enough)
- ✗ Publication-quality results (too coarse)

---

### High-Resolution Configuration
```python
python simulate.py --part-x 5.0 --part-y 5.0 --part-z 5.0 --voxel-size 100 --exp-label high_res
```

**Voxel Resolution**: 100 μm
**Runtime**: 10-20 hours
**Memory**: ~500 MB

#### What You GAIN:
1. **Geometric Accuracy**
   - Smooth track profiles (parabolic/Gaussian)
   - Accurate overlap calculations
   - Surface roughness captured (±50 μm)

2. **Thermal Detail**
   - Sharp temperature gradients
   - Precise melt pool boundary (1-2 voxels = 100-200 μm)
   - Accurate cooling rates

3. **Melt Pool Morphology**
   - Realistic elliptical/teardrop shape
   - Accurate depth (±50 μm)
   - Can distinguish keyhole vs conduction

4. **Physical Fidelity**
   - Heat-affected zone (HAZ) properly resolved
   - Inter-layer remelting depth accurate
   - Porosity/defect predictions more reliable

#### Use Cases for High-Resolution:
- ✓ Final production simulations
- ✓ Quantitative predictions (for experiments)
- ✓ Publication-quality results
- ✓ Defect/porosity analysis
- ✓ Microstructure input data (cooling rates)
- ✗ Quick parameter screening (too slow)

---

## Camera Resolution: When to Change It

### Default Camera Settings
```python
PerspectiveCameraCallback(
    plane_size=(0.06, 0.04),         # 6cm × 4cm view area (physical)
    pixel_size_xy=(0.001, 0.001),    # 1mm per pixel
    # → Gives resolution: 60×40 pixels
)
```

### When to Use Low Camera Resolution (60×40 pixels)
- Quick previews during development
- Monitoring simulation progress
- When file size matters
- Testing overlay appearance

**Pros**: Fast rendering, small files (~50 KB PNG)
**Cons**: Cannot see fine details in images

### When to Use High Camera Resolution (1920×1080 or higher)
```python
PerspectiveCameraCallback(
    resolution_wh=(1920, 1080),  # HD resolution
    save_images=True,
    dpi=300  # Publication quality
)
```

- Publication figures
- Presentations
- Detailed overlay analysis
- Marketing materials

**Pros**: Crisp, professional images
**Cons**: Larger files (~2-5 MB PNG), slightly slower

---

## Smaller Specimen: Resolution Retained?

**YES! Camera resolution is independent of specimen size.**

### Example: Small Specimen (2mm × 2mm × 1mm)

```python
# Scenario 1: Small specimen, default camera
python simulate.py --part-x 2.0 --part-y 2.0 --part-z 1.0 --voxel-size 200
# Camera still outputs 60×40 pixels
# But the part looks smaller in the view (6cm view area shows 2mm part)
```

**What happens**:
- The camera views a **fixed physical area** (6cm × 4cm by default)
- A 2mm part appears small in this view (like zoomed out)
- Output image is still 60×40 pixels
- The part occupies only ~20×13 pixels of the image (2mm/6cm × 60px ≈ 20px)

**To zoom in on small specimens**:

```python
PerspectiveCameraCallback(
    plane_size=(0.01, 0.01),         # 1cm × 1cm view (zoomed in!)
    pixel_size_xy=(0.00005, 0.00005),  # 50μm per pixel (matches voxel size)
    # → Gives resolution: 200×200 pixels
)
```

Now the 2mm part fills most of the view!

---

## Practical Recommendations

### For Development/Debugging
```python
# Fast simulation, low-res images
voxel_size_um=300,
resolution_wh=(60, 40)  # or use default
```
**Runtime**: Minutes
**Quality**: Good enough to see if it's working

### For Parameter Studies
```python
# Medium simulation, default camera
voxel_size_um=200,
# Use default camera resolution
```
**Runtime**: 1-2 hours per case
**Quality**: Adequate for trends

### For Final Results
```python
# High-res simulation, HD camera
voxel_size_um=100,
resolution_wh=(1920, 1080)
```
**Runtime**: 10-20 hours
**Quality**: Publication-ready

### For Small Parts with Detail
```python
# Adjust plane_size to match part size
PerspectiveCameraCallback(
    plane_size=(0.01, 0.01),  # Match your part size roughly
    voxel_size_um=100,        # Keep simulation accurate
    resolution_wh=(800, 800)  # Enough pixels for detail
)
```

---

## Summary Table

| Configuration | Voxel Size | Camera Res | Runtime | Image Quality | Physics Accuracy | Use Case |
|--------------|------------|------------|---------|---------------|------------------|----------|
| **Quick preview** | 300μm | 60×40 | Minutes | Low | Qualitative | Debugging |
| **Standard dev** | 200μm | 60×40 | 1-2h | Low | Medium | Development |
| **Production** | 100μm | 60×40 | 10-20h | Low | High | Data collection |
| **Publication** | 100μm | 1920×1080 | 10-20h | High | High | Papers/presentations |
| **High-speed test** | 300μm | 1920×1080 | Minutes | High* | Qualitative | Demos (blocky but crisp) |

*High image quality but shows blocky simulation artifacts clearly

---

## Key Takeaways

1. **Voxel resolution** = simulation accuracy (affects physics)
2. **Camera resolution** = image quality (affects visualization)
3. **They are independent** - you can mix and match
4. **Quick tests** lose geometric/thermal detail but capture overall behavior
5. **High resolution** gives accurate predictions but takes much longer
6. **Smaller specimens** don't change camera resolution automatically - adjust `plane_size` to zoom in
7. **Best practice**: Use quick tests for development, high-res for final results
