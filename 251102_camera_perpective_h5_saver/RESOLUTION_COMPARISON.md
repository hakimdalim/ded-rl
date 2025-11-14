# DED Simulation Resolution Comparison Guide

## Overview

This document provides a detailed comparison between quick test simulations and high-resolution simulations for Directed Energy Deposition (DED) processes, explaining the quality differences, trade-offs, and use cases for each approach.

---

## Configuration Comparison

### Quick Test Configuration (Rapid Prototyping)

**Typical Parameters:**
- **Voxel Size:** 200-300 micrometers (um)
- **Part Dimensions:** 2-5 mm height, 5 mm x 5 mm footprint
- **Build Volume:** 20 mm x 20 mm x 15 mm
- **Time Step:** 200 ms
- **Total Voxels:** ~50,000-150,000 voxels
- **Memory per Field:** 1-5 MB
- **Simulation Time:** Minutes to tens of minutes

**Example Quick Test:**
```python
SimulationRunner.from_human_units(
    build_volume_mm=(20.0, 20.0, 15.0),
    part_volume_mm=(5.0, 5.0, 2.0),  # Small part
    voxel_size_um=200.0,              # Coarse resolution
    delta_t_ms=200.0,
    scan_speed_mm_s=3.0,
    laser_power_W=600.0,
    powder_feed_g_min=2.0,
    hatch_spacing_um=700.0,
    layer_spacing_um=350.0
)
```

### High-Resolution Configuration (Production Analysis)

**Typical Parameters:**
- **Voxel Size:** 50-100 micrometers (um)
- **Part Dimensions:** 5+ mm height, 10 mm x 10 mm footprint
- **Build Volume:** 30 mm x 30 mm x 20 mm
- **Time Step:** 50-100 ms
- **Total Voxels:** 2-20 million voxels
- **Memory per Field:** 100-500 MB (potentially GB)
- **Simulation Time:** Hours to days

**Example High-Resolution:**
```python
SimulationRunner.from_human_units(
    build_volume_mm=(30.0, 30.0, 20.0),
    part_volume_mm=(10.0, 10.0, 5.0),  # Realistic part
    voxel_size_um=100.0,                # Fine resolution
    delta_t_ms=100.0,
    scan_speed_mm_s=3.0,
    laser_power_W=600.0,
    powder_feed_g_min=2.0,
    hatch_spacing_um=700.0,
    layer_spacing_um=350.0
)
```

---

## Quality Differences by Category

### 1. Spatial Features & Geometric Accuracy

#### Quick Test (200-300um voxels)

**What is Lost:**
- **Fine geometric details** smaller than ~0.5-1.0 mm cannot be resolved
- **Track edge definition:** Track boundaries appear stepped/blocky due to voxel discretization
- **Surface roughness:** Clad bead surface appears faceted rather than smooth
- **Overlap precision:** Inter-track and inter-layer overlap regions are poorly defined
- **Substrate interface:** Sharp transition between substrate and deposited material is blurred

**What is Retained:**
- **Overall part shape:** Gross dimensions and profile are captured
- **Track placement:** Centerline positions of tracks are accurate
- **Layer structure:** Number and vertical spacing of layers is correct
- **Build orientation:** Scan direction and strategy patterns are preserved

**Quantitative Impact:**
- Points per hatch spacing: ~2-4 points (minimum recommended: 5)
- Points along track length: 15-25 points for 5mm track (recommended: 100+)
- Geometric error: +/- 1-2 voxel widths (200-600 um)
- Feature resolution limit: ~0.5-1.0 mm

**Warning Messages:**
```
Low resolution across hatch spacing. Current configuration has 3.5 points
per hatch spacing (recommended minimum: 5). This might affect simulation
accuracy. Consider decreasing X voxel size from 200.0um to 140.0um or less.
```

#### High-Resolution (50-100um voxels)

**What is Gained:**
- **Smooth track boundaries:** Track edges follow true parabolic/Gaussian profiles
- **Surface topology:** Realistic clad bead surface with proper curvature
- **Overlap zones:** Precise capture of remelted regions between passes
- **Micro-features:** Can resolve features down to 200-400 um
- **Interface definition:** Sharp, accurate substrate-deposit boundary

**Quantitative Performance:**
- Points per hatch spacing: 7-14 points (exceeds minimum)
- Points along track length: 50-100+ points for 5mm track
- Geometric error: +/- 50-100 um
- Feature resolution limit: ~200-400 um

---

### 2. Thermal Details & Physics

#### Quick Test (200-300um voxels)

**What is Lost:**
- **Thermal gradient accuracy:** Temperature gradients are spatially averaged over large volumes
- **Peak temperature precision:** Melt pool peak temperatures may be underestimated due to spatial averaging
- **Cooling rate fidelity:** Rapid cooling near melt pool boundaries is smoothed out
- **Heat Affected Zone (HAZ):** Sharp thermal transitions are blurred across multiple voxels
- **Temporal resolution:** 200ms time steps miss fast thermal transients (<0.2s)

**What is Retained:**
- **Bulk thermal behavior:** Average temperature fields and heat flow patterns
- **Melt pool existence:** Presence and approximate location of molten region
- **Thermal history trends:** Overall heating/cooling cycles for each location
- **Steady-state patterns:** Time-averaged thermal fields are qualitatively correct

**Physical Consequences:**
- Melt pool dimensions are approximate (+/- 200-400 um)
- Cannot accurately predict microstructure (grain size, phase transformations)
- Cooling rates may be off by 10-50%
- Thermal stress calculations will have significant error

**Thermal Resolution:**
- Temperature field smoothing: ~3-5x voxel size (0.6-1.5 mm)
- Melt pool boundary uncertainty: +/- 1-2 voxels
- Cooling rate resolution: ~200ms timesteps (5 Hz sampling)

#### High-Resolution (50-100um voxels)

**What is Gained:**
- **Sharp thermal gradients:** Accurate capture of steep temperature changes
- **Peak temperature accuracy:** Better representation of maximum temperatures
- **Cooling rate precision:** Faster timesteps (50-100ms) capture thermal dynamics
- **HAZ definition:** Clear delineation of heat affected zones
- **Sub-voxel interpolation:** Melt pool dimensions calculated with sub-voxel accuracy

**Physical Benefits:**
- Melt pool dimensions accurate to +/- 50-100 um
- Reliable input for microstructure modeling
- Cooling rates accurate within 5-10%
- Better prediction of residual stress and distortion

**Thermal Resolution:**
- Temperature field smoothing: <1mm
- Melt pool boundary uncertainty: +/- 0.5 voxels (25-50 um)
- Cooling rate resolution: 50-100ms timesteps (10-20 Hz sampling)

---

### 3. Melt Pool Characteristics

#### Quick Test (200-300um voxels)

**Melt Pool Representation:**
- **Width:** Captured with 3-5 voxels across (~600-1500 um total)
- **Length:** Captured with 5-8 voxels along track
- **Depth:** May be only 2-4 voxels deep
- **Shape:** Blocky, stair-stepped appearance
- **Boundary:** Fuzzy transition due to thermal averaging

**Limitations:**
- Melt pool shape is highly discretized
- Cannot distinguish between different melting regimes (keyhole vs. conduction)
- Dilution ratio (substrate remelting) is approximate
- Powder catchment efficiency calculations are coarse

**Typical Dimensions (600W, 3mm/s):**
- Width: 1.5-2.0 mm (captured by 7-10 voxels at 200um)
- Length: 2.5-3.5 mm (captured by 12-17 voxels)
- Depth: 0.8-1.2 mm (captured by 4-6 voxels)

#### High-Resolution (50-100um voxels)

**Melt Pool Representation:**
- **Width:** Captured with 15-40 voxels across
- **Length:** Captured with 25-70 voxels along track
- **Depth:** 8-24 voxels deep
- **Shape:** Smooth, realistic ellipsoidal/parabolic profile
- **Boundary:** Sharp liquidus isotherm

**Capabilities:**
- Accurate melt pool morphology
- Distinguish convection patterns (if modeled)
- Precise dilution calculations
- Accurate powder distribution integration

**Typical Dimensions (600W, 3mm/s):**
- Width: 1.5-2.0 mm (captured by 15-40 voxels at 50-100um)
- Length: 2.5-3.5 mm (captured by 25-70 voxels)
- Depth: 0.8-1.2 mm (captured by 8-24 voxels)

---

### 4. Clad Bead Geometry

#### Quick Test (200-300um voxels)

**Clad Geometry Issues:**
- **Parabolic profile:** Approximated with only 3-5 discrete height levels
- **Bead width:** Quantized to nearest voxels (error ~100-300 um)
- **Bead height:** Quantized to vertical voxels (350um layer spacing = 1-2 voxels)
- **Track overlap:** Overlap regions poorly resolved
- **Build-up accuracy:** Cumulative height errors can reach 500-1000 um

**Impact on Part Quality:**
- Part height may differ from target by 0.5-1.0 mm
- Width variations are stepped rather than continuous
- Surface finish appears rough/faceted
- Inter-layer bonding zones are approximate

**Height Accumulation Example (10 layers):**
- Target height: 3.5 mm (10 layers x 350um)
- Actual representation: 3.2-3.8 mm (due to voxel quantization)
- Error: +/- 8-15% of target height

#### High-Resolution (50-100um voxels)

**Clad Geometry Fidelity:**
- **Parabolic profile:** Smooth representation with 8-15 discrete levels
- **Bead width:** Accurate to +/- 50-100 um
- **Bead height:** Multiple voxels per layer (3-7 voxels for 350um layer)
- **Track overlap:** Well-defined remelted and bonded regions
- **Build-up accuracy:** Cumulative errors <200 um

**Part Quality Predictions:**
- Part height accurate to +/- 100-200 um
- Width variations smoothly captured
- Surface finish realistically modeled
- Inter-layer bonding zones clearly defined

**Height Accumulation Example (10 layers):**
- Target height: 3.5 mm
- Actual representation: 3.45-3.55 mm
- Error: +/- 1-3% of target height

---

### 5. Powder Distribution & Mass Flow

#### Quick Test (200-300um voxels)

**Powder Modeling:**
- **Stream profile:** Gaussian/concentration profiles are coarse
- **Catchment integration:** Numerical integration uses ~200 points
- **Spatial distribution:** Powder concentration averaged over large voxels
- **Mass balance:** Overall mass flow is conserved but spatial details lost

**Consequences:**
- Dilution ratio (powder vs. substrate) is approximate
- Local composition variations not captured
- Clad dimension predictions rely on empirical corrections
- Cannot predict porosity from powder distribution defects

#### High-Resolution (50-100um voxels)

**Powder Modeling:**
- **Stream profile:** Smooth, realistic concentration gradients
- **Catchment integration:** Numerical integration uses 200-500 points
- **Spatial distribution:** Fine-scale powder concentration variations
- **Mass balance:** Accurate spatial distribution of deposited material

**Benefits:**
- Accurate dilution calculations
- Can model local composition gradients
- Better clad dimension predictions from first principles
- Potential to identify porosity-prone regions

---

### 6. Computational Cost

#### Quick Test (200-300um voxels)

**Computational Performance:**
- **Voxel count:** 50,000-150,000 voxels
- **Memory:** 1-5 MB per field (activation, temperature)
- **Simulation speed:** Real-time to 10x faster than real process
- **Storage (HDF5):** 10-100 MB for full simulation with compression
- **Hardware:** Runs on laptops, standard workstations

**Use Cases:**
- Algorithm development and testing
- Rapid parameter screening
- Code validation
- Educational demonstrations
- Proof-of-concept studies

**Typical Runtime (20mm part, 5 layers):**
- Setup: <5 seconds
- Execution: 2-10 minutes
- Visualization: Real-time capable

#### High-Resolution (50-100um voxels)

**Computational Performance:**
- **Voxel count:** 2-20 million voxels
- **Memory:** 100-500 MB per field (potentially GB for large builds)
- **Simulation speed:** 0.1-1x real-time (slower than actual process)
- **Storage (HDF5):** 1-50 GB for full simulation with compression
- **Hardware:** Requires high-performance workstations, cluster recommended

**Use Cases:**
- Production process optimization
- Detailed thermal analysis
- Microstructure prediction
- Defect prediction
- High-fidelity validation studies

**Typical Runtime (50mm part, 20 layers):**
- Setup: 10-30 seconds
- Execution: 2-24 hours
- Visualization: May require downsampling for real-time display

---

## Decision Matrix: When to Use Each Resolution

### Use Quick Test Resolution (200-300um) When:

1. **Developing new features** or algorithms
2. **Testing callback systems** or data collection
3. **Validating code changes** before full-scale runs
4. **Exploring parameter space** with many combinations
5. **Teaching/demonstrating** DED process physics
6. **Hardware is limited** (laptop, limited memory)
7. **Results needed quickly** (minutes to hours)
8. **Qualitative trends** are sufficient
9. **Part-level behavior** is more important than micro-scale details
10. **Initial feasibility studies** for new geometries

### Use High Resolution (50-100um) When:

1. **Optimizing production parameters** for real manufacturing
2. **Predicting microstructure** and mechanical properties
3. **Calculating residual stress** and distortion
4. **Validating against experiments** with thermal cameras/thermocouples
5. **Publishing scientific results** requiring accuracy
6. **Defect prediction** (porosity, lack-of-fusion, cracking)
7. **Quality certification** for critical applications
8. **Detailed thermal history** needed for each voxel
9. **Surface finish** and dimensional accuracy are critical
10. **Final design verification** before manufacturing

---

## Accuracy Validation Thresholds

### Recommended Discretization Criteria

The simulation automatically warns when discretization is insufficient:

**Track Length Resolution:**
- Minimum: 100 points along track length
- Quick test (5mm track, 200um voxels): 25 points - WARNING
- High-res (5mm track, 100um voxels): 50 points - WARNING
- High-res (10mm track, 100um voxels): 100 points - OK

**Hatch Spacing Resolution:**
- Minimum: 5 points per hatch spacing
- Quick test (700um hatch, 200um voxels): 3.5 points - WARNING
- High-res (700um hatch, 100um voxels): 7 points - OK

**Melt Pool Coverage:**
- Minimum: 5 voxels across melt pool width
- Recommended: 10+ voxels for accurate shape
- High-fidelity: 20+ voxels for sub-features

---

## Feature Resolution Limits

### Quick Test (200-300um voxels)

**Can Resolve:**
- Part-level dimensions (>5mm features)
- Layer structure and count
- Track placement and spacing
- Bulk temperature fields
- Overall activation volume

**Cannot Resolve:**
- Track edge details (<0.5mm)
- Melt pool fine structure
- Sharp thermal gradients
- Surface roughness
- Micro-scale defects

**Spatial Resolution Cutoff:** ~0.5-1.0 mm

### High Resolution (50-100um voxels)

**Can Resolve:**
- Detailed track geometry
- Melt pool morphology
- Sharp thermal boundaries
- Inter-track overlap zones
- Surface topology

**Cannot Resolve:**
- Grain structure (<10-50um)
- Dendrite arms (<1-10um)
- Individual powder particles (~50-150um diameter)
- Keyhole dynamics (requires <10um, CFD)

**Spatial Resolution Cutoff:** ~200-400 um

---

## Best Practices

### Workflow Recommendation

1. **Initial Development:** Start with quick test resolution (200-300um)
   - Validate setup and configuration
   - Test callbacks and data collection
   - Ensure simulation runs without errors

2. **Parameter Screening:** Use moderate resolution (150-200um)
   - Screen multiple parameter combinations
   - Identify promising regions in parameter space
   - Check for physical anomalies

3. **Final Analysis:** Use high resolution (50-100um)
   - Run selected parameter sets at high fidelity
   - Extract detailed thermal histories
   - Generate publication-quality results

### Configuration Guidelines

**For 2mm tall parts (testing):**
- Voxel size: 200um
- Time step: 200ms
- Expected runtime: 5-15 minutes
- Validation: Visual inspection of trends

**For 5mm tall parts (analysis):**
- Voxel size: 100-150um
- Time step: 100-150ms
- Expected runtime: 1-4 hours
- Validation: Check discretization warnings

**For 10mm+ tall parts (production):**
- Voxel size: 50-100um
- Time step: 50-100ms
- Expected runtime: 4-24+ hours
- Validation: Quantitative comparison with experiments

### Memory Management

**Quick Test:**
- Full volume storage feasible
- Real-time visualization possible
- All timesteps can be saved

**High Resolution:**
- Consider saving only key timesteps (every N steps)
- Use HDF5 compression (9x compression for activation, 2-4x for temperature)
- May need to visualize subsets or slices
- Monitor memory usage during simulation

---

## Summary

### Quick Test Resolution (200-300um voxels)

**Advantages:**
- Fast execution (minutes)
- Low memory requirements
- Suitable for laptops
- Ideal for development
- Good for parameter screening

**Disadvantages:**
- Coarse geometric representation
- Approximate thermal details
- Discretization warnings common
- Not suitable for final validation
- Limited physical accuracy

**Typical Error Margins:**
- Geometry: +/- 0.3-0.6 mm
- Temperature peaks: +/- 50-150K
- Melt pool size: +/- 0.2-0.5 mm
- Cooling rates: +/- 20-50%

### High Resolution (50-100um voxels)

**Advantages:**
- Accurate geometry
- Precise thermal fields
- Realistic melt pool shapes
- Suitable for validation
- Publication quality

**Disadvantages:**
- Slow execution (hours to days)
- High memory requirements
- Requires powerful hardware
- Large data storage needs
- May need cluster computing

**Typical Error Margins:**
- Geometry: +/- 50-150 um
- Temperature peaks: +/- 20-50K
- Melt pool size: +/- 50-150 um
- Cooling rates: +/- 5-10%

---

## Example Comparisons

### Memory Scaling

| Configuration | Voxels | Memory/Field | Total (4 fields) | HDF5 Size (20 steps) |
|--------------|--------|--------------|------------------|----------------------|
| Quick: 20x20x15mm @ 300um | 67k | 0.5 MB | 2 MB | 15 MB |
| Quick: 20x20x15mm @ 200um | 225k | 1.8 MB | 7 MB | 40 MB |
| High: 30x30x20mm @ 100um | 6M | 48 MB | 192 MB | 2.5 GB |
| High: 30x30x20mm @ 50um | 48M | 384 MB | 1.5 GB | 18 GB |

### Simulation Time Scaling

| Configuration | Steps/Layer | Layers | Total Steps | Runtime (estimate) |
|--------------|-------------|--------|-------------|-------------------|
| Quick: 2mm part @ 200um | 10 | 6 | 60 | 5 minutes |
| Quick: 5mm part @ 200um | 15 | 15 | 225 | 15 minutes |
| High: 5mm part @ 100um | 30 | 15 | 450 | 2 hours |
| High: 10mm part @ 50um | 60 | 30 | 1800 | 12 hours |

---

## Conclusion

**Choose quick test resolution (200-300um)** for development, testing, and rapid exploration where qualitative trends and overall behavior are sufficient.

**Choose high resolution (50-100um)** for production optimization, validation studies, and any application where quantitative accuracy is required.

**Progressive refinement** - Start coarse, refine as needed - is the most efficient approach for most projects.

The simulation's built-in validation warnings will alert you when discretization is insufficient for your chosen parameters, helping you make informed decisions about when resolution matters.

---

*Last Updated: 2025-10-23*
*Compatible with: hypo-simulations semi-analytical DED-LB process simulation*
