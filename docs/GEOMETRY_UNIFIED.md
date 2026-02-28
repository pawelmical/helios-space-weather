# HELIOS Geometry Unification - L1+L4+L5 Constellation

**Date:** February 1, 2026  
**Status:** UNIFIED - All files now use consistent L1+L4+L5 geometry

---

## Overview

The HELIOS project now uses **consistent L1+L4+L5 constellation geometry** across all validation files. This document explains the unified approach and rationale.

---

## Constellation Configuration

### **Observer Positions (Static Geometry)**

All files now use the same observer position function:

```python
def get_observer_position(longitude_deg: float, distance_au: float = 1.0) -> np.ndarray:
    """Get observer position given heliographic longitude.
    
    Earth is at longitude 0 deg, at 1 AU.
    L1 is at 0 deg, 0.99 AU (Sun-Earth line).
    L4 is at +60 deg, L5 is at -60 deg.
    """
    lon_rad = longitude_deg * DEG_TO_RAD
    r = distance_au * AU_IN_KM
    return np.array([r * np.cos(lon_rad), r * np.sin(lon_rad), 0.0])
```

### **L1+L4+L5 Positions**

| Observer | Longitude | Distance | Role |
|----------|-----------|----------|------|
| **L1** | 0° | 0.99 AU | Sun-Earth line (imaging + triangulation baseline) |
| **L4** | +60° | 1.00 AU | Leading Lagrange point (optimal triangulation with L1) |
| **L5** | -60° | 1.00 AU | Trailing Lagrange point (redundancy + far-side coverage) |

---

## Triangulation Strategy for Earth-Directed CMEs

### **Optimal Pairs (90° Intersection Angle)**

For **Earth-directed CMEs** (along Sun-Earth line at longitude 0°):

- **L1 + L4**: 90° intersection angle - **OPTIMAL**
- **L1 + L5**: 90° intersection angle - **OPTIMAL**
- **L4 + L5**: ~180° intersection angle - **DEGENERATE** (parallel lines)

### **Why L1+L4 is Used**

The **L1+L4 pair** is used for Earth-directed CME triangulation because:

1. **90° intersection angle** = optimal geometry (minimizes GDOP)
2. **Best spatial resolution**: 1.10 million km at 0.5 AU with σ=0.5°
3. **L1 on Sun-Earth line** provides direct Earth-threat assessment
4. **L4 at +60°** provides optimal stereoscopic baseline

**L5 role**: Provides redundancy if L4 fails, and covers far-side CMEs

---

## File Consistency

### **Files Using Unified L1+L4+L5 Geometry**

| File | Geometry Type | Triangulation Pair | Purpose |
|------|---------------|-------------------|---------|
| [geometry_verification.py](../helios_code/geometry_verification.py) | Static L1+L4+L5 | **L1+L4** | Mathematical proof of constellation optimality |
| [test_triangulation_constraint.py](../scripts/test_triangulation_constraint.py) | Static L1+L4+L5 | **L1+L4** | Single-event realistic test with tracking |
| [HELIOS_Colab_Demo.ipynb](../notebooks/HELIOS_Colab_Demo.ipynb) | Static L1+L4+L5 | **L1+L4** | Full pipeline demonstration |

---

## Verified Claims

### **1. Spatial Resolution (L1+L4 @ 0.5 AU)**

| Angular Uncertainty (σ) | Spatial Resolution | Solar Radii |
|------------------------|-------------------|-------------|
| 1.0° | ~2.2 million km | ~3.2 Rs |
| 0.5° | **1.10 million km** | **1.59 Rs** |
| 0.25° | ~0.55 million km | ~0.79 Rs |

### **2. Triangulation-Constrained Prediction Improvement**

From [test_triangulation_constraint.py](../scripts/test_triangulation_constraint.py) results:

```
PREDICTION ERROR:
   Standard:    4.0h (16.1% error)
   Constrained: 0.1h (0.4% error)
   IMPROVEMENT: 3.9h better (98% reduction)

UNCERTAINTY RANGE:
   Standard:    5.3h
   Constrained: 1.5h
   REDUCTION:   3.8h (72% tighter)
```

**Key Insight**: L1+L4 triangulation measurements at 5h, 10h, 15h, 20h allow real-time model fitting, correcting for unknown drag conditions and speed errors.

### **3. Coverage Analysis**

- **Total Coverage**: ~83.3% of heliosphere (mean ~83.5%)
- **Blind Spot**: ~60° on far side (opposite Earth)
- **Earth Threat Zone**: Fully covered (0° ± 90°)

### **4. GDOP Optimization**

- **L1+L4 GDOP**: ~1.2 (optimal)
- **L1+L5 GDOP**: ~1.2 (optimal)
- **L4+L5 GDOP**: ∞ (degenerate for Earth-directed CMEs)

---

## Physical Rationale

### **Why Not Use Dynamic Geometry?**

**Static geometry** (fixed L4/L5 at ±60°) is used for mathematical verification because:

1. **Simplifies analysis**: Clear, reproducible geometry
2. **Worst-case scenario**: Tests optimal performance without orbital variations
3. **Mission design reference**: Lagrange points are defined relative to Earth

**Dynamic geometry** (L4/L5 positions vary with Earth's orbit) is more realistic but:
- Adds complexity without changing fundamental conclusions
- L4/L5 positions always maintain ±60° from Earth by definition
- Used in `HELIOS_Colab_Demo.ipynb` for realistic pipeline demonstration

### **Why L1 Matters Despite Being "In the Middle"**

You're correct that L1 at 0° doesn't add much angular separation for **coverage**, but it's crucial for:

1. **Triangulation baseline**: Provides the 90° intersection with L4/L5
2. **Earth-threat imaging**: Direct view along Sun-Earth line
3. **Validation**: L1 acts as "ground truth" for model calibration
4. **Heritage mission compatibility**: SOHO/LASCO at L1 provides baseline comparison

---

## Unification Complete

**Status**: All geometry files now use L1+L4+L5 constellation with L1+L4 as the primary triangulation pair for Earth-directed CMEs.

**Completed**:
1. Updated `test_triangulation_constraint.py` to use L1+L4+L5 geometry
2. Updated `HELIOS_Colab_Demo.ipynb` to use L1+L4 for triangulation
3. Documented the optimal pair selection logic

**Result**: Consistent 1.10 million km spatial resolution at 0.5 AU (σ=0.5°) across all files using L1+L4 optimal pair.

---

## Quick Reference

```python
# Standard observer setup (all files)
obs_l1 = get_observer_position(0, 0.99)    # L1: 0.99 AU at 0°
obs_l4 = get_observer_position(60, 1.0)    # L4: 1.00 AU at +60°
obs_l5 = get_observer_position(-60, 1.0)   # L5: 1.00 AU at -60°

# Earth-directed CME target
target = np.array([0.5 * AU_IN_KM, 0.0, 0.0])  # 0.5 AU along x-axis

# Optimal triangulation pair for Earth-directed CMEs
los_l1 = (target - obs_l1) / np.linalg.norm(target - obs_l1)
los_l4 = (target - obs_l4) / np.linalg.norm(target - obs_l4)
# Intersection angle: ~90° (OPTIMAL)

# Degenerate pair (avoid for Earth-directed CMEs)
los_l4 = (target - obs_l4) / np.linalg.norm(target - obs_l4)
los_l5 = (target - obs_l5) / np.linalg.norm(target - obs_l5)
# Intersection angle: ~180° (PARALLEL - BAD)
```

---

**Document Maintained By**: Paweł Micał
**Last Updated**: February 15, 2026
