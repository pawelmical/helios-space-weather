# HELIOS Triangulation-Constrained Prediction Guide

## Overview

The `scripts/test_triangulation_constraint.py` script demonstrates the **core value proposition of HELIOS**: using stereoscopic triangulation measurements to significantly improve CME arrival predictions.

**Key Result**: Reduces prediction error by ~40-99% and uncertainty by ~70% compared to standard methods.

---

## How It Works

### 1. Simulate a "True" CME (Unknown Reality)

```python
# True parameters (what nature actually does - unknown to predictor)
true_gamma_0 = 3.926e-10 * 0.70   # 30% lower drag
true_n_power = 14.09 * 0.80       # 20% lower power-law exponent
true_initial_speed = 1750.0       # Actual speed
```

The script creates a realistic CME trajectory using parameters that **differ significantly** from our calibrated model to simulate real-world uncertainty.

**WHERE TO CHANGE**:
- **Line 35-37**: Adjust `true_gamma_0`, `true_n_power`, `true_initial_speed` to test different scenarios
- **Line 41**: Change `estimated_speed` to simulate coronagraph measurement errors

---

### 2. Extract Triangulation "Measurements"

```python
measurement_times = [5, 10, 15, 20]  # Hours after eruption
```

HELIOS provides CME position at multiple times via stereoscopic triangulation from L4/L5 viewpoints.

**WHERE TO CHANGE**:
- **Line 94**: Modify `measurement_times` list to use different observation times
- **Line 102**: Adjust `np.random.normal(0, 0.01)` to change measurement noise (1% default)

---

### 3. Standard Prediction (No Triangulation)

```python
standard = run_ensemble(initial_speed=estimated_speed, n_members=200, seed=42)
```

Uses only the coronagraph speed estimate and calibrated drag model with uncertainty.

**WHERE TO CHANGE**:
- **Line 112**: `n_members=200` → lower (50-100) for faster runs, higher (500+) for smoother distributions
- **Line 112**: `seed=42` → change for different random ensemble realizations

**Parameters** (in `helios_code/ensemble_propagation.py` → `run_ensemble()`):
- `v0_variation_percent=15.0` → initial speed uncertainty (±15%)
- `gamma_variation_percent=30.0` → drag coefficient uncertainty (±30%)
- `n_power_variation_percent=10.0` → power-law exponent uncertainty (±10%)

---

### 4. Triangulation-Constrained Prediction (HELIOS)

```python
constrained = triangulation_constrained_prediction(
    initial_speed=estimated_speed,
    measured_positions=measured_positions,
    n_ensemble=200,
    seed=42
)
```

**The Key Algorithm** (in `helios_code/ensemble_propagation.py`):

#### Step 1: Fit Model Parameters to Observations
```python
def objective(params):
    v0_factor, gamma_0, n_power = params
    # Simulate CME to each observation time
    # Compare predicted vs observed positions
    # Return position error + regularization
```

Uses **L-BFGS-B optimizer** with:
- **Bounds**: v0_factor ∈ [0.7, 1.4], gamma_0 ∈ [1e-11, 1e-8], n_power ∈ [3, 30]
- **Multi-start**: 3×4×4 = 48 initial guesses to find global minimum
- **Regularization**: Mild preference for calibrated defaults

**WHERE TO CHANGE** (in `helios_code/ensemble_propagation.py` → `triangulation_constrained_prediction()`):

**Line ~688**: Bounds
```python
bounds = [(0.7, 1.4), (1e-11, 1e-8), (3.0, 30.0)]
# Widen bounds if CMEs are very fast/slow
```

**Line ~721**: Regularization weight
```python
reg_weight = 0.001  # Lower = trust data more, higher = trust prior more
```

**Line ~689-696**: Multi-start grid
```python
for v0_init in [0.9, 1.0, 1.1]:          # Reduce starts for speed
    for g0_init in [1e-10, 3e-10, 5e-10, 1e-9]:
        for n_init in [8, 12, 16, 20]:
```

**Line ~694**: Optimizer settings
```python
method='L-BFGS-B',   # Options: 'L-BFGS-B' (fast), 'Nelder-Mead' (robust but slow)
options={'maxiter': 300}  # Lower for speed, higher for convergence
```

#### Step 2: Re-run Ensemble with Fitted Parameters
```python
# Much tighter variations around fitted values
v0_variation = ±5%     # vs ±15% standard
gamma_factor = ±10%    # vs ±30% standard
n_factor = ±5%         # vs ±10% standard
```

**WHERE TO CHANGE** (in `helios_code/ensemble_propagation.py` → lines ~735-738):
```python
v0_variation = 1 + np.random.uniform(-0.05, 0.05, n_ensemble)  # ±5%
gamma_factor = 1 + np.random.uniform(-0.10, 0.10, n_ensemble)  # ±10%
n_factor = 1 + np.random.uniform(-0.05, 0.05, n_ensemble)      # ±5%
```

---

## Key Configuration Points

### Fast Interactive Mode
For quick tests or demos:
```python
# scripts/test_triangulation_constraint.py
standard = run_ensemble(..., n_members=50)      # Line 112
constrained = triangulation_constrained_prediction(..., n_ensemble=50)  # Line 119
```

### High-Quality Production Mode
For publications or final results:
```python
standard = run_ensemble(..., n_members=500)
constrained = triangulation_constrained_prediction(..., n_ensemble=500)
```

### Optimizer Trade-offs

| Method | Speed | Robustness | When to Use |
|--------|-------|------------|-------------|
| **L-BFGS-B** (current) | Fast | Good | Interactive, demos, most cases |
| **Nelder-Mead** | Slow | Excellent | Final production, pathological cases |
| **Powell** | Medium | Good | Alternative if L-BFGS-B fails |

Change in `helios_code/ensemble_propagation.py` line ~692:
```python
result = minimize(objective, [...], method='L-BFGS-B', ...)  # ← Change here
```

---

## Performance Tuning

### If Script is Too Slow

1. **Reduce ensemble sizes** (50-100 members) → lines 112, 119 of `scripts/test_triangulation_constraint.py`
2. **Reduce multi-start grid** → `helios_code/ensemble_propagation.py` lines 689-696
3. **Lower maxiter** (100-200) → line 694
4. **Increase timestep `dt`** (0.01 instead of 0.005) → line 59 of test script AND line 697 of ensemble_propagation.py

### If Results are Unstable

1. **Increase ensemble sizes** (500+)
2. **Add more multi-start points**
3. **Increase regularization weight** (0.01 instead of 0.001)
4. **Switch to Nelder-Mead optimizer**

### If Optimizer Fails to Converge

1. **Widen bounds** on parameters
2. **Reduce regularization** (trust data more)
3. **Add more measurement times** (every 3-5 hours)
4. **Check measurement noise** isn't too large

---

## Expected Results

### Physics-Only (Current Implementation)

| Metric | Standard | Constrained | Improvement |
|--------|----------|-------------|-------------|
| **Error** | 4.0h (16%) | 0.0-2.3h (0-9%) | **40-99%** |
| **Uncertainty** | 5.3h | 1.5h | **72%** |
| **Speed Recovery** | 1674 km/s (wrong) | 1741-1753 km/s | ~99% accurate |

### With ML (Hypothetical Future)

After training on 100+ events: **~1.2h error (5%)**, **~0.8h uncertainty**

---

## Troubleshooting

### Script "Freezes"
**Cause**: Optimizer is busy computing (CPU-bound, no progress output)  
**Fix**: Check Task Manager for high CPU%, or add progress prints in `ensemble_propagation.py`

### Very Different Results Each Run
**Cause**: Random seed not set or ensemble too small  
**Fix**: Use `seed=42` and increase `n_members` to 200+

### Fitted Parameters Look Unrealistic
**Cause**: Overfitting to sparse measurements, weak regularization  
**Fix**: Increase `reg_weight` or add more measurement times

### "RuntimeError: Optimization failed"
**Cause**: Optimizer hit bounds or couldn't converge  
**Fix**: Widen bounds, reduce `maxiter`, or switch to Nelder-Mead

---

## File Structure

```
scripts/
├── test_triangulation_constraint.py   ← Triangulation demo script
├── run_complete_mvp.py                ← Main MVP demo
└── run_historical_validation.py       ← Metrics verification

helios_code/
└── ensemble_propagation.py            ← Core algorithm
    ├─ triangulation_constrained_prediction()  ← Main function
    ├─ run_ensemble()                          ← Standard prediction
    └─ calculate_cme_trajectory()              ← Physics propagation
```

---

## Quick Reference: Where to Edit

| What to Change | File | Line | Variable |
|----------------|------|------|----------|
| **Test scenario (true params)** | scripts/test_triangulation_constraint.py | 35-37 | `true_gamma_0`, `true_n_power`, `true_initial_speed` |
| **Measurement times** | scripts/test_triangulation_constraint.py | 94 | `measurement_times` |
| **Ensemble size (interactive)** | scripts/test_triangulation_constraint.py | 112, 119 | `n_members`, `n_ensemble` |
| **Optimizer method** | helios_code/ensemble_propagation.py | ~692 | `method='L-BFGS-B'` |
| **Optimizer iterations** | helios_code/ensemble_propagation.py | ~694 | `maxiter: 300` |
| **Multi-start grid** | helios_code/ensemble_propagation.py | 689-696 | Loop ranges |
| **Regularization** | helios_code/ensemble_propagation.py | ~721 | `reg_weight = 0.001` |
| **Constrained variations** | helios_code/ensemble_propagation.py | 735-738 | `uniform(-0.05, 0.05)` |
| **Timestep** | scripts/test_triangulation_constraint.py | 59 | `dt = 0.005` |

---

## Contact & Support

For questions about this implementation, see:
- Main codebase: `helios_code/ensemble_propagation.py`
- Test script: `scripts/test_triangulation_constraint.py`
- Notebook demo: `notebooks/HELIOS_Colab_Demo.ipynb`
