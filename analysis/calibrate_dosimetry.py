#!/usr/bin/env python3
"""
HELIOS Dosimetry Equation Calibration & Table Consistency Check
================================================================
Verifies the empirical coefficient in the dose equation against the
Bastille Day 2000 anchor point and recalibrates the severity table
so that Bz ranges, dose ranges, and crew-response protocols are
mathematically consistent.

Equation:
    D (mSv) = K * |Bz|^1.3 * sqrt(v_CME) * t_exposure

Anchor point (Bastille Day 2000):
    Bz = -60 nT,  v = 1673 km/s,  t = 10 h  →  D ≈ 1015 mSv

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
import pandas as pd

# ============================================================================
# 1. COEFFICIENT VERIFICATION
# ============================================================================

def verify_coefficient():
    """
    Check whether K = 0.02 reproduces the Bastille Day target dose.
    If not, solve for the correct K.
    """
    # Bastille Day 2000 anchor
    Bz      = 60      # |nT|
    v       = 1673    # km/s
    t       = 10      # hours
    target  = 1015    # mSv (whitepaper value)

    # Current equation
    K_current     = 0.02
    dose_current  = K_current * (Bz ** 1.3) * np.sqrt(v) * t

    # Solve for correct K
    K_correct = target / ((Bz ** 1.3) * np.sqrt(v) * t)

    error_pct = ((dose_current - target) / target) * 100

    return {
        'K_current':      K_current,
        'dose_current':   round(dose_current, 2),
        'target_dose':    target,
        'error_pct':      round(error_pct, 2),
        'K_correct':      K_correct,
        'K_correct_4dp':  round(K_correct, 4),
    }


# ============================================================================
# 2. BZ-RANGE CALCULATOR
# ============================================================================

def solve_bz_for_dose(dose_mSv, K, v_cme, t):
    """
    Invert  D = K * |Bz|^1.3 * sqrt(v) * t   →   |Bz|
    """
    denom = K * np.sqrt(v_cme) * t
    if denom <= 0:
        return 0.0
    return (dose_mSv / denom) ** (1.0 / 1.3)


def calculate_bz_thresholds(K, v_cme=800, t=10):
    """
    Calculate Bz values that correspond to dose boundaries
    [10, 50, 100, 200, 600] mSv for the given coefficient / speed / exposure.
    """
    dose_boundaries = [10, 50, 100, 200, 600]
    return {d: round(solve_bz_for_dose(d, K, v_cme, t), 1)
            for d in dose_boundaries}


# ============================================================================
# 3. SEVERITY TABLE GENERATOR
# ============================================================================

def generate_severity_table(K, v_cme=800, t=10):
    """
    Build a DataFrame whose Bz ranges are *derived* from the dose formula
    so that table and equation are guaranteed consistent.
    """
    bz = calculate_bz_thresholds(K, v_cme, t)

    table = pd.DataFrame({
        'Severity':       ['Low', 'Moderate', 'High', 'Extreme'],
        'Bz_Range_nT':    [
            f'-{bz[10]:.1f} to -{bz[50]:.1f}',
            f'-{bz[50]:.1f} to -{bz[100]:.1f}',
            f'-{bz[100]:.1f} to -{bz[200]:.1f}',
            f'< -{bz[200]:.1f}',
        ],
        'Dose_Range_mSv': ['10 – 50', '50 – 100', '100 – 200', '> 200'],
        'Crew_Response':  [
            'Enhanced monitoring',
            'Shelter-in-place advisory',
            'Mandatory shelter protocols',
            'EVA Abort, emergency shielding',
        ],
    })
    return table


# ============================================================================
# 4. MULTI-VELOCITY VALIDATION MATRIX
# ============================================================================

def validation_matrix(K, t=10):
    """
    Compute dose for a grid of (Bz, v_CME) combinations to confirm
    realistic values across the operational envelope.
    """
    bz_vals  = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    v_vals   = [400, 800, 1200, 1500, 2000]

    rows = []
    for bz in bz_vals:
        for v in v_vals:
            dose = K * (bz ** 1.3) * np.sqrt(v) * t
            rows.append({
                '|Bz| (nT)': bz,
                'v_CME (km/s)': v,
                'Dose (mSv)': round(dose, 1),
            })
    return pd.DataFrame(rows)


# ============================================================================
# 5. BASTILLE DAY RE-CHECK
# ============================================================================

def bastille_recheck(K):
    dose = K * (60 ** 1.3) * np.sqrt(1673) * 10
    err  = ((dose - 1015) / 1015) * 100
    return round(dose, 1), round(err, 2)


# ============================================================================
# MAIN – run full calibration
# ============================================================================

def main():
    sep = "=" * 80

    # ── Step 1: coefficient verification ─────────────────────────────────
    print(sep)
    print("STEP 1: COEFFICIENT VERIFICATION   (anchor = Bastille Day 2000)")
    print(sep)
    v = verify_coefficient()
    for k, val in v.items():
        print(f"  {k:20s}: {val}")

    # ── Decision ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("DECISION")
    print(sep)

    if abs(v['error_pct']) < 10:
        print("  [PASS]  K = 0.02 is within 10 % of Bastille Day target.")
        K = 0.02
    else:
        print(f"  [FAIL]  K = 0.02 gives {v['error_pct']:+.1f} % error  "
              f"({v['dose_current']:.0f} mSv vs 1015 mSv target).")
        K = v['K_correct']
        print(f"  --> Corrected coefficient:  K = {K:.6f}  "
              f"(rounded {v['K_correct_4dp']})")

    # ── Step 2: new severity table at reference speed ────────────────────
    ref_speed = 800   # km/s – "typical moderate CME"
    print(f"\n{sep}")
    print(f"STEP 2: CALIBRATED SEVERITY TABLE   (v_CME = {ref_speed} km/s, "
          f"t = 10 h, K = {K:.6f})")
    print(sep)
    table = generate_severity_table(K, v_cme=ref_speed)
    print(table.to_string(index=False))

    # Also show Bz thresholds for multiple speeds
    print(f"\n{sep}")
    print("STEP 2b: Bz THRESHOLDS BY CME SPEED")
    print(sep)
    print(f"  {'Speed':>8}   {'10 mSv':>10}  {'50 mSv':>10}  "
          f"{'100 mSv':>10}  {'200 mSv':>10}  {'600 mSv':>10}")
    print("  " + "-" * 68)
    for spd in [400, 600, 800, 1000, 1500, 2000]:
        th = calculate_bz_thresholds(K, v_cme=spd)
        print(f"  {spd:>6} km/s   "
              + "  ".join(f"-{th[d]:>7.1f}" for d in [10, 50, 100, 200, 600]))

    # ── Step 3: multi-velocity validation matrix ─────────────────────────
    print(f"\n{sep}")
    print("STEP 3: DOSE VALIDATION MATRIX   (rows = |Bz|, cols = v_CME)")
    print(sep)
    mat = validation_matrix(K)

    # Pivot for readability
    pivot = mat.pivot(index='|Bz| (nT)',
                      columns='v_CME (km/s)',
                      values='Dose (mSv)')
    print(pivot.to_string())

    # ── Step 4: Bastille Day final recheck ───────────────────────────────
    dose_bd, err_bd = bastille_recheck(K)
    print(f"\n{sep}")
    print("STEP 4: BASTILLE DAY FINAL RECHECK")
    print(sep)
    print(f"  K         = {K:.6f}")
    print(f"  Predicted = {dose_bd:.1f} mSv")
    print(f"  Target    = 1015 mSv")
    print(f"  Error     = {err_bd:+.2f} %")
    if abs(err_bd) < 10:
        print("  STATUS    = PASS")
    else:
        print("  STATUS    = FAIL")

    # ── Step 5: comparison old table vs new table ────────────────────────
    print(f"\n{sep}")
    print("STEP 5: OLD TABLE vs NEW TABLE COMPARISON")
    print(sep)
    old_bz = {
        'Low':      '-10 to -20',
        'Moderate':  '-20 to -35',
        'High':      '-35 to -50',
        'Extreme':    '< -50',
    }
    new_bz_dict = calculate_bz_thresholds(K, v_cme=ref_speed)
    new_bz = {
        'Low':      f'-{new_bz_dict[10]:.1f} to -{new_bz_dict[50]:.1f}',
        'Moderate':  f'-{new_bz_dict[50]:.1f} to -{new_bz_dict[100]:.1f}',
        'High':      f'-{new_bz_dict[100]:.1f} to -{new_bz_dict[200]:.1f}',
        'Extreme':    f'< -{new_bz_dict[200]:.1f}',
    }
    print(f"  {'Severity':>10}  {'Old Bz (nT)':>20}  {'New Bz (nT)':>25}")
    print("  " + "-" * 60)
    for sev in ['Low', 'Moderate', 'High', 'Extreme']:
        print(f"  {sev:>10}  {old_bz[sev]:>20}  {new_bz[sev]:>25}")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Equation:   D(mSv) = {K:.6f} * |Bz|^1.3 * sqrt(v_CME) * t")
    print(f"  Coefficient K = {K:.6f}  (was 0.02)")
    print(f"  Exponent      = 1.3 (unchanged)")
    print(f"  Ref speed     = {ref_speed} km/s (typical moderate CME)")
    print(f"  Exposure      = 10 h (pessimistic worst-case)")
    print(f"  Bastille Day  = {dose_bd:.1f} mSv  (target 1015, err {err_bd:+.1f}%)")

    return K, table


if __name__ == "__main__":
    K_final, final_table = main()
