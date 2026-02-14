#!/usr/bin/env python3
"""
HELIOS Radiation Severity Table Generator
==========================================
Generates Bz threshold tables, multi-velocity sensitivity analysis,
dose validation matrices, and Bastille Day verification.

Equation: D(mSv) = 0.0132 × |Bz|^1.3 × √v_CME × t_exposure

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
import csv
import os

# ============================================================================
# CONSTANTS
# ============================================================================

K = 0.0132          # Calibrated dose coefficient (8-event median)
ALPHA = 1.3         # Bz-energy exponent
T_EXPOSURE = 10.0   # Worst-case EVA hours

# Dose thresholds defining severity boundaries
DOSE_BOUNDARIES = [10, 50, 100, 200]  # mSv

# Severity class definitions
SEVERITY_CLASSES = [
    {'name': 'Low',      'dose_min': 10,  'dose_max': 50,   'response': 'Enhanced monitoring'},
    {'name': 'Moderate', 'dose_min': 50,  'dose_max': 100,  'response': 'Shelter-in-place advisory'},
    {'name': 'High',     'dose_min': 100, 'dose_max': 200,  'response': 'Mandatory shelter protocols'},
    {'name': 'Extreme',  'dose_min': 200, 'dose_max': None, 'response': 'EVA Abort, emergency shielding'},
]

# CME velocity scenarios
VELOCITIES = {
    'Slow':     400,
    'Moderate': 800,
    'Fast':     1500,
    'Extreme':  2500,
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# ============================================================================
# CORE MATH
# ============================================================================

def dose_from_bz(bz_nT: float, v_km_s: float, t_hours: float = T_EXPOSURE) -> float:
    """
    Forward: D(mSv) = K × |Bz|^α × √v × t
    """
    bz_abs = abs(bz_nT)
    if bz_abs < 1e-6:
        return 0.0
    return K * (bz_abs ** ALPHA) * np.sqrt(v_km_s) * t_hours


def bz_from_dose(dose_mSv: float, v_km_s: float, t_hours: float = T_EXPOSURE) -> float:
    """
    Inverse: |Bz| = (D / (K × √v × t))^(1/α)
    
    Returns positive Bz magnitude. Actual Bz is negative (southward).
    """
    denominator = K * np.sqrt(v_km_s) * t_hours
    if denominator <= 0 or dose_mSv <= 0:
        return 0.0
    return (dose_mSv / denominator) ** (1.0 / ALPHA)


def write_csv(filepath: str, headers: list, rows: list):
    """Write CSV with headers."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Saved: {filepath}")


# ============================================================================
# TASK 1: Primary Severity Table (v = 800 km/s)
# ============================================================================

def task1_severity_table_v800():
    """Calculate Bz thresholds at reference velocity 800 km/s."""
    print("=" * 80)
    print("TASK 1: SEVERITY TABLE — v_CME = 800 km/s")
    print("=" * 80)
    
    v = 800
    
    # First show the raw Bz inversions for key dose thresholds
    print(f"\n  Dose equation: D = {K} × |Bz|^{ALPHA} × √{v} × {T_EXPOSURE}")
    print(f"  √v = {np.sqrt(v):.4f}")
    print(f"  K × √v × t = {K} × {np.sqrt(v):.4f} × {T_EXPOSURE} = {K * np.sqrt(v) * T_EXPOSURE:.4f}")
    print()
    
    dose_points = [10, 50, 100, 200, 600]
    print(f"  {'Dose (mSv)':>12} → {'|Bz| (nT)':>12}   Derivation")
    print("  " + "-" * 65)
    for dose in dose_points:
        bz = bz_from_dose(dose, v)
        denom = K * np.sqrt(v) * T_EXPOSURE
        print(f"  {dose:>9} mSv → {bz:>9.1f} nT   ({dose}/{denom:.4f})^(1/{ALPHA}) = {bz:.2f}")
    
    # Build severity table
    print(f"\n  SEVERITY CLASSIFICATION TABLE (v = {v} km/s, t = {T_EXPOSURE}h)")
    print("  " + "-" * 90)
    print(f"  {'Severity':<12} {'Bz Min (nT)':>12} {'Bz Max (nT)':>12} {'Dose Min':>10} {'Dose Max':>10} {'Crew Response'}")
    print("  " + "-" * 90)
    
    rows = []
    for sev in SEVERITY_CLASSES:
        bz_min = bz_from_dose(sev['dose_min'], v)
        bz_max = bz_from_dose(sev['dose_max'], v) if sev['dose_max'] is not None else 80.0
        
        dose_min_str = f"{sev['dose_min']} mSv"
        dose_max_str = f"{sev['dose_max']} mSv" if sev['dose_max'] is not None else ">200 mSv"
        bz_max_str = f"{bz_max:.1f}" if sev['dose_max'] is not None else ">80"
        
        print(f"  {sev['name']:<12} {bz_min:>11.1f}  {bz_max_str:>12}  {dose_min_str:>10} {dose_max_str:>10} {sev['response']}")
        
        rows.append([
            sev['name'],
            round(bz_min, 1),
            round(bz_max, 1) if sev['dose_max'] is not None else '>80',
            sev['dose_min'],
            sev['dose_max'] if sev['dose_max'] is not None else '>200',
            sev['response']
        ])
    
    # Validation: forward-calculate dose at each Bz boundary
    print(f"\n  VALIDATION (forward dose at each Bz boundary):")
    print("  " + "-" * 50)
    for sev in SEVERITY_CLASSES:
        bz_lo = bz_from_dose(sev['dose_min'], v)
        d_check = dose_from_bz(bz_lo, v)
        match = "✓" if abs(d_check - sev['dose_min']) < 0.1 else "✗"
        print(f"    Bz={bz_lo:.2f} nT → D={d_check:.1f} mSv (expect {sev['dose_min']}) {match}")
        
        if sev['dose_max'] is not None:
            bz_hi = bz_from_dose(sev['dose_max'], v)
            d_check2 = dose_from_bz(bz_hi, v)
            match2 = "✓" if abs(d_check2 - sev['dose_max']) < 0.1 else "✗"
            print(f"    Bz={bz_hi:.2f} nT → D={d_check2:.1f} mSv (expect {sev['dose_max']}) {match2}")
    
    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, 'severity_table_v800.csv')
    write_csv(csv_path,
              ['Severity', 'Bz_Min_nT', 'Bz_Max_nT', 'Dose_Min_mSv', 'Dose_Max_mSv', 'Crew_Response'],
              rows)
    
    return rows


# ============================================================================
# TASK 2: Multi-Velocity Sensitivity Analysis
# ============================================================================

def task2_multivelocity_analysis():
    """Calculate Bz thresholds across multiple CME velocities."""
    print("\n" + "=" * 80)
    print("TASK 2: MULTI-VELOCITY Bz THRESHOLDS")
    print("=" * 80)
    
    dose_points = [10, 50, 100, 200]
    vel_names = list(VELOCITIES.keys())
    vel_values = list(VELOCITIES.values())
    
    # Header
    header_parts = [f"{'Dose (mSv)':>12}"]
    for name, v in VELOCITIES.items():
        header_parts.append(f"Bz @ {v} ({name})")
    
    print(f"\n  Bz thresholds (nT) by velocity (t = {T_EXPOSURE}h)")
    print("  " + "-" * 75)
    print(f"  {'Dose (mSv)':>12}  {'v=400':>10}  {'v=800':>10}  {'v=1500':>10}  {'v=2500':>10}")
    print(f"  {'':>12}  {'(Slow)':>10}  {'(Moderate)':>10}  {'(Fast)':>10}  {'(Extreme)':>10}")
    print("  " + "-" * 75)
    
    rows = []
    for dose in dose_points:
        bz_values = []
        parts = [f"  {dose:>9} mSv"]
        for v in vel_values:
            bz = bz_from_dose(dose, v)
            bz_values.append(round(bz, 1))
            parts.append(f"{bz:>10.1f}")
        print("  ".join(parts))
        rows.append([dose] + bz_values)
    
    # Key insight
    print(f"\n  KEY INSIGHT:")
    bz_slow = bz_from_dose(200, 400)
    bz_fast = bz_from_dose(200, 2500)
    print(f"    Extreme threshold (200 mSv):")
    print(f"      Slow CME (400):    |Bz| > {bz_slow:.1f} nT needed")
    print(f"      Extreme CME (2500): |Bz| > {bz_fast:.1f} nT needed")
    print(f"    → {bz_slow/bz_fast:.1f}× higher Bz needed at slow speed to reach same dose")
    
    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, 'bz_thresholds_multivelocity.csv')
    headers = ['Dose_mSv', 'Bz_v400', 'Bz_v800', 'Bz_v1500', 'Bz_v2500']
    write_csv(csv_path, headers, rows)
    
    return rows


# ============================================================================
# TASK 3: Dose Validation Matrix
# ============================================================================

def task3_dose_validation_matrix():
    """For each Bz boundary at v=800, calculate dose at all velocities."""
    print("\n" + "=" * 80)
    print("TASK 3: DOSE VALIDATION MATRIX")
    print("=" * 80)
    
    v_ref = 800
    vel_values = list(VELOCITIES.values())
    
    # Get Bz boundaries from severity thresholds
    bz_boundaries = set()
    for sev in SEVERITY_CLASSES:
        bz_boundaries.add(round(bz_from_dose(sev['dose_min'], v_ref), 1))
        if sev['dose_max'] is not None:
            bz_boundaries.add(round(bz_from_dose(sev['dose_max'], v_ref), 1))
    
    # Add extra values for completeness
    extra_bz = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75]
    for b in extra_bz:
        bz_boundaries.add(float(b))
    
    bz_sorted = sorted(bz_boundaries)
    
    print(f"\n  Dose (mSv) at each |Bz| across CME velocities (t = {T_EXPOSURE}h)")
    print("  " + "-" * 80)
    print(f"  {'|Bz| (nT)':>10}  {'v=400':>10}  {'v=800':>10}  {'v=1500':>10}  {'v=2500':>10}  {'Sev@800'}")
    print("  " + "-" * 80)
    
    rows = []
    for bz in bz_sorted:
        doses = []
        for v in vel_values:
            d = dose_from_bz(bz, v)
            doses.append(round(d, 1))
        
        # Severity at v=800
        d_800 = dose_from_bz(bz, 800)
        if d_800 < 10:
            sev = "Minimal"
        elif d_800 < 50:
            sev = "Low"
        elif d_800 < 100:
            sev = "Moderate"
        elif d_800 < 200:
            sev = "High"
        else:
            sev = "Extreme"
        
        print(f"  {bz:>10.1f}  {doses[0]:>10.1f}  {doses[1]:>10.1f}  {doses[2]:>10.1f}  {doses[3]:>10.1f}  {sev}")
        rows.append([bz] + doses)
    
    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, 'dose_validation_matrix.csv')
    headers = ['Bz_nT', 'Dose_v400', 'Dose_v800', 'Dose_v1500', 'Dose_v2500']
    write_csv(csv_path, headers, rows)
    
    # Cross-velocity severity shift analysis
    print(f"\n  CROSS-VELOCITY SEVERITY SHIFTS:")
    print("  " + "-" * 60)
    print(f"  A CME with |Bz|=15 nT:")
    for name, v in VELOCITIES.items():
        d = dose_from_bz(15, v)
        if d < 10: s = "Minimal"
        elif d < 50: s = "Low"
        elif d < 100: s = "Moderate"
        elif d < 200: s = "High"
        else: s = "Extreme"
        print(f"    at v={v:>5} km/s ({name:<8}): {d:>7.1f} mSv → {s}")
    
    return rows


# ============================================================================
# TASK 4: Bastille Day Verification
# ============================================================================

def task4_bastille_verification():
    """Verify calibration against Bastille Day 2000."""
    print("\n" + "=" * 80)
    print("TASK 4: BASTILLE DAY 2000 VERIFICATION")
    print("=" * 80)
    
    true_bz = -60.0
    pred_bz = -55.92
    v = 1674
    lit_dose = 1015.0  # Kim et al. 2015
    
    # Forward calculations
    dose_true = dose_from_bz(true_bz, v)
    dose_pred = dose_from_bz(pred_bz, v)
    
    # Errors
    err_true_vs_lit = (dose_true - lit_dose) / lit_dose * 100
    err_pred_vs_lit = (dose_pred - lit_dose) / lit_dose * 100
    err_pred_vs_true = (dose_pred - dose_true) / dose_true * 100
    
    print(f"""
  Event: Bastille Day 2000 (X5.7 flare, 2000-07-14)
  
  STEP-BY-STEP CALCULATION:
  
    D = K × |Bz|^α × √v × t
    D = {K} × |Bz|^{ALPHA} × √{v} × {T_EXPOSURE}
    
    √{v} = {np.sqrt(v):.4f}
    K × √v × t = {K} × {np.sqrt(v):.4f} × {T_EXPOSURE} = {K * np.sqrt(v) * T_EXPOSURE:.4f}
    
  FROM TRUE Bz ({true_bz} nT):
    |Bz|^{ALPHA} = {abs(true_bz)}^{ALPHA} = {abs(true_bz)**ALPHA:.2f}
    D = {K} × {abs(true_bz)**ALPHA:.2f} × {np.sqrt(v):.4f} × {T_EXPOSURE}
    D = {dose_true:.1f} mSv
    
  FROM PREDICTED Bz ({pred_bz} nT):
    |Bz|^{ALPHA} = {abs(pred_bz):.2f}^{ALPHA} = {abs(pred_bz)**ALPHA:.2f}
    D = {K} × {abs(pred_bz)**ALPHA:.2f} × {np.sqrt(v):.4f} × {T_EXPOSURE}
    D = {dose_pred:.1f} mSv
    
  LITERATURE VALUE: {lit_dose} mSv (Kim et al. 2015)
  
  COMPARISON:
    Dose(true Bz)  vs Literature:  {dose_true:.1f} vs {lit_dose:.0f} mSv  ({err_true_vs_lit:+.1f}%)
    Dose(pred Bz)  vs Literature:  {dose_pred:.1f} vs {lit_dose:.0f} mSv  ({err_pred_vs_lit:+.1f}%)
    Dose(pred)     vs Dose(true):  {dose_pred:.1f} vs {dose_true:.1f} mSv  ({err_pred_vs_true:+.1f}%)
""")
    
    # NASA limit check
    print(f"  NASA LIMIT CHECK:")
    nasa_limits = [('30-day', 250), ('Annual', 500), ('Career', 600)]
    for name, limit in nasa_limits:
        pct = dose_pred / limit * 100
        status = "EXCEEDED" if dose_pred > limit else "Within limit"
        print(f"    {name:<10}: {dose_pred:.0f}/{limit} mSv = {pct:.0f}%  → {status}")
    
    # Write CSV
    rows = [
        ['True_Bz', round(true_bz, 1), 'nT'],
        ['Predicted_Bz', round(pred_bz, 2), 'nT'],
        ['Velocity', v, 'km/s'],
        ['Exposure_Time', T_EXPOSURE, 'hours'],
        ['Dose_from_True_Bz', round(dose_true, 1), 'mSv'],
        ['Dose_from_Predicted_Bz', round(dose_pred, 1), 'mSv'],
        ['Literature_Value', lit_dose, 'mSv'],
        ['Error_True_vs_Lit', round(err_true_vs_lit, 1), '%'],
        ['Error_Pred_vs_Lit', round(err_pred_vs_lit, 1), '%'],
        ['Error_Pred_vs_True', round(err_pred_vs_true, 1), '%'],
        ['NASA_30day_Limit', 250, 'mSv'],
        ['NASA_30day_Exceeded_By', round(dose_pred - 250, 1), 'mSv'],
        ['Severity_Class', 'Extreme', ''],
        ['Severity_Confidence', 100.0, '%'],
    ]
    
    csv_path = os.path.join(OUTPUT_DIR, 'bastille_verification.csv')
    write_csv(csv_path, ['Parameter', 'Value', 'Unit'], rows)
    
    return dose_true, dose_pred


# ============================================================================
# FINAL MARKDOWN TABLE FOR WHITEPAPER
# ============================================================================

def generate_whitepaper_tables():
    """Generate publication-ready tables."""
    print("\n" + "=" * 80)
    print("WHITEPAPER-READY TABLES")
    print("=" * 80)
    
    v_ref = 800
    
    # Table 1: Primary severity classification
    print(f"""
  TABLE 1: HELIOS Radiation Severity Classification
  (Reference: v_CME = {v_ref} km/s, t = {T_EXPOSURE}h, deep-space unshielded)
  
  | Severity | Dose Range (mSv) | |Bz| Range (nT) | % NASA 30-day | Crew Response |
  |----------|------------------|-----------------|---------------|---------------|""")
    
    for sev in SEVERITY_CLASSES:
        bz_lo = bz_from_dose(sev['dose_min'], v_ref)
        if sev['dose_max'] is not None:
            bz_hi = bz_from_dose(sev['dose_max'], v_ref)
            dose_str = f"{sev['dose_min']}–{sev['dose_max']}"
            bz_str = f"{bz_lo:.1f}–{bz_hi:.1f}"
            pct = f"{sev['dose_min']/250*100:.0f}–{sev['dose_max']/250*100:.0f}%"
        else:
            dose_str = f">{sev['dose_min']}"
            bz_str = f">{bz_lo:.1f}"
            pct = f">{sev['dose_min']/250*100:.0f}%"
        
        print(f"  | {sev['name']:<8} | {dose_str:>16} | {bz_str:>15} | {pct:>13} | {sev['response']:<13} |")
    
    # Table 2: Multi-velocity reference
    print(f"""
  TABLE 2: |Bz| Thresholds by CME Velocity (nT)
  (D = {K} × |Bz|^{ALPHA} × √v × {T_EXPOSURE})
  
  | Dose (mSv) | v=400 km/s | v=800 km/s | v=1500 km/s | v=2500 km/s |
  |------------|------------|------------|-------------|-------------|""")
    
    for dose in [10, 50, 100, 200]:
        parts = [f"  | {dose:>10} |"]
        for v in [400, 800, 1500, 2500]:
            bz = bz_from_dose(dose, v)
            parts.append(f" {bz:>10.1f} |")
        print("".join(parts))
    
    # Table 3: Bastille Day summary
    dose_true = dose_from_bz(60, 1674)
    dose_pred = dose_from_bz(55.92, 1674)
    
    print(f"""
  TABLE 3: Bastille Day 2000 — End-to-End Validation
  
  | Parameter | Value | Unit |
  |-----------|-------|------|
  | CME Speed | 1674 | km/s |
  | True Bz (ACE) | -60.0 | nT |
  | NN Predicted Bz | -55.92 | nT |
  | Bz Error | 4.08 | nT (6.8%) |
  | Dose (true Bz) | {dose_true:.1f} | mSv |
  | Dose (predicted Bz) | {dose_pred:.1f} | mSv |
  | Literature dose | 1015 | mSv |
  | Pred. vs Literature | {(dose_pred-1015)/1015*100:+.1f} | % |
  | NASA 30-day limit | 250 | mSv |
  | Limit exceedance | {dose_pred/250*100:.0f} | % of limit |
  | Severity | Extreme | (100% conf.) |
""")

    # Table 4: Quick-reference operational table
    print(f"""
  TABLE 4: OPERATIONAL QUICK-REFERENCE
  (For real-time EVA Go/No-Go decisions)
  
  ┌───────────┬───────────────────────────────────────────────────────┐
  │           │              CME Speed (km/s)                        │
  │ Severity  ├──────────┬──────────┬───────────┬───────────────────┤
  │           │   400    │   800    │  1500     │  2500             │
  ├───────────┼──────────┼──────────┼───────────┼───────────────────┤""")
    
    for sev in SEVERITY_CLASSES:
        parts = [f"  │ {sev['name']:<9} │"]
        for v in [400, 800, 1500, 2500]:
            bz_lo = bz_from_dose(sev['dose_min'], v)
            if sev['dose_max'] is not None:
                bz_hi = bz_from_dose(sev['dose_max'], v)
                parts.append(f" {bz_lo:.0f}–{bz_hi:.0f} nT │")
            else:
                parts.append(f"  >{bz_lo:.0f} nT   │")
        # Pad last column
        line = "".join(parts)
        print(line)
    
    print(f"  └───────────┴──────────┴──────────┴───────────┴───────────────────┘")
    print(f"  Read as: |Bz| must exceed shown value to reach that severity level.")
    print(f"  All values assume t = {T_EXPOSURE}h EVA, deep-space (no magnetosphere).")


# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

def print_summary():
    """Print final summary with recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    v_ref = 800
    
    print(f"""
  CALIBRATION:
    K = {K} (median of 8 SPE events: Aug 1972, Bastille 2000, Halloween 2003,
              Jan 2005, Mar 1989, Sep 2017, May 2024, Carrington 1859)
    α = {ALPHA} (empirical Bz-energy scaling)
    
  REFERENCE THRESHOLDS (v = {v_ref} km/s):
    Low/Moderate boundary:   |Bz| = {bz_from_dose(50, v_ref):.1f} nT  (50 mSv)
    Moderate/High boundary:  |Bz| = {bz_from_dose(100, v_ref):.1f} nT  (100 mSv)
    High/Extreme boundary:   |Bz| = {bz_from_dose(200, v_ref):.1f} nT  (200 mSv)
    NASA 30-day limit:       |Bz| = {bz_from_dose(250, v_ref):.1f} nT  (250 mSv)
    
  BASTILLE DAY VERIFICATION:
    NN predicts Bz = -55.92 nT → Dose = {dose_from_bz(55.92, 1674):.1f} mSv
    Literature: 1015 mSv → Error: {(dose_from_bz(55.92, 1674)-1015)/1015*100:+.1f}%
    
  OPERATIONAL RECOMMENDATIONS:
    1. Use DOSE as primary classification metric (not Bz alone)
    2. Bz thresholds shift significantly with CME speed
    3. At v > 1500 km/s, even moderate |Bz| (~15 nT) crosses High threshold
    4. For EVA planning: always combine Bz + speed in dose formula
    5. Exposure time is linearly controllable — shortest EVA = safest
    
  FILES GENERATED:
    output/severity_table_v800.csv          — Primary severity table
    output/bz_thresholds_multivelocity.csv  — Multi-velocity Bz thresholds
    output/dose_validation_matrix.csv       — Cross-velocity dose matrix
    output/bastille_verification.csv        — Bastille Day verification
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print()
    print("█" * 80)
    print("██  HELIOS RADIATION SEVERITY TABLE GENERATOR")
    print("██  D(mSv) = 0.0132 × |Bz|^1.3 × √v_CME × t_exposure")
    print("█" * 80)
    
    task1_severity_table_v800()
    task2_multivelocity_analysis()
    task3_dose_validation_matrix()
    task4_bastille_verification()
    generate_whitepaper_tables()
    print_summary()
    
    print("Done. All 4 CSV files generated.\n")


if __name__ == "__main__":
    main()
