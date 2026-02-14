#!/usr/bin/env python3
"""
HELIOS Dose Calculation — Practical Guide
===========================================
How to estimate radiation exposure for deep-space missions using
the HELIOS dosimetry framework.

Framework: D_deepspace = 0.0132 × |Bz|^1.3 × √v_CME × t_exposure

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
from NeuralNetwork_ML.severity import (
    calculate_dose,
    calculate_severity,
    bz_to_severity_class,
    DosimetryResult,
)

# ============================================================================
# METHOD 1: Direct Dose Calculation
# ============================================================================

def example_direct_dose():
    """Calculate dose directly using the formula."""
    print("=" * 80)
    print("METHOD 1: Direct Dose Calculation")
    print("=" * 80)
    print()

    # Example: Strong CME approaching
    bz = -35  # nT (negative = southward, geoeffective)
    v = 1200  # km/s
    t = 10    # hours (default worst-case EVA)

    dose = calculate_dose(bz_nT=bz, speed_km_s=v, exposure_hours=t)

    print(f"Input parameters:")
    print(f"  |Bz| = {abs(bz)} nT")
    print(f"  v_CME = {v} km/s")
    print(f"  t_exposure = {t} hours")
    print()
    print(f"Calculation:")
    print(f"  D = 0.0132 × {abs(bz)}^1.3 × √{v} × {t}")
    print(f"  D = 0.0132 × {abs(bz)**1.3:.1f} × {np.sqrt(v):.2f} × {t}")
    print(f"  D = {dose:.1f} mSv")
    print()

    if dose < 50:
        level = "Low"
    elif dose < 100:
        level = "Moderate"
    elif dose < 250:
        level = "High"
    else:
        level = "Extreme"

    pct = (dose / 250) * 100
    print(f"Severity:  {level}")
    print(f"NASA 30-day limit:  {pct:.0f}% of 250 mSv")
    print()
    if level == "Extreme":
        print(f"⚠️  EXCEEDS NASA 30-DAY LIMIT — EVA ABORT RECOMMENDED")
    print()


# ============================================================================
# METHOD 2: Full Severity Calculation
# ============================================================================

def example_full_severity():
    """Calculate dose + severity classification."""
    print("=" * 80)
    print("METHOD 2: Full Severity Calculation (One-Shot)")
    print("=" * 80)
    print()

    bz = -60  # Bastille Day parameters
    v = 1673
    t = 10

    result = calculate_severity(bz_nT=bz, speed_km_s=v, exposure_hours=t)

    print(f"Input: Bz={bz} nT, v={v} km/s, t={t}h")
    print()
    print("Output (DosimetryResult object):")
    print(f"  dose_mSv:      {result.dose_mSv:.1f}")
    print(f"  severity_class: {result.severity_class}  (0=Low, 1=Mod, 2=High, 3=Extreme)")
    print(f"  severity_name:  {result.severity_name}")
    print(f"  bz_nT:          {result.bz_nT:.1f}")
    print(f"  speed_km_s:     {result.speed_km_s:.0f}")
    print(f"  exposure_hours: {result.exposure_hours:.1f}")
    print()
    print(f"Summary: {result.severity_name} event → {result.dose_mSv:.1f} mSv")
    print()


# ============================================================================
# METHOD 3: Dose Implicit in Bz-to-Severity Classification
# ============================================================================

def example_bz_to_severity():
    """Severity classification (dose calculated internally)."""
    print("=" * 80)
    print("METHOD 3: Severity from Bz + Speed (Dose Internal)")
    print("=" * 80)
    print()

    bz = -30
    v = 800

    sev_class, sev_name = bz_to_severity_class(bz_nT=bz, speed_km_s=v)
    dose = calculate_dose(bz_nT=bz, speed_km_s=v)

    print(f"Input: Bz={bz} nT, v={v} km/s")
    print()
    print(f"Output:")
    print(f"  Severity class: {sev_class}  ({sev_name})")
    print(f"  Implied dose:   {dose:.1f} mSv")
    print()
    print(f"Note: bz_to_severity_class() computes dose internally,")
    print(f"      then classifies. This is Option B (dose-based).")
    print()


# ============================================================================
# PRACTICAL MISSION SCENARIOS
# ============================================================================

def mission_scenarios():
    """Real-world EVA planning examples."""
    print("=" * 80)
    print("PRACTICAL SCENARIOS: Mission Planning")
    print("=" * 80)
    print()

    scenarios = [
        {
            "name": "Weak CME (Normal Operations)",
            "bz": -15,
            "v": 600,
            "desc": "Minor activity — operations continue"
        },
        {
            "name": "Moderate CME (Advisory)",
            "bz": -20,
            "v": 800,
            "desc": "Strong event — shelter-in-place recommended"
        },
        {
            "name": "Strong CME (Action Required)",
            "bz": -35,
            "v": 1000,
            "desc": "Major event — EVA cancellation likely"
        },
        {
            "name": "Extreme CME (Emergency)",
            "bz": -60,
            "v": 1673,
            "desc": "Bastille Day level — historical extreme"
        },
    ]

    print(f"{'Scenario':<30} {'|Bz|':>6} {'v':>7} {'Dose(mSv)':>12} {'Sev':>10} {'Action':>20}")
    print("-" * 100)

    for sc in scenarios:
        dose = calculate_dose(bz_nT=sc["bz"], speed_km_s=sc["v"], exposure_hours=10)

        if dose < 50:
            sev = "Low"
            action = "Continue ops"
        elif dose < 100:
            sev = "Moderate"
            action = "Advisory"
        elif dose < 250:
            sev = "High"
            action = "Shelter-in-place"
        else:
            sev = "Extreme"
            action = "EVA Abort"

        print(
            f"{sc['name']:<30} {abs(sc['bz']):>6} {sc['v']:>7} "
            f"{dose:>12.1f} {sev:>10} {action:>20}"
        )

    print()


# ============================================================================
# EXPOSURE TIME TRADEOFFS
# ============================================================================

def exposure_time_sensitivity():
    """Show dose vs EVA duration."""
    print("=" * 80)
    print("EVA DURATION vs DOSE: Time is Your Shield")
    print("=" * 80)
    print()

    bz = -25  # Moderate storm
    v = 800

    print(f"Scenario: Bz={bz} nT, v={v} km/s during SPE")
    print()
    print(f"{'EVA Duration':>15} {'Dose (mSv)':>15} {'Severity':>12} {'% 30-day Limit':>15}")
    print("-" * 60)

    for hours in [0.5, 1, 2, 4, 6, 8, 10, 12]:
        dose = calculate_dose(bz_nT=bz, speed_km_s=v, exposure_hours=hours)

        if dose < 50:
            sev = "Low"
        elif dose < 100:
            sev = "Moderate"
        elif dose < 250:
            sev = "High"
        else:
            sev = "Extreme"

        pct = (dose / 250) * 100

        print(f"{hours:>10.1f} hours {dose:>15.1f} {sev:>12} {pct:>14.1f}%")

    print()
    print("KEY INSIGHT: Every hour saved ≈ 22.5 mSv reduction")
    print("  → Emergency abort at 2h instead of 10h saves 180 mSv (~71%)")
    print()


# ============================================================================
# SPEED SENSITIVITY: Why Fast CMEs Are More Dangerous
# ============================================================================

def speed_sensitivity():
    """Show dose scaling with CME speed."""
    print("=" * 80)
    print("SPEED SENSITIVITY: Faster CMEs ∝ Higher Dose")
    print("=" * 80)
    print()

    bz = -30  # Fixed Bz
    t = 10

    print(f"Fixed: Bz={bz} nT, t={t}h  (varying CME speed)")
    print()
    print(f"{'v_CME (km/s)':>15} {'Dose (mSv)':>15} {'Severity':>12} {'√v Factor':>15}")
    print("-" * 60)

    for v in [400, 600, 800, 1000, 1500, 2000]:
        dose = calculate_dose(bz_nT=bz, speed_km_s=v, exposure_hours=t)

        if dose < 50:
            sev = "Low"
        elif dose < 100:
            sev = "Moderate"
        elif dose < 250:
            sev = "High"
        else:
            sev = "Extreme"

        sqrt_v = np.sqrt(v)

        print(f"{v:>10} km/s {dose:>15.1f} {sev:>12} {sqrt_v:>15.2f}")

    print()
    print("FORMULA: D ∝ √v")
    print("  → Doubling speed from 800→1600 km/s increases dose by ~41%")
    print("  → Fast CMEs (v>1500) are operationally more dangerous")
    print()


# ============================================================================
# COMPARING OLD vs NEW CALIBRATION
# ============================================================================

def old_vs_new():
    """Show impact of K=0.0121 vs K=0.0132."""
    print("=" * 80)
    print("CALIBRATION EVOLUTION: Single-Event vs Multi-Event")
    print("=" * 80)
    print()

    bz = -60
    v = 1673

    K_old = 0.0121  # Single-event fit (Bastille Day only)
    K_new = 0.0132  # Multi-event median (8 SPEs)

    dose_old = K_old * (abs(bz) ** 1.3) * np.sqrt(v) * 10
    dose_new = K_new * (abs(bz) ** 1.3) * np.sqrt(v) * 10

    print(f"Event: Bastille Day 2000 (Bz={bz}, v={v})")
    print()
    print(f"OLD (K=0.0121):  {dose_old:.1f} mSv  (error: {((dose_old-1015)/1015)*100:+.1f}%)")
    print(f"NEW (K=0.0132):  {dose_new:.1f} mSv  (error: {((dose_new-1015)/1015)*100:+.1f}%)")
    print()
    print(f"Target:          1015 mSv")
    print()
    print("Improvement: K=0.0132 (median of 8 events) is more robust than K=0.0121")
    print("  → Better for uncertain parameters")
    print("  → Captures multi-event statistics")
    print("  → Still within ±25% for all historical SPEs")
    print()


# ============================================================================
# PRACTICAL CODE SNIPPETS
# ============================================================================

def code_snippets():
    """Show how to use in real code."""
    print("=" * 80)
    print("CODE SNIPPETS: How to Use in Mission Operations")
    print("=" * 80)
    print()

    print("# Snippet 1: One-line dose calculation")
    print("-" * 60)
    print("from NeuralNetwork_ML.severity import calculate_dose")
    print()
    print("bz_nT = -40  # From real-time DSCOVR/ACE data")
    print("v_km_s = 1200")
    print("eva_duration_h = 6")
    print()
    print("dose = calculate_dose(bz_nT, v_km_s, eva_duration_h)")
    print("print(f'Estimated dose: {dose:.1f} mSv')")
    print()

    print("\n# Snippet 2: Full severity decision")
    print("-" * 60)
    print("from NeuralNetwork_ML.severity import calculate_severity")
    print()
    print("result = calculate_severity(bz_nT=-45, speed_km_s=1500)")
    print()
    print("if result.severity_name == 'Extreme':")
    print("    print('EVA ABORT: Dose exceeds 250 mSv')")
    print("    print(f'Predicted dose: {result.dose_mSv:.1f} mSv')")
    print("else:")
    print("    print(f'Go/No-Go: {result.severity_name} — {result.dose_mSv:.1f} mSv')")
    print()

    print("\n# Snippet 3: Real-time GOES integration")
    print("-" * 60)
    print("def eva_go_nogo(bz_nT, v_km_s, planned_eva_hours):")
    print("    '''Make EVA decision based on predicted dose.'''")
    print("    from NeuralNetwork_ML.severity import calculate_dose")
    print()
    print("    dose = calculate_dose(bz_nT, v_km_s, planned_eva_hours)")
    print("    nasa_30day_limit = 250  # mSv")
    print()
    print("    if dose > nasa_30day_limit:")
    print("        return 'NO-GO', f'Dose {dose:.0f} mSv exceeds limit'")
    print("    elif dose > 100:")
    print("        return 'CAUTION', f'High risk — {dose:.0f} mSv'")
    print("    else:")
    print("        return 'GO', f'Safe — {dose:.0f} mSv'")
    print()
    print("# Real-time usage:")
    print("data = get_latest_solar_wind()  # From DSCOVR")
    print("status, msg = eva_go_nogo(data['bz'], data['v'], eva_hours=6)")
    print("send_mission_alert(status, msg)")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("█" * 80)
    print("██  HELIOS DOSE CALCULATION GUIDE")
    print("██  Deep-Space Astronaut Radiation Exposure Estimation")
    print("█" * 80)
    print()

    example_direct_dose()
    example_full_severity()
    example_bz_to_severity()
    mission_scenarios()
    exposure_time_sensitivity()
    speed_sensitivity()
    old_vs_new()
    code_snippets()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Formula:  D_deepspace = 0.0132 × |Bz|^1.3 × √v_CME × t_exposure")
    print()
    print("Three ways to calculate:")
    print("  1. calculate_dose(bz, v, t)  → dose value")
    print("  2. calculate_severity(bz, v)  → full result with severity")
    print("  3. bz_to_severity_class(bz, v)  → class + name (dose internal)")
    print()
    print("NASA limits (deep-space):")
    print("  30-day: 250 mSv  (HELIOS Extreme threshold)")
    print("  Annual: 500 mSv")
    print("  Career: 600 mSv")
    print()
    print("Key facts:")
    print("  • No magnetosphere: astronauts get FULL SEP flux")
    print("  • Dose ∝ √v: fast CMEs are much more dangerous")
    print("  • Dose ∝ t: every hour of EVA is critical")
    print("  • Dose ∝ |Bz|^1.3: Bz is proxy for CME energy")
    print()


if __name__ == "__main__":
    main()
