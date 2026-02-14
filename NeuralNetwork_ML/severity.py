"""
HELIOS Deep-Space Severity Classification
==========================================
Dose-based severity thresholds for astronaut radiation exposure
in UNSHIELDED deep space (no magnetosphere).

Physics:
    D_deepspace (mSv) = 0.0132 × |Bz|^1.3 × √v_CME × t_exposure

Where:
    - Bz: Southward IMF component (nT) — proxy for CME magnetic energy
    - v_CME: CME speed (km/s)
    - t_exposure: Exposure duration (hours)

Calibrated parameters:
    - K = 0.0132  (median of 8 historical SPE reconstructions)
    - α = 1.3     (empirical Bz-energy scaling exponent)
    - t = 10 h    (pessimistic worst-case EVA)

Severity classification (Option B — DOSE-BASED):
    - Low (0):      10-50 mSv    |  4-20% of NASA 30-day limit
    - Moderate (1): 50-100 mSv   | 20-40% of NASA 30-day limit
    - High (2):     100-250 mSv  | 40-100% of NASA 30-day limit
    - Extreme (3):  >250 mSv     | Exceeds NASA 30-day limit (250 mSv)

Key design decisions:
    - Classification by PREDICTED DOSE, not by Bz ranges
    - Bz is a proxy for CME energy, not a direct radiation driver
    - In deep space there is no magnetospheric shielding
    - Extreme threshold aligns with NASA-STD-3001 30-day limit

References:
    - NASA-STD-3001 Vol.1 Rev A (2022)
    - NCRP Report 132
    - Townsend et al. 2003
    - Kim et al. 2015
    - Cucinotta et al. 2010

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from NeuralNetwork_ML.config import SEVERITY_CONFIG, BZ_CONFIG


@dataclass
class DosimetryResult:
    """Container for dosimetry calculation results."""
    dose_mSv: float
    severity_class: int
    severity_name: str
    bz_nT: float
    speed_km_s: float
    exposure_hours: float


def calculate_dose(
    bz_nT: float,
    speed_km_s: float,
    exposure_hours: float = None
) -> float:
    """
    Calculate deep-space radiation dose from CME parameters.

    Formula: D = 0.0132 × |Bz|^1.3 × √v × t

    This gives the UNSHIELDED free-space dose.  No magnetospheric
    protection is assumed (deep-space environment).

    Parameters
    ----------
    bz_nT : float
        Southward Bz component (negative values expected).
        Acts as a proxy for CME magnetic energy content.
    speed_km_s : float
        CME speed (km/s)
    exposure_hours : float, optional
        Exposure duration (default: 10 hours, pessimistic)

    Returns
    -------
    dose_mSv : float
        Estimated deep-space radiation dose in millisieverts
    """
    if exposure_hours is None:
        exposure_hours = SEVERITY_CONFIG['t_exposure_hours']

    # Use absolute value of Bz (formula expects positive)
    bz_abs = abs(bz_nT)

    # Ensure non-negative speed
    speed = max(0, speed_km_s)

    # Minimum Bz threshold (below this, dose is negligible)
    if bz_abs < 5.0:
        return 0.0

    # Dosimetry formula with configurable parameters
    coeff = SEVERITY_CONFIG['dose_coefficient']
    exponent = SEVERITY_CONFIG['bz_exponent']

    dose = coeff * (bz_abs ** exponent) * np.sqrt(speed) * exposure_hours

    return dose


def dose_to_severity_class(dose_mSv: float) -> Tuple[int, str]:
    """
    Convert dose to severity class.

    Parameters
    ----------
    dose_mSv : float
        Radiation dose in millisieverts

    Returns
    -------
    class_idx : int
        Severity class (0-3)
    class_name : str
        Human-readable class name
    """
    thresholds = SEVERITY_CONFIG['dose_thresholds']
    names = SEVERITY_CONFIG['class_names']

    if dose_mSv < thresholds['low'][1]:  # < 50 mSv
        return 0, names[0]
    elif dose_mSv < thresholds['moderate'][1]:  # < 100 mSv
        return 1, names[1]
    elif dose_mSv < thresholds['high'][1]:  # < 250 mSv
        return 2, names[2]
    else:  # >= 250 mSv (exceeds NASA 30-day limit)
        return 3, names[3]


def bz_to_severity_class(bz_nT: float, speed_km_s: float = 800) -> Tuple[int, str]:
    """
    Classify severity from Bz (and optional speed) via dose calculation.

    Option B approach: computes dose first, then classifies by dose range.
    A default reference speed (800 km/s) is used when speed is unknown,
    representing a typical moderate CME.

    Parameters
    ----------
    bz_nT : float
        Southward Bz component (negative for geoeffective)
    speed_km_s : float, optional
        CME speed in km/s (default: 800 — typical moderate CME)

    Returns
    -------
    class_idx : int
        Severity class (0-3)
    class_name : str
        Human-readable class name
    """
    dose = calculate_dose(bz_nT, speed_km_s)
    return dose_to_severity_class(dose)


def calculate_severity(
    bz_nT: float,
    speed_km_s: float,
    exposure_hours: Optional[float] = None
) -> DosimetryResult:
    """
    Full severity calculation with dosimetry.

    Parameters
    ----------
    bz_nT : float
        Southward Bz component (negative for geoeffective)
    speed_km_s : float
        CME arrival speed
    exposure_hours : float, optional
        Exposure duration (default from config)

    Returns
    -------
    result : DosimetryResult
        Complete dosimetry and severity information
    """
    if exposure_hours is None:
        exposure_hours = SEVERITY_CONFIG['t_exposure_hours']

    dose = calculate_dose(bz_nT, speed_km_s, exposure_hours)
    severity_class, severity_name = dose_to_severity_class(dose)

    return DosimetryResult(
        dose_mSv=dose,
        severity_class=severity_class,
        severity_name=severity_name,
        bz_nT=bz_nT,
        speed_km_s=speed_km_s,
        exposure_hours=exposure_hours
    )


def invert_dose_for_bz(
    target_dose_mSv: float,
    speed_km_s: float,
    exposure_hours: float = None
) -> float:
    """
    Invert dosimetry formula to find Bz for target dose.

    D = coeff * |Bz|^exp * sqrt(v) * t
    |Bz| = (D / (coeff * sqrt(v) * t))^(1/exp)

    Returns negative Bz (southward).

    Parameters
    ----------
    target_dose_mSv : float
        Target dose in millisieverts
    speed_km_s : float
        CME speed in km/s
    exposure_hours : float, optional
        Exposure duration (default from config)

    Returns
    -------
    bz_nT : float
        Bz value (negative) that produces the target dose
    """
    if exposure_hours is None:
        exposure_hours = SEVERITY_CONFIG['t_exposure_hours']

    coeff = SEVERITY_CONFIG['dose_coefficient']
    exponent = SEVERITY_CONFIG['bz_exponent']

    denominator = coeff * np.sqrt(speed_km_s) * exposure_hours
    if denominator <= 0:
        return 0.0

    bz_abs = (target_dose_mSv / denominator) ** (1.0 / exponent)
    return -bz_abs  # Southward is negative


def get_bz_thresholds_for_speed(speed_km_s: float) -> dict:
    """
    Calculate Bz thresholds for a given CME speed.

    This allows severity classification to account for speed variation.

    Parameters
    ----------
    speed_km_s : float
        CME arrival speed

    Returns
    -------
    thresholds : dict
        Bz thresholds between severity classes
    """
    exposure = SEVERITY_CONFIG['t_exposure_hours']

    return {
        'low_moderate': invert_dose_for_bz(50, speed_km_s, exposure),
        'moderate_high': invert_dose_for_bz(100, speed_km_s, exposure),
        'high_extreme': invert_dose_for_bz(250, speed_km_s, exposure),
    }


def get_dose_summary(bz_nT: float, speed_km_s: float) -> str:
    """
    Generate a human-readable dose summary.

    Parameters
    ----------
    bz_nT : float
        Southward Bz component
    speed_km_s : float
        CME arrival speed

    Returns
    -------
    summary : str
        Formatted summary string
    """
    result = calculate_severity(bz_nT, speed_km_s)

    lines = [
        f"Bz: {result.bz_nT:.1f} nT",
        f"Speed: {result.speed_km_s:.0f} km/s",
        f"Exposure: {result.exposure_hours:.1f} hours (deep-space, unshielded)",
        f"Dose: {result.dose_mSv:.1f} mSv",
        f"Severity: {result.severity_name} (class {result.severity_class})",
        f"NASA 30-day limit: {result.dose_mSv / 250 * 100:.0f}%",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Test severity calculations
    print("=" * 60)
    print("HELIOS Deep-Space Severity Classification - Test")
    print("=" * 60)
    print("Framework: No magnetosphere. Unshielded free-space dose.")
    print("Classification: by PREDICTED DOSE (Option B).")

    # Test cases
    test_cases = [
        (-15, 600, "Moderate CME"),
        (-25, 800, "Strong CME"),
        (-40, 800, "Major CME"),
        (-60, 1673, "Bastille Day 2000"),
        (-75, 1200, "Extreme CME"),
    ]

    print("\nSeverity Classification Results:")
    print("-" * 70)
    print(f"{'Bz (nT)':>10} {'Speed':>10} {'Dose (mSv)':>12} {'Class':>8} {'Name':>12} {'Description'}")
    print("-" * 70)

    for bz, speed, desc in test_cases:
        result = calculate_severity(bz, speed)
        print(f"{result.bz_nT:>10.1f} {result.speed_km_s:>10.0f} "
              f"{result.dose_mSv:>12.1f} {result.severity_class:>8d} "
              f"{result.severity_name:>12} {desc}")

    # Test Bastille Day event
    print("\n" + "=" * 60)
    print("Bastille Day 2000 — Deep Space")
    print("=" * 60)
    print(get_dose_summary(-60, 1673))

    # Show Bz thresholds for different speeds (illustrative)
    print("\n" + "=" * 60)
    print("Illustrative Bz Thresholds by CME Speed")
    print("(NOT used for classification — dose-based only)")
    print("=" * 60)
    print(f"{'Speed (km/s)':>15} {'Low→Mod':>12} {'Mod→High':>12} {'High→Ext':>12}")
    print("-" * 55)

    for speed in [400, 600, 800, 1000, 1500]:
        thresh = get_bz_thresholds_for_speed(speed)
        print(f"{speed:>15.0f} {thresh['low_moderate']:>12.1f} "
              f"{thresh['moderate_high']:>12.1f} {thresh['high_extreme']:>12.1f}")
