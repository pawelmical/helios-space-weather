"""
HELIOS Warning Generator
=========================
Generates structured JSON warnings and Markdown reports for crew safety.

Output formats:
    - JSON: Machine-readable warning with full telemetry
    - Markdown: Human-readable validation report

Crew Response Protocol (per NASA-STD-3001):
    - GREEN (Low):      Enhanced monitoring, continue operations
    - YELLOW (Moderate): Shelter advisory, relocate if available
    - ORANGE (High):     Mandatory shelter, suspend activities
    - RED (Extreme):     EVA abort, emergency shielding

Author: HELIOS Team
Date: February 2026
"""

import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import os

from NeuralNetwork_ML.config import SEVERITY_CONFIG


# Crew response protocols based on severity class
CREW_RESPONSES = {
    0: {  # Low (10-50 mSv, 4-20% NASA limit)
        "alert": "GREEN",
        "message": "Solar radiation event detected. Enhanced radiation monitoring recommended. Continue normal operations.",
        "actions": [
            "Activate personal dosimeters",
            "Log exposure start time",
            "Monitor radiation levels every 30 minutes",
            "Continue nominal operations"
        ],
        "critical": False
    },
    1: {  # Moderate (50-100 mSv, 20-40% NASA limit)
        "alert": "YELLOW",
        "message": "Moderate solar radiation event. Shelter advisory in effect. Relocate to shielded area if available.",
        "actions": [
            "Move to shielded area if available",
            "Suspend non-essential EVA operations",
            "Monitor dose rate continuously",
            "Prepare for possible shelter-in-place",
            "Report status to Mission Control"
        ],
        "critical": False
    },
    2: {  # High (100-250 mSv, 40-100% NASA limit)
        "alert": "ORANGE",
        "message": "HIGH RADIATION EVENT. MANDATORY SHELTER. Suspend all activities outside shielded areas.",
        "actions": [
            "MANDATORY: Enter radiation shelter immediately",
            "ABORT any ongoing EVA operations",
            "Seal all hatches to shelter module",
            "Activate backup life support systems",
            "Report shelter status to Mission Control",
            "Monitor personal dosimeter readings"
        ],
        "critical": True
    },
    3: {  # Extreme (>250 mSv, >100% NASA limit)
        "alert": "RED",
        "message": "EMERGENCY: EXTREME RADIATION EVENT. ALL EVA OPERATIONS MUST CEASE IMMEDIATELY. Deploy emergency shielding.",
        "actions": [
            "ABORT EVA IMMEDIATELY - priority emergency ingress",
            "Deploy all available emergency shielding",
            "Take shelter in deepest/most-shielded module",
            "Activate emergency dosimetry protocols",
            "Prepare for potential equipment damage",
            "Report EMERGENCY status to Mission Control",
            "Document crew locations and exposure estimates"
        ],
        "critical": True
    }
}


@dataclass
class CrewWarning:
    """Structured crew warning with actionable guidance."""
    alert_level: str            # "GREEN", "YELLOW", "ORANGE", "RED"
    message: str                # Human-readable warning message
    time_to_impact_hours: float # Estimated time until CME arrival
    recommended_actions: List[str]
    critical: bool              # True if immediate action required

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def generate_crew_warning(
    severity_class: int,
    severity_name: str,
    dose_mSv: float,
    tmr_status: str,
    transit_time_hours: float = 28.0
) -> CrewWarning:
    """
    Generate actionable crew warning based on severity and dose.

    Parameters
    ----------
    severity_class : int
        Severity class (0-3)
    severity_name : str
        Severity name ("Low", "Moderate", "High", "Extreme")
    dose_mSv : float
        Predicted radiation dose in millisieverts
    tmr_status : str
        TMR voting status ("FULL_FUSION", "EXTENDED_ANALYSIS", "ABORT")
    transit_time_hours : float
        Estimated CME transit time to Earth (default: 28 hours)

    Returns
    -------
    warning : CrewWarning
        Structured warning with actions
    """
    response = CREW_RESPONSES.get(severity_class, CREW_RESPONSES[3])

    # Adjust message and actions if TMR status is ABORT (unreliable prediction)
    message = response["message"]
    actions = response["actions"].copy()

    if tmr_status == "ABORT":
        message = "[TMR CONSENSUS FAILURE] " + message
        message += " WARNING: Prediction may be unreliable - use conservative action."
        actions.append("CAUTION: TMR voting failed - predictions may be unreliable")
        actions.append("Consider treating as one severity level higher")

    # Add dose context to message
    nasa_percent = (dose_mSv / SEVERITY_CONFIG['nasa_30day_limit_mSv']) * 100
    message += f" Predicted dose: {dose_mSv:.0f} mSv ({nasa_percent:.0f}% of NASA 30-day limit)."

    return CrewWarning(
        alert_level=response["alert"],
        message=message,
        time_to_impact_hours=transit_time_hours,
        recommended_actions=actions,
        critical=response["critical"]
    )


def generate_warning_json(
    event_info: Dict,
    ml_predictions: List[Dict],
    tmr_consensus: Dict,
    physical_model: Dict,
    crew_warning: CrewWarning,
    validation: Dict
) -> Dict:
    """
    Generate complete warning JSON matching the HELIOS MVP specification.

    Parameters
    ----------
    event_info : Dict
        Event metadata (name, timestamp, type, source_region, speed, width)
    ml_predictions : List[Dict]
        Individual satellite predictions from TMR
    tmr_consensus : Dict
        TMR voting consensus results
    physical_model : Dict
        Physical dosimetry calculation results
    crew_warning : CrewWarning
        Generated crew warning
    validation : Dict
        Validation metrics (ground truth comparison)

    Returns
    -------
    warning_json : Dict
        Complete warning in standard format
    """
    return {
        "helios_warning": {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),

            "event_info": event_info,

            "ml_ensemble": {
                "predictions": ml_predictions,
                "n_satellites": len(ml_predictions)
            },

            "tmr_consensus": tmr_consensus,

            "physical_model": physical_model,

            "crew_warning": crew_warning.to_dict(),

            "validation": validation
        }
    }


def save_warning_json(warning_json: Dict, output_path: str) -> str:
    """
    Save warning JSON to file.

    Parameters
    ----------
    warning_json : Dict
        Complete warning dictionary
    output_path : str
        Output file path

    Returns
    -------
    output_path : str
        Path where file was saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(warning_json, f, indent=2)

    return output_path


def generate_markdown_report(warning_json: Dict, output_path: str) -> str:
    """
    Generate human-readable Markdown validation report.

    Parameters
    ----------
    warning_json : Dict
        Complete warning dictionary
    output_path : str
        Output file path for Markdown

    Returns
    -------
    content : str
        Markdown content (also written to file)
    """
    w = warning_json["helios_warning"]

    lines = [
        "# HELIOS MVP Validation Report",
        "",
        f"**Generated:** {w['generated_at']}",
        f"**Version:** {w['version']}",
        "",
        "---",
        "",
        "## 1. Event Information",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Name** | {w['event_info'].get('name', 'N/A')} |",
        f"| **Timestamp** | {w['event_info'].get('timestamp', 'N/A')} |",
        f"| **Type** | {w['event_info'].get('type', 'N/A')} |",
        f"| **Source Region** | {w['event_info'].get('source_region', 'N/A')} |",
        f"| **CME Speed** | {w['event_info'].get('cme_speed_km_s', 'N/A')} km/s |",
        f"| **Angular Width** | {w['event_info'].get('angular_width_deg', 'N/A')}° |",
        "",
        "---",
        "",
        "## 2. ML Ensemble Predictions",
        "",
        "| Satellite | Bz (nT) | Uncertainty | Severity | Confidence |",
        "|-----------|---------|-------------|----------|------------|",
    ]

    for p in w["ml_ensemble"]["predictions"]:
        lines.append(
            f"| {p['satellite_id']} | {p['bz_mean']:.1f} | ±{p['bz_std']:.1f} nT | "
            f"{p['severity_name']} | {p['severity_confidence']*100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. TMR Consensus",
        "",
    ])

    tmr = w["tmr_consensus"]
    lines.extend([
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Vote Type** | {tmr['vote_type']} |",
        f"| **Status** | {tmr['status']} |",
        f"| **Consensus Bz** | {tmr['consensus_bz']:.1f} ±{tmr['consensus_bz_uncertainty']:.1f} nT |",
        f"| **Consensus Severity** | {tmr['consensus_severity_name']} (Class {tmr['consensus_severity']}) |",
        f"| **Bz Spread** | {tmr['agreement_bz_range']:.1f} nT |",
        f"| **Exact Agreement** | {'Yes' if tmr['agreement_severity_exact'] else 'No'} |",
        f"| **Within ±1 Tolerance** | {'Yes' if tmr['agreement_severity_tolerance'] else 'No'} |",
        "",
        "---",
        "",
        "## 4. Physical Model Validation",
        "",
    ])

    pm = w["physical_model"]
    lines.extend([
        f"**Dosimetry Formula:** D = K × |Bz|^α × √v × t",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| K (coefficient) | {pm['formula_params']['K']} |",
        f"| α (exponent) | {pm['formula_params']['alpha']} |",
        f"| t (exposure) | {pm['formula_params']['t_hours']} hours |",
        "",
        f"| Result | Value |",
        f"|--------|-------|",
        f"| **Calculated Dose** | {pm['dose_mSv']:.1f} mSv |",
        f"| **Physical Severity** | {pm['severity_name']} (Class {pm['severity_class']}) |",
        f"| **NASA 30-day Limit** | {pm['nasa_limit_percent']:.1f}% |",
        "",
        "---",
        "",
        "## 5. Crew Warning",
        "",
    ])

    cw = w["crew_warning"]
    alert_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴"}.get(cw['alert_level'], "⚪")

    lines.extend([
        f"### Alert Level: {alert_emoji} **{cw['alert_level']}**",
        "",
        f"**Critical:** {'YES - IMMEDIATE ACTION REQUIRED' if cw['critical'] else 'No'}",
        "",
        f"**Message:**",
        f"> {cw['message']}",
        "",
        f"**Time to Impact:** {cw['time_to_impact_hours']:.1f} hours",
        "",
        "### Recommended Actions:",
        "",
    ])

    for i, action in enumerate(cw["recommended_actions"], 1):
        lines.append(f"{i}. {action}")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Validation Metrics",
        "",
    ])

    val = w["validation"]
    bz_pass = val['bz_error'] < 10  # Target: <10 nT error
    sev_pass = val['severity_correct']
    consistency_pass = val['ml_physics_consistent']

    lines.extend([
        f"| Metric | Value | Status |",
        f"|--------|-------|--------|",
        f"| Ground Truth Bz | {val['ground_truth_bz']:.1f} nT | Reference |",
        f"| Predicted Bz | {val['predicted_bz']:.1f} nT | - |",
        f"| Bz Error | {val['bz_error']:.1f} nT ({val['bz_error_percent']:.1f}%) | {'✅ PASS' if bz_pass else '❌ FAIL'} |",
        f"| Severity Correct | {val['predicted_severity']} vs {val['ground_truth_severity']} | {'✅ PASS' if sev_pass else '❌ FAIL'} |",
        f"| ML-Physics Consistent | - | {'✅ PASS' if consistency_pass else '❌ FAIL'} |",
        "",
        "---",
        "",
        "## 7. Summary",
        "",
    ])

    all_pass = bz_pass and sev_pass and consistency_pass

    if all_pass:
        lines.extend([
            "### ✅ MVP VALIDATION SUCCESSFUL",
            "",
            "The HELIOS MVP successfully demonstrates:",
            "- 3 parallel ML inferences via Monte Carlo dropout",
            "- TMR voting with majority consensus",
            "- Physical model dose calculation",
            "- Automated crew warning generation",
            f"- Bastille Day 2000 prediction within acceptable error ({val['bz_error']:.1f} nT)",
        ])
    else:
        lines.extend([
            "### ⚠️ VALIDATION REQUIRES REVIEW",
            "",
            "Some validation criteria were not met:",
        ])
        if not bz_pass:
            lines.append(f"- Bz error ({val['bz_error']:.1f} nT) exceeds 10 nT target")
        if not sev_pass:
            lines.append(f"- Severity prediction incorrect")
        if not consistency_pass:
            lines.append(f"- ML and Physical model severities inconsistent")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by HELIOS MVP v1.0.0*",
    ])

    content = "\n".join(lines)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return content


if __name__ == "__main__":
    # Test warning generator
    print("=" * 60)
    print("Warning Generator Module - Test")
    print("=" * 60)

    # Test crew warnings for each severity
    print("\nCrew Warning Tests:")
    print("-" * 40)

    for severity in range(4):
        name = SEVERITY_CONFIG['class_names'][severity]
        warning = generate_crew_warning(
            severity_class=severity,
            severity_name=name,
            dose_mSv=[30, 75, 150, 400][severity],
            tmr_status="FULL_FUSION",
            transit_time_hours=28.0
        )
        print(f"\nSeverity {severity} ({name}):")
        print(f"  Alert: {warning.alert_level}")
        print(f"  Critical: {warning.critical}")
        print(f"  Message: {warning.message[:80]}...")
        print(f"  Actions: {len(warning.recommended_actions)} items")

    # Test TMR ABORT warning
    print("\n" + "-" * 40)
    print("TMR ABORT Warning:")
    abort_warning = generate_crew_warning(
        severity_class=3,
        severity_name="Extreme",
        dose_mSv=400,
        tmr_status="ABORT",
        transit_time_hours=28.0
    )
    print(f"  Alert: {abort_warning.alert_level}")
    print(f"  Message: {abort_warning.message[:100]}...")

    print("\nWarning generator tests completed!")
