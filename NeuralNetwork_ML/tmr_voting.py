"""
HELIOS Triple Modular Redundancy (TMR) Voting System
=====================================================
Implements redundant inference and voting logic for crew safety.

Architecture (3 onboard model inference passes via MC dropout):
    Model_A Inference ─┐
    Model_B Inference ─┼─> TMR Voting Diamond ─> Consensus
    Model_C Inference ─┘

Vote Types:
    3/3: Full Fusion (highest confidence)
    2/3: Extended Analysis (proceed with caution)
    1/3: Abort (system failure, use fallback)
    0/3: Abort (complete disagreement)

Tolerance Rules:
    Bz: ±10 nT tolerance for agreement
    Severity: ±1 class tolerance for majority (e.g., High and Extreme OK)

Author: HELIOS Team
Date: February 2026
"""

from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from enum import Enum

from NeuralNetwork_ML.config import SEVERITY_CONFIG, BZ_CONFIG


class TMRStatus(Enum):
    """TMR voting status codes."""
    FULL_FUSION = "FULL_FUSION"           # 3/3 agreement
    EXTENDED_ANALYSIS = "EXTENDED_ANALYSIS"  # 2/3 agreement
    ABORT = "ABORT"                        # <2/3 agreement


SEVERITY_NAMES = SEVERITY_CONFIG['class_names']  # ['Low', 'Moderate', 'High', 'Extreme']


@dataclass
class SatellitePrediction:
    """Single onboard model inference result (labeled as satellite prediction for operational context)."""
    satellite_id: str           # "Model_A", "Model_B", "Model_C"
    bz_mean: float              # Predicted Bz in nT
    bz_std: float               # Uncertainty (std dev)
    severity_class: int         # 0-3
    severity_name: str          # "Low", "Moderate", "High", "Extreme"
    severity_confidence: float  # Max softmax probability
    severity_probs: List[float] # All 4 class probabilities

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TMRConsensus:
    """TMR voting result."""
    consensus_bz: float                    # Consensus Bz value (nT)
    consensus_bz_uncertainty: float        # Std dev of predictions
    consensus_severity: int                # Voted severity class (0-3)
    consensus_severity_name: str           # "Low", "Moderate", "High", "Extreme"
    vote_type: str                         # "3/3", "2/3", "1/3", "0/3"
    status: str                            # "FULL_FUSION", "EXTENDED_ANALYSIS", "ABORT"
    agreement_bz_range: float              # Max spread in Bz predictions
    agreement_severity_exact: bool         # All 3 match exactly
    agreement_severity_tolerance: bool     # 2/3 within ±1 class
    individual_predictions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d


# ==============================================================================
# DEPRECATED LEGACY CODE - DO NOT USE IN PRODUCTION
# ==============================================================================
# The following functions were part of an early prototype (Jan-Feb 2026) that
# used Monte Carlo dropout to simulate ensemble behavior from a single model.
#
# REASON FOR DEPRECATION:
# This approach has been REPLACED by the production TMR system which uses
# three independently-trained models loaded from ensemble checkpoint (.pth).
# See run_complete_mvp.py for current implementation with load_ensemble().
#
# These functions are preserved for historical reference and transparency.
# ==============================================================================

# def enable_mc_dropout(model):
#     """
#     [DEPRECATED] Enable dropout layers for Monte Carlo inference.
#     """
#     for name, module in model.named_modules():
#         classname = module.__class__.__name__
#         if 'Dropout' in classname:
#             module.train()
#
#
# def run_mc_inference(
#     model,
#     feature_tensor,
#     bz_normalizer=None,
#     n_samples: int = 3,
#     device: str = 'cpu'
# ) -> List[SatellitePrediction]:
#     """
#     [DEPRECATED] Run Monte Carlo dropout inference.
#
#     This function simulated three models using dropout stochasticity.
#     REPLACED by true ensemble inference in production.
#     """
#     import torch
#     MODEL_IDS = ["Model_A", "Model_B", "Model_C"]
#     model.to(device)
#     model.eval()
#     enable_mc_dropout(model)
#     predictions = []
#     with torch.no_grad():
#         for i in range(n_samples):
#             bz_mean, bz_logvar, severity_logits = model(feature_tensor)
#             bz_norm = bz_mean.cpu().numpy().squeeze()
#             bz_logvar_val = bz_logvar.cpu().numpy().squeeze()
#             bz_std_norm = np.exp(0.5 * bz_logvar_val)
#             if bz_normalizer is None:
#                 bz_nT = float(bz_norm)
#                 bz_std_nT = float(bz_std_norm)
#             else:
#                 bz_nT = float(bz_normalizer.inverse_transform(np.array([bz_norm]))[0])
#                 bz_range = bz_normalizer.bz_max - bz_normalizer.bz_min
#                 bz_std_nT = float(bz_std_norm * bz_range)
#             probs = torch.softmax(severity_logits, dim=-1).cpu().numpy().squeeze()
#             sev_class = int(np.argmax(probs))
#             sev_conf = float(probs[sev_class])
#             predictions.append(SatellitePrediction(
#                 satellite_id=MODEL_IDS[i] if i < len(MODEL_IDS) else f"Model_{i}",
#                 bz_mean=float(bz_nT),
#                 bz_std=bz_std_nT,
#                 severity_class=sev_class,
#                 severity_name=SEVERITY_NAMES[sev_class],
#                 severity_confidence=sev_conf,
#                 severity_probs=probs.tolist()
#             ))
#     return predictions

# ==============================================================================
# END DEPRECATED CODE
# ==============================================================================


def tmr_vote(
    predictions: List[SatellitePrediction],
    bz_tolerance_nT: float = 10.0,
    severity_tolerance: int = 1
) -> TMRConsensus:
    """
    Perform TMR voting on satellite predictions.

    Voting rules (per primary_nominalops.drawio):
    - 3/3 exact severity match → FULL_FUSION (highest confidence)
    - 2/3 exact match OR all within ±1 tolerance → EXTENDED_ANALYSIS
    - <2/3 agreement → ABORT (use median, flag uncertainty)

    Bz Consensus:
    - 3/3: mean of all predictions
    - 2/3: mean of agreeing pair
    - <2/3: median (robust fallback)

    Parameters
    ----------
    predictions : List[SatellitePrediction]
        Three satellite predictions
    bz_tolerance_nT : float
        Tolerance for Bz agreement (default: 10 nT)
    severity_tolerance : int
        Tolerance for severity class (default: 1 class)

    Returns
    -------
    consensus : TMRConsensus
        Voting result with consensus values
    """
    if len(predictions) < 2:
        raise ValueError("TMR voting requires at least 2 predictions")

    # Extract values
    bz_values = np.array([p.bz_mean for p in predictions])
    severities = [p.severity_class for p in predictions]

    # Calculate Bz agreement metrics
    bz_range = float(np.max(bz_values) - np.min(bz_values))
    bz_within_tolerance = bz_range <= bz_tolerance_nT

    # Severity agreement metrics
    unique_severities = set(severities)
    severity_exact_3 = len(unique_severities) == 1

    # Find mode (most common severity)
    severity_mode = max(set(severities), key=severities.count)
    severity_majority_count = severities.count(severity_mode)

    # Check if all within tolerance of mode
    severity_within_tolerance = all(
        abs(s - severity_mode) <= severity_tolerance
        for s in severities
    )

    # Determine consensus based on voting rules
    if severity_exact_3:
        # Perfect agreement: 3/3
        vote_type = "3/3"
        status = TMRStatus.FULL_FUSION.value
        consensus_bz = float(np.mean(bz_values))

    elif severity_majority_count >= 2:
        # Majority agreement: 2/3
        vote_type = "2/3"
        status = TMRStatus.EXTENDED_ANALYSIS.value

        # Use mean of agreeing satellites
        agreeing_indices = [i for i, s in enumerate(severities) if s == severity_mode]
        consensus_bz = float(np.mean([bz_values[i] for i in agreeing_indices]))

    elif severity_within_tolerance:
        # All within tolerance but no exact majority: treat as 2/3 with caution
        vote_type = "2/3"
        status = TMRStatus.EXTENDED_ANALYSIS.value
        consensus_bz = float(np.mean(bz_values))

    else:
        # No agreement: abort
        vote_type = f"{severity_majority_count}/3"
        status = TMRStatus.ABORT.value
        consensus_bz = float(np.median(bz_values))

    return TMRConsensus(
        consensus_bz=consensus_bz,
        consensus_bz_uncertainty=float(np.std(bz_values)),
        consensus_severity=severity_mode,
        consensus_severity_name=SEVERITY_NAMES[severity_mode],
        vote_type=vote_type,
        status=status,
        agreement_bz_range=bz_range,
        agreement_severity_exact=severity_exact_3,
        agreement_severity_tolerance=severity_within_tolerance,
        individual_predictions=[p.to_dict() for p in predictions]
    )


def format_tmr_report(consensus: TMRConsensus) -> str:
    """
    Generate human-readable TMR voting report.

    Parameters
    ----------
    consensus : TMRConsensus
        Voting result

    Returns
    -------
    report : str
        Formatted report string
    """
    lines = [
        "=" * 60,
        "TMR VOTING REPORT",
        "=" * 60,
        "",
        "Individual Predictions:",
        "-" * 40,
    ]

    for p in consensus.individual_predictions:
        lines.append(
            f"  {p['satellite_id']}: Bz = {p['bz_mean']:.1f} ± {p['bz_std']:.1f} nT | "
            f"Severity = {p['severity_name']} ({p['severity_confidence']*100:.1f}%)"
        )

    lines.extend([
        "",
        "Consensus:",
        "-" * 40,
        f"  Vote Type: {consensus.vote_type}",
        f"  Status: {consensus.status}",
        f"  Consensus Bz: {consensus.consensus_bz:.1f} ± {consensus.consensus_bz_uncertainty:.1f} nT",
        f"  Consensus Severity: {consensus.consensus_severity_name} (class {consensus.consensus_severity})",
        "",
        "Agreement Metrics:",
        "-" * 40,
        f"  Bz Range (spread): {consensus.agreement_bz_range:.1f} nT",
        f"  Severity Exact Match: {'Yes' if consensus.agreement_severity_exact else 'No'}",
        f"  Severity Within ±1 Tolerance: {'Yes' if consensus.agreement_severity_tolerance else 'No'}",
        "=" * 60,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Test TMR voting with mock data
    print("=" * 60)
    print("TMR Voting Module - Test")
    print("=" * 60)

    # Test case 1: Perfect agreement (3/3)
    print("\nTest 1: Perfect Agreement (3/3)")
    test_predictions_1 = [
        SatellitePrediction("L1", -55.0, 5.0, 3, "Extreme", 0.92, [0.01, 0.02, 0.05, 0.92]),
        SatellitePrediction("L4", -56.0, 5.5, 3, "Extreme", 0.89, [0.02, 0.03, 0.06, 0.89]),
        SatellitePrediction("L5", -54.5, 4.8, 3, "Extreme", 0.91, [0.01, 0.02, 0.06, 0.91]),
    ]
    consensus_1 = tmr_vote(test_predictions_1)
    print(f"  Result: {consensus_1.vote_type} - {consensus_1.status}")
    print(f"  Consensus Bz: {consensus_1.consensus_bz:.1f} nT")

    # Test case 2: Majority agreement (2/3)
    print("\nTest 2: Majority Agreement (2/3)")
    test_predictions_2 = [
        SatellitePrediction("L1", -55.0, 5.0, 3, "Extreme", 0.90, [0.01, 0.02, 0.07, 0.90]),
        SatellitePrediction("L4", -56.0, 5.5, 3, "Extreme", 0.88, [0.02, 0.03, 0.07, 0.88]),
        SatellitePrediction("L5", -40.0, 6.0, 2, "High", 0.75, [0.05, 0.10, 0.75, 0.10]),
    ]
    consensus_2 = tmr_vote(test_predictions_2)
    print(f"  Result: {consensus_2.vote_type} - {consensus_2.status}")
    print(f"  Consensus Bz: {consensus_2.consensus_bz:.1f} nT")

    # Test case 3: Tolerance agreement (all within ±1)
    print("\nTest 3: Tolerance Agreement (within ±1)")
    test_predictions_3 = [
        SatellitePrediction("L1", -55.0, 5.0, 3, "Extreme", 0.90, [0.01, 0.02, 0.07, 0.90]),
        SatellitePrediction("L4", -50.0, 5.5, 2, "High", 0.85, [0.02, 0.03, 0.85, 0.10]),
        SatellitePrediction("L5", -52.0, 4.8, 2, "High", 0.80, [0.02, 0.08, 0.80, 0.10]),
    ]
    consensus_3 = tmr_vote(test_predictions_3)
    print(f"  Result: {consensus_3.vote_type} - {consensus_3.status}")
    print(f"  Consensus Bz: {consensus_3.consensus_bz:.1f} nT")

    # Test case 4: No agreement (ABORT)
    print("\nTest 4: No Agreement (ABORT)")
    test_predictions_4 = [
        SatellitePrediction("L1", -55.0, 5.0, 3, "Extreme", 0.70, [0.05, 0.10, 0.15, 0.70]),
        SatellitePrediction("L4", -30.0, 5.5, 1, "Moderate", 0.65, [0.10, 0.65, 0.15, 0.10]),
        SatellitePrediction("L5", -15.0, 4.8, 0, "Low", 0.60, [0.60, 0.20, 0.15, 0.05]),
    ]
    consensus_4 = tmr_vote(test_predictions_4)
    print(f"  Result: {consensus_4.vote_type} - {consensus_4.status}")
    print(f"  Consensus Bz: {consensus_4.consensus_bz:.1f} nT")

    # Print full report for test 1
    print("\n" + format_tmr_report(consensus_1))

    print("\nAll TMR voting tests completed!")
