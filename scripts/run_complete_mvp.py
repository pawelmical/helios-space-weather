#!/usr/bin/env python3
"""
HELIOS Complete MVP Pipeline
=============================
End-to-end demonstration with TMR voting for Bastille Day 2000.

Pipeline Steps:
    1. Load Bastille Day 2000 features (16D vector)
    2. Load trained ensemble (3 independent models with different seeds)
    3. Run inference on each model (L1, L4, L5 satellites)
    4. Perform TMR voting
    5. Calculate physical dose from consensus Bz
    6. Validate: compare ML severity vs dose-based severity
    7. Generate crew warning
    8. Output JSON warning + Markdown report

Usage:
    python scripts/run_complete_mvp.py
    python scripts/run_complete_mvp.py --model output/helios_final_model_proper.pth
    python scripts/run_complete_mvp.py --output output/mvp_results

Author: HELIOS Team
Date: February 2026
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
import numpy as np

# Path setup - ensure we can import from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import HELIOS modules (non-PyTorch first)
from NeuralNetwork_ML.features import create_bastille_day_features
from NeuralNetwork_ML.severity import calculate_dose, dose_to_severity_class
from NeuralNetwork_ML.config import SEVERITY_CONFIG, VALIDATION_TARGETS, BZ_THRESHOLDS, bz_to_severity
from NeuralNetwork_ML.tmr_voting import SatellitePrediction, tmr_vote
from NeuralNetwork_ML.warning_generator import (
    generate_crew_warning,
    generate_warning_json,
    save_warning_json,
    generate_markdown_report
)


# Severity names (BZ_THRESHOLDS and bz_to_severity imported from NeuralNetwork_ML.config
# as the single source of truth — see NeuralNetwork_ML/config.py)
SEVERITY_NAMES = ['Low', 'Moderate', 'High', 'Extreme']


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def bz_to_severity_probs(pred_bz: float, sigma: float) -> np.ndarray:
    """
    Derive severity probabilities by integrating Gaussian N(pred_bz, sigma^2)
    over each severity interval.
    """
    sigma = max(sigma, 0.5)  # floor to avoid degenerate distributions
    c30 = _normal_cdf(BZ_THRESHOLDS[0], pred_bz, sigma)
    c20 = _normal_cdf(BZ_THRESHOLDS[1], pred_bz, sigma)
    c10 = _normal_cdf(BZ_THRESHOLDS[2], pred_bz, sigma)
    probs = np.array([
        1.0 - c10,           # P(Low):      Bz > -10
        c10 - c20,           # P(Moderate): -20 < Bz <= -10
        c20 - c30,           # P(High):    -30 < Bz <= -20
        c30,                 # P(Extreme):  Bz <= -30
    ])
    return np.clip(probs, 0.0, 1.0)


def print_banner(title: str, width: int = 70):
    """Print a formatted section banner."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, title: str):
    """Print step header."""
    print(f"\n[STEP {step_num}] {title}")
    print("-" * 60)


def create_model_class():
    """
    Create the DualHeadBzModel class dynamically.
    This matches the architecture used in the trained models.
    """
    import torch
    import torch.nn as nn

    class DualHeadBzModel(nn.Module):
        """Dual-head: Bz regression (heteroscedastic) + severity classification."""

        def __init__(self, input_dim=16, hidden_dims=None, dropout=0.2, n_classes=4):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 256, 128, 64]

            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            self.encoder = nn.Sequential(*layers)

            self.bz_head = nn.Sequential(
                nn.Linear(hidden_dims[-1], 32), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(32, 2)   # (mean, log_variance)
            )

            self.sev_head = nn.Sequential(
                nn.Linear(hidden_dims[-1], 32), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(32, n_classes)
            )

        def forward(self, x):
            z = self.encoder(x)
            bz = self.bz_head(z)
            return bz[:, 0], bz[:, 1], self.sev_head(z)

    return DualHeadBzModel


def load_ensemble(model_path: str, device: str):
    """
    Load trained ensemble from checkpoint.

    Parameters
    ----------
    model_path : str
        Path to ensemble checkpoint (.pth file)
    device : str
        Compute device ('cpu' or 'cuda')

    Returns
    -------
    models : List[nn.Module]
        List of loaded models
    scaler : sklearn.preprocessing.StandardScaler
        Feature scaler from training
    model_config : dict
        Model configuration
    """
    import torch

    print(f"  Loading ensemble from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract components
    ensemble_states = checkpoint['ensemble_states']
    model_config = checkpoint['model_config']
    scaler = checkpoint['scaler']
    seeds = checkpoint.get('seeds', [42, 123, 456])

    print(f"  Model config: {model_config}")
    print(f"  Ensemble size: {len(ensemble_states)} models")
    print(f"  Seeds: {seeds}")

    # Create model class
    DualHeadBzModel = create_model_class()

    # Load each model
    models = []
    for i, state_dict in enumerate(ensemble_states):
        model = DualHeadBzModel(
            input_dim=model_config.get('input_dim', 16),
            hidden_dims=model_config.get('hidden_dims', [128, 256, 128, 64]),
            dropout=model_config.get('dropout', 0.2),
            n_classes=model_config.get('n_classes', 4)
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"    Loaded model {i+1} (seed={seeds[i] if i < len(seeds) else 'N/A'})")

    return models, scaler, model_config


def run_ensemble_inference(models, feature_tensor, scaler, device):
    """
    Run inference on all ensemble models.

    Parameters
    ----------
    models : List[nn.Module]
        Ensemble models
    feature_tensor : np.ndarray
        Raw feature vector (1, 16)
    scaler : dict or sklearn.preprocessing.StandardScaler
        Feature scaler (dict with 'mean' and 'std' keys, or sklearn object)
    device : str
        Compute device

    Returns
    -------
    predictions : List[SatellitePrediction]
        One prediction per model (satellite)
    """
    import torch

    SATELLITE_IDS = ["L1", "L4", "L5"]

    # Normalize features using the scaler from training
    if isinstance(scaler, dict):
        # Manual StandardScaler using dict format
        mean = np.array(scaler['mean'])
        std = np.array(scaler['std'])
        feature_norm = (feature_tensor.reshape(1, -1) - mean) / std
    else:
        # sklearn StandardScaler object
        feature_norm = scaler.transform(feature_tensor.reshape(1, -1))

    feature_gpu = torch.FloatTensor(feature_norm).to(device)

    predictions = []

    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            bz_mean, bz_logvar, sev_logits = model(feature_gpu)

            # Extract values
            bz_pred = float(bz_mean.cpu().numpy()[0])
            bz_logvar_val = float(bz_logvar.cpu().numpy()[0])
            bz_std = float(np.exp(0.5 * bz_logvar_val))

            # Derive severity from predicted Bz using Gaussian CDF
            probs = bz_to_severity_probs(bz_pred, bz_std)
            sev_class = int(np.argmax(probs))
            sev_conf = float(probs[sev_class])

            predictions.append(SatellitePrediction(
                satellite_id=SATELLITE_IDS[i] if i < len(SATELLITE_IDS) else f"SAT_{i}",
                bz_mean=bz_pred,
                bz_std=bz_std,
                severity_class=sev_class,
                severity_name=SEVERITY_NAMES[sev_class],
                severity_confidence=sev_conf,
                severity_probs=probs.tolist()
            ))

    return predictions


def run_complete_mvp(args):
    """
    Execute the complete HELIOS MVP pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    warning_json : Dict
        Complete warning output
    """
    import torch

    # =========================================================================
    # BANNER
    # =========================================================================
    print_banner("HELIOS COMPLETE MVP - TMR Voting Pipeline")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Device: {args.device}")

    # =========================================================================
    # STEP 1: Load Bastille Day Features
    # =========================================================================
    print_step(1, "Loading Bastille Day 2000 Features")

    features = create_bastille_day_features()
    feature_array = features.to_array()

    print(f"  Event: Bastille Day 2000 (2000-07-14 10:24 UT)")
    print(f"  CME Speed: {features.cme_speed:.0f} km/s")
    print(f"  Angular Width: {features.angular_width:.0f} deg")
    print(f"  Source Region: N{features.source_latitude:.0f}W{features.source_longitude:.0f}")
    print(f"  Feature Vector: shape={feature_array.shape}, dtype={feature_array.dtype}")

    # =========================================================================
    # STEP 2: Load Trained Ensemble
    # =========================================================================
    print_step(2, "Loading Trained Ensemble (3 Independent Models)")

    models, scaler, model_config = load_ensemble(args.model, args.device)

    # =========================================================================
    # STEP 3: Ensemble Inference (3 Models = 3 Satellites)
    # =========================================================================
    print_step(3, "Running Ensemble Inference (L1, L4, L5 Satellites)")

    predictions = run_ensemble_inference(models, feature_array, scaler, args.device)

    print(f"  Generated {len(predictions)} satellite predictions:")
    for p in predictions:
        print(f"    {p.satellite_id}: Bz = {p.bz_mean:.1f} +/- {p.bz_std:.1f} nT | "
              f"Severity = {p.severity_name} ({p.severity_confidence*100:.1f}%)")

    # =========================================================================
    # STEP 4: TMR Voting
    # =========================================================================
    print_step(4, "TMR Voting")

    consensus = tmr_vote(predictions, bz_tolerance_nT=10.0, severity_tolerance=1)

    print(f"  Vote Type: {consensus.vote_type}")
    print(f"  Status: {consensus.status}")
    print(f"  Consensus Bz: {consensus.consensus_bz:.1f} +/- {consensus.consensus_bz_uncertainty:.1f} nT")
    print(f"  Consensus Severity: {consensus.consensus_severity_name} (class {consensus.consensus_severity})")
    print(f"  Bz Spread: {consensus.agreement_bz_range:.1f} nT")
    print(f"  Exact Match: {consensus.agreement_severity_exact}")
    print(f"  Within +/-1 Tolerance: {consensus.agreement_severity_tolerance}")

    # =========================================================================
    # STEP 5: Physical Model Dose Calculation
    # =========================================================================
    print_step(5, "Physical Model (Dosimetry)")

    cme_speed = features.cme_speed
    exposure_hours = SEVERITY_CONFIG['t_exposure_hours']

    # Calculate dose from consensus Bz
    dose_mSv = calculate_dose(consensus.consensus_bz, cme_speed, exposure_hours)
    physical_class, physical_name = dose_to_severity_class(dose_mSv)
    nasa_30day_percent = (dose_mSv / SEVERITY_CONFIG['nasa_30day_limit_mSv']) * 100

    print(f"  Formula: D = {SEVERITY_CONFIG['dose_coefficient']} x |Bz|^{SEVERITY_CONFIG['bz_exponent']} x sqrt(v) x t")
    print(f"  Inputs:")
    print(f"    Bz = {consensus.consensus_bz:.1f} nT")
    print(f"    v = {cme_speed:.0f} km/s")
    print(f"    t = {exposure_hours:.0f} hours")
    print(f"  Results:")
    print(f"    Calculated Dose: {dose_mSv:.1f} mSv")
    print(f"    Physical Severity: {physical_name} (class {physical_class})")
    print(f"    NASA 30-day Limit: {nasa_30day_percent:.1f}%")

    # =========================================================================
    # STEP 6: Validation
    # =========================================================================
    print_step(6, "Validation Against Ground Truth")

    # Ground truth from VALIDATION_TARGETS
    ground_truth = VALIDATION_TARGETS['bastille_day_2000']
    ground_truth_bz = ground_truth['expected_bz']
    ground_truth_severity = ground_truth['expected_severity']

    # Calculate errors
    bz_error = abs(consensus.consensus_bz - ground_truth_bz)
    bz_error_percent = (bz_error / abs(ground_truth_bz)) * 100
    severity_correct = consensus.consensus_severity == ground_truth_severity

    # Check ML vs Physics consistency (within 1 class)
    ml_physics_consistent = abs(consensus.consensus_severity - physical_class) <= 1

    print(f"  Ground Truth Bz: {ground_truth_bz:.1f} nT")
    print(f"  Predicted Bz: {consensus.consensus_bz:.1f} nT")
    print(f"  Bz Error: {bz_error:.1f} nT ({bz_error_percent:.1f}%)")
    print(f"  Target Error: <{ground_truth['bz_tolerance']:.0f} nT")
    print(f"  Bz Validation: {'PASS' if bz_error < ground_truth['bz_tolerance'] else 'FAIL'}")
    print()
    print(f"  Ground Truth Severity: {ground_truth_severity} (Extreme)")
    print(f"  Predicted Severity: {consensus.consensus_severity} ({consensus.consensus_severity_name})")
    print(f"  Severity Validation: {'PASS' if severity_correct else 'FAIL'}")
    print()
    print(f"  ML-Physics Consistency: {'PASS' if ml_physics_consistent else 'FAIL'}")

    # =========================================================================
    # STEP 7: Generate Crew Warning
    # =========================================================================
    print_step(7, "Generating Crew Warning")

    # Estimate transit time (CME travel time to Earth)
    distance_km = 1.496e8  # 1 AU in km
    transit_time_hours = distance_km / (cme_speed * 3600)

    crew_warning = generate_crew_warning(
        severity_class=consensus.consensus_severity,
        severity_name=consensus.consensus_severity_name,
        dose_mSv=dose_mSv,
        tmr_status=consensus.status,
        transit_time_hours=transit_time_hours
    )

    print(f"  Alert Level: {crew_warning.alert_level}")
    print(f"  Critical: {'YES' if crew_warning.critical else 'No'}")
    print(f"  Time to Impact: {crew_warning.time_to_impact_hours:.1f} hours")
    print(f"  Message: {crew_warning.message[:100]}...")
    print(f"  Recommended Actions ({len(crew_warning.recommended_actions)}):")
    for i, action in enumerate(crew_warning.recommended_actions[:4], 1):
        print(f"    {i}. {action}")
    if len(crew_warning.recommended_actions) > 4:
        print(f"    ... and {len(crew_warning.recommended_actions) - 4} more")

    # =========================================================================
    # STEP 8: Generate Output Files
    # =========================================================================
    print_step(8, "Generating Output Files")

    os.makedirs(args.output, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build complete warning JSON
    warning_json = generate_warning_json(
        event_info={
            "name": "Bastille Day 2000",
            "timestamp": "2000-07-14T10:24:00Z",
            "type": "X5.7 Flare + Full Halo CME",
            "source_region": f"N{features.source_latitude:.0f}W{features.source_longitude:.0f}",
            "cme_speed_km_s": float(cme_speed),
            "angular_width_deg": float(features.angular_width)
        },
        ml_predictions=[p.to_dict() for p in predictions],
        tmr_consensus=consensus.to_dict(),
        physical_model={
            "dose_mSv": float(dose_mSv),
            "severity_class": int(physical_class),
            "severity_name": physical_name,
            "nasa_limit_percent": float(nasa_30day_percent),
            "formula_params": {
                "K": SEVERITY_CONFIG['dose_coefficient'],
                "alpha": SEVERITY_CONFIG['bz_exponent'],
                "t_hours": float(exposure_hours)
            }
        },
        crew_warning=crew_warning,
        validation={
            "ground_truth_bz": float(ground_truth_bz),
            "ground_truth_severity": int(ground_truth_severity),
            "predicted_bz": float(consensus.consensus_bz),
            "predicted_severity": int(consensus.consensus_severity),
            "bz_error": float(bz_error),
            "bz_error_percent": float(bz_error_percent),
            "severity_correct": bool(severity_correct),
            "ml_physics_consistent": bool(ml_physics_consistent)
        }
    )

    # Save JSON
    json_path = os.path.join(args.output, f"warning_{timestamp_str}.json")
    save_warning_json(warning_json, json_path)
    print(f"  JSON Warning: {json_path}")

    # Save Markdown report
    md_path = os.path.join(args.output, f"validation_report_{timestamp_str}.md")
    generate_markdown_report(warning_json, md_path)
    print(f"  Markdown Report: {md_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_banner("MVP EXECUTION COMPLETE")

    all_pass = (bz_error < ground_truth['bz_tolerance']) and severity_correct and ml_physics_consistent

    print(f"""
  RESULTS SUMMARY
  ---------------
  TMR Status:       {consensus.status}
  Vote Type:        {consensus.vote_type}

  Bz Prediction:    {consensus.consensus_bz:.1f} nT (true: {ground_truth_bz:.1f} nT)
  Bz Error:         {bz_error:.1f} nT ({bz_error_percent:.1f}%)
  Bz Validation:    {'PASS' if bz_error < ground_truth['bz_tolerance'] else 'FAIL'}

  Radiation Dose:   {dose_mSv:.1f} mSv
  NASA 30-day:      {nasa_30day_percent:.1f}%

  ML Severity:      {consensus.consensus_severity_name} (class {consensus.consensus_severity})
  Physics Severity: {physical_name} (class {physical_class})
  Severity Match:   {'PASS' if severity_correct else 'FAIL'}
  Consistency:      {'PASS' if ml_physics_consistent else 'FAIL'}

  Alert Level:      {crew_warning.alert_level}
  Critical:         {'YES' if crew_warning.critical else 'No'}

  OVERALL:          {'MVP VALIDATION SUCCESSFUL' if all_pass else 'REVIEW REQUIRED'}

  OUTPUT FILES
  ------------
  {json_path}
  {md_path}
""")

    return warning_json


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="HELIOS Complete MVP Pipeline with TMR Voting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_complete_mvp.py
  python scripts/run_complete_mvp.py --model output/helios_final_model_proper.pth
  python scripts/run_complete_mvp.py --output output/mvp_results --device cuda
        """
    )

    # Model path
    default_model = os.path.join(PROJECT_ROOT, "output", "helios_final_model_proper.pth")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=default_model,
        help=f"Path to trained model checkpoint (default: {default_model})"
    )

    # Output directory
    default_output = os.path.join(PROJECT_ROOT, "output", "mvp_results")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=default_output,
        help=f"Output directory for results (default: {default_output})"
    )

    # Device
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Compute device (cpu/cuda). Auto-detected if not specified."
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print(f"\nAvailable models in output/:")
        output_dir = os.path.join(PROJECT_ROOT, "output")
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        sys.exit(1)

    # Run pipeline
    try:
        warning_json = run_complete_mvp(args)
        return 0
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
