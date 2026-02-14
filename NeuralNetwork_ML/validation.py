"""
HELIOS Validation Module
=========================
Bastille Day 2000 case study for model validation.

Target metrics:
    - Predicted Bz: -60 +/- 7 nT (MAE target: 7 nT)
    - Predicted Severity: Extreme (class 3)

Event Details (Bastille Day 2000):
    - Date: 2000-07-14 10:24 UT
    - Speed: 1674 km/s
    - Width: 360 deg (full halo)
    - Source: N22W07 (22 deg North, 7 deg West)
    - Bz measured (ACE): -60 nT
    - Reference: Nishino et al. 2006

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime

from NeuralNetwork_ML.config import VALIDATION_TARGETS, BZ_CONFIG, SEVERITY_CONFIG
from NeuralNetwork_ML.features import CMEFeatures, create_bastille_day_features
from NeuralNetwork_ML.preprocessing import FeatureNormalizer, BzNormalizer

# Check if PyTorch is available without importing it
def _check_torch():
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

HAS_TORCH = _check_torch()


# Bastille Day 2000 event parameters
BASTILLE_DAY_PARAMS = {
    'date': '2000-07-14',
    'eruption_time': datetime(2000, 7, 14, 10, 24),
    'speed_km_s': 1674,
    'angular_width_deg': 360,  # Full halo
    'source_latitude': 22,
    'source_longitude': 7,
    'measured_bz': -60,  # nT (ACE measurement)
    'expansion_rate': 2.5,
    'acceleration': -150,
    'detection_time_hours': 0.5,
    'brightness_asymmetry': 1.5,
    'observation_completeness': 1.0,
}


def get_bastille_day_feature_vector() -> np.ndarray:
    """
    Get the 16-dimensional feature vector for Bastille Day 2000.

    Returns
    -------
    features : np.ndarray
        Shape (16,) feature vector
    """
    features = create_bastille_day_features()
    return features.to_array()


def validate_model(
    model,
    feature_normalizer: FeatureNormalizer,
    bz_normalizer: BzNormalizer,
    device: str = 'cpu'
) -> Dict:
    """
    Validate model on Bastille Day 2000 event.

    Parameters
    ----------
    model : HELIOSDualHeadModel
        Trained model
    feature_normalizer : FeatureNormalizer
        Feature normalizer used during training
    bz_normalizer : BzNormalizer
        Bz normalizer used during training
    device : str
        Device for inference

    Returns
    -------
    results : dict
        Validation results including predictions and metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    # Import torch locally to avoid module-level conflict
    import torch

    model.eval()
    model.to(device)

    # Get Bastille Day features
    features = create_bastille_day_features()
    feature_array = features.to_array()

    # Normalize
    feature_norm = feature_normalizer.transform(feature_array)
    feature_tensor = torch.from_numpy(feature_norm).float().unsqueeze(0).to(device)

    # Predict
    predictions = model.predict(feature_tensor)

    # Denormalize Bz
    bz_pred_norm = predictions['bz_mean'].cpu().numpy().squeeze()
    bz_std_norm = predictions['bz_std'].cpu().numpy().squeeze()

    bz_pred = bz_normalizer.inverse_transform(np.array([bz_pred_norm]))[0]
    # Uncertainty scaling (approximate)
    bz_std = bz_std_norm * (BZ_CONFIG['bz_max'] - BZ_CONFIG['bz_min'])

    severity_class = predictions['severity_class'].cpu().numpy().squeeze()
    severity_probs = predictions['severity_probs'].cpu().numpy().squeeze()

    # Ground truth
    targets = VALIDATION_TARGETS['bastille_day_2000']
    bz_true = targets['expected_bz']
    severity_true = targets['expected_severity']

    # Calculate metrics
    bz_error = abs(bz_pred - bz_true)
    bz_error_percent = 100 * bz_error / abs(bz_true)
    severity_correct = (severity_class == severity_true)

    # Determine if validation passes
    bz_passes = bz_error <= targets['bz_tolerance']
    severity_passes = severity_correct

    results = {
        # Predictions
        'bz_predicted': float(bz_pred),
        'bz_uncertainty': float(bz_std),
        'severity_predicted': int(severity_class),
        'severity_probs': severity_probs.tolist(),

        # Ground truth
        'bz_true': float(bz_true),
        'severity_true': int(severity_true),

        # Metrics
        'bz_mae': float(bz_error),
        'bz_error_percent': float(bz_error_percent),
        'severity_correct': bool(severity_correct),

        # Pass/Fail
        'bz_passes': bool(bz_passes),
        'severity_passes': bool(severity_passes),
        'validation_passes': bool(bz_passes and severity_passes),

        # Feature vector (for analysis)
        'features': features.to_dict(),
    }

    return results


def print_validation_report(results: Dict):
    """
    Print formatted validation report.

    Parameters
    ----------
    results : dict
        Results from validate_model()
    """
    severity_names = SEVERITY_CONFIG['class_names']

    print("\n" + "=" * 65)
    print("BASTILLE DAY 2000 VALIDATION REPORT")
    print("=" * 65)

    print("\nEvent Parameters:")
    print("-" * 65)
    print(f"  Date: {BASTILLE_DAY_PARAMS['date']}")
    print(f"  Speed: {BASTILLE_DAY_PARAMS['speed_km_s']} km/s")
    print(f"  Width: {BASTILLE_DAY_PARAMS['angular_width_deg']} deg (full halo)")
    print(f"  Source: N{BASTILLE_DAY_PARAMS['source_latitude']}W{BASTILLE_DAY_PARAMS['source_longitude']}")

    print("\nPrediction Results:")
    print("-" * 65)
    print(f"{'Metric':<25} {'Predicted':<15} {'True':<15} {'Status':<10}")
    print("-" * 65)

    bz_status = "PASS" if results['bz_passes'] else "FAIL"
    print(f"{'Bz (nT)':<25} {results['bz_predicted']:>10.1f}     {results['bz_true']:>10.1f}     {bz_status:<10}")
    print(f"{'Bz uncertainty (nT)':<25} {results['bz_uncertainty']:>10.1f}")
    print(f"{'Bz MAE (nT)':<25} {results['bz_mae']:>10.1f}     {'<= 7.0':>10}     {bz_status:<10}")

    pred_name = severity_names[results['severity_predicted']]
    true_name = severity_names[results['severity_true']]
    sev_status = "PASS" if results['severity_passes'] else "FAIL"
    print(f"{'Severity class':<25} {pred_name:>10}     {true_name:>10}     {sev_status:<10}")

    print("\nSeverity Probabilities:")
    print("-" * 65)
    for i, (name, prob) in enumerate(zip(severity_names, results['severity_probs'])):
        marker = " <--" if i == results['severity_predicted'] else ""
        print(f"  {name:<12}: {prob:6.2%}{marker}")

    print("\n" + "-" * 65)
    overall = "PASS" if results['validation_passes'] else "FAIL"
    print(f"{'OVERALL VALIDATION:':<25} {overall}")
    print("=" * 65)


def compute_validation_metrics(
    model,
    test_features: np.ndarray,
    test_bz: np.ndarray,
    test_severity: np.ndarray,
    feature_normalizer: FeatureNormalizer,
    bz_normalizer: BzNormalizer,
    device: str = 'cpu'
) -> Dict:
    """
    Compute validation metrics on a test dataset.

    Parameters
    ----------
    model : HELIOSDualHeadModel
        Trained model
    test_features : np.ndarray
        Test features (n_samples, 16)
    test_bz : np.ndarray
        Test Bz values (n_samples,)
    test_severity : np.ndarray
        Test severity classes (n_samples,)
    feature_normalizer : FeatureNormalizer
        Feature normalizer
    bz_normalizer : BzNormalizer
        Bz normalizer
    device : str
        Device for inference

    Returns
    -------
    metrics : dict
        Validation metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    # Import torch locally to avoid module-level conflict
    import torch

    model.eval()
    model.to(device)

    # Normalize features
    features_norm = feature_normalizer.transform(test_features)
    features_tensor = torch.from_numpy(features_norm).float().to(device)

    # Predict
    with torch.no_grad():
        predictions = model.predict(features_tensor)

    # Denormalize Bz predictions
    bz_pred_norm = predictions['bz_mean'].cpu().numpy().squeeze()
    bz_pred = bz_normalizer.inverse_transform(bz_pred_norm)

    severity_pred = predictions['severity_class'].cpu().numpy()

    # Compute metrics
    bz_mae = np.mean(np.abs(bz_pred - test_bz))
    bz_rmse = np.sqrt(np.mean((bz_pred - test_bz) ** 2))
    bz_mape = np.mean(np.abs((bz_pred - test_bz) / test_bz)) * 100

    severity_accuracy = np.mean(severity_pred == test_severity)

    # Per-class accuracy
    class_accuracies = {}
    for c in range(4):
        mask = test_severity == c
        if np.sum(mask) > 0:
            class_accuracies[c] = np.mean(severity_pred[mask] == c)
        else:
            class_accuracies[c] = None

    return {
        'bz_mae': float(bz_mae),
        'bz_rmse': float(bz_rmse),
        'bz_mape': float(bz_mape),
        'severity_accuracy': float(severity_accuracy),
        'class_accuracies': class_accuracies,
        'n_samples': len(test_bz),
    }


def print_metrics_report(metrics: Dict):
    """Print formatted metrics report."""
    severity_names = SEVERITY_CONFIG['class_names']

    print("\n" + "=" * 50)
    print("VALIDATION METRICS REPORT")
    print("=" * 50)

    print(f"\nBz Regression Metrics (n={metrics['n_samples']}):")
    print("-" * 50)
    print(f"  MAE:  {metrics['bz_mae']:.2f} nT")
    print(f"  RMSE: {metrics['bz_rmse']:.2f} nT")
    print(f"  MAPE: {metrics['bz_mape']:.1f}%")

    print(f"\nSeverity Classification Metrics:")
    print("-" * 50)
    print(f"  Overall Accuracy: {metrics['severity_accuracy']:.2%}")
    print("\n  Per-class Accuracy:")
    for c, acc in metrics['class_accuracies'].items():
        if acc is not None:
            print(f"    {severity_names[c]:<12}: {acc:.2%}")
        else:
            print(f"    {severity_names[c]:<12}: N/A (no samples)")

    print("=" * 50)


if __name__ == "__main__":
    print("=" * 60)
    print("HELIOS Validation - Bastille Day 2000")
    print("=" * 60)

    # Show Bastille Day feature vector
    print("\nBastille Day 2000 Feature Vector:")
    print("-" * 60)

    features = create_bastille_day_features()
    feature_dict = features.to_dict()

    from NeuralNetwork_ML.config import FEATURE_NAMES, FEATURE_BOUNDS

    for name in FEATURE_NAMES:
        value = feature_dict[name]
        bounds = FEATURE_BOUNDS[name]
        print(f"  {name:25s}: {value:10.2f}  [{bounds[0]:.1f}, {bounds[1]:.1f}]")

    print("\nAs numpy array:")
    arr = features.to_array()
    print(f"  Shape: {arr.shape}")
    print(f"  Values: {arr}")

    # Expected output after training
    print("\n" + "=" * 60)
    print("EXPECTED OUTPUT AFTER TRAINING:")
    print("=" * 60)
    print(f"  Predicted Bz: -60 +/- 7 nT")
    print(f"  Predicted Severity: Extreme (class 3)")
    print(f"  Target MAE: <= 7 nT")

    # Show what the validation report would look like
    print("\n" + "=" * 60)
    print("SAMPLE VALIDATION REPORT (placeholder values):")
    print("=" * 60)

    sample_results = {
        'bz_predicted': -58.5,
        'bz_uncertainty': 5.2,
        'bz_true': -60.0,
        'bz_mae': 1.5,
        'bz_passes': True,
        'severity_predicted': 3,
        'severity_true': 3,
        'severity_probs': [0.01, 0.02, 0.12, 0.85],
        'severity_passes': True,
        'validation_passes': True,
    }

    print_validation_report(sample_results)
