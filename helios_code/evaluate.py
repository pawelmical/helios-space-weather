"""
HELIOS Validation - Evaluation Metrics Module
===============================================
Metrics for detection and prediction performance.

Features:
- Confusion matrix computation
- POD (Probability of Detection) / Sensitivity
- FAR (False Alarm Rate)
- ROC curves and AUC
- Comparison: L1-only vs HELIOS multi-view

Author: HELIOS Team
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try importing sklearn for ROC/AUC
try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix as sk_confusion_matrix  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ConfusionMetrics:
    """Container for confusion matrix and derived metrics."""
    TP: int  # True Positives
    FN: int  # False Negatives (missed detections)
    FP: int  # False Positives (false alarms)
    TN: int  # True Negatives
    
    @property
    def POD(self) -> float:
        """Probability of Detection (Sensitivity, Recall)."""
        total_positives = self.TP + self.FN
        return self.TP / total_positives if total_positives > 0 else 0.0
    
    @property
    def FAR(self) -> float:
        """False Alarm Rate (1 - Precision)."""
        total_alarms = self.TP + self.FP
        return self.FP / total_alarms if total_alarms > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        total_alarms = self.TP + self.FP
        return self.TP / total_alarms if total_alarms > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)."""
        total_negatives = self.TN + self.FP
        return self.TN / total_negatives if total_negatives > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.POD == 0:
            return 0.0
        return 2 * (self.precision * self.POD) / (self.precision + self.POD)
    
    @property
    def CSI(self) -> float:
        """Critical Success Index (Threat Score)."""
        denom = self.TP + self.FN + self.FP
        return self.TP / denom if denom > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            'TP': self.TP,
            'FN': self.FN,
            'FP': self.FP,
            'TN': self.TN,
            'POD': self.POD,
            'FAR': self.FAR,
            'precision': self.precision,
            'specificity': self.specificity,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'CSI': self.CSI
        }


def compute_confusion(
    ground_truth: List[bool],
    predictions: List[bool]
) -> ConfusionMetrics:
    """
    Compute confusion matrix from ground truth and predictions.
    
    Parameters
    ----------
    ground_truth : list of bool
        True labels (True = CME event occurred)
    predictions : list of bool
        Predicted labels (True = CME detected)
        
    Returns
    -------
    metrics : ConfusionMetrics
        Confusion matrix and derived metrics
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have same length")
    
    TP = sum(1 for gt, pred in zip(ground_truth, predictions) if gt and pred)
    FN = sum(1 for gt, pred in zip(ground_truth, predictions) if gt and not pred)
    FP = sum(1 for gt, pred in zip(ground_truth, predictions) if not gt and pred)
    TN = sum(1 for gt, pred in zip(ground_truth, predictions) if not gt and not pred)
    
    return ConfusionMetrics(TP=TP, FN=FN, FP=FP, TN=TN)


def compute_confusion_from_events(
    events: List[Dict],
    detections: Dict[str, bool],
    key_event_id: str = 'event_id',
    key_has_cme: str = 'has_cme'
) -> ConfusionMetrics:
    """
    Compute confusion matrix from event list and detection results.
    
    Parameters
    ----------
    events : list of dict
        Event list with event_id and has_cme flag
    detections : dict
        Mapping event_id -> detected (bool)
    key_event_id : str
        Key for event ID in event dict
    key_has_cme : str
        Key for ground truth CME flag
        
    Returns
    -------
    metrics : ConfusionMetrics
    """
    ground_truth = []
    predictions = []
    
    for event in events:
        event_id = event[key_event_id]
        has_cme = event.get(key_has_cme, True)  # Default: assume CME
        detected = detections.get(event_id, False)
        
        ground_truth.append(has_cme)
        predictions.append(detected)
    
    return compute_confusion(ground_truth, predictions)


def compute_roc_curve(
    ground_truth: List[bool],
    scores: List[float],
    n_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.
    
    Parameters
    ----------
    ground_truth : list of bool
        True labels
    scores : list of float
        Detection confidence scores (0-1)
    n_thresholds : int
        Number of threshold values to evaluate
        
    Returns
    -------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates (POD)
    thresholds : np.ndarray
        Threshold values
    auc_value : float
        Area under the ROC curve
    """
    if SKLEARN_AVAILABLE:
        fpr, tpr, thresholds = roc_curve(ground_truth, scores)
        auc_value = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_value
    else:
        # Manual implementation
        thresholds = np.linspace(0, 1, n_thresholds)
        fpr_list = []
        tpr_list = []
        
        for thresh in thresholds:
            predictions = [s >= thresh for s in scores]
            metrics = compute_confusion(ground_truth, predictions)
            tpr_list.append(metrics.POD)
            fpr_list.append(1 - metrics.specificity)
        
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        
        # Compute AUC using trapezoidal rule
        # Sort by FPR
        sorted_idx = np.argsort(fpr)
        fpr_sorted = fpr[sorted_idx]
        tpr_sorted = tpr[sorted_idx]
        
        # Use scipy.integrate.trapezoid if available, else simple implementation
        try:
            from scipy.integrate import trapezoid
            auc_value = trapezoid(tpr_sorted, fpr_sorted)
        except ImportError:
            # Simple trapezoidal integration
            auc_value = 0.0
            for i in range(1, len(fpr_sorted)):
                auc_value += 0.5 * (tpr_sorted[i] + tpr_sorted[i-1]) * (fpr_sorted[i] - fpr_sorted[i-1])
        
        return fpr, tpr, thresholds, auc_value


def compare_detection_modes(
    events: List[Dict],
    l1_detections: Dict[str, bool],
    helios_detections: Dict[str, bool],
    l1_scores: Optional[Dict[str, float]] = None,
    helios_scores: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Compare L1-only vs HELIOS multi-view detection performance.
    
    Parameters
    ----------
    events : list of dict
        Event list with ground truth
    l1_detections : dict
        L1-only detection results {event_id: detected}
    helios_detections : dict
        HELIOS multi-view detection results
    l1_scores : dict, optional
        L1 confidence scores for ROC
    helios_scores : dict, optional
        HELIOS confidence scores for ROC
        
    Returns
    -------
    comparison : dict
        Comparison metrics for both modes
    """
    ground_truth = [event.get('has_cme', True) for event in events]
    
    # L1 metrics
    l1_preds = [l1_detections.get(e['event_id'], False) for e in events]
    l1_metrics = compute_confusion(ground_truth, l1_preds)
    
    # HELIOS metrics
    helios_preds = [helios_detections.get(e['event_id'], False) for e in events]
    helios_metrics = compute_confusion(ground_truth, helios_preds)
    
    comparison = {
        'L1_only': l1_metrics.to_dict(),
        'HELIOS': helios_metrics.to_dict(),
        'improvement': {
            'POD_improvement': helios_metrics.POD - l1_metrics.POD,
            'FAR_reduction': l1_metrics.FAR - helios_metrics.FAR,
            'accuracy_improvement': helios_metrics.accuracy - l1_metrics.accuracy,
            'F1_improvement': helios_metrics.f1_score - l1_metrics.f1_score
        }
    }
    
    # Add ROC/AUC if scores available
    if l1_scores is not None:
        l1_score_list = [l1_scores.get(e['event_id'], 0.0) for e in events]
        _, _, _, l1_auc = compute_roc_curve(ground_truth, l1_score_list)
        comparison['L1_only']['AUC'] = l1_auc
    
    if helios_scores is not None:
        helios_score_list = [helios_scores.get(e['event_id'], 0.0) for e in events]
        _, _, _, helios_auc = compute_roc_curve(ground_truth, helios_score_list)
        comparison['HELIOS']['AUC'] = helios_auc
        
        if l1_scores is not None:
            comparison['improvement']['AUC_improvement'] = helios_auc - l1_auc
    
    return comparison


def create_detection_report(
    events: List[Dict],
    instrument_detections: Dict[str, Dict[str, bool]],
    instrument_scores: Optional[Dict[str, Dict[str, float]]] = None
):
    """
    Create detection report table per instrument.
    
    Parameters
    ----------
    events : list of dict
        Event list with ground truth
    instrument_detections : dict
        {instrument_name: {event_id: detected}}
    instrument_scores : dict, optional
        {instrument_name: {event_id: confidence}}
        
    Returns
    -------
    df : pd.DataFrame
        Detection report per instrument
    """
    import pandas as pd
    
    ground_truth = [event.get('has_cme', True) for event in events]
    
    rows = []
    
    for instrument, detections in instrument_detections.items():
        predictions = [detections.get(e['event_id'], False) for e in events]
        metrics = compute_confusion(ground_truth, predictions)
        
        row = {
            'instrument': instrument,
            **metrics.to_dict()
        }
        
        # Add AUC if scores available
        if instrument_scores and instrument in instrument_scores:
            scores = [instrument_scores[instrument].get(e['event_id'], 0.0) 
                     for e in events]
            _, _, _, auc_val = compute_roc_curve(ground_truth, scores)
            row['AUC'] = auc_val
        
        rows.append(row)
    
    # Add combined HELIOS row (multi-instrument OR logic)
    helios_preds = []
    for event in events:
        event_id = event['event_id']
        detected = any(
            dets.get(event_id, False) 
            for dets in instrument_detections.values()
        )
        helios_preds.append(detected)
    
    helios_metrics = compute_confusion(ground_truth, helios_preds)
    rows.append({
        'instrument': 'HELIOS_combined',
        **helios_metrics.to_dict()
    })
    
    return pd.DataFrame(rows)


def evaluate_arrival_predictions(
    events: List[Dict],
    predictions: Dict[str, Dict]
) -> Dict:
    """
    Evaluate arrival time prediction accuracy.
    
    Parameters
    ----------
    events : list of dict
        Event list with actual_arrival_hours
    predictions : dict
        {event_id: {pred_arrival_h, pred_arrival_16_h, pred_arrival_84_h}}
        
    Returns
    -------
    metrics : dict
        Prediction performance metrics
    """
    errors = []
    abs_errors = []
    percent_errors = []
    within_uncertainty = 0
    
    for event in events:
        event_id = event['event_id']
        actual = event.get('actual_arrival_hours')
        
        if actual is None or event_id not in predictions:
            continue
        
        pred = predictions[event_id]
        pred_median = pred.get('pred_arrival_median_h', pred.get('pred_arrival_h'))
        
        error = pred_median - actual
        errors.append(error)
        abs_errors.append(abs(error))
        percent_errors.append(abs(error) / actual * 100)
        
        # Check if actual is within ensemble range
        pred_16 = pred.get('pred_arrival_16_h', pred_median - 5)
        pred_84 = pred.get('pred_arrival_84_h', pred_median + 5)
        if pred_16 <= actual <= pred_84:
            within_uncertainty += 1
    
    n = len(errors)
    
    if n == 0:
        return {'n_events': 0}
    
    return {
        'n_events': n,
        'mean_error_h': np.mean(errors),
        'mean_abs_error_h': np.mean(abs_errors),
        'median_abs_error_h': np.median(abs_errors),
        'std_error_h': np.std(errors),
        'rmse_h': np.sqrt(np.mean(np.array(errors)**2)),
        'mean_percent_error': np.mean(percent_errors),
        'within_uncertainty_fraction': within_uncertainty / n,
        'max_error_h': max(abs_errors),
        'min_error_h': min(abs_errors)
    }


def create_comparison_table(
    events: List[Dict],
    l1_predictions: Dict[str, Dict],
    helios_predictions: Dict[str, Dict]
):
    """
    Create comparison table: L1-only vs HELIOS predictions.
    
    Parameters
    ----------
    events : list of dict
        Event list
    l1_predictions : dict
        L1-only prediction results
    helios_predictions : dict
        HELIOS prediction results
        
    Returns
    -------
    df : pd.DataFrame
        Comparison table
    """
    import pandas as pd
    
    rows = []
    
    for event in events:
        event_id = event['event_id']
        actual_arrival = event.get('actual_arrival_hours')
        
        row = {
            'event_id': event_id,
            'actual_arrival_h': actual_arrival,
        }
        
        # L1 prediction
        if event_id in l1_predictions:
            l1_pred = l1_predictions[event_id]
            row['L1_pred_arrival_h'] = l1_pred.get('pred_arrival_median_h')
            if actual_arrival:
                row['L1_error_h'] = row['L1_pred_arrival_h'] - actual_arrival
        
        # HELIOS prediction
        if event_id in helios_predictions:
            h_pred = helios_predictions[event_id]
            row['HELIOS_pred_arrival_h'] = h_pred.get('pred_arrival_median_h')
            row['HELIOS_uncertainty_h'] = (h_pred.get('pred_arrival_84_h', 0) - 
                                           h_pred.get('pred_arrival_16_h', 0)) / 2
            if actual_arrival:
                row['HELIOS_error_h'] = row['HELIOS_pred_arrival_h'] - actual_arrival
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    metrics: ConfusionMetrics,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    metrics : ConfusionMetrics
        Confusion matrix metrics
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = np.array([[metrics.TN, metrics.FP],
                   [metrics.FN, metrics.TP]])
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    labels = ['No CME', 'CME']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', 
                          fontsize=20, fontweight='bold',
                          color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # Add metrics text
    metrics_text = (f"POD (Sensitivity): {metrics.POD:.2%}\n"
                   f"FAR: {metrics.FAR:.2%}\n"
                   f"Precision: {metrics.precision:.2%}\n"
                   f"F1 Score: {metrics.f1_score:.2f}")
    
    ax.text(1.3, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_comparison(
    ground_truth: List[bool],
    l1_scores: List[float],
    helios_scores: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves comparing L1-only vs HELIOS.
    
    Parameters
    ----------
    ground_truth : list of bool
        True labels
    l1_scores : list of float
        L1 confidence scores
    helios_scores : list of float
        HELIOS confidence scores
    save_path : str, optional
        Path to save figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # L1 ROC
    l1_fpr, l1_tpr, _, l1_auc = compute_roc_curve(ground_truth, l1_scores)
    ax.plot(l1_fpr, l1_tpr, 'b-', linewidth=2.5, 
            label=f'L1-only (AUC = {l1_auc:.3f})')
    
    # HELIOS ROC
    h_fpr, h_tpr, _, h_auc = compute_roc_curve(ground_truth, helios_scores)
    ax.plot(h_fpr, h_tpr, 'r-', linewidth=2.5, 
            label=f'HELIOS (AUC = {h_auc:.3f})')
    
    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Target points
    ax.scatter([0.15], [0.90], s=200, c='green', marker='*', zorder=10,
              label='Target (POD=0.90, FAR=0.15)')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / POD)', fontsize=12)
    ax.set_title('ROC Curve: L1-only vs HELIOS Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_performance_summary(
    l1_metrics: ConfusionMetrics,
    helios_metrics: ConfusionMetrics,
    target_pod: float = 0.90,
    target_far: float = 0.15
) -> None:
    """
    Print formatted performance summary.
    
    Parameters
    ----------
    l1_metrics : ConfusionMetrics
        L1-only detection metrics
    helios_metrics : ConfusionMetrics
        HELIOS multi-view metrics
    target_pod : float
        Target POD
    target_far : float
        Target FAR
    """
    print("\n" + "=" * 60)
    print("DETECTION PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'L1-only':<15} {'HELIOS':<15} {'Target':<15}")
    print("-" * 60)
    
    pod_status_l1 = "✓" if l1_metrics.POD >= target_pod else "✗"
    pod_status_h = "✓" if helios_metrics.POD >= target_pod else "✗"
    print(f"{'POD (Sensitivity)':<20} {l1_metrics.POD:.2%} {pod_status_l1:<5} {helios_metrics.POD:.2%} {pod_status_h:<5} {target_pod:.2%}")
    
    far_status_l1 = "✓" if l1_metrics.FAR <= target_far else "✗"
    far_status_h = "✓" if helios_metrics.FAR <= target_far else "✗"
    print(f"{'FAR':<20} {l1_metrics.FAR:.2%} {far_status_l1:<5} {helios_metrics.FAR:.2%} {far_status_h:<5} ≤{target_far:.2%}")
    
    print(f"{'Precision':<20} {l1_metrics.precision:.2%}       {helios_metrics.precision:.2%}")
    print(f"{'F1 Score':<20} {l1_metrics.f1_score:.3f}        {helios_metrics.f1_score:.3f}")
    print(f"{'Accuracy':<20} {l1_metrics.accuracy:.2%}       {helios_metrics.accuracy:.2%}")
    
    print("\n" + "-" * 60)
    print("IMPROVEMENT (HELIOS vs L1-only):")
    print(f"  POD improvement:      {(helios_metrics.POD - l1_metrics.POD)*100:+.1f} percentage points")
    print(f"  FAR reduction:        {(l1_metrics.FAR - helios_metrics.FAR)*100:+.1f} percentage points")
    print(f"  F1 improvement:       {helios_metrics.f1_score - l1_metrics.f1_score:+.3f}")
    
    if helios_metrics.POD >= target_pod and helios_metrics.FAR <= target_far:
        print("\n★ HELIOS meets both POD and FAR targets!")
    else:
        print("\n⚠ HELIOS does not yet meet all targets. Improvement plan:")
        if helios_metrics.POD < target_pod:
            print(f"   - Increase POD by {(target_pod - helios_metrics.POD)*100:.1f}pp")
        if helios_metrics.FAR > target_far:
            print(f"   - Reduce FAR by {(helios_metrics.FAR - target_far)*100:.1f}pp")
    
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("HELIOS Evaluation Module - Test")
    print("=" * 60)
    
    # Create test data
    ground_truth = [True, True, True, True, True, False, False, False, True, True]
    l1_predictions = [True, True, False, True, False, False, True, False, True, True]
    helios_predictions = [True, True, True, True, True, False, False, False, True, True]
    
    # Compute metrics
    print("\n1. Confusion matrix test:")
    l1_metrics = compute_confusion(ground_truth, l1_predictions)
    helios_metrics = compute_confusion(ground_truth, helios_predictions)
    
    print(f"   L1-only:  TP={l1_metrics.TP}, FN={l1_metrics.FN}, FP={l1_metrics.FP}, TN={l1_metrics.TN}")
    print(f"             POD={l1_metrics.POD:.2%}, FAR={l1_metrics.FAR:.2%}")
    
    print(f"   HELIOS:   TP={helios_metrics.TP}, FN={helios_metrics.FN}, FP={helios_metrics.FP}, TN={helios_metrics.TN}")
    print(f"             POD={helios_metrics.POD:.2%}, FAR={helios_metrics.FAR:.2%}")
    
    # Test ROC
    print("\n2. ROC curve test:")
    l1_scores = [0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.5, 0.1, 0.9, 0.8]
    helios_scores = [0.95, 0.9, 0.85, 0.88, 0.92, 0.15, 0.1, 0.05, 0.93, 0.91]
    
    _, _, _, l1_auc = compute_roc_curve(ground_truth, l1_scores)
    _, _, _, h_auc = compute_roc_curve(ground_truth, helios_scores)
    print(f"   L1 AUC:     {l1_auc:.3f}")
    print(f"   HELIOS AUC: {h_auc:.3f}")
    
    # Print summary
    print_performance_summary(l1_metrics, helios_metrics)
    
    print("\n" + "=" * 60)
    print("Test completed!")
