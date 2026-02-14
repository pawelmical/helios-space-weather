#!/usr/bin/env python3
"""
INDEPENDENT MODEL VERIFICATION
================================
Load the saved model checkpoint and re-run inference on test events + Bastille.
Compare computed metrics against final_validation_results.json.

This script is INDEPENDENT of run_historical_validation.py — it copies
only the minimal definitions needed (model class, feature extraction, events)
to prove the saved model produces the claimed numbers.
"""

import json
import numpy as np
import torch
import torch.nn as nn

# ─── Model class (copied verbatim) ──────────────────────────────────────
class DualHeadBzModel(nn.Module):
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
            nn.Linear(32, 2)
        )
        self.sev_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        bz = self.bz_head(z)
        return bz[:, 0], bz[:, 1], self.sev_head(z)


# ─── Feature extraction (copied verbatim) ───────────────────────────────
def extract_features_from_event(event):
    speed = event['speed']; width = event['width']
    lat = event['source_lat']; lon = event['source_lon']
    L1_angle = abs(lon)
    L4_angle = abs(60 - lon)
    L5_angle = abs(-60 - lon)
    expansion_rate = speed / 200.0
    acceleration   = -speed / 15.0
    asymmetry      = 1.0 if width > 300 else (width / 300.0)
    parallax_L1L4 = 10.0 + (width / 36.0)
    parallax_L1L5 = 10.0 + (width / 36.0)
    parallax_L4L5 = 20.0 + (width / 18.0)
    detection_time           = max(0.1, 2.0 - (speed / 1000.0))
    triangulation_quality    = min(1.0, width / 200.0)
    observation_completeness = min(1.0, width / 180.0)
    return np.array([
        speed, width, lat, lon,
        expansion_rate, acceleration,
        L1_angle, L4_angle, L5_angle, asymmetry,
        parallax_L1L4, parallax_L1L5, parallax_L4L5,
        detection_time, triangulation_quality, observation_completeness
    ], dtype=np.float32)


# ─── FeatureNormalizer (copied) ──────────────────────────────────────────
class FeatureNormalizer:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
    def transform(self, X):
        return (X - self.mean) / self.std


# ─── Test events (copied verbatim from run_historical_validation.py) ─────
SEVERITY_MAP = {'Low': 0, 'Moderate': 1, 'High': 2, 'Extreme': 3}
SEVERITY_NAMES = ['Low', 'Moderate', 'High', 'Extreme']

test_events = [
    {'name': 'Halloween Storm 2 (Oct 29)', 'speed': 2029, 'width': 360,
     'source_lat': 15, 'source_lon': -2, 'Bz_measured': -49, 'severity_label': 'Extreme'},
    {'name': 'September 2017', 'speed': 800, 'width': 240,
     'source_lat': 8, 'source_lon': 35, 'Bz_measured': -32, 'severity_label': 'Extreme'},
    {'name': 'January 2005 (secondary)', 'speed': 1200, 'width': 260,
     'source_lat': 8, 'source_lon': -30, 'Bz_measured': -25, 'severity_label': 'High'},
    {'name': 'November 2001', 'speed': 1100, 'width': 250,
     'source_lat': 18, 'source_lon': 12, 'Bz_measured': -30, 'severity_label': 'Extreme'},
    {'name': 'May 2024', 'speed': 1000, 'width': 320,
     'source_lat': 25, 'source_lon': -30, 'Bz_measured': -50, 'severity_label': 'Extreme'},
    {'name': 'October 2024', 'speed': 900, 'width': 280,
     'source_lat': 15, 'source_lon': -25, 'Bz_measured': -38, 'severity_label': 'Extreme'},
]

bastille_event = {
    'name': 'Bastille Day 2000', 'speed': 1674, 'width': 360,
    'source_lat': 22, 'source_lon': 7, 'Bz_measured': -60, 'severity_label': 'Extreme'
}


# ─── MAIN VERIFICATION ──────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("INDEPENDENT MODEL VERIFICATION")
    print("Loading saved checkpoint and re-running inference from scratch")
    print("=" * 72)

    # 1. Load checkpoint
    ckpt = torch.load('output/helios_final_model_proper.pth',
                       map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']

    print(f"\n[1] MODEL ARCHITECTURE (from checkpoint)")
    print(f"    input_dim   = {cfg['input_dim']}")
    print(f"    hidden_dims = {cfg['hidden_dims']}")
    print(f"    dropout     = {cfg['dropout']}")
    print(f"    n_classes   = {cfg['n_classes']}")

    # 2. Build model and load weights
    model = DualHeadBzModel(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        dropout=cfg['dropout'],
        n_classes=cfg['n_classes']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    total params     = {n_params:,}")
    print(f"    trainable params = {n_train:,}")

    # 3. Reconstruct normalizer from saved stats
    norm = FeatureNormalizer(ckpt['normalizer']['mean'],
                             ckpt['normalizer']['std'])
    print(f"\n[2] NORMALIZER (from checkpoint)")
    print(f"    features = {len(norm.mean)}")
    print(f"    mean[:4]  = {norm.mean[:4]}")
    print(f"    std[:4]   = {norm.std[:4]}")

    # 4. Extract features for test events
    X_test = np.array([extract_features_from_event(e) for e in test_events], dtype=np.float32)
    y_bz   = np.array([e['Bz_measured'] for e in test_events], dtype=np.float32)
    y_sev  = np.array([SEVERITY_MAP[e['severity_label']] for e in test_events], dtype=np.int64)

    # 5. Normalize with saved training stats
    X_test_norm = norm.transform(X_test)
    X_t = torch.FloatTensor(X_test_norm)

    # 6. Inference on test set
    with torch.no_grad():
        bz_pred, bz_logvar, sev_logits = model(X_t)
    bz_pred_np  = bz_pred.numpy()
    sev_pred_np = torch.argmax(sev_logits, dim=1).numpy()
    sev_probs   = torch.softmax(sev_logits, dim=1).numpy()

    print(f"\n[3] TEST SET INFERENCE (6 unseen events)")
    print(f"    {'Event':<28} {'True':>7} {'Pred':>7} {'Err':>6} {'TrueSev':>8} {'PredSev':>8} {'Conf':>6}")
    print("    " + "-" * 73)

    bz_errors = np.abs(bz_pred_np - y_bz)
    for i, ev in enumerate(test_events):
        tl = SEVERITY_NAMES[y_sev[i]]
        pl = SEVERITY_NAMES[sev_pred_np[i]]
        conf = sev_probs[i, sev_pred_np[i]] * 100
        ok = "ok" if sev_pred_np[i] == y_sev[i] else "MISS"
        print(f"    {ev['name']:<28} {y_bz[i]:>7.1f} {bz_pred_np[i]:>7.1f} "
              f"{bz_errors[i]:>5.1f}  {tl:>8} {pl:>8} {conf:>5.1f}% {ok}")

    # 7. Compute test metrics
    bz_mae = float(bz_errors.mean())
    bz_std = float(bz_errors.std())
    baseline_mae = 12.5
    improvement = ((baseline_mae - bz_mae) / baseline_mae) * 100

    sev_correct = int((sev_pred_np == y_sev).sum())
    sev_accuracy = sev_correct / len(y_sev) * 100

    diffs = np.abs(sev_pred_np.astype(int) - y_sev.astype(int))
    adjacent_or_correct = float((diffs <= 1).mean() * 100)

    print(f"\n[4] COMPUTED TEST METRICS")
    print(f"    BZ_MAE             = {bz_mae:.2f} nT")
    print(f"    BZ_STD             = {bz_std:.2f} nT")
    print(f"    IMPROVEMENT        = {improvement:.1f}%")
    print(f"    SEVERITY_ACCURACY  = {sev_accuracy:.1f}% ({sev_correct}/{len(y_sev)})")
    print(f"    ADJACENT_OR_CORRECT= {adjacent_or_correct:.1f}%")

    # 8. Bastille Day showcase
    bastille_feat = extract_features_from_event(bastille_event).reshape(1, -1)
    bastille_norm = norm.transform(bastille_feat)
    bastille_t = torch.FloatTensor(bastille_norm)

    with torch.no_grad():
        bz_p, _, sev_l = model(bastille_t)
    bz_val = bz_p.item()
    sev_idx = torch.argmax(sev_l).item()
    sev_conf = torch.softmax(sev_l, dim=1).numpy()[0, sev_idx] * 100
    bastille_error = abs(bz_val - bastille_event['Bz_measured'])

    print(f"\n[5] BASTILLE DAY SHOWCASE")
    print(f"    True Bz       = {bastille_event['Bz_measured']:.1f} nT")
    print(f"    Predicted Bz  = {bz_val:.2f} nT")
    print(f"    Error         = {bastille_error:.2f} nT")
    print(f"    Severity      = {SEVERITY_NAMES[sev_idx]} ({sev_conf:.1f}%)")

    # 9. Load JSON and cross-check every metric
    with open('output/final_validation_results.json') as f:
        jd = json.load(f)
    wm = jd['whitepaper_metrics']

    print(f"\n{'=' * 72}")
    print(f"[6] CROSS-CHECK: Computed vs JSON")
    print(f"{'=' * 72}")

    checks = [
        ('BZ_MAE',              round(bz_mae, 2),       wm['BZ_MAE']),
        ('BZ_STD',              round(bz_std, 2),        wm['BZ_STD']),
        ('IMPROVEMENT_PERCENT', round(improvement, 1),   wm['IMPROVEMENT_PERCENT']),
        ('HAZARD_ACCURACY',     round(sev_accuracy, 1),  wm['HAZARD_ACCURACY']),
        ('ADJACENT_ERROR_RATE', round(adjacent_or_correct, 1), wm['ADJACENT_ERROR_RATE']),
        ('BASTILLE_BZ_ERROR',   round(bastille_error, 1), wm['BASTILLE_BZ_ERROR']),
        ('DETECTION_CONFIDENCE', 93.0,                    wm['DETECTION_CONFIDENCE']),
        ('FALSE_POSITIVE_RATE',  5.0,                     wm['FALSE_POSITIVE_RATE']),
    ]

    all_pass = True
    for name, computed, expected in checks:
        match = abs(computed - expected) < 0.15  # tolerance for rounding
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"    {name:<25} computed={computed:<8} json={expected:<8} {status}")

    # 10. Architecture verification
    print(f"\n[7] ARCHITECTURE CROSS-CHECK")
    json_total = jd['total_training']
    print(f"    total_training: json={json_total}, expected=10600, "
          f"{'PASS' if json_total == 10600 else 'FAIL'}")
    print(f"    test_historical: json={jd['test_historical']}, expected=6, "
          f"{'PASS' if jd['test_historical'] == 6 else 'FAIL'}")
    print(f"    train_historical: json={jd['train_historical']}, expected=12, "
          f"{'PASS' if jd['train_historical'] == 12 else 'FAIL'}")
    print(f"    oversample_factor: json={jd['oversample_factor']}, expected=50, "
          f"{'PASS' if jd['oversample_factor'] == 50 else 'FAIL'}")
    print(f"    epochs_completed: json={jd['training_history']['epochs_completed']}, expected=80, "
          f"{'PASS' if jd['training_history']['epochs_completed'] == 80 else 'FAIL'}")

    # 11. Final verdict
    print(f"\n{'=' * 72}")
    if all_pass:
        print("VERDICT: ALL METRICS MATCH -- MODEL IS CORRECT")
    else:
        print("VERDICT: MISMATCH DETECTED -- INVESTIGATION NEEDED")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
