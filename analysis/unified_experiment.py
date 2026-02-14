#!/usr/bin/env python3
"""
HELIOS Unified Architecture Experiment
========================================

Fair head-to-head comparison of three configurations:

  1. BASELINE:     Simplified features + Uniform Bz + LayerNorm+GELU
  2. PHYSICS+LN:   Physics features + Physics Bz + LayerNorm+GELU
  3. PHYSICS+BN:   Physics features + Physics Bz + BatchNorm+ReLU

All configurations use:
  - Same 12/6/1 train/test/showcase split
  - 50x oversampled training historical + 10,000 synthetic events
  - Adam(lr=0.0005, wd=1e-5), ReduceLROnPlateau(patience=5)
  - Heteroscedastic loss (α=0.7) + CrossEntropy (β=0.3)
  - Class weights [1.0, 1.5, 3.0, 12.0]
  - 80 epochs max, early stopping patience=15
  - StandardScaler normalization (fitted on training data only)

Comparison:
  BASELINE vs PHYSICS+LN  → Does physics-based data/features help?
  PHYSICS+LN vs PHYSICS+BN → Which architecture wins?

Output: Per-event predictions + summary comparison table.
"""

import copy
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# NeuralNetwork_ML imports (physics-based features & data generation)
# ---------------------------------------------------------------------------
from NeuralNetwork_ML.config import FEATURE_BOUNDS, FEATURE_NAMES
from NeuralNetwork_ML.dataset_generator import (
    AU_IN_KM,
    create_historical_event,
    generate_bz_from_physics,
    generate_synthetic_dataset,
)
from NeuralNetwork_ML.features import (
    CMEFeatures,
    create_bastille_day_features,
    extract_features,
)
from NeuralNetwork_ML.severity import bz_to_severity_class


# ============================================================================
# CONSTANTS
# ============================================================================

SEVERITY_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Extreme": 3}
SEVERITY_NAMES = ["Low", "Moderate", "High", "Extreme"]

# Number of random seeds per configuration (increase for robustness)
N_SEEDS = 3
SEEDS = [42, 123, 456]


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ModelBatchNormReLU(nn.Module):
    """Architecture A: BatchNorm + ReLU + Dropout (from NeuralNetwork_ML/model.py)."""

    def __init__(self, input_dim=16, hidden_dims=None, dropout=0.15, n_classes=4):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 64]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)

        self.bz_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),  # mean + log_variance
        )
        self.sev_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        bz_out = self.bz_head(z)
        return bz_out[:, 0], bz_out[:, 1], self.sev_head(z)


class ModelLayerNormGELU(nn.Module):
    """Architecture B: LayerNorm + GELU + Dropout (from run_historical_validation.py)."""

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
            nn.Linear(32, 2),
        )
        self.sev_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        bz_out = self.bz_head(z)
        return bz_out[:, 0], bz_out[:, 1], self.sev_head(z)


# ============================================================================
# NORMALIZER
# ============================================================================

class StandardScaler:
    """Zero mean, unit variance — fitted on training data only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ============================================================================
# HISTORICAL EVENT DATABASE — IDENTICAL to run_historical_validation.py
# ============================================================================

def load_historical_events():
    """
    Load 19 historical CME events with proper train/test split.

    Training: 12 events (diverse across all severity classes)
    Test:     6 events (balanced representation, UNSEEN during training)
    Showcase: Bastille Day 2000 (completely excluded)
    """

    train_events = [
        # EXTREME — 4 training events
        {"name": "Halloween Storm 1 (Oct 28)", "date": "2003-10-28",
         "speed": 2459, "width": 360, "source_lat": 16, "source_lon": -8,
         "Bz_measured": -50, "severity_label": "Extreme"},
        {"name": "January 2005 Storm", "date": "2005-01-15",
         "speed": 2861, "width": 360, "source_lat": 14, "source_lon": -5,
         "Bz_measured": -55, "severity_label": "Extreme"},
        {"name": "July 2012 STEREO", "date": "2012-07-12",
         "speed": 1900, "width": 360, "source_lat": 15, "source_lon": -8,
         "Bz_measured": -55, "severity_label": "Extreme"},
        {"name": "November 2003", "date": "2003-11-20",
         "speed": 1660, "width": 360, "source_lat": 2, "source_lon": 8,
         "Bz_measured": -45, "severity_label": "Extreme"},
        # HIGH — 4 training events
        {"name": "April 2001", "date": "2001-04-02",
         "speed": 1200, "width": 280, "source_lat": 20, "source_lon": -10,
         "Bz_measured": -38, "severity_label": "High"},
        {"name": "December 2006", "date": "2006-12-13",
         "speed": 1774, "width": 360, "source_lat": 6, "source_lon": 38,
         "Bz_measured": -48, "severity_label": "Extreme"},
        {"name": "St. Patricks Day 2015", "date": "2015-03-17",
         "speed": 700, "width": 220, "source_lat": 22, "source_lon": -12,
         "Bz_measured": -22, "severity_label": "High"},
        {"name": "June 2015", "date": "2015-06-22",
         "speed": 750, "width": 230, "source_lat": 13, "source_lon": 20,
         "Bz_measured": -20, "severity_label": "High"},
        # MODERATE — 3 training events
        {"name": "August 2011", "date": "2011-08-05",
         "speed": 650, "width": 200, "source_lat": 21, "source_lon": 5,
         "Bz_measured": -15, "severity_label": "Moderate"},
        {"name": "March 2012", "date": "2012-03-07",
         "speed": 680, "width": 210, "source_lat": 18, "source_lon": -8,
         "Bz_measured": -17, "severity_label": "Moderate"},
        {"name": "April 2023", "date": "2023-04-24",
         "speed": 550, "width": 180, "source_lat": 12, "source_lon": 15,
         "Bz_measured": -14, "severity_label": "Moderate"},
        # LOW-BOUNDARY — 1 training event
        {"name": "February 2022", "date": "2022-02-03",
         "speed": 480, "width": 140, "source_lat": 12, "source_lon": 18,
         "Bz_measured": -12, "severity_label": "Moderate"},
    ]

    test_events = [
        # EXTREME — 2 test events
        {"name": "Halloween Storm 2 (Oct 29)", "date": "2003-10-29",
         "speed": 2029, "width": 360, "source_lat": 15, "source_lon": -2,
         "Bz_measured": -49, "severity_label": "Extreme"},
        {"name": "September 2017", "date": "2017-09-06",
         "speed": 800, "width": 240, "source_lat": 8, "source_lon": 35,
         "Bz_measured": -32, "severity_label": "Extreme"},
        # HIGH — 2 test events
        {"name": "January 2005 (secondary)", "date": "2005-01-17",
         "speed": 1200, "width": 260, "source_lat": 8, "source_lon": -30,
         "Bz_measured": -25, "severity_label": "High"},
        {"name": "November 2001", "date": "2001-11-24",
         "speed": 1100, "width": 250, "source_lat": 18, "source_lon": 12,
         "Bz_measured": -30, "severity_label": "Extreme"},
        # MODERATE / LOW — 2 test events
        {"name": "May 2024", "date": "2024-05-11",
         "speed": 1000, "width": 320, "source_lat": 25, "source_lon": -30,
         "Bz_measured": -50, "severity_label": "Extreme"},
        {"name": "October 2024", "date": "2024-10-10",
         "speed": 900, "width": 280, "source_lat": 15, "source_lon": -25,
         "Bz_measured": -38, "severity_label": "Extreme"},
    ]

    bastille_event = {
        "name": "Bastille Day 2000", "date": "2000-07-14",
        "speed": 1674, "width": 360, "source_lat": 22, "source_lon": 7,
        "Bz_measured": -60, "severity_label": "Extreme",
    }

    return train_events, test_events, bastille_event


# ============================================================================
# FEATURE EXTRACTION — TWO APPROACHES
# ============================================================================

def extract_features_simplified(event):
    """
    Simplified feature extraction (from run_historical_validation.py).
    Deterministic — no randomness. Geometry approximated from source_lon.
    """
    speed = event["speed"]
    width = event["width"]
    lat = event["source_lat"]
    lon = event["source_lon"]

    return np.array([
        speed,
        width,
        lat,
        lon,
        speed / 200.0,                                     # expansion_rate
        -speed / 15.0,                                     # acceleration
        abs(lon),                                           # L1_viewing_angle
        abs(60 - lon),                                     # L4_viewing_angle
        abs(-60 - lon),                                    # L5_viewing_angle
        1.0 if width > 300 else (width / 300.0),           # brightness_asymmetry
        10.0 + (width / 36.0),                             # parallax_L1L4
        10.0 + (width / 36.0),                             # parallax_L1L5
        20.0 + (width / 18.0),                             # parallax_L4L5
        max(0.1, 2.0 - (speed / 1000.0)),                 # detection_time
        min(1.0, width / 200.0),                           # triangulation_quality
        min(1.0, width / 180.0),                           # observation_completeness
    ], dtype=np.float32)


def extract_features_physics(event, seed=42):
    """
    Physics-based feature extraction (from NeuralNetwork_ML package).
    Uses actual HELIOS constellation geometry from helios_code/utils.py.
    Derives expansion_rate, acceleration, etc. from physical relationships.
    """
    event_data = {
        "event_id": event.get("name", "unknown").lower().replace(" ", "_")[:25],
        "date": event["date"],
        "speed": event["speed"],
        "width": event["width"],
        "bz": event["Bz_measured"],
        "source_lat": event["source_lat"],
        "source_lon": event["source_lon"],
        "notes": event.get("reference", ""),
    }
    synth_event = create_historical_event(event_data, seed=seed)
    return synth_event.features.to_array()


# ============================================================================
# BATCH CONVERSIONS
# ============================================================================

def events_to_arrays(events, mode="simplified", base_seed=42):
    """Convert event list to (features, bz, severity) arrays."""
    X, y_bz, y_sev = [], [], []
    for i, e in enumerate(events):
        if mode == "simplified":
            X.append(extract_features_simplified(e))
        else:
            X.append(extract_features_physics(e, seed=base_seed + i))
        y_bz.append(e["Bz_measured"])
        y_sev.append(SEVERITY_MAP[e["severity_label"]])
    return (np.array(X, np.float32),
            np.array(y_bz, np.float32),
            np.array(y_sev, np.int64))


# ============================================================================
# SYNTHETIC DATA GENERATION — TWO APPROACHES
# ============================================================================

def generate_synthetic_simplified(n_samples=10000, seed=42):
    """
    Baseline: uniform Bz per severity class.
    Reproduces run_historical_validation.py generate_synthetic_cme_dataset.
    """
    rng = np.random.RandomState(seed)

    n_low = int(n_samples * 0.20)
    n_mod = int(n_samples * 0.20)
    n_high = int(n_samples * 0.25)
    n_ext = n_samples - n_low - n_mod - n_high

    configs = [
        (0, n_low,  (250, 550),  (50, 160),  (-10, -1)),
        (1, n_mod,  (400, 750),  (100, 220), (-20, -8)),
        (2, n_high, (600, 1300), (180, 320), (-38, -16)),
        (3, n_ext,  (700, 3000), (200, 360), (-80, -28)),
    ]

    X_all, y_bz_all, y_sev_all = [], [], []
    for sev, n, (s_lo, s_hi), (w_lo, w_hi), (b_lo, b_hi) in configs:
        for _ in range(n):
            speed = rng.uniform(s_lo, s_hi)
            width = rng.uniform(w_lo, w_hi)
            event = {"speed": speed, "width": width,
                     "source_lat": rng.uniform(-40, 40),
                     "source_lon": rng.uniform(-60, 60)}
            X_all.append(extract_features_simplified(event))
            y_bz_all.append(rng.uniform(b_lo, b_hi))
            y_sev_all.append(sev)

    perm = rng.permutation(len(X_all))
    return (np.array(X_all, np.float32)[perm],
            np.array(y_bz_all, np.float32)[perm],
            np.array(y_sev_all, np.int64)[perm])


def generate_synthetic_physics(n_events=10000, seed=42):
    """
    Physics-based: Bz from calibrated physics model with stratified sampling.
    Uses NeuralNetwork_ML.dataset_generator.generate_synthetic_dataset.
    Removes Bastille Day augmentations for fairness (showcase is excluded).
    """
    features, bz, severity, ids = generate_synthetic_dataset(n_events, seed)
    # Remove Bastille augmentations — showcase must be completely excluded
    mask = np.array([not eid.startswith("bastille_aug") for eid in ids])
    return features[mask], bz[mask], severity[mask]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(feature_mode, train_events, test_events, bastille_event,
                 oversample_factor=50, n_synthetic=10000, data_seed=42):
    """
    Prepare combined training data and test data for one feature mode.

    Parameters
    ----------
    feature_mode : str
        'simplified' or 'physics'
    oversample_factor : int
        How many augmented copies of each historical training event.
    n_synthetic : int
        Number of synthetic events.
    data_seed : int
        Random seed for data generation.

    Returns
    -------
    dict with X_train, y_bz_train, y_sev_train, X_test, y_bz_test,
    y_sev_test, X_bastille, y_bz_bastille, n_synthetic_actual.
    """
    # --- Extract features from historical events ---
    X_train_hist, y_bz_train, y_sev_train = events_to_arrays(
        train_events, mode=feature_mode, base_seed=data_seed
    )
    X_test, y_bz_test, y_sev_test = events_to_arrays(
        test_events, mode=feature_mode, base_seed=data_seed + 1000
    )

    if feature_mode == "simplified":
        X_bastille = extract_features_simplified(bastille_event)
    else:
        X_bastille = extract_features_physics(bastille_event, seed=data_seed + 2000)

    # --- Generate synthetic data ---
    if feature_mode == "simplified":
        X_synth, y_bz_synth, y_sev_synth = generate_synthetic_simplified(n_synthetic, data_seed)
    else:
        X_synth, y_bz_synth, y_sev_synth = generate_synthetic_physics(n_synthetic, data_seed)

    # --- Oversample historical training events with noise augmentation ---
    aug_rng = np.random.RandomState(99)
    X_aug_list, y_bz_aug_list, y_sev_aug_list = [], [], []
    for i in range(len(X_train_hist)):
        for _ in range(oversample_factor):
            noise_feat = 1.0 + aug_rng.normal(0, 0.05, X_train_hist[i].shape)
            X_aug_list.append(X_train_hist[i] * noise_feat)
            y_bz_aug_list.append(y_bz_train[i] + aug_rng.normal(0, 1.5))
            y_sev_aug_list.append(y_sev_train[i])
    X_aug = np.array(X_aug_list, np.float32)
    y_bz_aug = np.array(y_bz_aug_list, np.float32)
    y_sev_aug = np.array(y_sev_aug_list, np.int64)

    # --- Combine synthetic + augmented historical ---
    X_combined = np.vstack([X_synth, X_aug])
    y_bz_combined = np.hstack([y_bz_synth, y_bz_aug])
    y_sev_combined = np.hstack([y_sev_synth, y_sev_aug])

    # Shuffle
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_bz_combined = y_bz_combined[perm]
    y_sev_combined = y_sev_combined[perm]

    return {
        "X_train": X_combined,
        "y_bz_train": y_bz_combined,
        "y_sev_train": y_sev_combined,
        "X_test": X_test,
        "y_bz_test": y_bz_test,
        "y_sev_test": y_sev_test,
        "X_bastille": X_bastille,
        "y_bz_bastille": bastille_event["Bz_measured"],
        "n_synthetic": len(X_synth),
        "n_augmented": len(X_aug),
    }


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def heteroscedastic_loss(bz_mean, bz_logvar, bz_true):
    """Heteroscedastic regression loss — learns per-prediction uncertainty."""
    precision = torch.exp(-bz_logvar)
    return (0.5 * precision * (bz_true - bz_mean) ** 2 + 0.5 * bz_logvar).mean()


# ============================================================================
# TRAINING + EVALUATION (single run)
# ============================================================================

def train_and_evaluate(config_name, model_class, model_kwargs,
                       data, device, train_seed=42, verbose=False):
    """
    Train one model and evaluate on test set + Bastille.

    Returns dict of metrics + per-event predictions.
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    X_train_raw = data["X_train"]
    y_bz_train = data["y_bz_train"]
    y_sev_train = data["y_sev_train"]
    X_test_raw = data["X_test"]
    y_bz_test = data["y_bz_test"]
    y_sev_test = data["y_sev_test"]
    X_bastille_raw = data["X_bastille"]
    y_bz_bastille = data["y_bz_bastille"]

    # --- Normalize ---
    normalizer = StandardScaler()
    X_train_norm = normalizer.fit_transform(X_train_raw.copy())
    X_test_norm = normalizer.transform(X_test_raw.copy())
    X_bastille_norm = normalizer.transform(X_bastille_raw.copy().reshape(1, -1))

    # --- 80/20 train/val split ---
    n = len(X_train_norm)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    X_t = torch.FloatTensor(X_train_norm[train_idx]).to(device)
    y_bz_t = torch.FloatTensor(y_bz_train[train_idx]).to(device)
    y_sev_t = torch.LongTensor(y_sev_train[train_idx]).to(device)

    X_v = torch.FloatTensor(X_train_norm[val_idx]).to(device)
    y_bz_v = torch.FloatTensor(y_bz_train[val_idx]).to(device)
    y_sev_v = torch.LongTensor(y_sev_train[val_idx]).to(device)

    train_loader = DataLoader(
        TensorDataset(X_t, y_bz_t, y_sev_t),
        batch_size=64, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_v, y_bz_v, y_sev_v),
        batch_size=64, shuffle=False, num_workers=0,
    )

    # --- Model ---
    model = model_class(**model_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False,
    )

    class_weights = torch.tensor([1.0, 1.5, 3.0, 12.0], device=device)
    criterion_sev = nn.CrossEntropyLoss(weight=class_weights)
    alpha, beta = 0.7, 0.3

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    final_epoch = 0

    t0 = time.time()
    for epoch in range(80):
        # Train
        model.train()
        for batch_X, batch_bz, batch_sev in train_loader:
            optimizer.zero_grad()
            bz_mean, bz_logvar, sev_logits = model(batch_X)
            loss = (alpha * heteroscedastic_loss(bz_mean, bz_logvar, batch_bz)
                    + beta * criterion_sev(sev_logits, batch_sev))
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss_sum, val_correct = 0.0, 0
        with torch.no_grad():
            for batch_X, batch_bz, batch_sev in val_loader:
                bz_mean, bz_logvar, sev_logits = model(batch_X)
                loss = (alpha * heteroscedastic_loss(bz_mean, bz_logvar, batch_bz)
                        + beta * criterion_sev(sev_logits, batch_sev))
                val_loss_sum += loss.item()
                val_correct += (torch.argmax(sev_logits, 1) == batch_sev).sum().item()

        avg_val = val_loss_sum / len(val_loader)
        val_acc = val_correct / len(y_sev_v) * 100
        scheduler.step(avg_val)

        if verbose and (epoch % 10 == 0 or epoch == 79):
            print(f"    [{config_name}] Epoch {epoch:3d} | val_loss={avg_val:.4f} | "
                  f"val_acc={val_acc:.1f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break
        final_epoch = epoch

    elapsed = time.time() - t0

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # --- Evaluate test set ---
    X_test_gpu = torch.FloatTensor(X_test_norm).to(device)
    with torch.no_grad():
        bz_pred, bz_logvar, sev_logits = model(X_test_gpu)
        bz_pred_np = bz_pred.cpu().numpy()
        bz_unc_np = np.exp(bz_logvar.cpu().numpy() / 2)  # std dev
        sev_pred_np = torch.argmax(sev_logits, 1).cpu().numpy()
        sev_probs_np = torch.softmax(sev_logits, 1).cpu().numpy()

    bz_errors = np.abs(bz_pred_np - y_bz_test)
    bz_mae = float(bz_errors.mean())
    bz_std = float(bz_errors.std())
    sev_correct = int((sev_pred_np == y_sev_test).sum())
    sev_accuracy = sev_correct / len(y_sev_test) * 100

    # Adjacent error rate
    adjacent_or_correct = float(
        (np.abs(sev_pred_np.astype(int) - y_sev_test.astype(int)) <= 1).mean() * 100
    )

    # Baseline comparison
    baseline_mae = 12.5  # nT — persistence model baseline
    improvement = ((baseline_mae - bz_mae) / baseline_mae) * 100

    # --- Evaluate Bastille Day ---
    X_bast_gpu = torch.FloatTensor(X_bastille_norm).to(device)
    with torch.no_grad():
        bz_p, bz_lv, sev_l = model(X_bast_gpu)
        bastille_pred = bz_p.item()
        bastille_sev = torch.argmax(sev_l).item()
        bastille_conf = torch.softmax(sev_l, 1).cpu().numpy()[0]
    bastille_error = abs(bastille_pred - y_bz_bastille)

    return {
        "config": config_name,
        "seed": train_seed,
        "bz_mae": round(bz_mae, 2),
        "bz_std": round(bz_std, 2),
        "improvement_pct": round(improvement, 1),
        "sev_accuracy": round(sev_accuracy, 1),
        "adjacent_or_correct": round(adjacent_or_correct, 1),
        "bastille_pred": round(bastille_pred, 1),
        "bastille_error": round(bastille_error, 1),
        "bastille_sev": SEVERITY_NAMES[bastille_sev],
        "bastille_conf": round(float(bastille_conf[bastille_sev]) * 100, 1),
        "best_val_loss": round(best_val_loss, 4),
        "epochs": final_epoch + 1,
        "train_time_s": round(elapsed, 1),
        # Per-event details
        "per_event": {
            "bz_pred": bz_pred_np.tolist(),
            "bz_unc": bz_unc_np.tolist(),
            "bz_error": bz_errors.tolist(),
            "sev_pred": sev_pred_np.tolist(),
            "sev_probs": sev_probs_np.tolist(),
        },
    }


# ============================================================================
# MULTI-SEED RUNNER
# ============================================================================

def run_config_multi_seed(config_name, model_class, model_kwargs, data,
                          device, seeds, verbose=False):
    """
    Run one configuration across multiple seeds. Returns aggregated results.
    """
    all_runs = []
    for s in seeds:
        result = train_and_evaluate(
            config_name, model_class, model_kwargs,
            data, device, train_seed=s, verbose=verbose,
        )
        all_runs.append(result)

    # Aggregate
    maes = [r["bz_mae"] for r in all_runs]
    accs = [r["sev_accuracy"] for r in all_runs]
    bast_errs = [r["bastille_error"] for r in all_runs]
    imps = [r["improvement_pct"] for r in all_runs]

    return {
        "config": config_name,
        "n_seeds": len(seeds),
        "bz_mae_mean": round(float(np.mean(maes)), 2),
        "bz_mae_std": round(float(np.std(maes)), 2),
        "sev_accuracy_mean": round(float(np.mean(accs)), 1),
        "sev_accuracy_std": round(float(np.std(accs)), 1),
        "bastille_error_mean": round(float(np.mean(bast_errs)), 1),
        "bastille_error_std": round(float(np.std(bast_errs)), 1),
        "improvement_mean": round(float(np.mean(imps)), 1),
        "runs": all_runs,
        "best_run": all_runs[int(np.argmin(maes))],
    }


# ============================================================================
# GPU SETUP
# ============================================================================

def setup_device():
    """Detect and configure GPU/CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name} ({vram:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("  CPU mode (no GPU detected)")
    return device


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_per_event_table(test_events, result):
    """Print per-event prediction breakdown for a single run."""
    pe = result["per_event"]
    y_bz_test = [e["Bz_measured"] for e in test_events]
    y_sev_test = [SEVERITY_MAP[e["severity_label"]] for e in test_events]

    print(f"\n     {'Event':<28} {'True':>6} {'Pred':>6} {'Err':>5} "
          f"{'Unc':>5} {'TrueSev':>8} {'PredSev':>8} {'Conf':>5}")
    print("     " + "-" * 78)

    for i, ev in enumerate(test_events):
        sev_true = SEVERITY_NAMES[y_sev_test[i]]
        sev_pred = SEVERITY_NAMES[pe["sev_pred"][i]]
        conf = pe["sev_probs"][i][pe["sev_pred"][i]] * 100
        ok = "Y" if pe["sev_pred"][i] == y_sev_test[i] else " "
        print(f"  {ok}  {ev['name']:<28} {y_bz_test[i]:>5.0f} "
              f"{pe['bz_pred'][i]:>6.1f} {pe['bz_error'][i]:>5.1f} "
              f"{pe['bz_unc'][i]:>5.1f} {sev_true:>8} {sev_pred:>8} "
              f"{conf:>5.1f}%")


def print_comparison_table(summaries):
    """Print side-by-side comparison of all configurations."""
    print("\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)

    header = (f"  {'Config':<22} {'Bz MAE (nT)':>14} {'Sev Acc (%)':>14} "
              f"{'Bastille Err':>14} {'Improvement':>12}")
    print(header)
    print("  " + "-" * 78)

    for s in summaries:
        if s["n_seeds"] > 1:
            mae_str = f"{s['bz_mae_mean']:.1f} ± {s['bz_mae_std']:.1f}"
            acc_str = f"{s['sev_accuracy_mean']:.0f} ± {s['sev_accuracy_std']:.0f}"
            bast_str = f"{s['bastille_error_mean']:.1f} ± {s['bastille_error_std']:.1f}"
            imp_str = f"{s['improvement_mean']:.0f}%"
        else:
            r = s["best_run"]
            mae_str = f"{r['bz_mae']:.1f}"
            acc_str = f"{r['sev_accuracy']:.0f}"
            bast_str = f"{r['bastille_error']:.1f}"
            imp_str = f"{r['improvement_pct']:.0f}%"
        print(f"  {s['config']:<22} {mae_str:>14} {acc_str:>14} "
              f"{bast_str:>14} {imp_str:>12}")

    # Identify winner
    best_idx = int(np.argmin([s["bz_mae_mean"] for s in summaries]))
    print(f"\n  >>> WINNER (lowest MAE): {summaries[best_idx]['config']}")

    print("=" * 90)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("  HELIOS UNIFIED ARCHITECTURE EXPERIMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    device = setup_device()

    # ── Load event split ──────────────────────────────────────────────────
    train_events, test_events, bastille_event = load_historical_events()
    print(f"\n  Events: {len(train_events)} train / {len(test_events)} test / 1 showcase")

    # ── Prepare data for each feature mode ────────────────────────────────
    print("\n  Preparing SIMPLIFIED data (baseline)...")
    data_simple = prepare_data("simplified", train_events, test_events, bastille_event)
    print(f"    Synthetic: {data_simple['n_synthetic']}  |  "
          f"Augmented historical: {data_simple['n_augmented']}  |  "
          f"Total training: {len(data_simple['X_train'])}")

    print("\n  Preparing PHYSICS data...")
    data_physics = prepare_data("physics", train_events, test_events, bastille_event)
    print(f"    Synthetic: {data_physics['n_synthetic']}  |  "
          f"Augmented historical: {data_physics['n_augmented']}  |  "
          f"Total training: {len(data_physics['X_train'])}")

    # ── Define configurations ─────────────────────────────────────────────
    model_kwargs = {"input_dim": 16, "hidden_dims": [128, 256, 128, 64], "n_classes": 4}

    configs = [
        {
            "name": "BASELINE (LN+GELU)",
            "model_class": ModelLayerNormGELU,
            "model_kwargs": {**model_kwargs, "dropout": 0.2},
            "data": data_simple,
        },
        {
            "name": "PHYSICS + LN+GELU",
            "model_class": ModelLayerNormGELU,
            "model_kwargs": {**model_kwargs, "dropout": 0.2},
            "data": data_physics,
        },
        {
            "name": "PHYSICS + BN+ReLU",
            "model_class": ModelBatchNormReLU,
            "model_kwargs": {**model_kwargs, "dropout": 0.15},
            "data": data_physics,
        },
    ]

    # ── Run all configurations ────────────────────────────────────────────
    seeds = SEEDS[:N_SEEDS]
    all_summaries = []

    for cfg in configs:
        print(f"\n{'─' * 90}")
        print(f"  RUNNING: {cfg['name']}  ({len(seeds)} seed(s))")
        print(f"{'─' * 90}")

        summary = run_config_multi_seed(
            cfg["name"], cfg["model_class"], cfg["model_kwargs"],
            cfg["data"], device, seeds, verbose=True,
        )
        all_summaries.append(summary)

        # Print best run details
        best = summary["best_run"]
        print(f"\n  Best run (seed={best['seed']}): "
              f"MAE={best['bz_mae']:.1f} nT | Acc={best['sev_accuracy']:.0f}% | "
              f"Bastille={best['bastille_error']:.1f} nT "
              f"({best['bastille_sev']} {best['bastille_conf']:.0f}%)")
        print_per_event_table(test_events, best)

    # ── Final comparison ──────────────────────────────────────────────────
    print_comparison_table(all_summaries)

    # ── Identify and save the winner ──────────────────────────────────────
    best_cfg_idx = int(np.argmin([s["bz_mae_mean"] for s in all_summaries]))
    winner = all_summaries[best_cfg_idx]
    winner_run = winner["best_run"]

    print(f"\n  Saving winner results...")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save comparison report
    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "unified_architecture_comparison",
        "seeds_per_config": len(seeds),
        "device": str(device),
        "winner": winner["config"],
        "summaries": [
            {
                "config": s["config"],
                "bz_mae_mean": s["bz_mae_mean"],
                "bz_mae_std": s["bz_mae_std"],
                "sev_accuracy_mean": s["sev_accuracy_mean"],
                "bastille_error_mean": s["bastille_error_mean"],
                "improvement_mean": s["improvement_mean"],
            }
            for s in all_summaries
        ],
        "winner_best_run": {
            "bz_mae": winner_run["bz_mae"],
            "bz_std": winner_run["bz_std"],
            "sev_accuracy": winner_run["sev_accuracy"],
            "bastille_pred": winner_run["bastille_pred"],
            "bastille_error": winner_run["bastille_error"],
            "bastille_sev": winner_run["bastille_sev"],
            "improvement_pct": winner_run["improvement_pct"],
            "epochs": winner_run["epochs"],
        },
    }

    # Convert numpy types for JSON
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        return obj

    report_path = output_dir / "experiment_comparison.json"
    with open(report_path, "w") as f:
        json.dump(to_native(report), f, indent=2)
    print(f"  Report saved: {report_path}")

    print(f"\n{'=' * 90}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Winner: {winner['config']} — MAE={winner['bz_mae_mean']:.1f} nT")
    print(f"{'=' * 90}")

    return all_summaries


if __name__ == "__main__":
    main()
