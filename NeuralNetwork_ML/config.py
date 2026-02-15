"""
HELIOS Neural Network Configuration
====================================
Hyperparameters, constants, and configuration for Bz prediction model.

Author: HELIOS Team
Date: February 2026
"""

import sys
import os
import importlib.util

# ============================================================================
# PHYSICAL CONSTANTS (imported from existing utils for consistency)
# ============================================================================
# Use importlib to avoid conflict with Python's built-in 'code' module
def _import_utils_constants():
    """Import constants from helios_code/utils.py using importlib."""
    try:
        utils_path = os.path.join(os.path.dirname(__file__), '..', 'helios_code', 'utils.py')
        if os.path.exists(utils_path):
            spec = importlib.util.spec_from_file_location("helios_utils", utils_path)
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            return utils_module.AU_IN_KM, utils_module.SOLAR_RADIUS_KM
    except Exception:
        pass
    # Fallback values
    return 1.496e8, 6.96e5  # km

AU_IN_KM, SOLAR_RADIUS_KM = _import_utils_constants()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
MODEL_CONFIG = {
    # Input
    'input_dim': 16,

    # Shared Encoder - INCREASED CAPACITY for better Bz learning
    'encoder_layers': [16, 128, 256, 128, 64],
    'encoder_activation': 'relu',
    'encoder_dropout': 0.15,  # Reduced dropout for better learning

    # Bz Regression Head
    'bz_head_layers': [64, 32, 2],  # Output: [mean, log_variance]
    'bz_activation': 'relu',

    # Severity Classification Head
    'severity_head_layers': [64, 32, 4],  # 4 classes
    'severity_activation': 'relu',

    # Loss weights (multi-task learning) - EMPHASIZE Bz MORE
    'alpha_bz': 0.8,      # Weight for Bz regression loss (increased)
    'beta_severity': 0.2,  # Weight for severity classification loss
}

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 15,  # More patience for imbalanced data
    'validation_split': 0.2,
    'random_seed': 42,

    # Class weighting for imbalanced severity classification
    'use_class_weights': True,
    'severity_class_weights': [1.0, 1.5, 3.0, 10.0],  # Low, Moderate, High, Extreme
}

# ============================================================================
# DATASET PARAMETERS
# ============================================================================
DATASET_CONFIG = {
    'n_synthetic_events': 10000,
    'n_historical_events': 20,

    # Stratified sampling - ensure sufficient extreme events
    'stratified_sampling': True,
    'target_class_distribution': [0.35, 0.30, 0.20, 0.15],  # Low, Moderate, High, Extreme
    'min_extreme_events': 1000,  # Ensure at least this many Extreme class

    # Extreme event augmentation
    'augment_extreme_events': True,
    'extreme_augmentation_factor': 10,  # Replicate extreme events

    # Speed distribution (log-normal)
    'speed_mu': 6.5,       # log(km/s), ~665 km/s median
    'speed_sigma': 0.5,    # log-normal std
    'speed_min': 300,      # km/s
    'speed_max': 3500,     # km/s

    # Angular width distribution
    'width_mean': 60.0,    # degrees
    'width_std': 25.0,
    'width_min': 15.0,
    'width_max': 360.0,    # Full halo
}

# ============================================================================
# FEATURE BOUNDS (for normalization)
# ============================================================================
FEATURE_BOUNDS = {
    'cme_speed': (300, 3500),           # km/s
    'angular_width': (15, 360),          # degrees
    'source_latitude': (-90, 90),        # degrees
    'source_longitude': (-180, 180),     # degrees
    'expansion_rate': (0.1, 5.0),        # Rs/hour
    'acceleration': (-500, 500),         # m/s^2
    'L1_viewing_angle': (0, 180),        # degrees
    'L4_viewing_angle': (0, 180),        # degrees
    'L5_viewing_angle': (0, 180),        # degrees
    'brightness_asymmetry': (0.1, 10.0), # ratio
    'parallax_L1L4': (0, 50),            # solar radii
    'parallax_L1L5': (0, 50),            # solar radii
    'parallax_L4L5': (0, 50),            # solar radii
    'detection_time': (0.1, 24),         # hours post-eruption
    'triangulation_quality': (0, 1),     # score
    'observation_completeness': (0, 1),  # score
}

# Feature names in order
FEATURE_NAMES = [
    'cme_speed', 'angular_width', 'source_latitude', 'source_longitude',
    'expansion_rate', 'acceleration', 'L1_viewing_angle', 'L4_viewing_angle',
    'L5_viewing_angle', 'brightness_asymmetry', 'parallax_L1L4', 'parallax_L1L5',
    'parallax_L4L5', 'detection_time', 'triangulation_quality', 'observation_completeness'
]

# ============================================================================
# BZ PHYSICS PARAMETERS
# ============================================================================
BZ_CONFIG = {
    'bz_min': -80.0,      # nT (extreme southward)
    'bz_max': 0.0,        # nT (we only model southward Bz)
    'noise_sigma': 5.0,   # nT
}

# ============================================================================
# SEVERITY THRESHOLDS (Dosimetry-based)
# ============================================================================
# ── DEEP-SPACE DOSIMETRY (no magnetosphere) ──
# Formula:  D_deepspace (mSv) = K × |Bz|^α × √v_CME × t_exposure
#
# K  = 0.0132  (median of 8 historical SPE reconstructions)
# α  = 1.3     (empirical Bz-energy scaling)
#
# Calibration events:
#   Aug 1972, Bastille Day 2000, Halloween 2003, Jan 2005,
#   Mar 1989, Sep 2017, May 2024, Carrington 1859.
#   Bastille Day 2000: 1106 mSv predicted vs 1015 observed (+9%).
#
# Context: Bz is an empirical PROXY for CME energy, not a direct
# radiation driver.  In deep space there is no magnetospheric
# shielding — astronauts receive the FULL solar energetic particle flux.
#
# References:
#   - Townsend et al. 2003 (Aug 1972 dose reconstruction)
#   - Kim et al. 2015 (SPE dose estimates)
#   - Cucinotta et al. 2010 (NASA Constellation programme)
#   - NCRP Report 132
#   - NASA-STD-3001 Vol.1 Rev A (2022)

SEVERITY_CONFIG = {
    't_exposure_hours': 10.0,  # Pessimistic worst-case EVA

    # Dosimetry formula coefficients
    'dose_coefficient': 0.0132,  # Median of 8 SPE events (robust)
    'bz_exponent': 1.3,          # Empirical Bz-energy scaling

    # ── Option B: DOSE-BASED severity classification ──
    # Classification is by PREDICTED DOSE, not by Bz ranges.
    # Extreme threshold = NASA 30-day limit (250 mSv).
    'dose_thresholds': {
        'low': (10, 50),        # Class 0:  4-20% of NASA 30-day limit
        'moderate': (50, 100),  # Class 1: 20-40% of NASA 30-day limit
        'high': (100, 250),     # Class 2: 40-100% of NASA 30-day limit
        'extreme': (250, None)  # Class 3: >100% — exceeds 30-day limit
    },

    # Illustrative Bz ranges (for documentation only — NOT used for
    # classification).  Actual Bz thresholds depend on v_CME.
    # Reference: v_CME = 800 km/s, t = 10 h.
    'bz_thresholds_illustrative': {
        'low': (-7, -5),          # ~10-50 mSv at 800 km/s
        'moderate': (-13, -7),    # ~50-100 mSv at 800 km/s
        'high': (-25, -13),       # ~100-250 mSv at 800 km/s
        'extreme': (-80, -25)     # >250 mSv at 800 km/s
    },

    # NASA dose limits (deep-space, NASA-STD-3001 Vol.1 Rev A)
    'nasa_30day_limit_mSv': 250,
    'nasa_annual_limit_mSv': 500,
    'nasa_career_limit_mSv': 600,

    'class_names': ['Low', 'Moderate', 'High', 'Extreme'],
    'n_classes': 4,
}

# ============================================================================
# VALIDATION TARGETS
# ============================================================================
VALIDATION_TARGETS = {
    'bastille_day_2000': {
        'date': '2000-07-14',
        'expected_bz': -60.0,  # nT (measured at ACE)
        'bz_tolerance': 10.0,  # MAE target per whitepaper Table 4
        'expected_severity': 3, # Extreme
        'speed_kms': 1674,
        'width_deg': 360,
        'source_lat': 22,
        'source_lon': 7,
    }
}

# ============================================================================
# OUTPUT PATHS
# ============================================================================
OUTPUT_CONFIG = {
    'model_dir': 'output/models',
    'checkpoint_dir': 'output/checkpoints',
    'logs_dir': 'output/logs',
    'figures_dir': 'output/figures',
}


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("HELIOS Neural Network Configuration")
    print("=" * 60)
    print(f"\nModel Architecture:")
    print(f"  Input dimension: {MODEL_CONFIG['input_dim']}")
    print(f"  Encoder layers: {MODEL_CONFIG['encoder_layers']}")
    print(f"  Loss weights: alpha={MODEL_CONFIG['alpha_bz']}, beta={MODEL_CONFIG['beta_severity']}")
    print(f"\nTraining:")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"\nDataset:")
    print(f"  Synthetic events: {DATASET_CONFIG['n_synthetic_events']}")
    print(f"  Historical events: {DATASET_CONFIG['n_historical_events']}")
    print(f"\nSeverity Classes (dose-based, deep-space):")
    for i, name in enumerate(SEVERITY_CONFIG['class_names']):
        dose = SEVERITY_CONFIG['dose_thresholds'][name.lower()]
        bz_ill = SEVERITY_CONFIG['bz_thresholds_illustrative'][name.lower()]
        print(f"  {i}: {name} - Dose: {dose} mSv, Bz(illustrative @800km/s): {bz_ill} nT")
