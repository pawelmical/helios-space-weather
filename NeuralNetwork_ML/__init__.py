"""
HELIOS Neural Network ML Module
================================
AI-based Bz prediction and severity classification for CME geoeffectiveness.

Architecture:
    Input (16 features) -> Shared Encoder [16->128->256->128->64]
                               |
              +---------------+---------------+
              |                               |
         Bz Head [64->32->2]          Severity Head [64->32->4]
         (mean, log_variance)          (4 class logits)

Usage:
    # Generate dataset
    from NeuralNetwork_ML.dataset_generator import generate_combined_dataset
    features, bz, severity, ids = generate_combined_dataset()

    # Train model (run from command line)
    python -m NeuralNetwork_ML.train

    # For PyTorch-dependent imports, import explicitly:
    from NeuralNetwork_ML.model import HELIOSDualHeadModel, create_model
    from NeuralNetwork_ML.preprocessing import CMEDataset, create_data_loaders

NOTE: PyTorch-dependent modules are NOT imported at package level to avoid
a naming conflict with Python's built-in 'code' module (the parent directory
contains a 'code/' folder). Import them explicitly when needed.

Author: HELIOS Team
Date: February 2026
"""

__version__ = "1.0.0"
__author__ = "HELIOS Team"

# Check if PyTorch is available (without importing it)
def _check_torch_available():
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

HAS_TORCH = _check_torch_available()

# Configuration (no PyTorch dependency)
from NeuralNetwork_ML.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DATASET_CONFIG,
    SEVERITY_CONFIG,
    FEATURE_NAMES,
    FEATURE_BOUNDS,
)

# Feature engineering (no PyTorch dependency)
from NeuralNetwork_ML.features import (
    CMEFeatures,
    extract_features,
    create_bastille_day_features,
)

# Severity classification (no PyTorch dependency)
from NeuralNetwork_ML.severity import (
    calculate_dose,
    calculate_severity,
    bz_to_severity_class,
    DosimetryResult,
)

# Preprocessing - only non-PyTorch parts
from NeuralNetwork_ML.preprocessing import (
    FeatureNormalizer,
    BzNormalizer,
)

# Dataset generation (no PyTorch dependency)
from NeuralNetwork_ML.dataset_generator import (
    generate_synthetic_dataset,
    generate_historical_dataset,
    generate_combined_dataset,
    generate_bz_from_physics,
    HISTORICAL_EVENTS,
)

__all__ = [
    '__version__',
    '__author__',
    'HAS_TORCH',
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    'DATASET_CONFIG',
    'SEVERITY_CONFIG',
    'FEATURE_NAMES',
    'FEATURE_BOUNDS',
    'CMEFeatures',
    'extract_features',
    'create_bastille_day_features',
    'calculate_dose',
    'calculate_severity',
    'bz_to_severity_class',
    'DosimetryResult',
    'FeatureNormalizer',
    'BzNormalizer',
    'generate_synthetic_dataset',
    'generate_historical_dataset',
    'generate_combined_dataset',
    'generate_bz_from_physics',
    'HISTORICAL_EVENTS',
]
