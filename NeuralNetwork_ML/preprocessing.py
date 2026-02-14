"""
HELIOS Data Preprocessing
==========================
Feature normalization, data loading, and PyTorch dataset classes.

NOTE: PyTorch imports are deferred to avoid conflict with the 'code' module
in the parent directory. Import torch inside functions that need it.

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import pandas as pd

from NeuralNetwork_ML.config import (
    FEATURE_BOUNDS, FEATURE_NAMES, TRAINING_CONFIG, BZ_CONFIG
)

# Check if PyTorch is available without importing it
def _check_torch():
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

HAS_TORCH = _check_torch()


@dataclass
class NormalizationStats:
    """Statistics for feature normalization."""
    means: np.ndarray
    stds: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray


class FeatureNormalizer:
    """
    Normalizes features to [0, 1] range using min-max scaling.

    Uses predefined bounds from config to ensure consistent normalization
    across training and inference.
    """

    def __init__(self, use_bounds: bool = True):
        """
        Parameters
        ----------
        use_bounds : bool
            If True, use predefined bounds from config.
            If False, fit bounds from data.
        """
        self.use_bounds = use_bounds
        self.mins = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.maxs = np.ones(len(FEATURE_NAMES), dtype=np.float32)
        self.fitted = False

        if use_bounds:
            self._set_bounds_from_config()
            self.fitted = True

    def _set_bounds_from_config(self):
        """Set normalization bounds from config."""
        for i, name in enumerate(FEATURE_NAMES):
            if name in FEATURE_BOUNDS:
                self.mins[i] = FEATURE_BOUNDS[name][0]
                self.maxs[i] = FEATURE_BOUNDS[name][1]

    def fit(self, X: np.ndarray):
        """
        Fit normalizer to data (only if use_bounds=False).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        if not self.use_bounds:
            self.mins = X.min(axis=0).astype(np.float32)
            self.maxs = X.max(axis=0).astype(np.float32)
            # Avoid division by zero
            self.maxs = np.where(self.maxs == self.mins, self.mins + 1, self.maxs)
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        X_normalized : np.ndarray
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        X_norm = (X - self.mins) / (self.maxs - self.mins)
        return np.clip(X_norm, 0, 1).astype(np.float32)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize features back to original scale."""
        return (X_norm * (self.maxs - self.mins) + self.mins).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_state(self) -> Dict:
        """Get normalizer state for serialization."""
        return {
            'mins': self.mins.tolist(),
            'maxs': self.maxs.tolist(),
            'use_bounds': self.use_bounds,
            'fitted': self.fitted
        }

    @classmethod
    def from_state(cls, state: Dict) -> 'FeatureNormalizer':
        """Create normalizer from saved state."""
        normalizer = cls(use_bounds=state['use_bounds'])
        normalizer.mins = np.array(state['mins'], dtype=np.float32)
        normalizer.maxs = np.array(state['maxs'], dtype=np.float32)
        normalizer.fitted = state['fitted']
        return normalizer


class BzNormalizer:
    """Normalizes Bz values to [0, 1] range."""

    def __init__(self, bz_min: float = None, bz_max: float = None):
        """
        Parameters
        ----------
        bz_min : float
            Minimum Bz value (default from config: -80 nT)
        bz_max : float
            Maximum Bz value (default from config: 0 nT)
        """
        self.bz_min = bz_min if bz_min is not None else BZ_CONFIG['bz_min']
        self.bz_max = bz_max if bz_max is not None else BZ_CONFIG['bz_max']

    def transform(self, bz: np.ndarray) -> np.ndarray:
        """Normalize Bz to [0, 1]."""
        bz_norm = (bz - self.bz_min) / (self.bz_max - self.bz_min)
        return np.clip(bz_norm, 0, 1).astype(np.float32)

    def inverse_transform(self, bz_norm: np.ndarray) -> np.ndarray:
        """Denormalize Bz back to nT."""
        return (bz_norm * (self.bz_max - self.bz_min) + self.bz_min).astype(np.float32)

    def get_state(self) -> Dict:
        """Get normalizer state for serialization."""
        return {
            'bz_min': self.bz_min,
            'bz_max': self.bz_max
        }

    @classmethod
    def from_state(cls, state: Dict) -> 'BzNormalizer':
        """Create normalizer from saved state."""
        return cls(bz_min=state['bz_min'], bz_max=state['bz_max'])


class CMEDataset:
    """
    PyTorch Dataset for CME Bz prediction.

    Each sample contains:
    - features: 16-dimensional normalized feature vector
    - bz_target: Normalized Bz value
    - severity_target: Severity class (0-3)

    NOTE: Inherits from torch.utils.data.Dataset dynamically to avoid
    import conflicts with the 'code' module.
    """

    def __init__(
        self,
        features: np.ndarray,
        bz_values: np.ndarray,
        severity_classes: np.ndarray,
        feature_normalizer: Optional[FeatureNormalizer] = None,
        bz_normalizer: Optional[BzNormalizer] = None
    ):
        """
        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, 16)
        bz_values : np.ndarray
            Bz values in nT (n_samples,)
        severity_classes : np.ndarray
            Severity class labels (n_samples,)
        feature_normalizer : FeatureNormalizer, optional
            Pre-fitted normalizer (creates new one if None)
        bz_normalizer : BzNormalizer, optional
            Pre-fitted Bz normalizer
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CMEDataset. Install with: pip install torch")

        # Import torch here to avoid module-level import conflict
        import torch
        self._torch = torch

        self.feature_normalizer = feature_normalizer or FeatureNormalizer(use_bounds=True)
        self.bz_normalizer = bz_normalizer or BzNormalizer()

        # Normalize features
        self.features = self.feature_normalizer.transform(features)

        # Normalize Bz
        self.bz_values = self.bz_normalizer.transform(bz_values)

        # Severity classes (integer labels)
        self.severity_classes = severity_classes.astype(np.int64)

        self.n_samples = len(features)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        features : torch.Tensor
            Shape (16,)
        bz_target : torch.Tensor
            Shape (1,) - normalized Bz
        severity_target : torch.Tensor
            Shape () - class label
        """
        torch = self._torch
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor([self.bz_values[idx]], dtype=torch.float32),
            torch.tensor(self.severity_classes[idx], dtype=torch.long)
        )


def create_data_loaders(
    dataset: CMEDataset,
    batch_size: int = None,
    validation_split: float = None,
    seed: int = None
) -> Tuple:
    """
    Create train and validation DataLoaders.

    Parameters
    ----------
    dataset : CMEDataset
        Full dataset
    batch_size : int
        Batch size (default from config)
    validation_split : float
        Fraction for validation (default from config)
    seed : int
        Random seed (default from config)

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    # Import torch here to avoid module-level import conflict
    import torch
    from torch.utils.data import DataLoader

    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    validation_split = validation_split or TRAINING_CONFIG['validation_split']
    seed = seed or TRAINING_CONFIG['random_seed']

    # Split dataset
    n_total = len(dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader


def load_events_csv(filepath: str) -> List[Dict]:
    """
    Load events from CSV file.

    Parameters
    ----------
    filepath : str
        Path to events CSV file

    Returns
    -------
    events : List[Dict]
        List of event dictionaries
    """
    df = pd.read_csv(filepath)
    return df.to_dict('records')


def prepare_dataset_arrays(
    features_list: List[np.ndarray],
    bz_list: List[float],
    severity_list: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare arrays for dataset creation.

    Parameters
    ----------
    features_list : List[np.ndarray]
        List of feature vectors
    bz_list : List[float]
        List of Bz values
    severity_list : List[int]
        List of severity classes

    Returns
    -------
    features : np.ndarray
        Shape (n_samples, 16)
    bz_values : np.ndarray
        Shape (n_samples,)
    severity_classes : np.ndarray
        Shape (n_samples,)
    """
    features = np.array(features_list, dtype=np.float32)
    bz_values = np.array(bz_list, dtype=np.float32)
    severity_classes = np.array(severity_list, dtype=np.int64)

    return features, bz_values, severity_classes


if __name__ == "__main__":
    # Test preprocessing
    print("=" * 60)
    print("HELIOS Preprocessing - Test")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Random features within bounds
    features = np.zeros((n_samples, 16), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        low, high = FEATURE_BOUNDS[name]
        features[:, i] = np.random.uniform(low, high, n_samples)

    # Random Bz values
    bz_values = np.random.uniform(-60, -10, n_samples).astype(np.float32)

    # Random severity classes
    severity_classes = np.random.randint(0, 4, n_samples).astype(np.int64)

    print(f"\nSample data created:")
    print(f"  Features shape: {features.shape}")
    print(f"  Bz values shape: {bz_values.shape}")
    print(f"  Severity shape: {severity_classes.shape}")

    # Test feature normalizer
    print("\nFeature Normalizer:")
    normalizer = FeatureNormalizer(use_bounds=True)
    features_norm = normalizer.transform(features)
    print(f"  Normalized range: [{features_norm.min():.3f}, {features_norm.max():.3f}]")

    # Test roundtrip
    features_rt = normalizer.inverse_transform(features_norm)
    error = np.abs(features - features_rt).max()
    print(f"  Roundtrip max error: {error:.6f}")

    # Test Bz normalizer
    print("\nBz Normalizer:")
    bz_norm = BzNormalizer()
    bz_normalized = bz_norm.transform(bz_values)
    print(f"  Normalized range: [{bz_normalized.min():.3f}, {bz_normalized.max():.3f}]")

    bz_rt = bz_norm.inverse_transform(bz_normalized)
    bz_error = np.abs(bz_values - bz_rt).max()
    print(f"  Roundtrip max error: {bz_error:.6f}")

    # Test PyTorch dataset
    if HAS_TORCH:
        print("\nPyTorch Dataset:")
        dataset = CMEDataset(features, bz_values, severity_classes)
        print(f"  Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"  Sample shapes: features={sample[0].shape}, bz={sample[1].shape}, severity={sample[2].shape}")

        # Test data loaders
        train_loader, val_loader = create_data_loaders(dataset, batch_size=16)
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # Get one batch
        batch = next(iter(train_loader))
        print(f"  Batch shapes: features={batch[0].shape}, bz={batch[1].shape}, severity={batch[2].shape}")
    else:
        print("\nPyTorch not installed - skipping Dataset tests")

    print("\nPreprocessing tests completed!")
