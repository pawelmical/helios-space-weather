"""
HELIOS Model Training Script
=============================
Train the dual-head Bz prediction model.

Usage:
    python -m NeuralNetwork_ML.train
    python -m NeuralNetwork_ML.train --epochs 100 --batch-size 64 --lr 1e-3

Author: HELIOS Team
Date: February 2026
"""

import os
import sys

# Fix for 'helios_code' module conflict - must be done BEFORE importing torch
# The project's 'helios_code/' directory no longer shadows Python's built-in 'code' module
_code_module_backup = sys.modules.pop('code', None)
_code_submodules = {k: v for k, v in list(sys.modules.items()) if k.startswith('code.')}
for k in _code_submodules:
    sys.modules.pop(k, None)

_mvptest_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path_backup = sys.path.copy()
sys.path = [p for p in sys.path if not (p and os.path.exists(os.path.join(p, 'helios_code', '__init__.py')))]

import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import json

from NeuralNetwork_ML.config import (
    TRAINING_CONFIG, OUTPUT_CONFIG, DATASET_CONFIG, SEVERITY_CONFIG
)
from NeuralNetwork_ML.dataset_generator import generate_combined_dataset
from NeuralNetwork_ML.preprocessing import (
    CMEDataset, create_data_loaders, FeatureNormalizer, BzNormalizer
)
from NeuralNetwork_ML.validation import validate_model, print_validation_report

# PyTorch imports
try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from NeuralNetwork_ML.model import create_model, create_loss_function, HELIOSDualHeadModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Restore sys.path and code module
sys.path = _path_backup
if _code_module_backup is not None:
    sys.modules['code'] = _code_module_backup
for k, v in _code_submodules.items():
    sys.modules[k] = v


def setup_directories():
    """Create output directories."""
    for dir_name, dir_path in OUTPUT_CONFIG.items():
        full_path = os.path.join(os.path.dirname(__file__), '..', dir_path)
        os.makedirs(full_path, exist_ok=True)


def get_output_path(subdir: str, filename: str) -> str:
    """Get full path for output file."""
    base = os.path.join(os.path.dirname(__file__), '..')
    return os.path.join(base, OUTPUT_CONFIG[subdir], filename)


def train_epoch(
    model: 'HELIOSDualHeadModel',
    train_loader,
    loss_fn,
    optimizer,
    device: str
) -> Dict[str, float]:
    """
    Train for one epoch.

    Parameters
    ----------
    model : HELIOSDualHeadModel
        Model to train
    train_loader : DataLoader
        Training data loader
    loss_fn : MultiTaskLoss
        Loss function
    optimizer : Optimizer
        Optimizer
    device : str
        Device

    Returns
    -------
    metrics : dict
        Training metrics for this epoch
    """
    model.train()
    total_loss = 0
    total_bz_loss = 0
    total_sev_loss = 0
    n_batches = 0

    for features, bz_target, severity_target in train_loader:
        features = features.to(device)
        bz_target = bz_target.to(device)
        severity_target = severity_target.to(device)

        optimizer.zero_grad()

        bz_mean, bz_logvar, severity_logits = model(features)
        loss, loss_dict = loss_fn(
            bz_mean, bz_logvar, severity_logits,
            bz_target, severity_target
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['total_loss']
        total_bz_loss += loss_dict['bz_loss']
        total_sev_loss += loss_dict['severity_loss']
        n_batches += 1

    return {
        'train_loss': total_loss / n_batches,
        'train_bz_loss': total_bz_loss / n_batches,
        'train_severity_loss': total_sev_loss / n_batches,
    }


def validate_epoch(
    model: 'HELIOSDualHeadModel',
    val_loader,
    loss_fn,
    device: str
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Parameters
    ----------
    model : HELIOSDualHeadModel
        Model to validate
    val_loader : DataLoader
        Validation data loader
    loss_fn : MultiTaskLoss
        Loss function
    device : str
        Device

    Returns
    -------
    metrics : dict
        Validation metrics for this epoch
    """
    model.eval()
    total_loss = 0
    total_bz_loss = 0
    total_sev_loss = 0
    correct_severity = 0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for features, bz_target, severity_target in val_loader:
            features = features.to(device)
            bz_target = bz_target.to(device)
            severity_target = severity_target.to(device)

            bz_mean, bz_logvar, severity_logits = model(features)
            loss, loss_dict = loss_fn(
                bz_mean, bz_logvar, severity_logits,
                bz_target, severity_target
            )

            total_loss += loss_dict['total_loss']
            total_bz_loss += loss_dict['bz_loss']
            total_sev_loss += loss_dict['severity_loss']

            pred_class = torch.argmax(severity_logits, dim=-1)
            correct_severity += (pred_class == severity_target).sum().item()
            total_samples += severity_target.size(0)
            n_batches += 1

    return {
        'val_loss': total_loss / n_batches,
        'val_bz_loss': total_bz_loss / n_batches,
        'val_severity_loss': total_sev_loss / n_batches,
        'val_severity_accuracy': correct_severity / total_samples,
    }


def train_model(args) -> Tuple['HELIOSDualHeadModel', Dict]:
    """
    Main training function.

    Parameters
    ----------
    args : Namespace
        Command line arguments

    Returns
    -------
    model : HELIOSDualHeadModel
        Trained model
    results : dict
        Training history and validation results
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    print("\n" + "=" * 65)
    print("HELIOS Bz Prediction Model Training")
    print("=" * 65)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup
    setup_directories()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Synthetic events: {args.n_synthetic}")

    # Generate dataset
    print("\nGenerating dataset...")
    features, bz_values, severity_classes, event_ids = generate_combined_dataset(
        n_synthetic=args.n_synthetic,
        seed=args.seed
    )
    n_historical = len(event_ids) - args.n_synthetic
    print(f"  Total samples: {len(features)}")
    print(f"  Synthetic: {args.n_synthetic}")
    print(f"  Historical: {n_historical}")

    # Show class distribution
    class_counts = np.bincount(severity_classes, minlength=4)
    print(f"\nSeverity class distribution:")
    for i, (name, count) in enumerate(zip(SEVERITY_CONFIG['class_names'], class_counts)):
        print(f"  {name}: {count} ({100*count/len(severity_classes):.1f}%)")

    # Create normalizers
    feature_normalizer = FeatureNormalizer(use_bounds=True)
    bz_normalizer = BzNormalizer()

    # Create dataset
    dataset = CMEDataset(
        features, bz_values, severity_classes,
        feature_normalizer, bz_normalizer
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        seed=args.seed
    )
    print(f"\nData loaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    model = create_model(device)
    n_params = model.count_parameters()
    print(f"\nModel:")
    print(f"  Parameters: {n_params:,}")

    # Create loss with class weights for imbalanced severity classification
    if TRAINING_CONFIG.get('use_class_weights', False):
        class_weights = torch.tensor(
            TRAINING_CONFIG['severity_class_weights'],
            dtype=torch.float32,
            device=device
        )
        print(f"  Class weights: {TRAINING_CONFIG['severity_class_weights']}")
    else:
        class_weights = None
    loss_fn = create_loss_function(class_weights=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("\n" + "-" * 65)
    print("Training...")
    print("-" * 65)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_bz_loss': [], 'val_bz_loss': [],
        'train_severity_loss': [], 'val_severity_loss': [],
        'val_severity_accuracy': [], 'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = get_output_path('model_dir', 'best_model.pt')

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)

        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        # Update scheduler
        scheduler.step(val_metrics['val_loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        for key, value in train_metrics.items():
            history[key].append(value)
        for key, value in val_metrics.items():
            history[key].append(value)
        history['lr'].append(current_lr)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_metrics['train_loss']:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"severity_acc={val_metrics['val_severity_accuracy']:.2%}, "
                  f"lr={current_lr:.2e}")

        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    print(f"\nLoading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Validate on Bastille Day
    print("\n" + "=" * 65)
    print("Validating on Bastille Day 2000...")
    validation_results = validate_model(
        model, feature_normalizer, bz_normalizer, device
    )
    print_validation_report(validation_results)

    # Save final artifacts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model with all training info
    model_path = get_output_path('model_dir', f'model_{timestamp}.pt')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'feature_normalizer': feature_normalizer.get_state(),
        'bz_normalizer': bz_normalizer.get_state(),
        'history': history,
        'args': vars(args),
        'validation_results': validation_results,
        'n_parameters': n_params,
        'device': device,
    }
    torch.save(save_dict, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save history as JSON
    history_path = get_output_path('logs_dir', f'history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'validation_results': validation_results,
            'args': vars(args),
        }, f, indent=2)
    print(f"History saved to: {history_path}")

    return model, {'history': history, 'validation': validation_results}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train HELIOS Bz prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--epochs', type=int, default=TRAINING_CONFIG['epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=TRAINING_CONFIG['weight_decay'],
        help='Weight decay for AdamW'
    )
    parser.add_argument(
        '--val-split', type=float, default=TRAINING_CONFIG['validation_split'],
        help='Validation split fraction'
    )
    parser.add_argument(
        '--patience', type=int, default=TRAINING_CONFIG['early_stopping_patience'],
        help='Early stopping patience'
    )
    parser.add_argument(
        '--n-synthetic', type=int, default=DATASET_CONFIG['n_synthetic_events'],
        help='Number of synthetic training events'
    )
    parser.add_argument(
        '--seed', type=int, default=TRAINING_CONFIG['random_seed'],
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    try:
        model, results = train_model(args)
        print("\n" + "=" * 65)
        print("Training complete!")
        print("=" * 65)

        # Summary
        val_results = results['validation']
        print(f"\nFinal Results:")
        print(f"  Bastille Day Bz Error: {val_results['bz_mae']:.1f} nT (target: <= 7 nT)")
        print(f"  Severity Correct: {val_results['severity_correct']}")
        print(f"  Validation Passes: {val_results['validation_passes']}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo install PyTorch, run:")
        print("  pip install torch")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
