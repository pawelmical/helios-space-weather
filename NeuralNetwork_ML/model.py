"""
HELIOS Neural Network Model
============================
Dual-head architecture for Bz regression and severity classification.

Architecture:
    Input (16) -> Shared Encoder [16->64->128->64]
                      |
         +-----------+-----------+
         |                       |
    Bz Head [64->32->2]    Severity Head [64->32->4]
    (mean, log_var)        (4 class logits)

Loss Function:
    L_total = alpha * L_bz + beta * L_severity

    Where:
    - L_bz: Heteroscedastic regression loss
    - L_severity: Cross-entropy classification loss
    - alpha = 0.7, beta = 0.3

NOTE: PyTorch imports are deferred to avoid conflict with the 'code' module
in the parent directory. torch is imported when classes are instantiated.

Author: HELIOS Team
Date: February 2026
"""

from typing import Tuple, Dict, Optional

from NeuralNetwork_ML.config import MODEL_CONFIG

# Check if PyTorch is available without importing it
def _check_torch():
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

HAS_TORCH = _check_torch()

# Import torch lazily only when needed
_torch = None
_nn = None
_F = None

def _get_torch():
    """Get torch module, importing it if necessary."""
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
    return _torch, _nn, _F


if HAS_TORCH:
    # Import torch with workaround for 'helios_code' module conflict
    # The project's 'helios_code/' directory no longer shadows Python's built-in 'code' module
    import sys
    import os

    # Save and remove the project's 'code' module from cache if present
    _code_module_backup = sys.modules.pop('code', None)
    _code_submodules = {k: v for k, v in sys.modules.items() if k.startswith('code.')}
    for k in _code_submodules:
        sys.modules.pop(k, None)

    # Temporarily fix sys.path
    _mvptest_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _path_backup = sys.path.copy()
    sys.path = [p for p in sys.path if not (p and os.path.exists(os.path.join(p, 'helios_code', '__init__.py')))]

    # Now import torch safely
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Restore sys.path
    sys.path = _path_backup

    # Restore the project's code module
    if _code_module_backup is not None:
        sys.modules['code'] = _code_module_backup
    for k, v in _code_submodules.items():
        sys.modules[k] = v

    class SharedEncoder(nn.Module):
        """
        Shared feature encoder.

        Architecture: 16 -> 64 -> 128 -> 64 with ReLU activations.
        """

        def __init__(
            self,
            input_dim: int = 16,
            hidden_dims: list = None,
            dropout: float = 0.2
        ):
            """
            Parameters
            ----------
            input_dim : int
                Input feature dimension (default: 16)
            hidden_dims : list
                Hidden layer dimensions (default from config)
            dropout : float
                Dropout rate (default: 0.2)
            """
            super().__init__()

            if hidden_dims is None:
                hidden_dims = MODEL_CONFIG['encoder_layers'][1:]  # [64, 128, 64]

            layers = []
            prev_dim = input_dim

            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = dim

            self.encoder = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through encoder.

            Parameters
            ----------
            x : torch.Tensor
                Input features (batch_size, 16)

            Returns
            -------
            encoded : torch.Tensor
                Encoded representation (batch_size, 64)
            """
            return self.encoder(x)


    class BzRegressionHead(nn.Module):
        """
        Bz prediction head with uncertainty estimation.

        Outputs both mean and log-variance for heteroscedastic loss.
        This allows the model to learn input-dependent uncertainty.
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 32
        ):
            """
            Parameters
            ----------
            input_dim : int
                Input dimension from encoder
            hidden_dim : int
                Hidden layer dimension
            """
            super().__init__()

            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc_mean = nn.Linear(hidden_dim, 1)
            self.fc_logvar = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Encoded features (batch_size, input_dim)

            Returns
            -------
            mean : torch.Tensor
                Predicted mean Bz (batch_size, 1)
            log_var : torch.Tensor
                Predicted log variance (batch_size, 1)
            """
            h = F.relu(self.bn1(self.fc1(x)))
            mean = self.fc_mean(h)
            log_var = self.fc_logvar(h)
            return mean, log_var


    class SeverityClassificationHead(nn.Module):
        """
        Severity classification head.

        4-class output: Low, Moderate, High, Extreme
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 32,
            n_classes: int = 4
        ):
            """
            Parameters
            ----------
            input_dim : int
                Input dimension from encoder
            hidden_dim : int
                Hidden layer dimension
            n_classes : int
                Number of output classes (default: 4)
            """
            super().__init__()

            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Encoded features (batch_size, input_dim)

            Returns
            -------
            logits : torch.Tensor
                Class logits (batch_size, n_classes)
            """
            h = F.relu(self.bn1(self.fc1(x)))
            logits = self.fc_out(h)
            return logits


    class HELIOSDualHeadModel(nn.Module):
        """
        Complete dual-head model for Bz and severity prediction.

        Multi-task learning with shared encoder.
        """

        def __init__(
            self,
            input_dim: int = 16,
            encoder_dims: list = None,
            dropout: float = 0.2,
            n_severity_classes: int = 4
        ):
            """
            Parameters
            ----------
            input_dim : int
                Input feature dimension (default: 16)
            encoder_dims : list
                Encoder hidden dimensions (default from config)
            dropout : float
                Dropout rate (default: 0.2)
            n_severity_classes : int
                Number of severity classes (default: 4)
            """
            super().__init__()

            if encoder_dims is None:
                encoder_dims = MODEL_CONFIG['encoder_layers'][1:]

            self.encoder = SharedEncoder(input_dim, encoder_dims, dropout)
            encoder_output_dim = self.encoder.output_dim

            self.bz_head = BzRegressionHead(encoder_output_dim)
            self.severity_head = SeverityClassificationHead(
                encoder_output_dim,
                n_classes=n_severity_classes
            )

        def forward(
            self,
            x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass through complete model.

            Parameters
            ----------
            x : torch.Tensor
                Input features (batch_size, 16)

            Returns
            -------
            bz_mean : torch.Tensor
                Predicted Bz mean (batch_size, 1)
            bz_logvar : torch.Tensor
                Predicted Bz log variance (batch_size, 1)
            severity_logits : torch.Tensor
                Severity class logits (batch_size, 4)
            """
            encoded = self.encoder(x)
            bz_mean, bz_logvar = self.bz_head(encoded)
            severity_logits = self.severity_head(encoded)
            return bz_mean, bz_logvar, severity_logits

        def predict(
            self,
            x: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            """
            Inference mode prediction.

            Parameters
            ----------
            x : torch.Tensor
                Input features (batch_size, 16)

            Returns
            -------
            dict with keys:
                'bz_mean': Predicted Bz (normalized)
                'bz_std': Predicted uncertainty (std dev)
                'severity_probs': Class probabilities
                'severity_class': Predicted class
            """
            self.eval()
            with torch.no_grad():
                bz_mean, bz_logvar, severity_logits = self.forward(x)
                bz_std = torch.exp(0.5 * bz_logvar)
                severity_probs = F.softmax(severity_logits, dim=-1)
                severity_class = torch.argmax(severity_probs, dim=-1)

            return {
                'bz_mean': bz_mean,
                'bz_std': bz_std,
                'severity_probs': severity_probs,
                'severity_class': severity_class
            }

        def count_parameters(self) -> int:
            """Count total trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    # ============================================================================
    # LOSS FUNCTIONS
    # ============================================================================

    class HeteroscedasticLoss(nn.Module):
        """
        Heteroscedastic regression loss with learned uncertainty.

        Loss = 0.5 * exp(-log_var) * (y - y_pred)^2 + 0.5 * log_var

        This allows the model to learn input-dependent uncertainty:
        - High uncertainty -> model is less penalized for errors
        - Low uncertainty -> model is more penalized for errors
        - Regularization term (0.5 * log_var) prevents trivial solution
        """

        def forward(
            self,
            mean: torch.Tensor,
            log_var: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute heteroscedastic loss.

            Parameters
            ----------
            mean : torch.Tensor
                Predicted mean (batch_size, 1)
            log_var : torch.Tensor
                Predicted log variance (batch_size, 1)
            target : torch.Tensor
                Ground truth (batch_size, 1)

            Returns
            -------
            loss : torch.Tensor
                Scalar loss value
            """
            precision = torch.exp(-log_var)
            loss = 0.5 * precision * (target - mean)**2 + 0.5 * log_var
            return loss.mean()


    class MultiTaskLoss(nn.Module):
        """
        Combined loss for multi-task learning.

        L_total = alpha * L_bz + beta * L_severity

        Where:
            L_bz: Heteroscedastic regression loss
            L_severity: Cross-entropy classification loss
        """

        def __init__(
            self,
            alpha: float = 0.7,
            beta: float = 0.3,
            class_weights: Optional[torch.Tensor] = None
        ):
            """
            Parameters
            ----------
            alpha : float
                Weight for Bz regression loss (default: 0.7)
            beta : float
                Weight for severity classification loss (default: 0.3)
            class_weights : torch.Tensor, optional
                Class weights for imbalanced classification
            """
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.bz_loss = HeteroscedasticLoss()
            self.severity_loss = nn.CrossEntropyLoss(weight=class_weights)

        def forward(
            self,
            bz_mean: torch.Tensor,
            bz_logvar: torch.Tensor,
            severity_logits: torch.Tensor,
            bz_target: torch.Tensor,
            severity_target: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute combined loss.

            Parameters
            ----------
            bz_mean : torch.Tensor
                Predicted Bz mean (batch_size, 1)
            bz_logvar : torch.Tensor
                Predicted Bz log variance (batch_size, 1)
            severity_logits : torch.Tensor
                Severity class logits (batch_size, 4)
            bz_target : torch.Tensor
                Ground truth Bz (batch_size, 1)
            severity_target : torch.Tensor
                Ground truth severity class (batch_size,)

            Returns
            -------
            total_loss : torch.Tensor
                Combined weighted loss
            loss_dict : dict
                Individual loss components
            """
            l_bz = self.bz_loss(bz_mean, bz_logvar, bz_target)
            l_severity = self.severity_loss(severity_logits, severity_target)

            total = self.alpha * l_bz + self.beta * l_severity

            return total, {
                'bz_loss': l_bz.item(),
                'severity_loss': l_severity.item(),
                'total_loss': total.item()
            }


    def create_model(device: str = 'cpu') -> HELIOSDualHeadModel:
        """
        Create model with default configuration.

        Parameters
        ----------
        device : str
            Device to place model on ('cpu' or 'cuda')

        Returns
        -------
        model : HELIOSDualHeadModel
            Initialized model
        """
        model = HELIOSDualHeadModel(
            input_dim=MODEL_CONFIG['input_dim'],
            encoder_dims=MODEL_CONFIG['encoder_layers'][1:],
            dropout=MODEL_CONFIG['encoder_dropout'],
            n_severity_classes=4
        )
        return model.to(device)


    def create_loss_function(class_weights: Optional[torch.Tensor] = None) -> MultiTaskLoss:
        """
        Create loss function with default weights.

        Parameters
        ----------
        class_weights : torch.Tensor, optional
            Class weights for imbalanced classification

        Returns
        -------
        loss_fn : MultiTaskLoss
            Initialized loss function
        """
        return MultiTaskLoss(
            alpha=MODEL_CONFIG['alpha_bz'],
            beta=MODEL_CONFIG['beta_severity'],
            class_weights=class_weights
        )

else:
    # Placeholder classes when PyTorch is not available
    class HELIOSDualHeadModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")

    def create_model(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

    def create_loss_function(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")


if __name__ == "__main__":
    print("=" * 60)
    print("HELIOS Neural Network Model - Test")
    print("=" * 60)

    if not HAS_TORCH:
        print("\nPyTorch not installed - skipping tests")
        print("Install with: pip install torch")
    else:
        import numpy as np

        # Create model
        print("\nCreating model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        model = create_model(device)
        print(f"  Total parameters: {model.count_parameters():,}")

        # Print architecture
        print("\nModel Architecture:")
        print(model)

        # Test forward pass
        print("\nForward Pass Test:")
        batch_size = 8
        x = torch.randn(batch_size, 16).to(device)

        model.train()
        bz_mean, bz_logvar, severity_logits = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Bz mean shape: {bz_mean.shape}")
        print(f"  Bz logvar shape: {bz_logvar.shape}")
        print(f"  Severity logits shape: {severity_logits.shape}")

        # Test predict
        print("\nPredict Test:")
        predictions = model.predict(x)
        print(f"  Bz mean: {predictions['bz_mean'].shape}")
        print(f"  Bz std: {predictions['bz_std'].shape}")
        print(f"  Severity probs: {predictions['severity_probs'].shape}")
        print(f"  Severity class: {predictions['severity_class'].shape}")

        # Test loss function
        print("\nLoss Function Test:")
        loss_fn = create_loss_function()

        bz_target = torch.randn(batch_size, 1).to(device)
        severity_target = torch.randint(0, 4, (batch_size,)).to(device)

        model.train()
        bz_mean, bz_logvar, severity_logits = model(x)
        total_loss, loss_dict = loss_fn(
            bz_mean, bz_logvar, severity_logits,
            bz_target, severity_target
        )
        print(f"  Total loss: {loss_dict['total_loss']:.4f}")
        print(f"  Bz loss: {loss_dict['bz_loss']:.4f}")
        print(f"  Severity loss: {loss_dict['severity_loss']:.4f}")

        # Test backward pass
        print("\nBackward Pass Test:")
        total_loss.backward()
        print("  Gradients computed successfully!")

        print("\nModel tests completed!")
