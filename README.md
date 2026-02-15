# HELIOS MVP - Space Weather CME Prediction System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pawelmical/helios-space-weather/blob/main/notebooks/HELIOS_Colab_Demo.ipynb)

## Overview

HELIOS MVP is a comprehensive space weather early warning system designed to detect coronal mass ejections (CMEs) and predict their radiation hazard severity for spacecraft crew safety. This system combines classical detection algorithms with machine learning for risk assessment.

**Key Features:**
- CME Detection using running-difference coronagraph analysis
- Stereoscopic Triangulation from L1/L4/L5 constellation
- Neural Network Bz prediction (PyTorch dual-head architecture)
- Radiation dosimetry calculations
- Triple Modular Redundancy (TMR) voting for reliability

## Quick Start

### Option 1: Google Colab (Recommended - No Setup Required)

Click the "Open in Colab" badge above, or open `notebooks/HELIOS_Colab_Demo.ipynb` in Google Colab.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/pawelmical/helios-space-weather.git
cd helios-space-weather

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the complete MVP demo
python scripts/run_complete_mvp.py
```

## Project Structure

```
helios-space-weather/
├── README.md                      # This file
├── requirements.txt               # Python dependencies (includes PyTorch)
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore file
│
├── helios_code/                   # Core detection & triangulation modules
│   ├── __init__.py
│   ├── detection.py               # CME detection algorithm
│   ├── triangulation.py           # Stereoscopic triangulation
│   ├── ensemble_propagation.py    # CME trajectory modeling
│   ├── evaluate.py                # Performance metrics
│   ├── utils.py                   # Utilities & constants
│   └── geometry_verification.py   # Geometry validation
│
├── NeuralNetwork_ML/              # Machine learning pipeline
│   ├── __init__.py
│   ├── config.py                  # ML configuration
│   ├── model.py                   # Dual-head NN architecture
│   ├── features.py                # Feature engineering
│   ├── severity.py                # Severity classification
│   ├── tmr_voting.py              # Triple modular redundancy
│   ├── warning_generator.py       # Crew warning system
│   ├── train.py                   # Model training
│   ├── preprocessing.py           # Data preprocessing
│   ├── dataset_generator.py       # Dataset generation
│   └── validation.py              # Model validation
│
├── scripts/
│   ├── run_complete_mvp.py        # Main MVP demo
│   ├── run_final_validation.py    # Train model + generate whitepaper metrics (v2 pipeline)
│   ├── run_historical_validation.py # (legacy - replaced by run_final_validation.py)
│   └── test_triangulation_constraint.py # Triangulation verification
│
├── notebooks/
│   └── HELIOS_Colab_Demo.ipynb    # Google Colab demo
│
├── data/
│   ├── events_list.csv            # CME event catalog
│   └── bastille_goes8_data.json   # Bastille Day 2000 data
│
├── output/
│   ├── helios_final_model_proper.pth  # Trained model
│   ├── final_validation_results.json  # Gold standard metrics
│   ├── mvp_results/                   # Validation run outputs
│   ├── geometry_verification.csv      # Geometry validation
│   ├── coverage_analysis.csv          # Coverage analysis
│   ├── timing_advantage.csv           # Timing metrics
│   └── ...                            # Additional CSV outputs
│
└── docs/                          # Technical documentation
    ├── GEOMETRY_UNIFIED.md        # Constellation geometry
    ├── TRIANGULATION_GUIDE.md     # Algorithm guide
    └── METRIC_CALCULATION_TRACE.md # Metric derivation
```

## Usage Examples

### Run Bastille Day 2000 Demo

```python
from helios_code.ensemble_propagation import run_ensemble

# Bastille Day CME parameters
ensemble = run_ensemble(initial_speed=1674, n_members=100)

print(f"Predicted arrival: {ensemble.arrival_median_hours:.1f} hours")
print(f"68% confidence: [{ensemble.arrival_16_hours:.1f}, {ensemble.arrival_84_hours:.1f}] hours")
print(f"Actual arrival: 28.5 hours")
```

### Neural Network Inference

```python
import torch
from NeuralNetwork_ML.features import create_bastille_day_features

# Load trained model
checkpoint = torch.load('output/helios_final_model_proper.pth', map_location='cpu')
print(f"Loaded {len(checkpoint['ensemble_states'])} ensemble models")

# Create feature vector
features = create_bastille_day_features()
print(f"16D feature vector created")
```

### Monte-Carlo Triangulation

```python
from helios_code.triangulation import montecarlo_triangulation
import numpy as np

# Run Monte-Carlo with 0.5° angular uncertainty
result = montecarlo_triangulation(
    r1=observer1_pos, u1=los1,
    r2=observer2_pos, u2=los2,
    sigma_deg=0.5, n_samples=1000
)
print(f"Spatial resolution: {result.delta_r_km/1e6:.2f} million km")
```

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Detection Confidence | 93.0% | CME detection rate |
| False Positive Rate | 5.0% | False alarm rate |
| Bz MAE | 6.5 nT | Mean absolute error |
| Hazard Accuracy | 83.3% | Severity classification |
| Bastille Day Error | 4.6 nT | Showcase event accuracy |

## Model Architecture

The dual-head neural network:
- **Input**: 16-dimensional feature vector
- **Encoder**: [128, 256, 128, 64] layers with LayerNorm + GELU
- **Bz Head**: Heteroscedastic regression (mean + variance)
- **Severity Head**: 4-class classification (Low, Moderate, High, Extreme)

## Citation

If using this framework, please cite:

```
Paweł Micał (2026). HELIOS: Space Weather CME Prediction System.
Space Weather Early Warning System Development.
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

Paweł Micał - pawelmical@icloud.com
