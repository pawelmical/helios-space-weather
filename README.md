# HELIOS MVP - Space Weather CME Prediction System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pawelmical/helios-space-weather/blob/main/notebooks/HELIOS_Colab_Demo.ipynb)

## Overview

HELIOS MVP is a conceptual space weather early warning system designed to detect coronal mass ejections (CMEs) and predict their radiation hazard severity for spacecraft crew safety. This system combines classical detection algorithms with machine learning for risk assessment. Note: This is a demonstration of the core algorithms and model architecture, not a fully operational system.

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

# Run the complete MVP demo (uses pre-trained model, ~5 seconds)
python scripts/run_complete_mvp.py
```

> **Note — two scripts, two purposes:**
> - `run_complete_mvp.py` — loads the committed pre-trained model and runs the full
>   Bastille Day 2000 TMR pipeline. This is what you want for the demo.
> - `run_final_validation.py` — retrains the entire 3-seed ensemble from scratch.
>   Only needed if you want to reproduce the training. Requires ~30–60 min and a
>   GPU is strongly recommended. Running this will overwrite `output/helios_final_model_proper.pth`.

## Project Structure

```
helios-space-weather/
├── README.md                      # This file
├── CHANGELOG.md                   # Version history
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
├── helios_orbits/                 # Orbital mechanics & mission design
│   ├── helios_orbital_mechanics.py    # CR3BP orbital propagation
│   ├── helios_orbital_params.txt      # Mission parameters summary
│   ├── helios_l1_halo_orbit.png       # L1 halo orbit visualization
│   ├── helios_l4_drift_orbit.png      # L4 drift orbit visualization
│   ├── helios_l5_drift_orbit.png      # L5 drift orbit visualization
│   └── helios_srp_comparison.png      # Solar radiation pressure effects
│
├── scripts/
│   ├── run_complete_mvp.py        # ⚡ DEMO: load saved model, run Bastille Day TMR pipeline (~5s)
│   ├── run_final_validation.py    # 🔁 RETRAIN: train 3-seed ensemble from scratch (GPU recommended, ~30-60 min)
│   └── test_triangulation_constraint.py # Triangulation geometry verification
│
├── notebooks/
│   └── HELIOS_Colab_Demo.ipynb    # Google Colab demo
│
├── data/
│   ├── events_list.csv            # CME event catalog
│   ├── bastille_goes8_data.json   # Bastille Day 2000 data
│   └── images/                    # Image assets
│       └── README.md              # Image folder documentation
│
├── output/
│   ├── helios_final_model_proper.pth  # Trained model checkpoint
│   ├── final_validation_results.json  # Gold standard metrics (authoritative)
│   ├── mvp_results/                   # TMR showcase outputs
│   ├── geometry_verification.csv      # Geometry validation results
│   ├── coverage_analysis.csv          # L1/L4/L5 sky-coverage analysis
│   ├── spatial_resolution_sweep.csv   # Triangulation resolution vs baseline
│   ├── timing_advantage.csv           # Detection timing vs L1-only baseline
│   ├── detection_windows_verification.csv  # SEZ detection windows
│   └── dose_validation_matrix.csv     # Dose calculation validation
│
├── docs/                          # Developer documentation
│   ├── GEOMETRY_UNIFIED.md        # Constellation geometry
│   ├── TRIANGULATION_GUIDE.md     # Algorithm guide
│   └── METRIC_CALCULATION_TRACE.md # Metric derivation
│
└── technical_documentation/       # Stakeholder documentation
    ├── README.md                  # Architecture overview
    ├── METRIC_PROVENANCE.md       # Metric derivation trace
    ├── whitepaper/                # HELIOS whitepaper (user-provided)
    ├── appendices/                # Technical appendices (user-provided)
    └── pitchdeck/                 # Investor presentation (user-provided)
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
| Severity Accuracy | 83.3% | 4-class classification rate |
| Improvement vs Baseline | 36% | Over geometric estimate |
| Bz MAE | 6.3 nT | Mean absolute error |
| Hazard Accuracy | 83.3% | Severity classification |
| Bastille Day Error | 5.2 nT | Showcase event accuracy |

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
