# HELIOS Technical Documentation

## Architecture Overview

HELIOS uses two parallel pipelines in this repository:

### Pipeline 1: Standalone Validation (`scripts/run_final_validation.py`)
- **Purpose:** Produces all metrics cited in the whitepaper
- **Self-contained:** Zero imports from `helios_code/` or `NeuralNetwork_ML/`
- **Training:** 12 historical events (50x augmented) + 10,000 synthetic = 10,600 samples
- **Test:** 6 completely unseen historical events
- **Showcase:** Bastille Day 2000 (excluded from all training)
- **Output:** `output/final_validation_results.json`

### Pipeline 2: Modular MVP (`scripts/run_complete_mvp.py`)
- **Purpose:** End-to-end TMR demonstration (3-satellite showcase)
- **Uses:** `helios_code/` for detection/triangulation, `NeuralNetwork_ML/` for model architecture
- **Loads:** Trained checkpoint from Pipeline 1
- **Output:** `output/mvp_results/`

### Why Two Pipelines?
Pipeline 1 is standalone for reproducibility - anyone can run one script and get the exact whitepaper metrics. Pipeline 2 demonstrates the full operational concept (detection -> triangulation -> inference -> TMR voting -> dosimetry -> crew warning).

## Validated Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Bz MAE (test set) | 6.3 +/- 5.6 nT | final_validation_results.json |
| Improvement vs geometric baseline | 36% | Dynamically computed |
| Severity accuracy | 83.3% (5/6) | 4-class classification |
| Adjacent-or-correct | 100% | All within +/-1 class |
| Bastille Day Bz prediction | -54.9 nT (error: 5.2 nT) | Ensemble consensus |
| Bastille Day confidence | 91.6% | Gaussian CDF integration |
| Inter-model spread | 2.7 nT | 3-seed ensemble |
| Predicted dose | 985 mSv (394% NASA limit) | Deterministic formula |

## Known MVP Limitations

1. **Synthetic Bz is uncorrelated with CME parameters** - `rng.uniform()` within severity class bounds. Causes regression-to-mean (~-55 nT) for deep-Extreme events. Future work: physics-based Bz generator.
2. **Detection confidence not measured** - Requires CNN + coronagraph imagery pipeline (operational system).
3. **Feature extraction simplified** - 16D deterministic features from 4 CME parameters. Operational system will use full geometric + in-situ data.
4. **Z-score normalization** - MVP uses StandardScaler. Operational system targets predefined [0,1] bounds for embedded hardware.

## File Reference

| File | Role |
|------|------|
| `scripts/run_final_validation.py` | **Authoritative** - produces all whitepaper metrics |
| `scripts/run_complete_mvp.py` | TMR showcase demonstration |
| `NeuralNetwork_ML/config.py` | Shared constants (thresholds, architecture) |
| `NeuralNetwork_ML/severity.py` | Dose-based severity (operational concept) |
| `NeuralNetwork_ML/model.py` | DualHeadBzModel architecture |
| `NeuralNetwork_ML/features.py` | Modular pipeline feature extraction |
| `NeuralNetwork_ML/dataset_generator.py` | Modular pipeline event catalog |
| `helios_code/detection.py` | Running-difference CME detection |
| `helios_code/triangulation.py` | Stereoscopic 3D reconstruction |
| `helios_code/ensemble_propagation.py` | Multi-satellite propagation |
| `helios_code/utils.py` | Geometry utilities |
| `output/final_validation_results.json` | **Ground truth** for all metrics |
| `output/mvp_results/` | TMR showcase output |

## Folder Structure

```
technical_documentation/
  README.md                 # This file
  METRIC_PROVENANCE.md      # Detailed metric derivation
  whitepaper/               # HELIOS whitepaper
  appendices/               # Technical appendices
  pitchdeck/                # Investor presentation materials
```
