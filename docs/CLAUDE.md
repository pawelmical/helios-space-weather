# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HELIOS MVP Validation Framework for space weather CME (Coronal Mass Ejection) detection, triangulation, and arrival prediction. Uses stereoscopic observation from L1+L4+L5 Lagrange points to improve Earth-directed CME tracking.

## Commands

```bash
# Setup environment
python -m venv helios_env
helios_env\Scripts\activate    # Windows
pip install -r requirements.txt

# Run full validation pipeline
python run_validation.py
python run_validation.py --helios-mode synthetic --n-events 10 --n-ensemble 100

# Run geometry verification suite
python run_geometry_verification.py
python run_geometry_verification.py --samples 500

# Run triangulation-constrained prediction test
python test_triangulation_constraint.py

# Test individual modules
python code/detection.py
python code/triangulation.py
python code/ensemble_propagation.py
python code/evaluate.py

# Run validation notebook
jupyter notebook notebooks/validation_run.ipynb
```

## Architecture

### Core Modules (code/)

- **detection.py**: CME detection using running-difference analysis on coronagraph images. Key class: `CMEDetector`
- **triangulation.py**: Stereoscopic 3D position estimation with Monte-Carlo uncertainty. Key functions: `triangulate_two_lines()`, `montecarlo_triangulation()`
- **ensemble_propagation.py**: Drag-based CME propagation with parameter uncertainty. Key functions: `run_ensemble()`, `triangulation_constrained_prediction()`, `calculate_cme_trajectory()`
- **evaluate.py**: Detection metrics (confusion matrix, ROC, AUC). Key function: `compute_confusion()`
- **utils.py**: Observer positions and coordinate utilities. Key: `get_observer_position()`, `AU_IN_KM`
- **geometry_verification.py**: Mathematical verification of constellation geometry (GDOP, coverage, timing)

### Constellation Geometry

The project uses L1+L4+L5 constellation with **L1+L4 as the optimal triangulation pair** for Earth-directed CMEs:

| Observer | Longitude | Distance | Role |
|----------|-----------|----------|------|
| L1 | 0° | 0.99 AU | Sun-Earth line imaging + triangulation baseline |
| L4 | +60° | 1.00 AU | Leading point - optimal 90° intersection with L1 |
| L5 | -60° | 1.00 AU | Trailing point - redundancy + far-side coverage |

L1+L4 achieves ~90° intersection angle (optimal GDOP), while L4+L5 is degenerate (~180°) for Earth-directed CMEs.

### Data Flow

1. Event catalog (`data/events_list.csv`) defines CME events with eruption times, speeds, and arrival times
2. Detection module processes coronagraph images (synthetic or real)
3. Triangulation module computes 3D CME positions from multi-viewpoint observations
4. Ensemble propagation predicts arrival times using drag-based physics model
5. Evaluation module compares L1-only vs HELIOS (multi-viewpoint) performance

### Key Constants

- `AU_IN_KM = 1.496e8` (km)
- Solar radius: `6.96e5` km
- Default angular precision: 0.5° (triangulation)
- Default ensemble size: 100-200 members

## Output Artifacts

Generated in `output/`:
- `results_validation.csv` - Per-event propagation results
- `detection_report.csv` - TP/FN/FP/TN metrics per instrument
- `triangulation_table.csv` - Monte-Carlo spatial resolution
- `geometry_verification.csv` - Constellation verification results
- `*_analysis.csv` - Detailed analysis reports
