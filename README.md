# HELIOS — Autonomous CME Early Warning System for Deep-Space Operations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pawelmical/helios-space-weather/blob/main/notebooks/HELIOS_Colab_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

HELIOS is a distributed space weather monitoring architecture delivering autonomous CME warnings **16–50 hours before geomagnetic impact** — protecting satellites, power grids, and deep-space crews from coronal mass ejections.

A Bastille Day 2000 CME (Bz = −60 nT, X5.7 flare, 1674 km/s) would deliver **985 mSv** of unshielded radiation to a deep-space crew — **394% of NASA's single-event SPE limit**. HELIOS predicted this event with 5.2 nT MAE and unanimous Extreme severity classification. Current operational systems provide 15–60 minutes of warning. HELIOS provides 16–50 hours.

> Research prototype · Paweł Micał · MIT License

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Bz MAE** | **6.3 ± 5.6 nT** | 6-event independent test set |
| **Severity Accuracy** | **83.3%** | 4-class: Low / Moderate / High / Extreme |
| **Bastille Day 2000 MAE** | **5.2 nT (8.6%)** | Held-out extreme event, ground truth: ACE −60 nT |
| **Consensus Bz** | −54.9 ± 1.1 nT | 3-model ensemble, 2.7 nT inter-model spread |
| **Predicted Dose** | 985 mSv | 394% NASA SPE limit (NASA-STD-3001 Vol.1 Rev.C) |
| **Warning Window** | 16–50 hours | Velocity-dependent: 800–2500 km/s CME range |
| **Detection Advantage** | +3.5–10.9 hours | vs single-point L1 observation |
| **Heliospheric Coverage** | 83.5% | 301° of 360°, zero blind spots on Earth-threat hemisphere |
| **Improvement vs Baseline** | 36% | Over physics-based geometric estimate |

---
## Why This Matters

While predictive models like NOAA's WSA-ENLIL can estimate Coronal Mass Ejection (CME) arrivals days in advance, they carry a high margin of error ($\pm$ 10-12 hours) and cannot reliably predict the storm's exact magnetic orientation (severity). Absolute, actionable confirmation of a CME's impact — provided by L1 monitors like DSCOVR and ACE — delivers only **15–60 minutes** of definitive warning. 

That tight window is operationally insufficient for:

- **Deep-space crews** requiring hours to terminate EVAs and reach shelter.
- **Satellite operators** needing time to transition critical hardware to safe mode.
- **Power grid operators** managing pre-emptive load shedding before intense geomagnetic storm onset.

Historical precedent highlights this vulnerability. The August 1972 solar particle event — occurring between Apollo 16 and Apollo 17 — would have been lethal to any crew in transit. The Bastille Day 2000 event would have critically exceeded safe exposure limits for EVAs. Even during the recent Gannon Storm (May 2024, G5), while operators had roughly 17 hours from initial CME detection to peak impact, the exact severity remained uncertain until the plasma physically crossed the L1 point..

HELIOS addresses this by distributing observation across three Lagrange point nodes — L1, L4, L5 — enabling stereoscopic CME characterization **hours before plasma arrival**, not minutes.

---

## Architecture

```
         L4 (60° ahead)
          ●
         / \
        /   \   ← 120° stereoscopic baseline
       /     \
  Sun ●  ─── ● L1 (upstream) ──→ Earth ○
       \     /
        \   /
         \ /
          ●
         L5 (60° behind)
```

**Three-node constellation.** L1 provides on-axis in-situ validation and upstream solar wind measurement. L4 and L5 provide off-axis coronagraph perspectives enabling stereoscopic 3D CME reconstruction — detecting Earth-directed events **hours earlier** than any single-point system.

**AI provides probabilistic inference only.** All crew-facing warnings execute through deterministic TMR-voted rule-based logic. No machine learning in the safety-critical decision path.

| Layer | Function |
|-------|----------|
| Detection | Running-difference coronagraph analysis, CME onset identification |
| Triangulation | Monte Carlo stereoscopic reconstruction, 1.10 Mkm spatial resolution |
| ML Inference | Dual-head neural network — Bz regression + severity classification |
| Safety Layer | Triple Modular Redundancy voting — 2/3 majority required for warning |
| Output | Crew-actionable alert: severity class, predicted dose (mSv), response timeline |

**Neural network:** 16D feature input → [128, 256, 128, 64] shared encoder (LayerNorm + GELU) → dual heads: heteroscedastic Bz regression (mean + variance) + 4-class severity classification.

---

## Try It

**One click — no setup required:**

Click the Colab badge above. Runs the full Bastille Day 2000 TMR pipeline in ~5 seconds.

**Expected output:**
```
Consensus Bz:       -54.9 ± 1.1 nT   (ground truth: -60.0 nT)
MAE:                5.2 nT  (8.6%)
Severity:           EXTREME  [3/3 unanimous]
Predicted dose:     985 mSv  (394% NASA SPE limit)
Warning lead time:  ~27 hours
```

**Local installation:**
```bash
git clone https://github.com/pawelmical/helios-space-weather.git
cd helios-space-weather
pip install -r requirements.txt
python scripts/run_complete_mvp.py   # ~5 seconds, uses pre-trained model
```

> To retrain from scratch (30–60 min, GPU recommended): `python scripts/run_final_validation.py`

---

## Performance Envelope

Warning lead time by CME velocity — time available for operational decision-making before Earth impact:

| CME Speed | Warning Window | Triangulation Window | L1 Transit |
|-----------|---------------|---------------------|------------|
| 800 km/s (slow) | **~50 hours** | 1.2–15.7 h | 51.4 h |
| 1500 km/s (moderate) | **~27 hours** | 0.6–8.4 h | 27.4 h |
| 2500 km/s (fast) | **~16 hours** | 0.4–5.0 h | 16.5 h |

System latency from coronagraph acquisition to Earth ground receipt: **1–10 minutes** (L1 node: ~1–2 min; L4/L5 nodes: ~9–10 min including light-propagation delay).

---

## Radiation Severity Classification

Autonomous hazard assessment maps predicted dose to four operational categories (NASA-STD-3001, Vol. 1, Rev. C):

| Severity | Dose Range | NASA SPE Limit | Crew Response |
|----------|-----------|----------------|---------------|
| Low | 10–50 mSv | 4–20% | Enhanced monitoring, continue operations |
| Moderate | 50–100 mSv | 20–40% | Shelter advisory |
| High | 100–250 mSv | 40–100% | Mandatory shelter, suspend EVA |
| **Extreme** | **>250 mSv** | **>100%** | **EVA abort, emergency shielding** |

Bastille Day 2000: 985 mSv predicted → **Extreme**, unanimous 3/3 TMR consensus.

---

## Repository Structure

```
helios-space-weather/
├── helios_code/          # Detection, triangulation, propagation modules
├── NeuralNetwork_ML/     # Dual-head NN, TMR voting, warning generator
├── helios_orbits/        # CR3BP orbital mechanics, halo orbit visualizations
├── scripts/              # run_complete_mvp.py · run_final_validation.py
├── notebooks/            # HELIOS_Colab_Demo.ipynb
├── data/                 # CME event catalog, Bastille Day ground truth
├── output/               # Trained model, validation results, coverage analysis
├── docs/                 # Geometry, triangulation, metric derivation guides
└── technical_documentation/  # Whitepaper · Appendices · Pitch deck
```

→ Full architecture documentation: [docs/](docs/)  
→ Orbital mechanics derivation: [docs/GEOMETRY_UNIFIED.md](docs/GEOMETRY_UNIFIED.md)  
→ Metric provenance: [technical_documentation/METRIC_PROVENANCE.md](technical_documentation/METRIC_PROVENANCE.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [Technical Whitepaper](technical_documentation/whitepaper/) | Full system architecture, orbital mechanics, AI pipeline, validation |
| [Appendix A — Orbital Mechanics](technical_documentation/appendices/) | CR3BP halo orbit derivation, SEZ compliance, propellant budget |
| [Appendix B — AI/ML Pipeline](technical_documentation/appendices/) | Feature engineering, training methodology, hyperparameters |
| [Pitch Deck](technical_documentation/pitchdeck/) | System overview for non-technical stakeholders |

---

## License & Contact

MIT License — see [LICENSE](LICENSE)

**Paweł Micał**  
📧 pawelmical@icloud.com  
🔗 [LinkedIn](https://www.linkedin.com/in/paweł-micał-0483bb38a/)  
📄 [Technical Whitepaper](technical_documentation/whitepaper/)

---
*HELIOS is a research prototype. Operational deployment requires multi-event validation on space-qualified hardware and integration with mission-specific communication architectures.*