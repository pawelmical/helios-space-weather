# Metric Provenance - How Each Number Was Generated

## Test Set Metrics
- **Source:** `run_final_validation.py` -> `evaluate_ensemble()` -> `final_validation_results.json`
- **MAE:** `np.abs(bz_ensemble - y_bz_t).mean()` = 6.29 nT (rounded: 6.3)
- **Std:** `np.abs(bz_ensemble - y_bz_t).std()` = 5.61 nT (rounded: 5.6)
- **Baseline:** `-0.1 * speed * sin(lat)` per test event, MAE = 9.85 nT (rounded: 9.9)
- **Improvement:** `(1 - 6.29/9.85) * 100` = 36.2% (rounded: 36%)
- **Severity accuracy:** 5/6 correct = 83.3%
- **Adjacent-or-correct:** 6/6 within +/-1 class = 100%

## Bastille Day Showcase
- **Source:** `run_final_validation.py` -> `evaluate_ensemble()` -> same JSON
- **Ensemble predictions:** L1=-55.18, L4=-56.04, L5=-53.33 (seeds 42,123,456)
- **Consensus Bz:** mean(-55.18, -56.04, -53.33) = -54.85 (rounded: -54.9)
- **Spread:** max - min = -53.33 - (-56.04) = 2.71 (rounded: 2.7)
- **Std:** std([-55.18, -56.04, -53.33]) = 1.11 (rounded: 1.1)
- **Error:** |(-54.85) - (-60.0)| = 5.15 (rounded: 5.2)
- **Relative error:** 5.15/60.0 = 8.6%
- **Confidence:** 91.6% (Gaussian CDF integration over Extreme threshold)

## Dose Calculation
- **Formula:** `D = 0.0132 * |Bz|^1.3 * sqrt(speed) * exposure_hours`
- **Inputs:** Bz = 54.85, speed = 1674 km/s, exposure = 10 hours
- **Calculation:** 0.0132 * 54.85^1.3 * sqrt(1674) * 10 = 984.9 mSv (rounded: 985)
- **NASA ratio:** 985 / 250 = 3.94 = 394%

## Detection Metrics (Future Work)
- **Detection Confidence:** NULL - requires CNN + coronagraph imagery pipeline
- **False Positive Rate:** NULL - requires operational detection system

These metrics were placeholder values in earlier versions (93%, 5%) based on literature estimates.
The current validation pipeline does not include a detection component, so these are marked
as future work items for the operational system.

## Data Sources

### Test Set Events (6 events)
| Event | Date | True Bz | Predicted Bz | Error |
|-------|------|---------|--------------|-------|
| Halloween Storm 2 | 2003-10-29 | -49.0 nT | -55.23 nT | 6.23 nT |
| September 2017 | 2017-09-06 | -32.0 nT | -31.63 nT | 0.37 nT |
| January 2005 | 2005-01-17 | -25.0 nT | -43.07 nT | 18.07 nT |
| November 2001 | 2001-11-24 | -30.0 nT | -35.07 nT | 5.07 nT |
| May 2024 | 2024-05-11 | -50.0 nT | -55.20 nT | 5.20 nT |
| October 2024 | 2024-10-10 | -38.0 nT | -35.23 nT | 2.77 nT |

### Showcase Event
| Event | Date | True Bz | Predicted Bz | Error |
|-------|------|---------|--------------|-------|
| Bastille Day 2000 | 2000-07-14 | -60.0 nT | -54.85 nT | 5.15 nT |

## Reproducibility

All metrics can be reproduced by running:
```bash
python scripts/run_final_validation.py
```

This will regenerate `output/final_validation_results.json` with identical values
(assuming same random seeds: 42, 123, 456).
