# CHANGELOG

## v2.0 (February 2026) - Current
- Fixed training/inference feature mismatch in features.py (triangulation_quality: 0.85 → computed)
- Proper train/test split: 12 train + 6 test + 1 showcase (Bastille Day excluded)
- 3-seed ensemble (42, 123, 456) with Bz-derived severity via Gaussian CDF
- Validated metrics: Bz MAE 6.5 nT, Bastille Day error 4.6 nT, severity accuracy 83.3%

## v1.0 (January 2026) - Deprecated
- Initial MVP with data leakage (all 19 events in training → artificially low 1.0 nT MAE)
- Replaced by v2.0 pipeline
