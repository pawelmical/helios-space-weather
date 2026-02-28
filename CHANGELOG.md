# CHANGELOG

## [1.1.0] - 2026-02-28

### Fixed
- Corrected all whitepaper metrics to match final_validation_results.json
- Fixed dataset_generator.py sync comment (independent pipeline documentation)
- Fixed los_z geometry bug in helios_code/utils.py
- Removed buggy mvp_results/ output (satellites identical, spread=0)
- Cleaned __pycache__ directories

### Added
- technical_documentation/ folder with architecture overview and metric provenance
- .gitignore entry for __pycache__/

### Changed
- README.md performance metrics updated to validated values
- All .md files unified to reference authoritative metrics
- Code comments updated to reflect current pipeline state

## v2.0 (February 2026)
- Fixed training/inference feature mismatch in features.py (triangulation_quality: 0.85 → computed)
- Proper train/test split: 12 train + 6 test + 1 showcase (Bastille Day excluded)
- 3-seed ensemble (42, 123, 456) with Bz-derived severity via Gaussian CDF
- Validated metrics: Bz MAE 6.3 nT, Bastille Day error 5.2 nT, severity accuracy 83.3%

## v1.0 (January 2026) - Deprecated
- Initial MVP with data leakage (all 19 events in training → artificially low 1.0 nT MAE)
- Replaced by v2.0 pipeline
