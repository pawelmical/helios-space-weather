"""
HELIOS Validation Code Package
==============================
CME detection, triangulation, propagation, and evaluation modules.
"""

from .detection import CMEDetector, DetectionResult, generate_synthetic_cme_images
from .triangulation import (
    triangulate_two_lines,
    estimate_point_from_observations,
    montecarlo_triangulation,
    TriangulationResult,
    MonteCarloResult
)
from .ensemble_propagation import (
    calculate_cme_trajectory,
    run_ensemble,
    propagate_event,
    EnsembleResult
)
from .evaluate import (
    compute_confusion,
    compute_roc_curve,
    ConfusionMetrics,
    print_performance_summary
)
from .utils import (
    get_observer_position,
    get_constellation_positions,
    AU_IN_KM
)

__version__ = "1.0.0"
__author__ = "HELIOS Team"
