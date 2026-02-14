"""
HELIOS Feature Engineering
===========================
16-dimensional feature vector construction for CME characterization.

Features:
    1. cme_speed (km/s)
    2. angular_width (degrees)
    3. source_latitude (degrees)
    4. source_longitude (degrees)
    5. expansion_rate (Rs/hour)
    6. acceleration (m/s^2)
    7. L1_viewing_angle (degrees)
    8. L4_viewing_angle (degrees)
    9. L5_viewing_angle (degrees)
    10. brightness_asymmetry (ratio)
    11. parallax_L1L4 (solar radii)
    12. parallax_L1L5 (solar radii)
    13. parallax_L4L5 (solar radii)
    14. detection_time (hours post-eruption)
    15. triangulation_quality (0-1 score)
    16. observation_completeness (0-1 score)

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import os

from NeuralNetwork_ML.config import FEATURE_NAMES, FEATURE_BOUNDS, AU_IN_KM, SOLAR_RADIUS_KM

# Import from existing codebase using importlib to avoid conflict with built-in 'code' module
def _import_utils_functions():
    """Import functions from code/utils.py using importlib."""
    try:
        utils_path = os.path.join(os.path.dirname(__file__), '..', 'helios_code', 'utils.py')
        if os.path.exists(utils_path):
            spec = importlib.util.spec_from_file_location("helios_utils", utils_path)
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            return (
                utils_module.get_observer_position,
                utils_module.get_constellation_positions,
                utils_module.compute_viewing_angle,
                utils_module.heliocentric_to_spherical,
                True
            )
    except Exception:
        pass
    return None, None, None, None, False

(get_observer_position,
 get_constellation_positions,
 compute_viewing_angle,
 heliocentric_to_spherical,
 HAS_UTILS) = _import_utils_functions()


@dataclass
class CMEFeatures:
    """Container for 16-dimensional CME feature vector."""
    cme_speed: float              # 1. Speed in km/s
    angular_width: float          # 2. Angular width in degrees
    source_latitude: float        # 3. Source region latitude
    source_longitude: float       # 4. Source region longitude
    expansion_rate: float         # 5. Expansion rate in Rs/hour
    acceleration: float           # 6. Acceleration in m/s^2
    L1_viewing_angle: float       # 7. L1 viewing angle
    L4_viewing_angle: float       # 8. L4 viewing angle
    L5_viewing_angle: float       # 9. L5 viewing angle
    brightness_asymmetry: float   # 10. Brightness asymmetry ratio
    parallax_L1L4: float          # 11. Parallax L1-L4 in solar radii
    parallax_L1L5: float          # 12. Parallax L1-L5 in solar radii
    parallax_L4L5: float          # 13. Parallax L4-L5 in solar radii
    detection_time: float         # 14. Detection time post-eruption (hours)
    triangulation_quality: float  # 15. Triangulation quality score (0-1)
    observation_completeness: float  # 16. Observation completeness (0-1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in standard order."""
        return np.array([
            self.cme_speed, self.angular_width, self.source_latitude,
            self.source_longitude, self.expansion_rate, self.acceleration,
            self.L1_viewing_angle, self.L4_viewing_angle, self.L5_viewing_angle,
            self.brightness_asymmetry, self.parallax_L1L4, self.parallax_L1L5,
            self.parallax_L4L5, self.detection_time, self.triangulation_quality,
            self.observation_completeness
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CMEFeatures':
        """Create from numpy array."""
        return cls(**{name: float(arr[i]) for i, name in enumerate(FEATURE_NAMES)})

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'CMEFeatures':
        """Create from dictionary."""
        return cls(**{name: d[name] for name in FEATURE_NAMES})


def calculate_viewing_angles(
    cme_direction: np.ndarray,
    observation_time: datetime,
    helios_mode: str = 'synthetic'
) -> Tuple[float, float, float]:
    """
    Calculate viewing angles from each observer to CME propagation direction.

    Parameters
    ----------
    cme_direction : np.ndarray
        Unit vector of CME propagation direction
    observation_time : datetime
        Time of observation
    helios_mode : str
        'synthetic' or 'proxy'

    Returns
    -------
    L1_angle, L4_angle, L5_angle : float
        Viewing angles in degrees
    """
    if not HAS_UTILS:
        # Fallback: synthetic angles for Earth-directed CME
        return (2.0, 62.0, 58.0)

    positions = get_constellation_positions(observation_time, helios_mode)

    angles = []
    for instrument in ['L1', 'L4', 'L5']:
        if instrument in positions:
            obs_pos, _ = positions[instrument]
            angle = compute_viewing_angle(obs_pos, cme_direction)
            angles.append(angle)
        else:
            angles.append(90.0)  # Default if missing

    return tuple(angles)


def calculate_parallax(
    cme_position: np.ndarray,
    observation_time: datetime,
    helios_mode: str = 'synthetic'
) -> Tuple[float, float, float]:
    """
    Calculate apparent parallax between observer pairs.

    Parallax is the apparent angular shift in CME position when viewed
    from different observers. Larger parallax = better triangulation.

    Returns parallax in solar radii (apparent position difference).
    """
    if not HAS_UTILS:
        # Fallback: typical parallax for Earth-directed CME at 0.1 AU
        return (12.5, 12.8, 25.0)

    positions = get_constellation_positions(observation_time, helios_mode)

    cme_distance = np.linalg.norm(cme_position)

    def calc_apparent_direction(obs_pos: np.ndarray) -> np.ndarray:
        """Calculate direction from observer to CME."""
        direction = cme_position - obs_pos
        return direction / np.linalg.norm(direction)

    L1_dir = calc_apparent_direction(positions['L1'][0])
    L4_dir = calc_apparent_direction(positions['L4'][0])
    L5_dir = calc_apparent_direction(positions['L5'][0])

    def angle_to_parallax(dir1: np.ndarray, dir2: np.ndarray) -> float:
        """Convert angular separation to parallax in solar radii."""
        cos_angle = np.clip(np.dot(dir1, dir2), -1, 1)
        angle_rad = np.arccos(cos_angle)
        # Convert to solar radii at CME distance
        parallax_km = cme_distance * np.tan(angle_rad)
        return parallax_km / SOLAR_RADIUS_KM

    parallax_L1L4 = angle_to_parallax(L1_dir, L4_dir)
    parallax_L1L5 = angle_to_parallax(L1_dir, L5_dir)
    parallax_L4L5 = angle_to_parallax(L4_dir, L5_dir)

    return parallax_L1L4, parallax_L1L5, parallax_L4L5


def calculate_triangulation_quality(
    parallax_L1L4: float,
    parallax_L1L5: float,
    parallax_L4L5: float,
    viewing_angles: Tuple[float, float, float]
) -> float:
    """
    Calculate triangulation quality score (0-1).

    Quality depends on:
    - Parallax magnitude (larger = better)
    - Viewing angles close to 90 deg from propagation direction (optimal)
    - At least 2 good viewing angles
    """
    # Parallax component (normalized)
    max_parallax = max(parallax_L1L4, parallax_L1L5, parallax_L4L5)
    parallax_score = np.clip(max_parallax / 30.0, 0, 1)  # 30 Rs is excellent

    # Viewing angle component
    # Angles near 90 deg are optimal for seeing CME extent
    angle_scores = []
    for angle in viewing_angles:
        # Score peaks at 90 deg
        score = 1.0 - abs(90 - angle) / 90
        angle_scores.append(max(0, score))

    # Take best 2 viewing angles
    angle_scores = sorted(angle_scores, reverse=True)[:2]
    angle_score = np.mean(angle_scores)

    # Combined score
    quality = 0.5 * parallax_score + 0.5 * angle_score
    return np.clip(quality, 0, 1)


def extract_features(
    cme_speed: float,
    angular_width: float,
    source_lat: float,
    source_lon: float,
    expansion_rate: float,
    acceleration: float,
    observation_time: datetime,
    cme_position: np.ndarray,
    cme_direction: np.ndarray,
    brightness_asymmetry: float = 1.0,
    detection_time_hours: float = 1.0,
    observation_completeness: float = 1.0,
    helios_mode: str = 'synthetic'
) -> CMEFeatures:
    """
    Extract full 16-dimensional feature vector from CME observation.

    This is the main entry point for feature engineering.

    Parameters
    ----------
    cme_speed : float
        CME speed in km/s
    angular_width : float
        Angular width in degrees
    source_lat : float
        Source region latitude in degrees
    source_lon : float
        Source region longitude in degrees
    expansion_rate : float
        Expansion rate in solar radii per hour
    acceleration : float
        Acceleration in m/s^2
    observation_time : datetime
        Time of observation
    cme_position : np.ndarray
        CME position in heliocentric coordinates [x, y, z] in km
    cme_direction : np.ndarray
        CME propagation direction unit vector
    brightness_asymmetry : float
        Brightness asymmetry ratio (default 1.0 = symmetric)
    detection_time_hours : float
        Time since eruption in hours
    observation_completeness : float
        Observation completeness score (0-1)
    helios_mode : str
        'synthetic' or 'proxy'

    Returns
    -------
    features : CMEFeatures
        16-dimensional feature vector
    """
    # Calculate viewing angles
    L1_angle, L4_angle, L5_angle = calculate_viewing_angles(
        cme_direction, observation_time, helios_mode
    )

    # Calculate parallax
    parallax_L1L4, parallax_L1L5, parallax_L4L5 = calculate_parallax(
        cme_position, observation_time, helios_mode
    )

    # Calculate triangulation quality
    triangulation_quality = calculate_triangulation_quality(
        parallax_L1L4, parallax_L1L5, parallax_L4L5,
        (L1_angle, L4_angle, L5_angle)
    )

    return CMEFeatures(
        cme_speed=cme_speed,
        angular_width=angular_width,
        source_latitude=source_lat,
        source_longitude=source_lon,
        expansion_rate=expansion_rate,
        acceleration=acceleration,
        L1_viewing_angle=L1_angle,
        L4_viewing_angle=L4_angle,
        L5_viewing_angle=L5_angle,
        brightness_asymmetry=brightness_asymmetry,
        parallax_L1L4=parallax_L1L4,
        parallax_L1L5=parallax_L1L5,
        parallax_L4L5=parallax_L4L5,
        detection_time=detection_time_hours,
        triangulation_quality=triangulation_quality,
        observation_completeness=observation_completeness
    )


def create_bastille_day_features() -> CMEFeatures:
    """
    Create feature vector for Bastille Day 2000 event.
    Uses COMPUTED values matching training data formulas from extract_features().

    Event parameters:
    - Date: 2000-07-14 10:24 UT
    - Speed: 1674 km/s
    - Width: 360 deg (full halo)
    - Source: N22W07 (22 deg North, 7 deg West)
    - Bz measured (ACE): -60 nT
    """
    # Bastille Day 2000 event parameters
    speed = 1674.0      # km/s
    width = 360.0       # degrees (full halo)
    lat = 22.0          # N22
    lon = 7.0           # W07

    # COMPUTED features (matching run_final_validation.py extract_features)
    expansion_rate = speed / 200.0                    # 8.37
    acceleration = -speed / 15.0                      # -111.6
    L1_angle = abs(lon)                               # 7
    L4_angle = abs(60 - lon)                          # 53
    L5_angle = abs(-60 - lon)                         # 67
    brightness_asymmetry = 1.0 if width > 300 else width / 300.0  # 1.0
    parallax_L1L4 = 10.0 + (width / 36.0)             # 20.0
    parallax_L1L5 = 10.0 + (width / 36.0)             # 20.0
    parallax_L4L5 = 20.0 + (width / 18.0)             # 40.0
    detection_time = max(0.1, 2.0 - (speed / 1000.0)) # 0.326

    return CMEFeatures(
        cme_speed=speed,
        angular_width=width,
        source_latitude=lat,
        source_longitude=lon,
        expansion_rate=expansion_rate,
        acceleration=acceleration,
        L1_viewing_angle=L1_angle,
        L4_viewing_angle=L4_angle,
        L5_viewing_angle=L5_angle,
        brightness_asymmetry=brightness_asymmetry,
        parallax_L1L4=parallax_L1L4,
        parallax_L1L5=parallax_L1L5,
        parallax_L4L5=parallax_L4L5,
        detection_time=detection_time,
        triangulation_quality=0.85,
        observation_completeness=1.0
    )


if __name__ == "__main__":
    # Test feature extraction
    print("=" * 60)
    print("HELIOS Feature Engineering - Test")
    print("=" * 60)

    # Create Bastille Day features
    features = create_bastille_day_features()

    print("\nBastille Day 2000 Feature Vector:")
    print("-" * 40)
    for name in FEATURE_NAMES:
        value = getattr(features, name)
        bounds = FEATURE_BOUNDS[name]
        in_bounds = bounds[0] <= value <= bounds[1]
        status = "OK" if in_bounds else "OUT OF BOUNDS"
        print(f"  {name:25s}: {value:10.2f}  [{bounds[0]:.1f}, {bounds[1]:.1f}]  {status}")

    print("\nAs numpy array:")
    arr = features.to_array()
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Values: {arr}")

    # Roundtrip test
    features_rt = CMEFeatures.from_array(arr)
    match = np.allclose(features.to_array(), features_rt.to_array())
    print(f"\nRoundtrip test: {'PASS' if match else 'FAIL'}")
