"""
HELIOS Dataset Generator
=========================
Generate synthetic CME events with physically-motivated Bz values.

Bz Generation Physics:
    Bz ~ f(speed, tilt_angle, helicity, width)

    Simplified model:
    Bz_base = -k * v^0.5 * sin(tilt) * (width/60)^0.3

    Where:
    - k: Scaling factor calibrated to historical events
    - v: CME speed (km/s)
    - tilt: Flux rope axis tilt angle
    - width: Angular width (degrees)

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from NeuralNetwork_ML.config import (
    DATASET_CONFIG, BZ_CONFIG, FEATURE_BOUNDS, AU_IN_KM, SOLAR_RADIUS_KM
)
from NeuralNetwork_ML.features import CMEFeatures, extract_features
from NeuralNetwork_ML.severity import bz_to_severity_class


@dataclass
class SyntheticEvent:
    """Container for synthetic CME event."""
    event_id: str
    features: CMEFeatures
    bz_true: float  # nT
    severity_class: int
    eruption_time: datetime
    metadata: Dict


# ============================================================================
# BZ PHYSICS MODEL
# ============================================================================

def generate_bz_from_physics(
    speed_km_s: float,
    angular_width_deg: float,
    tilt_angle_deg: float = None,
    helicity: int = None,
    seed: int = None
) -> float:
    """
    Generate Bz using flux rope physics model.

    The model captures the main dependencies:
    1. Speed: Faster CMEs have stronger compression -> stronger Bz
    2. Width: Wider CMEs have larger flux content
    3. Tilt angle: Determines how much of the azimuthal field projects southward
    4. Helicity: +1 or -1, determines initial field orientation

    Formula:
        Bz = -k * sqrt(v/1000) * sin(tilt) * (w/60)^0.3 * helicity
        + noise

    Calibrated to give:
        - Bastille Day (1674 km/s, halo): Bz ~ -60 nT
        - Typical fast CME (1200 km/s): Bz ~ -30 to -40 nT
        - Slow CME (500 km/s): Bz ~ -10 to -15 nT

    Parameters
    ----------
    speed_km_s : float
        CME speed in km/s
    angular_width_deg : float
        Angular width in degrees
    tilt_angle_deg : float, optional
        Flux rope tilt angle (random if not provided)
    helicity : int, optional
        Flux rope helicity +1 or -1 (random if not provided)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    bz : float
        Southward Bz component in nT (negative)
    """
    if seed is not None:
        np.random.seed(seed)

    # Random tilt angle if not provided (uniform -90 to 90)
    if tilt_angle_deg is None:
        tilt_angle_deg = np.random.uniform(-90, 90)

    # Random helicity if not provided
    if helicity is None:
        helicity = np.random.choice([-1, 1])

    # Calibration constant
    # Tuned so Bastille Day (1674 km/s, 360 deg) gives ~60 nT
    k = 35.0

    # Speed factor (normalized to 1000 km/s)
    speed_factor = np.sqrt(speed_km_s / 1000.0)

    # Width factor (normalized to 60 degrees)
    width_factor = (angular_width_deg / 60.0) ** 0.3

    # Tilt factor (sin gives southward projection)
    tilt_rad = np.deg2rad(tilt_angle_deg)
    tilt_factor = np.sin(tilt_rad)

    # Base Bz (always southward for geoeffective events)
    bz_base = -k * speed_factor * width_factor * abs(tilt_factor)

    # Apply helicity (can flip the sign in some configurations)
    # For simplicity, we ensure southward Bz for training
    if bz_base > 0:
        bz_base = -bz_base

    # Add noise
    noise = np.random.normal(0, BZ_CONFIG['noise_sigma'])
    bz = bz_base + noise

    # Clip to valid range
    bz = np.clip(bz, BZ_CONFIG['bz_min'], BZ_CONFIG['bz_max'])

    return bz


# ============================================================================
# SYNTHETIC EVENT GENERATION
# ============================================================================

def generate_synthetic_event(
    event_idx: int,
    base_time: datetime = None,
    seed: int = None
) -> SyntheticEvent:
    """
    Generate a single synthetic CME event with all features and labels.

    Parameters
    ----------
    event_idx : int
        Event index for ID generation
    base_time : datetime, optional
        Base time for event generation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    event : SyntheticEvent
        Complete synthetic event with features and labels
    """
    if seed is not None:
        np.random.seed(seed + event_idx)

    if base_time is None:
        base_time = datetime(2000, 1, 1)

    # Sample CME speed from log-normal distribution
    speed = np.random.lognormal(
        DATASET_CONFIG['speed_mu'],
        DATASET_CONFIG['speed_sigma']
    )
    speed = np.clip(speed, DATASET_CONFIG['speed_min'], DATASET_CONFIG['speed_max'])

    # Sample angular width
    width = np.random.normal(
        DATASET_CONFIG['width_mean'],
        DATASET_CONFIG['width_std']
    )
    width = np.clip(width, DATASET_CONFIG['width_min'], DATASET_CONFIG['width_max'])

    # Source location (biased toward active region belt)
    source_lat = np.random.normal(0, 15)  # +/-15 deg around equator
    source_lat = np.clip(source_lat, -45, 45)
    source_lon = np.random.uniform(-60, 60)  # Earth-visible disk

    # Expansion rate (correlated with speed)
    expansion_rate = 0.5 + (speed / 1000) * np.random.uniform(0.5, 1.5)
    expansion_rate = np.clip(expansion_rate, 0.1, 5.0)

    # Acceleration (fast CMEs decelerate, slow accelerate)
    if speed > 800:
        acceleration = np.random.uniform(-300, -50)  # Deceleration
    else:
        acceleration = np.random.uniform(0, 200)  # Acceleration

    # Detection time (faster CMEs detected earlier)
    detection_time = np.random.uniform(0.5, 3.0) * (1000 / speed)
    detection_time = np.clip(detection_time, 0.5, 12.0)

    # Brightness asymmetry
    brightness_asymmetry = np.random.lognormal(0, 0.3)
    brightness_asymmetry = np.clip(brightness_asymmetry, 0.3, 5.0)

    # Observation completeness (usually high for Earth-directed)
    observation_completeness = np.random.beta(8, 2)  # Biased toward 1.0

    # Generate observation time
    eruption_time = base_time + timedelta(days=event_idx)
    observation_time = eruption_time + timedelta(hours=detection_time)

    # CME position and direction (simplified for training)
    # Earth-directed CME at detection time
    distance_au = 0.1 + detection_time * speed / (AU_IN_KM / 3600)
    distance_au = min(distance_au, 0.3)  # Cap at 0.3 AU

    cme_position = np.array([distance_au * AU_IN_KM, 0, 0])
    cme_direction = np.array([1.0, 0, 0])  # Earth-directed

    # Extract features
    features = extract_features(
        cme_speed=speed,
        angular_width=width,
        source_lat=source_lat,
        source_lon=source_lon,
        expansion_rate=expansion_rate,
        acceleration=acceleration,
        observation_time=observation_time,
        cme_position=cme_position,
        cme_direction=cme_direction,
        brightness_asymmetry=brightness_asymmetry,
        detection_time_hours=detection_time,
        observation_completeness=observation_completeness,
        helios_mode='synthetic'
    )

    # Generate Bz using physics model
    tilt_angle = np.random.uniform(-90, 90)
    bz_true = generate_bz_from_physics(
        speed_km_s=speed,
        angular_width_deg=width,
        tilt_angle_deg=tilt_angle
    )

    # Get severity class (dose-based: requires both Bz and speed)
    severity_class, _ = bz_to_severity_class(bz_true, speed)

    return SyntheticEvent(
        event_id=f"synthetic_{event_idx:05d}",
        features=features,
        bz_true=bz_true,
        severity_class=severity_class,
        eruption_time=eruption_time,
        metadata={
            'tilt_angle': tilt_angle,
            'type': 'synthetic'
        }
    )


def generate_extreme_event(
    event_idx: int,
    base_time: datetime = None,
    seed: int = None
) -> SyntheticEvent:
    """
    Generate an EXTREME severity synthetic event.

    Forces parameters that produce Bz < -50 nT (Extreme class).

    Parameters
    ----------
    event_idx : int
        Event index for ID generation
    base_time : datetime, optional
        Base time for event generation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    event : SyntheticEvent
        Extreme severity synthetic event
    """
    if seed is not None:
        np.random.seed(seed + event_idx)

    if base_time is None:
        base_time = datetime(2000, 1, 1)

    # Force EXTREME event parameters:
    # High speed (1400-3500 km/s)
    speed = np.random.uniform(1400, 3500)

    # Wide CME (mostly halos, 200-360 deg)
    width = np.random.uniform(200, 360)

    # Source near disk center (geoeffective)
    source_lat = np.random.normal(0, 20)
    source_lat = np.clip(source_lat, -45, 45)
    source_lon = np.random.uniform(-40, 40)

    # Expansion rate (high for fast CMEs)
    expansion_rate = 1.5 + (speed / 1000) * np.random.uniform(0.8, 1.5)
    expansion_rate = np.clip(expansion_rate, 1.0, 5.0)

    # Deceleration (fast CMEs decelerate)
    acceleration = np.random.uniform(-400, -100)

    # Early detection
    detection_time = np.random.uniform(0.3, 1.5) * (1000 / speed)
    detection_time = np.clip(detection_time, 0.3, 4.0)

    # Brightness asymmetry
    brightness_asymmetry = np.random.lognormal(0.2, 0.3)
    brightness_asymmetry = np.clip(brightness_asymmetry, 0.5, 5.0)

    # High observation completeness
    observation_completeness = np.random.beta(12, 1)

    eruption_time = base_time + timedelta(days=event_idx)
    observation_time = eruption_time + timedelta(hours=detection_time)

    distance_au = 0.1 + detection_time * speed / (AU_IN_KM / 3600)
    distance_au = min(distance_au, 0.3)

    cme_position = np.array([distance_au * AU_IN_KM, 0, 0])
    cme_direction = np.array([1.0, 0, 0])

    features = extract_features(
        cme_speed=speed,
        angular_width=width,
        source_lat=source_lat,
        source_lon=source_lon,
        expansion_rate=expansion_rate,
        acceleration=acceleration,
        observation_time=observation_time,
        cme_position=cme_position,
        cme_direction=cme_direction,
        brightness_asymmetry=brightness_asymmetry,
        detection_time_hours=detection_time,
        observation_completeness=observation_completeness,
        helios_mode='synthetic'
    )

    # Force high tilt angle for extreme Bz
    tilt_angle = np.random.choice([-1, 1]) * np.random.uniform(60, 90)
    bz_true = generate_bz_from_physics(
        speed_km_s=speed,
        angular_width_deg=width,
        tilt_angle_deg=tilt_angle
    )

    # Ensure it's actually extreme (Bz < -50)
    # If not extreme enough, scale it
    if bz_true > -50:
        bz_true = np.random.uniform(-70, -50)

    severity_class = 3  # Extreme

    return SyntheticEvent(
        event_id=f"extreme_{event_idx:05d}",
        features=features,
        bz_true=bz_true,
        severity_class=severity_class,
        eruption_time=eruption_time,
        metadata={
            'tilt_angle': tilt_angle,
            'type': 'synthetic_extreme'
        }
    )


def generate_high_event(
    event_idx: int,
    base_time: datetime = None,
    seed: int = None
) -> SyntheticEvent:
    """
    Generate a HIGH severity synthetic event.

    Forces parameters that produce Bz between -35 and -50 nT (High class).
    """
    if seed is not None:
        np.random.seed(seed + event_idx)

    if base_time is None:
        base_time = datetime(2000, 1, 1)

    # Force HIGH event parameters:
    speed = np.random.uniform(1000, 2000)
    width = np.random.uniform(120, 300)

    source_lat = np.random.normal(0, 18)
    source_lat = np.clip(source_lat, -45, 45)
    source_lon = np.random.uniform(-50, 50)

    expansion_rate = 1.0 + (speed / 1000) * np.random.uniform(0.6, 1.2)
    expansion_rate = np.clip(expansion_rate, 0.5, 4.0)

    if speed > 1200:
        acceleration = np.random.uniform(-250, -50)
    else:
        acceleration = np.random.uniform(-100, 50)

    detection_time = np.random.uniform(0.5, 2.0) * (1000 / speed)
    detection_time = np.clip(detection_time, 0.4, 6.0)

    brightness_asymmetry = np.random.lognormal(0.1, 0.25)
    brightness_asymmetry = np.clip(brightness_asymmetry, 0.4, 4.0)

    observation_completeness = np.random.beta(10, 1.5)

    eruption_time = base_time + timedelta(days=event_idx)
    observation_time = eruption_time + timedelta(hours=detection_time)

    distance_au = 0.1 + detection_time * speed / (AU_IN_KM / 3600)
    distance_au = min(distance_au, 0.3)

    cme_position = np.array([distance_au * AU_IN_KM, 0, 0])
    cme_direction = np.array([1.0, 0, 0])

    features = extract_features(
        cme_speed=speed,
        angular_width=width,
        source_lat=source_lat,
        source_lon=source_lon,
        expansion_rate=expansion_rate,
        acceleration=acceleration,
        observation_time=observation_time,
        cme_position=cme_position,
        cme_direction=cme_direction,
        brightness_asymmetry=brightness_asymmetry,
        detection_time_hours=detection_time,
        observation_completeness=observation_completeness,
        helios_mode='synthetic'
    )

    tilt_angle = np.random.choice([-1, 1]) * np.random.uniform(45, 75)
    bz_true = generate_bz_from_physics(
        speed_km_s=speed,
        angular_width_deg=width,
        tilt_angle_deg=tilt_angle
    )

    # Ensure it's in High range (-35 to -50)
    if bz_true > -35 or bz_true <= -50:
        bz_true = np.random.uniform(-50, -35)

    severity_class = 2  # High

    return SyntheticEvent(
        event_id=f"high_{event_idx:05d}",
        features=features,
        bz_true=bz_true,
        severity_class=severity_class,
        eruption_time=eruption_time,
        metadata={
            'tilt_angle': tilt_angle,
            'type': 'synthetic_high'
        }
    )


def generate_synthetic_dataset(
    n_events: int = None,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate full synthetic dataset with STRATIFIED SAMPLING.

    Ensures sufficient representation of all severity classes,
    especially Extreme events which are naturally rare.

    Parameters
    ----------
    n_events : int, optional
        Number of events (default from config: 10000)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    features : np.ndarray
        Shape (n_events, 16)
    bz_values : np.ndarray
        Shape (n_events,)
    severity_classes : np.ndarray
        Shape (n_events,)
    event_ids : List[str]
    """
    n_events = n_events or DATASET_CONFIG['n_synthetic_events']
    seed = seed or 42

    np.random.seed(seed)

    features_list = []
    bz_list = []
    severity_list = []
    event_ids = []

    base_time = datetime(2000, 1, 1)

    # Target distribution: 35% Low, 30% Moderate, 20% High, 15% Extreme
    n_extreme = max(int(n_events * 0.15), DATASET_CONFIG.get('min_extreme_events', 500))
    n_high = int(n_events * 0.20)
    # NOTE: Bastille Day augmentation removed — injecting near-copies of the
    # sole validation target into training data constitutes data leakage and
    # inflates validation metrics. The model must generalise without seeing it.
    n_regular = n_events - n_extreme - n_high

    print(f"  Generating stratified dataset:")
    print(f"    Regular events: {n_regular}")
    print(f"    High events: {n_high}")
    print(f"    Extreme events: {n_extreme}")

    # 1. Generate regular events (natural distribution)
    for i in range(n_regular):
        event = generate_synthetic_event(i, base_time, seed + i)
        features_list.append(event.features.to_array())
        bz_list.append(event.bz_true)
        severity_list.append(event.severity_class)
        event_ids.append(event.event_id)

    # 2. Generate HIGH severity events
    for i in range(n_high):
        event = generate_high_event(n_regular + i, base_time, seed + n_regular + i)
        features_list.append(event.features.to_array())
        bz_list.append(event.bz_true)
        severity_list.append(event.severity_class)
        event_ids.append(event.event_id)

    # 3. Generate EXTREME severity events
    for i in range(n_extreme):
        event = generate_extreme_event(n_regular + n_high + i, base_time, seed + n_regular + n_high + i)
        features_list.append(event.features.to_array())
        bz_list.append(event.bz_true)
        severity_list.append(event.severity_class)
        event_ids.append(event.event_id)

    return (
        np.array(features_list, dtype=np.float32),
        np.array(bz_list, dtype=np.float32),
        np.array(severity_list, dtype=np.int64),
        event_ids
    )


# ============================================================================
# HISTORICAL EVENTS
# ============================================================================

# Well-characterized historical events with estimated Bz values
# References: CDAW CME catalog, ACE/DSCOVR measurements
# NOTE: This event catalog serves the modular pipeline (train.py).
# run_final_validation.py has its own independent event database.
# Parameter differences (e.g. initial vs arrival speed) are expected.
HISTORICAL_EVENTS = [
    # NOTE: Bastille Day 2000 is intentionally EXCLUDED from this list.
    # It is the sole validation/showcase target — including it here would
    # constitute data leakage for the modular pipeline (train.py).
    # For the production pipeline, see run_final_validation.py which has
    # its own explicit train/test/showcase split.
    {
        'event_id': 'halloween_2003_1',
        'date': '2003-10-28',
        'speed': 2459,
        'width': 360,
        'bz': -50,
        'source_lat': 16,
        'source_lon': -8,
        'notes': 'X17.2 flare, Halloween storms'
    },
    {
        'event_id': 'halloween_2003_2',
        'date': '2003-10-29',
        'speed': 2029,
        'width': 360,
        'bz': -49,
        'source_lat': -15,
        'source_lon': 2,
        'notes': 'X10 flare, second Halloween CME'
    },
    {
        'event_id': 'carrington_proxy',
        'date': '2003-11-04',
        'speed': 2657,
        'width': 360,
        'bz': -70,
        'source_lat': -19,
        'source_lon': -80,
        'notes': 'X28+ flare, Carrington-class proxy (saturated detectors)'
    },
    {
        'event_id': 'easter_2001',
        'date': '2001-04-15',
        'speed': 1199,
        'width': 245,
        'bz': -30,
        'source_lat': -20,
        'source_lon': -85,
        'notes': 'X14.4 flare'
    },
    {
        'event_id': 'july_2012_farside',
        'date': '2012-07-23',
        'speed': 3050,
        'width': 360,
        'bz': -75,  # Estimated if Earth-directed
        'source_lat': -15,
        'source_lon': 120,  # Far-side
        'notes': 'Carrington-class, missed Earth'
    },
    {
        'event_id': 'sept_2017',
        'date': '2017-09-06',
        'speed': 1571,
        'width': 360,
        'bz': -32,
        'source_lat': -9,
        'source_lon': 34,
        'notes': 'X9.3 flare, largest of Cycle 24'
    },
    {
        'event_id': 'march_1989',
        'date': '1989-03-10',
        'speed': 1200,
        'width': 300,
        'bz': -40,
        'source_lat': 35,
        'source_lon': 55,
        'notes': 'Quebec blackout event'
    },
    {
        'event_id': 'nov_2001',
        'date': '2001-11-04',
        'speed': 1100,
        'width': 360,
        'bz': -30,
        'source_lat': 18,
        'source_lon': 12,
        'notes': 'X1.0 flare with fast CME'
    },
    {
        'event_id': 'dec_2006',
        'date': '2006-12-13',
        'speed': 1774,
        'width': 360,
        'bz': -48,
        'source_lat': 6,
        'source_lon': 38,
        'notes': 'X3.4 flare, first STEREO observation'
    },
    # More moderate events for class balance
    {
        'event_id': 'july_2000',
        'date': '2000-07-11',
        'speed': 1078,
        'width': 180,
        'bz': -22,
        'source_lat': 18,
        'source_lon': 0,
        'notes': 'Pre-Bastille Day CME'
    },
    {
        'event_id': 'nov_2003_late',
        'date': '2003-11-20',
        'speed': 1660,
        'width': 360,
        'bz': -42,
        'source_lat': 1,
        'source_lon': -4,
        'notes': 'M-class flare, fast CME'
    },
    {
        'event_id': 'jan_2005',
        'date': '2005-01-15',
        'speed': 2861,
        'width': 360,
        'bz': -55,
        'source_lat': 14,
        'source_lon': 6,
        'notes': 'X2.6 flare'
    },
    {
        'event_id': 'sept_2005',
        'date': '2005-09-07',
        'speed': 2257,
        'width': 360,
        'bz': -45,
        'source_lat': -8,
        'source_lon': 77,
        'notes': 'X17.0 flare'
    },
    {
        'event_id': 'dec_2001',
        'date': '2001-12-28',
        'speed': 1446,
        'width': 300,
        'bz': -28,
        'source_lat': 19,
        'source_lon': -40,
        'notes': 'M-class flare'
    },
    # Lower severity events
    {
        'event_id': 'apr_2000',
        'date': '2000-04-04',
        'speed': 1188,
        'width': 160,
        'bz': -18,
        'source_lat': 16,
        'source_lon': -66,
        'notes': 'C9.7 flare'
    },
    {
        'event_id': 'aug_2002',
        'date': '2002-08-14',
        'speed': 1309,
        'width': 180,
        'bz': -20,
        'source_lat': -10,
        'source_lon': 54,
        'notes': 'M-class flare'
    },
    {
        'event_id': 'may_2003',
        'date': '2003-05-28',
        'speed': 1366,
        'width': 190,
        'bz': -24,
        'source_lat': 8,
        'source_lon': -20,
        'notes': 'X3.6 flare'
    },
    {
        'event_id': 'jan_2002',
        'date': '2002-01-10',
        'speed': 1794,
        'width': 360,
        'bz': -35,
        'source_lat': 23,
        'source_lon': 35,
        'notes': 'C-class flare, fast CME'
    },
    {
        'event_id': 'oct_2000',
        'date': '2000-10-25',
        'speed': 770,
        'width': 120,
        'bz': -15,
        'source_lat': -14,
        'source_lon': -10,
        'notes': 'M-class flare, moderate CME'
    },
]


def create_historical_event(event_data: Dict, seed: int = None) -> SyntheticEvent:
    """
    Create a SyntheticEvent from historical event data.

    Uses measured parameters and generates the unmeasured features
    with appropriate uncertainty.

    Parameters
    ----------
    event_data : Dict
        Historical event data dictionary
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    event : SyntheticEvent
        Event with features derived from historical data
    """
    if seed is not None:
        np.random.seed(seed)

    # Parse date
    eruption_time = datetime.strptime(event_data['date'], '%Y-%m-%d')
    eruption_time = eruption_time.replace(hour=12)  # Assume noon

    speed = event_data['speed']
    width = event_data['width']
    bz = event_data['bz']
    source_lat = event_data['source_lat']
    source_lon = event_data['source_lon']

    # Generate derived features
    expansion_rate = 0.5 + (speed / 1000) * np.random.uniform(0.8, 1.2)

    if speed > 1500:
        acceleration = np.random.uniform(-200, -50)
    else:
        acceleration = np.random.uniform(-50, 100)

    detection_time = np.random.uniform(0.5, 2.0) * (1000 / speed)
    detection_time = np.clip(detection_time, 0.3, 6.0)

    brightness_asymmetry = np.random.lognormal(0, 0.2)
    observation_completeness = np.random.beta(9, 1)

    observation_time = eruption_time + timedelta(hours=detection_time)

    # CME position at observation
    distance_au = 0.1 + detection_time * speed / (AU_IN_KM / 3600)
    distance_au = min(distance_au, 0.25)

    cme_position = np.array([distance_au * AU_IN_KM, 0, 0])
    cme_direction = np.array([1.0, 0, 0])

    features = extract_features(
        cme_speed=speed,
        angular_width=width,
        source_lat=source_lat,
        source_lon=source_lon,
        expansion_rate=expansion_rate,
        acceleration=acceleration,
        observation_time=observation_time,
        cme_position=cme_position,
        cme_direction=cme_direction,
        brightness_asymmetry=brightness_asymmetry,
        detection_time_hours=detection_time,
        observation_completeness=observation_completeness,
        helios_mode='synthetic'
    )

    severity_class, _ = bz_to_severity_class(bz, speed)

    return SyntheticEvent(
        event_id=event_data['event_id'],
        features=features,
        bz_true=bz,
        severity_class=severity_class,
        eruption_time=eruption_time,
        metadata={
            'type': 'historical',
            'notes': event_data.get('notes', '')
        }
    )


def generate_historical_dataset(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate dataset from historical events.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility

    Returns
    -------
    features : np.ndarray
        Shape (n_events, 16)
    bz_values : np.ndarray
        Shape (n_events,)
    severity_classes : np.ndarray
        Shape (n_events,)
    event_ids : List[str]
    """
    features_list = []
    bz_list = []
    severity_list = []
    event_ids = []

    for i, event_data in enumerate(HISTORICAL_EVENTS):
        event = create_historical_event(event_data, seed + i)
        features_list.append(event.features.to_array())
        bz_list.append(event.bz_true)
        severity_list.append(event.severity_class)
        event_ids.append(event.event_id)

    return (
        np.array(features_list, dtype=np.float32),
        np.array(bz_list, dtype=np.float32),
        np.array(severity_list, dtype=np.int64),
        event_ids
    )


def generate_combined_dataset(
    n_synthetic: int = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate combined synthetic + historical dataset.

    Parameters
    ----------
    n_synthetic : int, optional
        Number of synthetic events (default from config)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    features : np.ndarray
        Shape (n_total, 16)
    bz_values : np.ndarray
        Shape (n_total,)
    severity_classes : np.ndarray
        Shape (n_total,)
    event_ids : List[str]
    """
    # Synthetic events
    syn_features, syn_bz, syn_severity, syn_ids = generate_synthetic_dataset(n_synthetic, seed)

    # Historical events
    hist_features, hist_bz, hist_severity, hist_ids = generate_historical_dataset(seed + 10000)

    # Combine
    features = np.vstack([syn_features, hist_features])
    bz_values = np.concatenate([syn_bz, hist_bz])
    severity = np.concatenate([syn_severity, hist_severity])
    event_ids = syn_ids + hist_ids

    return features, bz_values, severity, event_ids


if __name__ == "__main__":
    # Test dataset generation
    print("=" * 60)
    print("HELIOS Dataset Generator - Test")
    print("=" * 60)

    # Test Bz physics model
    print("\nBz Physics Model:")
    print("-" * 40)
    test_cases = [
        (500, 60, "Slow, narrow"),
        (1000, 180, "Medium, wide"),
        (1674, 360, "Bastille Day"),
        (2500, 360, "Extreme fast"),
    ]

    for speed, width, desc in test_cases:
        bz = generate_bz_from_physics(speed, width, tilt_angle_deg=60, seed=42)
        print(f"  {desc:20s}: v={speed:5.0f} km/s, w={width:3.0f} deg -> Bz={bz:6.1f} nT")

    # Test synthetic event generation
    print("\nSynthetic Event Generation:")
    print("-" * 40)
    event = generate_synthetic_event(0, seed=42)
    print(f"  Event ID: {event.event_id}")
    print(f"  Bz: {event.bz_true:.1f} nT")
    print(f"  Severity: {event.severity_class}")
    print(f"  Features shape: {event.features.to_array().shape}")

    # Test small synthetic dataset
    print("\nSmall Synthetic Dataset (100 events):")
    print("-" * 40)
    features, bz, severity, ids = generate_synthetic_dataset(n_events=100, seed=42)
    print(f"  Features shape: {features.shape}")
    print(f"  Bz range: [{bz.min():.1f}, {bz.max():.1f}] nT")
    print(f"  Severity distribution: {np.bincount(severity, minlength=4)}")

    # Test historical dataset
    print("\nHistorical Dataset:")
    print("-" * 40)
    hist_features, hist_bz, hist_severity, hist_ids = generate_historical_dataset()
    print(f"  Number of events: {len(hist_ids)}")
    print(f"  Features shape: {hist_features.shape}")
    print(f"  Bz range: [{hist_bz.min():.1f}, {hist_bz.max():.1f}] nT")
    print(f"  Severity distribution: {np.bincount(hist_severity, minlength=4)}")

    # List historical events
    print("\nHistorical Events:")
    print(f"  {'Event ID':<25} {'Speed':>8} {'Width':>8} {'Bz':>8} {'Class':>8}")
    print("  " + "-" * 60)
    for i, event_id in enumerate(hist_ids):
        print(f"  {event_id:<25} {hist_features[i, 0]:>8.0f} "
              f"{hist_features[i, 1]:>8.0f} {hist_bz[i]:>8.1f} {hist_severity[i]:>8d}")

    print("\nDataset generation tests completed!")
