"""
HELIOS Validation - Utility Functions
======================================
Observer positions, coordinate transformations, and helper functions.

Supports:
- SOHO (L1), STEREO A/B ephemeris
- Synthetic L1/L4/L5 observer positions
- Coordinate transformations (heliocentric)

Author: HELIOS Team
Date: January 2026
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List, Union
import warnings

# Constants
AU_IN_KM = 1.496e8  # 1 Astronomical Unit in kilometers
SOLAR_RADIUS_KM = 6.96e5  # Solar radius in km
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


def get_observer_position(
    instrument: str,
    time: datetime,
    helios_mode: str = "proxy"
) -> Tuple[np.ndarray, float]:
    """
    Get observer position in heliocentric coordinates.
    
    Parameters
    ----------
    instrument : str
        Observer/instrument name: 'SOHO', 'STEREO-A', 'STEREO-B',
        'L1', 'L4', 'L5' (for synthetic HELIOS mode)
    time : datetime, optional
        Observation time. If None, uses Bastille Day (2000-07-14 10:30)
    helios_mode : str
        'proxy' - Use actual spacecraft positions (SOHO at L1, STEREO as L4/L5 proxy)
        'synthetic' - Use synthetic L1/L4/L5 positions based on Earth orbit
        
    Returns
    -------
    position : np.ndarray
        Position in heliocentric cartesian coordinates [x, y, z] in km
    distance_au : float
        Distance from Sun in AU
    """
    # Default time: Bastille Day event
    if time is None:
        time = datetime(2000, 7, 14, 10, 30, 0)
    
    # Get Earth position (simplified model - circular orbit)
    earth_pos = _get_earth_position(time)
    earth_distance = np.linalg.norm(earth_pos)
    
    if helios_mode == "proxy":
        return _get_proxy_position(instrument, time, earth_pos)
    else:
        return _get_synthetic_position(instrument, time, earth_pos)


def _get_earth_position(time: datetime) -> np.ndarray:
    """
    Calculate Earth's position in heliocentric coordinates.
    
    Uses simplified circular orbit model.
    Reference: J2000 ecliptic coordinates.
    """
    # Days since J2000 epoch
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    days_since_j2000 = (time - j2000).total_seconds() / 86400.0
    
    # Earth's mean longitude (simplified)
    # Mean longitude at J2000: ~100.46 degrees
    # Mean motion: 360 / 365.25 degrees per day
    mean_longitude_deg = 100.46 + 0.9856474 * days_since_j2000
    mean_longitude_rad = np.deg2rad(mean_longitude_deg % 360)
    
    # Earth position (circular orbit approximation at 1 AU)
    x = AU_IN_KM * np.cos(mean_longitude_rad)
    y = AU_IN_KM * np.sin(mean_longitude_rad)
    z = 0.0  # Ecliptic plane
    
    return np.array([x, y, z])


def _get_proxy_position(
    instrument: str, 
    time: datetime, 
    earth_pos: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Get actual spacecraft positions (or approximations).
    
    SOHO: At L1, ~1.5 million km sunward of Earth (~0.99 AU)
    STEREO-A: Leads Earth in orbit (drifts ~22 degrees/year)
    STEREO-B: Trails Earth in orbit (until 2014, then lost contact)
    """
    earth_distance = np.linalg.norm(earth_pos)
    earth_direction = earth_pos / earth_distance
    
    if instrument.upper() in ['SOHO', 'L1']:
        # L1 is ~1.5 million km sunward of Earth
        l1_distance = 0.99 * AU_IN_KM
        position = earth_direction * l1_distance
        return position, l1_distance / AU_IN_KM
    
    elif instrument.upper() == 'STEREO-A':
        # STEREO-A separation angle (approximate for historical dates)
        # Launched 2006, separated at ~22 deg/year
        separation_angle = _get_stereo_separation(time, spacecraft='A')
        position = _rotate_position(earth_pos, separation_angle)
        return position, np.linalg.norm(position) / AU_IN_KM
    
    elif instrument.upper() == 'STEREO-B':
        # STEREO-B separation (trailing, negative angle)
        separation_angle = _get_stereo_separation(time, spacecraft='B')
        position = _rotate_position(earth_pos, separation_angle)
        return position, np.linalg.norm(position) / AU_IN_KM
    
    elif instrument.upper() == 'L4':
        # Use STEREO-A as L4 proxy
        separation_angle = _get_stereo_separation(time, spacecraft='A')
        position = _rotate_position(earth_pos, separation_angle)
        return position, np.linalg.norm(position) / AU_IN_KM
    
    elif instrument.upper() == 'L5':
        # Use STEREO-B as L5 proxy
        separation_angle = _get_stereo_separation(time, spacecraft='B')
        position = _rotate_position(earth_pos, separation_angle)
        return position, np.linalg.norm(position) / AU_IN_KM
    
    else:
        raise ValueError(f"Unknown instrument: {instrument}")


def _get_synthetic_position(
    instrument: str,
    time: datetime,
    earth_pos: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Get synthetic observer positions for HELIOS constellation.
    
    L1: Sun-Earth line, ~0.99 AU
    L4: Leading Lagrange point, +60 degrees from Earth, 1 AU
    L5: Trailing Lagrange point, -60 degrees from Earth, 1 AU
    """
    earth_distance = np.linalg.norm(earth_pos)
    earth_direction = earth_pos / earth_distance
    
    if instrument.upper() in ['SOHO', 'L1']:
        l1_distance = 0.99 * AU_IN_KM
        position = earth_direction * l1_distance
        return position, l1_distance / AU_IN_KM
    
    elif instrument.upper() in ['STEREO-A', 'L4']:
        # L4: +60 degrees ahead of Earth
        position = _rotate_position(earth_pos, 60.0)
        return position, np.linalg.norm(position) / AU_IN_KM
    
    elif instrument.upper() in ['STEREO-B', 'L5']:
        # L5: -60 degrees behind Earth
        position = _rotate_position(earth_pos, -60.0)
        return position, np.linalg.norm(position) / AU_IN_KM
    
    else:
        raise ValueError(f"Unknown instrument: {instrument}")


def _get_stereo_separation(time: datetime, spacecraft: str = 'A') -> float:
    """
    Calculate STEREO separation angle from Earth.
    
    STEREO launched October 26, 2006.
    Separation rate: ~22 degrees per year.
    STEREO-A leads (positive), STEREO-B trails (negative).
    """
    launch_date = datetime(2006, 10, 26)
    
    if time < launch_date:
        # Before STEREO launch, use fixed angles for simulation
        # (representing what HELIOS would observe)
        if spacecraft == 'A':
            return 60.0  # Synthetic L4
        else:
            return -60.0  # Synthetic L5
    
    # Years since launch
    years_since_launch = (time - launch_date).days / 365.25
    
    # Separation rate ~22 degrees per year
    separation = 22.0 * years_since_launch
    
    # Cap at reasonable values
    separation = min(separation, 180.0)
    
    if spacecraft == 'A':
        return separation  # Leading
    else:
        return -separation  # Trailing


def _rotate_position(position: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate position vector in the ecliptic plane.
    
    Positive angle = counterclockwise (leading in orbit)
    Negative angle = clockwise (trailing in orbit)
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix for z-axis (ecliptic pole)
    x_new = position[0] * cos_a - position[1] * sin_a
    y_new = position[0] * sin_a + position[1] * cos_a
    z_new = position[2]
    
    return np.array([x_new, y_new, z_new])


def compute_line_of_sight(
    observer_pos: np.ndarray,
    cme_position_angle_deg: float,
    cme_latitude_deg: float = 0.0
) -> np.ndarray:
    """
    Compute line-of-sight unit vector from observer toward CME.
    
    Parameters
    ----------
    observer_pos : np.ndarray
        Observer position in heliocentric coordinates [x, y, z] in km
    cme_position_angle_deg : float
        CME position angle as observed from the instrument (0 = North, 90 = West)
    cme_latitude_deg : float
        CME heliographic latitude
        
    Returns
    -------
    los : np.ndarray
        Line-of-sight unit vector pointing from observer toward Sun/CME
    """
    # Convert position angle to radians
    pa_rad = np.deg2rad(cme_position_angle_deg)
    lat_rad = np.deg2rad(cme_latitude_deg)
    
    # Direction from observer toward Sun center
    sun_direction = -observer_pos / np.linalg.norm(observer_pos)
    
    # Build local coordinate system at observer
    # z-axis: toward ecliptic north
    z_local = np.array([0, 0, 1])
    
    # x-axis: toward Sun (approximately)
    x_local = sun_direction
    
    # y-axis: completes right-handed system
    y_local = np.cross(z_local, x_local)
    y_local = y_local / np.linalg.norm(y_local)
    
    # Recalculate z to be orthogonal
    z_local = np.cross(x_local, y_local)
    z_local = z_local / np.linalg.norm(z_local)
    
    # Position angle is measured from North (z) toward West (y)
    # Latitude is elevation from ecliptic
    los_x = np.cos(lat_rad)
    los_y = np.cos(lat_rad) * np.sin(pa_rad)
    los_z = np.sin(lat_rad) * np.cos(pa_rad) + np.sin(lat_rad)
    
    # Transform to heliocentric coordinates
    los = los_x * x_local + los_y * y_local + los_z * z_local
    los = los / np.linalg.norm(los)
    
    return los


def compute_viewing_angle(
    observer_pos: np.ndarray,
    cme_direction: np.ndarray
) -> float:
    """
    Compute viewing angle between observer LOS and CME propagation direction.
    
    Parameters
    ----------
    observer_pos : np.ndarray
        Observer position in heliocentric coordinates
    cme_direction : np.ndarray
        CME propagation direction unit vector
        
    Returns
    -------
    angle_deg : float
        Viewing angle in degrees
    """
    observer_direction = observer_pos / np.linalg.norm(observer_pos)
    cos_angle = np.dot(observer_direction, cme_direction)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    return np.rad2deg(angle_rad)


def heliocentric_to_spherical(position: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert heliocentric Cartesian coordinates to spherical.
    
    Returns
    -------
    r : float
        Distance from Sun in km
    lon : float
        Heliographic longitude in degrees
    lat : float
        Heliographic latitude in degrees
    """
    x, y, z = position
    r = np.linalg.norm(position)
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arcsin(z / r)) if r > 0 else 0.0
    return r, lon, lat


def spherical_to_heliocentric(
    r: float,
    lon_deg: float,
    lat_deg: float
) -> np.ndarray:
    """
    Convert spherical coordinates to heliocentric Cartesian.
    
    Parameters
    ----------
    r : float
        Distance from Sun in km
    lon_deg : float
        Heliographic longitude in degrees
    lat_deg : float
        Heliographic latitude in degrees
        
    Returns
    -------
    position : np.ndarray
        Position in heliocentric coordinates [x, y, z] in km
    """
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return np.array([x, y, z])


def time_to_hours_since_event(
    time: datetime,
    event_time: datetime
) -> float:
    """Convert datetime to hours since event start."""
    return (time - event_time).total_seconds() / 3600.0


def hours_to_time(
    hours: float,
    event_time: datetime
) -> datetime:
    """Convert hours since event to datetime."""
    return event_time + timedelta(hours=hours)


def parse_datetime(dt_string: str) -> datetime:
    """
    Parse datetime string in various formats.
    
    Supported formats:
    - 'YYYY-MM-DD HH:MM:SS'
    - 'YYYY-MM-DD HH:MM'
    - 'YYYY-MM-DDTHH:MM:SS'
    - 'YYYY-MM-DD'
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse datetime: {dt_string}")


def get_constellation_positions(
    time: datetime,
    helios_mode: str = "synthetic",
    include_instruments: Optional[List[str]] = None
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Get positions for all observers in the constellation.
    
    Parameters
    ----------
    time : datetime
        Observation time
    helios_mode : str
        'proxy' or 'synthetic'
    include_instruments : list, optional
        List of instruments to include. Default: ['L1', 'L4', 'L5']
        
    Returns
    -------
    positions : dict
        Dictionary mapping instrument name to (position, distance_au)
    """
    if include_instruments is None:
        include_instruments = ['L1', 'L4', 'L5']
    
    positions = {}
    for instrument in include_instruments:
        try:
            pos, dist = get_observer_position(instrument, time, helios_mode)
            positions[instrument] = (pos, dist)
        except ValueError as e:
            warnings.warn(f"Could not get position for {instrument}: {e}")
    
    return positions


def angular_separation(
    pos1: np.ndarray,
    pos2: np.ndarray
) -> float:
    """
    Calculate angular separation between two positions as seen from the Sun.
    
    Returns angle in degrees.
    """
    dir1 = pos1 / np.linalg.norm(pos1)
    dir2 = pos2 / np.linalg.norm(pos2)
    cos_angle = np.dot(dir1, dir2)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    return np.rad2deg(angle_rad)


def create_output_directory(base_path: str = "output") -> str:
    """Create output directory if it doesn't exist."""
    import os
    os.makedirs(base_path, exist_ok=True)
    return base_path


def save_dataframe_to_csv(
    df,
    filename: str,
    output_dir: str = "output"
) -> str:
    """Save pandas DataFrame to CSV in output directory."""
    import os
    create_output_directory(output_dir)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    # Test observer positions
    test_time = datetime(2000, 7, 14, 10, 30, 0)  # Bastille Day
    
    print("=" * 60)
    print("HELIOS Utils - Observer Position Test")
    print("=" * 60)
    print(f"Test time: {test_time}")
    print()
    
    for mode in ['proxy', 'synthetic']:
        print(f"\n{mode.upper()} MODE:")
        print("-" * 40)
        
        for instrument in ['SOHO', 'STEREO-A', 'STEREO-B', 'L1', 'L4', 'L5']:
            try:
                pos, dist = get_observer_position(instrument, test_time, mode)
                _, lon, lat = heliocentric_to_spherical(pos)
                print(f"  {instrument:12s}: {dist:.3f} AU, lon={lon:+7.1f}°, lat={lat:+5.1f}°")
            except Exception as e:
                print(f"  {instrument:12s}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
