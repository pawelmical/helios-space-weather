"""
HELIOS Validation - Stereoscopic Triangulation Module
======================================================
Line-of-sight triangulation for 3D CME position estimation.

Features:
- Two-observer triangulation (LOS intersection)
- Multi-observer least squares solution
- Monte-Carlo uncertainty analysis
- Spatial resolution estimation

Author: HELIOS Team
Date: January 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

# Constants
AU_IN_KM = 1.496e8
DEG_TO_RAD = np.pi / 180.0


@dataclass
class TriangulationResult:
    """Container for triangulation results."""
    position: np.ndarray  # 3D position in km
    distance_au: float
    uncertainty_km: float
    method: str
    n_observers: int
    residual: float  # Distance between LOS at closest approach
    
    def to_dict(self) -> Dict:
        return {
            'x_km': self.position[0],
            'y_km': self.position[1],
            'z_km': self.position[2],
            'distance_au': self.distance_au,
            'uncertainty_km': self.uncertainty_km,
            'uncertainty_au': self.uncertainty_km / AU_IN_KM,
            'method': self.method,
            'n_observers': self.n_observers,
            'residual_km': self.residual
        }


@dataclass 
class MonteCarloResult:
    """Container for Monte-Carlo triangulation results."""
    mean_position: np.ndarray
    std_position: np.ndarray
    median_position: np.ndarray
    quantile_16: np.ndarray
    quantile_84: np.ndarray
    delta_r_km: float  # Spatial resolution (1-sigma)
    n_samples: int
    perturbation_deg: float
    all_positions: np.ndarray  # All MC samples
    
    def to_dict(self) -> Dict:
        return {
            'mean_x_km': self.mean_position[0],
            'mean_y_km': self.mean_position[1],
            'mean_z_km': self.mean_position[2],
            'mean_r_au': np.linalg.norm(self.mean_position) / AU_IN_KM,
            'std_x_km': self.std_position[0],
            'std_y_km': self.std_position[1],
            'std_z_km': self.std_position[2],
            'delta_r_km': self.delta_r_km,
            'delta_r_au': self.delta_r_km / AU_IN_KM,
            'n_samples': self.n_samples,
            'perturbation_deg': self.perturbation_deg
        }


def triangulate_two_lines(
    r1: np.ndarray,
    u1: np.ndarray,
    r2: np.ndarray,
    u2: np.ndarray
) -> Tuple[np.ndarray, float, float, float]:
    """
    Find the closest point between two lines in 3D space.
    
    Each line is defined as: P(t) = r + t*u
    
    Parameters
    ----------
    r1 : np.ndarray
        Origin point of line 1 (observer 1 position)
    u1 : np.ndarray
        Direction vector of line 1 (line of sight)
    r2 : np.ndarray
        Origin point of line 2 (observer 2 position)
    u2 : np.ndarray
        Direction vector of line 2 (line of sight)
        
    Returns
    -------
    midpoint : np.ndarray
        Midpoint between the closest approach points
    distance : float
        Distance between the two lines at closest approach
    s1 : float
        Parameter for line 1 at closest approach
    s2 : float
        Parameter for line 2 at closest approach
    """
    # Normalize direction vectors
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    
    # Vector from r1 to r2
    w0 = r1 - r2
    
    # Dot products
    a = np.dot(u1, u1)  # = 1 (normalized)
    b = np.dot(u1, u2)
    c = np.dot(u2, u2)  # = 1 (normalized)
    d = np.dot(u1, w0)
    e = np.dot(u2, w0)
    
    # Denominator
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:
        # Lines are parallel
        s1 = 0.0
        s2 = d / b if abs(b) > 1e-10 else 0.0
    else:
        s1 = (b * e - c * d) / denom
        s2 = (a * e - b * d) / denom
    
    # Closest points on each line
    p1 = r1 + s1 * u1
    p2 = r2 + s2 * u2
    
    # Midpoint and distance
    midpoint = (p1 + p2) / 2
    distance = np.linalg.norm(p2 - p1)
    
    return midpoint, distance, s1, s2


def estimate_point_from_observations(
    observer_positions: List[np.ndarray],
    line_of_sight_vectors: List[np.ndarray]
) -> TriangulationResult:
    """
    Estimate 3D point from multiple observer lines of sight.
    
    For N>2 observers, uses least squares to find the point
    that minimizes the sum of squared distances to all lines.
    
    Parameters
    ----------
    observer_positions : list of np.ndarray
        Observer positions in heliocentric coordinates [x, y, z]
    line_of_sight_vectors : list of np.ndarray
        Unit vectors pointing from each observer toward the target
        
    Returns
    -------
    result : TriangulationResult
        Triangulation result including position and uncertainty
    """
    n_obs = len(observer_positions)
    
    if n_obs < 2:
        raise ValueError("Need at least 2 observers for triangulation")
    
    if n_obs == 2:
        # Direct two-line intersection
        r1, r2 = observer_positions
        u1, u2 = line_of_sight_vectors
        
        midpoint, distance, s1, s2 = triangulate_two_lines(r1, u1, r2, u2)
        
        return TriangulationResult(
            position=midpoint,
            distance_au=np.linalg.norm(midpoint) / AU_IN_KM,
            uncertainty_km=distance / 2,  # Half the separation
            method='two_line_intersection',
            n_observers=2,
            residual=distance
        )
    
    else:
        # Multi-observer least squares
        # Set up system: minimize sum of |P - (r_i + t_i * u_i)|^2
        # This leads to: A @ [P; t_1; t_2; ...] = b
        
        # Build the normal equations
        # We want to find P that minimizes sum of squared distances to all lines
        
        # Matrix form: (I - u_i @ u_i.T) @ P = (I - u_i @ u_i.T) @ r_i
        # Sum over all observers
        
        A = np.zeros((3, 3))
        b = np.zeros(3)
        
        for r_i, u_i in zip(observer_positions, line_of_sight_vectors):
            u_i = u_i / np.linalg.norm(u_i)
            projection_matrix = np.eye(3) - np.outer(u_i, u_i)
            A += projection_matrix
            b += projection_matrix @ r_i
        
        # Solve the system
        try:
            position = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            position = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Calculate residuals (distances to each line)
        residuals = []
        for r_i, u_i in zip(observer_positions, line_of_sight_vectors):
            u_i = u_i / np.linalg.norm(u_i)
            # Distance from point to line
            v = position - r_i
            dist = np.linalg.norm(v - np.dot(v, u_i) * u_i)
            residuals.append(dist)
        
        mean_residual = np.mean(residuals)
        
        return TriangulationResult(
            position=position,
            distance_au=np.linalg.norm(position) / AU_IN_KM,
            uncertainty_km=mean_residual,
            method='least_squares_multi_observer',
            n_observers=n_obs,
            residual=mean_residual
        )


def montecarlo_triangulation(
    r1: np.ndarray,
    u1: np.ndarray,
    r2: np.ndarray,
    u2: np.ndarray,
    sigma_deg: float = 1.0,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> MonteCarloResult:
    """
    Monte-Carlo triangulation with angular perturbations.
    
    Perturbs the line-of-sight vectors by random angles drawn from
    a Gaussian distribution to estimate spatial resolution.
    
    Parameters
    ----------
    r1, r2 : np.ndarray
        Observer positions
    u1, u2 : np.ndarray
        Line-of-sight direction vectors
    sigma_deg : float
        Standard deviation of angular perturbation in degrees
    n_samples : int
        Number of Monte-Carlo samples
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    result : MonteCarloResult
        Monte-Carlo analysis results
    """
    if seed is not None:
        np.random.seed(seed)
    
    sigma_rad = sigma_deg * DEG_TO_RAD
    
    # Normalize input vectors
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    
    positions = []
    
    for _ in range(n_samples):
        # Perturb each LOS vector
        u1_perturbed = _perturb_direction(u1, sigma_rad)
        u2_perturbed = _perturb_direction(u2, sigma_rad)
        
        # Triangulate
        midpoint, _, _, _ = triangulate_two_lines(r1, u1_perturbed, r2, u2_perturbed)
        positions.append(midpoint)
    
    positions = np.array(positions)
    
    # Statistics
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)
    median_pos = np.median(positions, axis=0)
    q16 = np.percentile(positions, 16, axis=0)
    q84 = np.percentile(positions, 84, axis=0)
    
    # Spatial resolution: 1-sigma spread in 3D
    distances_from_mean = np.linalg.norm(positions - mean_pos, axis=1)
    delta_r = np.std(distances_from_mean)
    
    return MonteCarloResult(
        mean_position=mean_pos,
        std_position=std_pos,
        median_position=median_pos,
        quantile_16=q16,
        quantile_84=q84,
        delta_r_km=delta_r,
        n_samples=n_samples,
        perturbation_deg=sigma_deg,
        all_positions=positions
    )


def _perturb_direction(
    u: np.ndarray,
    sigma_rad: float
) -> np.ndarray:
    """
    Perturb a direction vector by a random angle.
    
    The perturbation is applied in a random direction perpendicular
    to the original vector.
    """
    # Generate random perturbation angles
    theta = np.random.normal(0, sigma_rad)  # Polar angle perturbation
    phi = np.random.uniform(0, 2 * np.pi)   # Azimuthal angle of perturbation
    
    # Find two orthogonal vectors perpendicular to u
    if abs(u[0]) < 0.9:
        v1 = np.cross(u, np.array([1, 0, 0]))
    else:
        v1 = np.cross(u, np.array([0, 1, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    
    # Perturbed direction
    u_perturbed = (u * np.cos(theta) + 
                   (v1 * np.cos(phi) + v2 * np.sin(phi)) * np.sin(theta))
    
    return u_perturbed / np.linalg.norm(u_perturbed)


def analyze_spatial_resolution(
    baseline_km: float,
    target_distance_au: float,
    sigma_deg_list: List[float] = [1.0, 0.5, 0.25],
    n_samples: int = 1000
) -> Dict[float, float]:
    """
    Analyze spatial resolution as function of angular uncertainty.
    
    Creates a simple geometry with two observers separated by
    a specified baseline, observing a target at a given distance.
    
    Parameters
    ----------
    baseline_km : float
        Separation between observers in km
    target_distance_au : float
        Distance to target from Sun in AU
    sigma_deg_list : list of float
        Angular uncertainties to test (degrees)
    n_samples : int
        Monte-Carlo samples per test
        
    Returns
    -------
    results : dict
        Mapping from sigma_deg to delta_r (km)
    """
    target_distance_km = target_distance_au * AU_IN_KM
    
    # Set up simple geometry
    # Observer 1 at origin, Observer 2 at +y direction
    r1 = np.array([-baseline_km/2, 0, 0])
    r2 = np.array([+baseline_km/2, 0, 0])
    
    # Target along +x axis at specified distance
    target = np.array([target_distance_km, 0, 0])
    
    # Lines of sight toward target
    u1 = target - r1
    u1 = u1 / np.linalg.norm(u1)
    u2 = target - r2
    u2 = u2 / np.linalg.norm(u2)
    
    results = {}
    
    for sigma_deg in sigma_deg_list:
        mc_result = montecarlo_triangulation(
            r1, u1, r2, u2,
            sigma_deg=sigma_deg,
            n_samples=n_samples,
            seed=42
        )
        results[sigma_deg] = mc_result.delta_r_km
    
    return results


def triangulate_cme_feature(
    observer_positions: Dict[str, np.ndarray],
    position_angles: Dict[str, float],
    latitudes: Optional[Dict[str, float]] = None
) -> TriangulationResult:
    """
    Triangulate a CME feature from multi-instrument observations.
    
    Parameters
    ----------
    observer_positions : dict
        Mapping instrument name -> position [x, y, z] in km
    position_angles : dict
        Mapping instrument name -> position angle (degrees, 0=N, 90=W)
    latitudes : dict, optional
        Mapping instrument name -> heliographic latitude (degrees)
        
    Returns
    -------
    result : TriangulationResult
    """
    from .utils import compute_line_of_sight
    
    obs_list = []
    los_list = []
    
    for instrument in observer_positions:
        if instrument not in position_angles:
            continue
            
        pos = observer_positions[instrument]
        pa = position_angles[instrument]
        lat = latitudes.get(instrument, 0.0) if latitudes else 0.0
        
        los = compute_line_of_sight(pos, pa, lat)
        
        obs_list.append(pos)
        los_list.append(los)
    
    return estimate_point_from_observations(obs_list, los_list)


def create_triangulation_table(
    events: List[Dict],
    mc_results: Dict[str, Dict[float, MonteCarloResult]]
):
    """
    Create triangulation results table.
    
    Parameters
    ----------
    events : list of dict
        Event information
    mc_results : dict
        {event_id: {sigma_deg: MonteCarloResult}}
        
    Returns
    -------
    df : pd.DataFrame
        Triangulation table
    """
    import pandas as pd
    
    rows = []
    
    for event in events:
        event_id = event.get('event_id', 'unknown')
        
        if event_id not in mc_results:
            continue
            
        for sigma_deg, mc in mc_results[event_id].items():
            row = {
                'event_id': event_id,
                'sigma_deg': sigma_deg,
                'mean_r_au': np.linalg.norm(mc.mean_position) / AU_IN_KM,
                'delta_r_km': mc.delta_r_km,
                'delta_r_solar_radii': mc.delta_r_km / 6.96e5,
                'std_x_km': mc.std_position[0],
                'std_y_km': mc.std_position[1],
                'std_z_km': mc.std_position[2],
                'n_samples': mc.n_samples
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def compute_degraded_mode_resolution(
    constellation_configs: List[str],
    target_distances_au: List[float] = [0.5, 1.0],
    sigma_deg: float = 0.5,
    n_samples: int = 1000
):
    """
    Compare spatial resolution for different constellation configurations.
    
    Parameters
    ----------
    constellation_configs : list of str
        Configurations to test: ['L1-only', 'L1+L4', 'L1+L5', 'L1+L4+L5', 'L4+L5']
    target_distances_au : list of float
        Target distances to test
    sigma_deg : float
        Angular uncertainty
    n_samples : int
        Monte-Carlo samples
        
    Returns
    -------
    df : pd.DataFrame
        Comparison table
    """
    import pandas as pd
    
    # Define observer positions (simplified, at 1 AU)
    observer_templates = {
        'L1': np.array([0.99 * AU_IN_KM, 0, 0]),  # Sunward of Earth
        'L4': np.array([0.5 * AU_IN_KM, 0.866 * AU_IN_KM, 0]),  # +60 deg
        'L5': np.array([0.5 * AU_IN_KM, -0.866 * AU_IN_KM, 0])   # -60 deg
    }
    
    rows = []
    
    for config in constellation_configs:
        observers = config.replace(' ', '').split('+')
        
        if len(observers) < 2:
            # Single observer - cannot triangulate
            for r_au in target_distances_au:
                rows.append({
                    'configuration': config,
                    'n_observers': 1,
                    'target_r_au': r_au,
                    'delta_r_km': np.inf,
                    'can_triangulate': False
                })
            continue
        
        # Get observer positions
        obs_positions = [observer_templates[obs] for obs in observers]
        
        for r_au in target_distances_au:
            target_km = r_au * AU_IN_KM
            target = np.array([target_km, 0, 0])  # Along Sun-Earth line
            
            # Lines of sight
            los_vectors = [(target - pos) / np.linalg.norm(target - pos) 
                          for pos in obs_positions]
            
            if len(observers) == 2:
                mc = montecarlo_triangulation(
                    obs_positions[0], los_vectors[0],
                    obs_positions[1], los_vectors[1],
                    sigma_deg=sigma_deg,
                    n_samples=n_samples
                )
                delta_r = mc.delta_r_km
            else:
                # Multi-observer: average pairwise
                deltas = []
                for i in range(len(observers)):
                    for j in range(i+1, len(observers)):
                        mc = montecarlo_triangulation(
                            obs_positions[i], los_vectors[i],
                            obs_positions[j], los_vectors[j],
                            sigma_deg=sigma_deg,
                            n_samples=n_samples // len(observers)
                        )
                        deltas.append(mc.delta_r_km)
                delta_r = np.mean(deltas)
            
            rows.append({
                'configuration': config,
                'n_observers': len(observers),
                'target_r_au': r_au,
                'delta_r_km': delta_r,
                'delta_r_solar_radii': delta_r / 6.96e5,
                'can_triangulate': True
            })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("HELIOS Triangulation Module - Test")
    print("=" * 60)
    
    # Test two-line triangulation
    print("\n1. Two-line triangulation test:")
    r1 = np.array([0, -AU_IN_KM/2, 0])  # Observer 1
    r2 = np.array([0, +AU_IN_KM/2, 0])  # Observer 2
    
    target = np.array([0.5 * AU_IN_KM, 0, 0])  # Target at 0.5 AU
    
    u1 = target - r1
    u1 = u1 / np.linalg.norm(u1)
    u2 = target - r2
    u2 = u2 / np.linalg.norm(u2)
    
    midpoint, distance, s1, s2 = triangulate_two_lines(r1, u1, r2, u2)
    
    print(f"   True target:     [{target[0]/AU_IN_KM:.3f}, {target[1]/AU_IN_KM:.3f}, {target[2]/AU_IN_KM:.3f}] AU")
    print(f"   Triangulated:    [{midpoint[0]/AU_IN_KM:.3f}, {midpoint[1]/AU_IN_KM:.3f}, {midpoint[2]/AU_IN_KM:.3f}] AU")
    print(f"   LOS distance:    {distance/1e6:.1f} thousand km")
    
    # Test Monte-Carlo
    print("\n2. Monte-Carlo uncertainty analysis:")
    for sigma in [1.0, 0.5, 0.25]:
        mc = montecarlo_triangulation(r1, u1, r2, u2, sigma_deg=sigma, n_samples=1000, seed=42)
        print(f"   σ = {sigma}°: ΔR = {mc.delta_r_km/1e6:.2f} million km = {mc.delta_r_km/6.96e5:.1f} solar radii")
    
    # Test spatial resolution analysis
    print("\n3. Spatial resolution vs distance:")
    baseline = 2 * AU_IN_KM  # L4-L5 separation
    for r_au in [0.5, 1.0]:
        results = analyze_spatial_resolution(baseline, r_au)
        print(f"   R = {r_au} AU:")
        for sigma, delta_r in results.items():
            print(f"      σ = {sigma}°: ΔR = {delta_r/1e6:.2f} million km")
    
    print("\n" + "=" * 60)
    print("Test completed!")
