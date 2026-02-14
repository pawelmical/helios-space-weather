"""Test triangulation-constrained prediction - REALISTIC SCENARIO.

This test simulates a realistic scenario where:
1. The true CME parameters are SIGNIFICANTLY different from our calibrated model
2. Initial speed estimate has uncertainty
3. Drag conditions vary from our baseline

The key insight: HELIOS triangulation tracks the ACTUAL CME, so it can correct
for these unknown factors in real-time.
"""
import sys
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent / 'code'
sys.path.insert(0, str(code_dir))

import numpy as np
from ensemble_propagation import (  # type: ignore
    calculate_cme_trajectory, 
    run_ensemble,
    triangulation_constrained_prediction,
    AU_IN_KM
)
from triangulation import triangulate_two_lines  # type: ignore

# Helper function for observer positions (matches geometry_verification.py)
def get_observer_position(longitude_deg: float, distance_au: float = 1.0) -> np.ndarray:
    """Get observer position given heliographic longitude.
    
    Earth is at longitude 0 deg, at 1 AU.
    L1 is at 0 deg, 0.99 AU (Sun-Earth line).
    L4 is at +60 deg, L5 is at -60 deg.
    """
    DEG_TO_RAD = np.pi / 180.0
    lon_rad = longitude_deg * DEG_TO_RAD
    r = distance_au * AU_IN_KM
    return np.array([r * np.cos(lon_rad), r * np.sin(lon_rad), 0.0])


np.random.seed(12345)  # For reproducibility

print('TRIANGULATION-CONSTRAINED PREDICTION - REALISTIC TEST')
print('='*70)
print('Using L1+L4+L5 constellation geometry (static, optimal for Earth-directed CMEs)')
print('='*70)

# Simulate the "TRUE" CME trajectory (this is what nature does)
# We'll use SIGNIFICANTLY different parameters - this is the model uncertainty we face!
print('1. Simulating TRUE CME trajectory (nature):')
print('   (Reality differs significantly from our calibrated model)')

# True parameters (UNKNOWN to predictor - these are what actually happen)
# In reality, drag conditions vary event-to-event
true_gamma_0 = 3.926e-10 * 0.70   # 30% LOWER - weaker drag, faster CME
true_n_power = 14.09 * 0.80       # 20% LOWER - different heliospheric profile
true_initial_speed = 1750.0       # Actual speed (we'll estimate 1674 based on coronagraph)

# What we ESTIMATE from coronagraph observations (with error)
estimated_speed = 1674.0  # Our estimate has ~5% error

print(f'   True initial speed: {true_initial_speed} km/s')
print(f'   Estimated speed:    {estimated_speed} km/s (5% error)')
print(f'   True gamma_0:       {true_gamma_0:.4e} (30% lower than calibrated)')
print(f'   True n_power:       {true_n_power:.2f} (20% lower than calibrated)')

# Get true trajectory
distance_km = 1e6
speed = true_initial_speed  # Use TRUE initial speed
time = 0.0
dt = 0.005

true_times = [0.0]
true_distances = [distance_km / AU_IN_KM]
true_speeds = [speed]

while distance_km < AU_IN_KM:
    r_au = distance_km / AU_IN_KM
    gamma = true_gamma_0 * (1.0 + (r_au / 0.605) ** true_n_power)
    delta_v = speed - 450.0
    drag = -gamma * delta_v * abs(delta_v)
    dt_s = dt * 3600
    speed = max(speed + drag * dt_s, 450.0)
    distance_km += speed * dt_s
    time += dt
    true_times.append(time)
    true_distances.append(distance_km / AU_IN_KM)
    true_speeds.append(speed)

true_times = np.array(true_times)
true_distances = np.array(true_distances)
true_speeds = np.array(true_speeds)
true_arrival = true_times[-1]
true_speed_earth = true_speeds[-1]

print(f'   True arrival: {true_arrival:.2f}h')
print(f'   True speed at Earth: {true_speed_earth:.0f} km/s')
print(f'   (Compare: calibrated model predicts ~28.5h for 1674 km/s)')

# Extract "measurements" at 5, 10, 15, 20 hours (what HELIOS triangulation provides)
# Use L1+L4 pair for triangulation (optimal 90° intersection for Earth-directed CMEs)
measurement_times = [5, 10, 15, 20]
measured_positions = []

# Observer positions (L1+L4+L5 constellation)
obs_l1 = get_observer_position(0, 0.99)   # L1: Sun-Earth line, 0.99 AU
obs_l4 = get_observer_position(60, 1.0)   # L4: +60°, 1 AU
obs_l5 = get_observer_position(-60, 1.0)  # L5: -60°, 1 AU

print()
print('2. HELIOS Constellation Setup:')
print(f'   L1: 0.99 AU at 0° (Sun-Earth line)')
print(f'   L4: 1.00 AU at +60° (leading Lagrange point)')
print(f'   L5: 1.00 AU at -60° (trailing Lagrange point)')
print(f'   Triangulation pair: L1+L4 (90° intersection - optimal!)')

print()
print('3. Triangulation measurements (from L1+L4 pair):')
for t_target in measurement_times:
    idx = np.argmin(np.abs(true_times - t_target))
    t_meas = true_times[idx]
    r_true = true_distances[idx]
    v_meas = true_speeds[idx]
    
    # True CME position at this time (Earth-directed, along x-axis)
    cme_pos_true = np.array([r_true * AU_IN_KM, 0.0, 0.0])
    
    # Lines of sight from L1 and L4 toward CME
    los_l1 = cme_pos_true - obs_l1
    los_l1 = los_l1 / np.linalg.norm(los_l1)
    los_l4 = cme_pos_true - obs_l4
    los_l4 = los_l4 / np.linalg.norm(los_l4)
    
    # Add angular measurement noise (coronagraph uncertainty: ~0.5°)
    sigma_rad = 0.5 * np.pi / 180.0
    theta1 = np.random.normal(0, sigma_rad)
    phi1 = np.random.uniform(0, 2 * np.pi)
    theta4 = np.random.normal(0, sigma_rad)
    phi4 = np.random.uniform(0, 2 * np.pi)
    
    # Perturb LOS vectors
    def perturb_los(u, theta, phi):
        if abs(u[0]) < 0.9:
            v1 = np.cross(u, np.array([1.0, 0.0, 0.0]))
        else:
            v1 = np.cross(u, np.array([0.0, 1.0, 0.0]))
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(u, v1)
        u_new = u * np.cos(theta) + (v1 * np.cos(phi) + v2 * np.sin(phi)) * np.sin(theta)
        return u_new / np.linalg.norm(u_new)
    
    los_l1_noisy = perturb_los(los_l1, theta1, phi1)
    los_l4_noisy = perturb_los(los_l4, theta4, phi4)
    
    # Triangulate position from noisy measurements
    cme_pos_measured, _, _, _ = triangulate_two_lines(obs_l1, los_l1_noisy, obs_l4, los_l4_noisy)
    r_meas = np.linalg.norm(cme_pos_measured) / AU_IN_KM
    
    measured_positions.append((t_meas, r_meas))
    
    # Calculate triangulation error
    pos_error_km = np.linalg.norm(cme_pos_measured - cme_pos_true)
    print(f'   t={t_meas:.1f}h: r={r_meas:.3f} AU (true: {r_true:.3f} AU, error: {pos_error_km/1e6:.2f} Mkm, v={v_meas:.0f} km/s)')

print()
print('4. Standard prediction (no triangulation):')
print('   Using estimated speed and calibrated drag model')
standard = run_ensemble(initial_speed=estimated_speed, n_members=200, seed=42)
range_std = standard.arrival_84_hours - standard.arrival_16_hours
err_std = abs(standard.arrival_median_hours - true_arrival)
print(f'   Predicted arrival: {standard.arrival_median_hours:.1f}h (68%: {standard.arrival_16_hours:.1f}-{standard.arrival_84_hours:.1f}h)')
print(f'   68% range: {range_std:.1f}h')
print(f'   Error vs TRUE: {err_std:.1f}h ({err_std/true_arrival*100:.1f}%)')
print(f'   TRUE arrival within prediction range? {standard.arrival_16_hours <= true_arrival <= standard.arrival_84_hours}')

print()
print('5. Triangulation-constrained prediction (HELIOS advantage):')
print('   Using estimated speed BUT fitting to L1+L4 TRIANGULATED positions')
constrained = triangulation_constrained_prediction(
    initial_speed=estimated_speed,  # We still use our estimate
    measured_positions=measured_positions,  # But we have REAL position data!
    n_ensemble=200,
    seed=42
)
arr_med = constrained['arrival_median_h']
arr_16 = constrained['arrival_16_h']
arr_84 = constrained['arrival_84_h']
range_con = constrained['constrained_arrival_range_h']
err_con = abs(arr_med - true_arrival)
reduction = constrained['uncertainty_reduction_percent']

print(f'   Predicted arrival: {arr_med:.1f}h (68%: {arr_16:.1f}-{arr_84:.1f}h)')
print(f'   68% range: {range_con:.1f}h')
print(f'   Error vs TRUE: {err_con:.1f}h ({err_con/true_arrival*100:.1f}%)')
print(f'   TRUE arrival within prediction range? {arr_16 <= true_arrival <= arr_84}')
print(f'   Fitted v0_effective: {constrained["v0_effective_kms"]:.0f} km/s (true: {true_initial_speed} km/s, est: {estimated_speed})')
print(f'   Fitted gamma_0: {constrained["gamma_0_fitted"]:.4e} (true: {true_gamma_0:.4e})')
print(f'   Fitted n_power: {constrained["n_power_fitted"]:.2f} (true: {true_n_power:.2f})')

print()
print('='*70)
print('SUMMARY: HELIOS VALUE PROPOSITION')
print('='*70)
print()
print('PREDICTION ERROR (how wrong we are):')
print(f'   Standard:    {err_std:.1f}h ({err_std/true_arrival*100:.1f}% error)')
print(f'   Constrained: {err_con:.1f}h ({err_con/true_arrival*100:.1f}% error)')
improvement_error = (err_std - err_con) / err_std * 100 if err_std > 0 else 0
print(f'   IMPROVEMENT:  {err_std - err_con:.1f}h better ({improvement_error:.0f}% reduction)')
print()
print('UNCERTAINTY RANGE (our confidence interval):')
print(f'   Standard:    {range_std:.1f}h')
print(f'   Constrained: {range_con:.1f}h')
print(f'   REDUCTION:   {range_std - range_con:.1f}h ({reduction:.0f}% tighter)')
print()
print('KEY INSIGHT - L1+L4+L5 CONSTELLATION VALUE:')
print('   Without HELIOS: We rely on calibrated model + L1 coronagraph speed estimate')
print('                   Model/solar wind variations cause large errors')
print('   With HELIOS:    L1+L4 triangulation TRACKS the actual CME at 5h, 10h, 15h, 20h')
print('                   Real position data corrects for unknown drag/speed factors')
print('                   L5 provides redundancy and far-side coverage')
print()
if err_con < err_std:
    print(f'   -> HELIOS reduced prediction error by {err_std - err_con:.1f} hours!')
else:
    print(f'   -> HELIOS constrained the uncertainty range (error similar)')
print(f'   -> Provides {range_std - range_con:.1f} hours tighter warning window')
