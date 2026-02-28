#!/usr/bin/env python3
"""
HELIOS Constellation — Appendix A: Constellation Orbital Mechanics
===================================================================

Production-quality computation of orbital parameters for the HELIOS 3-node
distributed space-weather monitoring system at Sun–Earth L1, L4, and L5.

Outputs
-------
- Console summary table (also saved to helios_orbital_params.txt)
- helios_L1_halo_3D.png            — 3-D halo trajectory + SEZ cone
- helios_L1_halo_projections.png   — X-Y / X-Z / Y-Z projections
- helios_sez_analysis.png          — Solar Exclusion Zone analysis
- helios_srp_comparison.png        — SRP perturbation comparison

References (cited inline where used)
-------------------------------------
[1] Richardson D.L. (1980), "Analytic Construction of Periodic Orbits
    About the Collinear Points," Celestial Mechanics 22(3), 241–253.
[2] Howell K.C. (1984), "Three-Dimensional, Periodic, 'Halo' Orbits,"
    Celestial Mechanics 32(1), 53–71.
[3] Howell K.C. & Pernicka H.J. (1988), "Station-Keeping Method for
    Libration Point Trajectories," Celestial Mechanics 41, 107–124.
[4] Lo M.W. et al. (2001), "Genesis Mission Design,"
    J. Astronautical Sciences 49(1), 169–184.  SK ≈ 3.8 m/s/yr.
[5] Koon W.S. et al. (2002), "Low Energy Transfer to the Moon,"
    Automatica 38(4), 571–583.  Genesis: Ay ≈ 780 000 km, Az ≈ 290 000 km.
[6] Tsiolkovsky rocket equation — propellant sizing.

Author : HELIOS project
Date   : 2026-02-17
"""

from __future__ import annotations

import time as _timer
import datetime
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import sqrt, pi, cos, sin, arctan2, arcsin
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# ══════════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS & SYSTEM PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# --- Fundamental constants (NIST / JPL DE440) ---
M_SUN   = 1.98892e30       # kg  —  Solar mass
M_EARTH = 5.97219e24       # kg  —  Earth mass
G_CONST = 6.67430e-11      # m³ kg⁻¹ s⁻²   — gravitational constant

# CR3BP mass parameter  μ = M₂ / (M₁ + M₂)  (Sun–Earth)
MU = M_EARTH / (M_SUN + M_EARTH)   # ≈ 3.003467e-6
# The user-specified exact value (JPL convention with lunar mass lumped):
MU = 3.040357e-6

# Characteristic length  L* = Sun–Earth distance
L_STAR = 1.496e8           # km   (1 AU)
L_STAR_M = L_STAR * 1e3    # m

# Characteristic time  T* = 1 / ω_Earth
OMEGA_EARTH = 2.0 * pi / (365.25 * 86400.0)     # rad/s
T_STAR = 1.0 / OMEGA_EARTH                       # s   ≈ 5.022642e6 s

# For output convenience
TU_TO_SEC  = T_STAR
TU_TO_DAYS = T_STAR / 86400.0

# Velocity unit  V* = L* / T*
V_STAR = L_STAR_M / T_STAR          # m/s
V_STAR_KM = V_STAR / 1e3            # km/s

# Solar Radiation Pressure parameters (L1 node)
P_SRP  = 4.56e-6       # N/m²  at 1 AU  [solar radiation pressure]
CR_SRP = 1.5            # reflectivity coefficient (conservative)
A_SRP  = 15.0           # m²    effective cross-section
M_SC   = 905.0          # kg    dry mass (L1 node)

# SRP acceleration in dimensional (m/s²) and normalized units
A_SRP_DIM = CR_SRP * P_SRP * (A_SRP / M_SC)     # m/s²
# Normalize: a_norm = a_dim * T*² / L*_m
A_SRP_NORM = A_SRP_DIM * T_STAR**2 / L_STAR_M

# Celestial body radii (for eclipse geometry)
R_SUN_KM   = 696_340.0     # km
R_EARTH_KM = 6_371.0       # km

print(f"[init] μ = {MU:.10e}")
print(f"[init] L* = {L_STAR:.3e} km,  T* = {T_STAR:.6e} s  "
      f"({TU_TO_DAYS:.4f} days)")
print(f"[init] V* = {V_STAR_KM:.6f} km/s")
print(f"[init] SRP accel (dimensional) = {A_SRP_DIM:.4e} m/s²")
print(f"[init] SRP accel (normalized)  = {A_SRP_NORM:.4e}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# §2  L1 POSITION — Newton–Raphson on quintic equation
# ══════════════════════════════════════════════════════════════════════════════

def solve_L1(mu: float, tol: float = 1e-15, maxiter: int = 200) -> float:
    """
    Solve for the collinear L1 Lagrange point in the Sun–Earth CR3BP.

    The x-coordinate satisfies a quintic polynomial derived from setting
    the gradient of the pseudo-potential to zero on the x-axis (y = z = 0).

    Uses Newton–Raphson iteration.

    Returns
    -------
    x_L1 : float   (normalized CR3BP coordinate, Sun at -μ, Earth at 1-μ)
    """
    # L1 lies between Sun and smaller primary (Earth).
    # Let γ = 1 - μ - x_L1  (distance from L1 to Earth, positive toward Sun).
    # The quintic for γ:
    #   γ⁵ - (3-μ)γ⁴ + (3-2μ)γ³ - μγ² + 2μγ - μ = 0
    # We use Newton–Raphson in x directly.

    # Initial guess (Hill's approximation)
    gamma0 = (mu / 3.0) ** (1.0 / 3.0)
    x0 = 1.0 - mu - gamma0

    x = x0
    for _ in range(maxiter):
        r1 = x + mu           # distance to Sun (at -μ)
        r2 = x - (1.0 - mu)   # signed distance to Earth (at 1-μ)
        # f  = x - (1-μ)/r1² * sign + μ/r2² * sign  (equilibrium condition)
        f  = x - (1.0 - mu) / r1**2 * np.sign(r1) + mu / r2**2 * np.sign(r2)
        # Actually, the full expression on the x-axis:
        # ẍ = 0: x - (1-μ)(x+μ)/|x+μ|³ - μ(x-1+μ)/|x-1+μ|³ = 0
        r1abs = abs(r1)
        r2abs = abs(r2)
        f = x - (1.0 - mu) * (x + mu) / r1abs**3 - mu * (x - 1.0 + mu) / r2abs**3
        fp = 1.0 - (1.0 - mu) / r1abs**3 + 3.0 * (1.0 - mu) * (x + mu)**2 / r1abs**5 \
             - mu / r2abs**3 + 3.0 * mu * (x - 1.0 + mu)**2 / r2abs**5
        dx = -f / fp
        x += dx
        if abs(dx) < tol:
            break
    return x


X_L1 = solve_L1(MU)
GAMMA_L1 = 1.0 - MU - X_L1    # distance from L1 to Earth

print(f"[L1] x_L1  = {X_L1:.15f}  (normalized)")
print(f"[L1] γ_L1  = {GAMMA_L1:.15e}")
print(f"[L1] L1–Earth distance = {GAMMA_L1 * L_STAR:,.1f} km")
print(f"[L1] L1–Sun  distance  = {(X_L1 + MU) * L_STAR:,.1f} km")
print()

# ══════════════════════════════════════════════════════════════════════════════
# §3  RICHARDSON 3RD ORDER HALO ORBIT INITIAL CONDITIONS  [1]
# ══════════════════════════════════════════════════════════════════════════════

def richardson_halo_ic(
    mu: float,
    gamma: float,
    x_L1: float,
    Az_km: float,
    L_star: float = L_STAR,
    m: int = 1,
) -> dict:
    """
    Compute 3rd-order Richardson (1980) analytical initial conditions for a
    Northern (m = +1) or Southern (m = -1) halo orbit about L1.

    Parameters
    ----------
    mu     : CR3BP mass parameter
    gamma  : L1 distance to second primary (Earth)
    x_L1   : L1 x-coordinate (normalized)
    Az_km  : desired out-of-plane amplitude [km]
    L_star : dimensional length unit [km]
    m      : +1 = Northern halo, -1 = Southern halo

    Returns
    -------
    dict with keys: state_norm (6,), state_dim_km (6,), Ax_km, Ay_km, Az_km,
                    period_norm, period_days, omega_p, c_coeffs ...

    Reference
    ---------
    [1] Richardson (1980), Celestial Mechanics 22(3), 241–253, Eqs. 4-14.
    """

    # --- Legendre polynomial coefficients c_n  [1, Eq. 4] ---
    # c_n = (1/γ³) [ (±1)^n μ + (-1)^n (1-μ) γ^(n+1) / (1 ∓ γ)^(n+1) ]
    # For L1 (between bodies), sign convention:
    #   c_n = (1/γ³)[ μ + (-1)^n (1-μ) γ^(n+1) / (1 - γ)^(n+1) ]
    #   (using d₁ = 1-μ-x_L1 = γ, and the Sun is at distance d_sun = x_L1+μ ≈ 1-γ)
    def cn(n):
        return (1.0 / gamma**3) * (
            mu + (-1)**n * (1.0 - mu) * gamma**(n + 1) / (1.0 - gamma)**(n + 1)
        )

    c2 = cn(2)
    c3 = cn(3)
    c4 = cn(4)

    # --- Characteristic frequencies  [1, Eqs. 6–8] ---
    # λ² + (c₂ - 2)λ² - (c₂ - 1)(1 + 2c₂) = 0   ← eigenvalue equation
    # Solving: λ_p (in-plane frequency)
    # Discriminant of λ²:
    poly_b = c2 - 2.0
    poly_c = -(c2 - 1.0) * (1.0 + 2.0 * c2)
    disc = poly_b**2 - 4.0 * poly_c
    lam2 = (-poly_b + sqrt(disc)) / 2.0      # positive root
    lam_p = sqrt(lam2)                        # planar eigenvalue (real, positive)

    # k, δ from [1]
    k = (lam2 + 1.0 + 2.0 * c2) / (2.0 * lam_p)   # Eq. (7)

    # Out-of-plane frequency  ν
    # ν² = c₂   → ν = √c₂   (for L1 halo, to leading order)
    # Actually [1] has  ν² = c₂  only to zeroth order; the full expression:
    nu  = sqrt(c2)    # vertical frequency (zeroth order)

    # Δ = ν² - c₂ = 0 to zeroth order  (used in higher-order terms)
    delta_n = nu**2 - c2  # should ≈ 0

    # --- 2nd and 3rd order coefficients  [1, Table 1 / Eqs. 10-14] ---
    d1 = 3.0 * lam_p**2 / k * (k * (6.0 * lam_p**2 - 1.0) - 2.0 * lam_p)
    d2 = 8.0 * lam_p**2 / k * (k * (11.0 * lam_p**2 - 1.0) - 2.0 * lam_p)

    a21 = 3.0 * c3 * (k**2 - 2.0) / (4.0 * (1.0 + 2.0 * c2))
    a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2))
    a23 = -3.0 * c3 * lam_p / (4.0 * k * d1) * (
        3.0 * k**3 * lam_p - 6.0 * k * (k - lam_p) + 4.0
    )
    a24 = -3.0 * c3 * lam_p / (4.0 * k * d1) * (
        2.0 + 3.0 * k * lam_p
    )

    b21 = -3.0 * c3 * lam_p / (2.0 * d1) * (3.0 * k * lam_p - 4.0)
    b22 = 3.0 * c3 * lam_p / d1

    d21 = -c3 / (2.0 * lam_p**2)

    a31 = -9.0 * lam_p / (4.0 * d2) * (
        4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k**2)
    ) + (9.0 * lam_p**2 + 1.0 - c2) / (2.0 * d2) * (
        3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * k**2)
    )

    a32 = -1.0 / d2 * (
        9.0 * lam_p / 4.0 * (4.0 * c3 * (k * a24 - b22) + k * c4)
        + 3.0 / 2.0 * (9.0 * lam_p**2 + 1.0 - c2) * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
    )

    b31 = 3.0 / (8.0 * d2) * (
        8.0 * lam_p * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * k**2))
        + (9.0 * lam_p**2 + 1.0 + 2.0 * c2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k**2))
    )

    b32 = 1.0 / d2 * (
        9.0 * lam_p * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
        + 3.0 / 8.0 * (9.0 * lam_p**2 + 1.0 + 2.0 * c2) * (
            4.0 * c3 * (k * a24 - b22) + k * c4
        )
    )

    d31 = 3.0 / (64.0 * lam_p**2) * (
        4.0 * c3 * a24 + c4
    )

    d32 = 3.0 / (64.0 * lam_p**2) * (
        4.0 * c3 * (a23 - d21) + c4 * (4.0 + k**2)
    )

    # --- Frequency correction (s1, s2)  [1, Eq. 14] ---
    s1 = (
        3.0 / 2.0 * c3 * (2.0 * a21 * (k**2 - 2.0) - a23 * (k**2 + 2.0) - 2.0 * k * b21)
        - 3.0 / 8.0 * c4 * (3.0 * k**4 - 8.0 * k**2 + 8.0)
    ) / (2.0 * lam_p * (lam_p * (1.0 + k**2) - 2.0 * k))

    s2 = (
        3.0 / 2.0 * c3 * (2.0 * a22 * (k**2 - 2.0) + a24 * (k**2 + 2.0) + 2.0 * k * b22 + 5.0 * d21)
        + 3.0 / 8.0 * c4 * (12.0 - k**2)
    ) / (2.0 * lam_p * (lam_p * (1.0 + k**2) - 2.0 * k))

    # --- Amplitude constraint  [1]: l1 Ax² + l2 Az² + Δ = 0  ---
    l1 = -3.0 / 2.0 * c3 * (2.0 * a21 + a23 + 5.0 * d21) - 3.0 / 8.0 * c4 * (12.0 - k**2) + 2.0 * lam_p**2 * s1
    l2 =  3.0 / 2.0 * c3 * (a24 - 2.0 * a22) + 9.0 / 8.0 * c4 + 2.0 * lam_p**2 * s2

    # Normalize amplitudes
    Az_norm = Az_km / (gamma * L_star)

    # From constraint l1 Ax² + l2 Az² + Δ = 0, Δ ≈ 0:
    # Ax² = -(l2 Az² + Δ) / l1   [1, Eq. 14]
    Ax_norm2 = -(l2 * Az_norm**2 + delta_n) / l1
    if Ax_norm2 < 0:
        raise ValueError(
            f"No real Ax for Az = {Az_km:.0f} km (Ax² = {Ax_norm2:.4e}). "
            "Requested amplitude is outside the Richardson validity envelope."
        )
    Ax_norm = sqrt(Ax_norm2)

    # Corrected frequency
    omega1 = 0.0  # 1st order correction is zero for halo orbits
    omega2 = s1 * Ax_norm**2 + s2 * Az_norm**2
    omega = 1.0 + omega2   # ω = 1 + ω₂ Ax² + ...

    # Period in normalized time (relative to γ-scaled time τ = lam_p * t)
    period_tau = 2.0 * pi / (lam_p * omega)

    # Dimensional amplitudes
    Ax_km_out = Ax_norm * gamma * L_star
    Ay_norm = k * Ax_norm
    Ay_km_out = Ay_norm * gamma * L_star

    # --- Initial state (τ = 0, φ = 0)  [1, Eqs. 9] ---
    # At t = 0 with phase angle φ = 0:
    #   x = a21 Ax² + a22 Az² - Ax cos(0) + (a23 Ax² - a24 Az²) cos(2·0)
    #       + (a31 Ax³ - a32 Ax Az²) cos(3·0)
    #   y = 0
    #   z = m δₙ Az [1 + 2 d21 Ax cos(0) + ...]   → δₙ ≈ 1 here (sign)
    #   ẋ = 0
    #   ẏ = ...
    #   ż = 0

    # φ = 0, ψ = φ + phase_diff;  for Northern halo: ψ = π/2 when φ = 0
    # Convention: at t = 0 → cos(τ₁·0) = 1 (x displacement max)
    # Then y should = 0 (phase set so sin(0) = 0).

    tau1 = lam_p * omega  # scaled frequency

    # x(0)  [Richardson notation: ξ in γ-scaled coords, origin at L1]:
    x_local = (a21 * Ax_norm**2 + a22 * Az_norm**2
               - Ax_norm
               + (a23 * Ax_norm**2 - a24 * Az_norm**2)
               + (a31 * Ax_norm**3 - a32 * Ax_norm * Az_norm**2))

    # y(0) = 0 by phase choice

    # z(0):
    z_local = Az_norm * (1.0 + 2.0 * d21 * Ax_norm + (d32 * Az_norm**2 + d31 * Ax_norm**2))
    z_local *= m  # m = +1 for Northern

    # ẋ(0) = 0 by phase choice

    # ẏ(0):
    # dy/dt at τ=0:  (k Ax + 2(b21 Ax² - b22 Az²) + 3(b31 Ax³ - b32 Ax Az²))
    # multiplied by tau1 (chain rule)
    ydot_local = (k * Ax_norm
                  + 2.0 * (b21 * Ax_norm**2 - b22 * Az_norm**2)
                  + 3.0 * (b31 * Ax_norm**3 - b32 * Ax_norm * Az_norm**2))
    ydot_local *= tau1

    # ż(0) = 0 by phase choice

    # Convert from γ-scaled local coords (origin at L1) to CR3BP rotating frame:
    # X_CR3BP = x_L1 + γ * x_local   (x_local is measured from L1, positive toward Earth)
    # BUT Richardson uses ξ positive AWAY from the closer primary (i.e., toward Sun for L1).
    # So X_CR3BP = x_L1 - γ * x_local   (minus sign: toward Sun is -x direction)
    # Wait — actually Richardson defines ξ toward the SMALLER body for L1.
    # For L1 between Sun and Earth: ξ > 0 points toward Earth (i.e., +x direction
    # since Earth is at 1-μ and L1 < 1-μ).
    # Therefore: X_CR3BP = x_L1 + γ * x_local  ... but conventionally people write
    # X = x_L1 ± γ ξ depending on convention.  We must be careful.
    #
    # Richardson's convention: the local coordinate ξ is positive toward the
    # nearer primary.  For L1, nearer primary is Earth → ξ > 0 toward Earth.
    # Since Earth is at x = 1-μ > x_L1, this means ξ > 0 → X increases.
    #
    # Actually, the standard Richardson convention for L1:
    #   X = x_L1 + γ·ξ   (ξ positive toward Earth)
    #   but in his formulation Ax appears with a MINUS sign in the x-expansion,
    #   so the orbit displacement at τ₁ t = 0 is x_local < 0, meaning the
    #   spacecraft is displaced toward the Sun.
    #
    # Let's just use the standard mapping:
    # In the rotating frame centered at L1, scaled by γ:
    #   X_rot = x_L1 + γ * ξ   (ξ = x_local)
    #   Y_rot = γ * η           (η from y equation)
    #   Z_rot = γ * ζ

    x0_norm = x_L1 + gamma * x_local
    y0_norm = 0.0
    z0_norm = gamma * z_local

    # Velocities: dX/dt = γ * dξ/dt,  dξ/dt already in normalized time
    vx0_norm = 0.0
    vy0_norm = gamma * ydot_local
    vz0_norm = 0.0

    state_norm = np.array([x0_norm, y0_norm, z0_norm, vx0_norm, vy0_norm, vz0_norm])

    # Dimensional
    state_dim_km = np.array([
        x0_norm * L_star,
        0.0,
        z0_norm * L_star,
        0.0,
        vy0_norm * V_STAR_KM,
        0.0,
    ])

    return {
        "state_norm": state_norm,
        "state_dim_km": state_dim_km,
        "Ax_km": Ax_km_out,
        "Ay_km": Ay_km_out,
        "Az_km": Az_km,
        "Ax_norm": Ax_norm,
        "Ay_norm": Ay_norm,
        "Az_norm": Az_norm,
        "period_norm": period_tau,
        "period_days": period_tau * TU_TO_DAYS,
        "omega_p": lam_p,
        "omega": omega,
        "tau1": tau1,
        "c2": c2, "c3": c3, "c4": c4,
        "k": k,
        "delta_n": delta_n,
        "gamma": gamma,
        "m_halo": m,
        "s1": s1, "s2": s2,
        "l1": l1, "l2": l2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §4  CR3BP + SRP EQUATIONS OF MOTION
# ══════════════════════════════════════════════════════════════════════════════

def cr3bp_eom(t: float, state: np.ndarray, mu: float,
              srp: bool = False, a_srp: float = 0.0) -> np.ndarray:
    """
    Equations of motion in the Sun–Earth Circular Restricted 3-Body Problem,
    optionally including Solar Radiation Pressure (SRP).

    State vector: [x, y, z, vx, vy, vz]  (normalized rotating frame).

    The pseudo-potential is
        Ω = ½(x² + y²) + (1-μ)/r₁ + μ/r₂

    SRP acts radially away from the Sun (located at x = -μ).

    Parameters
    ----------
    t       : time (not used explicitly — autonomous system)
    state   : (6,) state vector [x, y, z, ẋ, ẏ, ż]
    mu      : mass parameter
    srp     : whether to include SRP perturbation
    a_srp   : magnitude of SRP acceleration [normalized]

    Returns
    -------
    dstate  : (6,) time-derivative of state
    """
    x, y, z, vx, vy, vz = state

    # Distances to primaries
    r1 = sqrt((x + mu)**2 + y**2 + z**2)          # distance to Sun (at -μ, 0, 0)
    r2 = sqrt((x - 1.0 + mu)**2 + y**2 + z**2)    # distance to Earth (at 1-μ, 0, 0)

    # Accelerations from pseudo-potential ∂Ω/∂q
    ax = (2.0 * vy + x
          - (1.0 - mu) * (x + mu) / r1**3
          - mu * (x - 1.0 + mu) / r2**3)

    ay = (-2.0 * vx + y
          - (1.0 - mu) * y / r1**3
          - mu * y / r2**3)

    az = (-(1.0 - mu) * z / r1**3
          - mu * z / r2**3)

    # SRP: radially away from Sun (at x = -μ)
    if srp and a_srp > 0:
        # Unit vector from Sun to spacecraft
        dx_sun = x + mu
        dy_sun = y
        dz_sun = z
        r_sun = sqrt(dx_sun**2 + dy_sun**2 + dz_sun**2)
        # a_SRP is constant at 1 AU; for accurate modeling we could scale
        # with 1/r², but near L1 the distance to Sun ≈ 1 AU, so the correction
        # is < 1%.  We keep it constant for this analysis.
        ax += a_srp * dx_sun / r_sun
        ay += a_srp * dy_sun / r_sun
        az += a_srp * dz_sun / r_sun

    return np.array([vx, vy, vz, ax, ay, az])


def cr3bp_eom_stm(t: float, state_stm: np.ndarray, mu: float) -> np.ndarray:
    """
    CR3BP equations of motion augmented with the State Transition Matrix (STM)
    variational equations (no SRP, for stability analysis).

    state_stm : (42,) = [x, y, z, vx, vy, vz, Φ₁₁, Φ₁₂, ..., Φ₆₆]

    The STM satisfies  Φ̇ = A(t) Φ,  where A is the Jacobian of the EOM.

    Reference: [2] Howell (1984).
    """
    s = state_stm[:6]
    phi = state_stm[6:].reshape((6, 6))

    x, y, z, vx, vy, vz = s
    r1 = sqrt((x + mu)**2 + y**2 + z**2)
    r2 = sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    # Partial derivatives of the pseudo-potential (Ω)
    # Ωxx, Ωyy, Ωzz, Ωxy, Ωxz, Ωyz
    r1_3 = r1**3
    r1_5 = r1**5
    r2_3 = r2**3
    r2_5 = r2**5

    Uxx = (1.0
           - (1.0 - mu) / r1_3 + 3.0 * (1.0 - mu) * (x + mu)**2 / r1_5
           - mu / r2_3 + 3.0 * mu * (x - 1.0 + mu)**2 / r2_5)

    Uyy = (1.0
           - (1.0 - mu) / r1_3 + 3.0 * (1.0 - mu) * y**2 / r1_5
           - mu / r2_3 + 3.0 * mu * y**2 / r2_5)

    Uzz = (-(1.0 - mu) / r1_3 + 3.0 * (1.0 - mu) * z**2 / r1_5
           - mu / r2_3 + 3.0 * mu * z**2 / r2_5)

    Uxy = (3.0 * (1.0 - mu) * (x + mu) * y / r1_5
           + 3.0 * mu * (x - 1.0 + mu) * y / r2_5)

    Uxz = (3.0 * (1.0 - mu) * (x + mu) * z / r1_5
           + 3.0 * mu * (x - 1.0 + mu) * z / r2_5)

    Uyz = (3.0 * (1.0 - mu) * y * z / r1_5
           + 3.0 * mu * y * z / r2_5)

    # Jacobian A(t) of the CR3BP EOM
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 0] = Uxx;   A[3, 1] = Uxy;   A[3, 2] = Uxz
    A[3, 4] = 2.0
    A[4, 0] = Uxy;   A[4, 1] = Uyy;   A[4, 2] = Uyz
    A[4, 3] = -2.0
    A[5, 0] = Uxz;   A[5, 1] = Uyz;   A[5, 2] = Uzz

    # STM derivative
    dphi = A @ phi

    ds = cr3bp_eom(t, s, mu, srp=False)

    return np.concatenate([ds, dphi.flatten()])


def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """
    Compute the Jacobi constant C for a given state in the CR3BP rotating frame.

    C = 2Ω - v²  where Ω is the pseudo-potential and v is the velocity magnitude.
    """
    x, y, z, vx, vy, vz = state
    r1 = sqrt((x + mu)**2 + y**2 + z**2)
    r2 = sqrt((x - 1.0 + mu)**2 + y**2 + z**2)
    Omega = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2
    v2 = vx**2 + vy**2 + vz**2
    return 2.0 * Omega - v2


# ══════════════════════════════════════════════════════════════════════════════
# §5  PROPAGATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def propagate(state0: np.ndarray, t_span: Tuple[float, float],
              mu: float, srp: bool = False, a_srp: float = 0.0,
              n_eval: int = 8000,
              rtol: float = 1e-12, atol: float = 1e-14) -> object:
    """
    Propagate CR3BP trajectory with scipy RK45.

    Returns scipy OdeResult with dense output.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(
        cr3bp_eom, t_span, state0,
        args=(mu, srp, a_srp),
        method="RK45",
        rtol=rtol, atol=atol,
        t_eval=t_eval,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol


def propagate_stm(state0: np.ndarray, t_span: Tuple[float, float],
                  mu: float, n_eval: int = 4000) -> object:
    """
    Propagate CR3BP trajectory + STM (variational equations) for stability
    analysis.  No SRP included (linear stability of natural orbit).

    Reference: [2] Howell (1984).
    """
    phi0 = np.eye(6).flatten()
    y0 = np.concatenate([state0, phi0])
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(
        cr3bp_eom_stm, t_span, y0,
        args=(mu,),
        method="RK45",
        rtol=1e-12, atol=1e-14,
        t_eval=t_eval,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"STM integration failed: {sol.message}")
    return sol


def find_period(state0: np.ndarray, T_guess: float, mu: float,
                tol: float = 1e-8) -> Tuple[np.ndarray, float]:
    """
    Find a periodic halo orbit near *state0* using single-shooting
    differential correction (Howell 1984 [2]).

    Strategy
    --------
    1. Propagate the state (with STM) until the first y = 0 crossing after
       a significant fraction of the expected half-period → half-period T/2.
    2. At y = 0, target vx(T/2) = 0 by adjusting vy0 (keep x0, z0 fixed to
       preserve the desired amplitude).
    3. Iterate Newton-style until |vx(T/2)| < tol.
    4. Full period T = 2 × T_half.

    For the simpler 1-variable correction (fixing x0 and z0, varying vy0),
    the update formula is:

        δvy0 = -vx_f / (∂vx_f/∂vy0 − ax_f · ∂y_f/∂vy0 / vy_f)

    (accounting for the variable integration time to the y = 0 surface).

    Returns
    -------
    state_corr : (6,) corrected initial state
    T_period   : float  full period in normalized TU
    """
    state = state0.copy()
    T_half_guess = T_guess / 2.0
    t_half = T_half_guess          # fallback in case loop never sets it
    xf = state.copy()              # fallback
    maxiter = 50

    for iteration in range(maxiter):
        # ── Propagate state + STM until y = 0 crossing ──
        phi0 = np.eye(6).flatten()
        y0_aug = np.concatenate([state, phi0])

        # We use a non-terminal event, then pick the first crossing
        # after a minimum time threshold (avoids t ≈ 0 false trigger).
        def y_event(t, s, *args):
            return s[1]   # y = 0

        y_event.terminal = False
        y_event.direction = 0   # detect all crossings

        sol = solve_ivp(
            cr3bp_eom_stm, [0, T_guess * 1.2], y0_aug,
            args=(mu,),
            method="RK45", rtol=1e-13, atol=1e-15,
            events=y_event, dense_output=True,
        )

        # Filter crossings: skip t < 20% of expected half-period
        t_min = 0.2 * T_half_guess
        valid = [tc for tc in sol.t_events[0] if tc > t_min]

        if len(valid) == 0:
            print(f"[DC] Warning: no y=0 crossing found at iter {iteration}")
            break

        # Pick the first valid crossing (should be near T/2)
        t_half = valid[0]
        sf = sol.sol(t_half)
        xf = sf[:6]
        phi = sf[6:].reshape((6, 6))

        vx_f = xf[3]
        vz_f = xf[5]

        # Check convergence
        if abs(vx_f) < tol and abs(vz_f) < tol:
            break

        # Accelerations at the crossing point (needed for variable-time correction)
        dsdt = cr3bp_eom(t_half, xf, mu, srp=False)
        vy_f  = xf[4]
        ax_f  = dsdt[3]  # ẍ
        az_f  = dsdt[5]  # z̈

        if abs(vy_f) < 1e-15:
            print("[DC] Warning: vy at half-period ≈ 0, no correction possible.")
            break

        # ── 2-variable correction: vary x0 and vy0 to target vxf = 0, vzf = 0 ──
        # Partial derivatives corrected for variable endpoint:
        # J[i,j] = Φ(row_i, col_j) − (a_i / vy_f) · Φ(2, col_j)
        # where row_i corresponds to vx (row 4→index 3) or vz (row 6→index 5)
        # and col_j corresponds to x0 (col 1→index 0) or vy0 (col 5→index 4)

        J = np.zeros((2, 2))
        J[0, 0] = phi[3, 0] - (ax_f / vy_f) * phi[1, 0]   # ∂vxf/∂x0
        J[0, 1] = phi[3, 4] - (ax_f / vy_f) * phi[1, 4]   # ∂vxf/∂vy0
        J[1, 0] = phi[5, 0] - (az_f / vy_f) * phi[1, 0]   # ∂vzf/∂x0
        J[1, 1] = phi[5, 4] - (az_f / vy_f) * phi[1, 4]   # ∂vzf/∂vy0

        rhs = np.array([-vx_f, -vz_f])

        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            print("[DC] Warning: singular Jacobian, stopping correction.")
            break

        # Apply correction (with damping for robustness)
        alpha = 1.0
        state[0] += alpha * delta[0]   # update x0
        state[4] += alpha * delta[1]   # update vy0

    T_period = 2.0 * t_half

    vx_res = abs(xf[3])
    vz_res = abs(xf[5])
    converged = vx_res < tol and vz_res < tol
    status = "Converged" if converged else "NOT converged"
    print(f"  [DC] {status} in {iteration + 1} iterations: "
          f"|vx(T/2)| = {vx_res:.2e}, |vz(T/2)| = {vz_res:.2e}")

    if not converged:
        raise RuntimeError(
            f"DC did not converge: |vx|={vx_res:.2e}, |vz|={vz_res:.2e}"
        )

    return state, T_period


def find_halo_with_target_az(
    state0_rich: np.ndarray,
    T_guess: float,
    mu: float,
    az_target_km: float,
    L_star: float = L_STAR,
    max_iter: int = 15,
    az_tol: float = 500.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Find a periodic halo orbit whose out-of-plane (Az) amplitude matches
    *az_target_km* within *az_tol*.

    Strategy: run differential correction (find_period) iteratively while
    scaling z₀ so the measured z-peak of the periodic orbit converges to
    the desired value.

    Returns
    -------
    state_corr : (6,) corrected IC for the periodic halo orbit
    T_period   : float  period in normalized TU
    az_actual  : float  measured Az amplitude [km]
    """
    state = state0_rich.copy()
    state_corr = state.copy()
    T_period = T_guess
    az_actual = az_target_km  # init

    for az_iter in range(max_iter):
        state_corr, T_period = find_period(state, T_period, mu, tol=1e-10)

        # Measure actual Az from propagation
        sol = propagate(state_corr, (0, T_period), mu, srp=False, n_eval=4000)
        z_arr = sol.y[2] * L_star
        az_actual = (np.max(z_arr) - np.min(z_arr)) / 2.0

        err = abs(az_actual - az_target_km)
        print(f"    [Az iter {az_iter}] Az_actual = {az_actual:,.0f} km "
              f"(target = {az_target_km:,.0f} km, err = {err:,.0f} km)")

        if err < az_tol:
            break

        # Scale z0 for next iteration
        scale = az_target_km / az_actual
        state = state_corr.copy()
        state[2] *= scale                   # adjust z0
        # keep x0 and vy0 from the last DC convergence

    return state_corr, T_period, az_actual


# ══════════════════════════════════════════════════════════════════════════════
# §6  SEZ (Solar Exclusion Zone) ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sez_analysis(sol, mu: float) -> dict:
    """
    Compute angular separation between spacecraft and Sun–Earth line
    at each time step.

    θ(t) = arctan( √(y² + z²) / |x − x_Sun| )

    where x_Sun = −μ in the rotating frame.

    Returns dict with theta (degrees), violation fractions for 3°, 5°, 10°.
    """
    x, y, z = sol.y[0], sol.y[1], sol.y[2]

    # Angular separation from Sun–Earth line
    # Sun is at (-μ, 0, 0).  Earth is at (1-μ, 0, 0).
    # The Sun–Earth line is the x-axis in the rotating frame.
    # Angular separation of spacecraft from the Sun–Earth line as seen from Earth:
    dx = x - (1.0 - mu)   # x relative to Earth
    rho = np.sqrt(y**2 + z**2)  # perpendicular distance from x-axis
    r_sc_earth = np.sqrt(dx**2 + y**2 + z**2)

    # Angle as seen from Earth
    theta_deg = np.degrees(np.arctan2(rho, np.abs(dx)))

    # For SEZ: the relevant angle is the angular separation of the spacecraft
    # from the Sun as seen from Earth.
    # Sun direction from Earth: toward (-μ - (1-μ), 0, 0) = (-1, 0, 0)
    # S/C direction from Earth: (x - (1-μ), y, z)
    # cos(angle) = dot / (|r_1| |r_2|)
    sun_dir = np.array([-1.0, 0.0, 0.0])  # Sun direction (unit) from Earth
    cos_angle = (dx * (-1.0)) / (r_sc_earth)  # simplified since sun_dir is unit x
    # Actually cos_angle = -dx / r_sc_earth
    cos_angle = -dx / r_sc_earth
    theta_sez = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # Violations
    violations = {}
    for cone in [3.0, 5.0, 10.0]:
        frac = np.sum(theta_sez < cone) / len(theta_sez) * 100.0
        violations[cone] = frac

    min_sep = np.min(theta_sez)

    return {
        "theta_sez": theta_sez,
        "time_days": sol.t * TU_TO_DAYS,
        "violations": violations,
        "min_separation": min_sep,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §7  ECLIPSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def eclipse_analysis(sol, mu: float) -> dict:
    """
    Check whether the spacecraft enters Earth's umbra or penumbra shadow cone.

    Geometry:
    - Earth blocks sunlight → shadow extends behind Earth (in +x direction
      in the rotating frame, since Sun is at -μ).
    - L1 is between Sun and Earth → spacecraft never enters Earth shadow.
    - Verify quantitatively.

    Umbra half-angle:
        α_umbra = arcsin(R_Sun / d_Sun) - arcsin(R_Earth / d_Earth)
        (where d_Sun, d_Earth are distances from s/c to Sun/Earth centers)

    For L1 halo orbits, the spacecraft is between Sun and Earth, so the
    Earth shadow is cast in the opposite direction.  Eclipse fraction
    should be exactly 0%.
    """
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    n = len(x)

    in_umbra = np.zeros(n, dtype=bool)
    in_penumbra = np.zeros(n, dtype=bool)

    for i in range(n):
        # Spacecraft position (dimensional km for geometry)
        xs = x[i] * L_STAR
        ys = y[i] * L_STAR
        zs = z[i] * L_STAR

        # Earth at (1-μ)*L_STAR, 0, 0
        xe = (1.0 - mu) * L_STAR
        # Sun at -μ*L_STAR, 0, 0
        x_sun = -mu * L_STAR

        # For s/c to be in Earth's shadow, it must be on the anti-Sun side of Earth
        # i.e., x_sc > x_Earth (farther from Sun than Earth).
        # For L1 orbits, x_sc < x_Earth, so no eclipse is possible.
        if xs > xe:
            # Check shadow cone
            d_earth = sqrt((xs - xe)**2 + ys**2 + zs**2)
            # Perpendicular distance from Earth-Sun line
            rho = sqrt(ys**2 + zs**2)
            # Shadow cone:  r_shadow(d) = R_Earth - d * (R_Earth - R_Sun) / d_SE
            # (this is a linear approximation)
            d_SE = xe - x_sun   # Earth-Sun distance
            # Distance along shadow axis from Earth
            d_along = xs - xe
            r_umbra = R_EARTH_KM * (1.0 - d_along * R_SUN_KM / (d_SE * R_EARTH_KM))
            r_penumbra = R_EARTH_KM + d_along * (R_SUN_KM - R_EARTH_KM) / d_SE

            if rho < r_umbra and r_umbra > 0:
                in_umbra[i] = True
            if rho < r_penumbra:
                in_penumbra[i] = True

    frac_umbra   = np.sum(in_umbra)   / n * 100.0
    frac_penumbra = np.sum(in_penumbra) / n * 100.0

    return {
        "frac_umbra": frac_umbra,
        "frac_penumbra": frac_penumbra,
        "in_umbra": in_umbra,
        "in_penumbra": in_penumbra,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §8  STABILITY ANALYSIS (Monodromy Matrix & Eigenvalues)
# ══════════════════════════════════════════════════════════════════════════════

def stability_analysis(state0: np.ndarray, T_period: float, mu: float) -> dict:
    """
    Propagate State Transition Matrix (STM) for one period to obtain the
    monodromy matrix M = Φ(T, 0).

    Extract eigenvalues → stability index ν = (|λ_u| + 1/|λ_u|) / 2.
    Interpret: ν > 1 → orbit is unstable.

    Characteristic divergence timescale: τ_div = T / ln(|λ_unstable|).

    Reference: [2] Howell (1984).
    """
    sol = propagate_stm(state0, (0, T_period), mu, n_eval=4000)

    # Extract monodromy matrix
    phi_final = sol.y[6:, -1].reshape((6, 6))

    eigenvalues = np.linalg.eigvals(phi_final)

    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]

    # The most unstable eigenvalue (largest magnitude)
    lam_u = eigenvalues[0]
    lam_u_mag = np.abs(lam_u)

    # Stability index
    nu = (lam_u_mag + 1.0 / lam_u_mag) / 2.0

    # Divergence timescale
    tau_div_norm = T_period / np.log(lam_u_mag)  # normalized time units
    tau_div_days = tau_div_norm * TU_TO_DAYS

    return {
        "monodromy": phi_final,
        "eigenvalues": eigenvalues,
        "lambda_unstable": lam_u,
        "lambda_u_mag": lam_u_mag,
        "stability_index": nu,
        "diverge_time_norm": tau_div_norm,
        "diverge_time_days": tau_div_days,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §9  DELTA-V & PROPELLANT BUDGET  [3,4,6]
# ══════════════════════════════════════════════════════════════════════════════

def delta_v_budget(m_dry: float = 905.0) -> dict:
    """
    Compute ΔV allocations and propellant masses for L1 and L4/L5 nodes.

    L1 station-keeping: Floquet mode targeting [3], empirical ~ 2–5 m/s/yr,
      Genesis actual 3.8 m/s/yr [4].  Baseline: 4.0 m/s/yr.

    Propellant via Tsiolkovsky [6]:  m_prop = m_dry * (exp(ΔV / (Isp g₀)) - 1)
    """
    g0 = 9.80665  # m/s²

    # --- L1 Node ---
    sk_rate = 4.0           # m/s/yr  (station-keeping rate)
    sk_10yr = sk_rate * 10  # m/s
    sk_12yr = sk_rate * 12  # m/s
    sk_margin = sk_12yr * 1.2   # 20% margin → 57.6 m/s
    sk_budget = 60.0            # rounded UP to 60 m/s (conservative allocation)

    # L1 HOI via stable manifold approach [5]
    # ASSUMPTION: launch vehicle (e.g., Falcon 9) delivers spacecraft to C3 ≈ 0
    # targeting the stable manifold inbound to L1.  LEO→L1 cruise ΔV is the
    # launch vehicle's responsibility and is NOT included here.
    # Genesis: <6 m/s deterministic (Koon et al., Automatica 38, 2002)
    # Budget: manifold inj ~10 + MCC ~20 + final HOI ~10 + margin ~10
    #
    # LAUNCH-DATE DEPENDENCY: This 50 m/s allocation is a launch-date-independent
    # feasibility estimate. The actual HOI ΔV will vary with:
    #   (a) Launch date — manifold tube phase angle changes with Sun-Earth-Moon
    #       geometry over a ~178-day halo orbit period.
    #   (b) C3 value — different launch windows yield different hyperbolic excess
    #       energies; residual ΔV must be absorbed by the spacecraft.
    #   (c) Transfer time — longer cruise = different manifold attachment point
    #       and different final insertion ΔV.
    #   (d) Navigation errors — TCM budget is launch-window-specific.
    # A dedicated launch-window trajectory study (STK or Monte Carlo) is
    # required to refine this estimate to PDR/CDR level.
    hoi_margin = 50.0       # m/s  (stable manifold insertion, post-delivery)

    contingency = 15.0      # m/s  (attitude + contingency)

    total_L1 = hoi_margin + sk_budget + contingency

    # Propellant: chemical bipropellant for HOI  (Isp = 310 s)
    Isp_chem = 310.0    # s
    m_prop_hoi = m_dry * (np.exp(hoi_margin / (Isp_chem * g0)) - 1.0)

    # Propellant: electric ion thruster for SK  (Isp = 2500 s, xenon)
    Isp_ion = 2500.0    # s
    m_prop_sk_10 = m_dry * (np.exp(sk_10yr / (Isp_ion * g0)) - 1.0)
    m_prop_sk_12 = m_dry * (np.exp(sk_12yr / (Isp_ion * g0)) - 1.0)

    # Total xenon for SK (12 yr with margin)
    m_prop_sk_budget = m_dry * (np.exp(sk_budget / (Isp_ion * g0)) - 1.0)

    # --- L4/L5 Nodes ---
    # ASSUMPTION: same launch vehicle delivery philosophy as L1 — spacecraft
    # delivered via low-energy ballistic transfer to Earth's Trojan region
    # (C3 ≈ 0, correct 60° phasing). LEO→L4/L5 cruise ΔV is NOT included.
    # The 50 m/s covers final orbit insertion + phasing corrections only.
    # This is a feasibility-level estimate; a dedicated transfer trajectory
    # study is required for PDR-level budgets.
    #
    # LAUNCH-DATE DEPENDENCY: L4/L5 TOI (Tadpole Orbit Insertion) is more
    # strongly launch-date-dependent than L1, because the 60° phasing of
    # L4/L5 relative to Earth translates a launch timing error directly into
    # phasing-correction ΔV.  A ±3-month launch slip can add ~10–20 m/s to
    # the TOI budget for L4/L5.
    # The current 50 m/s already absorbs typical phasing uncertainties at
    # concept-study level; PDR requires a dedicated Earth-to-Trojan trajectory
    # analysis tied to the selected launch window.
    l45_sk_rate  = 1.5       # m/s/yr (Jupiter perturbation dominant)
    l45_sk_12yr  = l45_sk_rate * 12.0   # 18 m/s
    l45_sk_margin = np.ceil(l45_sk_12yr * 1.2)  # 21.6 → 22, with margin → 26
    l45_sk_budget = 26.0     # m/s (as specified)

    l45_toi = 50.0           # m/s  (Tadpole Orbit Insertion, post-delivery)
    l45_contingency = 10.0   # m/s

    l45_total = l45_toi + l45_sk_budget + l45_contingency

    # L4/L5 propellant (chemical bipropellant for TOI, xenon for SK)
    m_l45_dry = 808.0   # kg  (L4/L5 nodes)
    m_l45_prop_toi = m_l45_dry * (np.exp(l45_toi / (Isp_chem * g0)) - 1.0)
    m_l45_prop_sk  = m_l45_dry * (np.exp(l45_sk_budget / (Isp_ion * g0)) - 1.0)
    m_l45_prop_xenon_10 = m_l45_dry * (np.exp((l45_sk_rate * 10) / (Isp_ion * g0)) - 1.0)

    return {
        "L1": {
            "sk_rate": sk_rate,
            "sk_10yr": sk_10yr,
            "sk_12yr": sk_12yr,
            "sk_budget": sk_budget,
            "hoi": hoi_margin,
            "contingency": contingency,
            "total_dv": hoi_margin + sk_budget + contingency,
            "Isp_chem": Isp_chem,
            "Isp_ion": Isp_ion,
            "m_prop_hoi": m_prop_hoi,
            "m_prop_sk_10yr": m_prop_sk_10,
            "m_prop_sk_12yr": m_prop_sk_12,
            "m_prop_sk_budget": m_prop_sk_budget,
            "m_dry": m_dry,
        },
        "L45": {
            "sk_rate": l45_sk_rate,
            "sk_12yr": l45_sk_12yr,
            "sk_budget": l45_sk_budget,
            "toi": l45_toi,
            "contingency": l45_contingency,
            "total_dv": l45_total,
            "m_dry": m_l45_dry,
            "m_prop_toi": m_l45_prop_toi,
            "m_prop_sk_xenon": m_l45_prop_sk,
            "m_prop_sk_10yr_xenon": m_l45_prop_xenon_10,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# §10  SRP PERTURBATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def srp_perturbation_analysis(sol_nosrp, sol_srp,
                              state0: np.ndarray, mu: float) -> dict:
    """
    Quantify the effect of Solar Radiation Pressure on the L1 halo orbit.

    Metrics:
    - SRP acceleration magnitude vs CR3BP acceleration at L1
    - Position drift after 1 period without correction
    - Extra ΔV per year attributable to SRP

    Reference: spacecraft parameters defined in §1.
    """
    # SRP acceleration magnitude (dimensional)
    a_srp_dim = A_SRP_DIM   # m/s²

    # CR3BP natural acceleration magnitude AT L1 (centrifugal + gravitational)
    x_l1 = X_L1
    r1_l1 = x_l1 + mu           # distance to Sun
    r2_l1 = (1.0 - mu) - x_l1   # distance to Earth
    # At L1, d²x/dt² = 0 in equilibrium, but the individual terms:
    a_grav_sun  = (1.0 - mu) / r1_l1**2
    a_grav_earth = mu / r2_l1**2
    a_cent = x_l1  # centrifugal = ω² r (but ω = 1 in normalized units)
    # Net (should be ~0 at equilibrium): a_cent - a_grav_sun + a_grav_earth
    # For ratio, use centrifugal as reference:
    a_cr3bp_ref = a_cent  # ~ 1 in normalized units

    ratio = A_SRP_NORM / a_cr3bp_ref * 100.0   # percentage

    # Position drift after 1 orbit
    # Compare end states of SRP vs no-SRP propagation
    T_half_idx = len(sol_nosrp.t) // 2  # ~1 period
    pos_nosrp = sol_nosrp.y[:3, T_half_idx] * L_STAR  # km
    pos_srp   = sol_srp.y[:3, T_half_idx] * L_STAR

    drift_1period_km = np.linalg.norm(pos_srp - pos_nosrp)

    # Full 2-period drift
    pos_nosrp_end = sol_nosrp.y[:3, -1] * L_STAR
    pos_srp_end   = sol_srp.y[:3, -1] * L_STAR
    drift_2period_km = np.linalg.norm(pos_srp_end - pos_nosrp_end)

    # 30-day drift (find index closest to 30 days)
    t_days = sol_nosrp.t * TU_TO_DAYS
    idx_30 = np.argmin(np.abs(t_days - 30.0))
    pos_nosrp_30 = sol_nosrp.y[:3, idx_30] * L_STAR
    pos_srp_30   = sol_srp.y[:3, idx_30] * L_STAR
    drift_30day_km = np.linalg.norm(pos_srp_30 - pos_nosrp_30)

    # Extra ΔV from SRP per year
    # Approximate: Δv_SRP ≈ a_SRP × Δt  (impulse approximation)
    dt_year = 365.25 * 86400.0  # seconds
    dv_srp_year = a_srp_dim * dt_year  # m/s/yr

    return {
        "a_srp_dim": a_srp_dim,
        "a_srp_norm": A_SRP_NORM,
        "ratio_pct": ratio,
        "drift_1period_km": drift_1period_km,
        "drift_2period_km": drift_2period_km,
        "drift_30day_km": drift_30day_km,
        "dv_srp_per_year": dv_srp_year,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §11  PLOTTING FUNCTIONS (publication quality, 300 DPI)
# ══════════════════════════════════════════════════════════════════════════════

# Consistent color palette (professional aerospace)
C_HELIOS   = "#0055A4"    # HELIOS primary blue
C_GENESIS  = "#808080"    # Genesis gray
C_SRP      = "#CC3311"    # SRP red
C_NOSRP    = "#004488"    # no-SRP blue
C_L1       = "#EE7733"    # L1 marker orange
C_SUN      = "#DDAA33"    # Sun gold
C_EARTH    = "#009988"    # Earth teal

FIGDIR = Path(".")  # save in current directory


def plot_3d_halo(sol, mu, Az_km, sez_data, outdir=FIGDIR):
    """
    Plot 1 — 3-D Halo orbit in L1-centered local coordinates.

    Uses ξ = (x − x_L1), η = y, ζ = z  in units of 10³ km.
    Shows only 1 period for clarity (periodic orbit overlaps itself).
    Includes a minimal SEZ 5° wireframe cone from Earth toward Sun.
    Camera angle chosen to reveal the 3-D saddle-like shape.

    Saves: helios_L1_halo_3D.png
    """
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Convert to L1-centered coordinates in 10³ km
    x_L1_km = X_L1 * L_STAR
    x_local = (sol.y[0] * L_STAR - x_L1_km) / 1e3   # ξ  (10³ km)
    y_local = sol.y[1] * L_STAR / 1e3                 # η  (10³ km)
    z_local = sol.y[2] * L_STAR / 1e3                 # ζ  (10³ km)
    t_days  = sol.t * TU_TO_DAYS

    # Find 1-period cutoff (half the 2-period propagation)
    n_half = len(sol.t) // 2
    x1 = x_local[:n_half]
    y1 = y_local[:n_half]
    z1 = z_local[:n_half]
    t1 = t_days[:n_half]

    # Color by orbital phase (viridis: dark → bright)
    t_norm = (t1 - t1[0]) / (t1[-1] - t1[0])
    cmap = plt.cm.viridis
    n = len(x1)
    for i in range(n - 1):
        ax.plot(x1[i:i+2], y1[i:i+2], z1[i:i+2],
                color=cmap(t_norm[i]), linewidth=2.2, alpha=0.92)

    # Mark L1 at origin
    ax.scatter([0], [0], [0], c=C_L1, s=120, marker="^",
               zorder=10, label="L1 (origin)", edgecolors='black', linewidths=0.5)

    # Mark starting point
    ax.scatter([x1[0]], [y1[0]], [z1[0]], c='#CC0000', s=60,
               marker='o', zorder=10, edgecolors='black', linewidths=0.5,
               label='Orbit start')

    # ── SEZ 5° cone (wireframe rings from Earth toward Sun) ──
    # Earth sits at ξ = +γ·L*/1e3 in local coords (≈ +1498 × 10³ km)
    # Cone extends from Earth in the -ξ direction (toward Sun)
    # We clip it to the orbital region so it doesn't dominate the view
    cone_angle = np.radians(5.0)
    earth_xi = GAMMA_L1 * L_STAR / 1e3   # ≈ 1497.6  (10³ km)
    theta_ring = np.linspace(0, 2 * pi, 80)

    # Draw rings at a few ξ positions spanning the orbit's ξ range
    xi_min_orb, xi_max_orb = np.min(x1), np.max(x1)
    ring_positions = np.linspace(xi_min_orb - 30, xi_max_orb + 30, 6)
    for xi_pos in ring_positions:
        dist_from_earth = earth_xi - xi_pos   # always positive (Earth is far +ξ)
        r_ring = dist_from_earth * np.tan(cone_angle)
        y_ring = r_ring * np.cos(theta_ring)
        z_ring = r_ring * np.sin(theta_ring)
        ax.plot(np.full_like(theta_ring, xi_pos), y_ring, z_ring,
                color='red', alpha=0.20, linewidth=0.6)

    # Draw two axial lines along the cone boundary for visual cue
    xi_ends = np.array([xi_min_orb - 30, xi_max_orb + 30])
    for angle_offset in [0, pi/2, pi, 3*pi/2]:
        r_ends = (earth_xi - xi_ends) * np.tan(cone_angle)
        y_line = r_ends * np.cos(angle_offset)
        z_line = r_ends * np.sin(angle_offset)
        ax.plot(xi_ends, y_line, z_line,
                color='red', alpha=0.12, linewidth=0.4, linestyle='--')

    # Label the cone
    mid_xi = (xi_min_orb + xi_max_orb) / 2
    mid_r = (earth_xi - mid_xi) * np.tan(cone_angle)
    ax.text(mid_xi, mid_r + 15, 0,
            "SEZ 5°", color='red', fontsize=7, alpha=0.7,
            ha='center', va='bottom')

    # Set axis limits tightly around the orbit (with 15% padding)
    pad = 0.15
    for arr, setter in [(x1, ax.set_xlim), (y1, ax.set_ylim), (z1, ax.set_zlim)]:
        lo, hi = np.min(arr), np.max(arr)
        margin = (hi - lo) * pad
        setter(lo - margin, hi + margin)

    # Camera angle: elev=25°, azim=135° — reveals the saddle/halo shape
    ax.view_init(elev=25, azim=135)

    # Axis labels in local coords
    ax.set_xlabel("ξ  (10³ km)", fontsize=10, labelpad=10)
    ax.set_ylabel("η  (10³ km)", fontsize=10, labelpad=10)
    ax.set_zlabel("ζ  (10³ km)", fontsize=10, labelpad=8)
    

    # Legend — positioned to avoid overlapping direction label
    ax.legend(fontsize=8, loc="upper right",
              bbox_to_anchor=(1.0, 0.95), framealpha=0.9)

    # Direction annotation — top-left, clear of legend
    ax.text2D(0.02, 0.97,
        "← Sun                  Earth →",
        transform=ax.transAxes, fontsize=9, color='#555555',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.9))

    # Colorbar for orbital phase
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(0, t1[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.10)
    cbar.set_label("Time (days)", fontsize=9)

    # Amplitude annotations — bottom-center, clear of axes
    ax.text2D(0.50, 0.02,
        f"Ax = {(np.max(x1)-np.min(x1))/2:,.0f} × 10³ km   |"
        f"   Ay = {(np.max(y1)-np.min(y1))/2:,.0f} × 10³ km   |"
        f"   Az = {(np.max(z1)-np.min(z1))/2:,.0f} × 10³ km",
        transform=ax.transAxes, fontsize=8.5, color='#333333',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFDD',
                  edgecolor='#999999', alpha=0.90))

    plt.tight_layout()
    path = outdir / "helios_L1_halo_3D.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_projections(sols: dict, mu, outdir=FIGDIR):
    """
    Plot 2 — Three orthographic projections (X-Y, X-Z, Y-Z).

    Compare Genesis (Az=290k, dashed gray) vs HELIOS (Az=250k, solid blue).

    Saves: helios_L1_halo_projections.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    projections = [
        (0, 1, "X (km)", "Y (km)"),
        (0, 2, "X (km)", "Z (km)"),
        (1, 2, "Y (km)", "Z (km)"),
    ]

    for ax, (i, j, xlabel, ylabel) in zip(axes, projections):
        for label, sol_info in sols.items():
            sol = sol_info["sol"]
            style = sol_info.get("style", "-")
            color = sol_info.get("color", C_HELIOS)
            lw = sol_info.get("lw", 1.5)
            qi = sol.y[i] * L_STAR
            qj = sol.y[j] * L_STAR
            ax.plot(qi, qj, style, color=color, linewidth=lw,
                    label=label, alpha=0.85)

        # Mark L1
        l1_coords = [X_L1 * L_STAR, 0, 0]
        earth_coords = [(1 - mu) * L_STAR, 0, 0]
        ax.plot(l1_coords[i], l1_coords[j], "^", color=C_L1, ms=8, zorder=5)
        ax.annotate("L1", (l1_coords[i], l1_coords[j]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3))

    # X-Z projection: clean for publication (debug annotations removed)

    fig.suptitle("HELIOS L1 Halo Orbit \u2014 Orthographic Projections", fontsize=13, y=1.02)
    plt.tight_layout()
    path = outdir / "helios_L1_halo_projections.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_sez(sez_results: dict, outdir=FIGDIR):
    """
    Plot 3 — SEZ analysis.

    Top: θ(t) vs time with threshold lines.
    Bottom: bar chart of violation fractions for each Az.

    Saves: helios_sez_analysis.png
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={"height_ratios": [2, 1]})

    colors_az = {"200k": "#117733", "250k": C_HELIOS, "290k": C_GENESIS}

    # Top panel: θ(t)
    for label, sez in sez_results.items():
        c = colors_az.get(label.split()[0], C_HELIOS)
        ax1.plot(sez["time_days"], sez["theta_sez"], color=c,
                 linewidth=0.8, label=label, alpha=0.85)

    for cone, ls in [(3.0, ":"), (5.0, "--"), (10.0, "-.")]:
        ax1.axhline(cone, color="red", linestyle=ls, alpha=0.5,
                    label=f"SEZ {cone:.0f}°")

    ax1.set_xlabel("Time (days)", fontsize=10)
    ax1.set_ylabel("Angular separation (°)", fontsize=10)
    ax1.set_title("Solar Exclusion Zone Analysis", fontsize=12)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Shade the 10-deg SEZ violation window for HELIOS 250k
    for t_start_shade, t_end_shade in [(0, 4.0), (173.7, 177.74)]:
        ax1.axvspan(t_start_shade, t_end_shade,
                    alpha=0.12, color='red',
                    label='10° SEZ entry (250k)' if t_start_shade == 0 else None)

    # Bottom panel: minimum angular separation comparison
    az_labels_plot = list(sez_results.keys())
    min_angles = [sez_results[lbl]["min_separation"] for lbl in az_labels_plot]
    colors_bar = ["#CC3311" if a < 5.0 else "#DDAA33" if a < 10.0 else "#117733"
                  for a in min_angles]

    x_pos = np.arange(len(az_labels_plot))
    bars = ax2.bar(x_pos, min_angles, color=colors_bar, width=0.5, alpha=0.85)

    # Reference lines
    for thresh, ls, ref_label in [(3.0, ':', 'SEZ 3° hard limit'),
                                   (5.0, '--', 'SEZ 5° requirement'),
                                   (10.0, '-.', 'SEZ 10° nominal')]:
        ax2.axhline(thresh, color='red', linestyle=ls, alpha=0.7, label=ref_label)

    # Annotate bars with value + violation status
    for bar, angle, lbl in zip(bars, min_angles, az_labels_plot):
        status = "✓ OK" if angle >= 5.0 else "✗ FAIL"
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{angle:.2f}°\n{status}", ha='center', fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(az_labels_plot, fontsize=10)
    ax2.set_ylabel("Minimum Angular Separation (°)", fontsize=10)
    ax2.set_title("SEZ Compliance: Minimum Angular Separation from Sun–Earth Line", fontsize=11)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_ylim(0, max(min_angles) * 1.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = outdir / "helios_sez_analysis.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_srp_comparison(sol_nosrp, sol_srp, T_period, outdir=FIGDIR):
    """
    Plot 4 — SRP perturbation: component residuals + drift magnitude.

    Top 3 panels: Δx(t), Δy(t), Δz(t) in km  (SRP minus no-SRP).
    Bottom panel: total position drift |\u0394r|(t).

    The previous version overlaid two full orbits, but at 30-day scale
    the SRP drift (~433 km) is 0.06% of the orbital amplitude (~705,000 km),
    making the two trajectories visually identical.  Residual plots reveal
    the perturbation structure clearly.

    Saves: helios_srp_comparison.png
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.32, wspace=0.30)
    ax_dx = fig.add_subplot(gs[0, 0])
    ax_dy = fig.add_subplot(gs[0, 1])
    ax_dz = fig.add_subplot(gs[0, 2])
    ax_mag = fig.add_subplot(gs[1, :])

    t_days = sol_nosrp.t * TU_TO_DAYS

    # Component residuals in km
    dx = (sol_srp.y[0] - sol_nosrp.y[0]) * L_STAR  # Δx
    dy = (sol_srp.y[1] - sol_nosrp.y[1]) * L_STAR  # Δy
    dz = (sol_srp.y[2] - sol_nosrp.y[2]) * L_STAR  # Δz

    # Total drift magnitude
    drift_mag = np.sqrt(dx**2 + dy**2 + dz**2)

    # ── Top panels: component residuals ──
    comp_data = [
        (ax_dx, dx, "Δx (radial)",   "#004488"),
        (ax_dy, dy, "Δy (cross-track)", "#DDAA33"),
        (ax_dz, dz, "Δz (out-of-plane)",  "#BB5566"),
    ]
    for ax, d, label, color in comp_data:
        ax.plot(t_days, d, color=color, linewidth=1.5)
        ax.fill_between(t_days, 0, d, alpha=0.12, color=color)
        ax.set_xlabel("Time (days)", fontsize=10)
        ax.set_ylabel(f"{label} (km)", fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        # Annotate final value
        ax.text(0.97, 0.92, f"{d[-1]:+.1f} km",
                transform=ax.transAxes, fontsize=9, ha='right',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.set_title(label, fontsize=10, fontweight='bold', color=color)

    # ── Bottom panel: total drift magnitude ──
    ax_mag.plot(t_days, drift_mag, color=C_SRP, linewidth=2.0)
    ax_mag.fill_between(t_days, 0, drift_mag, alpha=0.12, color=C_SRP)
    ax_mag.set_xlabel("Time (days)", fontsize=11)
    ax_mag.set_ylabel("Position drift |\u0394r| (km)", fontsize=11)
    ax_mag.set_title(f"Total SRP-Induced Position Drift "
                     f"(a_SRP = {A_SRP_DIM:.2e} m/s², "
                     f"drift at {t_days[-1]:.0f} days: {drift_mag[-1]:,.0f} km)",
                     fontsize=11)
    ax_mag.grid(True, alpha=0.3)
    ax_mag.axhline(drift_mag[-1], color='gray', linestyle=':', alpha=0.5)
    ax_mag.text(t_days[-1]*0.98, drift_mag[-1]*1.04,
                f"{drift_mag[-1]:,.0f} km", ha='right', fontsize=10,
                color=C_SRP, fontweight='bold')

    # Annotate SRP budget context
    dv_yr = A_SRP_DIM * 365.25 * 86400  # m/s/yr
    ax_mag.text(0.02, 0.92,
        f"SRP \u0394V \u2248 {dv_yr:.1f} m/s/yr (absorbed in SK budget)",
        transform=ax_mag.transAxes, fontsize=9, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', alpha=0.85))

    fig.suptitle(
        f"SRP Perturbation Analysis \u2014 L1 Halo Orbit ({t_days[-1]:.0f}-day window)",
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = outdir / "helios_srp_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# §12  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = _timer.perf_counter()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    outdir = Path(".")
    print("=" * 70)
    print("HELIOS CONSTELLATION — APPENDIX A: Orbital Mechanics Computation")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # A. Richardson IC for three Az amplitudes
    # ------------------------------------------------------------------
    az_targets = [200_000.0, 250_000.0, 290_000.0]
    ic_results = {}

    for Az in az_targets:
        tag = f"{Az/1e3:.0f}k"
        print(f"── Richardson IC: Az = {tag} km ──")
        res = richardson_halo_ic(MU, GAMMA_L1, X_L1, Az, L_STAR, m=1)
        ic_results[tag] = res

        s = res["state_norm"]
        sd = res["state_dim_km"]
        print(f"  Ax = {res['Ax_km']:,.1f} km,  Ay = {res['Ay_km']:,.1f} km,  "
              f"Az = {res['Az_km']:,.1f} km")
        print(f"  Period (Richardson) = {res['period_days']:.2f} days")
        print(f"  Normalized IC:")
        labels = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
        for lbl, val in zip(labels, s):
            print(f"    {lbl:4s} = {val:+.10e}")
        print(f"  Dimensional IC:")
        units = ["km", "km", "km", "km/s", "km/s", "km/s"]
        for lbl, val, u in zip(labels, sd, units):
            print(f"    {lbl:4s} = {val:+.6e} {u}")
        print()

    # ------------------------------------------------------------------
    # B. Differential correction + period refinement (250k km)  [2]
    #    With Az-targeting to match design amplitude exactly.
    # ------------------------------------------------------------------
    helios = ic_results["250k"]
    state0_rich = helios["state_norm"].copy()

    print("── Differential correction & period refinement  [Howell 1984] ──")
    T_rich = helios["period_norm"]
    state0, T_period, az_actual = find_halo_with_target_az(
        state0_rich, T_rich, MU, az_target_km=250_000.0,
        az_tol=200.0, max_iter=20,
    )
    T_period_days = T_period * TU_TO_DAYS
    print(f"  Richardson period:  {T_rich * TU_TO_DAYS:.4f} days  "
          f"({T_rich:.10f} TU)")
    print(f"  Corrected  period:  {T_period_days:.4f} days  "
          f"({T_period:.10f} TU)")
    print(f"  Final Az:           {az_actual:,.0f} km")
    print(f"  Corrected IC:")
    labels_ic = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
    for lbl, val in zip(labels_ic, state0):
        print(f"    {lbl:4s} = {val:+.10e}")
    print()

    # ------------------------------------------------------------------
    # C. Jacobi constant check
    # ------------------------------------------------------------------
    C0 = jacobi_constant(state0, MU)
    print(f"── Jacobi constant at t=0:  C = {C0:.10f} ──")
    print()

    # ------------------------------------------------------------------
    # D. Propagate 2 periods: with and without SRP
    # ------------------------------------------------------------------
    T_total = 2.0 * T_period
    print(f"── Propagating 2 periods ({T_total * TU_TO_DAYS:.2f} days) ──")

    print("  [1/2] No SRP ...")
    sol_nosrp = propagate(state0, (0, T_total), MU, srp=False, n_eval=10000)
    print("  [2/2] With SRP ...")
    sol_srp   = propagate(state0, (0, T_total), MU, srp=True,
                          a_srp=A_SRP_NORM, n_eval=10000)

    # Short 30-day propagation for SRP comparison plot
    T_30day = 30.0 / TU_TO_DAYS
    print("  [+] 30-day SRP comparison for plot ...")
    sol_nosrp_30 = propagate(state0, (0, T_30day), MU, srp=False, n_eval=3000)
    sol_srp_30   = propagate(state0, (0, T_30day), MU, srp=True,
                              a_srp=A_SRP_NORM, n_eval=3000)

    # Jacobi constant drift
    C_final = jacobi_constant(sol_nosrp.y[:, -1], MU)
    print(f"  Jacobi constant (no SRP): C₀ = {C0:.10f}, C_final = {C_final:.10f}, "
          f"ΔC = {abs(C_final - C0):.2e}")
    print()

    # Differentially correct comparison orbits (200k, 290k) for periodicity.
    # 200k uses continuation from 250k DC result (more robust than raw
    # Richardson ICs, which can fail DC at lower amplitudes).
    # 290k starts from Richardson ICs (converges reliably).
    print("── Correcting comparison orbits (differential correction) ──")
    sols_compare = {}
    for tag in ["200k", "290k"]:
        r = ic_results[tag]
        target_az = float(tag.replace("k", "")) * 1000.0
        try:
            if tag == "200k":
                # Continuation from 250k: scale z0 toward target Az
                s0_cont = state0.copy()
                s0_cont[2] *= target_az / az_actual
                T_guess_cont = T_period
            else:
                s0_cont = r["state_norm"].copy()
                T_guess_cont = r["period_norm"]
            s0_c, T_c, az_c = find_halo_with_target_az(
                s0_cont, T_guess_cont, MU, az_target_km=target_az,
                az_tol=500.0, max_iter=15,
            )
            sol_c = propagate(s0_c, (0, 2.0 * T_c), MU, srp=False, n_eval=8000)
            sols_compare[tag] = {"sol": sol_c, "T_period": T_c, "ic": r}
            print(f"  {tag}: DC converged, Az = {az_c:,.0f} km, "
                  f"period = {T_c * TU_TO_DAYS:.2f} days")
        except Exception as e:
            print(f"  {tag}: DC failed ({e}), falling back to Richardson IC")
            sol_c = propagate(r["state_norm"], (0, 2.0 * r["period_norm"]),
                              MU, srp=False, n_eval=8000)
            sols_compare[tag] = {"sol": sol_c, "T_period": r["period_norm"], "ic": r}
    print()

    # ------------------------------------------------------------------
    # E. SEZ analysis for all three amplitudes
    # ------------------------------------------------------------------
    print("── SEZ Analysis ──")
    sez_results = {}

    sez_250 = sez_analysis(sol_nosrp, MU)
    sez_results["250k (HELIOS)"] = sez_250

    for tag in ["200k", "290k"]:
        sez_r = sez_analysis(sols_compare[tag]["sol"], MU)
        sez_results[f"{tag}"] = sez_r

    print(f"  {'Az':>10s} | {'SEZ 3°':>8s} | {'SEZ 5°':>8s} | {'SEZ 10°':>8s} | Min sep")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for label, sez in sez_results.items():
        v = sez["violations"]
        print(f"  {label:>10s} | {v[3.0]:7.2f}% | {v[5.0]:7.2f}% | {v[10.0]:7.2f}% | "
              f"{sez['min_separation']:.2f}°")
    print()

    # ------------------------------------------------------------------
    # F. Eclipse analysis
    # ------------------------------------------------------------------
    print("── Eclipse Analysis ──")
    ecl = eclipse_analysis(sol_nosrp, MU)
    print(f"  Umbra fraction:    {ecl['frac_umbra']:.4f}%")
    print(f"  Penumbra fraction: {ecl['frac_penumbra']:.4f}%")
    if ecl["frac_umbra"] == 0.0 and ecl["frac_penumbra"] == 0.0:
        print("  ✓ L1 halo orbit avoids Earth shadow entirely (as expected).")
    print()

    # ------------------------------------------------------------------
    # G. Stability analysis (STM / monodromy matrix)
    # ------------------------------------------------------------------
    print("── Stability Analysis (Monodromy Matrix) ──")
    stab = stability_analysis(state0, T_period, MU)
    print(f"  Monodromy matrix eigenvalues:")
    for i, ev in enumerate(stab["eigenvalues"]):
        print(f"    λ_{i+1} = {ev.real:+.6e} {ev.imag:+.6e}j   "
              f"|λ| = {abs(ev):.6e}")
    print(f"  Unstable eigenvalue: |λ_u| = {stab['lambda_u_mag']:.6f}")
    print(f"  Stability index ν = {stab['stability_index']:.4f}  "
          f"(orbit is {'unstable' if stab['stability_index'] > 1 else 'stable'})")
    print(f"  Divergence time = {stab['diverge_time_days']:.1f} days "
          "(without station-keeping)")
    print()

    # ------------------------------------------------------------------
    # H. ΔV budget
    # ------------------------------------------------------------------
    print("── ΔV Budget ──")
    budget = delta_v_budget(M_SC)
    L1b = budget["L1"]
    L45b = budget["L45"]

    print(f"  L1 NODE:")
    print(f"    HOI allocation:      {L1b['hoi']:.0f} m/s "
          f"(chemical, Isp={L1b['Isp_chem']:.0f}s)")
    print(f"    Station-keeping:     {L1b['sk_budget']:.0f} m/s "
          f"(electric, Isp={L1b['Isp_ion']:.0f}s, 12 yr + margin)")
    print(f"    Contingency:         {L1b['contingency']:.0f} m/s")
    print(f"    TOTAL:               {L1b['total_dv']:.0f} m/s")
    print(f"    Prop mass (HOI):     {L1b['m_prop_hoi']:.1f} kg (bipropellant)")
    print(f"    Xenon (SK, 10yr):    {L1b['m_prop_sk_10yr']:.1f} kg")
    print(f"    Xenon (SK, 12yr):    {L1b['m_prop_sk_12yr']:.1f} kg")
    print(f"    Xenon (SK, budget):  {L1b['m_prop_sk_budget']:.1f} kg")
    print()
    print(f"  L4/L5 NODES:")
    print(f"    TOI allocation:      {L45b['toi']:.0f} m/s  (Tadpole Orbit Insertion)")
    print(f"    Station-keeping:     {L45b['sk_budget']:.0f} m/s "
          f"(12 yr + margin)")
    print(f"    Contingency:         {L45b['contingency']:.0f} m/s")
    print(f"    TOTAL:               {L45b['total_dv']:.0f} m/s")
    print(f"    Prop mass (TOI):     {L45b['m_prop_toi']:.1f} kg (bipropellant)")
    print(f"    Xenon (SK, 10yr):    {L45b['m_prop_sk_10yr_xenon']:.1f} kg")
    print(f"    Xenon (SK, budget):  {L45b['m_prop_sk_xenon']:.1f} kg")
    print()

    # ------------------------------------------------------------------
    # I. SRP perturbation quantification
    # ------------------------------------------------------------------
    print("── SRP Perturbation Analysis ──")
    srp_info = srp_perturbation_analysis(sol_nosrp, sol_srp, state0, MU)
    print(f"  SRP acceleration (dim):   {srp_info['a_srp_dim']:.4e} m/s²")
    print(f"  SRP acceleration (norm):  {srp_info['a_srp_norm']:.4e}")
    print(f"  SRP / CR3BP centrifugal:  {srp_info['ratio_pct']:.4f}%")
    print(f"  Position drift (30 days): {srp_info['drift_30day_km']:,.0f} km")
    print(f"  Position drift (1 period):{srp_info['drift_1period_km']:,.0f} km")
    print(f"  Position drift (2 periods):{srp_info['drift_2period_km']:,.0f} km")
    print(f"  Extra ΔV from SRP:        {srp_info['dv_srp_per_year']:.2f} m/s/year")
    print(f"  → SRP contributes ~{srp_info['dv_srp_per_year']:.1f} m/s/year "
          "to station-keeping budget")
    print()

    # ------------------------------------------------------------------
    # J. Generate all plots
    # ------------------------------------------------------------------
    print("── Generating publication-quality plots ──")

    # Plot 1: 3-D halo orbit + SEZ cone
    plot_3d_halo(sol_nosrp, MU, 250_000.0, sez_250, outdir)

    # Plot 2: Projections — HELIOS (250k) + Genesis (290k)
    proj_sols = {
        "HELIOS (250k km)": {
            "sol": sol_nosrp, "style": "-", "color": C_HELIOS, "lw": 1.5,
        },
        "Genesis (290k km)": {
            "sol": sols_compare["290k"]["sol"],
            "style": "--", "color": C_GENESIS, "lw": 1.0,
        },
    }
    plot_projections(proj_sols, MU, outdir)

    # Plot 3: SEZ analysis
    plot_sez(sez_results, outdir)

    # Plot 4: SRP comparison (30-day window)
    plot_srp_comparison(sol_nosrp_30, sol_srp_30, T_period, outdir)

    # ------------------------------------------------------------------
    # K. Final summary table → console + text file
    # ------------------------------------------------------------------
    # Use corrected IC for the summary (state0 has been refined by DC)
    s0 = state0   # corrected, normalized
    sd = np.array([
        state0[0] * L_STAR,
        state0[1] * L_STAR,
        state0[2] * L_STAR,
        state0[3] * V_STAR_KM,
        state0[4] * V_STAR_KM,
        state0[5] * V_STAR_KM,
    ])

    # Compute actual amplitudes from corrected trajectory
    x_traj = sol_nosrp.y[0] * L_STAR
    y_traj = sol_nosrp.y[1] * L_STAR
    z_traj = sol_nosrp.y[2] * L_STAR
    Ax_actual = (np.max(x_traj) - np.min(x_traj)) / 2.0
    Ay_actual = (np.max(y_traj) - np.min(y_traj)) / 2.0
    Az_actual = (np.max(z_traj) - np.min(z_traj)) / 2.0

    summary_lines = []

    def P(line=""):
        summary_lines.append(line)
        print(line)

    P("═" * 70)
    P("HELIOS CONSTELLATION - ORBITAL MECHANICS PARAMETERS")
    P(f"Computed: {timestamp} | CR3BP Sun-Earth | μ = {MU:.6e}")
    P("═" * 70)
    P()
    P("L1 NODE - NORTHERN HALO ORBIT")
    P(f"  Orbit type:         Northern L1 Halo (Richardson 1980 + differential correction)")
    P(f"  Az amplitude:       {Az_actual:,.0f} km")
    P(f"  Ay amplitude:       {Ay_actual:,.0f} km")
    P(f"  Ax amplitude:       {Ax_actual:,.0f} km")
    P(f"  Jacobi constant C:  {C0:.8f}")
    P(f"  Period T:           {T_period_days:.2f} days")
    P(f"  Stability index ν:  {stab['stability_index']:.4f}")
    P(f"  Diverge time:       {stab['diverge_time_days']:.1f} days  (without station-keeping)")
    P(f"  SEZ margin (±5°):   {sez_250['min_separation']:.2f}°  (minimum angular separation)")
    P(f"  Eclipse fraction:   {ecl['frac_umbra']:.2f}%  (expected: 0%)")
    P()
    P(f"  Corrected IC (normalized CR3BP units):")
    P(f"    x0  = {s0[0]:+.10e}")
    P(f"    y0  = {s0[1]:+.10e}")
    P(f"    z0  = {s0[2]:+.10e}")
    P(f"    vx0 = {s0[3]:+.10e}")
    P(f"    vy0 = {s0[4]:+.10e}")
    P(f"    vz0 = {s0[5]:+.10e}")
    P()
    P(f"  Corrected IC (dimensional: km, km/s):")
    P(f"    x0  = {sd[0]:+.6e} km")
    P(f"    y0  = {sd[1]:+.1f} km")
    P(f"    z0  = {sd[2]:+.6e} km")
    P(f"    vx0 = {sd[3]:+.6e} km/s")
    P(f"    vy0 = {sd[4]:+.10e} km/s")
    P(f"    vz0 = {sd[5]:+.6e} km/s")
    P()
    P(f"  SRP Perturbation:")
    P(f"    SRP acceleration:  {srp_info['a_srp_dim']:.4e} m/s²")
    P(f"    SRP/CR3BP ratio:   {srp_info['ratio_pct']:.4f}%")
    P(f"    SRP ΔV per year:   {srp_info['dv_srp_per_year']:.2f} m/s/year")
    P()
    P(f"  DELTA-V BUDGET (conservative, 20% margin):")
    P(f"    HOI allocation:    {L1b['hoi']:.0f} m/s (chemical, bipropellant, Isp={L1b['Isp_chem']:.0f}s)")
    P(f"    Station-keeping:   {L1b['sk_budget']:.0f} m/s (electric, xenon, Isp={L1b['Isp_ion']:.0f}s, 12yr)")
    P(f"    Contingency:       {L1b['contingency']:.0f} m/s")
    P(f"    TOTAL:             {L1b['total_dv']:.0f} m/s")
    P()
    P(f"  PROPELLANT MASSES:")
    P(f"    Chemical (HOI):    {L1b['m_prop_hoi']:.1f} kg (bipropellant)")
    P(f"    Xenon (SK, 10yr):  {L1b['m_prop_sk_10yr']:.1f} kg")
    P(f"    Xenon (SK, 12yr):  {L1b['m_prop_sk_12yr']:.1f} kg")
    P()
    P(f"  XENON MASS BREAKDOWN:")
    P(f"    Baseline (40 m/s): {L1b['m_prop_sk_10yr']:.1f} kg")
    P(f"    Budget   (60 m/s): {L1b['m_prop_sk_budget']:.1f} kg")
    P(f"    With 20% ullage:   {L1b['m_prop_sk_budget'] * 1.20:.1f} kg")
    P(f"    Note: 20% ullage margin covers tank residuals and loading tolerances.")
    P()
    P("L4/L5 NODES - TADPOLE ORBITS")
    P(f"  Orbit type:         Tadpole orbit (linearly stable)")
    P(f"  Station-keeping:    {L45b['sk_rate']:.1f} m/s/year (Jupiter perturbation dominant)")
    P(f"  TOI allocation:     {L45b['toi']:.0f} m/s  (Tadpole Orbit Insertion, chemical bipropellant, Isp={L1b['Isp_chem']:.0f}s)")
    P(f"  SK budget (12yr):   {L45b['sk_budget']:.0f} m/s (with 20% margin)")
    P(f"  Contingency:        {L45b['contingency']:.0f} m/s")
    P(f"  TOTAL ΔV:           {L45b['total_dv']:.0f} m/s")
    P(f"  Dry mass:           {L45b['m_dry']:.0f} kg")
    P(f"  Prop (TOI, chem):   {L45b['m_prop_toi']:.1f} kg")
    P(f"  Xenon (SK, 10yr):   {L45b['m_prop_sk_10yr_xenon']:.1f} kg")
    P(f"  Xenon (SK, budget): {L45b['m_prop_sk_xenon']:.1f} kg")
    P()
    P("═" * 70)

    # Save summary to text file
    summary_path = outdir / "helios_orbital_params.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n[output] Summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    elapsed = _timer.perf_counter() - t_start
    print(f"\n[done] Total computation time: {elapsed:.2f} s")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
