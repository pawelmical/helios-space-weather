#!/usr/bin/env python3
"""
HELIOS Geometry Verification Runner
===================================

Runs the complete geometry verification suite and generates output reports.

Usage:
    python run_geometry_verification.py [--samples N]

Outputs:
    output/geometry_verification.csv - Main verification report
    output/coverage_analysis.csv - Detailed coverage analysis
    output/gdop_analysis.csv - GDOP by baseline configuration
    output/spatial_resolution_sweep.csv - Resolution vs distance/precision

Author: HELIOS Team
Date: January 2026
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. CSV output will be limited.")

from helios_code.geometry_verification import (
    run_full_geometry_verification,
    generate_verification_report,
    analyze_full_constellation,
    verify_coverage_three_methods,
    run_spatial_resolution_sweep,
    verify_timing_advantage_claim,
    calculate_triangulation_gdop,
    verify_spatial_resolution,
    monte_carlo_resolution
)


def create_output_dir(path: str = "output") -> str:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def generate_coverage_report(output_dir: str = "output") -> str:
    """Generate detailed coverage analysis CSV."""
    if not HAS_PANDAS:
        return None

    results = verify_coverage_three_methods()

    rows = []
    for method in results['methods']:
        r = results['individual_results'][method]
        rows.append({
            'method': method,
            'coverage_percent': r.total_coverage_percent,
            'blind_spot_deg': r.blind_spot_deg,
            'blind_spot_center_deg': r.blind_spot_center_deg,
            'earth_distance_deg': r.earth_threat_zone_deg
        })

    # Add summary row
    rows.append({
        'method': 'MEAN',
        'coverage_percent': results['mean_coverage_percent'],
        'blind_spot_deg': results['blind_spot_deg'],
        'blind_spot_center_deg': float('nan'),
        'earth_distance_deg': results['earth_threat_distance_deg']
    })

    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, 'coverage_analysis.csv')
    df.to_csv(filepath, index=False)

    return filepath


def generate_gdop_report(output_dir: str = "output") -> str:
    """Generate GDOP analysis CSV for various configurations."""
    if not HAS_PANDAS:
        return None

    rows = []

    # Two-observer configurations
    baselines = [30, 45, 60, 90, 120, 150, 180]
    for baseline in baselines:
        result = calculate_triangulation_gdop(
            [baseline/2, -baseline/2],
            target_distance_au=0.5
        )
        rows.append({
            'configuration': f'2-obs ({baseline} deg baseline)',
            'n_observers': 2,
            'baseline_deg': baseline,
            'gdop': result.gdop_value,
            'hdop': result.hdop,
            'vdop': result.vdop,
            'pdop': result.pdop,
            'condition_number': result.condition_number,
            'is_optimal': result.is_optimal
        })

    # Named configurations
    named_configs = [
        ('L1+L4', [0, 60]),
        ('L1+L5', [0, -60]),
        ('L4+L5', [60, -60]),
        ('L1+L4+L5', [0, 60, -60])
    ]

    for name, angles in named_configs:
        result = calculate_triangulation_gdop(angles, 0.5)
        rows.append({
            'configuration': name,
            'n_observers': len(angles),
            'baseline_deg': result.baseline_angle_deg,
            'gdop': result.gdop_value,
            'hdop': result.hdop,
            'vdop': result.vdop,
            'pdop': result.pdop,
            'condition_number': result.condition_number,
            'is_optimal': result.is_optimal
        })

    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, 'gdop_analysis.csv')
    df.to_csv(filepath, index=False)

    return filepath


def generate_resolution_report(n_samples: int, output_dir: str = "output") -> str:
    """Generate spatial resolution sweep CSV."""
    if not HAS_PANDAS:
        return None

    results = run_spatial_resolution_sweep(
        distances_au=[0.3, 0.5, 0.7, 1.0],
        precisions_deg=[0.25, 0.5, 1.0],
        n_samples=n_samples
    )

    rows = []
    for r in results:
        rows.append({
            'target_distance_au': r.target_distance_au,
            'angular_precision_deg': r.angular_precision_deg,
            'resolution_km': r.spatial_resolution_km,
            'resolution_million_km': r.spatial_resolution_km / 1e6,
            'resolution_solar_radii': r.spatial_resolution_solar_radii,
            'percentile_16_km': r.percentile_16_km,
            'percentile_84_km': r.percentile_84_km,
            'n_samples': r.n_samples,
            'configuration': r.configuration
        })

    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, 'spatial_resolution_sweep.csv')
    df.to_csv(filepath, index=False)

    return filepath


def generate_timing_report(output_dir: str = "output") -> str:
    """Generate timing advantage CSV."""
    if not HAS_PANDAS:
        return None

    timing = verify_timing_advantage_claim()

    rows = []
    for r in timing['results']:
        rows.append({
            'cme_speed_km_s': r.cme_speed_km_s,
            'single_point_hours': r.single_point_detection_hours,
            'stereo_hours': r.stereo_detection_hours,
            'advantage_hours': r.advantage_hours,
            'improvement_percent': r.improvement_percent,
            'earth_directed': r.earth_directed
        })

    # Summary row
    rows.append({
        'cme_speed_km_s': 'MEAN',
        'single_point_hours': np.mean([r.single_point_detection_hours for r in timing['results']]),
        'stereo_hours': np.mean([r.stereo_detection_hours for r in timing['results']]),
        'advantage_hours': timing['mean_advantage_hours'],
        'improvement_percent': np.mean([r.improvement_percent for r in timing['results']]),
        'earth_directed': True
    })

    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, 'timing_advantage.csv')
    df.to_csv(filepath, index=False)

    return filepath


def print_summary(results: dict):
    """Print executive summary of verification results."""
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY - GEOMETRY VERIFICATION RESULTS")
    print("=" * 70)

    # Coverage
    cov = results['coverage_verification']
    print(f"""
COVERAGE VERIFICATION:
  Method 1 (Ray Tracing):        {cov['coverage_percentages'][0]:.1f}%
  Method 2 (Angular Arithmetic): {cov['coverage_percentages'][1]:.1f}%
  Method 3 (Set Theoretic):      {cov['coverage_percentages'][2]:.1f}%
  -----------------------------------------
  MEAN COVERAGE:                 {cov['mean_coverage_percent']:.1f}%
  Standard Deviation:            {cov['std_coverage_percent']:.2f}%
  Blind Spot:                    {cov['blind_spot_deg']:.1f} deg
  Blind Spot Distance to Earth:  {cov['earth_threat_distance_deg']:.0f} deg
  Methods Agree:                 {'YES' if cov['methods_agree'] else 'NO'}
""")

    # GDOP
    gdop = results['gdop_verification']
    print(f"""TRIANGULATION GEOMETRY:
  L1+L4 Intersection Angle:      {gdop['l1_l4_intersection_deg']:.1f} deg
  L1+L4 Resolution @ 0.5 AU:     {gdop['l1_l4_resolution_km']/1e6:.2f} million km
  L4-L5 Baseline:                {gdop['l4_l5_baseline_deg']} deg
  Best Pair for Earth CME:       {gdop['best_pair']}
  Verification:                  {'PASSED' if gdop['verification_passed'] else 'FAILED'}
""")

    # Resolution
    res = results['spatial_resolution']
    print(f"""SPATIAL RESOLUTION (Monte Carlo, N={res.n_samples}):
  Target Distance:               {res.target_distance_au} AU
  Angular Precision:             {res.angular_precision_deg} deg
  -----------------------------------------
  SPATIAL RESOLUTION:            {res.spatial_resolution_km/1e6:.2f} million km
  In Solar Radii:                {res.spatial_resolution_solar_radii:.2f} Rs
  16th Percentile:               {res.percentile_16_km/1e6:.2f} million km
  84th Percentile:               {res.percentile_84_km/1e6:.2f} million km
""")

    # Timing
    timing = results['timing_verification']
    print(f"""TIMING ADVANTAGE:
  Minimum Advantage:             {timing['min_advantage_hours']:.1f} hours
  Maximum Advantage:             {timing['max_advantage_hours']:.1f} hours
  Mean Advantage:                {timing['mean_advantage_hours']:.1f} hours
  Claimed Range:                 {timing['claim_range']}
  Actual Range:                  {timing['actual_range']}
  Claim Verified:                {'YES' if timing['claim_verified'] else 'NO'}
""")

    # Overall
    print("=" * 70)
    print("FINAL VERIFICATION STATUS")
    print("=" * 70)
    all_passed = results['all_verifications_passed']
    print(f"""
  Triangulation Geometry:        {'PASSED' if gdop['verification_passed'] else 'FAILED'}
  Coverage Analysis:             {'PASSED' if cov['verification_passed'] else 'FAILED'}
  Spatial Resolution:            {'PASSED' if results['resolution_verified'] else 'FAILED'}
  Timing Advantage:              {'PASSED' if timing['claim_verified'] else 'FAILED'}
  -----------------------------------------
  OVERALL STATUS:                {'ALL PASSED' if all_passed else 'SOME FAILED'}
""")


def main():
    parser = argparse.ArgumentParser(
        description='HELIOS Geometry Verification Suite'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=500,
        help='Number of Monte Carlo samples (default: 500)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("HELIOS GEOMETRY VERIFICATION SUITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Monte Carlo samples: {args.samples}")
    print("=" * 70)

    # Create output directory
    create_output_dir(args.output)

    # Run full verification
    print("\nRunning verification suite...")
    results = run_full_geometry_verification(n_samples=args.samples)

    # Generate reports
    print("\nGenerating output reports...")

    if HAS_PANDAS:
        # Main verification report
        main_report = generate_verification_report(results, args.output)
        print(f"  Created: {main_report}")

        # Coverage analysis
        coverage_report = generate_coverage_report(args.output)
        print(f"  Created: {coverage_report}")

        # GDOP analysis
        gdop_report = generate_gdop_report(args.output)
        print(f"  Created: {gdop_report}")

        # Resolution sweep
        resolution_report = generate_resolution_report(args.samples, args.output)
        print(f"  Created: {resolution_report}")

        # Timing advantage
        timing_report = generate_timing_report(args.output)
        print(f"  Created: {timing_report}")
    else:
        print("  Warning: pandas not available, skipping CSV generation")

    # Print summary
    if not args.quiet:
        print_summary(results)

    # Final status
    print("\n" + "=" * 70)
    if results['all_verifications_passed']:
        print("VERIFICATION COMPLETE - ALL CLAIMS VERIFIED")
    else:
        print("VERIFICATION COMPLETE - SOME CLAIMS NOT VERIFIED")
    print("=" * 70)

    return 0 if results['all_verifications_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
