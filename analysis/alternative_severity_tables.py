#!/usr/bin/env python3
"""
VELOCITY-DEPENDENT SEVERITY TABLES
===================================
Generate separate Bz thresholds for slow/moderate/fast CME categories
to address the v-dependence problem.

Author: HELIOS Team
Date: February 2026
"""

import numpy as np
import pandas as pd

K = 0.0121

def solve_bz_for_dose(dose_mSv, v_cme, t=10):
    """Invert dose formula to get Bz threshold."""
    return (dose_mSv / (K * np.sqrt(v_cme) * t)) ** (1.0 / 1.3)

# ============================================================================
# OPTION A: Velocity-Binned Tables
# ============================================================================

def generate_velocity_binned_tables():
    """Create separate tables for slow/moderate/fast CMEs."""
    
    velocity_categories = [
        ('Slow CME', 500, '300-700 km/s'),
        ('Moderate CME', 900, '700-1200 km/s'),
        ('Fast CME', 1600, '1200-2500 km/s'),
    ]
    
    dose_boundaries = [50, 100, 200]
    
    print("="*80)
    print("OPTION A: VELOCITY-BINNED SEVERITY TABLES")
    print("="*80)
    print()
    
    for cat_name, v_ref, v_range in velocity_categories:
        print(f"{cat_name} ({v_range})")
        print("-" * 80)
        
        bz_50  = solve_bz_for_dose(50, v_ref)
        bz_100 = solve_bz_for_dose(100, v_ref)
        bz_200 = solve_bz_for_dose(200, v_ref)
        
        table = pd.DataFrame({
            'Severity': ['Low', 'Moderate', 'High', 'Extreme'],
            'Bz_Range_nT': [
                f'-{bz_50:.0f} to -{bz_100:.0f}',
                f'-{bz_100:.0f} to -{bz_200:.0f}',
                f'-{bz_200:.0f} to -{bz_200*1.5:.0f}',
                f'< -{bz_200*1.5:.0f}',
            ],
            'Dose_Range_mSv': ['10-50', '50-100', '100-200', '>200'],
        })
        
        print(table.to_string(index=False))
        print()


# ============================================================================
# OPTION B: Dose-Based Classification (Recommended!)
# ============================================================================

def generate_dose_based_table():
    """Use DOSE directly as the metric, not Bz."""
    
    print("="*80)
    print("OPTION B: DOSE-BASED SEVERITY (RECOMMENDED)")
    print("="*80)
    print()
    print("Instead of Bz ranges, classify by PREDICTED DOSE:")
    print()
    
    table = pd.DataFrame({
        'Severity': ['Low', 'Moderate', 'High', 'Extreme'],
        'Predicted_Dose': ['10-50 mSv', '50-100 mSv', '100-200 mSv', '>200 mSv'],
        'NASA_30day_Limit': ['<20%', '20-40%', '40-80%', '>80%'],
        'Crew_Response': [
            'Enhanced monitoring',
            'Shelter-in-place advisory',
            'Mandatory shelter protocols',
            'EVA Abort, emergency shielding'
        ],
        'Example_Conditions': [
            'Bz=−15, v=600',
            'Bz=−20, v=800',
            'Bz=−30, v=700',
            'Bz=−40, v>1000 OR Bz>−23, v>1500'
        ]
    })
    
    print(table.to_string(index=False))
    print()
    print("ADVANTAGES:")
    print("  ✓ Directly tied to safety limits (NASA career dose)")
    print("  ✓ No confusion about velocity dependence")
    print("  ✓ Model outputs dose → classification is automatic")
    print("  ✓ Clear for mission planners")


# ============================================================================
# OPTION C: 2D Lookup Table
# ============================================================================

def generate_2d_lookup():
    """Create a 2D (Bz, velocity) severity matrix."""
    
    print("\n" + "="*80)
    print("OPTION C: 2D SEVERITY MATRIX (Bz × Velocity)")
    print("="*80)
    print()
    
    bz_bins = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60])
    v_bins = np.array([400, 600, 800, 1000, 1500, 2000])
    
    # Compute dose matrix
    matrix = []
    for bz in bz_bins:
        row = []
        for v in v_bins:
            dose = K * (bz ** 1.3) * np.sqrt(v) * 10
            
            # Severity code
            if dose < 50:
                sev = 'L'
            elif dose < 100:
                sev = 'M'
            elif dose < 200:
                sev = 'H'
            else:
                sev = 'X'
            
            row.append(f"{sev}({dose:.0f})")
        matrix.append(row)
    
    df = pd.DataFrame(matrix, 
                      index=[f'{bz}' for bz in bz_bins],
                      columns=[f'{v}' for v in v_bins])
    
    df.index.name = '|Bz| (nT)'
    df.columns.name = 'v (km/s)'
    
    print(df.to_string())
    print()
    print("Legend: L=Low, M=Moderate, H=High, X=Extreme  (dose in mSv)")
    print()
    print("USAGE: Lookup predicted Bz and v to get severity + dose estimate")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n")
    print("█"*80)
    print("█  ALTERNATIVE SEVERITY CLASSIFICATION SCHEMES")
    print("█  Problem: Static Bz table doesn't account for velocity dependence")
    print("█"*80)
    print()
    
    generate_velocity_binned_tables()
    generate_dose_based_table()
    generate_2d_lookup()
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()
    print("Use OPTION B (Dose-Based) for operational simplicity:")
    print("  1. ML model predicts Bz and velocity")
    print("  2. Compute dose: D = 0.0121 × |Bz|^1.3 × √v × t")
    print("  3. Classify severity by dose range directly")
    print("  4. No ambiguity, no lookup tables needed")
    print()
    print("For whitepapers/documentation:")
    print("  • Still show example Bz ranges with explicit v=800 km/s note")
    print("  • Add velocity caveat: 'Thresholds scale with √v'")
    print("  • Emphasize that DOSE is the operational metric")
    print()


if __name__ == "__main__":
    main()
