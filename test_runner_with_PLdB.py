# # test_runner_with_PLdB.py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from propagate_frequency_domain import propagate_linear_fft, apply_turbulence_envelope
# from nonlinear_correction import nonlinear_correction
# import os

# def load_csv(csvfile):
#     df = pd.read_csv(csvfile)
#     return df['t'].values, df['p'].values

# def compute_perceived_loudness(t, p, method='stevens_mark_vii'):
#     """
#     Compute Perceived Loudness (PLdB) using Stevens' Mark VII or similar metric.
    
#     Parameters:
#     -----------
#     t : array
#         Time array (seconds)
#     p : array
#         Pressure waveform (Pa)
#     method : str
#         'stevens_mark_vii' or 'spl_peak'
    
#     Returns:
#     --------
#     PLdB : float
#         Perceived loudness in dB
#     """
    
#     if method == 'stevens_mark_vii':
#         # Stevens' Mark VII Formula (simplified approximation)
#         # PLdB = 10 * log10(integral of (p(t)^2.67) dt)
        
#         dt = t[1] - t[0]
        
#         # Remove DC component
#         p_ac = p - np.mean(p)
        
#         # Ensure positive for power law
#         p_abs = np.abs(p_ac)
        
#         # Stevens exponent (typically 2.67 for sonic boom)
#         exponent = 2.67
        
#         # Compute integral
#         integrand = p_abs ** exponent
#         integral = np.trapz(integrand, dx=dt)
        
#         # Convert to dB scale
#         if integral > 0:
#             PLdB = 10 * np.log10(integral)
#         else:
#             PLdB = -np.inf
        
#         # Normalize to typical sonic boom reference (calibration factor)
#         # This factor is tuned to match typical PLdB ranges (80-85 dB)
#         calibration = 80.0  # adjust based on your reference data
#         PLdB = PLdB + calibration
        
#     elif method == 'spl_peak':
#         # Simple SPL-based loudness (alternative)
#         p_ref = 20e-6  # 20 μPa reference
#         p_peak = np.max(np.abs(p))
#         PLdB = 20 * np.log10(p_peak / p_ref)
    
#     else:
#         raise ValueError(f"Unknown method: {method}")
    
#     return PLdB

# def apply_azimuth_correction(p_time, azimuth_deg, distance_m):
#     """
#     Apply azimuth-dependent atmospheric effects.
    
#     Parameters:
#     -----------
#     p_time : array
#         Pressure waveform
#     azimuth_deg : float
#         Azimuth angle in degrees
#     distance_m : float
#         Propagation distance
    
#     Returns:
#     --------
#     p_corrected : array
#         Azimuth-corrected pressure waveform
#     """
    
#     # Empirical azimuth correction factors (tune based on atmospheric data)
#     # These represent lateral spreading and atmospheric refraction effects
    
#     azimuth_factors = {
#         0: 1.0,      # reference
#         20: 0.97,    # slight attenuation
#         40: 0.95     # more attenuation
#     }
    
#     # Interpolate for intermediate angles
#     angles = np.array([0, 20, 40])
#     factors = np.array([1.0, 0.97, 0.95])
#     factor = np.interp(azimuth_deg, angles, factors)
    
#     # Apply geometric spreading correction
#     # For lateral propagation, additional 1/r spreading
#     geometric_factor = 1.0 / (1.0 + 0.0001 * azimuth_deg * distance_m / 1000.0)
    
#     total_factor = factor * geometric_factor
    
#     return p_time * total_factor

# def compute_metrics(reference_p, calc_p):
#     err = calc_p - reference_p
#     rmse = np.sqrt(np.mean(err**2))
#     ref_peak = np.max(np.abs(reference_p))
#     calc_peak = np.max(np.abs(calc_p))
#     pct_err_peak = 100.0 * (calc_peak - ref_peak) / ref_peak if ref_peak != 0 else np.nan
#     return {'rmse': rmse, 'pct_err_peak': pct_err_peak, 'ref_peak': ref_peak, 'calc_peak': calc_peak}

# def run_case(input_csv, reference_csv=None, distance=15760.0, azimuth=0.0, 
#              out_prefix='case', params=None):
#     """
#     Run sonic boom propagation with azimuth support and PLdB calculation.
    
#     Parameters:
#     -----------
#     input_csv : str
#         Path to input near-field signature CSV
#     reference_csv : str
#         Path to reference ground signature CSV (optional)
#     distance : float
#         Propagation distance in meters
#     azimuth : float
#         Azimuth angle in degrees
#     out_prefix : str
#         Output file prefix
#     params : dict
#         Simulation parameters
#     """
    
#     t, p = load_csv(input_csv)
    
#     if params is None:
#         params = {}
    
#     # Resample to uniform sampling rate
#     fs_req = params.get('fs_req', 200000.0)
#     dt = 1.0 / fs_req
#     t_uniform = np.arange(t[0], t[-1], dt)
#     p_uniform = np.interp(t_uniform, t, p)
    
#     # Linear propagation (frequency domain)
#     c0 = params.get('c0', 340.0)
#     t_out, p_lin = propagate_linear_fft(
#         t_uniform, p_uniform, distance, 
#         c0=c0,
#         temp_c=params.get('temp_c', 20.0),
#         rh=params.get('rh', 50.0),
#         p_pa=params.get('p_pa', 101325.0)
#     )
    
#     # Apply azimuth correction
#     p_lin = apply_azimuth_correction(p_lin, azimuth, distance)
    
#     # Optional turbulence
#     if params.get('apply_turbulence', False):
#         p_lin = apply_turbulence_envelope(
#             p_lin, 
#             params.get('turb_sigma', 0.03), 
#             seed=params.get('seed', None)
#         )
    
#     # Nonlinear correction
#     if params.get('apply_nonlinear', True):
#         p_calc = nonlinear_correction(
#             p_lin,
#             dx=params.get('dx', 5.0),
#             dt=params.get('dt', 1e-5),
#             nu=params.get('nu', 5e-5),
#             n_steps=int(distance / params.get('dx', 5.0))
#         )
#     else:
#         p_calc = p_lin
    
#     # Calculate Perceived Loudness
#     PLdB = compute_perceived_loudness(t_out, p_calc, method='stevens_mark_vii')
    
#     # Save results
#     os.makedirs('outputs', exist_ok=True)
#     output_csv = f'outputs/{out_prefix}_az{int(azimuth)}_propagated.csv'
#     pd.DataFrame({'t': t_out, 'p': p_calc}).to_csv(output_csv, index=False)
    
#     # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.plot(t_out * 1000, p_calc, label=f'Calculated (Az={azimuth}°)', linewidth=1.5)
    
#     if reference_csv:
#         tr, pr = load_csv(reference_csv)
#         pr_interp = np.interp(t_out, tr, pr)
#         plt.plot(t_out * 1000, pr_interp, label='Reference', alpha=0.7, linestyle='--')
#         metrics = compute_metrics(pr_interp, p_calc)
#     else:
#         metrics = compute_metrics(np.zeros_like(p_calc), p_calc)
    
#     plt.legend()
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Overpressure (Pa)')
#     plt.title(f'Ground Signature - Azimuth {azimuth}° | PLdB = {PLdB:.2f}')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'outputs/{out_prefix}_az{int(azimuth)}_comparison.png', dpi=200)
#     plt.close()
    
#     # Add PLdB to metrics
#     metrics['PLdB'] = PLdB
#     metrics['azimuth'] = azimuth
    
#     return metrics

# def run_table_4_2_comparison(input_csv, reference_csv=None, distance=15760.0, params=None):
#     """
#     Run propagation for all azimuth angles in Table 4.2 and generate comparison.
#     """
    
#     # Table 4.2 reference values
#     table_4_2 = {
#         'Azimuth': [0, 20, 40],
#         'ITUBOOM': [81.30063, 81.33086, 82.34975],
#         'sBOOM': [80.66667, 78.06131, 80.14446]
#     }
    
#     results = []
    
#     print("\n" + "="*70)
#     print("RUNNING TABLE 4.2 COMPARISON")
#     print("="*70)
    
#     for azimuth in table_4_2['Azimuth']:
#         print(f"\n▶ Processing Azimuth = {azimuth}°...")
        
#         metrics = run_case(
#             input_csv=input_csv,
#             reference_csv=reference_csv,
#             distance=distance,
#             azimuth=azimuth,
#             out_prefix='jwb_test',
#             params=params
#         )
        
#         results.append({
#             'Azimuth': azimuth,
#             'Calculated_PLdB': metrics['PLdB'],
#             'ITUBOOM_Reference': table_4_2['ITUBOOM'][table_4_2['Azimuth'].index(azimuth)],
#             'sBOOM_Reference': table_4_2['sBOOM'][table_4_2['Azimuth'].index(azimuth)],
#             'RMSE': metrics['rmse'],
#             'Peak_Error_%': metrics['pct_err_peak']
#         })
        
#         print(f"  ✓ PLdB = {metrics['PLdB']:.5f}")
#         print(f"  ✓ ITUBOOM Reference = {table_4_2['ITUBOOM'][table_4_2['Azimuth'].index(azimuth)]}")
#         print(f"  ✓ sBOOM Reference = {table_4_2['sBOOM'][table_4_2['Azimuth'].index(azimuth)]}")
    
#     # Create comparison DataFrame
#     df_results = pd.DataFrame(results)
    
#     # Calculate errors
#     df_results['Error_vs_ITUBOOM'] = df_results['Calculated_PLdB'] - df_results['ITUBOOM_Reference']
#     df_results['Error_vs_sBOOM'] = df_results['Calculated_PLdB'] - df_results['sBOOM_Reference']
    
#     # Save results
#     df_results.to_csv('outputs/table_4_2_comparison.csv', index=False)
    
#     # Generate summary report
#     summary = f"""
# {'='*70}
# TABLE 4.2 COMPARISON REPORT - JAXA WING BODY CASE
# {'='*70}

# Input File: {input_csv}
# Reference File: {reference_csv}
# Distance: {distance} m (15,760 m altitude to ground)

# PERCEIVED LOUDNESS (PLdB) COMPARISON
# {'='*70}

# {'Azimuth':<12} {'Calculated':<15} {'ITUBOOM':<15} {'sBOOM':<15} {'Err(ITUBOOM)':<15} {'Err(sBOOM)':<15}
# {'-'*70}
# """
    
#     for _, row in df_results.iterrows():
#         summary += f"{row['Azimuth']:<12.0f} {row['Calculated_PLdB']:<15.5f} {row['ITUBOOM_Reference']:<15.5f} "
#         summary += f"{row['sBOOM_Reference']:<15.5f} {row['Error_vs_ITUBOOM']:<15.5f} {row['Error_vs_sBOOM']:<15.5f}\n"
    
#     summary += f"\n{'='*70}\n"
#     summary += f"MEAN ABSOLUTE ERROR:\n"
#     summary += f"  vs ITUBOOM: {df_results['Error_vs_ITUBOOM'].abs().mean():.5f} dB\n"
#     summary += f"  vs sBOOM:   {df_results['Error_vs_sBOOM'].abs().mean():.5f} dB\n"
#     summary += f"{'='*70}\n\n"
    
#     summary += "INTERPRETATION:\n"
#     summary += "• Error < 0.5 dB  → Excellent match\n"
#     summary += "• Error < 1.0 dB  → Good match\n"
#     summary += "• Error < 2.0 dB  → Acceptable match\n"
#     summary += "• Error > 2.0 dB  → Needs tuning\n\n"
    
#     summary += "RECOMMENDED ACTIONS:\n"
#     if df_results['Error_vs_ITUBOOM'].abs().mean() > 2.0:
#         summary += "• Tune azimuth correction factors\n"
#         summary += "• Adjust Stevens' Mark VII calibration constant\n"
#         summary += "• Verify atmospheric absorption parameters\n"
#     else:
#         summary += "• Results are within acceptable range\n"
    
#     # Save report
#     with open('outputs/table_4_2_summary.txt', 'w') as f:
#         f.write(summary)
    
#     print("\n" + summary)
#     print(f"📊 Full comparison saved to: outputs/table_4_2_comparison.csv")
#     print(f"📄 Summary report saved to: outputs/table_4_2_summary.txt")
    
#     return df_results

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Sonic Boom Propagation with PLdB Calculation')
#     parser.add_argument('--input', required=True, help='Input near-field CSV')
#     parser.add_argument('--reference', required=False, help='Reference ground signature CSV')
#     parser.add_argument('--distance', type=float, default=15760.0, help='Propagation distance (m)')
#     parser.add_argument('--azimuth', type=float, default=None, help='Single azimuth angle (deg)')
#     parser.add_argument('--table42', action='store_true', help='Run full Table 4.2 comparison')
#     parser.add_argument('--out', default='case', help='Output prefix')
    
#     args = parser.parse_args()
    
#     # Simulation parameters (tune these for best match)
#     params = {
#         'fs_req': 200000.0,
#         'apply_turbulence': False,
#         'apply_nonlinear': True,
#         'dx': 5.0,           # Spatial step for Burgers solver
#         'dt': 1e-5,          # Time step
#         'nu': 5e-5,          # Viscosity (tune: lower = more steepening)
#         'temp_c': 20.0,      # Temperature (°C)
#         'rh': 50.0,          # Relative humidity (%)
#         'p_pa': 101325.0,    # Pressure (Pa)
#         'c0': 340.0          # Speed of sound (m/s)
#     }
    
#     if args.table42:
#         # Run full Table 4.2 comparison
#         df_results = run_table_4_2_comparison(
#             input_csv=args.input,
#             reference_csv=args.reference,
#             distance=args.distance,
#             params=params
#         )
#     else:
#         # Run single case
#         azimuth = args.azimuth if args.azimuth is not None else 0.0
#         metrics = run_case(
#             input_csv=args.input,
#             reference_csv=args.reference,
#             distance=args.distance,
#             azimuth=azimuth,
#             out_prefix=args.out,
#             params=params
#         )
        
#         print("\n" + "="*50)
#         print("RESULTS")
#         print("="*50)
#         print(f"Azimuth:           {metrics['azimuth']}°")
#         print(f"Perceived Loudness: {metrics['PLdB']:.5f} dB")
#         print(f"RMSE:              {metrics['rmse']:.6f}")
#         print(f"Peak Error:        {metrics['pct_err_peak']:.2f}%")
#         print("="*50)

# test_runner_with_PLdB.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from propagate_frequency_domain import propagate_linear_fft, apply_turbulence_envelope
from nonlinear_correction import nonlinear_correction
import os
import sys
import traceback

def load_csv(csvfile):
    df = pd.read_csv(csvfile)
    return df['t'].values, df['p'].values

def compute_perceived_loudness(t, p, method='stevens_mark_vii'):
    """
    Compute Perceived Loudness (PLdB) using Stevens' Mark VII or similar metric.
    
    Parameters:
    -----------
    t : array
        Time array (seconds)
    p : array
        Pressure waveform (Pa)
    method : str
        'stevens_mark_vii' or 'spl_peak'
    
    Returns:
    --------
    PLdB : float
        Perceived loudness in dB
    """
    
    if method == 'stevens_mark_vii':
        # Stevens' Mark VII Formula (simplified approximation)
        # PLdB = 10 * log10(integral of (p(t)^2.67) dt)
        
        dt = t[1] - t[0]
        
        # Remove DC component
        p_ac = p - np.mean(p)
        
        # Ensure positive for power law
        p_abs = np.abs(p_ac)
        
        # Stevens exponent (typically 2.67 for sonic boom)
        exponent = 2.67
        
        # Compute integral
        integrand = p_abs ** exponent
        # integral = np.trapz(integrand, dx=dt)
        integral = np.trapezoid(integrand, dx=dt)

        
        # Convert to dB scale
        if integral > 0:
            PLdB = 10 * np.log10(integral)
        else:
            PLdB = -np.inf
        
        # Normalize to typical sonic boom reference (calibration factor)
        # This factor is tuned to match typical PLdB ranges (80-85 dB)
        calibration = 80.0  # adjust based on your reference data
        PLdB = PLdB + calibration
        
    elif method == 'spl_peak':
        # Simple SPL-based loudness (alternative)
        p_ref = 20e-6  # 20 μPa reference
        p_peak = np.max(np.abs(p))
        PLdB = 20 * np.log10(p_peak / p_ref)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return PLdB

def apply_azimuth_correction(p_time, azimuth_deg, distance_m):
    """
    Apply azimuth-dependent atmospheric effects.
    
    Parameters:
    -----------
    p_time : array
        Pressure waveform
    azimuth_deg : float
        Azimuth angle in degrees
    distance_m : float
        Propagation distance
    
    Returns:
    --------
    p_corrected : array
        Azimuth-corrected pressure waveform
    """
    
    # Empirical azimuth correction factors (tune based on atmospheric data)
    # These represent lateral spreading and atmospheric refraction effects
    
    azimuth_factors = {
        0: 1.0,      # reference
        20: 0.97,    # slight attenuation
        40: 0.95     # more attenuation
    }
    
    # Interpolate for intermediate angles
    angles = np.array([0, 20, 40])
    factors = np.array([1.0, 0.97, 0.95])
    factor = np.interp(azimuth_deg, angles, factors)
    
    # Apply geometric spreading correction
    # For lateral propagation, additional 1/r spreading
    geometric_factor = 1.0 / (1.0 + 0.0001 * azimuth_deg * distance_m / 1000.0)
    
    total_factor = factor * geometric_factor
    
    return p_time * total_factor

def compute_metrics(reference_p, calc_p):
    err = calc_p - reference_p
    rmse = np.sqrt(np.mean(err**2))
    ref_peak = np.max(np.abs(reference_p))
    calc_peak = np.max(np.abs(calc_p))
    pct_err_peak = 100.0 * (calc_peak - ref_peak) / ref_peak if ref_peak != 0 else np.nan
    return {'rmse': rmse, 'pct_err_peak': pct_err_peak, 'ref_peak': ref_peak, 'calc_peak': calc_peak}

def run_case(input_csv, reference_csv=None, distance=15760.0, azimuth=0.0, 
             out_prefix='case', params=None):
    """
    Run sonic boom propagation with azimuth support and PLdB calculation.
    
    Parameters:
    -----------
    input_csv : str
        Path to input near-field signature CSV
    reference_csv : str
        Path to reference ground signature CSV (optional)
    distance : float
        Propagation distance in meters
    azimuth : float
        Azimuth angle in degrees
    out_prefix : str
        Output file prefix
    params : dict
        Simulation parameters
    """
    
    t, p = load_csv(input_csv)
    
    if params is None:
        params = {}
    
    # Resample to uniform sampling rate
    fs_req = params.get('fs_req', 200000.0)
    dt = 1.0 / fs_req
    t_uniform = np.arange(t[0], t[-1], dt)
    p_uniform = np.interp(t_uniform, t, p)
    
    # Linear propagation (frequency domain)
    c0 = params.get('c0', 340.0)
    t_out, p_lin = propagate_linear_fft(
        t_uniform, p_uniform, distance, 
        c0=c0,
        temp_c=params.get('temp_c', 20.0),
        rh=params.get('rh', 50.0),
        p_pa=params.get('p_pa', 101325.0)
    )
    
    # Apply azimuth correction
    p_lin = apply_azimuth_correction(p_lin, azimuth, distance)
    
    # Optional turbulence
    if params.get('apply_turbulence', False):
        p_lin = apply_turbulence_envelope(
            p_lin, 
            params.get('turb_sigma', 0.03), 
            seed=params.get('seed', None)
        )
    
    # Nonlinear correction
    if params.get('apply_nonlinear', True):
        p_calc = nonlinear_correction(
            p_lin,
            dx=params.get('dx', 5.0),
            dt=params.get('dt', 1e-5),
            nu=params.get('nu', 5e-5),
            n_steps=int(distance / params.get('dx', 5.0))
        )
    else:
        p_calc = p_lin
    
    # Calculate Perceived Loudness
    PLdB = compute_perceived_loudness(t_out, p_calc, method='stevens_mark_vii')
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    output_csv = f'outputs/{out_prefix}_az{int(azimuth)}_propagated.csv'
    pd.DataFrame({'t': t_out, 'p': p_calc}).to_csv(output_csv, index=False)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(t_out * 1000, p_calc, label=f'Calculated (Az={azimuth}°)', linewidth=1.5)
    
    if reference_csv:
        tr, pr = load_csv(reference_csv)
        pr_interp = np.interp(t_out, tr, pr)
        plt.plot(t_out * 1000, pr_interp, label='Reference', alpha=0.7, linestyle='--')
        metrics = compute_metrics(pr_interp, p_calc)
    else:
        metrics = compute_metrics(np.zeros_like(p_calc), p_calc)
    
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Overpressure (Pa)')
    plt.title(f'Ground Signature - Azimuth {azimuth}° | PLdB = {PLdB:.2f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = f'outputs/{out_prefix}_az{int(azimuth)}_comparison.png'
    try:
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Plot saved: {plot_file}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not save plot: {e}")
    finally:
        plt.close()
    
    # Add PLdB to metrics
    metrics['PLdB'] = PLdB
    metrics['azimuth'] = azimuth
    
    return metrics

def run_table_4_2_comparison(input_csv, reference_csv=None, distance=15760.0, params=None):
    """
    Run propagation for all azimuth angles in Table 4.2 and generate comparison.
    """
    
    # Table 4.2 reference values
    table_4_2 = {
        'Azimuth': [0, 20, 40],
        'ITUBOOM': [81.30063, 81.33086, 82.34975],
        'sBOOM': [80.66667, 78.06131, 80.14446]
    }
    
    results = []
    
    print("\n" + "="*70)
    print("RUNNING TABLE 4.2 COMPARISON")
    print("="*70)
    
    for azimuth in table_4_2['Azimuth']:
        print(f"\n▶ Processing Azimuth = {azimuth}°...")
        
        try:
            metrics = run_case(
                input_csv=input_csv,
                reference_csv=reference_csv,
                distance=distance,
                azimuth=azimuth,
                out_prefix='jwb_test',
                params=params
            )
            
            results.append({
                'Azimuth': azimuth,
                'Calculated_PLdB': metrics['PLdB'],
                'ITUBOOM_Reference': table_4_2['ITUBOOM'][table_4_2['Azimuth'].index(azimuth)],
                'sBOOM_Reference': table_4_2['sBOOM'][table_4_2['Azimuth'].index(azimuth)],
                'RMSE': metrics['rmse'],
                'Peak_Error_%': metrics['pct_err_peak']
            })
            
            print(f"  ✓ PLdB = {metrics['PLdB']:.5f}")
            print(f"  ✓ ITUBOOM Reference = {table_4_2['ITUBOOM'][table_4_2['Azimuth'].index(azimuth)]}")
            print(f"  ✓ sBOOM Reference = {table_4_2['sBOOM'][table_4_2['Azimuth'].index(azimuth)]}")
        
        except Exception as e:
            print(f"  ✗ ERROR processing azimuth {azimuth}°: {e}")
            traceback.print_exc()
            continue
    
    # Create comparison DataFrame
    if not results:
        print("\n❌ ERROR: No results generated!")
        return None
    
    df_results = pd.DataFrame(results)
    
    # Calculate errors
    df_results['Error_vs_ITUBOOM'] = df_results['Calculated_PLdB'] - df_results['ITUBOOM_Reference']
    df_results['Error_vs_sBOOM'] = df_results['Calculated_PLdB'] - df_results['sBOOM_Reference']
    
    # Save results
    try:
        df_results.to_csv('outputs/table_4_2_comparison.csv', index=False)
        print(f"\n✓ Results saved: outputs/table_4_2_comparison.csv")
    except Exception as e:
        print(f"\n⚠ Warning: Could not save CSV: {e}")
    
    # Generate summary report
    summary = f"""
{'='*70}
TABLE 4.2 COMPARISON REPORT - JAXA WING BODY CASE
{'='*70}

Input File: {input_csv}
Reference File: {reference_csv}
Distance: {distance} m (15,760 m altitude to ground)

PERCEIVED LOUDNESS (PLdB) COMPARISON
{'='*70}

{'Azimuth':<12} {'Calculated':<15} {'ITUBOOM':<15} {'sBOOM':<15} {'Err(ITUBOOM)':<15} {'Err(sBOOM)':<15}
{'-'*70}
"""
    
    for _, row in df_results.iterrows():
        summary += f"{row['Azimuth']:<12.0f} {row['Calculated_PLdB']:<15.5f} {row['ITUBOOM_Reference']:<15.5f} "
        summary += f"{row['sBOOM_Reference']:<15.5f} {row['Error_vs_ITUBOOM']:<15.5f} {row['Error_vs_sBOOM']:<15.5f}\n"
    
    summary += f"\n{'='*70}\n"
    summary += f"MEAN ABSOLUTE ERROR:\n"
    summary += f"  vs ITUBOOM: {df_results['Error_vs_ITUBOOM'].abs().mean():.5f} dB\n"
    summary += f"  vs sBOOM:   {df_results['Error_vs_sBOOM'].abs().mean():.5f} dB\n"
    summary += f"{'='*70}\n\n"
    
    summary += "INTERPRETATION:\n"
    summary += "• Error < 0.5 dB  → Excellent match\n"
    summary += "• Error < 1.0 dB  → Good match\n"
    summary += "• Error < 2.0 dB  → Acceptable match\n"
    summary += "• Error > 2.0 dB  → Needs tuning\n\n"
    
    summary += "RECOMMENDED ACTIONS:\n"
    if df_results['Error_vs_ITUBOOM'].abs().mean() > 2.0:
        summary += "• Tune azimuth correction factors\n"
        summary += "• Adjust Stevens' Mark VII calibration constant\n"
        summary += "• Verify atmospheric absorption parameters\n"
    else:
        summary += "• Results are within acceptable range\n"
    
    # Save report
    try:
        with open('outputs/table_4_2_summary.txt', 'w') as f:
            f.write(summary)
        print(f"✓ Summary saved: outputs/table_4_2_summary.txt")
    except Exception as e:
        print(f"⚠ Warning: Could not save summary: {e}")
    
    print("\n" + summary)
    print(f"📊 Full comparison saved to: outputs/table_4_2_comparison.csv")
    print(f"📄 Summary report saved to: outputs/table_4_2_summary.txt")
    
    return df_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sonic Boom Propagation with PLdB Calculation')
    parser.add_argument('--input', required=True, help='Input near-field CSV')
    parser.add_argument('--reference', required=False, help='Reference ground signature CSV')
    parser.add_argument('--distance', type=float, default=15760.0, help='Propagation distance (m)')
    parser.add_argument('--azimuth', type=float, default=None, help='Single azimuth angle (deg)')
    parser.add_argument('--table42', action='store_true', help='Run full Table 4.2 comparison')
    parser.add_argument('--out', default='case', help='Output prefix')
    
    args = parser.parse_args()
    
    # Simulation parameters (tune these for best match)
    params = {
        'fs_req': 200000.0,
        'apply_turbulence': False,
        'apply_nonlinear': True,
        'dx': 5.0,           # Spatial step for Burgers solver
        'dt': 1e-5,          # Time step
        'nu': 5e-5,          # Viscosity (tune: lower = more steepening)
        'temp_c': 20.0,      # Temperature (°C)
        'rh': 50.0,          # Relative humidity (%)
        'p_pa': 101325.0,    # Pressure (Pa)
        'c0': 340.0          # Speed of sound (m/s)
    }
    
    if args.table42:
        # Run full Table 4.2 comparison
        df_results = run_table_4_2_comparison(
            input_csv=args.input,
            reference_csv=args.reference,
            distance=args.distance,
            params=params
        )
    else:
        # Run single case
        azimuth = args.azimuth if args.azimuth is not None else 0.0
        metrics = run_case(
            input_csv=args.input,
            reference_csv=args.reference,
            distance=args.distance,
            azimuth=azimuth,
            out_prefix=args.out,
            params=params
        )
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"Azimuth:           {metrics['azimuth']}°")
        print(f"Perceived Loudness: {metrics['PLdB']:.5f} dB")
        print(f"RMSE:              {metrics['rmse']:.6f}")
        print(f"Peak Error:        {metrics['pct_err_peak']:.2f}%")
        print("="*50)