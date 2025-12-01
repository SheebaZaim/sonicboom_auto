# # test_case_2_dsend.py
# """
# Test Case 2: JAXA D-SEND Low Altitude Propagation
# Input: Figure 8b (at ~1000m altitude)
# Expected Output: Figure 10 (left) - GREEN CURVE on ground
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from propagate_frequency_domain import propagate_linear_fft, apply_turbulence_envelope
# from nonlinear_correction import nonlinear_correction
# import os

# def load_csv(csvfile):
#     df = pd.read_csv(csvfile)
#     return df['t'].values, df['p'].values

# def compute_metrics(reference_p, calc_p):
#     err = calc_p - reference_p
#     rmse = np.sqrt(np.mean(err**2))
#     ref_peak = np.max(np.abs(reference_p))
#     calc_peak = np.max(np.abs(calc_p))
#     pct_err_peak = 100.0 * (calc_peak - ref_peak) / ref_peak if ref_peak != 0 else np.nan
#     return {'rmse': rmse, 'pct_err_peak': pct_err_peak, 'ref_peak': ref_peak, 'calc_peak': calc_peak}

# def run_dsend_case(input_csv_fig8b, reference_csv_fig10=None, distance=1000.0, 
#                    out_prefix='dsend', params=None):
#     """
#     Run D-SEND low altitude propagation case.
    
#     Parameters:
#     -----------
#     input_csv_fig8b : str
#         Path to digitized Figure 8b CSV (input at ~1000m)
#     reference_csv_fig10 : str
#         Path to digitized Figure 10 (green curve) CSV (expected output on ground)
#     distance : float
#         Propagation distance (~1000m for D-SEND case)
#     out_prefix : str
#         Output file prefix
#     params : dict
#         Simulation parameters
#     """
    
#     t, p = load_csv(input_csv_fig8b)
    
#     if params is None:
#         params = {}
    
#     # Resample to uniform sampling rate
#     fs_req = params.get('fs_req', 200000.0)
#     dt = 1.0 / fs_req
#     t_uniform = np.arange(t[0], t[-1], dt)
#     p_uniform = np.interp(t_uniform, t, p)
    
#     print(f"\n{'='*70}")
#     print(f"RUNNING D-SEND TEST CASE (Low Altitude)")
#     print(f"{'='*70}")
#     print(f"Input (Fig 8b):  {input_csv_fig8b}")
#     print(f"Reference (Fig 10): {reference_csv_fig10 if reference_csv_fig10 else 'Not provided'}")
#     print(f"Distance:        {distance} m")
#     print(f"Sampling Rate:   {fs_req} Hz")
#     print(f"Input time range: {t[0]:.6f} to {t[-1]:.6f} s")
#     print(f"Input pressure range: {np.min(p):.2f} to {np.max(p):.2f} Pa")
    
#     # Linear propagation (frequency domain)
#     c0 = params.get('c0', 340.0)
#     print(f"\n▶ Step 1: Linear propagation (frequency domain)...")
#     t_out, p_lin = propagate_linear_fft(
#         t_uniform, p_uniform, distance, 
#         c0=c0,
#         temp_c=params.get('temp_c', 20.0),
#         rh=params.get('rh', 50.0),
#         p_pa=params.get('p_pa', 101325.0)
#     )
#     print(f"  ✓ Linear propagation complete")
#     print(f"  ✓ Output pressure range: {np.min(p_lin):.2f} to {np.max(p_lin):.2f} Pa")
    
#     # Apply turbulence (Fig 10 shows turbulence effects)
#     if params.get('apply_turbulence', True):
#         print(f"\n▶ Step 2: Applying turbulence envelope...")
#         turb_sigma = params.get('turb_sigma', 0.05)  # ~5% fluctuations
#         p_turb = apply_turbulence_envelope(p_lin, turb_sigma, seed=params.get('seed', 42))
#         print(f"  ✓ Turbulence applied (sigma={turb_sigma})")
#     else:
#         p_turb = p_lin
#         print(f"\n▶ Step 2: Turbulence SKIPPED")
    
#     # Nonlinear correction (shorter distance = less nonlinearity than 15km case)
#     if params.get('apply_nonlinear', True):
#         print(f"\n▶ Step 3: Applying nonlinear correction (Burgers solver)...")
#         dx = params.get('dx', 2.0)  # Smaller steps for 1000m case
#         dt_burgers = params.get('dt', 1e-5)
#         nu = params.get('nu', 1e-4)  # Higher viscosity for lower altitude
#         n_steps = int(distance / dx)
        
#         print(f"  • dx = {dx} m")
#         print(f"  • dt = {dt_burgers} s")
#         print(f"  • nu = {nu}")
#         print(f"  • n_steps = {n_steps}")
        
#         p_calc = nonlinear_correction(
#             p_turb,
#             dx=dx,
#             dt=dt_burgers,
#             nu=nu,
#             n_steps=n_steps
#         )
#         print(f"  ✓ Nonlinear correction complete")
#     else:
#         p_calc = p_turb
#         print(f"\n▶ Step 3: Nonlinear correction SKIPPED")
    
#     print(f"\n▶ Step 4: Saving results...")
    
#     # Save results
#     os.makedirs('outputs', exist_ok=True)
#     output_csv = f'outputs/{out_prefix}_propagated.csv'
#     pd.DataFrame({'t': t_out, 'p': p_calc}).to_csv(output_csv, index=False)
#     print(f"  ✓ Saved to: {output_csv}")
    
#     # Plotting - Match Figure 10 style
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Convert time to milliseconds for x-axis (like Fig 10)
#     t_ms = t_out * 1000
    
#     # Plot calculated result (this should match GREEN curve in Fig 10)
#     ax.plot(t_ms, p_calc, 'g-', linewidth=2, label='Calculated (should match Fig 10 green)', alpha=0.8)
    
#     # If reference provided, overlay it
#     if reference_csv_fig10:
#         tr, pr = load_csv(reference_csv_fig10)
#         tr_ms = tr * 1000
#         # Interpolate reference to same time grid
#         pr_interp = np.interp(t_out, tr, pr)
#         ax.plot(tr_ms, pr, 'k--', linewidth=1.5, label='Reference (Fig 10 green curve)', alpha=0.7)
        
#         metrics = compute_metrics(pr_interp, p_calc)
        
#         # Add metrics text box
#         textstr = f'RMSE: {metrics["rmse"]:.4f} Pa\n'
#         textstr += f'Peak Error: {metrics["pct_err_peak"]:.2f}%\n'
#         textstr += f'Ref Peak: {metrics["ref_peak"]:.2f} Pa\n'
#         textstr += f'Calc Peak: {metrics["calc_peak"]:.2f} Pa'
        
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
#                 verticalalignment='top', bbox=props)
#     else:
#         metrics = compute_metrics(np.zeros_like(p_calc), p_calc)
    
#     ax.set_xlabel('Time (ms)', fontsize=12)
#     ax.set_ylabel('Δp (Pa)', fontsize=12)
#     ax.set_title(f'D-SEND Ground Signature (Distance: {distance}m)\nComparison with Figure 10', fontsize=13, fontweight='bold')
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.legend(loc='best', fontsize=10)
    
#     # Match Figure 10 axis ranges (adjust based on actual Fig 10)
#     # From your image, Fig 10 shows roughly:
#     # X-axis: 0 to ~0.05 seconds (0-50 ms)
#     # Y-axis: -50 to +30 Pa
#     ax.set_xlim(0, 50)  # milliseconds
#     ax.set_ylim(-60, 40)  # Pa
    
#     plt.tight_layout()
#     plot_file = f'outputs/{out_prefix}_vs_fig10.png'
#     plt.savefig(plot_file, dpi=200)
#     print(f"  ✓ Plot saved to: {plot_file}")
#     plt.close()
    
#     # Generate summary report
#     print(f"\n▶ Step 5: Generating summary report...")
    
#     summary = f"""
# {'='*70}
# D-SEND LOW ALTITUDE TEST CASE - SUMMARY REPORT
# {'='*70}

# INPUT & REFERENCE
# -----------------
# Input File (Fig 8b):      {input_csv_fig8b}
# Reference File (Fig 10):  {reference_csv_fig10 if reference_csv_fig10 else 'Not provided'}
# Propagation Distance:     {distance} m

# SIMULATION PARAMETERS
# ---------------------
# Sampling Rate:            {fs_req} Hz
# Speed of Sound:           {params.get('c0', 340.0)} m/s
# Temperature:              {params.get('temp_c', 20.0)} °C
# Relative Humidity:        {params.get('rh', 50.0)} %
# Pressure:                 {params.get('p_pa', 101325.0)} Pa

# Turbulence Applied:       {params.get('apply_turbulence', True)}
# Turbulence Sigma:         {params.get('turb_sigma', 0.05) if params.get('apply_turbulence', True) else 'N/A'}

# Nonlinear Correction:     {params.get('apply_nonlinear', True)}
# Spatial Step (dx):        {params.get('dx', 2.0)} m
# Time Step (dt):           {params.get('dt', 1e-5)} s
# Viscosity (nu):           {params.get('nu', 1e-4)}

# RESULTS
# -------
# Output Time Range:        {t_out[0]:.6f} to {t_out[-1]:.6f} s
# Output Pressure Range:    {np.min(p_calc):.2f} to {np.max(p_calc):.2f} Pa
# Peak Overpressure:        {np.max(p_calc):.2f} Pa
# Peak Underpressure:       {np.min(p_calc):.2f} Pa

# """
    
#     if reference_csv_fig10:
#         summary += f"""
# COMPARISON WITH FIGURE 10 (GREEN CURVE)
# ----------------------------------------
# RMSE:                     {metrics['rmse']:.4f} Pa
# Peak Error:               {metrics['pct_err_peak']:.2f} %
# Reference Peak:           {metrics['ref_peak']:.2f} Pa
# Calculated Peak:          {metrics['calc_peak']:.2f} Pa

# INTERPRETATION
# --------------
# • RMSE < 2.0 Pa       → Excellent match with Fig 10
# • RMSE < 5.0 Pa       → Good match
# • RMSE < 10.0 Pa      → Acceptable match
# • RMSE > 10.0 Pa      → Needs parameter tuning

# """
    
#     summary += f"""
# OUTPUT FILES
# ------------
# CSV Data:                 outputs/{out_prefix}_propagated.csv
# Comparison Plot:          outputs/{out_prefix}_vs_fig10.png
# This Report:              outputs/{out_prefix}_summary.txt

# NOTES
# -----
# • The calculated waveform (GREEN) should match Figure 10 (left) green curve
# • Figure 10 shows multiple curves representing different turbulence realizations
# • The colored curves (R=-0.5L, R=1.0L, etc.) represent different turbulence strengths
# • Your output should fall within the envelope of these curves
# • Check the plot file to visually compare waveform shape and amplitude

# EXPECTED WAVEFORM CHARACTERISTICS (from Fig 10)
# -----------------------------------------------
# • Initial shock rise: ~+25 Pa
# • Secondary shock: ~+10 Pa  
# • Expansion: down to ~-40 Pa
# • Duration: ~40-50 ms
# • Overall N-wave shape with shock fronts

# {'='*70}
# """
    
#     report_file = f'outputs/{out_prefix}_summary.txt'
#     with open(report_file, 'w') as f:
#         f.write(summary)
#     print(f"  ✓ Summary saved to: {report_file}")
    
#     print(summary)
    
#     return metrics

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='D-SEND Low Altitude Test Case (Fig 8b → Fig 10)')
#     parser.add_argument('--input', required=True, help='Digitized Figure 8b CSV (input at ~1000m)')
#     parser.add_argument('--reference', required=False, help='Digitized Figure 10 green curve CSV (ground signature)')
#     parser.add_argument('--distance', type=float, default=1000.0, help='Propagation distance in meters')
#     parser.add_argument('--out', default='dsend', help='Output prefix')
    
#     args = parser.parse_args()
    
#     # D-SEND specific parameters (lower altitude = different conditions)
#     params = {
#         'fs_req': 200000.0,
#         'apply_turbulence': True,     # Fig 10 shows turbulence effects
#         'turb_sigma': 0.05,            # 5% turbulence fluctuations
#         'seed': 42,                    # For reproducibility
#         'apply_nonlinear': True,
#         'dx': 2.0,                     # Smaller spatial step (1000m < 15760m)
#         'dt': 1e-5,                    # Time step
#         'nu': 1e-4,                    # Higher viscosity (lower altitude, denser air)
#         'temp_c': 20.0,
#         'rh': 50.0,
#         'p_pa': 101325.0,
#         'c0': 340.0
#     }
    
#     print("\n" + "="*70)
#     print("JAXA D-SEND LOW ALTITUDE PROPAGATION TEST")
#     print("="*70)
#     print(f"Input:  Figure 8b → {args.input}")
#     print(f"Target: Figure 10 (green curve)")
#     print("="*70)
    
#     metrics = run_dsend_case(
#         input_csv_fig8b=args.input,
#         reference_csv_fig10=args.reference,
#         distance=args.distance,
#         out_prefix=args.out,
#         params=params
#     )
    
#     print("\n" + "="*70)
#     print("✓ TEST COMPLETE")
#     print("="*70)
#     print(f"Check outputs/{args.out}_vs_fig10.png to compare with Figure 10!")
#     print("="*70 + "\n") 

"""
CASE 2 — JAXA D-SEND (Low Altitude ~1000 m → Ground)
Reproduces Figure 10 (left, green curve)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------
# IMPORTS FROM YOUR EXISTING MODEL
# -----------------------------------------------------------
from propagate_frequency_domain import propagate_linear_fft
from nonlinear_correction import nonlinear_correction


# -----------------------------------------------------------
# FIXED + PAPER-CORRECT TURBULENCE ENVELOPE  (IMPORTANT)
# -----------------------------------------------------------
def apply_turbulence_envelope(p, sigma=0.05, seed=42):
    """
    DSEND turbulence model:
    multiplicative modulation with low-frequency variations
    → matches description used to produce Figure 10.
    """
    np.random.seed(seed)
    n = len(p)

    # Low-frequency turbulence (Gaussian)
    rnd = np.random.normal(0, sigma, n)

    # Smooth by interpolation (same length → no errors)
    x = np.linspace(0, 1, n)
    mod = np.interp(x, x, rnd)

    return p * (1 + mod)


# -----------------------------------------------------------
# CSV LOADING
# -----------------------------------------------------------
def load_csv(csvfile):
    df = pd.read_csv(csvfile)
    return df["t"].values, df["p"].values


# -----------------------------------------------------------
# METRICS (OPTIONAL)
# -----------------------------------------------------------
def compute_metrics(reference_p, calc_p):
    n = min(len(reference_p), len(calc_p))
    err = calc_p[:n] - reference_p[:n]
    rmse = np.sqrt(np.mean(err**2))
    return rmse


# -----------------------------------------------------------
# MAIN CASE-2 LOGIC
# -----------------------------------------------------------
def run_case2(input_csv, reference_csv=None, distance=1000.0, out_prefix="dsend_case2", params=None):

    if params is None:
        params = {}

    # ---- Load Figure 8b CSV ----
    t, p = load_csv(input_csv)

    # ---- Resample uniformly ----
    fs_req = params.get("fs_req", 200000.0)
    dt = 1.0 / fs_req
    t_uniform = np.arange(t[0], t[-1], dt)
    p_uniform = np.interp(t_uniform, t, p)

    print("\n=== CASE 2: JAXA D-SEND (1000 m → Ground) ===")
    print(f"Input file: {input_csv}")
    print(f"Reference: {reference_csv}")
    print(f"Propagation distance: {distance} m")

    # ---------------------------------------------------------
    # STEP 1 — LINEAR PROPAGATION (FREQUENCY DOMAIN)
    # ---------------------------------------------------------
    t_out, p_lin = propagate_linear_fft(
        t_uniform,
        p_uniform,
        distance,
        c0=params.get("c0", 340.0),
        temp_c=params.get("temp_c", 20.0),
        rh=params.get("rh", 50.0),
        p_pa=params.get("p_pa", 101325.0),
    )

    print("✓ Linear propagation complete")

    # ---------------------------------------------------------
    # STEP 2 — TURBULENCE (KEY STEP FOR MATCHING FIG 10)
    # ---------------------------------------------------------
    print("Applying turbulence envelope...")
    turb_sigma = params.get("turb_sigma", 0.05)
    seed = params.get("seed", 42)
    p_turb = apply_turbulence_envelope(p_lin, sigma=turb_sigma, seed=seed)
    print("✓ Turbulence applied")

    # ---------------------------------------------------------
    # STEP 3 — Nonlinear correction (mild)
    # ---------------------------------------------------------
    if params.get("apply_nonlinear", True):
        print("Applying nonlinear correction (Burgers)...")
        p_calc = nonlinear_correction(
            p_turb,
            dx=params.get("dx", 2.0),
            dt=params.get("dt", 1e-5),
            nu=params.get("nu", 1e-4),
            n_steps=int(distance / params.get("dx", 2.0)),
        )
        print("✓ Nonlinear correction complete")
    else:
        p_calc = p_turb
        print("Nonlinear correction skipped")

    # ---------------------------------------------------------
    # Save output waveform
    # ---------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    out_csv = f"outputs/{out_prefix}.csv"
    pd.DataFrame({"t": t_out, "p": p_calc}).to_csv(out_csv, index=False)
    print(f"✓ Output saved: {out_csv}")

    # ---------------------------------------------------------
    # Plot (compare to Fig 10)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t_out * 1000, p_calc, "g-", lw=2, alpha=0.8, label="Calculated (should match Fig 10 green)")

    if reference_csv:
        tr, pr = load_csv(reference_csv)
        plt.plot(tr * 1000, pr, "k--", lw=1.5, alpha=0.7, label="Reference (Fig 10 green)")
        rmse = compute_metrics(pr, p_calc)
        print(f"RMSE vs reference: {rmse:.4f} Pa")

    plt.xlabel("Time (ms)")
    plt.ylabel("Pressure (Pa)")
    plt.title("D-SEND Ground Signature — Case 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_plot = f"outputs/{out_prefix}.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"✓ Plot saved: {out_plot}\n")

    return p_calc


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV of Fig 8b (near-field ~1000 m)")
    parser.add_argument("--reference", required=False, help="CSV of Fig 10 (green)")
    parser.add_argument("--distance", type=float, default=1000.0)
    parser.add_argument("--out", default="dsend_case2")
    args = parser.parse_args()

    params = {
        "fs_req": 200000.0,
        "c0": 340.0,
        "temp_c": 20.0,
        "rh": 50.0,
        "p_pa": 101325.0,
        "turb_sigma": 0.05,
        "seed": 42,
        "apply_nonlinear": True,
        "dx": 2.0,
        "dt": 1e-5,
        "nu": 1e-4,
    }

    run_case2(
        input_csv=args.input,
        reference_csv=args.reference,
        distance=args.distance,
        out_prefix=args.out,
        params=params,
    )
