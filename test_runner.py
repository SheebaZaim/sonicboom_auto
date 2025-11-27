# # test_runner.py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from propagate_frequency_domain import propagate_linear_fft, apply_turbulence_envelope
# from nonlinear_correction import  nonlinear_correction
# try:
#     from scipy.stats import rmse
# except ImportError:
#     rmse = None  # fallback if scipy or rmse not available
# import os
# import math

# def load_csv(csvfile):
#     df = pd.read_csv(csvfile)
#     return df['t'].values, df['p'].values

# def compute_metrics(reference_p, calc_p):
#     # return RMSE and percent error at peak over reference peak amplitude
#     err = calc_p - reference_p
#     rmse = np.sqrt(np.mean(err**2))
#     # percent error in peak amplitude
#     ref_peak = np.max(np.abs(reference_p))
#     calc_peak = np.max(np.abs(calc_p))
#     pct_err_peak = 100.0 * (calc_peak - ref_peak) / ref_peak if ref_peak != 0 else np.nan
#     return {'rmse': rmse, 'pct_err_peak': pct_err_peak, 'ref_peak': ref_peak, 'calc_peak': calc_peak}

# def run_case(input_csv, reference_csv=None, distance=15760.0, out_prefix='case', params=None):
#     t,p = load_csv(input_csv)
#     if params is None:
#         params = {}
#     fs_req = params.get('fs_req', 200000.0)  # sampling rate to resample to
#     # if needed, resample (assume input t uniformly sampled)
#     dt = 1.0/fs_req
#     t_uniform = np.arange(t[0], t[-1], dt)
#     p_uniform = np.interp(t_uniform, t, p)
#     # linear propagation
#     c0 = params.get('c0', 340.0)
#     t_out, p_lin = propagate_linear_fft(t_uniform, p_uniform, distance, c0=c0,
#                                         temp_c=params.get('temp_c',20.0),
#                                         rh=params.get('rh',50.0),
#                                         p_pa=params.get('p_pa',101325.0))
#     # optional turbulence
#     if params.get('apply_turbulence', False):
#         p_lin = apply_turbulence_envelope(p_lin, params.get('turb_sigma', 0.03), seed=params.get('seed', None))
#     # optional nonlinear correction
#     if params.get('apply_nonlinear', True):
#         p_calc = nonlinear_correction(
#     p_lin,
#     dx=params.get('dx', 5.0),            # e.g. 5 meters
#     dt=params.get('dt', 1e-5),           # small time step
#     nu=params.get('nu', 5e-5),           # tuned viscosity
#     n_steps=int(params.get('distance', 15760) / params.get('dx', 5.0))
# )
#     else:
#         p_calc = p_lin
        
#     # save results
#     os.makedirs('outputs', exist_ok=True)
#     pd.DataFrame({'t': t_out, 'p': p_calc}).to_csv(f'outputs/{out_prefix}_propagated.csv', index=False)
#     # plotting
#     plt.figure(figsize=(8,4))
#     plt.plot(t_out, p_calc, label='calculated')
#     if reference_csv:
#         tr, pr = load_csv(reference_csv)
#         pr_interp = np.interp(t_out, tr, pr)
#         plt.plot(t_out, pr_interp, label='reference', alpha=0.7)
#         metrics = compute_metrics(pr_interp, p_calc)
#     else:
#         metrics = compute_metrics(np.zeros_like(p_calc), p_calc)
#     plt.legend()
#     plt.xlabel('Time (s)')
#     plt.ylabel('Pressure (Pa)')
#     plt.title(f'Propagation result {out_prefix}')
#     plt.tight_layout()
#     plt.savefig(f'outputs/{out_prefix}_comparison.png', dpi=200)
#     plt.close()
#     # return metrics
#     return metrics

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', required=True)
#     parser.add_argument('--reference', required=False)
#     parser.add_argument('--distance', required=False, type=float, default=15760.0)
#     parser.add_argument('--out', required=False, default='case')
#     args = parser.parse_args()
#     params = {
#         'fs_req': 200000.0,
#         'apply_turbulence': False,
#         'apply_nonlinear': True,
#         'nonlin_intensity': 0.06,
#         'temp_c': 20.0,
#         'rh': 50.0,
#         'p_pa': 101325.0
#     }
#     metrics = run_case(args.input, args.reference, distance=args.distance, out_prefix=args.out, params=params)
#     print("Metrics:", metrics)

# # --- AUTO SUMMARY REPORT GENERATION ---

# summary_text = f"""
# SUMMARY REPORT
# =====================
# Input File     : {args.input}
# Reference File : {args.reference}
# Distance       : {args.distance}
# Output Prefix  : {args.out}

# Metrics
# -------
# RMSE               : {metrics['rmse']:.4f}
# Peak Error (%)     : {metrics['pct_err_peak']:.2f} %
# Reference Peak     : {metrics['ref_peak']:.4f}
# Calculated Peak    : {metrics['calc_peak']:.4f}

# Interpretation
# --------------
# • RMSE → Overall deviation. Lower = better match.
# • Peak Error (%) → Difference at the highest value.
# • Reference vs Calculated peak → How close the auto digitized result is to expected.

# Comment
# -------
# The summary is auto-generated based on comparison between digitized data and reference plot.
# Check the PNG comparison results in 'outputs/' folder.
# """

# # Save report
# report_path = f"outputs/{args.out}_summary.txt"
# # with open(report_path, "w") as f:
# #     f.write(summary_text)
# with open(report_path, "w", encoding="utf-8") as f:
#     f.write(summary_text)

# print(f"\n📄 Summary report generated at: {report_path}\n")

# test_runner.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from propagate_frequency_domain import propagate_linear_fft, apply_turbulence_envelope
from nonlinear_correction import nonlinear_correction
import os
import argparse

# -------- Utility Functions -------- #

def load_csv(csvfile):
    df = pd.read_csv(csvfile)
    return df['t'].values, df['p'].values

def compute_metrics(reference_p, calc_p):
    err = calc_p - reference_p
    rmse = np.sqrt(np.mean(err ** 2))
    ref_peak = np.max(np.abs(reference_p))
    calc_peak = np.max(np.abs(calc_p))
    pct_err_peak = 100.0 * (calc_peak - ref_peak) / ref_peak if ref_peak != 0 else np.nan
    return {
        'rmse': rmse,
        'pct_err_peak': pct_err_peak,
        'ref_peak': ref_peak,
        'calc_peak': calc_peak
    }

def run_case(input_csv, reference_csv=None, distance=15760.0, out_prefix='case', params=None):
    t, p = load_csv(input_csv)
    params = params or {}

    # Resampling
    dt = 1.0 / params.get('fs_req', 200000.0)
    t_uniform = np.arange(t[0], t[-1], dt)
    p_uniform = np.interp(t_uniform, t, p)

    # Linear propagation
    t_out, p_lin = propagate_linear_fft(
        t_uniform,
        p_uniform,
        distance,
        c0=params.get('c0', 340.0),
        temp_c=params.get('temp_c', 20.0),
        rh=params.get('rh', 50.0),
        p_pa=params.get('p_pa', 101325.0)
    )

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
            n_steps=int(distance / params.get('dx', 5.0))  # Correct nonlinear steps
        )
    else:
        p_calc = p_lin

    # Ensure output folder exists
    os.makedirs('outputs', exist_ok=True)

    # Save CSV
    out_csv_path = f'outputs/{out_prefix}_propagated.csv'
    pd.DataFrame({'t': t_out, 'p': p_calc}).to_csv(out_csv_path, index=False)

    # Plot comparison
    plt.figure(figsize=(8, 4))
    plt.plot(t_out, p_calc, label='Calculated')

    if reference_csv:
        tr, pr = load_csv(reference_csv)
        pr_interp = np.interp(t_out, tr, pr)
        plt.plot(t_out, pr_interp, label='Reference', alpha=0.7)
        metrics = compute_metrics(pr_interp, p_calc)
    else:
        metrics = compute_metrics(np.zeros_like(p_calc), p_calc)

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.title(f'Propagation result: {out_prefix}')
    plt.tight_layout()
    plt.savefig(f'outputs/{out_prefix}_comparison.png', dpi=200)
    plt.close()

    return metrics


# -------- Main Execution -------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--reference', required=False)
    parser.add_argument('--distance', type=float, default=15760.0)
    # parser.add_argument('--out', default=None)  # Allow dynamic output naming
    parser.add_argument('--out', nargs='?', const=None, default=None,
                    help="Optional output prefix. If not provided, auto-assigned from input file.")

    args = parser.parse_args()

    # Figure mapping logic
    base_input = os.path.splitext(os.path.basename(args.input))[0].lower()

    mapping = {
        "fig4.4": "fig4.5",
        "fig8b": "fig10"
    }

    # Dynamic output name
    if args.out:
        out_prefix = args.out
    else:
        out_prefix = mapping.get(base_input, base_input)

    print(f"\n📌 Assigned Output Prefix: {out_prefix}\n")

    params = {
        'fs_req': 200000.0,
        'apply_turbulence': False,
        'apply_nonlinear': True,
        'nonlin_intensity': 0.06,
        'temp_c': 20.0,
        'rh': 50.0,
        'p_pa': 101325.0
    }

    metrics = run_case(args.input, args.reference, args.distance, out_prefix, params)
    print("Metrics:", metrics)

    # Generate summary
    summary_text = f"""
SUMMARY REPORT
=====================
Input File     : {args.input}
Reference File : {args.reference}
Distance       : {args.distance}
Output Prefix  : {out_prefix}

Metrics
-------
RMSE               : {metrics['rmse']:.4f}
Peak Error (%)     : {metrics['pct_err_peak']:.2f} %
Reference Peak     : {metrics['ref_peak']:.4f}
Calculated Peak    : {metrics['calc_peak']:.4f}

Interpretation
--------------
• RMSE → Measures overall deviation (Lower = Better).
• Peak Error (%) → Difference at highest pressure point.
• Reference vs Calculated Peak → Accuracy of digitized waveform.

Comments
--------
✔ Correct auto-matching of output to target figure.
✔ Comparison plot saved.
✔ Summary generated automatically.
"""

    report_path = f"outputs/{out_prefix}_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n📄 Summary report saved at: {report_path}\n")
