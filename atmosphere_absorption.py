# atmosphere_absorption.py
import numpy as np

def air_properties(temperature_c=15.0, rel_humidity=50.0, pressure_pa=101325.0):
    """
    Return basic air properties (Kelvin, speed of sound, density) for given T, RH, p.
    Tunable: use more precise ISA formulas if needed.
    """
    T = temperature_c + 273.15
    # approximate speed of sound
    c = 331.3 * np.sqrt(1 + temperature_c / 273.15)
    rho = pressure_pa / (287.05 * T)
    return {'T': T, 'c': c, 'rho': rho, 'p': pressure_pa, 'RH': rel_humidity}

def bass_sutherland_alpha(f, T=293.15, p=101325.0, RH=50.0):
    """
    Practical frequency-dependent absorption coefficient (Np/m) approximating Bass & Sutherland
    f: Hz array or scalar
    T: Kelvin
    p: Pa
    RH: %
    This is a practical fit. Tune constants for <5% accuracy in your cases.
    """
    # Convert to more convenient units
    f = np.asarray(f, dtype=float)
    # reference conditions
    T0 = 293.15
    p0 = 101325.0
    # classical absorption ~ f^2 * T^(-3/2) * p^-1 (this is simplified)
    # molecular relaxation peaks around specific rotational/vibrational modes (O2, N2, H2O)
    # We'll implement an empirical sum of relaxation-like terms:
    # alpha = A * f^2 / (f^2 + f_r1^2) + B * f^2 / (f^2 + f_r2^2) + viscous_term * f^2
    # Constants below are starting guesses; tune for your conditions.
    A = 1e-12 * (p0/p) * (T0/T)**(1.5)
    B = 3e-12 * (p0/p) * (T0/T)**(1.2)
    visc = 1e-16 * (p0/p)
    # relaxation corner frequencies (Hz) — tune if needed
    f_r1 = 1e3 * (p/p0) * (T/T0)
    f_r2 = 5e3 * (p/p0) * (T/T0)
    alpha = A * f**2 / (f**2 + f_r1**2) + B * f**2 / (f**2 + f_r2**2) + visc * f**2
    # ensure non-negative
    return np.maximum(0.0, alpha)  # Np/m

def compute_alpha_for_freqs(freqs, temp_c=20.0, rh=50.0, p_pa=101325.0):
    T = temp_c + 273.15
    return bass_sutherland_alpha(freqs, T=T, p=p_pa, RH=rh)
