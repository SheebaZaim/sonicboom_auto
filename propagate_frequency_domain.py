# propagate_frequency_domain.py
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from atmosphere_absorption import compute_alpha_for_freqs

# def propagate_linear_fft(t_in, p_in, distance_m, c0=340.0, temp_c=20.0, rh=50.0, p_pa=101325.0):
#     """
#     Perform linear propagation in frequency domain: apply phase shift and amplitude attenuation
#     using alpha(f) from atmosphere model. Returns propagated time series at same sampling grid.
#     """
#     n = len(t_in)
#     dt = t_in[1] - t_in[0]
#     fs = 1.0/dt
#     freqs = rfftfreq(n, dt)
#     Pf = rfft(p_in)
#     # Compute alpha (Np/m)
#     alpha = compute_alpha_for_freqs(freqs, temp_c=temp_c, rh=rh, p_pa=p_pa)
#     # apply propagation: amplitude attenuation exp(-alpha * distance) and phase shift exp(-i*2pi*f*travel_time)
#     travel_time = distance_m / c0
#     H = np.exp(-alpha * distance_m) * np.exp(-1j * 2*np.pi * freqs * travel_time)
#     Pf_out = Pf * H
#     p_out = out  

def propagate_linear_fft(t_in, p_in, distance_m, c0=340.0, temp_c=20.0, rh=50.0, p_pa=101325.0):
    """
    Perform linear propagation in frequency domain: apply phase shift and amplitude attenuation
    using alpha(f) from atmosphere model. Returns propagated time series at same sampling grid.
    """
    from numpy.fft import rfft, irfft, rfftfreq
    import numpy as np

    n = len(t_in)
    dt = t_in[1] - t_in[0]
    fs = 1.0 / dt

    freqs = rfftfreq(n, dt)
    Pf = rfft(p_in)

    # Compute alpha (Np/m)
    alpha = compute_alpha_for_freqs(freqs, temp_c=temp_c, rh=rh, p_pa=p_pa)

    # Apply propagation: amplitude attenuation and phase shift
    travel_time = distance_m / c0
    H = np.exp(-alpha * distance_m) * np.exp(-1j * 2 * np.pi * freqs * travel_time)
    Pf_out = Pf * H

    # Inverse FFT to time domain
    # p_out = irfft(Pf_out, n=n)

    # return t_in, p_out  # return both time and propagated pressure

    p_out = irfft(Pf_out, n=len(p_in))
    return t_in, p_out
    


def apply_turbulence_envelope(p_time, ensemble_sigma, seed=None):
    """
    Multiply the time waveform by a slowly varying envelope representing turbulence focusing/defocusing.
    ensemble_sigma: standard deviation of log-amplitude (e.g., 0.05 ~ 5% fluctuations)
    """
    np.random.seed(seed)
    n = len(p_time)
    # slow varying low-freq modulator (use random low-pass)
    freqs = np.fft.rfftfreq(n, d=1.0/len(p_time))
    mod = np.interp(np.linspace(0,1,n), np.linspace(0,1,n), np.random.normal(0,1,10))
    mod = np.convolve(np.abs(mod), np.ones(1000)/1000, mode='same')
    mod = (mod - np.mean(mod)) / np.std(mod)
    amp = np.exp(ensemble_sigma * mod)
    return p_time * amp
