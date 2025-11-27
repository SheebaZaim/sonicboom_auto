# # nonlinear_correction.py
# import numpy as np
# from scipy.signal import savgol_filter

# def empirical_shock_sharpening(p_time, method='empirical', intensity=0.05):
#     """
#     Very simple empirical sharpening: amplify high gradients to approximate nonlinear steepening.
#     intensity: how much to sharpen (0..0.2)
#     """
#     dp = np.gradient(p_time)
#     dp2 = dp.copy()
#     # boost where gradient is large
#     boost = 1.0 + intensity * np.tanh(np.abs(dp)/np.std(dp))
#     # integrate back (approx)
#     p_new = np.cumsum(dp * boost)
#     p_new = p_new - np.mean(p_new) + np.mean(p_time)
#     # smooth small-scale noise
#     p_new = savgol_filter(p_new, 101, 3)
#     return p_new

# def burgers_time_domain(p, dx, dt, c0=340.0, nu=1e-5, n_steps=100):
#     """
#     Very coarse Burgers solver in time domain to approximate nonlinear steepening.
#     This is provided as a fallback — to reach high accuracy you'd need a specialized solver.
#     p: pressure time series (treated as initial spatial waveform), dx: propagation step size
#     dt: time step
#     nu: viscosity-like damping
#     """
#     u = p.copy()
#     for _ in range(n_steps):
#         dudx = np.gradient(u, dx)
#         u = u - dt * u * dudx + nu * dt * np.gradient(np.gradient(u, dx), dx)
#     return u


# nonlinear_correction.py
import numpy as np

def burgers_solver(p, dx, dt, c0=340.0, nu=1e-5, n_steps=200):
    """
    Solve Burgers' equation to simulate nonlinear propagation using a finite-difference method.

    Parameters:
    -----------
    p : array
        Initial pressure waveform.
    dx : float
        Spatial step (choose small, e.g. 1–2 meters or lower).
    dt : float
        Time step (must satisfy stability condition CFL ≤ 1).
    c0 : float
        Ambient speed of sound (m/s).
    nu : float
        Artificial viscosity term (shock-capturing factor). Lower = more steepening.
    n_steps : int
        Number of propagation steps.

    Returns:
    --------
    p : array
        Final propagated pressure waveform.
    """

    u = p.copy()

    # CFL condition: dt < dx / max(u)
    max_u = np.max(np.abs(u))
    if max_u != 0:
        cfl = dt * max_u / dx
        if cfl > 1:
            print(f"⚠ WARNING: CFL={cfl:.2f} > 1, adjusting dt for stability")
            dt = dx / max_u * 0.9

    for _ in range(n_steps):
        dudx = np.gradient(u, dx)
        d2udx2 = np.gradient(np.gradient(u, dx), dx)
        u = u - dt * (u * dudx) + nu * dt * d2udx2

    # Normalize to maintain same mean pressure
    u = u - np.mean(u) + np.mean(p)

    return u

def nonlinear_correction(p_time, dx=1.0, dt=1e-5, c0=340.0, nu=1e-5, n_steps=200):
    """
    Wrapper for standard nonlinear propagation.
    """
    return burgers_solver(p_time, dx, dt, c0=c0, nu=nu, n_steps=n_steps)
