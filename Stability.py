"""
Stability Module
This creates a recallable function for
solving the ODE.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Environmental Constants
# These are fixed for all simulations
M_WAVE_AMP = 2e5 # N·m (From Syasya's notebook, a constant)
OMEGA_WAVE = 0.5 # rad/s (From Syasya's notebook, a constant)

# The ODE Function

def _pitch_ode_model(t, y, I_total, C_h, K_total, t_wind, M_wind_series):
    """
    Defines the 2nd-order ODE for spar-buoy pitch dynamics.
    
    This version looks up the wind moment at time 't' from the
    provided 'M_wind_series' array.
    """
    theta, theta_dot = y
    
    # Wind Gust Logic
    # At the current time 't', find the wind moment
    # by interpolating from the gust profile.
    M_wind_at_t = np.interp(t, t_wind, M_wind_series)
    
    # Wave Logic
    M_wave = M_WAVE_AMP * np.sin(OMEGA_WAVE * t)
    
    # Sum forces
    M_external = M_wind_at_t + M_wave
    
    # Solve ODE
    # I*theta_ddot + C_h*theta_dot + K_total*theta = M_external
    theta_ddot = (M_external - C_h * theta_dot - K_total * theta) / I_total
    
    return [theta_dot, theta_ddot]

# Easy to Recall Function

def run_stability_simulation(I_total, C_h, K_total, t_wind, M_wind_series, t_span=(0, 200)):
    """
    Runs the stability simulation for ONE set of design parameters
    using a time-varying wind profile.
    
    Returns
    (t_eval, solution_object)
        A tuple containing the time array and the full 'solve_ivp' solution
    """
    y0 = [0, 0] # Initial conditions [theta=0, theta_dot=0]
    t_eval = np.linspace(t_span[0], t_span[1], 2000) # 2000 points
    
    # Pack the parameters for the solver.
    ode_args = (I_total, C_h, K_total, t_wind, M_wind_series)
    
    # Solve the ODE
    sol = solve_ivp(
        fun=_pitch_ode_model,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=ode_args,
        method='RK45' # Use RK45 as its very reliable we can compre with other numerical methods
    )
    
    return t_eval, sol
