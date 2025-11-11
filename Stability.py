"""
Stability Module
This creates a recallable function for
solving the ODE.
"""

import numpy as np
from scipy.integrate import solve_ivp

# SECTION 1: Environmental Constants
# These are fixed for all simulations
M_WAVE_AMP = 2e5 # N·m (From Syasya's notebook) constant wave moment amplitude
OMEGA_WAVE = 0.5 # rad/s (From Syasya's notebook) constant wave freqeuncy


# SECTION 2: The ODE Function
def _pitch_ode_model(t, y, I_total, C_h, K_total, t_wind, M_wind_series):
    """
    Defines the 2nd-order ODE for spar-buoy pitch dynamics.
        Parameters from solve_ivp:
        
    t : float
        The current time step that the solver is at.
    y : list [float, float]
        The current state of the system at time 't'.
        y[0] is the pitch angle (theta)
        y[1] is the pitch velocity (theta_dot)
        
    Parameters from 'ode_args':
    I_total: float
        Total system Inertia (I)
    C_h: float
        Total hydrodynamic Damping (C)
    K_total: float
        Total restoring Stiffness (K)
    t_wind: np.array
        The time array for the wind gust profile (e.g., [0, 10, ...])
    M_wind_series: np.array
        The wind moment array for the gust profile (e.g., [5e7, 6e7, ...])
    """
    theta, theta_dot = y
    
    # 1. Wind Gust Logic
    """
    At the current solver time 't', look up the wind moment
    by interpolating from the gust profile. This is essential
    as the solver (RK45) takes adaptive steps (e.g., t=50.1, t=50.22)
    that are not in our 10-second data.
    """
    M_wind_at_t = np.interp(t, t_wind, M_wind_series)
    
    # 2. Wave Logic
    # Calculate the sinusoidal wave moment at time 't'
    M_wave = M_WAVE_AMP * np.sin(OMEGA_WAVE * t)
    
    # 3. Sum forces
    M_external = M_wind_at_t + M_wave

    # Solve the ODE
    """
    I*theta_ddot + C_h*theta_dot + K_total*theta = M_external
    becomes
    theta_ddot = (M_external - C_h*theta_dot - K_total*theta) / I_total
     """
    
    theta_ddot = (M_external - C_h * theta_dot - K_total * theta) / I_total
    
    return [theta_dot, theta_ddot]


# SECTION 3: Easy to Recall Function

def run_stability_simulation(I_total, C_h, K_total, t_wind, M_wind_series, t_span=(0, 200)):
    """
    Runs the stability simulation for one set of design parameters
    using a time-varying wind profile.
    
    Returns
    (t_eval, solution_object)
        A tuple containing the time array and the full 'solve_ivp' solution
    """
    y0 = [0, 0] # Initial conditions [theta=0, theta_dot=0]
    t_eval = np.linspace(t_span[0], t_span[1], 2000) # Creating a time array with 2000 data points
    
    # Pack the parameters for the ODE solver.
    ode_args = (I_total, C_h, K_total, t_wind, M_wind_series)
    
    # Solve the ODE using fourth order runge-katta
    sol = solve_ivp(
        fun=_pitch_ode_model,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=ode_args,
        # We explicitly choose 'RK45' (Runge-Kutta 4-5) because it is
        # an adaptive-step-size solver. It is far more efficient and
        # accurate for an oscillating system than a fixed-step
        # method like Euler's, as we will justify in our report.
        method='RK45'
    )
    
    return t_eval, sol


# SECTION 4: TEST BLOCK
if __name__ == "__main__":
    """
    This allows us to add test code here to run this file
    directly to make sure it works.
    
    This code will not run when the file is imported.
    """
    
    print("Testing Stability.py directly")
    
    # 1. Create simple test parameters
    I_test = 1e10
    C_test = 3e8
    K_test = 5e9
    
    # 2. Create a simple test wind profile (constant wind)
    t_wind_test = np.array([0, 200])
    M_wind_series_test = np.array([7.1e7, 7.1e7])
    
    # 3. Run the simulation
    print("Running test simulation...")
    t_sim, solution = run_stability_simulation(
        I_test, C_test, K_test, t_wind_test, M_wind_series_test
    )
    
    # 4. Analyse and plot the test
    max_pitch = np.max(np.abs(np.degrees(solution.y[0])))
    print("Test simulation complete.")
    print(f"Max pitch angle reached: {max_pitch:.2f} degrees")
    
    print("Generating test plot 'stability_test_run.png'...")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(t_sim, np.degrees(solution.y[0]), label="Pitch (deg)")
    plt.xlabel("Time (s)"); plt.ylabel("Pitch Angle (deg)")
    plt.title("Stability.py Test Run")
    plt.grid(True); plt.legend()
    plt.savefig("stability_test_run.png")
    print("Test complete.")
