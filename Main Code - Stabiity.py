"""
CMM3 Group Project: Main Stability Analysis
Project: Optimisation of a Floating Offshore Wind Turbine

Description:
This script performs the main design analysis by integrating all
three custom-built modules:
1. `inertia_model`: Calculates total system MMI.
2. `hydrodynamics_model`: Calculates stiffness (K) and damping (C).
3. `stability_model`: Solves the ODE for pitch dynamics.

Workflow:
1. Defines a realistic wind gust profile.
2. Loops through a range of blade lengths (60m-120m).
3. For each blade length, it calls the modules to find I, K, and C.
4. It runs the full ODE simulation to find the max pitch angle.
5. The results (L vs. Pitch) are plotted.
6. A numerical root-finding method (Secant) is used to find the
   *exact* blade length that results in the 10-degree design limit.
"""

import numpy as np
import matplotlib.pyplot as plt
# scipy.optimize contains the root-finding solvers
from scipy.optimize import root_scalar

# SECTION 1: IMPORT YOUR THREE MODULES
# These imports now match your finalized .py files
from MMI import calculate_total_inertia
from Hydrodynamics import calculate_stiffness, calculate_damping
from Stability import run_stability_simulation

# SECTION 2: SIMULATION CONSTANTS & DESIGN PARAMETERS

# --- Aerodynamic Parameters (for forcing function) ---
# These are fixed parameters for the wind moment calculation
RHO_AIR = 1.225         # [kg/m^3]
HUB_HEIGHT = 87.6       # [m] (Height of hub, matches NACELLE_HUB_CM_Z)
C_T = 0.8               # [-] Assumed Thrust Coefficient
V_WIND_MEAN = 12.0      # [m/s] (Mean wind speed)

# --- Design Constraint ---
THETA_LIMIT_DEG = 10.0  # [deg] Max allowed pitch angle

# --- Simulation Time ---
T_SPAN = (0, 200)       # [s] Run simulation for 200 seconds
T_TRANSIENT = 20        # [s] Ignore the first 20s for max pitch analysis


# SECTION 3: HELPER FUNCTIONS (FORCING FUNCTIONS)

def create_wind_gust_profile(t_span, v_mean=12.0, gust_strength=1.5):
    """
    Creates a realistic wind profile with fixed gusts.
    This provides the time-varying wind speed for the simulation.
    """
    # Create time steps every 10 seconds
    t_points = np.arange(t_span[0], t_span[1] + 10, 10)
    
    # Define the velocity multiplier for each 10s interval
    multipliers = [
        1.0, 1.1, 1.0, 1.2, 1.4, # 0-40s
        gust_strength, 1.3, 1.1, 1.0, 1.1, # 50-90s
        1.2, 1.0, gust_strength * 1.2, 1.5, 1.3, # 100-140s
        1.1, 1.0, 1.1, 1.2, 1.0, 1.0 # 150-200s
    ]
    
    # Create a step-function profile
    # np.repeat creates [1.0, 1.0, 1.1, 1.1, ...]
    v_wind_values = [m * v_mean for m in multipliers]
    v_wind = np.repeat(v_wind_values, 2)
    
    # Create the corresponding time array
    # [0, 9.99, 10, 19.99, 20, ...]
    t_wind = np.sort(np.concatenate([t_points, t_points[1:] - 0.01]))
    t_wind[0] = t_span[0] # Ensure it starts at t=0
    
    # Ensure t_wind and v_wind are the same length
    v_wind = v_wind[:len(t_wind)]

    return t_wind, v_wind

def calculate_wind_moment_series(t_wind, v_wind_series, blade_length_m):
    """
    Calculates the full time-series of wind moment from a wind profile.
    M = F * h = (0.5 * rho * A * C_T * V^2) * h
    """
    D_rotor = blade_length_m * 2
    A_rotor = np.pi * (D_rotor**2) / 4
    
    # M = 0.5 * rho * A * C_T * V^2 * h
    M_wind_series = 0.5 * RHO_AIR * A_rotor * C_T * (v_wind_series**2) * HUB_HEIGHT
    return M_wind_series


# SECTION 4: OBJECTIVE FUNCTION (FOR ROOT FINDING)
# (This is the "First Class" part for Rubric 2A)

def find_pitch_error(L, t_wind, v_wind, target_pitch_deg):
    """
    This is the "objective function" for the root finder.
    It takes one variable 'L' (blade length) and finds the
    error between its max pitch and the target pitch.
    
    The root finder will find L such that: error = 0.
    
    Args:
        L (float): The current blade length guess.
        t_wind (np.array): Wind time array.
        v_wind (np.array): Wind velocity array.
        target_pitch_deg (float): The design limit (10.0).
    
    Returns:
        float: The error (max_pitch_deg - target_pitch_deg).
    """
    
    print(f"  Root Finder Guess: L = {L:.3f} m...")
    
    # --- 1. Run the full simulation for this L ---
    
    # a) Call MMI module (This is a clean, modular call)
    I_total = calculate_total_inertia(L)
    
    # b) Call Hydrodynamics module
    K_total = calculate_stiffness(L)
    C_h = calculate_damping(K_total, I_total)
    
    # c) Calculate the wind moment profile for this L
    M_wind_series = calculate_wind_moment_series(t_wind, v_wind, L)
    
    # d) Call Stability module
    t_sim, solution = run_stability_simulation(
        I_total, C_h, K_total,
        t_wind, M_wind_series, T_SPAN
    )
    
    # --- 2. Analyze Results ---
    # Find max pitch *after* the initial transient
    settled_mask = t_sim > T_TRANSIENT
    max_pitch_rad = np.max(np.abs(solution.y[0][settled_mask]))
    max_pitch_deg = np.degrees(max_pitch_rad)
    
    # --- 3. Return the Error ---
    error = max_pitch_deg - target_pitch_deg
    
    return error


# SECTION 5: MAIN ANALYSIS SCRIPT
if __name__ == "__main__":
    
    print("CMM3 Group Project: Main STABILITY Analysis Started")

    # 1. Define the design cases to test (for the plot)
    blade_lengths_to_test = np.linspace(60, 120, 13) # 13 cases
    
    results_table = []
    print(f"Running {len(blade_lengths_to_test)} design cases (60m to 120m)...")

    # 2. Create the single Wind Gust Profile for all simulations
    t_wind, v_wind = create_wind_gust_profile(T_SPAN, v_mean=V_WIND_MEAN)

    # 3. Main Simulation Loop (to get data for the plot)
    for L in blade_lengths_to_test:
        
        print(f"  Running case: Blade Length = {L:.1f}m...")
        
        # a) Call MMI module
        I_total = calculate_total_inertia(L)
        
        # b) Call Hydrodynamics module
        K_total = calculate_stiffness(L)
        C_h = calculate_damping(K_total, I_total)
        
        # c) Calculate the wind moment profile for this blade size
        M_wind_series = calculate_wind_moment_series(t_wind, v_wind, L)
        
        # d) Call Stability module
        t_sim, solution = run_stability_simulation(
            I_total, C_h, K_total,
            t_wind, M_wind_series, T_SPAN
        )
        
        # --- Analyze Results ---
        # Find max pitch *after* the initial 20s transient
        settled_mask = t_sim > T_TRANSIENT
        max_pitch_rad = np.max(np.abs(solution.y[0][settled_mask]))
        max_pitch_deg = np.degrees(max_pitch_rad)
        
        # Store results for this case
        results_table.append((L, max_pitch_deg))

    # SECTION 6: PRINT RESULTS TABLE
    
    print("\nSimulation Complete. Final Stability Results:")
    
    # Get results into numpy arrays for easy plotting/analysis
    L_results = np.array([row[0] for row in results_table])
    Pitch_results = np.array([row[1] for row in results_table])

    print("\n--- Stability Table (Data for Rubric 2C) ---")
    print("Blade Length (m) | Max Pitch (deg) | Status")
    print("-" * 50)
    
    for L, Pitch in zip(L_results, Pitch_results):
        status = "PASS" if Pitch <= THETA_LIMIT_DEG else "FAIL"
        print(f" {L:<17.1f} | {Pitch:<15.2f} | {status}")

    # SECTION 7: ROOT FINDING ANALYSIS (Rubric 2A)
    
    print("\n--- Root Finding Analysis (Secant Method) ---")
    print(f"Finding exact blade length for {THETA_LIMIT_DEG} deg pitch limit...")
    
    # We need to pass the "extra" arguments to our objective function
    args_for_solver = (t_wind, v_wind, THETA_LIMIT_DEG)
    
    # We need two initial guesses. Let's pick two from our
    # results that are near the 10-degree mark.
    # (e.g., L=80m and L=90m)
    x0_guess = 80.0
    x1_guess = 90.0
    
    try:
        # We call the solver.
        # It will repeatedly call find_pitch_error(L, *args_for_solver)
        # until the error is zero.
        sol = root_scalar(
            f=find_pitch_error,      # The function to solve
            args=args_for_solver,    # Extra args to pass to 'f'
            method='secant',         # Justify this in your report!
            x0=x0_guess,             # First initial guess
            x1=x1_guess              # Second initial guess
        )
        
        if sol.converged:
            max_L_allowed = sol.root
            print("\nROOT FINDING SUCCESS:")
            print(f"The maximum allowed blade length is: {max_L_allowed:.2f} m")
        else:
            print(f"\nROOT FINDING FAILED: {sol.flag}")
            max_L_allowed = None

    except Exception as e:
        print(f"\nRoot finding could not be run: {e}")
        print("This may be because the pitch never crosses 10 degrees.")
        max_L_allowed = None


    # SECTION 8: FINAL PLOT (Rubric 3B)
    
    print("\nGenerating final stability plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the simulation data
    ax.plot(L_results, Pitch_results, 's--', label='Max Pitch (Simulation)', zorder=10)
    
    # Plot the 10-degree limit
    ax.axhline(THETA_LIMIT_DEG, color='red', linestyle=':', 
               label=f"Design Limit ({THETA_LIMIT_DEG}°)")

    # If the root finder worked, plot the exact point
    if max_L_allowed:
        ax.plot(max_L_allowed, THETA_LIMIT_DEG, 'rX', markersize=15, 
                label=f"Exact Limit: {max_L_allowed:.2f} m", zorder=11)
        ax.axvline(max_L_allowed, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Blade Length (m)', fontsize=12)
    ax.set_ylabel('Max Pitch Angle (deg)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    
    fig.suptitle('Design Analysis: Stability vs. Blade Length', fontsize=16)
    fig.legend(loc='best')
    plt.tight_layout()
    
    plt.savefig("stability_analysis_curve.png")
    print("Saved final plot to 'stability_analysis_curve.png'")