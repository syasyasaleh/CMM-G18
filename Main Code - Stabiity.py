"""
Main Stability Analysis Script
Optimisation of a Floating Offshore Wind Turbine

Description:
1. Importing the seperate modules: `MMI`, `Hydrodynamics`, `Stability`
2. Defining the design cases (blade lengths 60m-120m).
3. Using a realistic NREL-based scaling law for blade weight.
4. Creating a realistic "10-second gust" wind profile.
5. Looping through each case, calling the modules to get I, K, and C.
6. Running the ODE solver for each case.
7. Collecting the results for Regression.
8. Plotting the final design curve.
9. Finding the exact 10-degree limit.
"""

import numpy as np
import matplotlib.pyplot as plt

# SECTION 1: IMPORT YOUR OUR CODES
from MMI import blade_moment_of_inertia
from Hydrodynamics import calculate_stiffness, calculate_damping
from Stability import run_stability_simulation

# SECTION 2: SYSTEM CONSTANTS & DESIGN ASSUMPTIONS

# Platform
I_PLATFORM = 8e9 # Platform inertia [kg·m^2] (From Syasya/Eric)
I_ADDED_FACTOR = 0.5 # Added mass inertia as a factor of platform
NUM_BLADES = 3

# Environment & Aerodynamics
RHO_AIR = 1.225 # kg/m^3
HUB_HEIGHT = 90.0 # m, above buoyancy
C_T = 0.8 # Thrust Coefficient
V_WIND_MEAN = 12.0 # m/s (Mean wind speed)

# Design Constraint
THETA_LIMIT_DEG = 10.0 # Max allowed pitch angle

# SECTION 3: FUNCTIONS

def create_wind_gust_profile(t_span=(0, 200), v_mean=12.0, gust_strength=1.5):
    """
    Creates a realistic wind profile with fixed gusts
    changing every 10 seconds.
    """
    
    # Create time steps every 10 seconds, plus the end of the step
    t_points = np.arange(t_span[0], t_span[1] + 10, 10)
    t_wind = np.sort(np.concatenate([t_points, t_points[1:] - 0.01]))
    
    # Define the velocity multiplier for each 10s interval
    multipliers = [
        1.0, 1.1, 1.0, 1.2, 1.4, # 0-40s
        gust_strength, 1.3, 1.1, 1.0, 1.1, # 50-90s
        1.2, 1.0, gust_strength * 1.2, 1.5, 1.3, # 100-140s
        1.1, 1.0, 1.1, 1.2, 1.0, 1.0 # 150-200s (last one)
    ]
    
    v_wind_values = [m * v_mean for m in multipliers]
    v_wind = np.repeat(v_wind_values, 2)
    
    # Ensure t_wind and v_wind are the same length
    v_wind = v_wind[:len(t_wind)]

    return t_wind, v_wind

def calculate_wind_moment_series(t_wind, v_wind_series, D_rotor):
    """
    Calculates the full time-series of wind moment from a wind profile.
    """
    A_rotor = np.pi * (D_rotor**2) / 4
    # M = 0.5 * rho * A * C_T * V^2 * h
    M_wind_series = 0.5 * RHO_AIR * A_rotor * C_T * (v_wind_series**2) * HUB_HEIGHT
    return M_wind_series

def get_realistic_blade_weight_N(blade_length_m):
    """
    Calculates blade weight based on scaling law.
    Source:https://www.researchgate.net/figure/Wind-turbine-blade-mass-as-a
    -function-of-the-blade-length-as-reproduced-from-Liu-and_fig2_375961938
    Mass (kg) = a * L^2.32
    a = 1.29
    
    Mass (kg) = 1.29 * L^2.32
    Weight (N) = Mass * g
    """
    g = 9.81
    # This is our realistic, citable formula
    blade_mass_kg = 1.29 * (blade_length_m ** 2.32)
    blade_weight_N = blade_mass_kg * g
    return blade_weight_N

# SECTION 4: MAIN ANALYSIS
if __name__ == "__main__":
    
    print("CMM3 Group 18: Main STABILITY Analysis Started")

    # 1. Define the design cases to test (range 60m-120m)
    blade_lengths_to_test = np.linspace(60, 120, 15)
    
    results_table = []
    print(f"Running {len(blade_lengths_to_test)} design cases (60m to 120m)...")

    # 2. Create the Wind Gust Profile
    t_wind, v_wind = create_wind_gust_profile(v_mean=V_WIND_MEAN)

    # 3. Main Simulation Loop
    for L in blade_lengths_to_test:
        
        print(f"  Running case: Blade Length = {L:.1f}m...")
        
        # Use scaling law for realistic weight
        W = get_realistic_blade_weight_N(L)
        D_rotor = L * 2
        
        # call functions
        
        # a) Call MMI module
        I_blade = blade_moment_of_inertia(W, L, model="rod")
        I_rotor = I_blade * NUM_BLADES
        I_total = I_PLATFORM + I_rotor + (I_PLATFORM * I_ADDED_FACTOR)
        
        # b) Call Hydrodynamics module
        #    This will use Syasya's new 60-120m data
        K_total = calculate_stiffness(L)
        C_h = calculate_damping(K_total, I_total)
        
        # c) Calculate the wind moment profile for this blade size
        M_wind_series = calculate_wind_moment_series(t_wind, v_wind, D_rotor)
        
        # d) Call Stability module
        t_sim, solution = run_stability_simulation(I_total, C_h, K_total, t_wind, M_wind_series)
        
        # --- Analyze Results ---
        # Find max pitch *after* the initial 20s transient
        settled_mask = t_sim > 20
        max_pitch_rad = np.max(np.abs(solution.y[0][settled_mask]))
        max_pitch_deg = np.degrees(max_pitch_rad)
        
        # Store results for this case
        results_table.append((L, max_pitch_deg))

    # SECTION 5: FINAL RESULTS & ANALYSIS
    
    print("Simulation Complete. Final Stability Results:")
    
    # 4. Print Regression Table (Data for Rubric 2C)
    print("\n--- Stability Table for Regression (Rubric 2C) ---")
    print("Blade Length (m) | Max Pitch (deg)")
    print("-" * 50)
    
    L_results = np.array([row[0] for row in results_table])
    Pitch_results = np.array([row[1] for row in results_table])

    for i in range(len(L_results)):
        L = L_results[i]
        Pitch = Pitch_results[i]
        status = "PASS" if Pitch <= THETA_LIMIT_DEG else "FAIL"
        print(f" {L:<17.1f} | {Pitch:<15.2f} ({status})")

    # 5. Run Root Finding
    # We find the *exact* blade length where pitch crosses 10 degrees
    # We use interpolation (again!) to find the root.
    try:
        # Check if the pitch ever *crossed* the limit
        if np.max(Pitch_results) > THETA_LIMIT_DEG and np.min(Pitch_results) < THETA_LIMIT_DEG:
            # np.interp(target_pitch, pitch_values, length_values)
            max_L_allowed = np.interp(THETA_LIMIT_DEG, Pitch_results, L_results)
            print("--- Root Finding Analysis ---")
            print(f"The maximum allowed blade length to stay under {THETA_LIMIT_DEG} degrees is: {max_L_allowed:.2f} m")
        else:
            print("--- Root Finding Analysis ---")
            print(f"Pitch did not cross the {THETA_LIMIT_DEG} degree limit in this range.")

    except Exception as e:
        print(f"Could not run root finding: {e}")

    # 6. Plot Final Design Curves (Rubric 3B)
    print("\nGenerating final stability plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax.set_xlabel('Blade Length (m)')
    ax.set_ylabel('Max Pitch Angle (deg)')
    ax.plot(L_results, Pitch_results, 's--', label='Max Pitch (deg)')
    ax.tick_params(axis='y')
    ax.grid(True)
    
    ax.axhline(THETA_LIMIT_DEG, color=color, linestyle=':', label=f"Limit ({THETA_LIMIT_DEG}°)")
    ax.set_ylim(bottom=0)

    fig.suptitle('Design Analysis: Stability vs. Blade Length (60m-120m)', fontsize=16)
    fig.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig("stability_analysis_curve.png")
    print("Saved final plot to 'stability_analysis_curve.png'")