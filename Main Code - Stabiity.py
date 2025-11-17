"""Main Stability Analysis
In this code we will:
Determine the maximum blade length (L) for a floating spar buoy
turbine that keeps the maximum pitch angle under a 10-degree limit.

We will do this by:
Find the root of a "black box" simulation:
f(L) = (Max Pitch for L) - 10

We will implement three required numerical methods:
1.  ODE Solver (4th order runge-katta):
    Called from the imported `Stability.py` module.
2.  Interpolation:
    - Used in `Hydrodynamics.py` to find K_h vs. L.
    - Used in `Stability.py` to apply the wind gust during
      the adaptive ODE steps.
3.  For Root Finding we will use the false position method:
    See Section 3 of this file to find the root of f(L).

The flow of the code:
1.  Section 5 (`if __name__ == "__main__"`) starts the script.
2.  It calls the root finder (`find_root_by_false_position`).
3.  The root finder repeatedly calls the main simulation
    (`find_pitch_for_length`), which in turn calls all imported modules.
4.  The results are saved, printed to a table, and plotted."""


# Library imports
import numpy as np
import matplotlib.pyplot as plt
# Import functions from our 3 mini codes
from MMI import calculate_total_inertia
from Hydrodynamics import calculate_stiffness, calculate_damping
from Stability import run_stability_simulation

# Section 1: Set our simulation constants and design parameters
# Aerodynamic Parameters for forcing function
"""These are the fixed parameters for the wind moment calculation"""
RHO_AIR = 1.225 # [kg/m^3]
HUB_HEIGHT = 87.6 # [m] Height of hub, matches NACELLE_HUB_CM_Z
C_T = 0.8 # [-] We assume this thrust coefficient
V_WIND_MEAN = 12 # [m/s] Mean wind speed

# Design Constraint
THETA_LIMIT_DEG = 10.0 # [deg] Max allowable pitch angle for our model
""" Note: If our root finding give a blade length, L,
with a corresponding angle greater than our constraint of 10 deg,
the deisgn will fail."""

# Simulation Time
T_SPAN = (0, 200) # [s] We will run the simulation for 200 seconds
T_TRANSIENT = 20 # [s] We will also ignore the first 20s as this will be messy
# i.e. stationary turbine then sudden external forces causes noisy initial data

# False position root finding parameters
ROOT_FIND_BRACKET = [60, 120] # [a, b] initial bracket
ROOT_FIND_ITERATIONS = 25 # N, i.e. the max number of iterations


# Section 2: External forcing functions
def create_wind_gust_profile(t_span, v_mean=12.0, gust_strength=1.5):
    """Creates a realistic wind profile with fixed gusts.
    This provides a time-varying wind speed for the simulation."""
    
    # Create time steps every 10 seconds
    t_points = np.arange(t_span[0], t_span[1] + 10, 10)
    
    # Define the velocity multiplier for each 10s interval
    multipliers = [ 1.0, 1.1, 1.0, 1.2, 1.4, # 0 to 40s
        1.5, 1.3, 1.1, 1.0, 1.1, # 50 to 90s
        1.2, 1.0, 1.8, 1.5, 1.3, # 100 to 140s
        1.1, 1.0, 1.1, 1.2, 1.0, 1.0] # 150 to 200s
       
    
    # Create a step-function to lop through each mulitplier 'm'
    v_wind_values = [m * v_mean for m in multipliers]
    v_wind = np.repeat(v_wind_values, 2)
    
    # Create the corresponding time array
    t_wind = np.sort(np.concatenate([t_points, t_points[1:] - 0.01]))
    t_wind[0] = t_span[0]
    v_wind = v_wind[:len(t_wind)]

    """Here we are creating an instant step in wind.
    When the ODE solver asks 'np.intero. for the wind at t=5s,
    it interpolates between (0,12) and (9.99,12) Or if it asks at t=15,
    it interpolates between (10,13.2) and (19.99,13.2).
    We also make sure that the two lists of wind speed and time are the same length.
    """
    return t_wind, v_wind

def calculate_wind_moment_series(t_wind, v_wind_series, blade_length_m):
    """Calculates the wind moment from all the wind profiles.
    M = F * h = (0.5 * rho * A * C_T * V^2) * h """
    D_rotor = blade_length_m * 2
    A_rotor = np.pi * (D_rotor**2) / 4
    
    M_wind_series = 0.5 * RHO_AIR * A_rotor * C_T * (v_wind_series**2) * HUB_HEIGHT
    return M_wind_series


# Section 3: Root finding - False position method
def find_root_by_false_position(bracket, iterations, t_wind, v_wind, history_list):
    """Finds the root and populates the history_list for plotting.
    This function contains:
    1. The Objective Function (find_pitch_error)
    2. The Bracket Check
    3. The False Position loop"""
    
    # 1. Define the Objective Function
    def find_pitch_error(L):
        """Runs the full simulation for one length L and returns the
        error from the 10-degree limit."""
        # Run the full simulation by calling the main function in section 4
        pitch = find_pitch_for_length(L, t_wind, v_wind)
        error = pitch - THETA_LIMIT_DEG # the error is f(L) = pitch(L) - 10 deg
        
        # Add this result to the history list for plotting to avoid repetitive, expensive computations
        history_list.append((L, pitch))
        
        print(f"Root Finder Guess: L={L:.3f} m, Pitch={pitch:.3f} deg, Error={error:.9f} deg")
        return error

    # Start the finding the root
    print("Starting to find the root using the false position method")
    a_n = bracket[0] # These values are passed in by section 5
    b_n = bracket[1]
    N = iterations

    try:
        # 2. Validate the bracket by running the objective function simulation twice
        f_a = find_pitch_error(a_n)
        f_b = find_pitch_error(b_n)
        
        if f_a * f_b >= 0: # If f_a and f_b have the same sign then the code will fail as the root isn't captured
            print(f"\nROOT FINDING FAILED: The initial bracket [{a_n}, {b_n}] is not valid.")
            print(f"f(a)_error = {f_a:.2f}, f(b)_error = {f_b:.2f} (must have opposite signs)")
            return None
        
        # 3. The false position method loop runs 25 times (max)
        for n in range(1, N + 1):
            denom = f_b - f_a # This is the denominator of our false position formula
            if abs(denom) < 1e-6: return None # Fails if the root is not within our pre-set error
                
            m_n = a_n - f_a * (b_n - a_n) / denom # This is the false position formula from lecture
            f_m_n = find_pitch_error(m_n) # run the ODE for the new guess m_n for new error f_m_n
            
            if abs(f_m_n) < 1e-6:
                print(f"\nFound solution in {n+2} total iterations.") # We add 2 as we ran 2 simulations earlier to test bracket
                return m_n
            
            if f_a * f_m_n < 0: # Check for opposite signs
                b_n, f_b = m_n, f_m_n # If true then we replace the value of b_n as m_n i.e [a_n, b_n]
            elif f_b * f_m_n < 0:
                a_n, f_a = m_n, f_m_n # If f_b and f_m_n have opposite signs then the root is between m_n and b_n, so update a_n
            else:
                return None # Fails
                
        print(f"\nFailed to converge in {N} iterations.")
        return a_n - f_a * (b_n - a_n) / denom # Return the last best guess if we've gone over 25 iterations

    except Exception as e:
        print(f"\nRoot finding could not be run: {e}")
        return None


# Section 4: The main simulation function

def find_pitch_for_length(L, t_wind, v_wind):
    """
    This is the main simulation function.
    It runs the full simulation for a single blade length 'L'
    and returns its maximum pitch angle.
    """
    
    # Call the MMI mini code
    I_total = calculate_total_inertia(L)
    
    # Call the Hydrodynamics mini code
    K_total = calculate_stiffness(L)
    C_h = calculate_damping(K_total, I_total) # Recall that this interpolates for L's
    
    # Calculate the wind moment profile for this L
    M_wind_series = calculate_wind_moment_series(t_wind, v_wind, L)
    
    # Call the Stability mini code i.e. The ODE solver
    t_sim, solution = run_stability_simulation(
        I_total, C_h, K_total,
        t_wind, M_wind_series, T_SPAN
    )
    
    # Analyse the results
    settled_mask = t_sim > T_TRANSIENT # Recall we are filtering out the intial set up stage i.e. 20s
    max_pitch_rad = np.max(np.abs(solution.y[0][settled_mask]))
    """Here we get absolute value of the 2000 point array of pitch angles (in adians) 'solution.y[0]',
    and make sure we are getting the values between 20s to 200s,
    and from that we are finding the single largest value"""
    max_pitch_deg = np.degrees(max_pitch_rad)
    return max_pitch_deg


# Section 5: The main analysis
if __name__ == "__main__": # Starting our code from here

    # 1. Create the single Wind Gust Profile for all simulations
    t_wind, v_wind = create_wind_gust_profile(T_SPAN, v_mean=V_WIND_MEAN)

    # 2. Create an empty list to store the data from the wind gusts for the plot
    plot_data_history = []

    # 3. ROOT FINDING & DATA GATHERING
    # This ONE function call runs all the simulations
    # and populates the plot_data_history list.
    max_L_allowed = find_root_by_false_position(
        ROOT_FIND_BRACKET, ROOT_FIND_ITERATIONS,
        t_wind, v_wind,
        history_list=plot_data_history # Pass the empty list
    )

    # Section 6: Print a table of results to see how the root finding method performed
    """We will print a table including the blade length, pitch angle
    and whether this was a pass or fail i.e. <10deg?"""
    print("Results Table")
    
    # Sort the table by blade length
    plot_data_history.sort() 
    
    L_results = np.array([row[0] for row in plot_data_history])
    Pitch_results = np.array([row[1] for row in plot_data_history])
    print("-" * 45)
    print("Blade length (m) | Max pitch (deg) | Result")
    print("-" * 45)
    
    for L, Pitch in zip(L_results, Pitch_results):
        status = "PASS" if Pitch <= THETA_LIMIT_DEG else "FAIL"
        print(f" {L:<15.2f} | {Pitch:<15.2f} | {status}")


    # Section 7: Get a plot of our results from root finding
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the simulation data
    ax.plot(L_results, Pitch_results, 's--', label='Max Pitch', zorder=10)
    
    # Plot the 10-degree limit
    ax.axhline(THETA_LIMIT_DEG, color='red', linestyle=':', 
               label=f"Design Limit ({THETA_LIMIT_DEG} deg)")

    # If the root finder worked, plot the exact point
    if max_L_allowed:
        ax.plot(max_L_allowed, THETA_LIMIT_DEG, 
                label=f"Exact Limit: {max_L_allowed:.2f} m", zorder=11)
        ax.axvline(max_L_allowed, color='red', linestyle=':')

    ax.set_xlabel('Blade Length, L, (m)', fontsize=14)
    ax.set_ylabel('Max Pitch Angle, theta, (deg)', fontsize=14)
    ax.grid(True, linestyle='--')
    ax.set_ylim(bottom=0)
    
    fig.suptitle('Design Analysis: Stability vs. Blade Length', fontsize=16)
    ax.legend(loc='best'); plt.tight_layout()
    
    plt.savefig("stability_analysis_curve.png")
    print("-" * 45)
