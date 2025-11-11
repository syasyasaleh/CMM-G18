
"""Power analysis of the wind turbine:
we will determine the Annual Energy Production (AEP) for the spar buoy
turbine across a range of blade lengths of 60m to 160m by calculating the AEP
by combining two models:
1. A power curve P(v, L)
2. A Weibull wind distribution H(v)

We will implement the regression numerical method
by fitting a polynomial to the baseline NREL 5MW data.

The code overall does the following:
1.  Define the baseline NREL 5MW power curve data.
2.  Run `np.polyfit` (poly regression) ONCE to create a single 'base_power_fn'.
3.  Define a scaling function that applies blade length L to this base function.
4.  Define a function to generate a realistic Weibull wind distribution.
5.  Loop from L=60 to L=160, call the scaling function, and calculate AEP.
6.  Print a final table and plot (L vs AEP) for the report.
7.  Plot two justification graphs: one for the Regression and one for the Weibull."""
import numpy as np
import matplotlib.pyplot as plt

# Section 1: Wind Distribution (Weibull)

# Parameters for a typical North Sea site
"""TNO innovation for life Offshore wind energy 
deployment in the North Sea by 2030:
long-term measurement campaign. EPL, 
2016-2022"""
WEIBULL_A = 10.566 # Scale parameter [m/s]
"""This parameter either expands or contarcts the distibutio along the wind speed axis."""
WEIBULL_K = 2.113 # Shape parameter [-]
"""If k is less than 1 this indicated a distrubtion with a higher fequency of low wind speeds
IF k is > 1 this indicates a more consistent wind speed around the median."""
MAX_WIND_SPEED = 25 # Max speed to check (matches cut-out)
"""Justification of the parameter values:
We are using citable parameters for a real site.
We have selected the data at 91m altitude, which is the
closest available data point to our turbine's 87.6m hub height."""

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    """
    This function returns an array of hours per year for each wind speed (1 to max_v)
    using a Weibull distribution.
    """
    hours_in_year = 365 * 24
    # Create an array for wind speeds [0, 1, 2, ..., max_v]
    v = np.arange(0, max_v + 1)
    
    # Weibull Probability Density Function (PDF) for wind speed
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    """This function describes the likelihood of observing a specifc wind speed v,
    and is useful when we want to estimate the wind energy potential,
    and analysing wind speed distributions across different data sets"""
    
    # Probability of wind in each 1 m/s
    # We can approximate this by the PDF value
    prob_v = pdf
    
    # Normalise the probabilities to sum to 1 to account for wind speeds > max_v
    prob_v_normalised = prob_v / np.sum(prob_v)
    
    # Calculate the hours per year for each wind speed
    hours_v = prob_v_normalised * hours_in_year
    
    return hours_v, v


# Section 2: Power curve model using regression

# This is the data from the NREL 5MW spec (L=61.5m)
# (Wind Speed (m/s), Power (kW)) using 5mw Fig9-1 Pg32
BASE_POWER_CURVE_DATA = np.array([
    (3, 100),
    (4, 250),
    (5, 475),
    (6, 800),
    (7, 1250),
    (8, 1900),
    (9, 2700),
    (10, 3717),
    (11, 4924),
    (11.4, 5000), # This is the rated wind speed
    (12, 5000),
    (13, 5000),
    (14, 5000),
    (15, 5000),
    (16, 5000),
    (17, 5000),
    (18, 5000),
    (19, 5000),
    (20, 5000),
    (21, 5000),
    (22, 5000),
    (23, 5000),
    (24, 5000),
    (25, 5000) ])# Cut off speed

# Generator limit (5 MW)
GENERATOR_LIMIT_W = 5_000_000.0
BASE_BLADE_LENGTH = 61.5  # [m]

def create_scaled_power_curve_fn(L, base_power_fn):
    """
    This is a "factory" function. It takes a blade length L and
    the single base_power_fn (from the regression) and
    returns a NEW function, `scaled_power_curve_fn`,
    that is specific to that blade length.
    """
    
    def scaled_power_curve_fn(v):
        # Call the base regression function (passed in as an argument)
        # and multiply by 1000 to get W from kW
        base_power_w = base_power_fn(v) * 1000
        
        # 3. The model using a scalling assumption
        """This is our scalling asumption which is that power ir proportional to the swept area A=pi*L^2,
        so the new power is scaled by the ratio of the areas (L/L_base)^2"""
        scaled_power_w = base_power_w * (L / BASE_BLADE_LENGTH)**2
        
        # 4. Apply the constraints
        # a) Cannot produce power below cut-in 3 m/s
        if v < 3.0:
            return 0.0
        # b) Cannot produce more than the stated generator limit of 5MW
        if scaled_power_w > GENERATOR_LIMIT_W:
            return GENERATOR_LIMIT_W
        # c) Shut down at the cut off data of 25 m/s
        if v > 25.0:
            return 0.0
        # d) Safety check: Polynomials can dip negative
        if scaled_power_w < 0:
            return 0.0
            
        return scaled_power_w

    # Return the new, specific function
    return scaled_power_curve_fn


# Section 3: The AEP calculation

def calculate_aep(power_curve_fn, wind_dist_hours):
    """THis function calculates the AEP for a given power curve and wind distribution.
    AEP = SUM(P(v) * H(v))"""
    total_energy_wh = 0.0
    
    # Loop from v=1 to v=max_v
    for v in range(1, len(wind_dist_hours)):
        # Get the power [W] at this wind speed
        power_w = power_curve_fn(v)
        
        # Get the hours/year at this wind speed
        hours = wind_dist_hours[v]
        
        # Energy [W·h] = power [W] * hours [h] for the specifc wind speed
        energy_wh = power_w * hours
        
        total_energy_wh += energy_wh
        
    # Convert from W·h to MW·h (divide by 1,000,000)
    total_energy_mwh = total_energy_wh / 1_000_000.0
    return total_energy_mwh


# Section 4: The main analysis section

if __name__ == "__main__":

    # 1. Get the wind distribution
    # Now returns BOTH the hours and the wind speeds for plotting
    wind_dist, v_wind_speeds = get_wind_distribution(WEIBULL_A, WEIBULL_K)
    
    # 2. Define the design cases to test
    blade_lengths_to_test = np.arange(60, 161, 5) # Test up to 160m so we can see a larger scale
    """We will include up to a balde length of 160m to highlight that this model assumes perfect stability,
    so we need to compare against our stbailty code results i.e. max picth 10 deg @ L=83m."""
    
    print(f"Running {len(blade_lengths_to_test)} design cases for 60m to 160m")
    
    aep_results = []

    # Run the regression calculation
    v_data = BASE_POWER_CURVE_DATA[:, 0]
    p_data_kw = BASE_POWER_CURVE_DATA[:, 1]
    coefficients = np.polyfit(v_data, p_data_kw, 6)
    
    # Create the single, reusable base function
    base_power_fn_for_plot = np.poly1d(coefficients)

    # 3. The main simulation loop
    for L in blade_lengths_to_test:
        
        # a) Create the specific power curve for this L
        # This is a much faster call as it only does the scaling
        power_fn = create_scaled_power_curve_fn(L, base_power_fn_for_plot)
        
        # b) Calculate the AEP
        aep = calculate_aep(power_fn, wind_dist)
        
        aep_results.append((L, aep))
        print(f"  L = {L:3d} m, AEP = {aep:,.0f} MWh/yr")

    # 4. Print a table of results
    print(" AEP Results Table")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 30)
    for L, aep in aep_results:
        print(f" {L:<15.2f} | {aep:,.0f}")
        
    # 5. Plot the final power curve (AEP vs. Length)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    L_plot = np.array([row[0] for row in aep_results])
    AEP_plot = np.array([row[1] for row in aep_results])
    
    ax.plot(L_plot, AEP_plot, 'bo-', label='AEP (MWh/yr)')
    ax.set_xlabel('Blade Length, L, (m)', fontsize=14)
    ax.set_ylabel('Annual Energy Production, AEP, (MWh/yr)', fontsize=14)
    ax.grid(True, linestyle='--')
    ax.set_ylim(bottom=0) # Make plot start at 0
    fig.suptitle('Design Analysis: AEP vs. Blade Length', fontsize=18)
    ax.legend(loc='best'); plt.tight_layout()
    plt.savefig("power_analysis_curve.png")
    
    
    # 6. Plot the Weibull distribution justification plot
    """Weibull plot justification:
    This plot is for the report to visualise our cited assumption
    for the wind profile. It shows the number of hours per year
    (y-axis) that each wind speed (x-axis) occurs.
    
    This visually justifies our AEP calculation and shows the
    "realism" of our chosen wind site model."""
    fig_weibull, ax_weibull = plt.subplots(figsize=(10, 6))
    
    ax_weibull.bar(v_wind_speeds, wind_dist, label='Hours per Year', width=0.8)
    
    ax_weibull.set_xlabel('Wind Speed, v, (m/s)', fontsize=12)
    ax_weibull.set_ylabel('Hours per Year (h)', fontsize=12)
    ax_weibull.grid(True, linestyle='--')
    ax_weibull.set_xlim(left=0)
    fig_weibull.suptitle(f'Wind Site Model: Weibull Distribution (A={WEIBULL_A}, k={WEIBULL_K})', fontsize=16)
    ax_weibull.legend(loc='best'); plt.tight_layout()
    plt.savefig("wind_distribution.png")
    
    
    # 7. Plot the regression justification plot
    """Regression justfiication plot:
    
    This plot is also for the report to justify our choice of
    numerical method.
    
    We must prove that our chosen method (6th-order polynomial
    regression) is a good model for the raw NREL data.
    
    What it shows:
    - The raw NREL data is plotted as red squares.
    - Our smooth, 6th-order regression model (the 'base_power_fn')
      is plotted as a solid blue line.
    
    Notice that:
    This plot provides visual proof that our regression model is an
    excellent best fit for the raw data, capturing the complex
    'S' shape of the power curve. The dips (e.g., at v=3, 20m/s)
    are known numerical artifacts of fitting a smooth polynomial
    to data with sharp chnanges in direction.
    Our model is robust to this, as our constraints
    (e.g., <0 or >5MW) correct for these artifacts."""

    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    
    # Plot the original noisy data points
    v_data_raw = BASE_POWER_CURVE_DATA[:, 0]
    p_data_raw = BASE_POWER_CURVE_DATA[:, 1]
    ax_reg.plot(v_data_raw, p_data_raw, 'rs', label='NREL 5MW Raw Data (L=61.5m)')
    
    # Plot the smooth best fit line from our regression
    # We create a new x-axis for a smooth line
    v_smooth = np.linspace(0, 30, 200)
    
    # Use the 'base_power_fn_for_plot' we saved earlier.
    p_smooth_kw = base_power_fn_for_plot(v_smooth)
    
    # Create a "mask" to only plot the valid operational range
    # (between cut-in at 3 m/s and cut-out at 25 m/s)
    op_range_mask = (v_smooth >= 3) & (v_smooth <= 25)
    
    # Apply constraints to the plot data
    p_smooth_kw[p_smooth_kw > 5000] = 5000 # Cap at 5MW
    p_smooth_kw[p_smooth_kw < 0] = 0   # Floor at 0MW
    
    # Plot the model only where the mask is true
    ax_reg.plot(v_smooth[op_range_mask], p_smooth_kw[op_range_mask], 
                'b-', lw=2, label='6th-Order Regression Fit (Our Model)')
    
    ax_reg.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax_reg.set_ylabel('Power (kW)', fontsize=12)
    ax_reg.grid(True, linestyle='--')
    ax_reg.set_ylim(bottom=0)
    ax_reg.set_xlim(left=0)
    fig_reg.suptitle('Justification of The Regression Model Fit', fontsize=16)
    ax_reg.legend(loc='best'); plt.tight_layout()
    plt.savefig("regression_justifcation.png")
