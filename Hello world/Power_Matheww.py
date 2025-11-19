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
WEIBULL_A = 10.566 # Scale parameter [m/s]
WEIBULL_K = 2.113 # Shape parameter [-]
MAX_WIND_SPEED = 25 # Max speed to check (matches cut-out)

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    """
    This function returns an array of hours per year for each wind speed (1 to max_v)
    using a Weibull distribution.
    """
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v = pdf
    prob_v_normalised = prob_v / np.sum(prob_v)
    hours_v = prob_v_normalised * hours_in_year
    
    return hours_v, v


# Section 2: Power curve model using regression

BASE_POWER_CURVE_DATA = np.array([
    (2.4, 0),
    (3, 100),
    (4, 250),
    (5, 475),
    (6, 800),
    (7, 1250),
    (8, 1900),
    (9, 2700),
    (10, 3717),
    (11, 4924),
    (11.4, 5000),
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
    (25, 5000)
])

GENERATOR_LIMIT_W = 5_000_000.0
BASE_BLADE_LENGTH = 61.5  # [m]

def create_scaled_power_curve_fn(L, base_power_fn):

    def scaled_power_curve_fn(v):
        base_power_w = base_power_fn(v) * 1000
        scaled_power_w = base_power_w * (L / BASE_BLADE_LENGTH)**2
        
        if v < 1.75:
            return 0.0
        if scaled_power_w > GENERATOR_LIMIT_W:
            return GENERATOR_LIMIT_W
        if v > 25.0:
            return 0.0
        if scaled_power_w < 0:
            return 0.0
            
        return scaled_power_w

    return scaled_power_curve_fn


# Section 3: The AEP calculation

def calculate_aep(power_curve_fn, wind_dist_hours):
    total_energy_wh = 0.0
    for v in range(1, len(wind_dist_hours)):
        power_w = power_curve_fn(v)
        hours = wind_dist_hours[v]
        energy_wh = power_w * hours
        total_energy_wh += energy_wh
    
    total_energy_mwh = total_energy_wh / 1_000_000.0
    return total_energy_mwh


# Section 4: The main analysis section

if __name__ == "__main__":

    wind_dist, v_wind_speeds = get_wind_distribution(WEIBULL_A, WEIBULL_K)
    blade_lengths_to_test = np.arange(60, 161, 5)
    
    print(f"Running {len(blade_lengths_to_test)} design cases for 60m to 160m")
    
    aep_results = []

    v_data = BASE_POWER_CURVE_DATA[:, 0]
    p_data_kw = BASE_POWER_CURVE_DATA[:, 1]
    coefficients = np.polyfit(v_data, p_data_kw, 6)
    base_power_fn_for_plot = np.poly1d(coefficients)

    for L in blade_lengths_to_test:
        power_fn = create_scaled_power_curve_fn(L, base_power_fn_for_plot)
        aep = calculate_aep(power_fn, wind_dist)
        
        aep_results.append((L, aep))
        print(f"  L = {L:3d} m, AEP = {aep:,.0f} MWh/yr")


    print(" AEP Results Table")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 30)
    for L, aep in aep_results:
        print(f" {L:<15.2f} | {aep:,.0f}")
        

    # 5. Plot the final power curve
    fig, ax = plt.subplots(figsize=(12, 8))
    L_plot = np.array([row[0] for row in aep_results])
    AEP_plot = np.array([row[1] for row in aep_results])
    
    ax.plot(L_plot, AEP_plot, 'bo-', label='AEP (MWh/yr)')
    ax.set_xlabel('Blade Length, L, (m)', fontsize=14)
    ax.set_ylabel('Annual Energy Production, AEP, (MWh/yr)', fontsize=14)
    ax.grid(True, linestyle='--')
    ax.set_ylim(bottom=0)
    fig.suptitle('Design Analysis: AEP vs. Blade Length', fontsize=18)
    ax.legend(loc='best'); plt.tight_layout()
    plt.savefig("power_analysis_curve.png")
    plt.show()  # <-- Added


    # 6. Plot the Weibull distribution
    fig_weibull, ax_weibull = plt.subplots(figsize=(10, 6))
    ax_weibull.bar(v_wind_speeds, wind_dist, label='Hours per Year', width=0.8)
    
    ax_weibull.set_xlabel('Wind Speed, v, (m/s)', fontsize=12)
    ax_weibull.set_ylabel('Hours per Year (h)', fontsize=12)
    ax_weibull.grid(True, linestyle='--')
    ax_weibull.set_xlim(left=0)
    fig_weibull.suptitle(f'Wind Site Model: Weibull Distribution (A={WEIBULL_A}, k={WEIBULL_K})', fontsize=16)
    ax_weibull.legend(loc='best'); plt.tight_layout()
    plt.savefig("wind_distribution.png")
    plt.show()  # <-- Added


    # 7. Plot the regression justification
    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    v_data_raw = BASE_POWER_CURVE_DATA[:, 0]
    p_data_raw = BASE_POWER_CURVE_DATA[:, 1]
    ax_reg.plot(v_data_raw, p_data_raw, 'rs', label='NREL 5MW Raw Data (L=61.5m)')
    
    v_smooth = np.linspace(0, 30, 200)
    p_smooth_kw = base_power_fn_for_plot(v_smooth)
    
    op_range_mask = (v_smooth >= 3) & (v_smooth <= 25)
    p_smooth_kw[p_smooth_kw > 5000] = 5000
    p_smooth_kw[p_smooth_kw < 0] = 0
    
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
    plt.show()  # <-- Added
