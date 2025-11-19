import numpy as np
import matplotlib.pyplot as plt

# Section 1: Wind Distribution (Weibull)

# Parameters for a typical North Sea site
WEIBULL_A = 10.566 # Scale parameter [m/s]
WEIBULL_K = 2.113 # Shape parameter [-]
MAX_WIND_SPEED = 25 # Max speed to check (matches cut-out)
CUT_IN_SPEED = 3.0
U_RATED = 11.4 # Rated speed for NREL 5MW

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    """
    This function returns an array of hours per year for each wind speed (1 to max_v)
    using a Weibull distribution.
    """
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    # Weibull PDF
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v_normalised = pdf / np.sum(pdf)
    hours_v = prob_v_normalised * hours_in_year
    
    return hours_v, v


# --------------------------------------------------------
# Section 2: Power curve model using a Smooth Regression
# --------------------------------------------------------

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
GENERATOR_LIMIT_KW = 5000.0
BASE_BLADE_LENGTH = 61.5  # [m]

def build_smooth_base_power_fn(v_data, p_data_kw, poly_order=6):
    """
    Fits a polynomial, shifts it so P(cut-in)=0, scales it so P(rated)=5000 kW,
    and applies clipping for a smooth, physical power curve.
    """
    # 1. Fit original 6th-order polynomial
    poly_coeff = np.polyfit(v_data, p_data_kw, poly_order)
    p_poly = np.poly1d(poly_coeff)

    # 2. Shift to make P(CUT_IN_SPEED) = 0
    offset = p_poly(CUT_IN_SPEED)
    def p_shifted(v):
        return p_poly(v) - offset

    # 3. Scale to make P(U_RATED) = GENERATOR_LIMIT_KW (5000 kW)
    val_at_rated = p_shifted(U_RATED)
    scale_factor = GENERATOR_LIMIT_KW / val_at_rated if val_at_rated > 0 else 1.0

    def base_power_fn(v):
        v_arr = np.atleast_1d(v) # Handle scalar or array input
        pv_poly = p_shifted(v_arr) * scale_factor
        
        # 4. Clipping and Plateau at Zero:
        # Power must be zero below cut-in speed
        pv = np.where(v_arr < CUT_IN_SPEED, 0.0, pv_poly)
        
        # Power must be clipped at generator limit (5000 kW)
        pv = np.clip(pv, 0.0, GENERATOR_LIMIT_KW)
        
        # Power must be zero above cut-out speed
        pv = np.where(v_arr > MAX_WIND_SPEED, 0.0, pv)
        
        return pv if np.isscalar(v) else pv
        
    return base_power_fn

def create_scaled_power_curve_fn(L, base_power_fn):

    def scaled_power_curve_fn(v):
        # base_power_fn returns power in kW
        p_kw = base_power_fn(v)
        p_w = p_kw * 1000.0
        
        # Scale by area ratio (L/BASE_BLADE_LENGTH)^2
        scaled_power_w = p_w * (L / BASE_BLADE_LENGTH)**2
        
        # Apply final operational limits
        if v < CUT_IN_SPEED or v > MAX_WIND_SPEED:
            return 0.0
        
        # Clip at generator limit
        if scaled_power_w > GENERATOR_LIMIT_W:
            return GENERATOR_LIMIT_W
            
        return max(0.0, scaled_power_w)

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
    
    # --- New base power function generation ---
    v_data = BASE_POWER_CURVE_DATA[:, 0]
    p_data_kw = BASE_POWER_CURVE_DATA[:, 1]
    base_power_fn = build_smooth_base_power_fn(v_data, p_data_kw)
    # ------------------------------------------

    print(f"Running {len(blade_lengths_to_test)} design cases for 60m to 160m")
    
    aep_results = []

    for L in blade_lengths_to_test:
        # Use the new, smooth base_power_fn
        power_fn = create_scaled_power_curve_fn(L, base_power_fn) 
        aep = float(calculate_aep(power_fn, wind_dist))
        
        aep_results.append((L, aep))
        print(f"  L = {L:3d} m, AEP = {aep:,.0f} MWh/yr")


    print("\n AEP Results Table")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 30)
    for L, aep in aep_results:
        print(f" {L:<15.2f} | {aep:,.0f}")
        

    # 5. Plot the final power curve (AEP vs L)
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
    plt.show()


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
    plt.show()


    # 7. Plot the regression justification (using the new smooth model)
    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    v_data_raw = BASE_POWER_CURVE_DATA[:, 0]
    p_data_raw = BASE_POWER_CURVE_DATA[:, 1]
    ax_reg.plot(v_data_raw, p_data_raw, 'rs', label='NREL 5MW Raw Data')
    
    v_smooth = np.linspace(0, 30, 300)
    p_smooth_kw = base_power_fn(v_smooth) # Use the new smooth function
    
    # We plot the full curve (0 to 30 m/s) to show the plateau and cut-out
    ax_reg.plot(v_smooth, p_smooth_kw, 
                'b-', lw=2, label='Shifted, Scaled & Plateaued Regression Fit')
    
    ax_reg.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax_reg.set_ylabel('Power (kW)', fontsize=12)
    ax_reg.grid(True, linestyle='--')
    ax_reg.set_ylim(bottom=0)
    ax_reg.set_xlim(left=0)
    fig_reg.suptitle('Regression Fit with Low-Speed Plateau (P=0 below Cut-In)', fontsize=16)
    ax_reg.legend(loc='best'); plt.tight_layout()
    plt.savefig("regression_justifcation_smooth_plateau.png")
    plt.show()