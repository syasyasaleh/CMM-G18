"""
Updated power analysis script.

Changes:
- Implements a simplified Cp(lambda, beta) model with efficiency,
  and uses it to compute power vs wind speed for pitch=0deg and pitch=10deg.
- Adjusts the polynomial regression so power starts at zero at cut-in (3 m/s)
  and scales to match rated power at rated wind speed (11.4 m/s).
- Keeps the original AEP vs blade-length loop (scaled regression).
- Saves plots and calls plt.show() for VS Code display.

Reference (used for justification / model assumptions): Johlas et al., "Floating
Platform Effects on Power Generation..." NREL (2021). The paper documents that
time-averaged rotor pitch reduces power and provides NREL 5MW reference turbine
specifications (D = 126 m, rated = 11.4 m/s), which motivate including pitch
effects. :contentReference[oaicite:1]{index=1}
"""
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Site / Weibull parameters
# -----------------------------
WEIBULL_A = 10.566  # scale [m/s]
WEIBULL_K = 2.113   # shape [-]
MAX_WIND_SPEED = 25 # cut-out [m/s]
CUT_IN_SPEED = 3.0
RATED_SPEED = 11.4  # from NREL 5MW reference. :contentReference[oaicite:2]{index=2}

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    # Weibull PDF
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v_normalised = pdf / np.sum(pdf)
    hours_v = prob_v_normalised * hours_in_year
    return hours_v, v

# -----------------------------
# Base NREL power curve data (kW) - same as you provided
# -----------------------------
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
    (11.4, 5000), # rated
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
BASE_BLADE_LENGTH = 61.5  # [m] NREL baseline

# -----------------------------
# Regression: fit a polynomial but shift so P(cut-in)=0 and scale to match rated
# -----------------------------
def build_shifted_scaled_base_power_fn(v_data, p_data_kw):
    """
    Fit polynomial, then shift so power at cut-in (3 m/s) is zero,
    and scale so power at rated speed (11.4 m/s) matches rated power.
    Returns a function that gives power in kW for any v (not constrained to generator limit).
    """
    # Fit 6th-order as before
    coeffs = np.polyfit(v_data, p_data_kw, 6)
    p_poly = np.poly1d(coeffs)
    
    # Evaluate offset at cut-in
    offset = p_poly(CUT_IN_SPEED)
    # Shifted polynomial (so P(cut-in)=0)
    def p_shifted(v):
        return p_poly(v) - offset
    
    # Evaluate shifted value at rated to get scale factor
    val_at_rated = p_shifted(RATED_SPEED)
    if val_at_rated <= 0:
        # fallback: avoid divide by zero — just return original poly clipped
        def base_fn(v):
            pv = p_poly(v)
            pv = np.where(pv < 0, 0.0, pv)
            return pv
        return base_fn
    
    scale = 5000.0 / val_at_rated  # scale so shifted value at rated becomes 5000 kW
    
    def base_fn(v):
        # If v is array-like handle vectors, else scalar
        v_arr = np.array(v, copy=False)
        pv = p_shifted(v_arr) * scale
        # clip negatives
        pv = np.where(pv < 0.0, 0.0, pv)
        # for values > rated region, cap at rated (5 MW) in kW
        pv = np.where(pv > 5000.0, 5000.0, pv)
        return pv
    return base_fn

# Build the base power function once
v_data = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw = BASE_POWER_CURVE_DATA[:, 1]
base_power_fn_kw = build_shifted_scaled_base_power_fn(v_data, p_data_kw)
# base_power_fn_kw returns kW

# -----------------------------
# Cp model (simplified), inspired by discussion in the PDF about pitch reducing power.
# The PDF documents that a time-averaged rotor pitch reduces power; we implement
# a Cp(λ, β) that decreases with pitch and with off-optimal TSR. :contentReference[oaicite:3]{index=3}
# -----------------------------
# Mechanical/electrical efficiency factor (generator + drivetrain + misc)
ETA_MECH = 0.95

def cp_simplified(lambda_tsr, beta_deg, lambda_opt=8.0, cp_max=0.48, k_beta=0.02):
    """
    cp_simplified - returns Cp (unitless) for given tip speed ratio and pitch angle.
    - lambda_opt: optimal TSR for which Cp peaks (typical large turbines: 6-10)
    - cp_max: maximum attainable Cp at beta=0 and lambda=lambda_opt (below Betz limit)
    - k_beta: Cp reduction fraction per degree of pitch (2% per degree as a tunable param)
    """
    # pitch reduction factor
    pitch_factor = max(0.0, 1.0 - k_beta * abs(beta_deg))
    # TSR shape: quadratic drop-off from lambda_opt
    tsr_term = 1.0 - ((lambda_tsr - lambda_opt) / lambda_opt)**2
    cp_raw = cp_max * max(0.0, tsr_term) * pitch_factor
    cp = ETA_MECH * cp_raw
    return cp

def power_from_cp(rho, v, L, cp):
    """
    Convert Cp to power [W]: P = 0.5 * rho * A * v^3 * Cp
    where A = pi * L^2 (blade length L taken as rotor radius).
    """
    A = np.pi * L**2
    P = 0.5 * rho * A * v**3 * cp
    return P

# -----------------------------
# Create scaled power curve function using either regression scaling (as before)
# or the Cp model for a particular blade length
# -----------------------------
def create_scaled_power_curve_fn(L, base_power_fn, use_cp=False, cp_params=None):
    """
    Returns a function power_curve_fn(v) that gives power in W for wind speed v:
    - If use_cp==False: uses shifted+scaled regression (base_power_fn) and scales
      by area ratio (L/BASE_BLADE_LENGTH)^2 then applies limits (cut-in/cut-out/generator).
    - If use_cp==True: computes Cp via cp_simplified and returns aerodynamic power,
      capped by generator limit. cp_params can include keys like 'beta_deg' and 'lambda_opt'.
    """
    def scaled_power_curve_fn(v):
        # scalar v
        if v < CUT_IN_SPEED or v > MAX_WIND_SPEED:
            return 0.0
        
        if not use_cp:
            # regression-based approach (base_power_fn returns kW)
            p_kw = base_power_fn(v)
            p_w = p_kw * 1000.0
            scaled = p_w * (L / BASE_BLADE_LENGTH)**2
            if scaled > GENERATOR_LIMIT_W:
                return GENERATOR_LIMIT_W
            if scaled < 0:
                return 0.0
            return scaled
        else:
            # physics-based Cp approach
            # default cp params
            beta_deg = cp_params.get('beta_deg', 0.0) if cp_params else 0.0
            lambda_opt = cp_params.get('lambda_opt', 8.0) if cp_params else 8.0
            cp_max = cp_params.get('cp_max', 0.48) if cp_params else 0.48
            k_beta = cp_params.get('k_beta', 0.02) if cp_params else 0.02
            rho = cp_params.get('rho', 1.225) if cp_params else 1.225
            
            # We model variable speed control in below-rated region:
            # For v <= RATED_SPEED: assume generator maintains lambda ~ lambda_opt (variable speed).
            # So, lambda_tsr = lambda_opt (by setting rotor speed accordingly).
            # For v > RATED_SPEED: assume rotor speed constant (set by lambda at rated).
            # Compute lambda_tsr:
            R = L  # blade length as radius
            if v <= RATED_SPEED:
                lambda_tsr = lambda_opt
            else:
                # lambda at rated: lambda_rated = lambda_opt (by definition above)
                # but when v increases above rated, the lambda falls proportionally
                lambda_tsr = lambda_opt * (RATED_SPEED / v)
            
            cp = cp_simplified(lambda_tsr, beta_deg, lambda_opt=lambda_opt, cp_max=cp_max, k_beta=k_beta)
            P_w = power_from_cp(rho, v, L, cp)
            # Apply generator cap and safety floor
            if P_w > GENERATOR_LIMIT_W:
                P_w = GENERATOR_LIMIT_W
            if P_w < 0:
                P_w = 0.0
            # cut-out/cut-in already handled
            return P_w
    return scaled_power_curve_fn

# -----------------------------
# AEP calculation (unchanged)
# -----------------------------
def calculate_aep(power_curve_fn, wind_dist_hours):
    total_energy_wh = 0.0
    for v in range(1, len(wind_dist_hours)):
        power_w = power_curve_fn(v)
        hours = wind_dist_hours[v]
        energy_wh = power_w * hours
        total_energy_wh += energy_wh
    total_energy_mwh = total_energy_wh / 1_000_000.0
    return total_energy_mwh

# -----------------------------
# Main analysis
# -----------------------------
if __name__ == "__main__":
    # Wind distribution
    wind_dist, v_wind_speeds = get_wind_distribution(WEIBULL_A, WEIBULL_K)
    
    # Blade lengths to test (original sweep)
    blade_lengths_to_test = np.arange(60, 161, 5)
    print(f"Running {len(blade_lengths_to_test)} design cases for 60m to 160m")
    
    aep_results = []
    for L in blade_lengths_to_test:
        # using regression scaling (not Cp) for AEP sweep
        power_fn = create_scaled_power_curve_fn(L, base_power_fn_kw, use_cp=False)
        aep = calculate_aep(power_fn, wind_dist)
        aep_results.append((L, aep))
        print(f"  L = {L:3d} m, AEP = {aep:,.0f} MWh/yr")
    
    # Print table
    print("\n AEP Results Table")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 36)
    for L, aep in aep_results:
        print(f" {L:<15.2f} | {aep:,.0f}")
    
    # Plot 1: AEP vs Blade Length (regression scaled)
    fig, ax = plt.subplots(figsize=(11, 7))
    L_plot = np.array([row[0] for row in aep_results])
    AEP_plot = np.array([row[1] for row in aep_results])
    ax.plot(L_plot, AEP_plot, 'bo-', label='AEP (MWh/yr) - regression scaled')
    ax.set_xlabel('Blade Length, L (m)', fontsize=12)
    ax.set_ylabel('Annual Energy Production, AEP (MWh/yr)', fontsize=12)
    ax.grid(True, linestyle='--')
    ax.set_ylim(bottom=0)
    fig.suptitle('Design Analysis: AEP vs. Blade Length (Regression-scaled)', fontsize=14)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig("power_analysis_curve.png")
    plt.show()
    
    # -----------------------------
    # Physics-based Cp comparison plot:
    # Power vs Wind Speed for pitch = 0 deg and pitch = 10 deg
    # Use representative blade length in 70-90 m range: choose 80 m
    # -----------------------------
    representative_L = 80.0  # [m] (midpoint of 70-90m)
    v_array = np.linspace(0.0, 25.0, 200)
    pitch_angles = [0.0, 10.0]  # degrees

    plt.figure(figsize=(11, 7))
    for beta in pitch_angles:
        P_vals_mw = []
        # Use Cp parameters - can be tuned
        cp_params = {
            'beta_deg': beta,
            'lambda_opt': 8.0,
            'cp_max': 0.48,
            'k_beta': 0.02,
            'rho': 1.225
        }
        power_fn_cp = create_scaled_power_curve_fn(representative_L, base_power_fn_kw, use_cp=True, cp_params=cp_params)
        
        for v in v_array:
            P_w = power_fn_cp(v)
            P_vals_mw.append(P_w / 1e6)
        
        plt.plot(v_array, P_vals_mw, label=f"Pitch = {beta:.0f}° (Cp model, L={representative_L} m)")
    
    # Also plot the shifted + scaled regression (for the same L) for comparison
    # so user sees both approaches on same plot.
    power_fn_reg = create_scaled_power_curve_fn(representative_L, base_power_fn_kw, use_cp=False)
    P_reg = [power_fn_reg(v) / 1e6 for v in v_array]
    plt.plot(v_array, P_reg, 'k--', lw=1.5, label=f"Regression-scaled (L={representative_L} m)")
    
    plt.xlabel('Wind speed (m/s)', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.title('Power vs Wind Speed — Cp model (0° & 10° pitch) and Regression (for comparison)')
    plt.grid(True, linestyle='--')
    plt.legend(loc='best')
    plt.xlim(0, 25)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("power_vs_windspeed_cp_pitch_comparison.png")
    plt.show()
    
    # -----------------------------
    # Weibull justification plot (as before)
    # -----------------------------
    fig_w, ax_w = plt.subplots(figsize=(10, 6))
    ax_w.bar(v_wind_speeds, wind_dist, label='Hours per Year', width=0.8)
    ax_w.set_xlabel('Wind Speed, v (m/s)', fontsize=12)
    ax_w.set_ylabel('Hours per Year (h)', fontsize=12)
    ax_w.grid(True, linestyle='--')
    fig_w.suptitle(f'Wind Site Model: Weibull Distribution (A={WEIBULL_A}, k={WEIBULL_K})', fontsize=14)
    ax_w.legend(loc='best')
    plt.tight_layout()
    plt.savefig("wind_distribution.png")
    plt.show()
    
    # -----------------------------
    # Regression justification plot (as before) but using the shifted/scaled base_fn
    # -----------------------------
    fig_r, ax_r = plt.subplots(figsize=(10, 6))
    v_data_raw = BASE_POWER_CURVE_DATA[:, 0]
    p_data_raw = BASE_POWER_CURVE_DATA[:, 1]
    ax_r.plot(v_data_raw, p_data_raw, 'rs', label='NREL 5MW Raw Data (L=61.5m)')
    
    v_smooth = np.linspace(0, 30, 300)
    p_smooth_kw = base_power_fn_kw(v_smooth)
    # Already shifted/scaled, but mask operational range
    op_range_mask = (v_smooth >= CUT_IN_SPEED) & (v_smooth <= MAX_WIND_SPEED)
    ax_r.plot(v_smooth[op_range_mask], p_smooth_kw[op_range_mask], 'b-', lw=2, label='Shifted+Scaled Regression Fit')
    
    ax_r.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax_r.set_ylabel('Power (kW)', fontsize=12)
    ax_r.grid(True, linestyle='--')
    ax_r.set_ylim(bottom=0)
    ax_r.set_xlim(left=0)
    fig_r.suptitle('Regression Justification: shifted & scaled to start at zero at cut-in', fontsize=14)
    ax_r.legend(loc='best')
    plt.tight_layout()
    plt.savefig("regression_justification_shifted.png")
    plt.show()
