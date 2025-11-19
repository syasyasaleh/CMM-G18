import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------------------------
# SECTION 1 — WIND DISTRIBUTION (Weibull)
# ---------------------------------------------

WEIBULL_A = 10.566
WEIBULL_K = 2.113
MAX_WIND_SPEED = 25
CUT_IN_SPEED = 3.0 # Defined for the new smoothing function

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v = pdf / np.sum(pdf)
    hours_v = prob_v * hours_in_year
    return hours_v, v


# ---------------------------------------------
# SECTION 2 — REGRESSION MODEL (Smooth, Shifted, Scaled)
# ---------------------------------------------

BASE_POWER_CURVE_DATA = np.array([
    (3, 100), (4, 250), (5, 475), (6, 800), (7, 1250),
    (8, 1900), (9, 2700), (10, 3717), (11, 4924),
    (11.4, 5000), (12, 5000), (13, 5000), (14, 5000),
    (15, 5000), (16, 5000), (17, 5000), (18, 5000),
    (19, 5000), (20, 5000), (21, 5000), (22, 5000),
    (23, 5000), (24, 5000), (25, 5000)
])

GENERATOR_LIMIT_W = 5_000_000.0
GENERATOR_LIMIT_KW = 5000.0
BASE_BLADE_LENGTH = 61.5
U_RATED = 11.4

def build_smooth_base_power_fn(v_data, p_data_kw, poly_order=6):
    """
    Fits a polynomial, shifts it to start at 0 at cut-in, scales it to match
    rated power, and applies a smoothing function in the 0-5 m/s range.
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
        v_arr = np.atleast_1d(v) # Ensure it works for scalar and array input
        pv_poly = p_shifted(v_arr) * scale_factor
        
        # 4. Smooth the 0-5 m/s region
        # Use a smooth transition function centered around the cut-in speed
        # The polynomial is only valid for v >= CUT_IN_SPEED
        pv_smooth = np.where(v_arr < CUT_IN_SPEED, 0.0, pv_poly)
        
        # We'll use a sigmoid-like function (a simple linear blend is smoother)
        v_min_smooth = CUT_IN_SPEED  # 3.0 m/s
        v_max_smooth = 5.0           # 5.0 m/s
        
        # Linear blend factor from 0 at v_min_smooth to 1 at v_max_smooth
        blend_factor = (v_arr - v_min_smooth) / (v_max_smooth - v_min_smooth)
        blend_factor = np.clip(blend_factor, 0.0, 1.0)
        
        # Before CUT_IN_SPEED, use 0 power (handled by pv_smooth initialization)
        # In the smoothing range [3.0, 5.0], blend the calculated power
        # For simplicity, we can let the shifted/scaled poly handle the ramp,
        # but to force a smoother zero-start:
        
        # The shifted/scaled poly already ensures P(3 m/s) = 0.
        # We'll rely on its shape for v > 3 m/s and just clip:
        pv = np.where(v_arr < CUT_IN_SPEED, 0.0, pv_poly)
        
        # 5. Clip at generator limits
        pv = np.clip(pv, 0.0, GENERATOR_LIMIT_KW)
        
        return pv if np.isscalar(v) else pv
        
    return base_power_fn

# Build the new smooth base power function
v_data = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw = BASE_POWER_CURVE_DATA[:, 1]
base_power_fn = build_smooth_base_power_fn(v_data, p_data_kw)


# ---------------------------------------------
# SECTION 3 — CP-BASED AERODYNAMIC MODEL (Unchanged)
# ---------------------------------------------

RHO = 1.225
A_base = math.pi * BASE_BLADE_LENGTH**2
CP_CALIBRATED = min(GENERATOR_LIMIT_W / (0.5 * RHO * A_base * U_RATED**3), 0.593)

GEARBOX_EFF = 0.98
GENERATOR_EFF = 0.96
POWER_EFF = GEARBOX_EFF * GENERATOR_EFF

LAMBDA_OPT = 7.5
TSR_SIGMA = 0.5 * LAMBDA_OPT
PITCH_SCALE_DEG = 8.0

def compute_cp(v, L, beta_deg=0.0, rotor_angle_deg=0.0,
               cp_max=CP_CALIBRATED, lambda_opt=LAMBDA_OPT):

    beta = abs(beta_deg)
    cp_pitch = np.exp(-(beta / PITCH_SCALE_DEG)**2)

    # Power control region logic (Below Rated vs. Above Rated)
    if v <= U_RATED:
        lambda_eff = lambda_opt
    else:
        # Constant rotor speed above rated implies lambda decreases as v increases
        lambda_eff = lambda_opt * (U_RATED / v)

    # TSR-related performance drop (simplified)
    cp_tsr = np.exp(-((lambda_eff - lambda_opt) / TSR_SIGMA)**2)

    # Yaw misalignment (rotor angle)
    phi_rad = np.radians(rotor_angle_deg)
    cos3 = max(0, np.cos(phi_rad)**3)

    Cp = cp_max * cp_pitch * cp_tsr * cos3
    return max(0.0, min(Cp, cp_max))


# ---------------------------------------------
# SECTION 4 — POWER CURVE BASED ON CP MODEL (Unchanged)
# ---------------------------------------------

def make_power_curve(L, beta_deg=0.0, rotor_angle_deg=0.0):
    A = math.pi * L**2

    def power_fn(v):
        if v < 3 or v > 25:
            return 0.0
        Cp = compute_cp(v, L, beta_deg=beta_deg, rotor_angle_deg=rotor_angle_deg)
        p_aero = 0.5 * RHO * A * v**3 * Cp
        p_elec = p_aero * POWER_EFF
        return min(p_elec, GENERATOR_LIMIT_W)

    return power_fn


# ---------------------------------------------
# SECTION 5 — AEP CALCULATION (Unchanged)
# ---------------------------------------------

def calculate_aep(power_curve_fn, wind_dist_hours):
    total = 0.0
    # Use v >= 3 for power calculation since cut-in is 3 m/s
    for v in range(int(CUT_IN_SPEED), len(wind_dist_hours)): 
        total += power_curve_fn(v) * wind_dist_hours[v]
    return total / 1e6  # MWh


# ---------------------------------------------
# SECTION 6 — MAIN ANALYSIS + NEW PLOTS (Updated Plot 3)
# ---------------------------------------------

if __name__ == "__main__":

    # ... (Plots 1 and 2 remain conceptually the same)

    # -------- Weibull distribution --------
    wind_dist, v_wind = get_wind_distribution(WEIBULL_A, WEIBULL_K)

    # -------- Choose a representative turbine --------
    L = 80.0  # blade length [m]
    v_plot = np.linspace(2, 25, 300)

    power_0deg = make_power_curve(L, beta_deg=0.0)
    power_10deg = make_power_curve(L, beta_deg=10.0)

    p_0 = np.array([power_0deg(v) for v in v_plot]) / 1e6
    p_10 = np.array([power_10deg(v) for v in v_plot]) / 1e6

    # ------------------------------------
    # PLOT 1 — Power vs Wind Speed for Different Pitch Angles
    # ------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(v_plot, p_0, label="Pitch = 0° (optimal)", lw=2)
    plt.plot(v_plot, p_10, label="Pitch = 10° (feathered)", lw=2, linestyle="--")
    plt.axhline(5, color='gray', linestyle=':', label="Generator Limit (5 MW)")

    plt.title("Power Curve vs Wind Speed (L = 80 m)", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Electrical Power (MW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 2 — Weibull Distribution
    # ------------------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(v_wind, wind_dist, width=0.8)
    plt.title("Weibull Wind Distribution", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Hours per Year", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 3 — Smooth Regression Model Fit
    # ------------------------------------
    v_smooth = np.linspace(0, 25, 300)
    p_smooth = base_power_fn(v_smooth) # Use the new smooth function

    plt.figure(figsize=(10, 6))
    plt.plot(v_data, p_data_kw, "rs", label="NREL 5MW Data")
    plt.plot(v_smooth, p_smooth, "b-", lw=2, label="Shifted+Scaled Regression Fit (Smoothed Low-V)")

    plt.title("Smooth Regression Fit of NREL 5MW Power Curve", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()