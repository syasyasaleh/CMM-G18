import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d # Need to import for more complex interpolation, but we'll use np.interp

# ---------------------------------------------
# SECTION 1 — WIND DISTRIBUTION (Weibull)
# ---------------------------------------------

WEIBULL_A = 10.566
WEIBULL_K = 2.113
MAX_WIND_SPEED = 25
CUT_OUT_SPEED = 25.0
CUT_IN_SPEED = 3.0 # Defined for operational check

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v = pdf / np.sum(pdf)
    hours_v = prob_v * hours_in_year
    return hours_v, v


# ---------------------------------------------
# SECTION 2 — INTERPOLATION MODEL
# ---------------------------------------------

BASE_POWER_CURVE_DATA = np.array([
    # Added 0 power at a point below cut-in to ensure a flat zero plateau
    (0.0, 0),
    (2.4, 0),
    (3, 100), (4, 250), (5, 475), (6, 800), (7, 1250),
    (8, 1900), (9, 2700), (10, 3717), (11, 4924),
    (11.4, 5000), (12, 5000), (13, 5000), (14, 5000),
    (15, 5000), (16, 5000), (17, 5000), (18, 5000),
    (19, 5000), (20, 5000), (21, 5000), (22, 5000),
    (23, 5000), (24, 5000), 
    (25, 5000),
    # Added 0 power at a point above cut-out to ensure a flat zero plateau
    (25.1, 0),
    (30.0, 0)
])

GENERATOR_LIMIT_W = 5_000_000.0
GENERATOR_LIMIT_KW = 5000.0
BASE_BLADE_LENGTH = 61.5

# Extract wind speed (x) and power (y) data
v_data = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw = BASE_POWER_CURVE_DATA[:, 1]

# --- Interpolation Function ---
def base_power_fn(v):
    """
    Interpolates the power curve data (v_data, p_data_kw) using linear interpolation.
    np.interp handles extrapolation by using the first/last data points, 
    but we have manually added 0-power points to handle the plateaus at 0.
    """
    p_interp = np.interp(v, v_data, p_data_kw)
    
    # We must still ensure the interpolated result is physically constrained,
    # though our data points help enforce this.
    p_interp = np.clip(p_interp, 0.0, GENERATOR_LIMIT_KW)
    
    return p_interp

# ---------------------------------------------
# SECTION 3 — CP-BASED AERODYNAMIC MODEL (Unchanged)
# ---------------------------------------------

RHO = 1.225
U_RATED = 11.4

# Calibrate Cp_max using the base rotor
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

    if v <= U_RATED:
        lambda_eff = lambda_opt
    else:
        lambda_eff = lambda_opt * (U_RATED / v)

    cp_tsr = np.exp(-((lambda_eff - lambda_opt) / TSR_SIGMA)**2)

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
        # Operational range handled by the CP model and explicit checks
        if v < CUT_IN_SPEED or v > CUT_OUT_SPEED:
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
    for v in range(1, len(wind_dist_hours)):
        # Ensure result is a scalar float for printing/storage
        total += float(power_curve_fn(v)) * wind_dist_hours[v] 
    return total / 1e6  # MWh


# ---------------------------------------------
# SECTION 6 — MAIN ANALYSIS + NEW PLOTS (Updated Plot 3)
# ---------------------------------------------

if __name__ == "__main__":

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
    # PLOT 3 — Interpolation Model Fit
    # ------------------------------------
    v_smooth = np.linspace(0, 25, 300)
    p_smooth = base_power_fn(v_smooth) # Use the interpolation function

    plt.figure(figsize=(10, 6))
    # Note: We must plot the *original* data points to see the fit
    v_data_op = BASE_POWER_CURVE_DATA[(BASE_POWER_CURVE_DATA[:,0] >= CUT_IN_SPEED) & (BASE_POWER_CURVE_DATA[:,0] <= CUT_OUT_SPEED), 0]
    p_data_kw_op = BASE_POWER_CURVE_DATA[(BASE_POWER_CURVE_DATA[:,0] >= CUT_IN_SPEED) & (BASE_POWER_CURVE_DATA[:,0] <= CUT_OUT_SPEED), 1]
    
    plt.plot(v_data_op, p_data_kw_op, "rs", label="NREL 5MW Operational Data")
    plt.plot(v_smooth, p_smooth, "b-", lw=2, label="Linear Interpolation Fit")

    plt.title("Interpolation Fit of NREL 5MW Power Curve", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()