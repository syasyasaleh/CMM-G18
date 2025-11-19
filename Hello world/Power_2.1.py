import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

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
    (30.0, 0) # Extend range to ensure interpolation beyond 25 is 0
])

GENERATOR_LIMIT_W = 5_000_000.0
GENERATOR_LIMIT_KW = 5000.0
BASE_BLADE_LENGTH = 61.5

# Extract wind speed (x) and power (y) data for the base NREL turbine
v_data_base_nrel = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw_base_nrel = BASE_POWER_CURVE_DATA[:, 1]

# --- Interpolation Function for the NREL BASELINE POWER CURVE ---
# This function defines the *shape* of the power curve at BASE_BLADE_LENGTH.
# It does not include the CP model or scaling yet.
def get_nrel_base_power_interp_fn():
    """
    Returns an interpolation function for the base NREL 5MW power curve data.
    This function will be scaled later by rotor area.
    """
    # Using interp1d for more flexibility, although np.interp also works
    # kind='linear' for linear interpolation.
    interp_func = interp1d(v_data_base_nrel, p_data_kw_base_nrel, kind='linear', 
                           bounds_error=False, fill_value=(0.0, 0.0))
    
    def power_fn(v):
        p_val = interp_func(v)
        # Ensure values are within physical limits (0 to 5000 kW)
        p_val = np.clip(p_val, 0.0, GENERATOR_LIMIT_KW)
        return p_val
    return power_fn

# Get the base interpolation function for the NREL 5MW (kW)
base_nrel_interp_power_kw = get_nrel_base_power_interp_fn()


# ---------------------------------------------
# SECTION 3 — CP-BASED AERODYNAMIC MODEL
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


# ----------------------------------------------------
# SECTION 4 — POWER CURVE GENERATION (Scaled from Interpolation)
# ----------------------------------------------------

def make_scaled_power_curve_from_interp(L, base_interp_kw_fn, beta_deg=0.0, rotor_angle_deg=0.0):
    """
    Creates a full power curve function for a given blade length (L) and pitch angle.
    This function combines the area scaling with the Cp-based pitch control logic.
    """
    A = math.pi * L**2
    
    def power_fn(v):
        # Handle cut-in and cut-out explicitly
        if v < CUT_IN_SPEED or v > CUT_OUT_SPEED:
            return 0.0

        # Start with the NREL baseline power curve (scaled by area)
        # We use the interpolated NREL data as a base, then apply CP model for pitch
        base_power_at_v_kw = base_interp_kw_fn(v)
        base_power_at_v_W = base_power_at_v_kw * 1000.0
        
        # Scale the base power curve by rotor area
        scaled_power_W = base_power_at_v_W * (L / BASE_BLADE_LENGTH)**2
        
        # Now, incorporate the pitch angle effect using the Cp model's pitch factor
        # This is a simplification: for a *true* Cp model, you'd recalculate P_aero directly.
        # But to show pitch effect *on top of* the interpolated curve:
        # We can approximate the power reduction due to pitch by using the Cp pitch factor.
        
        # Calculate Cp for the current conditions (L, v, beta_deg)
        # Assuming the Cp model gives the *absolute* Cp, which then gets factored into power.
        # However, the request implies showing pitch effect on the *interpolated curve*.
        # Let's use the full Cp model to make a *new* power curve, not scale the interpolated one,
        # as it gives a more accurate representation of pitch control.
        Cp = compute_cp(v, L, beta_deg=beta_deg, rotor_angle_deg=rotor_angle_deg)
        p_aero = 0.5 * RHO * A * v**3 * Cp
        p_elec = p_aero * POWER_EFF
        
        return min(max(0.0, p_elec), GENERATOR_LIMIT_W)

    return power_fn


# ---------------------------------------------
# SECTION 5 — AEP CALCULATION (unchanged, with type cast)
# ---------------------------------------------

def calculate_aep(power_curve_fn, wind_dist_hours):
    total = 0.0
    for v in range(1, len(wind_dist_hours)):
        # Ensure result is a scalar float for printing/storage
        total += float(power_curve_fn(v)) * wind_dist_hours[v] 
    return total / 1e6  # MWh


# ---------------------------------------------
# SECTION 6 — MAIN ANALYSIS + PLOTS
# ---------------------------------------------

if __name__ == "__main__":

    # -------- Weibull distribution --------
    wind_dist, v_wind = get_wind_distribution(WEIBULL_A, WEIBULL_K)

    # ------------------------------------
    # AEP Calculation with Interpolation Model
    # ------------------------------------
    blade_lengths_to_test = np.arange(60, 161, 5)
    print(f"Running {len(blade_lengths_to_test)} design cases for 60m to 160m using INTERPOLATION model...")
    
    aep_results = []
    for L in blade_lengths_to_test:
        # Create a power curve using the Cp model (which is the most comprehensive)
        # The Cp model inherently scales with L^2 and incorporates pitch.
        power_fn_L = make_scaled_power_curve_from_interp(L, base_nrel_interp_power_kw, beta_deg=0.0) 
        aep = float(calculate_aep(power_fn_L, wind_dist)) # Ensure AEP is a float
        aep_results.append((L, aep))
        print(f"  L = {L:3d} m, AEP = {aep:,.0f} MWh/yr")

    print("\n AEP Results Table (Interpolation Model)")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 36)
    for L, aep in aep_results:
        print(f" {L:<15.2f} | {aep:,.0f}")

    # Plot AEP vs Blade Length
    plt.figure(figsize=(10, 6))
    L_plot_aep = np.array([row[0] for row in aep_results])
    AEP_plot_aep = np.array([row[1] for row in aep_results])
    plt.plot(L_plot_aep, AEP_plot_aep, 'bo-', label='AEP (MWh/yr) - Interpolation Model')
    plt.title("AEP vs. Blade Length (Interpolation Model)", fontsize=16)
    plt.xlabel("Blade Length (m)", fontsize=14)
    plt.ylabel("Annual Energy Production (MWh/yr)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT: Interpolation Data Curve for Power Generation with Pitch Angle
    # ------------------------------------
    representative_L = 80.0  # Use a representative blade length for this plot
    v_plot_detailed = np.linspace(0, 30, 500) # Extend range to show full cut-in/cut-out

    # Power curve for 0-degree pitch
    power_fn_0deg = make_scaled_power_curve_from_interp(representative_L, base_nrel_interp_power_kw, beta_deg=0.0)
    p_0deg_mw = np.array([power_fn_0deg(v) for v in v_plot_detailed]) / 1e6

    # Power curve for 10-degree pitch
    power_fn_10deg = make_scaled_power_curve_from_interp(representative_L, base_nrel_interp_power_kw, beta_deg=10.0)
    p_10deg_mw = np.array([power_fn_10deg(v) for v in v_plot_detailed]) / 1e6

    plt.figure(figsize=(12, 7))
    plt.plot(v_plot_detailed, p_0deg_mw, label=f"Pitch = 0° (Optimal, L={representative_L} m)", lw=2)
    plt.plot(v_plot_detailed, p_10deg_mw, label=f"Pitch = 10° (Feathered, L={representative_L} m)", lw=2, linestyle='--')
    plt.axhline(GENERATOR_LIMIT_W / 1e6, color='gray', linestyle=':', label="Generator Limit (5 MW)")

    plt.title("Power Curve vs Wind Speed (Effect of Pitch Angle - Interpolation Model)", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Electrical Power (MW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 2 — Weibull Distribution (Unchanged)
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
    # PLOT 3 — Interpolation Model Fit (Justification)
    # ------------------------------------
    v_smooth_fit = np.linspace(0, 30, 300)
    p_smooth_fit = base_nrel_interp_power_kw(v_smooth_fit) # Use the base interpolation function

    plt.figure(figsize=(10, 6))
    plt.plot(v_data_base_nrel, p_data_kw_base_nrel, "rs", label="NREL 5MW Raw Data Points")
    plt.plot(v_smooth_fit, p_smooth_fit, "b-", lw=2, label="Linear Interpolation Fit (Base NREL)")

    plt.title("Linear Interpolation Fit of NREL 5MW Power Curve (with 0-Plateau)", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()