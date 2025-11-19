import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate # Importing the full interpolate module for splrep/splev

# ---------------------------------------------
# SECTION 1 — WIND DISTRIBUTION (Weibull)
# ---------------------------------------------

WEIBULL_A = 10.566
WEIBULL_K = 2.113
MAX_WIND_SPEED = 25
CUT_OUT_SPEED = 25.0
CUT_IN_SPEED = 3.0 

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v = pdf / np.sum(pdf)
    hours_v = prob_v * hours_in_year
    return hours_v, v


# ---------------------------------------------
# SECTION 2 — SPLINE INTERPOLATION MODEL
# ---------------------------------------------

BASE_POWER_CURVE_DATA = np.array([
    # Padding points to ensure the spline knows the curve is flat at 0
    (0.0, 0),
    (1.75, 0),
    (3, 100), (4, 250), (5, 475), (6, 800), (7, 1250),
    (8, 1900), (9, 2700), (10, 3717), (11, 4924),
    (11.4, 5000), (12, 5000), (13, 5000), (14, 5000),
    (15, 5000), (16, 5000), (17, 5000), (18, 5000),
    (19, 5000), (20, 5000), (21, 5000), (22, 5000),
    (23, 5000), (24, 5000), 
    (25, 5000),
])

GENERATOR_LIMIT_W = 5_000_000.0
GENERATOR_LIMIT_KW = 5000.0
BASE_BLADE_LENGTH = 61.5

# Extract wind speed (x) and power (y) data
v_data_base_nrel = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw_base_nrel = BASE_POWER_CURVE_DATA[:, 1]

# --- SPLINE Interpolation Function ---
def get_nrel_base_power_spline_fn():
    """
    Returns a Spline interpolation function for the base NREL 5MW data.
    Uses splrep with s=0 (no smoothing, passes through all points).
    """
    # 1. Generate the B-spline representation (tck)
    # s=0 ensures it passes through every data point exactly
    tck = interpolate.splrep(v_data_base_nrel, p_data_kw_base_nrel, s=0)
    
    def power_fn(v):
        # 2. Evaluate the spline
        # splev can handle arrays or scalars
        p_val = interpolate.splev(v, tck, der=0)
        
        # Splines can sometimes "overshoot" or "undershoot" (oscillate) 
        # near sharp corners (like at 3m/s or 25m/s). 
        # We clip to ensure physical realism (no negative power, no power > max).
        p_val = np.clip(p_val, 0.0, GENERATOR_LIMIT_KW)
        return p_val
        
    return power_fn

# Get the base spline function for the NREL 5MW (kW)
base_nrel_spline_power_kw = get_nrel_base_power_spline_fn()


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

def compute_cp_pitch_factor(beta_deg):
    """
    Calculates only the degradation factor due to pitch angle.
    This allows us to scale the Spline Curve (which is optimal pitch) 
    by the relative loss of efficiency due to pitching.
    """
    beta = abs(beta_deg)
    cp_pitch = np.exp(-(beta / PITCH_SCALE_DEG)**2)
    return cp_pitch


# ----------------------------------------------------
# SECTION 4 — POWER CURVE GENERATION
# ----------------------------------------------------

def make_scaled_power_curve_from_spline(L, base_spline_kw_fn, beta_deg=0.0):
    """
    Creates a full power curve function for a given blade length (L) and pitch angle.
    
    LOGIC UPDATE:
    Instead of recalculating power from theoretical Cp (which ignored the spline),
    we now take the Spline Power (Optimal) and multiply it by the Pitch Factor.
    This ensures the curve is 'based off' the interpolation.
    """
    
    def power_fn(v):
        # Operational limits
        if v < CUT_IN_SPEED or v > CUT_OUT_SPEED:
            return 0.0

        # 1. Get Base Power from Spline (kW) - Represents Optimal Pitch (0 deg)
        base_power_at_v_kw = base_spline_kw_fn(v)
        base_power_at_v_W = base_power_at_v_kw * 1000.0
        
        # 2. Scale by Area (Blade Length)
        scaled_power_opt_W = base_power_at_v_W * (L / BASE_BLADE_LENGTH)**2
        
        # 3. Apply Pitch Derating
        # We apply the factor that reduces Cp based on pitch angle.
        # Power_Actual = Power_Optimal * Pitch_Factor
        pitch_factor = compute_cp_pitch_factor(beta_deg)
        
        p_elec = scaled_power_opt_W * pitch_factor
        
        return min(max(0.0, p_elec), GENERATOR_LIMIT_W)

    return power_fn


# ---------------------------------------------
# SECTION 5 — AEP CALCULATION
# ---------------------------------------------

def calculate_aep(power_curve_fn, wind_dist_hours):
    total = 0.0
    for v in range(1, len(wind_dist_hours)):
        total += float(power_curve_fn(v)) * wind_dist_hours[v] 
    return total / 1e6  # MWh


# ---------------------------------------------
# SECTION 6 — MAIN ANALYSIS + REQUESTED PLOTS
# ---------------------------------------------

if __name__ == "__main__":

    # -------- Weibull distribution --------
    wind_dist, v_wind = get_wind_distribution(WEIBULL_A, WEIBULL_K)
    
    # Setup for plots
    representative_L = 80.0
    v_plot_detailed = np.linspace(0, 30, 500) # Higher res for smooth spline visualization

    # ------------------------------------
    # PLOT 1: SPLINE FIT JUSTIFICATION
    # ------------------------------------
    # Visualizing how the spline fits the raw data
    v_smooth_fit = np.linspace(0, 25, 500)
    p_smooth_fit = base_nrel_spline_power_kw(v_smooth_fit) 

    plt.figure(figsize=(10, 6))
    # Plot raw points
    plt.plot(v_data_base_nrel, p_data_kw_base_nrel, "rs", markersize=6, label="Raw Data Points")
    # Plot Spline line
    plt.plot(v_smooth_fit, p_smooth_fit, "b-", lw=2, label="Spline Interpolation (s=0)")
    
    plt.title("Spline Interpolation of NREL 5MW Power Curve", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 2: POWER vs WIND SPEED FOR 0, 5, 10 DEG PITCH
    # ------------------------------------
    
    plt.figure(figsize=(12, 8))
    
    # Pitch angles to request: 0, 5, 10
    pitch_angles = [0, 5, 10]
    colors = ['b', 'g', 'r'] # Blue, Green, Red
    
    for i, beta in enumerate(pitch_angles):
        # Create the power curve function for this specific pitch
        # This now uses the SPLINE curve as the base
        power_fn = make_scaled_power_curve_from_spline(representative_L, base_nrel_spline_power_kw, beta_deg=beta)
        
        # Calculate power across the wind speed range
        p_mw = np.array([power_fn(v) for v in v_plot_detailed]) / 1e6
        
        # Plot
        label_text = f"Pitch = {beta}°"
        if beta == 0: label_text += " (Optimal)"
        
        plt.plot(v_plot_detailed, p_mw, color=colors[i], lw=2, label=label_text)

    # Add Generator Limit line
    plt.axhline(GENERATOR_LIMIT_W / 1e6, color='gray', linestyle=':', label="Generator Limit (5 MW)")

    plt.title(f"Power Curve vs Wind Speed by Pitch Angle (Based on Spline)\n(L={representative_L}m)", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Electrical Power (MW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.xlim(0, 25)
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # AEP CALCULATION LOOP (Verification)
    # ------------------------------------
    blade_lengths_to_test = np.arange(60, 161, 10) 
    print("\nCalculated AEP using Spline Model (Pitch=0°):")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 36)
    
    for L in blade_lengths_to_test:
        power_fn_L = make_scaled_power_curve_from_spline(L, base_nrel_spline_power_kw, beta_deg=0.0)
        aep = float(calculate_aep(power_fn_L, wind_dist))
        print(f" {L:<15.2f} | {aep:,.0f}")