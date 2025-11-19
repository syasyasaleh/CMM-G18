import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate

# ---------------------------------------------
# SECTION 1 — WIND DISTRIBUTION (Weibull)
# ---------------------------------------------

WEIBULL_A = 10.566
WEIBULL_K = 2.113
MAX_WIND_SPEED = 25
CUT_OUT_SPEED = 25.0
CUT_IN_SPEED = 1.5

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

v_data_base_nrel = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw_base_nrel = BASE_POWER_CURVE_DATA[:, 1]

# --- SPLINE Interpolation Function ---
def get_nrel_base_power_spline_fn():
    """
    Returns a Spline interpolation function for the base NREL 5MW data.
    """
    # Smoothing factor s=20000 allows the curve to be smooth rather than jagged
    tck = interpolate.splrep(v_data_base_nrel, p_data_kw_base_nrel, s=20000)
    
    def power_fn(v):
        p_val = interpolate.splev(v, tck, der=0)
        p_val = np.clip(p_val, 0.0, GENERATOR_LIMIT_KW)
        return p_val
        
    return power_fn

base_nrel_spline_power_kw = get_nrel_base_power_spline_fn()


# ---------------------------------------------
# SECTION 3 — PITCH FACTOR MODEL
# ---------------------------------------------

RHO = 1.225
PITCH_SCALE_DEG = 8.0

def compute_cp_pitch_factor(beta_deg):
    """
    Calculates only the degradation factor due to pitch angle.
    This ensures we preserve the smooth Spline shape and just scale it down.
    """
    beta = abs(beta_deg)
    # Gaussian drop-off for pitch efficiency
    cp_pitch = np.exp(-(beta / PITCH_SCALE_DEG)**2)
    return cp_pitch


# ----------------------------------------------------
# SECTION 4 — POWER CURVE GENERATION
# ----------------------------------------------------

def make_scaled_power_curve_from_spline(L, base_spline_kw_fn, beta_deg=0.0):
    """
    Creates a full power curve function for a given blade length (L) and pitch angle.
    """
    
    def power_fn(v):
        if v < CUT_IN_SPEED or v > CUT_OUT_SPEED:
            return 0.0

        # 1. Get Base Power from Smoothed Spline (kW)
        base_power_at_v_kw = base_spline_kw_fn(v)
        base_power_at_v_W = base_power_at_v_kw * 1000.0
        
        # 2. Scale by Area (Blade Length)
        scaled_power_opt_W = base_power_at_v_W * (L / BASE_BLADE_LENGTH)**2
        
        # 3. Apply Pitch Derating Factor
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
# SECTION 6 — MAIN ANALYSIS + PLOTS
# ---------------------------------------------

if __name__ == "__main__":

    # -------- Weibull distribution --------
    wind_dist, v_wind = get_wind_distribution(WEIBULL_A, WEIBULL_K)
    
    representative_L = 80.0
    v_plot_detailed = np.linspace(0, 30, 500) 

    # ------------------------------------
    # PLOT 1: SPLINE FIT JUSTIFICATION
    # ------------------------------------
    v_smooth_fit = np.linspace(0, 25, 500)
    p_smooth_fit = base_nrel_spline_power_kw(v_smooth_fit) 

    plt.figure(figsize=(10, 6))
    plt.plot(v_data_base_nrel, p_data_kw_base_nrel, "rs", markersize=6, label="Raw Data Points")
    plt.plot(v_smooth_fit, p_smooth_fit, "b-", lw=2, label="Smoothed Spline Interpolation")
    
    plt.title("Spline Interpolation of Power Curve", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 2: POWER vs WIND SPEED FOR PITCH ANGLES
    # ------------------------------------
    plt.figure(figsize=(12, 8))
    
    pitch_angles = [0, 5, 7.5, 10]
    colors = ['y', 'g', 'r', 'b']
    
    for i, beta in enumerate(pitch_angles):
        # Using the spline-based function ensures the curves are smooth
        power_fn = make_scaled_power_curve_from_spline(representative_L, base_nrel_spline_power_kw, beta_deg=beta)
        p_mw = np.array([power_fn(v) for v in v_plot_detailed]) / 1e6
        
        label_text = f"Pitch = {beta}°"
        if beta == 0: label_text += " (Optimal)"
        
        plt.plot(v_plot_detailed, p_mw, color=colors[i], lw=2, label=label_text)

    plt.axhline(GENERATOR_LIMIT_W / 1e6, color='gray', linestyle=':', label="Generator Limit (5 MW)")

    plt.title(f"Smoothed Power Curves by Pitch Angle (L={representative_L}m)", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Electrical Power (MW)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.xlim(0, 25)
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 3: WEIBULL DISTRIBUTION
    # ------------------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(v_wind, wind_dist, color='skyblue', edgecolor='black', width=0.8)
    plt.title(f"Weibull Wind Distribution (A={WEIBULL_A}, k={WEIBULL_K})", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Hours per Year", fontsize=14)
    plt.grid(True, linestyle='--', axis='y')
    plt.xlim(0, 25)
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # PLOT 4: AEP vs BLADE LENGTH
    # ------------------------------------
    blade_lengths_to_test = np.arange(60, 161, 5) 
    aep_results = []
    
    print("\nCalculated AEP using Smoothed Spline Model (Pitch=0°):")
    print("Blade Length (m) | AEP (MWh/yr)")
    print("-" * 36)
    
    for L in blade_lengths_to_test:
        power_fn_L = make_scaled_power_curve_from_spline(L, base_nrel_spline_power_kw, beta_deg=0.0)
        aep = float(calculate_aep(power_fn_L, wind_dist))
        aep_results.append((L, aep))
        print(f" {L:<15.2f} | {aep:,.0f}")

    # Unpack results for plotting
    L_vals = [x[0] for x in aep_results]
    AEP_vals = [x[1] for x in aep_results]

    plt.figure(figsize=(10, 6))
    plt.plot(L_vals, AEP_vals, 'bo-', linewidth=2, markersize=6)
    plt.title("Annual Energy Production (AEP) vs. Blade Length", fontsize=16)
    plt.xlabel("Blade Length (m)", fontsize=14)
    plt.ylabel("AEP (MWh/yr)", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()