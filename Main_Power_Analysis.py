import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate  # Full interpolate module for splrep/splev

# --------------------------------------------- SECTION 1 — WIND DISTRIBUTION (Weibull) ---------------------------------------------

WEIBULL_A = 10.566
WEIBULL_K = 2.113
MAX_WIND_SPEED = 30
CUT_OUT_SPEED = 30.0
CUT_IN_SPEED = 1.5  # Approx. friction-break threshold

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    """Return hourly wind distribution using a Weibull PDF."""
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)

    # Weibull PDF
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)

    # Normalised probability across integer wind bins
    prob_v = pdf / np.sum(pdf)
    hours_v = prob_v * hours_in_year
    return hours_v, v


# --------------------------------------------- SECTION 2 — SPLINE INTERPOLATION MODEL ---------------------------------------------

# NREL 5MW raw curve data (kW)
BASE_POWER_CURVE_DATA = np.array([
    (0.0, 0), (1.75, 0),
    (3,100),(4,250),(5,475),(6,800),(7,1250),
    (8,1900),(9,2700),(10,3717),(11,4924),
    (11.4,5000),(12,5000),(13,5000),(14,5000),
    (15,5000),(16,5000),(17,5000),(18,5000),
    (19,5000),(20,5000),(21,5000),(22,5000),
    (23,5000),(24,5000),(25,5000),
])

GENERATOR_LIMIT_W = 5_000_000.0
GENERATOR_LIMIT_KW = 5000.0
BASE_BLADE_LENGTH = 61.5

# Extract wind & power arrays for convenience
v_data_base_nrel = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw_base_nrel = BASE_POWER_CURVE_DATA[:, 1]

def get_nrel_base_power_spline_fn():
    """
    Create a spline interpolation of the NREL 5MW curve.
    s=0 forces exact fit through all data points.
    """

    # Spline representation (tck)
    tck = interpolate.splrep(v_data_base_nrel, p_data_kw_base_nrel, s=0)

    def power_fn(v):
        """Evaluate spline at wind speed v (kW). Clipped to physical limits."""
        p_val = interpolate.splev(v, tck, der=0)
        return np.clip(p_val, 0.0, GENERATOR_LIMIT_KW)

    return power_fn

base_nrel_spline_power_kw = get_nrel_base_power_spline_fn()


# --------------------------------------------- SECTION 3 — CP-BASED AERODYNAMIC MODEL ---------------------------------------------

RHO = 1.225
U_RATED = 11.4

# Calibrate Cp so the reference turbine reaches 5MW at rated wind speed
A_base = math.pi * BASE_BLADE_LENGTH**2
CP_CALIBRATED = min(GENERATOR_LIMIT_W / (0.5 * RHO * A_base * U_RATED**3), 0.593)

# Mechanical/electrical efficiencies
GEARBOX_EFF = 0.98
GENERATOR_EFF = 0.96
POWER_EFF = GEARBOX_EFF * GENERATOR_EFF

# Typical TSR & pitch mapping constants
LAMBDA_OPT = 7.5
TSR_SIGMA = 0.5 * LAMBDA_OPT       # Width of TSR efficiency curve
PITCH_SCALE_DEG = 8.0             # Pitch sensitivity factor

def compute_cp(v, L, beta_deg=0.0, rotor_angle_deg=0.0,
               cp_max=CP_CALIBRATED, lambda_opt=LAMBDA_OPT):
    """
    Compute power coefficient Cp using pitch, TSR and rotor misalignment effects.
    Designed to remain simple but physically meaningful.
    """

    # ---- Pitch penalty term (Gaussian decay) ----
    beta = abs(beta_deg)
    cp_pitch = np.exp(-(beta / PITCH_SCALE_DEG)**2)

    # ---- TSR handling (constant below rated, reducing above) ----
    if v <= U_RATED:
        lambda_eff = lambda_opt
    else:
        lambda_eff = lambda_opt * (U_RATED / v)

    # TSR-dependent efficiency (Gaussian peak around optimum TSR)
    cp_tsr = np.exp(-((lambda_eff - lambda_opt) / TSR_SIGMA)**2)

    # ---- Rotor misalignment effect (cos^3 law) ----
    phi_rad = np.radians(rotor_angle_deg)
    cos3 = max(0, np.cos(phi_rad)**3)

    # ---- Combined Cp ----
    Cp = cp_max * cp_pitch * cp_tsr * cos3
    return max(0.0, min(Cp, cp_max))


# ---------------------------------------------------- SECTION 4 — POWER CURVE GENERATION ----------------------------------------------------

def make_scaled_power_curve_from_spline(L, base_spline_kw_fn, beta_deg=0.0, rotor_angle_deg=0.0):
    """
    Build a full power curve function combining:
      1) Spline-based NREL curve
      2) Area scaling from blade length
      3) Cp-based aerodynamic pitch losses
    """

    A = math.pi * L**2

    def power_fn(v):
        # Cut-in & cut-out behaviour
        if v < CUT_IN_SPEED or v > CUT_OUT_SPEED:
            return 0.0

        # Base spline power (kW → W)
        base_power_W = base_spline_kw_fn(v) * 1000.0

        # Scale by rotor area ratio
        scaled_power_W = base_power_W * (L / BASE_BLADE_LENGTH)**2

        # Aerodynamic model (Cp)
        Cp = compute_cp(v, L, beta_deg=beta_deg, rotor_angle_deg=rotor_angle_deg)
        p_aero = 0.5 * RHO * A * v**3 * Cp
        p_elec = p_aero * POWER_EFF

        # Limit to generator rating
        return min(max(0.0, p_elec), GENERATOR_LIMIT_W)

    return power_fn


# --------------------------------------------- SECTION 5 — AEP CALCULATION ---------------------------------------------

def calculate_aep(power_curve_fn, wind_dist_hours):
    """Integrate power curve over yearly wind-speed probability (MWh)."""
    total = 0.0
    for v in range(1, len(wind_dist_hours)):
        total += float(power_curve_fn(v)) * wind_dist_hours[v]
    return total / 1e6  # Convert Wh → MWh


# --------------------------------------------- SECTION 6 — MAIN ANALYSIS + PLOTS ---------------------------------------------

if __name__ == "__main__":

    # Weibull input distribution
    wind_dist, v_wind = get_wind_distribution(WEIBULL_A, WEIBULL_K)

    representative_L = 72.8
    v_plot_detailed = np.linspace(0, 30, 500)

    # ---------- PLOT 1 — Validate Spline Fit ----------
    v_smooth_fit = np.linspace(0, 30, 500)
    p_smooth_fit = base_nrel_spline_power_kw(v_smooth_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(v_data_base_nrel, p_data_kw_base_nrel, "rs", label="Raw Data")
    plt.plot(v_smooth_fit, p_smooth_fit, "b-", lw=2, label="Spline Fit")
    plt.title("Spline Interpolation of NREL 5MW Power Curve")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.grid(True, linestyle='--')
    plt.xlim(0, 25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- PLOT 2 — Power Curve vs Pitch ----------
    plt.figure(figsize=(12, 8))
    pitch_angles = [0, 5, 7.5, 10]
    colors = ['c', 'g', 'r', 'm']

    for i, beta in enumerate(pitch_angles):
        power_fn = make_scaled_power_curve_from_spline(
            representative_L, base_nrel_spline_power_kw, beta_deg=beta
        )
        p_mw = np.array([power_fn(v) for v in v_plot_detailed]) / 1e6
        plt.plot(v_plot_detailed, p_mw, color=colors[i], lw=2,
                 label=f"Pitch = {beta}°" + (" (Optimal)" if beta == 0 else ""))

    plt.axhline(GENERATOR_LIMIT_W/1e6, linestyle=':', color='gray', label="Rated 5MW")
    plt.title("Power Curve vs Wind Speed by Pitch Angle")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (MW)")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- AEP Example ----------
    power_fn_L = make_scaled_power_curve_from_spline(72.8, base_nrel_spline_power_kw, beta_deg=0.0)
    aep = float(calculate_aep(power_fn_L, wind_dist))

    print("\nAnnual Energy Production:", aep, "MWh")
    print("Annual Profit (£):", aep * 0.2635 * 1000)

    blade_lengths_to_test = np.arange(60, 161, 10)
    print("\nAEP vs Blade Length:")
    for L in blade_lengths_to_test:
        power_fn_L = make_scaled_power_curve_from_spline(L, base_nrel_spline_power_kw, beta_deg=0.0)
        aep = float(calculate_aep(power_fn_L, wind_dist))
        print(f"{L:6.1f} m -> {aep:,.0f} MWh")

# ---------- PLOT 3 — Power vs Pitch at Rated Speed ----------
pitch_range = np.linspace(0, 20, 200)
v_test = 12
L = 72.8

power_values = []
for beta in pitch_range:
    P = make_scaled_power_curve_from_spline(
        L, base_nrel_spline_power_kw, beta_deg=beta
    )(v_test)
    power_values.append(P / 1e6)

plt.figure(figsize=(10, 7))
plt.plot(pitch_range, power_values, lw=2)
plt.xlabel("Pitch Angle (°)")
plt.ylabel("Power (MW)")
plt.title(f"Power vs Pitch at {v_test} m/s")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# ---------- PLOT — Weibull Distribution ----------
plt.figure(figsize=(10, 6))
plt.bar(v_wind, wind_dist, width=0.8, color='skyblue', edgecolor='black')
plt.title("Weibull Wind Speed Distribution")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Hours per Year")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------- PLOT — AEP vs Blade Length ----------
aep_values = []
for L in blade_lengths_to_test:
    aep_values.append(calculate_aep(
        make_scaled_power_curve_from_spline(L, base_nrel_spline_power_kw),
        wind_dist
    ))

plt.figure(figsize=(10, 6))
plt.plot(blade_lengths_to_test, aep_values, 'bo-', lw=2)
plt.title("AEP vs Blade Length")
plt.xlabel("Blade Length (m)")
plt.ylabel("AEP (MWh)")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()
