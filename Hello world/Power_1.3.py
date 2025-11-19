"""
Improved Cp model + monotonic interpolation (PCHIP fallback) + plotting.

Features:
- Uses a monotonic interpolant (PCHIP if available) on the NREL NREL 5MW power points
  to avoid polynomial oscillations (removes dips around 20 m/s).
- Implements an advanced Cp model:
    Cp(lambda, beta) = cp_max * gaussian(lambda; lambda_opt, sigma) * pitch_factor(beta) * loss_factors
  where cp_max is calibrated so Cp*area*0.5*rho*v^3 hits 5MW at rated for beta=0.
- Applies drivetrain + miscellaneous efficiency.
- Caps aerodynamic power at generator (5MW) and enforces cut-in/cut-out.
- Plots raw data, the monotonic interpolant ("regression"), Cp-based curves for beta=0° and beta=10°,
  and ensures the 10° curve reaches 5MW but at higher wind speed than 0°.
"""
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Basic turbine/site params
# ------------------------
CUT_IN = 3.0           # m/s
RATED = 11.4           # m/s
CUT_OUT = 25.0         # m/s
GEN_LIMIT_W = 5_000_000.0  # W (5 MW)
RHO = 1.225            # kg/m^3 (air density)
BASE_BLADE_LENGTH = 61.5  # m (NREL rotor radius used as baseline)
A = np.pi * BASE_BLADE_LENGTH**2

# ------------------------
# NREL data (v [m/s], P [kW]) - same points you've used
# ------------------------
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
v_raw = BASE_POWER_CURVE_DATA[:, 0]
p_raw_kw = BASE_POWER_CURVE_DATA[:, 1]

# ------------------------
# Build monotonic interpolation of the raw data (safe "regression")
# Prefer PCHIP for smooth monotonic behaviour; fall back to linear interpolation
# ------------------------
use_pchip = False
interp_func = None
try:
    from scipy.interpolate import PchipInterpolator
    interp_func = PchipInterpolator(v_raw, p_raw_kw, extrapolate=False)
    use_pchip = True
    # will return kW
except Exception:
    # fallback: linear interpolation (monotonic between points)
    def interp_linear(vq):
        vq_arr = np.array(vq)
        # np.interp returns nan for out-of-bounds if left/right not provided -> we want 0 outside cut range
        vals = np.interp(vq_arr, v_raw, p_raw_kw, left=0.0, right=0.0)
        return vals
    interp_func = interp_linear
    use_pchip = False

# Helper: regression-style power function that returns Watts (monotonic)
def regression_power_w(v):
    """
    v: scalar or array
    returns: power in W (floats) from monotonic interpolant of the raw NREL data,
             clipped to [0, GEN_LIMIT] and zero outside cut-in/cut-out.
    """
    v_arr = np.array(v, copy=False)
    # evaluate interpolant (kW)
    if use_pchip:
        p_kw = interp_func(v_arr)
        # Pchip returns nan outside domain; set to 0 there
        p_kw = np.where(np.isnan(p_kw), 0.0, p_kw)
    else:
        p_kw = interp_func(v_arr)
    # clip negatives
    p_kw = np.where(p_kw < 0.0, 0.0, p_kw)
    # clip outside range to zero
    p_kw = np.where((v_arr < CUT_IN) | (v_arr > CUT_OUT), 0.0, p_kw)
    # convert to W
    p_w = p_kw * 1000.0
    p_w = np.where(p_w > GEN_LIMIT_W, GEN_LIMIT_W, p_w)
    return p_w

# ------------------------
# Advanced Cp model
# - Gaussian-like Cp dependence on TSR (smooth peaked curve)
# - Pitch reduces maximum Cp via a non-linear pitch factor (stronger than linear)
# - Global loss factors: tip/root/park losses / drivetrain efficiency
# - cp_max will be calibrated so beta=0° at rated gives generator-rated power (5 MW)
# ------------------------
# physics parameters for Cp shape (user-tunable)
lambda_opt = 8.0       # typical optimal tip-speed ratio for large blades
sigma_lambda = 3.0     # controls width of TSR peak (larger => broader)
pitch_k = 0.035        # pitch sensitivity constant (higher => stronger reduction with pitch)
drivetrain_eff = 0.95  # mechanical + electrical efficiency
aero_misc_loss = 0.98  # other aerodynamic multiplier (tip/root/wake losses)

def pitch_factor(beta_deg):
    """
    Non-linear pitch factor that reduces Cp as pitch increases.
    Uses a saturating function so large pitch angles never give negative Cp.
    """
    # Example: factor = 1 / (1 + k * beta^1.2)  (saturating)
    return 1.0 / (1.0 + pitch_k * (abs(beta_deg) ** 1.2))

def cp_profile(lambda_tsr, beta_deg, cp_max):
    """
    cp_profile: returns Cp (unitless) = cp_max * gaussian(lambda) * pitch_factor * aero_misc
    gaussian(lambda) = exp( -0.5 * ((lambda - lambda_opt) / sigma_lambda)^2 )
    multiplied by drivetrain/aero efficiency separately in final power computation.
    """
    tsr_term = np.exp(-0.5 * ((lambda_tsr - lambda_opt) / sigma_lambda)**2)
    pf = pitch_factor(beta_deg)
    cp = cp_max * tsr_term * pf * aero_misc_loss
    # enforce non-negative
    return max(0.0, cp)

# We'll calibrate cp_max so that for beta=0 and for the control policy below, P(rated) ≈ GEN_LIMIT_W
# Control policy for TSR/rotor speed:
# - below rated: assume variable-speed control holds lambda≈lambda_opt (so cp near cp_max),
# - above rated: rotor speed limited so lambda falls proportional to 1/v (lambda = lambda_opt * RATED / v).
def compute_cp_max_for_rating():
    """
    Determine cp_max such that for beta=0 and v=RATED the aerodynamic power
    (including drivetrain_eff) equals the generator limit (approx).
    Because below rated we assume lambda ~ lambda_opt,
    at rated speed if variable speed keeps lambda_opt, Cp at rated equals cp_max * tsr_term(=1).
    So P = 0.5 * rho * A * v^3 * cp_max * drivetrain_eff
    => cp_max = GEN_LIMIT_W / (0.5 * rho * A * RATED^3 * drivetrain_eff)
    """
    denom = 0.5 * RHO * A * (RATED ** 3) * drivetrain_eff
    cp_needed = GEN_LIMIT_W / denom
    # Ensure cp_needed not above Betz (0.593). If it is, clip and accept slight mismatch.
    cp_needed = min(cp_needed, 0.59)
    return cp_needed

cp_max_calibrated = compute_cp_max_for_rating()

# ------------------------
# Power from Cp model (returns W)
# ------------------------
def power_from_cp_model(v, beta_deg):
    """
    Computes aerodynamic power (W) for wind speed v and pitch beta_deg using the cp_profile.
    Uses the control policy: lambda= lambda_opt for v<=RATED, lambda decreases for v>RATED.
    Caps at generator limit and enforces cut-in/out.
    """
    if v < CUT_IN or v > CUT_OUT:
        return 0.0
    # tip-speed ratio:
    if v <= RATED:
        lambda_tsr = lambda_opt
    else:
        lambda_tsr = lambda_opt * (RATED / v)
    cp = cp_profile(lambda_tsr, beta_deg, cp_max_calibrated)
    P_aero = 0.5 * RHO * A * v**3 * cp
    # apply drivetrain / electrical efficiency
    P_elec = P_aero * drivetrain_eff
    # cap at generator limit
    if P_elec > GEN_LIMIT_W:
        P_elec = GEN_LIMIT_W
    return P_elec

# ------------------------
# Plot: monotonic interpolant (regression) and Cp curves for 0° and 10°.
# Also show that no dip appears and the 10° reaches 5MW later.
# ------------------------
if __name__ == "__main__":
    # wind axis
    v_plot = np.linspace(0.0, 25.0, 400)

    # regression curve (monotonic interpolant) in MW
    P_reg_w = regression_power_w(v_plot)
    P_reg_mw = P_reg_w / 1e6

    # Cp-based curves
    P_cp_0 = np.array([power_from_cp_model(v, 0.0) for v in v_plot]) / 1e6
    P_cp_10 = np.array([power_from_cp_model(v, 10.0) for v in v_plot]) / 1e6

    # Raw data points in MW
    p_raw_mw = p_raw_kw / 1000.0

    plt.figure(figsize=(12, 8))
    plt.scatter(v_raw, p_raw_mw, color='tab:red', label='NREL raw data (points)', zorder=6)
    plt.plot(v_plot, P_reg_mw, color='tab:blue', lw=2.2, label='Monotonic interpolant (safe regression)')
    plt.plot(v_plot, P_cp_0, color='tab:green', lw=2.0, label='Cp model (Pitch = 0°)')
    plt.plot(v_plot, P_cp_10, color='tab:orange', lw=2.0, linestyle='--', label='Cp model (Pitch = 10°)')

    plt.xlabel('Wind speed (m/s)', fontsize=13)
    plt.ylabel('Power (MW)', fontsize=13)
    plt.title('Improved Cp Model + Monotonic Interpolant\n(no polynomial dip at 20 m/s; 10° reaches 5 MW later)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 25)
    plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("improved_cp_monotonic_plot.png")
    plt.show()

    # Diagnostics: show wind speed where curves hit 5 MW
    def speed_to_reach_5mw(p_array, v_array):
        # find first v where p >= 5 MW
        idx = np.where(p_array >= 5.0)[0]
        return float(v_array[idx[0]]) if idx.size > 0 else None

    v_0_hit = speed_to_reach_5mw(P_cp_0, v_plot)
    v_10_hit = speed_to_reach_5mw(P_cp_10, v_plot)
    print(f"cp_max calibrated = {cp_max_calibrated:.4f}")
    print(f"Pitch 0° reaches 5 MW at v = {v_0_hit} m/s (should be ~{RATED} m/s).")
    print(f"Pitch10° reaches 5 MW at v = {v_10_hit} m/s (should be > {RATED} m/s).")

    # Print sample values around 11-20 m/s to show no dip
    print("\nSample outputs (v, regression_MW, Cp0_MW, Cp10_MW):")
    for v_sample in [10.0, 11.4, 14.0, 16.0, 18.0, 20.0]:
        print(f" v={v_sample:5.1f} : reg={regression_power_w(v_sample)/1e6:6.3f} MW,"
              f" cp0={power_from_cp_model(v_sample,0)/1e6:6.3f} MW,"
              f" cp10={power_from_cp_model(v_sample,10)/1e6:6.3f} MW")
