# Spline-interpolation-based power curves for different pitch angles
# Paste into VS Code / Jupyter and run.
# Requires: numpy, matplotlib. SciPy optional (better spline).
# If SciPy is not installed the code uses a smooth Hermite-like piecewise cubic fallback.

import numpy as np
import matplotlib.pyplot as plt
import math

# Try SciPy spline; if not available, we'll use a safe fallback
use_scipy = True
try:
    from scipy import interpolate
except Exception:
    use_scipy = False
    print("scipy.interpolate not found — using a smooth piecewise cubic fallback (no external deps).")

# ------------------------------
# Base NREL-like power curve data
# ------------------------------
BASE_POWER_CURVE_DATA = np.array([
    (0.0, 0),
    (2, 0),
    (3, 100), (4, 250), (5, 475), (6, 800), (7, 1250),
    (8, 1900), (9, 2700), (10, 3717), (11, 4924),
    (11.4, 5000), (12, 5000), (13, 5000), (14, 5000),
    (15, 5000), (16, 5000), (17, 5000), (18, 5000),
    (19, 5000), (20, 5000), (21, 5000), (22, 5000),
    (23, 5000), (24, 5000), (25, 5000),
    (25.1, 0),
    (30.0, 0)
])
v_data = BASE_POWER_CURVE_DATA[:,0]
p_data_kw = BASE_POWER_CURVE_DATA[:,1]

# plotting grid
v_plot = np.linspace(0, 25, 500)

# ------------------------------
# Spline (or fallback) interpolant
# ------------------------------
if use_scipy:
    # cubic interpolating spline
    tck = interpolate.splrep(v_data, p_data_kw, s=0, k=3)
    p_spline_kw = interpolate.splev(v_plot, tck, der=0)
else:
    # Piecewise cubic Hermite-like interpolant (C1 continuous)
    # Implementation: compute node slopes (monotone-friendly) and evaluate Hermite basis
    def piecewise_cubic(x, y, x_eval):
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        h = np.diff(x)
        delta = np.diff(y) / h
        m = np.zeros(n)
        m[0] = delta[0]
        m[-1] = delta[-1]
        for i in range(1, n-1):
            if delta[i-1] * delta[i] <= 0:
                m[i] = 0.0
            else:
                m[i] = (delta[i-1] + delta[i]) / 2.0
        x_eval = np.asarray(x_eval)
        y_eval = np.zeros_like(x_eval, dtype=float)
        for idx, xv in enumerate(x_eval):
            if xv <= x[0]:
                y_eval[idx] = y[0]; continue
            if xv >= x[-1]:
                y_eval[idx] = y[-1]; continue
            i = np.searchsorted(x, xv) - 1
            if i < 0: i = 0
            if i >= n-1: i = n-2
            h_i = x[i+1] - x[i]
            t = (xv - x[i]) / h_i
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            yv = h00*y[i] + h10*h_i*m[i] + h01*y[i+1] + h11*h_i*m[i+1]
            y_eval[idx] = yv
        return y_eval

    p_spline_kw = piecewise_cubic(v_data, p_data_kw, v_plot)

# ensure physical bounds (kW)
p_spline_kw = np.clip(p_spline_kw, 0.0, 5000.0)

# ------------------------------
# Pitch penalty + scaling + efficiencies
# ------------------------------
BASE_BLADE_LENGTH = 61.5  # reference blade length for the NREL power curve
GENERATOR_LIMIT_W = 5_000_000.0
GEARBOX_EFF = 0.98
GENERATOR_EFF = 0.96
POWER_EFF = GEARBOX_EFF * GENERATOR_EFF
PITCH_SCALE_DEG = 8.0
CUT_IN = 3.0
CUT_OUT = 25.0

def pitch_loss_factor(beta_deg):
    """Gaussian-like pitch penalty (1 at 0°, decreasing with beta)."""
    return math.exp(- (abs(beta_deg) / PITCH_SCALE_DEG)**2)

def make_spline_power_curve_with_pitch(L, beta_deg=0.0):
    scale = (L / BASE_BLADE_LENGTH)**2
    pitch_factor = pitch_loss_factor(beta_deg)
    def power_fn(v):
        # sample kW from interpolant (v_plot/p_spline_kw)
        pk_kW = float(np.interp(v, v_plot, p_spline_kw))
        p_w = pk_kW * 1000.0 * scale * pitch_factor * POWER_EFF
        if v < CUT_IN or v > CUT_OUT:
            return 0.0
        return float(min(max(p_w, 0.0), GENERATOR_LIMIT_W))
    return power_fn

# ------------------------------
# Create power curves for multiple pitch angles
# ------------------------------
L = 80.0  # blade length to test
pitch_angles = [0.0, 5.0, 10.0, 15.0]
curves_MW = {}
for beta in pitch_angles:
    fn = make_spline_power_curve_with_pitch(L, beta)
    curves_MW[beta] = np.array([fn(v) for v in v_plot]) / 1e6  # convert to MW

# ------------------------------
# Plot: power vs wind speed for different pitch angles
# ------------------------------
plt.figure(figsize=(10,6))
for beta in pitch_angles:
    plt.plot(v_plot, curves_MW[beta], label=f'Pitch {beta:.0f}°', linewidth=2)
plt.axhline(5.0, color='gray', linestyle=':', label='Generator limit (5 MW)')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Electrical power (MW)')
plt.title(f'Spline-based Power Curves (L = {L} m) — different pitch angles')
plt.xlim(0,25)
plt.ylim(0,5.5)
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# Also show original points + interpolant (kW)
# ------------------------------
plt.figure(figsize=(10,6))
plt.plot(v_data, p_data_kw, 'ro', label='Original NREL data (kW)')
plt.plot(v_plot, p_spline_kw, 'b-', lw=2, label=('Spline' if use_scipy else 'Cubic-Hermite-like'))
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (kW)')
plt.title('Interpolant vs original NREL points (kW)')
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# Compute & print AEP for each pitch (using Weibull hours)
# ------------------------------
def get_weibull_hours(A=10.566, k=2.113, max_v=25):
    hours_in_year = 365*24
    v = np.arange(0, max_v + 1)
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob = pdf / np.sum(pdf)
    return prob * hours_in_year, v

wind_hours, wind_bins = get_weibull_hours()

for beta in pitch_angles:
    fn = make_spline_power_curve_with_pitch(L, beta)
    aep_wh = sum(fn(v) * wind_hours[v] for v in range(1, len(wind_hours)))
    print(f"AEP (L={L} m, pitch={beta:.0f}°) = {aep_wh/1e6:,.0f} MWh/yr")
