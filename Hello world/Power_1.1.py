import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# SECTION 1 — WIND SPEED DISTRIBUTION (Weibull)
# ------------------------------------------------------

WEIBULL_A = 10.566
WEIBULL_K = 2.113
MAX_WIND_SPEED = 25

def get_wind_distribution(A, k, max_v=MAX_WIND_SPEED):
    hours_in_year = 365 * 24
    v = np.arange(0, max_v + 1)
    pdf = (k / A) * (v / A)**(k - 1) * np.exp(-(v / A)**k)
    prob_v = pdf / np.sum(pdf)
    hours_v = prob_v * hours_in_year
    return hours_v, v, pdf


# ------------------------------------------------------
# SECTION 2 — ORIGINAL NREL 5MW DATA + REGRESSION MODEL
# ------------------------------------------------------

BASE_POWER_CURVE_DATA = np.array([
    (3, 100), (4, 250), (5, 475), (6, 800), (7, 1250),
    (8, 1900), (9, 2700), (10, 3717), (11, 4924),
    (11.4, 5000), (12, 5000), (13, 5000), (14, 5000),
    (15, 5000), (16, 5000), (17, 5000), (18, 5000),
    (19, 5000), (20, 5000), (21, 5000), (22, 5000),
    (23, 5000), (24, 5000), (25, 5000)
])

GENERATOR_LIMIT_W = 5_000_000
BASE_BLADE_LENGTH = 61.5  # meters

# Regression fit
v_data = BASE_POWER_CURVE_DATA[:, 0]
p_data_kw = BASE_POWER_CURVE_DATA[:, 1]
poly_coeff = np.polyfit(v_data, p_data_kw, 6)
base_power_fn = np.poly1d(poly_coeff)


# ------------------------------------------------------
# SECTION 3 — SCALING THE POWER CURVE WITH BLADE LENGTH
# (Your previous method)
# ------------------------------------------------------

def make_scaled_power_curve(L):
    scale_factor = (L / BASE_BLADE_LENGTH) ** 2  

    def power_fn(v):
        if v < 3 or v > 25:
            return 0.0
        P_kW = base_power_fn(v) * scale_factor
        P_kW = min(P_kW, 5000)  # generator limit
        return P_kW * 1000  # convert kW → W
    return power_fn


def calculate_aep(power_curve_fn, wind_dist_hours):
    total_energy = 0.0
    for v in range(1, len(wind_dist_hours)):
        total_energy += power_curve_fn(v) * wind_dist_hours[v]
    return total_energy / 1e6  # MWh


# ------------------------------------------------------
# SECTION 4 — GENERATE THE THREE PLOTS
# ------------------------------------------------------

if __name__ == "__main__":

    # ---- Wind distribution ----
    wind_dist, v_wind, pdf = get_wind_distribution(WEIBULL_A, WEIBULL_K)

    # ---- Plot 1: Wind speed vs hours per year ----
    plt.figure(figsize=(10, 6))
    plt.bar(v_wind, wind_dist, width=0.8)
    plt.title("Wind Speed vs Hours per Year (Weibull Distribution)", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Hours per Year", fontsize=14)
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: Regression justification ----
    v_smooth = np.linspace(0, 25, 300)
    p_smooth = base_power_fn(v_smooth)
    p_smooth[p_smooth < 0] = 0
    p_smooth[p_smooth > 5000] = 5000

    plt.figure(figsize=(10, 6))
    plt.plot(v_data, p_data_kw, "ro", label="NREL 5MW Data")
    plt.plot(v_smooth, p_smooth, "b-", lw=2, label="6th-order Regression Fit")
    plt.title("Regression Justification for NREL 5MW Power Curve", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=14)
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Blade Length vs AEP ----
    blade_lengths = np.linspace(40, 120, 30)
    AEP_values = []

    for L in blade_lengths:
        power_curve = make_scaled_power_curve(L)
        AEP = calculate_aep(power_curve, wind_dist)
        AEP_values.append(AEP)

    plt.figure(figsize=(10, 6))
    plt.plot(blade_lengths, AEP_values, lw=2)
    plt.title("Blade Length vs Annual Energy Production (AEP)", fontsize=16)
    plt.xlabel("Blade Length (m)", fontsize=14)
    plt.ylabel("AEP (MWh/year)", fontsize=14)
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.show()
