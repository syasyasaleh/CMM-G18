"""
blade_length_power_drag_weight.py

Purpose:
 - Compute turbine power vs blade length (physical model, constant wind speed)
 - Compute surrogate aerodynamic thrust from power
 - Compute aerodynamic drag on blades using projected blade area
 - Compute blade weight from mass-per-length
 - Fit quadratic regression for Power vs Blade Length and plot results

Assumptions / Notes (important):
 - Power computed from P = 0.5 * rho * Cp * A * v^3, where A = pi * L^2 (L = blade length / rotor radius)
 - Surrogate thrust T_N approximated as T = c_t * P / v  (P in Watts, v in m/s). This is a dimensional heuristic.
 - Drag computed as D = 0.5 * rho * Cd * A_proj * v^2, where A_proj approximates the projected blade area:
     A_proj = n_blades * L * chord_mean
   (this is a simple projection; true blade drag is more complex)
 - Weight: blade mass = mass_per_length * L per blade, total = n_blades * mass_per_length * L * g
 - All constants (Cp, c_t, Cd, chord, mass_per_length, wind speed) are editable near top of script.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# User-editable parameters
# -------------------------
rho_air = 1.225        # air density (kg/m^3)
Cp = 0.42              # power coefficient (typical)
v_design = 12.0        # design/representative wind speed (m/s)
g = 9.81               # gravity (m/s^2)

# Blade / turbine geometry
n_blades = 3
# We treat 'blade_length' as the radius (R): single blade length in meters (tip to hub distance)
blade_lengths = np.array([30, 40, 50, 60, 70, 80, 90, 100], dtype=float)  # m

# Drag / structural assumptions
chord_mean = 3.0       # mean chord of blade (m) — adjust if you have better data
Cd_blade = 1.2         # drag coefficient for blade projected area (dimensionless) — shape dependent
mass_per_length = 50.0 # kg/m per blade (linear mass density) — adjust (50 kg/m is a reasonable order-of-magnitude)
                       # e.g., a 50 m blade then mass ≈ 2500 kg

# Thrust surrogate constant (T ≈ c_t * P / v)
# If you have a better Ct-based method, replace this.
c_t = 1.5

# -------------------------
# Physics calculations
# -------------------------
def power_from_length(L, rho=rho_air, Cp=Cp, v=v_design):
    """Return power in Watts for blade length L (m) at wind speed v using swept area scaling."""
    A = np.pi * L**2
    P_W = 0.5 * rho * Cp * A * v**3
    return P_W

def thrust_from_power(P_W, v=v_design, c_t=c_t):
    """Surrogate thrust (N) from power (W) at wind speed v."""
    v_safe = np.maximum(v, 1e-6)
    T_N = c_t * P_W / v_safe
    return T_N

def drag_on_blades(L, rho=rho_air, Cd=Cd_blade, chord=chord_mean, n=n_blades, v=v_design):
    """Aerodynamic drag (N) estimated from projected blade area A_proj = n_blades * L * chord_mean"""
    A_proj = n * L * chord  # m^2
    D_N = 0.5 * rho * Cd * A_proj * v**2
    return D_N

def blade_weight(L, mass_per_length=mass_per_length, n=n_blades, g=g):
    """Total blade weight force (N) = n_blades * (mass_per_length * L) * g"""
    mass_total = n * mass_per_length * L
    W_N = mass_total * g
    return W_N

# Compute arrays
P_W = power_from_length(blade_lengths)         # Watts
P_MW = P_W / 1e6                               # MW for plotting
T_N = thrust_from_power(P_W)                   # thrust in N
D_N = drag_on_blades(blade_lengths)            # drag in N
W_N = blade_weight(blade_lengths)              # weight in N

# Also compute ratios / derived quantities for reporting
thrust_over_drag = T_N / (D_N + 1e-9)
thrust_over_weight = T_N / (W_N + 1e-9)

# -------------------------
# Regression: fit quadratic P(L) in MW
# -------------------------
coeffs = np.polyfit(blade_lengths, P_MW, 2)   # quadratic fit (expected due to L^2)
poly_fit_MW = np.polyval(coeffs, blade_lengths)

# R^2 (simple manual)
y = P_MW
yhat = poly_fit_MW
ss_res = np.sum((y - yhat)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(10, 8))

# Power plot (top)
ax1 = plt.subplot(2,1,1)
ax1.plot(blade_lengths, P_MW, 'o-', label='Power (MW) from physics formula')
ax1.plot(blade_lengths, poly_fit_MW, '--', label=f'Quadratic fit (R²={r2:.4f})')
ax1.set_ylabel('Power (MW)')
ax1.set_title('Blade length vs Power, Thrust, Drag, Weight (at v = {:.1f} m/s)'.format(v_design))
ax1.grid(True)
ax1.legend(loc='upper left')

# Forces plot (bottom)
ax2 = plt.subplot(2,1,2)
ax2.plot(blade_lengths, T_N/1000.0, 'o-', label='Thrust (kN) [surrogate from power]')
ax2.plot(blade_lengths, D_N/1000.0, 's--', label='Drag on blades (kN)')
ax2.plot(blade_lengths, W_N/1000.0, 'x-.', label='Blade weight (kN)')
ax2.set_xlabel('Blade length (m)')
ax2.set_ylabel('Force (kN)')
ax2.grid(True)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

# -------------------------
# Print summary to console
# -------------------------
print("Quadratic regression for P(L) [MW]:")
print(f"  P(L) = {coeffs[0]:.6e} * L^2 + {coeffs[1]:.6e} * L + {coeffs[2]:.6e}")
print(f"  R^2 = {r2:.6f}")
print()
print("Sample table (blade length, power MW, thrust kN, drag kN, weight kN):")
for L, p, t, d, w in zip(blade_lengths, P_MW, T_N/1000.0, D_N/1000.0, W_N/1000.0):
    print(f"L={L:5.1f} m | P={p:7.3f} MW | Thrust={t:8.2f} kN | Drag={d:7.2f} kN | Weight={w:7.2f} kN")

# -------------------------
# Optional: estimate an "optimal" blade length if we try to maximize power
# but constrain on a simple force ratio or a weight limit.
# Example heuristic: choose the largest L within a limit where (Thrust + Drag) <= F_allowable
# (you should replace with your actual constraint, e.g. platform pitch limit resulting from ODE)
# -------------------------
F_allowable_kN = 500.0  # example platform horizontal force capacity (kN) - change to realistic number
total_horizontal_kN = (T_N + D_N)/1000.0
feasible_mask = total_horizontal_kN <= F_allowable_kN
if feasible_mask.any():
    feasible_lengths = blade_lengths[feasible_mask]
    # choose the one with max power among feasible
    best_idx = np.argmax(P_MW[feasible_mask])
    best_L = feasible_lengths[best_idx]
    print()
    print(f"Example feasibility check with F_allowable = {F_allowable_kN} kN:")
    print(f"  Best feasible blade length (from provided grid) = {best_L} m, P = {np.max(P_MW[feasible_mask]):.3f} MW")
else:
    print()
    print(f"No blade length in the sample grid meets the example horizontal force limit {F_allowable_kN} kN.")
    print("Adjust F_allowable or supply a wider grid of blade lengths to search.")

# End of script
