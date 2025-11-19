# Blade Length vs Power Regression
# ---------------------------------
# Models how turbine power generation varies with blade length
# assuming constant wind speed and air density.

import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
rho = 1.225        # air density (kg/m^3)
Cp = 0.42          # power coefficient
v = 12             # constant wind speed (m/s)

# --- Blade length data (m) ---
blade_lengths = np.array([40, 50, 60, 70, 80, 90, 100])

# --- Power calculation ---
# Power = 0.5 * rho * Cp * A * v^3, where A = π * R^2
powers = 0.5 * rho * Cp * np.pi * blade_lengths**2 * v**3  # in watts
powers_MW = powers / 1e6  # convert to MW

# --- Regression: fit polynomial to relationship ---
coeffs = np.polyfit(blade_lengths, powers_MW, 2)  # quadratic (expected)
poly_fit = np.polyval(coeffs, blade_lengths)

# --- Print regression model ---
print("Quadratic Regression Model for Power vs Blade Length:")
print(f"P(L) = {coeffs[0]:.4e}L² + {coeffs[1]:.4e}L + {coeffs[2]:.4e}")

# --- Plot results ---
plt.figure(figsize=(8,5))
plt.scatter(blade_lengths, powers_MW, color='red', label='Data (calculated)')
plt.plot(blade_lengths, poly_fit, '--b', label='Quadratic Regression Fit')
plt.xlabel('Blade Length (m)')
plt.ylabel('Power Output (MW)')
plt.title('Blade Length vs Power Generation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
