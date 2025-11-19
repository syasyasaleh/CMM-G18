# Wind speed vs Power regression for Siemens SWT-6.0-154 (Hywind-style turbine)
# Models the relationship between wind speed (m/s) and power (MW)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Manufacturer-like parameters
P_rated = 6.0     # MW
v_cut_in = 4.0    # m/s
v_rated = 13.0    # m/s
v_cut_out = 25.0  # m/s

# Define realistic turbine power curve 
def power_curve(v):
    v = np.array(v)
    P = np.zeros_like(v)
    # Rising cubic region between cut-in and rated
    mask_rise = (v >= v_cut_in) & (v < v_rated)
    P[mask_rise] = P_rated * ((v[mask_rise] - v_cut_in) / (v_rated - v_cut_in))**3
    # Rated region (constant power)
    mask_plateau = (v >= v_rated) & (v <= v_cut_out)
    P[mask_plateau] = P_rated
    # Zero below cut-in and above cut-out
    return P

# Generate synthetic data
v_data = np.linspace(0, 30, 50)
P_data = power_curve(v_data)

# Polynomial regression - 3rd degree
coeffs = np.polyfit(v_data, P_data, 3)
poly_fit = np.polyval(coeffs, v_data)

# Spline interpolation
spline_fit = PchipInterpolator(v_data, P_data)
P_spline = spline_fit(v_data)

# Print regression equation
print("Cubic Polynomial Regression Model:")
print(f"P(v) = {coeffs[0]:.4e}v³ + {coeffs[1]:.4e}v² + {coeffs[2]:.4e}v + {coeffs[3]:.4e}")

# Plot results 
plt.figure(figsize=(8,5))
plt.plot(v_data, P_data, 'o', label='Sample Data (Wind Speed vs Power)')
plt.plot(v_data, poly_fit, '--', label='Cubic Polynomial Fit')
plt.plot(v_data, P_spline, '-', label='Spline Fit')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (MW)')
plt.title('Wind Speed vs Power Generation (Siemens SWT-6.0-154)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

