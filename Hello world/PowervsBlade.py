import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Blade lengths
L = np.linspace(20, 150, 20)

# "True" model: aerodynamic power minus drag/weight penalty
# k1 controls scale, k2 controls penalty for weight, drag and limits
def realistic_power(L, k1, k2):
    return k1 * L**2 * np.exp(-k2 * L)

# Generate synthetic data
true_k1, true_k2 = 0.015, 0.02
P_true = realistic_power(L, true_k1, true_k2)

# Add small noise to simulate measurement variation
rng = np.random.default_rng(42)
P_measured = P_true * (1 + 0.05 * rng.standard_normal(len(L)))

# Regression fit to measured data
popt, _ = curve_fit(realistic_power, L, P_measured, p0=[0.01, 0.01])
P_fit = realistic_power(L, *popt)

# Find optimal (maximum) blade length analytically from fitted curve
L_opt = 2 / popt[1]  # derivative of L^2 * e^{-k2 L} → 2 - k2 L = 0

# Plot
plt.figure(figsize=(8,5))
plt.scatter(L, P_measured, color='r', label='Synthetic data')
plt.plot(L, P_fit, 'b-', label='Fitted bell-shaped regression')
plt.axvline(L_opt, color='g', linestyle='--', label=f'Optimal blade length ≈ {L_opt:.1f} m')
plt.xlabel('Blade Length (m)')
plt.ylabel('Power Output (MW, relative)')
plt.title('Realistic Blade Length vs Power Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Fitted model: P(L) = {popt[0]:.4f} * L^2 * exp(-{popt[1]:.4f} * L)")
print(f"Optimal blade length for max power ≈ {L_opt:.1f} m")