import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Synthetic experimental data
L = np.linspace(20, 150, 20)

def realistic_power(L, k1, k2):
    return k1 * L**2 * np.exp(-k2 * L)

true_k1, true_k2 = 0.015, 0.02
P_true = realistic_power(L, true_k1, true_k2)

rng = np.random.default_rng(42)
P_measured = P_true * (1 + 0.05 * rng.standard_normal(len(L)))

# Fit regression
popt, _ = curve_fit(realistic_power, L, P_measured, p0=[0.01, 0.01])
k1_fit, k2_fit = popt
P_fit = realistic_power(L, k1_fit, k2_fit)

# Optimal blade length (mathematical optimum)
L_opt = 2 / k2_fit
P_opt = realistic_power(L_opt, k1_fit, k2_fit)

# Allowed blade length
max_L_allowed = 83.47   
P_allowed = realistic_power(max_L_allowed, k1_fit, k2_fit)

# Plot with horizontal 
plt.figure(figsize=(8,5))

plt.scatter(L, P_measured, color='r', label='Synthetic measured data')
plt.plot(L, P_fit, 'b-', label='Fitted regression')

# Vertical lines for optimum and allowed
plt.axvline(L_opt, color='g', linestyle='--', label=f'Optimal L ≈ {L_opt:.1f} m')
plt.axvline(max_L_allowed, color='purple', linestyle='--',
            label=f'Allowed L ≈ {max_L_allowed:.1f} m')

# Horizontal line for allowed power
plt.axhline(P_allowed, color='orange', linestyle='--',
            label=f'Power at allowed L ≈ {P_allowed:.3f}')

# Mark the intersection point
plt.plot(max_L_allowed, P_allowed, 'ko', markersize=8)

# Label the point on the graph
plt.text(max_L_allowed + 2, P_allowed,
         f"P_allowed = {P_allowed:.3f}",
         fontsize=10, color='black', va='bottom')

plt.xlabel('Blade Length (m)')
plt.ylabel('Power Output (relative units)')
plt.title('Blade Length vs Power — With Allowed Power Marked')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 6 — Print results
print(f"Fitted model: P(L) = {k1_fit:.4f} * L^2 * exp(-{k2_fit:.4f} * L)")
print(f"Optimal blade length (max power): L_opt = {L_opt:.2f} m,   P_opt = {P_opt:.4f}")
print(f"Your allowed blade length:         L_allowed = {max_L_allowed:.2f} m")
print(f"Predicted power at L_allowed:      P_allowed = {P_allowed:.4f}")
