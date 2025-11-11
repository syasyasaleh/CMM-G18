"""
Compute the tower mass moment of inertia about the tower centroid using Table 2-1 data.
Assumptions made:
- Use the elevation nodes from Table 2-1 and interpret the TMassDen as kg/m.
- Interpret the TwFAIner as a sectional mass-moment-of-inertia per unit length (kg·m).
- Use the midpoint/trapezoid averaging for the segment values.
- Shift the segment inertias to the tower CM via the parallel-axis theorem.

Output: I_CM (kg·m^2). Also prints I about tower base (z=10.0 m) and tower top (z=87.6 m)
so we can see how realistic this value is conceptually.
"""

import numpy as np

# Table 2-1 data (nodes). Elevations [m], mass density TMassDen [kg/m], TwFAIner [kg·m]
# (values copied from the OC3 table 2-1. Distributed Tower Properties)
z = np.array([10.00, 17.76, 25.52, 33.28, 41.04, 48.80, 56.56, 64.32, 72.08, 79.84, 87.60])
tmassden = np.array([4667.00, 4345.28, 4034.76, 3735.44, 3447.32, 3170.40,
                     2904.69, 2650.18, 2406.88, 2174.77, 1953.87])
twfainer = np.array([24443.7, 20952.2, 17847.0, 15098.5, 12678.6, 10560.1,
                     8717.2, 7124.9, 5759.8, 4599.3, 3622.1])

# Known tower properties
z_cm = 43.4 # tower centroid [m]
m_total = 249718 # total tower mass [kg]

# Prepare segments
dz = np.diff(z) # segment lengths, length = len(z)-1
z_mid = 0.5*(z[:-1] + z[1:]) # segment centroid elevations
lambda_seg = 0.5*(tmassden[:-1] + tmassden[1:]) # avg mass-per-length per segment
m_seg = lambda_seg * dz # segment masses

# Local segment inertia: interpret TwFAIner as per-unit-length sectional inertia
twfainer_seg = 0.5*(twfainer[:-1] + twfainer[1:]) # avg per-unit-length TwFAIner
I_local_seg = twfainer_seg * dz # local inertia of each segment about its own centroid axis

# Parallel-axis shift to tower CM and sum
d = z_mid - z_cm
I_shifted = I_local_seg + m_seg * d**2
I_CM = I_shifted.sum()

# Sanity check: sum the mass from segments vs reported total (check its 249 718 kg)
mass_from_segments = m_seg.sum()

# Inertia about base (z = 10.0) and top (z = 87.6)
def shift_inertia(I_cm, M, z_from, z_to):
    d = z_to - z_from
    return I_cm + M * d**2

I_about_base = shift_inertia(I_CM, m_total, z_cm, 10.0)
I_about_top = shift_inertia(I_CM, m_total, z_cm, 87.6)

# Results
print(f"I_CM (about tower centroid) = {I_CM:0.6e} kg·m^2")
print(f"Mass from segments = {mass_from_segments} kg (reported total = {m_total} kg)")
print(f"I_about_base (z=10.0 m) = {I_about_base:0.6e} kg·m^2")
print(f"I_about_top  (z=87.6 m) = {I_about_top:0.6e} kg·m^2")

