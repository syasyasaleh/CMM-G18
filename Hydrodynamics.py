"""
Hydrodynamics Module
Original: Syasya
Refactored by: Matthew
Using values from (RAW)
Re-writing the code so its easier to integrate in main code
"""

import numpy as np

# 1. Blade Lengths
BLADE_LENGTH_LOOKUP = np.array([60, 70, 80, 90, 100, 110, 120])

# 2. Hydrostatic Stiffness
# Taken from Syasya's 'K and Ch wind turbine code'
KH_LOOKUP = np.array([
    2.71870041e+09, # K_h for 60m
    2.70999437e+09, # K_h for 70m
    2.69948682e+09, # K_h for 80m
    2.68710299e+09, # K_h for 90m
    2.67277562e+09, # K_h for 100m
    2.65644390e+09 # K_h for 110m
    2.63594952e+09 # K_h for 120m
])

# 3. Mooring Stiffness
K_M_Factor = 0.20 # Mooring stiffness is 20% of hydrostatic 

# 4. Damping Ratio
ZETA = 0.20 # Assuming a reasonable damping ratio 

# Functions that are easy to import into our main code for stability

def calculate_stiffness(blade_length):
    """
    Calculates total restoring stiffness (K_total) for a
    given blade length.
    """
    
    # 1. Use np.interp to find the K_h for the exact blade_length
    K_h = np.interp(blade_length, BLADE_LENGTH_LOOKUP, KH_LOOKUP)
    
    # 2. Calculate k_m based on k_h
    K_m = K_h * K_M_Factor
    
    # 3. Add the constant mooring stiffness
    K_total = K_h + K_m
    
    return K_total

def calculate_damping(K_total, I_total):
    """
    Calculates total hydrodynamic damping (C_h) from the damping ratio.
    """
    C_h = 2 * ZETA * np.sqrt(K_total * I_total)
    
    return C_h
