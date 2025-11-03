"""
Hydrodynamics Module
Original: Syasya
Refactored by: Matthew
Using values from (RAW)
"""

import numpy as np

# 1. Blade Lengths
BLADE_LENGTH_LOOKUP = np.array([0, 10, 20, 30, 40, 50, 60, 70])

# 2. Hydrostatic Stiffness
KH_LOOKUP = np.array([
    2.73897174e+09, 2.73539926e+09, 2.73182780e+09, 2.72825738e+09,
    2.72468799e+09, 2.72111962e+09, 2.71755228e+09, 2.71398598e+09
])

# 3. Mooring Stiffness
K_MOORING = 2121318150.0

# 4. Damping Ratio (ROW 2 of table)
ZETA = 0.20

# Functions that are easy to import

def calculate_stiffness(blade_length):
    """
    Calculates total restoring stiffness (K_total) for a
    given blade length.
    """
    
    # 1. Use np.interp to find the K_h for the exact blade_length
    K_h = np.interp(blade_length, BLADE_LENGTH_LOOKUP, KH_LOOKUP)
    
    # 2. Add the constant mooring stiffness
    K_total = K_h + K_MOORING
    
    return K_total

def calculate_damping(K_total, I_total):
    """
    Calculates total hydrodynamic damping (C_h) from the damping ratio.
    """
    C_h = 2 * ZETA * np.sqrt(K_total * I_total)
    
    return C_h