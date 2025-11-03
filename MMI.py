"""
Mass Moment of Inertia (MMI) Code
Original Author: Eric
Refactored by: Matthew
"""
# Note we need to set the paramter values which we can alter for regression in the main analysis code
# We can also change the names of functions etc

import math

def blade_moment_of_inertia(blade_weight_N, blade_length_m, model='rod'):
    """
    Estimates the MMI of a single turbine blade about the hub.
    """
    g = 9.81
    m = blade_weight_N / g

    if model.lower() == "rod":
        # Assumes a uniform slender rod: I = 1/3 * m * L^2
        I = (1/3) * m * blade_length_m**2
    elif model.lower() == "point":
        # Assumes all mass is at the tip: I = m * L^2
        I = m * blade_length_m**2
    else:
        raise ValueError("model must be 'rod' or 'point'")

    return I