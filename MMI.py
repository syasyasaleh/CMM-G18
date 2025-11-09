"""
Mass Moment of Inertia (MMI) Model for a Floating Spar Buoy Turbine

This module calculates the total MMI of the turbine
about the platform's pitch axis (Center of Buoyancy).

It is designed to be imported by other analysis scripts (Main Code - Stability.py).
The main public function is:
    - calculate_total_inertia(blade_length_m)
"""

import numpy as np

# Creating a function for variable mass

def _get_realistic_blade_mass_kg(blade_length_m):
    """
    Calculates the blade mass based on a citable scaling law.
    """
    # Scaling law: m = a*L**b = 1.17 * L^2.32
    # (see the reference in the report)
    return 1.17 * (blade_length_m ** 2.32)


# Set the system constants
# References for the data values used are in the report and taken from other members codes
# All coordinates are relative to Still Water Level (SWL)
PITCH_AXIS_Z = -45.5 # [m] (Pitch axis is at Center of Buoyancy, CB) 
"""
Using Syasya's calculation:
Distance from bottom of platform to SWL
- distance from bottom to Centre Buoyancy
91m - 45.5m = 45.5m
"""

# Part 1: Platform (Spar Buoy) Table 3-1 of the Floating System
PLATFORM_MASS = 7466330 # [kg] T.3-1 F.S
PLATFORM_CM_Z = -89.9155 # [m] (Z-coord of platform's own CM) T.3-1 F.S
PLATFORM_MMI_CM = 4.22923e9 # [kg·m^2] (Inertia about its own CM) T.3-1 F.S

# Part 2: Tower
TOWER_MASS = 249718 # [kg] T.32-2 F.S
TOWER_CM_Z = 43.4 # [m] (Z-coord of tower's own CM) F.S
TOWER_MMI_CM = 1.19e8 # [kg·m^2] (Inertia about its own CM) using F.S tower properties
# See Tower_mmi_CM.py for detail

# Part 3: Nacelle + Hub Assembly
NACELLE_MASS = 240000 # [kg] T.4-1 W.T
HUB_MASS = 56780 # [kg] T.4-1 W.T
NACELLE_HUB_CM_Z = 87.6 # [m] (Z-coord of the combined hub/nacelle CM) T.4-1 W.T

# (Inertia of Rotor + Nacelle about their combined CM)
NACELLE_HUB_MMI_CM = 115926 + 2607890 # [kg·m^2] T.4-1 W.T

# Part 4: Hydrodynamic Added Mass
I_ADDED_MASS = 3.8e10 # [kg·m^2] Fig.4-4 F.S
"""
JUSTIFIABLE SIMPLIFICATION
This value (3.8e10) is sourced from Fig 4-4 of the F.S. report,
where it is referenced to the Still Water Level (SWL, z=0.0).
For this 1-DOF model, we are assuming this value is a
reasonable approximation for the added mass MMI about our
system pitch axis (CB, z=-45.5).

A full Parallel Axis Theorem transfer was not performed
because the required hydrodynamic mass term 
was not available in our specification document.
"""



# Main public function

def calculate_total_inertia(blade_length_m):
    """
    Calculates the TOTAL Mass Moment of Inertia (MMI) for the
    entire floating turbine system about its pitch axis (the CB).
    
    It uses the Parallel Axis Theorem (I = I_cm + m * d^2) to move
    the inertia of each component to the system's pitch axis.
    """
    
    # a) Platform Inertia (Moved to Pitch Axis)
    d_plat = PLATFORM_CM_Z - PITCH_AXIS_Z
    I_platform = PLATFORM_MMI_CM + (PLATFORM_MASS * (d_plat ** 2))
    
    # b) Tower Inertia (Moved to Pitch Axis)
    d_tower = TOWER_CM_Z - PITCH_AXIS_Z
    I_tower = TOWER_MMI_CM + (TOWER_MASS * (d_tower ** 2))
    
    # c) Nacelle + Hub Inertia (Moved to Pitch Axis)
    m_nacelle_hub = NACELLE_MASS + HUB_MASS
    d_nacelle_hub = NACELLE_HUB_CM_Z - PITCH_AXIS_Z
    I_nacelle_hub = NACELLE_HUB_MMI_CM + (m_nacelle_hub * (d_nacelle_hub ** 2))

    # d) Blade Inertia (Moved to Pitch Axis)
    m_one_blade = _get_realistic_blade_mass_kg(blade_length_m)
    I_one_blade_about_root = (1/3) * m_one_blade * (blade_length_m ** 2)
    d_blades = d_nacelle_hub  
    I_blades_variable = 3 * (I_one_blade_about_root + (m_one_blade * (d_blades ** 2)))
    
    # e) Added Mass Inertia (Simplified)
    # Per our simplification, we use the sourced value directly
    # and assume it's a reasonable approximation at our pitch axis.
    I_added_mass_moved = I_ADDED_MASS
    
    # f) Total System MMI
    # Sum all parts
    I_total = (I_platform + I_tower + I_nacelle_hub + 
               I_blades_variable + I_added_mass_moved)
    
    return I_total

# 4. Testing code run for our blade length ranges
if __name__ == "__main__":
    
    print("Running a test for our range of blade lengths")
    
    # Test our specified range of blade lengths
    print("Testing analysis range (60m to 120m)")
    blade_lengths_m = np.arange(60, 121, 10)
    
    for L in blade_lengths_m:
        I_total = calculate_total_inertia(L)
        print(f"L = {L:3d} m,  I_total = {I_total:.4e} kg·m^2")
