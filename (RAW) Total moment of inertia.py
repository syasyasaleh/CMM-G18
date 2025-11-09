"""
Moment of Inertia Calculator for Spar Offshore Wind Turbine

I_total = I_platform + I_rotor + I_added

This script estimates:
Platform moment of inertia
Rotor moment of inertia
Added (hydrodynamic) inertia
and sums them for the total system inertia.

All units: SI (kg, m, kg·m²)
"""

import math

# 1. Platform Moment of Inertia
def calculate_platform_inertia(mass_platform, radius_platform, height_platform=None):
    """
    Calculate the platform's rotational inertia about its central (pitch/roll) axis.
    By default, assumes a solid cylinder rotating about its central axis.

    I_platform = 0.5 * m * r^2

    If height_platform is provided, you can use it to adjust the axis (optional).
    """
    I_platform = 0.5 * mass_platform * radius_platform**2
    return I_platform



# 2. Rotor Moment of Inertia
def calculate_rotor_inertia(rotor_mass, rotor_radius):
    """
    Calculate the rotor's moment of inertia assuming a solid disk (blades + hub).

    I_rotor = 0.5 * m * r^2
    """
    I_rotor = 0.5 * rotor_mass * rotor_radius**2
    return I_rotor


# 3. Added (Hydrodynamic) Inertia
def calculate_added_inertia(added_mass, reference_radius):
    """
    Calculate the added (hydrodynamic) moment of inertia.

    I_added = added_mass * r^2
    """
    I_added = added_mass * reference_radius**2
    return I_added


# 4. Total Moment of Inertia
def total_inertia(I_platform, I_rotor, I_added):
    """
    Sum the components.
    """
    return I_platform + I_rotor + I_added


# Example Usage
if __name__ == "__main__":
    # Example parameters (replace with real design data)
    mass_platform = 8.0e6       # kg
    radius_platform = 6.5       # m
    mass_rotor = 1.1e5          # kg
    radius_rotor = 63.0         # m
    added_mass = 1.5e6          # kg (hydrodynamic added mass)
    added_radius = 6.5          # m (approx same as platform radius)

    # Calculate each component
    I_platform = calculate_platform_inertia(mass_platform, radius_platform)
    I_rotor = calculate_rotor_inertia(mass_rotor, radius_rotor)
    I_added = calculate_added_inertia(added_mass, added_radius)

    # Total
    I_total = total_inertia(I_platform, I_rotor, I_added)

    # Print results
    print("\n=== Offshore Spar Wind Turbine Inertia Results ===")
    print(f"Platform Inertia: {I_platform:,.2e} kg·m²")
    print(f"Rotor Inertia:    {I_rotor:,.2e} kg·m²")
    print(f"Added Inertia:    {I_added:,.2e} kg·m²")
    print("----------------------------------------------")
    print(f"Total Inertia:    {I_total:,.2e} kg·m²\n")
