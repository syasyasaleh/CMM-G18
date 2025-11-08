import numpy as np
# Geometry
radius=7
A=np.pi*radius**2
mass = 10000000 #kg
# Depth limits (z = 0 at waterline, negative downward)
z_top = 0
z_bottom = -91
# Define zones (m)
zones = [
    {"z1": 0, "z2": -61, "rho": 300},    # Air-filled section
    {"z1": -61, "z2": -71, "rho": 1025}, # Water ballast
    {"z1": -71, "z2": -91, "rho": 5000}  # Solid ballast
]
Mz_total=0
M_total=0

for zone in zones:
    z1, z2, rho = zone["z1"], zone["z2"], zone["rho"]
    dz = abs(z2 - z1)

    M_zone = rho * A * dz     #mass of that zone(ballast,water,air)
    z_cg_zone = 0.5 * (z1 + z2)
    
    M_total += M_zone
    Mz_total += M_zone * z_cg_zone

z_total = Mz_total / M_total

print(f"Total mass={M_total} kg")
print(f"Total centroid={z_total} m")
Height_tower=83 #m
Diameter_nacelle=5 #m
Height_narcelle=88

Blade_length= np.arange(60, 120, 10) #m

#Mass #kg
Mass_blade=1.17 * (Blade_length** 2.32) #kg
Mass_nacelle=350000 
Mass_tower=670000

y_cg_list = []

for blade_length in Blade_length:
    
    y_cg_blade = 0.4*blade_length+Height_tower
    y_cg_tower = Height_tower/2
    y_cg_nacelle = Height_tower+(Diameter_nacelle/2)
    
    My_total=Mass_blade*y_cg_blade + Mass_nacelle*y_cg_nacelle + Mass_tower*Height_tower
    Mm_total=Mass_blade+Mass_nacelle+Mass_tower

    y_total = My_total/Mm_total

    y_cg_list.append(y_total)

print (f"total mass above water={Mm_total}kg")
print (f"y cg above water surface={y_total}m")

M_final=M_total+Mm_total
Mcg_final= M_total*(abs(z_bottom-z_total))+ (Mm_total*(y_total+abs(z_bottom)))

cg_final= Mcg_final/M_final

print(f"mass final={M_final}kg")
print (f"cg final from bottom platform={cg_final}")

#centre of buoyancy 
#only account for the submerged part of the wind turbine (the spar buoy platform)

from scipy.integrate import quad

def volume_cylinder(h):
    return np.pi*(radius**2)*h

I, err=quad(volume_cylinder, z_top, z_bottom)

cb_final=I/volume_cylinder(z_top-z_bottom)

print(f"cb from bottom of platform={cb_final}m")

#metacentric height

metacentric_height= cb_final-cg_final

print (f"metacentric height={metacentric_height}m")

# finding restoring moment of pitching movement

displaced_volume =np.pi*(radius**2)*(z_top-z_bottom)
seawater_density=1025 #kg/m3
g=9.81 #m/s2

Restoring_moment = seawater_density*g*displaced_volume*metacentric_height

print(f"restoring moment (K)={Restoring_moment}")

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

import math 
import pandas as pd

Restoring_hydrostatic_moment=np.array([2.73897174e+09,2.73539926e+09,2.73182780e+09,2.72825738e+09,2.72468799e+09,2.72111962e+09,2.71755228e+09,2.71398598e+09])
hydrodynamic_damping_list=[]

delta=np.arange(0,1,0.2)

for d in np.arange(0,1,0.2):
    Ch_per_Kh=[] #damping list for every restoring hydrostatic moment
    for Kh in Restoring_hydrostatic_moment:
        hydrodynamic_damping=2*d*math.sqrt(Kh* I_total) 
        Ch_per_Kh.append(hydrodynamic_damping) 
    hydrodynamic_damping_list.append(Ch_per_Kh)

hydrodynamic_damping_array=np.array(hydrodynamic_damping_list)

# Create column names dynamically
columns = [f"Ch{i+1}" for i in range(len(Restoring_hydrostatic_moment))]

# Combine damping ratio + Ch values into DataFrame
df = pd.DataFrame(hydrodynamic_damping_array, columns=columns)
df.insert(0, "Damping Ratio", delta)

# Print nicely
print(df.to_markdown(index=False, floatfmt=".2e"))

