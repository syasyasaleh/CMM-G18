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

Blade_length= np.arange(0, 80, 10) #m

#Mass #kg
Mass_blade=300*Blade_length
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
