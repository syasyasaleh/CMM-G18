"""
Mass Moment of Inertia (MMI) Code
Original Author: Eric
Refactored by: Matthew
Rewritten to make it easy to improt into the main code

Tis code calculates the MMI for a single blade. The main
script will call this function, multiplying
by the number of blades, and adding platform/added inertia
to get the total inertia.
"""
# Note we need to set the paramter values which we can alter for regression in the main analysis code

def blade_moment_of_inertia(blade_weight_N, blade_length_m, model='rod'):
    """
    Estimates the MMI of a single turbine blade about the hub.
      
    Our model treats the blade as a slender rod,
    which is a common first-order approximation.

     Parameters
    blade_weight_N:
        The total weight of one blade in Newtons (N).
        This value will be supplied by main_analysis.py from the
        realistic scaling law (get_realistic_blade_weight_N).
        
    blade_length_m:
        The full length of one blade in meters (m).
        This value will be supplied by main_analysis.py.

    model: str, optional
        The simplified physical model to use.
        'rod' (default): Assumes a uniform slender rod (I = 1/3 * m * L^2)
        'point': Assumes all mass is at the tip (I = m * L^2)
    """
    g = 9.81
    m = blade_weight_N / g

    if model.lower() == "rod":
        # Assumes a uniform slender rod: I = 1/3 * m * L^2
        I = (1/3) * m * blade_length_m**2
    elif model.lower() == "point":
        # Assumes all mass is at the tip: I = m * L^2, a lot less accurate!
        I = m * blade_length_m**2
    else:
        raise ValueError("model must be 'rod' or 'point'")

    return I
# Testing the code
if __name__ == "__main__":
    """
    This allows us to add test code here to run this file
    directly to make sure it works.
    
    This code will NOT run when the file is imported by
    main code.
    """
    print("--- Testing mmi.py directly ---")
    
    # Use the test values from Eric's original "raw" file
    test_weight = 85000
    test_length = 85
    
    # Call the function just like the main script will
    test_I = blade_moment_of_inertia(test_weight, test_length, model="rod")
    
    print(f"Test Case: {test_length}m blade, {test_weight}N weight")
    print(f"Calculated Single-Blade MMI: {test_I:,.2f} kg·m^2")
