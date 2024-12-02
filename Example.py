# Calculates a Site Occupancy from a CASTEP AIMD simulation
# import functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from site_utilities import *

# Initial Variables
# Path to the AIMD trajectory file
md_file = "Example.md"
# Define the atoms of interest
molecule_atoms = ['H1', 'H2', 'O1']
masses = np.array([1.00784, 1.00784, 15.999])  # Atomic masses for molecule atoms
surface_B = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']  # B surface atoms
surface_N = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']  # N surface atoms
key_B = ['B2', 'B5', 'B6', 'B3']  # for creating additional sites
key_N = ['N3', 'N2']  # for creating additional sites
# Define size of the grid of points to be used (angstrom)
xmin, xmax = -1, 7
ymin, ymax = -5, 8
# cell angle
cell_angle = -40.9
# likely maximum distance between the closest site and center of mass (angstroms)
expected_distance = 1
tolerence_distance = 0.5

## begin calculation
# Combine all atoms of interest into a single list
atoms_of_interest = surface_B + surface_N + molecule_atoms

# Get positions of the atoms of interest at every timestep
# The returned 'atom_positions' is a dictionary with atom labels as keys and positions as numpy arrays
atom_positions = get_positions(atoms_of_interest, md_file)

# Get the total number of timesteps from the number of in the first index as a list
first_atom_positions = list(atom_positions.values())[0]
total_timesteps = first_atom_positions.shape[0]

# Get unit cell constants and angles
constants, angles = get_unit_cell(md_file)
a = constants[0]
b = constants[1]
gamma = angles[2]

# List to hold the final categorization of positions at each timestep
nearest_position_list = []
all_positions = []
all_com = []

# Loop through the timesteps (doing 1/10 of the data set for illustration)
for t in range(0, total_timesteps, 10):
    # Get positions of relovent atoms at timestep t
    # positions of the molecule
    molecule_positions = np.array([atom_positions[molecule_atoms[0]][t],
                                   atom_positions[molecule_atoms[1]][t],
                                   atom_positions[molecule_atoms[2]][t]])
    # N3, N2 needed for site calculations
    nitrogen_positions = np.array([atom_positions[key_N[0]][t], atom_positions[key_N[1]][t]])
    # B2, B5, B6 needed for site calculations
    boron_positions = np.array([atom_positions[key_B[0]][t], atom_positions[key_B[1]][t], atom_positions[key_B[2]][t]])

    # Calculate positions for bridge sites and hole sites around N atom
    hole_position = np.mean([nitrogen_positions[0], boron_positions[1]], axis=0)
    bond_position1 = np.mean([nitrogen_positions[0], boron_positions[0]], axis=0)
    bond_position2 = np.mean([nitrogen_positions[0], boron_positions[2]], axis=0)
    bond_vert_position1 = np.mean([nitrogen_positions[0], atom_positions[key_B[3]][t]], axis=0)

    # Dictionary to store and access positions
    positions = {
        "hole": hole_position,
        "bond1": bond_position1,
        "bond2": bond_position2,
        "bond3": bond_vert_position1,
        "Ntop1": nitrogen_positions[0],
        "Ntop2": nitrogen_positions[1],
        "BTop1": boron_positions[0],
        "BTop2": boron_positions[1]
    }

    # Further positions around the hole site close to B and N atoms
    # based on average position between hole site and outer atoms
    nearB1 = (positions["BTop2"] + positions["hole"]) / 2
    nearN1 = (positions["Ntop1"] + positions["hole"]) / 2
    nearB2 = (positions["hole"] + positions["BTop1"]) / 2
    nearN2 = (positions["hole"] + positions["Ntop2"]) / 2
    # additional positions to create symmetrical set of points
    nearB3 = nearB2 + (nearN1 - positions["BTop1"])
    nearN3 = nearN2 + (nearN1 - positions["BTop1"])

    # Update the positions dictionary with newly calculated positions
    positions.update({
        "nearB1": nearB1,
        "nearN1": nearN1,
        "nearB2": nearB2,
        "nearN2": nearN2,
        "nearB3": nearB3,
        "nearN3": nearN3
    })

    # Remove duplicate B and N atom keys used only for calculating site positions
    positions.pop("BTop2", None)
    positions.pop("Ntop2", None)

    # uncomment to check positions arround an atom
    # plot_positions(new_positions)
    # plt.show()

    # Now create the new positions across the entire unit cell based on the position of the Nitrogen atoms
    # Dictionary to store new positions
    new_positions = {}

    # Loop round every nitrogen atom and add appropriate positions accross the entire unit cell
    for nitrogen_atom in surface_N:
        # For every key and position in the dictionary
        for key, original_position in positions.items():
            # get the position of the nitrogen atom
            new_nitrogen_position = atom_positions[nitrogen_atom][t]
            # calculate difference between the original nitrogen and the site
            displacement = original_position - positions['Ntop1']
            # apply this displacement to the new position
            new_position = new_nitrogen_position + displacement

            # Create unique key based on the site name and the nitrogen atom ID
            new_key = f"{key}_{nitrogen_atom}"
            # store in new directory
            new_positions[new_key] = new_position

    # uncomment to check positions arround each nitrogen atom
    # plot_positions(new_positions)
    # plt.show()

    # Calculate the center of mass for the molecule
    com = calculate_center_of_mass(molecule_positions, masses)

    # Find the center of the new dictionary positions to be used to check the com is within
    middle_value = find_middle_value(new_positions)
    middle2D = [middle_value[0], middle_value[1]]

    # Calculate site expansion ratio to ensure the com is found within the set of points whilst maintaining periodicity
    expansion_ratio_x, expansion_ratio_y = calculate_expansion_ratio(com, middle2D, xmin, xmax, ymin, ymax)

    # Expand the number of points such that the com is captured based on the unit cell size
    expanded_positions = expand_positions_in_rotated_unit_cell(new_positions,
                                                               a, b, gamma, cell_angle,
                                                               expansion_ratio_x,
                                                               expansion_ratio_y,
                                                               xmin, xmax, ymin, ymax)

    # # uncomment to check positions expanded atoms contain com
    # plot_positions(expanded_positions)
    # plt.scatter(com[0], com[1], c='blue')
    # plt.show()

    # Filter out positions that are outside the maximum expected distance of sites
    filtered_positions = filter_positions(expanded_positions,
                                          com[0] - expected_distance,
                                          com[0] + expected_distance,
                                          com[1] - expected_distance,
                                          com[1] + expected_distance)

    # uncomment to check positions expanded atoms contain com
    # plot_positions(filtered_positions)
    # plt.scatter(com[0], com[1], c='blue')
    # plt.show()

    # Find the nearest position to the center of mass
    nearest_position_key, category, min_distance = find_nearest_position(com, filtered_positions)

    # check to ensure the value is within the specified tolerance
    if min_distance > tolerence_distance:
        print("at timestep " + str(t) + " nearest site is: " + str(min_distance) + " not included")

    else:
        # Store the category of the nearest position
        nearest_position_list.append(category)

        # Further combine categories based on equivalent sites (e.g., "bond1/2/3" to "bond", "nearB" to "nearB")
        nearest_position_list = ["bond" if item.startswith("bond") else item for item in nearest_position_list]
        nearest_position_list = ["nearB" if item.startswith("nearB") else item for item in nearest_position_list]
        nearest_position_list = ["nearN" if item.startswith("nearN") else item for item in nearest_position_list]

        # store all com sampled
        all_com.append(com)

    # store the sites of the first timestep for plotting
    if t == 0:
        plot_positions = new_positions


# Calculate the probabilities of positions
probabilities = calculate_probabilities(nearest_position_list)

# uncomment to save the results
# Save the results (min_distance and position list) to a text file
# data_file_name = md_file + '.position_data.txt'
# with open(data_file_name, 'w') as f:
#     f.write("Nearest position at each timestep\n")
#     for position in nearest_position_list:
#         f.write(f"{position}\n")

# Save the results (min_distance and position list) to a text file
# site_file_name = md_file + '.site_data.txt'
# with open(site_file_name, 'w') as f:
#     f.write("Positions of sites at each timestep\n")
#     for site, position in all_positions:  # Loop through each dictionary
#         f.write(f"{site}: {position.tolist()}\n")  # Convert NumPy arrays to lists for readability
#     f.write("\n")  # Add a blank line between timesteps for clarity

# print("Data files written")

# # Define positions for plotting the primitive cell
# positions_prim = {
#     "hole": hole_position,
#     "bond": bond_position1,
#     "Ntop1": nitrogen_positions[0],
#     "BTop1": boron_positions[0],
#     "nearN": nearN1,
#     "nearB": nearB2
# }

# Map probabilities to positions using same ones as generated originally
positions_prim_probs = {
    "nearB1": probabilities["nearB"],
    "nearN1": probabilities["nearN"],
    "nearB2": probabilities["nearB"],
    "nearN2": probabilities["nearN"],
    "nearB3": probabilities["nearB"],
    "nearN3": probabilities["nearN"],
    "hole": probabilities["hole"],
    "bond1": probabilities["bond"],
    "bond2": probabilities["bond"],
    "bond3": probabilities["bond"],
    "Ntop1": probabilities["Ntop1"],
    "BTop1": probabilities["BTop1"]
}

# Expand positions to ensure periodic boundary conditions and plot the probabilities
expanded_positions = expand_positions_in_rotated_unit_cell(plot_positions, a, b, gamma, cell_angle, 1, 1, xmin, xmax, ymin, ymax)

# Create a dictionary to store the probabilities for each position
position_probabilities = {}

# Iterate over the positions dictionary
for key, position in expanded_positions.items():
    # Extract the root from the key (e.g., "nearN1" from "nearN1_-2_-2")
    root = key.split('_')[0]

    # Apply the probability from the probabilities dictionary if the root exists
    if root in positions_prim_probs:
        # Assign the probability to the corresponding position key
        position_probabilities[key] = positions_prim_probs[root]


plot_probabilities(position_probabilities, expanded_positions)

print("done")
