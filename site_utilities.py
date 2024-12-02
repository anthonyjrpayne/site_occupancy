import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.cm as cm


def get_data(atom_list, md_file, value):
    """
    Reads the .md file and returns a specified set of data

    Parameters
    ----------
    atom_list : list
        Atoms of interest to get the data for
        Can select atoms using the atom ID as appears in the .md file e.g. ["H1", "H2", "H3"]

    md_file : string
        The file to read

    value : string
        The data set to return
        'R' Position, 'V' Velocity, 'F' Force

    Returns
    -------
    data : list of arrays
        Data for each atom at every time step
        In the format data[atom][time][Atom, AtomNumber, x,y,z, value, ID]
    """

    data = []
    for atom in atom_list:
        with open(md_file, "r") as f:
            list0 = []
            for line in f:
                if value in line:
                    line = line.split()
                    identifier = line[0] + line[1]
                    if atom == identifier:
                        line.append(identifier)
                        list0.append(line)

            data.append(list0)
    return data


def get_positions(atoms, md_file):
    """
    Get the positions of specified atoms at every timestep

    Parameters
    ----------
    atoms : list
        List of atom IDs of interest, e.g., ["H1", "H2", "H3"]

    md_file : str
        Path to the file containing molecular dynamics data

    Returns
    -------
    atom_positions : dict
        A dictionary where each key is an atom ID (number and atom) and the value is a 2D numpy array
        Positions at time t can be accessed e.g [atom_positions['H1'][t] giving [x, y, z] cordinates
    """

    # get position data for the atoms of interest in the .md file
    data = get_data(atoms, md_file, 'R')
    atom_positions = {}

    # to angstroms
    to_ang = 0.529177249

    # Convert data to a dictionary of numpy arrays
    for atom in data:
        atom_id = atom[0][7]
        coords = np.array([[float(step[2]) * to_ang,  # Convert to Angstroms
                            float(step[3]) * to_ang,
                            float(step[4]) * to_ang] for step in atom])
        atom_positions[atom_id] = coords

    return atom_positions


def get_unit_cell(md_file):
    """
    Calculates angle between two vectors

    Parameters
    ----------
    md_file : string
         The file to read

    Returns
    -------
    time : list
        the number of time steps as a list

    total_energy : list
        the total energy for every time step

    hamiltonian_energy : list
        the hamiltonian_energy for every time step

    kinetic_energy : list
        the kinetic_energy for every time step
    """

    data = []
    with open(md_file, "r") as f:
        for line in f:
            if '<-- h' in line:
                line = line.split()
                data.append(line)
            if len(data) == 3:
                break


    # Cartesian components of lattice vectors
    a_cartesian = np.array([float(data[0][0]), float(data[0][1]), float(data[0][2])])
    b_cartesian = np.array([float(data[1][0]), float(data[1][1]), float(data[1][2])])
    c_cartesian = np.array([float(data[2][0]), float(data[2][1]), float(data[2][2])])

    # Lattice constants
    a = np.linalg.norm(a_cartesian)
    b = np.linalg.norm(b_cartesian)
    c = np.linalg.norm(c_cartesian)


    # Angles in radians
    cos_alpha = np.dot(b_cartesian, c_cartesian) / (b * c)
    cos_beta = np.dot(a_cartesian, c_cartesian) / (a * c)
    cos_gamma = np.dot(a_cartesian, b_cartesian) / (a * b)

    alpha = np.arccos(cos_alpha)
    beta = np.arccos(cos_beta)
    gamma = np.arccos(cos_gamma)

    # Convert angles to degrees if needed
    alpha_deg = np.degrees(alpha)
    beta_deg = np.degrees(beta)
    gamma_deg = np.degrees(gamma)

    # to angstroms
    to_ang = 0.529177249

    constants = [a * to_ang, b * to_ang, c * to_ang]
    angles = [alpha_deg, beta_deg, gamma_deg]

    return constants, angles


def calculate_center_of_mass(positions, masses):
    """
    Calculate the center of mass (COM) in 2D

    Parameters:
    positions (array-like): A list of [x, y] coordinates for each atom
    masses (array-like): A list of masses for each atom, same length as positions

    Returns:
    array: [x_COM, y_COM], the center of mass coordinates
    """
    positions = np.array(positions)
    masses = np.array(masses)

    # Weighted averages for x and y
    x_com = np.sum(positions[:, 0] * masses) / np.sum(masses)
    y_com = np.sum(positions[:, 1] * masses) / np.sum(masses)

    return np.array([x_com, y_com])


def find_middle_value(coordinate_dict):
    """
    Finds the center of a set of 2D coordinates

    Parameters:
    - coordinate_dict (dict): Dictionary of coordinates in the format {name: np.array([x, y])}

    Returns:
    - np.array: The middle value (center of mass) as a 2D point [x, y]
    """
    # Extract all coordinate arrays into a single numpy array via a list of values
    all_coords = np.array(list(coordinate_dict.values()))

    # Calculate the center of mass (average of all coordinates)
    middle_value = np.mean(all_coords, axis=0)

    return middle_value


def calculate_expansion_ratio(point, middle_value, xmin, xmax, ymin, ymax):
    """
    Calculate the integer expansion ratio needed to capture a point outside the given range, maintaining periodicity

    Parameters:
    - point (array-like): The x, y coordinates of the point [x, y]
    - middle_value (array-like): The center of the range [x_mid, y_mid]
    - xmin, xmax, ymin, ymax (float): Current bounds of the range

    Returns:
    - expansion_ratio (int): The integer ratio by which to expand the range to capture the point
    """
    x, y = point
    x_mid, y_mid = middle_value

    # Calculate the range of points in each direction
    current_x_range = xmax - xmin
    current_y_range = ymax - ymin

    # Determine how far the point is from the center value
    x_distance = abs(x - x_mid)
    y_distance = abs(y - y_mid)

    # Compute the required range to include the point
    required_x_range = 2 * x_distance  # Double the distance to ensure periodicity
    required_y_range = 2 * y_distance

    # Calculate the expansion ratios for x and y (+2 to ensure that the range is wide enough to capture)
    expansion_ratio_x = math.ceil(required_x_range / current_x_range) +2 if required_x_range > current_x_range else 1
    expansion_ratio_y = math.ceil(required_y_range / current_y_range) +2 if required_y_range > current_y_range else 1

    return expansion_ratio_x, expansion_ratio_y


def expand_positions_in_rotated_unit_cell(positions, a, b, gamma, rotation_angle, n_x, n_y, xmin, xmax, ymin, ymax):
    """
    Expands atomic positions in a 2D grid, accounting for a rotated unit cell and ensuring all points are captured

    Parameters:
    - positions (dict): Dictionary of atom names as keys and their positions (numpy arrays) as values
    - a, b (float): Lattice constants in Ångstroms for the unit cell.
    - gamma (float): Lattice angle in degrees for the unit cell.
    - rotation_angle (float): Rotation angle of the unit cell in degrees.
    - n_x, n_y (int): Initial number of unit cells to replicate in the x and y directions
    - xmin, xmax, ymin, ymax (float): Plot bounds to ensure coverage.

    Returns:
    - expanded_positions (dict): Dictionary of new atomic positions, with unique names
    """
    # Convert angles to radians
    gamma_rad = np.radians(gamma)
    rotation_rad = np.radians(rotation_angle)

    # Define lattice vectors in 2D based on a, b, and gamma
    a_vec = np.array([a, 0])  # Lattice vector a
    b_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad)])  # Lattice vector b

    # Apply rotation to the lattice vectors
    rotation_matrix = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])
    a_vec_rotated = np.dot(rotation_matrix, a_vec)
    b_vec_rotated = np.dot(rotation_matrix, b_vec)

    # Calculate effective width and height of the rotated unit cell
    unit_cell_width = max(abs(a_vec_rotated[0]), abs(b_vec_rotated[0]))
    unit_cell_height = max(abs(a_vec_rotated[1]), abs(b_vec_rotated[1]))

    # Adjust n_x and n_y based on plot bounds
    plot_width = xmax - xmin
    plot_height = ymax - ymin
    n_x = max(n_x, int(np.ceil(plot_width / unit_cell_width)))
    n_y = max(n_y, int(np.ceil(plot_height / unit_cell_height)))

    expanded_positions = {}

    # Loop through the atoms and generate replicated positions
    for atom, pos in positions.items():
        pos_2d = pos[:2]  # Use only x and y for 2D replication
        for i in range(-n_x, n_x + 1):  # Loop over x-direction
            for j in range(-n_y, n_y + 1):  # Loop over y-direction
                # Calculate new position
                translation = i * a_vec_rotated + j * b_vec_rotated
                new_pos = pos_2d + translation

                # Create a unique name for the replicated atom
                new_atom_name = f'{atom}_{i}_{j}'
                expanded_positions[new_atom_name] = new_pos

    return expanded_positions


def filter_positions(positions, x_min, x_max, y_min, y_max):
    """
    Filter positions based on specified x and y ranges

    Parameters:
    positions (dict): A dictionary where keys are position labels and values are arrays/lists with x, y coordinates
    x_min, x_max : float
        Minimum and maximum values for the x-coordinate
    y_min, y_max : float
        Minimum and maximum values for the y-coordinate

    Returns:
    dict: A filtered dictionary with only positions within the specified ranges
    """
    filtered_positions = {}
    for key, position in positions.items():
        x, y = position[:2]  # Extract x and y coordinates
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_positions[key] = position
    return filtered_positions


def find_nearest_position(center_of_mass, positions):
    """
    Find the nearest position to the center of mass and categorize it

    Parameters:
    center_of_mass (array-like): The center of mass coordinates as [x, y]
    positions (dict): A dictionary where keys are position labels (e.g., 'hole_N1_-1_0')
                      and values are arrays/lists with x, y coordinates

    Returns:
    tuple: Nearest position key (full label) and its category (e.g., 'hole')
    """
    min_distance = float('inf')  # Use infinity as the initial comparison value
    nearest_position_key = None

    for key, position in positions.items():
        position_xy = np.array(position[:2])  # Extract x, y coordinates
        distance = np.linalg.norm(center_of_mass - position_xy)
        if distance < min_distance:
            min_distance = distance
            nearest_position_key = key

    # Check if a nearest position was found
    if nearest_position_key is None:
        print("No nearest position found.")
        return None, None  # Return None for both if no position was found

    # Extract the general category from the position key (e.g., 'hole' from 'hole_N1_-1_0')
    category = nearest_position_key.split('_')[0]

    return nearest_position_key, category, min_distance


def plot_probabilities(probabilities, positions):
    """
    Plots the probability distribution of positions using a contour plot

    Parameters:
    - probabilities (dict): Dictionary of positions and their probabilities
    - positions (dict): Dictionary of position names as keys and their coordinates (numpy arrays) as values
    """
    # Prepare data for plotting
    x_coords = []
    y_coords = []
    prob_values = []

    for pos, prob in probabilities.items():
        if pos in positions:  # Ensure the position exists in the dictionary
            x_coords.append(positions[pos][0])
            y_coords.append(positions[pos][1])
            prob_values.append(prob)

    # Set the grid range from -3 to 3 on both axes
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4

    # Create a grid for contour plotting
    X_grid, Y_grid = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]

    # Interpolate the probabilities on the grid
    points = np.array([x_coords, y_coords]).T
    Z_grid = griddata(points, prob_values, (X_grid, Y_grid), method='cubic')

    # Get the min and max probability values for color scaling
    pmin = min(prob_values)
    pmax = max(prob_values)

    # Define contour levels
    levels = np.linspace(pmin, pmax, 200)
    line_levels = np.linspace(pmin, pmax, 20)

    # Plotting the contour plot
    plt.figure(figsize=(8, 6))
    CS = plt.contourf(X_grid, Y_grid, Z_grid, cmap=cm.Spectral, levels=levels, vmax=pmax, vmin=pmin)
    LS = plt.contour(X_grid, Y_grid, Z_grid, levels=line_levels, colors='k', linewidths=0.1, vmax=pmax, vmin=pmin)

    plt.colorbar(CS, format="%.3f", ticks=np.arange(pmin, pmax + 0.01, 0.05), orientation='vertical', fraction=0.02,
                 pad=0.1)

    plt.title('Probability Distribution')

    # Set axis limits to -3 to 3
    plt.xlim(x_min+1, x_max-1)
    plt.ylim(y_min+1, y_max-1)

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def calculate_probabilities(closest_positions):
    """
    Calculates the probability distribution of positions

    Parameters:
    - closest_positions (list): List of the closest position names at each timestep

    Returns:
    - probabilities (dict): Dictionary of positions and their probabilities
    """
    # Count occurrences of each position
    unique_positions, counts = np.unique(closest_positions, return_counts=True)
    total_timesteps = len(closest_positions)

    # Calculate probabilities
    probabilities = {pos: count / total_timesteps for pos, count in zip(unique_positions, counts)}
    return probabilities


def plot_positions(positions):
    """
    Plot the positions of sites

    Parameters
    ----------
    positions : dict
        Dictionary of positions where keys are labels and values are numpy arrays of shape (3,)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract and plot each position
    for label, pos in positions.items():
        ax.scatter(pos[0], pos[1], label=label, s=10, c='red')  # Use X and Y only for 2D
        # ax.text(pos[0], pos[1], label, fontsize=10)

    # Set axis labels and equal aspect ratio
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig('plot.png', transparent=True)
