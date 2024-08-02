import numpy as np


def lj_potential(dist_matrix, epsilon, sigma, cutoff):
    """
    Calculate the Lennard-Jones potential energy based on distances, with a cutoff.

    E = 4 * epsilon * ((sigma/dist)^12 - (sigma/dist)^6) for dist <= cutoff
    E = 0 for dist > cutoff

    :param dist_matrix: A distance matrix of shape (G, N).
    :param epsilon: A NumPy array of shape (N) for Lennard-Jones potential depths.
    :param sigma: A NumPy array of shape (N) for finite distances at which the
    inter-particle potential is zero.
    :param cutoff: A scalar value for the cutoff distance.
    :return: A NumPy array of shape (G) for the interaction energy.
    """
    epsilon = epsilon.reshape(1, -1)
    sigma = sigma.reshape(1, -1)

    # Calculate the Lennard-Jones potential
    p = 4 * epsilon * ((sigma / dist_matrix) ** 12 - (sigma / dist_matrix) ** 6)

    # Apply the cutoff: set energy to 0 where the distance is greater than the cutoff
    energy = np.where(dist_matrix <= cutoff, p, 0.0)  # (G, N)

    # Sum the energy along the N axis
    energy = np.sum(energy, axis=1)  # (G,)

    return energy


def gaussian(dist_matrix, height, width):
    """
    Calculate the Simplified Gaussian potential energy based on distances.

    G = height * exp(-(dist/sigma)^2) for dist <= cutoff
    G = 0 for dist > cutoff

    :param dist_matrix: A distance matrix of shape (G, N).
    :param height: A scalar value for the Gaussian potential height or amplitude.
    :param width: A scalar value for the Gaussian potential width.
    :return: A NumPy array of shape (G) for the calculated grids.
    """
    # Calculate the Gaussian potential
    p = height * np.exp(-((dist_matrix / width) ** 2))  # (G, N)

    # Sum the energy along the N axis
    energy = np.max(p, axis=1)  # (G,)

    return energy
