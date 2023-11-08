import numpy as np
from numba import jit


@jit(nopython=True)
def calculate_lj_potential(dist_matrix, epsilon, sigma, cutoff):
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
    lj_potential = (
        4 * epsilon * ((sigma / dist_matrix) ** 12 - (sigma / dist_matrix) ** 6)
    )

    # Apply the cutoff: set energy to 0 where the distance is greater than the cutoff
    energy = np.where(dist_matrix <= cutoff, lj_potential, 0.0)  # (G, N)

    # Sum the energy for each atom
    energy = np.sum(energy, axis=1)  # (G,)

    return energy


@jit(nopython=True)
def calculate_gaussian(dist_matrix, center, amplitude, width):
    """
    Calculate the Gaussian function value for a given distance.

    G = amplitude * exp(-(dist_matrix ** 2 / (2 * width ** 2))) #TODO: check the formula

    :param dist_matrix: A distance matrix of shape (G, N).
    :param center: The center position of the Gaussian function (mean value).
    :param amplitude: The amplitude of the Gaussian function.
    :param width: The width of the Gaussian (standard deviation).
    :return: A NumPy array of shape (G) for the Gaussian function values.
    """
    gaussian = amplitude * np.exp(-(dist_matrix**2 / (2 * width**2)))  # (G, N)

    # Sum the gaussian for each atom
    gaussian = np.sum(gaussian, axis=1)  # (G,)

    return gaussian
