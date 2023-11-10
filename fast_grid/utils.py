import numpy as np
from numba import jit


@jit(nopython=True)
def mic_distance_matrix(pos1, pos2, cell_vectors):
    """
    Calculate the distance matrix between two sets of positions with minimum image convention
    for a periodic cell.

    :param pos1: An array of shape (N1, 3) containing the positions of the first set of atoms.
    :param pos2: An array of shape (N2, 3) containing the positions of the second set of atoms.
    :param cell_vectors: A (3, 3) array where rows are the vectors defining the unit cell.
    :return: A distance matrix of shape (N1, N2).
    """
    N1 = pos1.shape[0]
    N2 = pos2.shape[0]
    distance_matrix = np.zeros((N1, N2), dtype=np.float64)
    inverse_cell = np.linalg.inv(cell_vectors)

    for i in range(N1):
        for j in range(N2):
            diff = pos1[i] - pos2[j]

            # Transform diff to the cell basis
            diff_cell_basis = np.dot(inverse_cell, diff)

            # Apply the minimum image convention in the cell basis
            diff_cell_basis -= np.round(diff_cell_basis)

            # Transform back to the original space
            diff = np.dot(cell_vectors, diff_cell_basis)

            # Calculate the distance
            distance = np.sqrt(np.sum(diff**2))

            distance_matrix[i, j] = distance

    return distance_matrix
