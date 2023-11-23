import numpy as np
from numpy.typing import NDArray
from numba import jit


def check_inputs_energy_grid(
    pos1: NDArray[np.float64],
    pos2: NDArray[np.float64],
    cell_vectors: NDArray[np.float64],
    inverse_cell: NDArray[np.float64],
    cutoff: float,
    energy_grid: NDArray[np.float64],
    epsilon: NDArray[np.float64] = None,
    sigma: NDArray[np.float64] = None,
    gaussian_height: float = None,
    gaussian_width: float = None,
):
    """Check inputs to use energy_grid_cython.

    :param pos1: Grid positions for energy grid, expected shape (G, 3).
    :param pos2: Positions for atoms, expected shape (N, 3).
    :param cell_vectors: Cell vectors, expected shape (3, 3).
    :param inverse_cell: Inverse cell, expected shape (3, 3).
    :param cutoff: Cutoff distance, expected to be a non-negative float.
    :param energy_grid: Zero array for energy grid, expected shape (G,).
    :param epsilon: Mixing epsilon, expected shape (N,).
    :param sigma: Mixing sigma, expected shape (N,).
    :param gaussian_height: Gaussian height
    :param gaussian_width: Gaussian width
    """
    assert pos1.shape[1] == 3 and pos1.ndim == 2, "pos1 must be of shape (G, 3)"
    assert pos2.shape[1] == 3 and pos2.ndim == 2, "pos2 must be of shape (N, 3)"
    assert cell_vectors.shape == (3, 3), "cell_vectors must be of shape (3, 3)"
    assert inverse_cell.shape == (3, 3), "inv_cell must be of shape (3, 3)"
    assert (
        isinstance(cutoff, float) and cutoff >= 0
    ), "cutoff must be a non-negative float"
    assert energy_grid.shape == (pos1.shape[0],), "energy_grid must be of shape (G,)"
    if epsilon is not None:
        assert (
            epsilon.ndim == 1 and epsilon.shape[0] == pos2.shape[0]
        ), "epsilon must be of shape (N,)"
    if sigma is not None:
        assert (
            sigma.ndim == 1 and sigma.shape[0] == pos2.shape[0]
        ), "sigma must be of shape (N,)"
    if gaussian_height is not None:
        assert isinstance(gaussian_height, float), "gaussian_height must be a float"
    if gaussian_width is not None:
        assert isinstance(gaussian_width, float), "gaussian_width must be a float"
    assert (
        pos1.dtype == np.float64
        and pos2.dtype == np.float64
        and cell_vectors.dtype == np.float64
        and inverse_cell.dtype == np.float64
        and epsilon.dtype == np.float64
        and sigma.dtype == np.float64
        and energy_grid.dtype == np.float64
    ), "All array inputs must be of type np.float64"


# deprecated function
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
    distance_matrix = np.empty((N1, N2), dtype=np.float32)
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
