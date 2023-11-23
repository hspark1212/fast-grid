# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np

import cython
cimport numpy as np
from cython.parallel import prange
from libc.math cimport round, exp


def gaussian_cython(np.ndarray[np.float64_t, ndim=2] pos1,
                            np.ndarray[np.float64_t, ndim=2] pos2,
                            np.ndarray[np.float64_t, ndim=2] cell_vectors,
                            np.ndarray[np.float64_t, ndim=2] inverse_cell,
                            float height,
                            float width,
                            float cutoff,
                            np.ndarray[np.float64_t, ndim=1] energy_grid,
                            ):
    
    cdef int G = pos1.shape[0] # grid size
    cdef int N = pos2.shape[0] # number of atoms
    cdef int i, j = 0
    cdef float diff_x, diff_y, diff_z
    cdef float diff_cell_basis_x, diff_cell_basis_y, diff_cell_basis_z
    cdef float r2, lj6, lj12, inv_r2, inv_r6, inv_r12, e, s, s6, s12 #remove this line
    cdef float energy
    cdef float threshold = 1e-10
    cdef float width_squared = width * width
    cdef float cutoff_squared = cutoff * cutoff

    for i in prange(G, nogil=True):
        energy = 0.0
        for j in range(N):
            diff_x = pos1[i, 0] - pos2[j, 0]
            diff_y = pos1[i, 1] - pos2[j, 1]
            diff_z = pos1[i, 2] - pos2[j, 2]
            
            # Matrix multiplication with the inverse cell matrix
            diff_cell_basis_x = (
                inverse_cell[0, 0] * diff_x
                + inverse_cell[0, 1] * diff_y
                + inverse_cell[0, 2] * diff_z
            )
            diff_cell_basis_y = (
                inverse_cell[1, 0] * diff_x
                + inverse_cell[1, 1] * diff_y
                + inverse_cell[1, 2] * diff_z
            )
            diff_cell_basis_z = (
                inverse_cell[2, 0] * diff_x
                + inverse_cell[2, 1] * diff_y
                + inverse_cell[2, 2] * diff_z
            )

            # Applying the minimum image convention
            diff_cell_basis_x = diff_cell_basis_x - round(diff_cell_basis_x)
            diff_cell_basis_y = diff_cell_basis_y - round(diff_cell_basis_y)
            diff_cell_basis_z = diff_cell_basis_z - round(diff_cell_basis_z)

            # Transforming back to the original space
            diff_x = (
                cell_vectors[0, 0] * diff_cell_basis_x
                + cell_vectors[0, 1] * diff_cell_basis_y
                + cell_vectors[0, 2] * diff_cell_basis_z
            )
            diff_y = (
                cell_vectors[1, 0] * diff_cell_basis_x
                + cell_vectors[1, 1] * diff_cell_basis_y
                + cell_vectors[1, 2] * diff_cell_basis_z
            )
            diff_z = (
                cell_vectors[2, 0] * diff_cell_basis_x
                + cell_vectors[2, 1] * diff_cell_basis_y
                + cell_vectors[2, 2] * diff_cell_basis_z
            )
            
            # Calculating the distance
            r2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
            
            if r2 < cutoff_squared and r2 > threshold:
                # Calculate Guassian
                energy += height * exp(r2 / width_squared)

        energy_grid[i] += energy
            
    return energy_grid