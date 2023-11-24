# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport exp


def minimum_image_triclinic(np.ndarray[np.float64_t, ndim=1] dx, np.ndarray[np.float64_t, ndim=2] box):
    cdef int ix, iy, iz
    cdef float rx, ry0, ry1, rz0, rz1, rz2, dsq
    cdef float dsq_min = np.finfo(np.float64).max
    cdef np.ndarray[np.float64_t, ndim=1] dx_min = np.zeros(3, dtype=np.float64)

    for ix in range(-1, 2):
        rx = dx[0] + box[0, 0] * ix
        for iy in range(-1, 2):
            ry0 = rx + box[1, 0] * iy
            ry1 = dx[1] + box[1, 1] * iy
            for iz in range(-1, 2):
                rz0 = ry0 + box[2, 0] * iz
                rz1 = ry1 + box[2, 1] * iz
                rz2 = dx[2] + box[2, 2] * iz
                dsq = rz0 * rz0 + rz1 * rz1 + rz2 * rz2
                if dsq < dsq_min:
                    dsq_min = dsq
                    dx_min[0] = rz0
                    dx_min[1] = rz1
                    dx_min[2] = rz2

    dx[:] = dx_min


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_potential_cython(np.ndarray[np.float64_t, ndim=2] pos1,
                              np.ndarray[np.float64_t, ndim=2] pos2, 
                              np.ndarray[np.float64_t, ndim=2] box,
                              np.ndarray[np.float64_t, ndim=1] epsilon,
                              np.ndarray[np.float64_t, ndim=1] sigma,
                              float cutoff):
    cdef int i, j
    cdef int n = pos1.shape[0]
    cdef int m = pos2.shape[0]
    cdef double r2, lj6, lj12, inv_r2, inv_r6, inv_r12, e, s, s6, s12, energy
    cdef double threshold = 1e-10
    cdef double cutoff_squared = cutoff * cutoff
    
    cdef np.ndarray[np.float64_t, ndim=1] energy_grid = np.zeros((n), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] diff = np.zeros(3, dtype=np.float64)

    for i in prange(n, nogil=True):
        energy = 0.0
        for j in range(m):
            diff[0] = pos2[j, 0] - pos1[i, 0]
            diff[1] = pos2[j, 1] - pos1[i, 1]
            diff[2] = pos2[j, 2] - pos1[i, 2]
            
            with gil:
                minimum_image_triclinic(diff, box)
            r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

            if r2 < cutoff_squared and r2 > threshold:
                e = epsilon[j]
                s = sigma[j]
                s6 = s * s * s * s * s * s
                s12 = s6 * s6
                lj6 = 4 * e * s6
                lj12 = 4 * e * s12
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                inv_r12 = inv_r6 * inv_r6
                
                energy += (lj12 * inv_r12) - (lj6 * inv_r6)
        
        energy_grid[i] += energy
    
    return energy_grid


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gaussian_cython(np.ndarray[np.float64_t, ndim=2] pos1,
                              np.ndarray[np.float64_t, ndim=2] pos2, 
                              np.ndarray[np.float64_t, ndim=2] box,
                              float height,
                              float width,
                              float cutoff):
    cdef int i, j
    cdef int n = pos1.shape[0]
    cdef int m = pos2.shape[0]
    cdef double r2, energy
    cdef double threshold = 1e-10
    cdef double width_squared = width * width
    cdef double cutoff_squared = cutoff * cutoff
    
    cdef np.ndarray[np.float64_t, ndim=1] energy_grid = np.zeros((n), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] diff = np.zeros(3, dtype=np.float64)

    for i in prange(n, nogil=True):
        energy = 0.0
        for j in range(m):
            diff[0] = pos2[j, 0] - pos1[i, 0]
            diff[1] = pos2[j, 1] - pos1[i, 1]
            diff[2] = pos2[j, 2] - pos1[i, 2]
            
            with gil:
                minimum_image_triclinic(diff, box)
            r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

            if r2 < cutoff_squared and r2 > threshold:
                energy += height * exp(r2 / width_squared)
        
        energy_grid[i] += energy
    
    return energy_grid