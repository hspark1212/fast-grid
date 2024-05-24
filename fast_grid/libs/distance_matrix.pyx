# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sqrt


def distance_matrix_triclinic_cython(np.ndarray[np.float64_t, ndim=2] pos1,
                              np.ndarray[np.float64_t, ndim=2] pos2, 
                              np.ndarray[np.float64_t, ndim=2] box):
    cdef int i, j
    cdef int n = pos1.shape[0]
    cdef int m = pos2.shape[0]
    cdef double r2

    cdef int ix, iy, iz
    cdef float rx, ry0, ry1, rz0, rz1, rz2, dsq
    cdef float dsq_min = np.finfo(np.float64).max
    
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((n, m), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] diff = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dx_min = np.zeros(3, dtype=np.float64)

    for i in prange(n, nogil=True):
        for j in range(m):
            diff[0] = pos2[j, 0] - pos1[i, 0]
            diff[1] = pos2[j, 1] - pos1[i, 1]
            diff[2] = pos2[j, 2] - pos1[i, 2]

            dsq_min = 10000.
            dx_min[0] = 0
            dx_min[1] = 0
            dx_min[2] = 0

            for ix in range(-1, 2):
                rx = diff[0] + box[0, 0] * ix
                for iy in range(-1, 2):
                    ry0 = rx + box[1, 0] * iy
                    ry1 = diff[1] + box[1, 1] * iy
                    for iz in range(-1, 2):
                        rz0 = ry0 + box[2, 0] * iz
                        rz1 = ry1 + box[2, 1] * iz
                        rz2 = diff[2] + box[2, 2] * iz
                        dsq = rz0 * rz0 + rz1 * rz1 + rz2 * rz2
                        if dsq < dsq_min:
                            dsq_min = dsq
                            dx_min[0] = rz0
                            dx_min[1] = rz1
                            dx_min[2] = rz2
                    
            r2 = dx_min[0] * dx_min[0] + dx_min[1] * dx_min[1] + dx_min[2] * dx_min[2]
            distances[i, j] = sqrt(r2)
    
    return distances