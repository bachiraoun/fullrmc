"""
This is a C compiled Cython generated module to transform coordinates. It contains the following methods.
                   
**transform_coordinates**: It transforms coordinates array.
    :Arguments:
       #. transMatrix (float32 array): The (3,3) transformation matrix
       #. coords (float32 array): The (N,3) coordinates array.
                                         
    :Returns:
       #. transCoords (float32 array): The (N,3) transformed coordinates array.
"""

from libc.math cimport sqrt, abs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
#from cython.parallel import prange

# declare types
NUMPY_FLOAT32 = np.float32
NUMPY_INT32   = np.int32
ctypedef np.float32_t C_FLOAT32
ctypedef np.int32_t   C_INT32


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def transform_coordinates( np.ndarray[C_FLOAT32, ndim=2] transMatrix not None,
                           np.ndarray[C_FLOAT32, ndim=2] coords not None):
    # declare variables
    cdef C_INT32 i
    # declare transCoords
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] transCoords = np.empty((coords.shape[0],3), dtype=NUMPY_FLOAT32)
    # loop
    for i from 0 <= i < coords.shape[0]:
        transCoords[i,0] = coords[i,0]*transMatrix[0,0] + coords[i,1]*transMatrix[1,0] + coords[i,2]*transMatrix[2,0]
        transCoords[i,1] = coords[i,0]*transMatrix[0,1] + coords[i,1]*transMatrix[1,1] + coords[i,2]*transMatrix[2,1]
        transCoords[i,2] = coords[i,0]*transMatrix[0,2] + coords[i,1]*transMatrix[1,2] + coords[i,2]*transMatrix[2,2]
    # return transformed
    return transCoords
    
