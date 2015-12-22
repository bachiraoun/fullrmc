"""
This is a C compiled Cython generated module to compute reciprocal basis.
                   
**get_reciprocal_basis**: It computes the reciprocal basis vectors array.
    :Arguments:
       #. basis (float32 array): The (3,3) basis vectors array.
                                         
    :Returns:
       #. rbasis (float32 array): The (3,3) normalized with the basis volume reciprocal basis vectors array.
       #. volume (float32): The basis volume.
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
def get_reciprocal_basis( np.ndarray[C_FLOAT32, ndim=2] basis not None):
    # declare rbasis
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] rbasis = np.empty((3,3), dtype=NUMPY_FLOAT32)
    # get reciprocal basis
    rbasis[0,0] = basis[1,1]*basis[2,2] - basis[2,1]*basis[1,2]
    rbasis[0,1] = - ( basis[0,1]*basis[2,2] - basis[2,1]*basis[0,1] )
    rbasis[0,2] = basis[0,1]*basis[1,2] - basis[1,1]*basis[0,2]
    rbasis[1,0] = - ( basis[1,0]*basis[2,2] - basis[2,0]*basis[1,2] )
    rbasis[1,1] = basis[0,0]*basis[2,2] - basis[2,0]*basis[0,2]
    rbasis[1,2] = - ( basis[0,0]*basis[1,2] - basis[1,0]*basis[0,2] )
    rbasis[2,0] = basis[1,0]*basis[2,1] - basis[2,0]*basis[1,1]
    rbasis[2,1] = - ( basis[0,0]*basis[2,1] - basis[2,0]*basis[0,1] )
    rbasis[2,2] = basis[0,0]*basis[1,1] - basis[1,0]*basis[0,1]
    # find volume
    vol = basis[0,0]*rbasis[0,0] + basis[1,0]*rbasis[0,1] + basis[2,0]*rbasis[0,2]
    # normalize with volume
    rbasis[0,0] = rbasis[0,0]/vol
    rbasis[0,1] = rbasis[0,1]/vol
    rbasis[0,2] = rbasis[0,2]/vol
    rbasis[1,0] = rbasis[1,0]/vol
    rbasis[1,1] = rbasis[1,1]/vol
    rbasis[1,2] = rbasis[1,2]/vol
    rbasis[2,0] = rbasis[2,0]/vol
    rbasis[2,1] = rbasis[2,1]/vol
    rbasis[2,2] = rbasis[2,2]/vol
    # return rbasis
    return rbasis, vol


   
    