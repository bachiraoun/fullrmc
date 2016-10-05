"""
This is a C compiled module to compute 
boundary conditions related calculations
"""

from libc.math cimport sqrt
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


# declare constants
cdef C_FLOAT32 BOX_LENGTH      = 1.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5
cdef C_FLOAT32 FLOAT32_ZERO    = 0.0
cdef C_FLOAT32 FLOAT32_ONE     = 1.0
cdef C_INT32   INT32_ONE       = 1


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x)
    C_FLOAT32 ceil(C_FLOAT32 x)
    C_FLOAT32 sqrt(C_FLOAT32 x)

cdef inline C_FLOAT32 round(C_FLOAT32 num):
    return floor(num + HALF_BOX_LENGTH) if (num > FLOAT32_ZERO) else ceil(num - HALF_BOX_LENGTH)
    



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def get_reciprocal_basis( np.ndarray[C_FLOAT32, ndim=2] basis not None):
    """
    Computes reciprocal box matrix.
    
    :Arguments:
       #. basis (float32 array): The (3,3) box matrix
                                         
    :Returns:
       #. rbasis (float32 array): The (3,3) reciprocal box matrix.
    """
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
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def transform_coordinates( np.ndarray[C_FLOAT32, ndim=2] transMatrix not None,
                           np.ndarray[C_FLOAT32, ndim=2] coords not None):
    """
    Transforms coordinates array using a transformation matrix.
    
    :Arguments:
       #. transMatrix (float32 array): The (3,3) transformation matrix
       #. coords (float32 array): The (N,3) coordinates array.
                                         
    :Returns:
       #. transCoords (float32 array): The (N,3) transformed coordinates array.
    """
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
    


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def box_coordinates_real_distances( C_INT32                       atomIndex, 
                                    ndarray[C_INT32, ndim=1]      indexes not None,
                                    np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                    np.ndarray[C_FLOAT32, ndim=2] basis not None):
                           
    """
    Computes atomic real distances given box coordinates.
    
    :Arguments:
       #. atomIndex (int32): The index of atom to compute the distance from.
       #. indexes (int32 array): The list of atom indexes to compute the distance to
       #. boxCoords (float32 array): The (N,3) box coordinates array.
       #. basis (float32 array): The (3,3) box matrix
       
    :Returns:
       #. distances (float32 array): The (N,) distances array.
    """
    # declare variables
    cdef C_INT32 i, ii
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz, distance,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # declare transCoords
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] distances = np.empty((indexes.shape[0],), dtype=NUMPY_FLOAT32)
    # loop
    ii = -1
    for i in indexes:
        ii += 1
        # calculate difference
        box_dx = boxCoords[i,0]-atomBox_x
        box_dy = boxCoords[i,1]-atomBox_y
        box_dz = boxCoords[i,2]-atomBox_z
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # calculate distance         
        distances[ii] = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
    # return distances
    return distances
        
        

        