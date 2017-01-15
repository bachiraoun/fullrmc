"""
This is a C compiled module to compute atomic bonds.
"""                      
from libc.math cimport sqrt, fabs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from fullrmc.Core.pairs_distances import pairs_distances_to_point, point_to_point_distance

# declare types
NUMPY_FLOAT32 = np.float32
NUMPY_INT32   = np.int32
ctypedef np.float32_t C_FLOAT32
ctypedef np.int32_t   C_INT32

# declare constants
cdef C_FLOAT32 FLOAT_ZERO      = 0.0
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_FLOAT32 BOX_LENGTH      = 1.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5
cdef C_INT32   INT32_ZERO      = 0
cdef C_INT32   INT32_ONE       = 1




@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef C_FLOAT32 _single_bond(C_FLOAT32 bondLength,
                            C_FLOAT32 lower,
                            C_FLOAT32 upper,
                            bint      reduceDistanceToUpper = False,
                            bint      reduceDistanceToLower = False ):
    """
    It calculates the bonds constraint of a distances array.
    
    :Arguments:
       #. bondLength (float32): The bond distance
       #. lower (float32): The bond lower limit or minimum bond length allowed.
       #. upper (float32): The bond upper limit or maximum bond length allowed.
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
                  
    :Returns:
       #. reducedBond: The reduced bond length
    """
    # declare variables
    cdef C_FLOAT32 reducedBond
    # compute reducedBond
    if bondLength>=lower and bondLength<=upper:
        reducedBond = FLOAT_ZERO     
    elif reduceDistanceToUpper:
        reducedBond = fabs(upper-bondLength)
    elif reduceDistanceToLower:
        reducedBond = fabs(lower-bondLength)
    else:
        if bondLength > (lower+upper)/FLOAT_TWO:
            reducedBond = fabs(upper-bondLength)
        else:
            reducedBond = fabs(lower-bondLength)
    # return
    return reducedBond
    
 
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_bonds_coords( ndarray[C_INT32, ndim=1]   idx1 not None,
                       ndarray[C_INT32, ndim=1]   idx2 not None,
                       ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                       ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                       ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                       ndarray[C_FLOAT32, ndim=2] basis not None,
                       bint                       isPBC,
                       bint                       reduceDistanceToUpper = False,
                       bint                       reduceDistanceToLower = False,
                       C_INT32                    ncores = 1):    
    """
    It calculates the bonds constraint of box coordinates.
    
    :Arguments:
       #. idx1 (int32 (n,) numpy.ndarray): First atoms index array
       #. idx2 (int32 (n,) numpy.ndarray): Second atoms index array
       #. lowerLimit (float32 (n,) numpy.ndarray): Lower limit or minimum bond length allowed.
       #. upperLimit (float32 (n,) numpy.ndarray): Upper limit or minimum bond length allowed.
       #. boxCoords (float32 (n,3) numpy.ndarray): The atomic coordinates array.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.       
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. bondsLength: The calculated bonds length
       #. reducedLengths: The calculated reduced bonds length
    """
    cdef C_INT32 i, numberOfIndexes
    cdef C_FLOAT32 bondLength, reducedLength
    numberOfIndexes = <C_INT32>len(lowerLimit)
    # create bondsLength and reducedBonds list
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] bondsLength    = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedLengths = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32) 
    # loop atoms
    for i from 0 <= i < numberOfIndexes:
        bondLength  = point_to_point_distance( point1 = boxCoords[ idx1[i],: ], 
                                               point2 = boxCoords[ idx2[i],: ], 
                                               basis  = basis,
                                               isPBC  = isPBC)
        reducedLength = _single_bond(bondLength            = bondLength,
                                     lower                 = lowerLimit[i],
                                     upper                 = upperLimit[i],
                                     reduceDistanceToUpper = reduceDistanceToUpper,
                                     reduceDistanceToLower = reduceDistanceToLower )
        # append lists
        bondsLength[i]    = bondLength
        reducedLengths[i] = reducedLength
    # return bondsLength and reducedRistances
    return bondsLength, reducedLengths    
    
    
    






