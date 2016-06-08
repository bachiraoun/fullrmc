"""
This is a C compiled module to compute atomic bonds.
"""                      
from libc.math cimport sqrt, abs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from fullrmc.Core.pairs_distances import pairs_distances_to_point

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
def single_bonds_dists( ndarray[C_FLOAT32, ndim=1]    distances not None,
                        np.ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                        np.ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                        bint                          reduceDistanceToUpper = False,
                        bint                          reduceDistanceToLower = False):
    """
    It calculates the bonds constraint of a distances array.
    
    :Arguments:
       #. distances (float32 array): The distances array.
       #. lowerLimit (float32 array): The (numberOfBondedAtoms) array for lower limit or minimum bond length allowed.
       #. upperLimit (float32 array): The (numberOfBondedAtoms) array for upper limit or maximum bond length allowed.
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
                  
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. bondsLength: The calculated bonds length
          #. reducedDistances: The reduced bonds length
    """
    # declare variables
    cdef C_INT32 i, numberOfIndexes
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 distance, reducedDistance
    # create bondsLength and reducedDistances
    numberOfIndexes = <C_INT32>len(distances)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] bondsLength      = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedDistances = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    # loop
    for i from 0 <= i < numberOfIndexes:
        # get distance         
        distance = distances[i]
        # compute reduced distance
        lower = lowerLimit[i]
        upper = upperLimit[i]
        if distance>=lower and distance<=upper:
            reducedDistance = FLOAT_ZERO     
        elif reduceDistanceToUpper:
            reducedDistance = abs(upper-distance)
        elif reduceDistanceToLower:
            reducedDistance = abs(lower-distance)
        else:
            if distance > (lower+upper)/FLOAT_TWO:
                reducedDistance = abs(upper-distance)
            else:
                reducedDistance = abs(lower-distance)
        # increment histograms
        bondsLength[i]      = distance
        reducedDistances[i] = reducedDistance
    # return result
    return {"bondsLength":bondsLength, "reducedDistances":reducedDistances}

    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_bonds_coords( dict                       bonds not None, 
                       ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                       ndarray[C_FLOAT32, ndim=2] basis not None,
                       bint                       isPBC,
                       bint                       reduceDistanceToUpper = False,
                       bint                       reduceDistanceToLower = False,
                       C_INT32                    ncores = 1):    
    """
    It calculates the bonds constraint of box coordinates.
    
    :Arguments:
       #. bonds (python dictionary): The bonds dictionary. Where keys are atoms indexes and values are dictionary of bondedIndexes array, lowerLimit array, upperLimit array
       #. boxCoords (float32 (n,3) numpy.ndarray): The atomic coordinates array.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.       
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. bondsLength: The calculated bonds length
          #. reducedDistances: The reduced bonds length
    """
    # create bondsLength and reducedDistances list
    bondsResult = {}
    # loop atoms
    for atomIndex, bond in bonds.items():
        distances = pairs_distances_to_point( point  = boxCoords[ atomIndex ], 
                                              coords = boxCoords[ bond["indexes"] ],
                                              basis  = basis,
                                              isPBC  = isPBC,
                                              ncores = ncores)  
        result = single_bonds_dists( distances             = distances, 
                                     lowerLimit            = bond["lower"],
                                     upperLimit            = bond["upper"],
                                     reduceDistanceToUpper = reduceDistanceToUpper,
                                     reduceDistanceToLower = reduceDistanceToLower)
        # append lists
        bondsResult[atomIndex] = result
    return bondsResult     
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_bonds_dists( dict                       bonds not None, 
                      ndarray[C_FLOAT32, ndim=2] distances not None,
                      bint                       reduceDistanceToUpper = False,
                      bint                       reduceDistanceToLower = False):    
    """
    It calculates the bonds constraint given atomic distances.
    
    :Arguments:
       #. bonds (python dictionary): The bonds dictionary. Where keys are atoms indexes and values are dictionary of bondedIndexes array, lowerLimit array, upperLimit array
       #. distances (float32 (n,1) numpy.ndarray): The atomic distances array.
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
       
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. bondsLength: The calculated bonds length
          #. reducedDistances: The reduced bonds length
    """
    cdef C_INT32 i=INT32_ZERO
    # create bondsLength and reducedDistances list
    bondsResult = {}
    # loop atoms
    for atomIndex, bond in bonds.items():
        result = single_bonds_dists( distances             = distances[:,i], 
                                     lowerLimit            = bond["lower"],
                                     upperLimit            = bond["upper"],
                                     reduceDistanceToUpper = reduceDistanceToUpper,
                                     reduceDistanceToLower = reduceDistanceToLower)
        # append lists
        bondsResult[atomIndex] = result
        # increment i
        i += INT32_ONE
    return bondsResult        
    
    
    
    
  
    