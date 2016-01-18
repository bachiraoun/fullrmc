"""
This is a C compiled Cython generated module to calculate bonds constraints. It contains the following methods.
       
**single_bonds**: It calculates the bonds constraint of a single atom.
    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. bondedIndexes (int32 array): The bonded atoms indexes array.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. lowerLimit (float32 array): The (numberOfBondedAtoms) array for lower limit or minimum bond length allowed.
       #. upperLimit (float32 array): The (numberOfBondedAtoms) array for upper limit or maximum bond length allowed.
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
                  
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. bondsLength: The calculated bonds length
          #. reducedDistances: The reduced bonds length
           
**full_bonds**: It calculates the bonds constraint of all atoms given a bonds dictionary.
    :Arguments:
       #. bonds (python dictionary): The bonds dictionary. Where keys are atoms indexes and values are dictionary of bondedIndexes array, lowerLimit array, upperLimit array
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. reduceDistanceToUpper (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce bonds length found out of limits to the difference between the bond length and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
                
    :Returns:
       #. result (python dictionary): where keys are atomsIndexes and values are dictionaries of exactly two keys as such.\n
          #. bondsLength: The calculated bonds length
          #. reducedDistances: The reduced bonds length
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

# declare constants
cdef C_FLOAT32 FLOAT_ZERO      = 0.0
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_FLOAT32 BOX_LENGTH      = 1.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5



cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x)
    C_FLOAT32 ceil(C_FLOAT32 x)
    C_FLOAT32 sqrt(C_FLOAT32 x)

cdef inline C_FLOAT32 round(C_FLOAT32 num):
    return floor(num + HALF_BOX_LENGTH) if (num > FLOAT_ZERO) else ceil(num - HALF_BOX_LENGTH)
    


       
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_bonds( C_INT32 atomIndex,
                  ndarray[C_INT32, ndim=1] bondedIndexes not None, 
                  ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                  ndarray[C_FLOAT32, ndim=2] basis not None,
                  np.ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                  np.ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                  bint reduceDistanceToUpper = False,
                  bint reduceDistanceToLower = False):
    # declare variables
    cdef C_INT32 i, numberOfIndexes, inLoopAtomIndex
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz, distance, reducedDistance
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get number of bonded indexes
    numberOfIndexes = <C_INT32>len(bondedIndexes)
    # create bondsLength and reducedDistances
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] bondsLength      = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedDistances = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # loop
    for i from 0 <= i < numberOfIndexes:
        # get inLoopAtomIndex
        inLoopAtomIndex = bondedIndexes[i]
        # calculate difference
        box_dx = boxCoords[inLoopAtomIndex,0]-atomBox_x
        box_dy = boxCoords[inLoopAtomIndex,1]-atomBox_y
        box_dz = boxCoords[inLoopAtomIndex,2]-atomBox_z
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # calculate distance         
        distance = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
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
def full_bonds( dict bonds not None, 
                ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                ndarray[C_FLOAT32, ndim=2] basis not None,
                bint reduceDistanceToUpper = False,
                bint reduceDistanceToLower = False):    
    # create bondsLength and reducedDistances list
    bondsResult = {}
    # loop atoms
    for atomIndex, bond in bonds.items():
        result = single_bonds( atomIndex = atomIndex,
                               bondedIndexes = bond["indexes"], 
                               boxCoords = boxCoords,
                               basis = basis,
                               lowerLimit = bond["lower"],
                               upperLimit = bond["upper"],
                               reduceDistanceToUpper = reduceDistanceToUpper,
                               reduceDistanceToLower = reduceDistanceToLower)
        # append lists
        bondsResult[atomIndex] = result
    return bondsResult
    

                                           
   
    