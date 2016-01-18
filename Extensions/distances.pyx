"""
This is a C compiled Cython generated module to calculate distances constraints. 
It contains the following methods.
                   
**single_distances**: It calculates the distances constraint of a single atom.
    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. dintra (float32 array): The (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.
       #. lowerLimit (float32 array): The (numberOfElements,numberOfElements,1) array of lower distance limits.
       #. upperLimit (float32 array): The (numberOfElements,numberOfElements,1) array of upper distance limits.
       #. interMolecular (bool): Whether to consider inter-molecular distances. DEFAULT: True
       #. intraMolecular (bool): Whether to consider intra-molecular distances. DEFAULT: True
       #. countWithinLimits (bool): Whether to count distances and atoms found within the lower and upper limits or outside. DEFAULT: True
       #. reduceDistanceToUpper (bool): Whether to reduce counted distances to the difference between the found distance and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce counted distances to the difference between the found distance and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
       #. reduceDistance (bool): Whether to reduce counted distances to the difference between the found distance and the closest limit. When True, this flag may lose its priority if any of reduceDistanceToLower or reduceDistanceToUpper is True. DEFAULT: False
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True

    :Returns:
       #. dintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.

**multiple_distances**: It calculates the distances constraint of multiple atoms. It creates the inter and intra-molecular histogram distances and numbers arrays and calls single_distances method for every desired atom index.
    :Arguments:
       #. indexes (int32 array): The atoms indexes array.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. lowerLimit (float32 array): The (numberOfElements,numberOfElements,1) array of lower distance limits.
       #. upperLimit (float32 array): The (numberOfElements,numberOfElements,1) array of upper distance limits.
       #. interMolecular (bool): Whether to consider inter-molecular distances. DEFAULT: True
       #. intraMolecular (bool): Whether to consider intra-molecular distances. DEFAULT: True
       #. countWithinLimits (bool): Whether to count distances and atoms found within the lower and upper limits or outside.
       #. reduceDistanceToUpper (bool): Whether to reduce counted distances to the difference between the found distance and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce counted distances to the difference between the found distance and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
       #. reduceDistance (bool): Whether to reduce counted distances to the difference between the found distance and the closest limit. When True, this flag may lose its priority if any of reduceDistanceToLower or reduceDistanceToUpper is True. DEFAULT: False
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True
                     
    :Returns:
       #. dintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.

**full_distances**: It calculates the distances constraint for all atoms. It calls multiple_distances method for all atoms.
    :Arguments:
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. lowerLimit (float32 array): The (numberOfElements,numberOfElements,1) array of lower distance limits.
       #. upperLimit (float32 array): The (numberOfElements,numberOfElements,1) array of upper distance limits.
       #. interMolecular (bool): Whether to consider inter-molecular distances. DEFAULT: True
       #. intraMolecular (bool): Whether to consider intra-molecular distances. DEFAULT: True
       #. countWithinLimits (bool): Whether to count distances and atoms found within the lower and upper limits or outside. DEFAULT: True
       #. reduceDistanceToUpper (bool): Whether to reduce counted distances to the difference between the found distance and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceDistanceToLower (bool): Whether to reduce counted distances to the difference between the found distance and the lower limit. When True, this flag may lose its priority for reduceDistanceToUpper if the later is True. DEFAULT: False
       #. reduceDistance (bool): Whether to reduce counted distances to the difference between the found distance and the closest limit. When True, this flag may lose its priority if any of reduceDistanceToLower or reduceDistanceToUpper is True. DEFAULT: False
        
    :Returns:
       #. dintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.
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
cdef C_FLOAT32 BOX_LENGTH      = 1.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5
cdef C_FLOAT32 FLOAT_ZERO      = 0.0
cdef C_FLOAT32 FLOAT_ONE       = 1.0
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_INT32   INT32_ONE       = 1


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
def single_distances( C_INT32 atomIndex, 
                      ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                      ndarray[C_FLOAT32, ndim=2] basis not None,
                      ndarray[C_INT32, ndim=1] moleculeIndex not None,
                      ndarray[C_INT32, ndim=1] elementIndex not None,
                      ndarray[C_FLOAT32, ndim=3] dintra not None,
                      ndarray[C_FLOAT32, ndim=3] dinter not None,
                      ndarray[C_INT32, ndim=3] nintra not None,
                      ndarray[C_INT32, ndim=3] ninter not None,
                      np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                      np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                      bint interMolecular = True,
                      bint intraMolecular = True,
                      bint countWithinLimits = True,
                      bint reduceDistanceToUpper = False,
                      bint reduceDistanceToLower = False,
                      bint reduceDistance = False,
                      bint allAtoms = True ):
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_INT32 binIndex
    cdef C_INT32 atomMoleculeIndex, inLoopMoleculeIndex, atomElementIndex, inLoopElementIndex
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz, distance,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # get atom molecule and symbol
    atomMoleculeIndex  = moleculeIndex[atomIndex]
    atomElementIndex = elementIndex[atomIndex]
    # start index
    if allAtoms:
        startIndex = <C_INT32>0
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>boxCoords.shape[0]
    # loop
    for i from startIndex <= i < endIndex:
        if i == atomIndex: continue
        inLoopMoleculeIndex = moleculeIndex[i]
        # whether atoms are of the same molecule and intramolecular is not needed
        if not intraMolecular and inLoopMoleculeIndex==atomMoleculeIndex:
           continue
        # whether atoms are not of the same molecule and intermolecular is not needed
        if not interMolecular and not inLoopMoleculeIndex==atomMoleculeIndex:
           continue
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
        distance = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
        # get in loop element index
        inLoopElementIndex = elementIndex[i]
        # check limits
        lower = lowerLimit[inLoopElementIndex,atomElementIndex,0]
        upper = upperLimit[inLoopElementIndex,atomElementIndex,0]
        if countWithinLimits:
            if distance<lower:
                continue
            if distance>=upper:
                continue
        elif (distance>=lower) and (distance<upper):
                continue
        # reduce distance to the smaller difference between distance and limits.
        if reduceDistanceToUpper:
            distance = abs(upper-distance)
        elif reduceDistanceToLower:
            distance = abs(lower-distance)
        elif reduceDistance:
            if distance > (lower+upper)/FLOAT_TWO:
                distance = abs(upper-distance)
            else:
                distance = abs(lower-distance)
        # increment histograms
        #print startIndex, i, inLoopElementIndex,atomElementIndex, lower, upper,  <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz), distance
        if inLoopMoleculeIndex == atomMoleculeIndex:
            dintra[atomElementIndex,inLoopElementIndex,0] += distance
            nintra[atomElementIndex,inLoopElementIndex,0] += INT32_ONE
            #print startIndex, i, atomElementIndex, inLoopElementIndex, lower, upper,  <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz), distance
        else:
            dinter[atomElementIndex,inLoopElementIndex,0] += distance
            ninter[atomElementIndex,inLoopElementIndex,0] += INT32_ONE



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multiple_distances( ndarray[C_INT32, ndim=1] indexes not None,
                        np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                        np.ndarray[C_FLOAT32, ndim=2] basis not None,
                        ndarray[C_INT32, ndim=1] moleculeIndex not None,
                        ndarray[C_INT32, ndim=1] elementIndex not None,
                        C_INT32 numberOfElements,
                        np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                        np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                        bint interMolecular=True,
                        bint intraMolecular=True,
                        bint countWithinLimits = True,
                        bint reduceDistanceToUpper = False,
                        bint reduceDistanceToLower = False,
                        bint reduceDistance = False,
                        bint allAtoms=True ):    
    # declare variables
    cdef C_INT32 i, ii
    # check lowerLimit array size
    shape = lowerLimit.shape
    assert shape[0] == shape[1] and shape[0]==numberOfElements, "lowerLimit array must have numberOfElements columns and numberOfElements rows"
    assert shape[2] == 1, "lowerLimit array third dimension must have a length of exactly 1"
    # create histograms
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] dintra = np.zeros((numberOfElements,numberOfElements,1), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] dinter = np.zeros((numberOfElements,numberOfElements,1), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_INT32,    mode="c", ndim=3] nintra = np.zeros((numberOfElements,numberOfElements,1), dtype=NUMPY_INT32)
    cdef ndarray[C_INT32,    mode="c", ndim=3] ninter = np.zeros((numberOfElements,numberOfElements,1), dtype=NUMPY_INT32)
    # loop atoms
    for i in indexes:
    #for i in prange(boxCoords.shape[0]-1):
        single_distances( atomIndex = i, 
                          boxCoords = boxCoords,
                          basis = basis,
                          moleculeIndex = moleculeIndex,
                          elementIndex = elementIndex,
                          dintra = dintra,
                          dinter = dinter,
                          nintra = nintra,
                          ninter = ninter,
                          lowerLimit = lowerLimit,
                          upperLimit = upperLimit,
                          interMolecular = interMolecular,
                          intraMolecular = intraMolecular,
                          countWithinLimits = countWithinLimits,
                          reduceDistance = reduceDistance,
                          reduceDistanceToLower = reduceDistanceToLower,
                          reduceDistanceToUpper = reduceDistanceToUpper,
                          allAtoms = allAtoms )
    return nintra, dintra, ninter, dinter
    


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_distances( np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                    np.ndarray[C_FLOAT32, ndim=2] basis not None,
                    ndarray[C_INT32, ndim=1] moleculeIndex not None,
                    ndarray[C_INT32, ndim=1] elementIndex not None,
                    C_INT32 numberOfElements,
                    np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                    np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                    bint interMolecular=True,
                    bint intraMolecular=True,
                    bint reduceDistanceToUpper=False,
                    bint reduceDistanceToLower=False,
                    bint reduceDistance=False,
                    bint countWithinLimits=True):    
    # get number of atoms
    cdef numberOfAtoms = <C_INT32>boxCoords.shape[0]
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(numberOfAtoms, dtype=NUMPY_INT32)
    # calculate histograms
    return multiple_distances( indexes=indexes,
                               boxCoords = boxCoords,
                               basis = basis,
                               moleculeIndex = moleculeIndex,
                               elementIndex = elementIndex,
                               numberOfElements = numberOfElements,
                               lowerLimit = lowerLimit,
                               upperLimit = upperLimit,
                               interMolecular = interMolecular,
                               intraMolecular = intraMolecular,
                               countWithinLimits = countWithinLimits,
                               reduceDistance = reduceDistance,
                               reduceDistanceToLower = reduceDistanceToLower,
                               reduceDistanceToUpper = reduceDistanceToUpper,
                               allAtoms=False )
                                           
   
    