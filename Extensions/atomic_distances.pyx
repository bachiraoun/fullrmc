"""
This is a C compiled module to compute atomic inter-molecular distances.
"""                
from libc.math cimport sqrt
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from cython.parallel import prange
from fullrmc.Core.pairs_distances import pairs_distances_to_indexcoords

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
cdef C_INT32   INT32_ZERO      = 0


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x) nogil
    C_FLOAT32 ceil(C_FLOAT32 x)  nogil
    C_FLOAT32 sqrt(C_FLOAT32 x)  nogil
    C_FLOAT32 abs(C_FLOAT32 x)   nogil # not sure why abs(-1.1) = 1 not 1.1, it is rounding. so we won't use it.
    C_FLOAT32 fabs(C_FLOAT32 x)  nogil 

            
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _single_atomic_distances_dists( C_INT32          atomIndex,
                                          C_INT32          atomMoleculeIndex,
                                          C_INT32          atomElementIndex,
                                          C_FLOAT32[:]     distances,
                                          C_INT32[:]       moleculeIndex,
                                          C_INT32[:]       elementIndex,
                                          C_FLOAT32[:,:,:] dintra,
                                          C_FLOAT32[:,:,:] dinter,
                                          C_INT32[:,:,:]   nintra,
                                          C_INT32[:,:,:]   ninter,
                                          C_FLOAT32[:,:,:] lowerLimit,
                                          C_FLOAT32[:,:,:] upperLimit,
                                          bint             interMolecular = True,
                                          bint             intraMolecular = True,
                                          bint             countWithinLimits = True,
                                          bint             reduceDistanceToUpper = False,
                                          bint             reduceDistanceToLower = False,
                                          bint             reduceDistance = False,
                                          bint             allAtoms = True, # added OCT 2016
                                          C_INT32          ncores = 1):
    # declare variables
    cdef C_FLOAT32 distance, upper, lower
    cdef C_INT32 i, startIndex, endIndex
    cdef C_INT32 inLoopMoleculeIndex, inLoopElementIndex
    cdef C_INT32 num_threads = ncores
    # start index
    if allAtoms:
        startIndex = INT32_ZERO
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>distances.shape[0]
    # start openmp loop
    for i in prange(startIndex, endIndex, INT32_ONE, nogil=True, schedule="static", num_threads=num_threads): # added OCT 2016
    #for i in prange(INT32_ZERO, <C_INT32>distances.shape[0], INT32_ONE, nogil=True, schedule="static", num_threads=num_threads):
    #for i from startIndex <= i < endIndex:
        if i == atomIndex: continue
        inLoopMoleculeIndex = moleculeIndex[i]
        # whether atoms are of the same molecule and intramolecular is not needed
        if (not intraMolecular) and (inLoopMoleculeIndex==atomMoleculeIndex):
           continue
        # whether atoms are not of the same molecule and intermolecular is not needed
        if (not interMolecular) and (not inLoopMoleculeIndex==atomMoleculeIndex):
           continue
        # get distance         
        distance = distances[i]
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
            distance = fabs(upper-distance)
        elif reduceDistanceToLower:
            distance = fabs(lower-distance)
        elif reduceDistance:
            if distance > (lower+upper)/FLOAT_TWO:
                distance = fabs(upper-distance)
            else:
                distance = fabs(lower-distance)
        # increment histograms
        #with gil: print atomIndex, atomIndex, lower, upper, distances[i], distance
        if inLoopMoleculeIndex == atomMoleculeIndex:
            dintra[atomElementIndex,inLoopElementIndex,0] += distance
            nintra[atomElementIndex,inLoopElementIndex,0] += INT32_ONE
        else:
            #with gil: print atomIndex, i, atomElementIndex,inLoopElementIndex, ' distance: ',distance, distances[i], ' lower, upper: ',lower, upper, upper-distances[i], abs(upper-distances[i]), fabs(upper-distances[i])
            dinter[atomElementIndex,inLoopElementIndex,0] += distance
            ninter[atomElementIndex,inLoopElementIndex,0] += INT32_ONE
            



            
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atomic_distances_dists_serial( C_INT32                       atomIndex, 
                                          ndarray[C_FLOAT32, ndim=1]    distances not None,
                                          ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                          ndarray[C_INT32, ndim=1]      elementIndex not None,
                                          ndarray[C_FLOAT32, ndim=3]    dintra not None,
                                          ndarray[C_FLOAT32, ndim=3]    dinter not None,
                                          ndarray[C_INT32, ndim=3]      nintra not None,
                                          ndarray[C_INT32, ndim=3]      ninter not None,
                                          np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                                          np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                                          bint                          interMolecular = True,
                                          bint                          intraMolecular = True,
                                          bint                          countWithinLimits = True,
                                          bint                          reduceDistanceToUpper = False,
                                          bint                          reduceDistanceToLower = False,
                                          bint                          reduceDistance = False,
                                          bint                          allAtoms = True):
    """
    Computes the inter-molecular distances constraint of a single atom given a distances array.
    
    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. distances (float32 array): The distances array of the atom with the rest of atoms.
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
    """
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_INT32 binIndex
    cdef C_INT32 atomMoleculeIndex, inLoopMoleculeIndex, atomElementIndex, inLoopElementIndex
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 distance,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get atom molecule and symbol
    atomMoleculeIndex = moleculeIndex[atomIndex]
    atomElementIndex  = elementIndex[atomIndex]
    # start index
    if allAtoms:
        startIndex = <C_INT32>0
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>distances.shape[0]
    # loop
    for i from startIndex <= i < endIndex:
        if i == atomIndex: continue
        inLoopMoleculeIndex = moleculeIndex[i]
        # whether atoms are of the same molecule and intramolecular is not needed
        if (not intraMolecular) and (inLoopMoleculeIndex==atomMoleculeIndex):
           continue
        # whether atoms are not of the same molecule and intermolecular is not needed
        if (not interMolecular) and (not inLoopMoleculeIndex==atomMoleculeIndex):
           continue
        # get distance         
        distance = distances[i]
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
            distance = fabs(upper-distance) # abs is rounding abs(-1.1) = 1 not 1.1
        elif reduceDistanceToLower:
            distance = fabs(lower-distance)
        elif reduceDistance:
            if distance > (lower+upper)/FLOAT_TWO:
                distance = fabs(upper-distance)
            else:
                distance = fabs(lower-distance)
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
def single_atomic_distances_dists( C_INT32                       atomIndex, 
                                   ndarray[C_FLOAT32, ndim=1]    distances not None,
                                   ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                   ndarray[C_INT32, ndim=1]      elementIndex not None,
                                   ndarray[C_FLOAT32, ndim=3]    dintra not None,
                                   ndarray[C_FLOAT32, ndim=3]    dinter not None,
                                   ndarray[C_INT32, ndim=3]      nintra not None,
                                   ndarray[C_INT32, ndim=3]      ninter not None,
                                   np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                                   np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                                   bint                          interMolecular = True,
                                   bint                          intraMolecular = True,
                                   bint                          countWithinLimits = True,
                                   bint                          reduceDistanceToUpper = False,
                                   bint                          reduceDistanceToLower = False,
                                   bint                          reduceDistance = False,
                                   bint                          allAtoms = True,
                                   C_INT32                       ncores = 1 ):
    """
    Computes the inter-molecular distances constraint of a single atom given a distances array.
    
    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. distances (float32 array): The distances array of the atom with the rest of atoms.
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
       #. ncores (int32) [default=1]: The number of cores to use.

    :Returns:
       #. dintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.
    """
    # declare variables
    cdef C_INT32 i
    cdef C_INT32 atomMoleculeIndex, atomElementIndex
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get atom molecule and symbol
    atomMoleculeIndex = moleculeIndex[atomIndex]
    atomElementIndex  = elementIndex[atomIndex]
    # loop
    _single_atomic_distances_dists( atomIndex             = atomIndex,
                                    atomMoleculeIndex     = atomMoleculeIndex,
                                    atomElementIndex      = atomElementIndex,
                                    distances             = distances,
                                    moleculeIndex         = moleculeIndex,
                                    elementIndex          = elementIndex,
                                    dintra                = dintra,
                                    dinter                = dinter,
                                    nintra                = nintra,
                                    ninter                = ninter,
                                    lowerLimit            = lowerLimit,
                                    upperLimit            = upperLimit,
                                    interMolecular        = interMolecular,
                                    intraMolecular        = intraMolecular,
                                    countWithinLimits     = countWithinLimits,
                                    reduceDistanceToUpper = reduceDistanceToUpper,
                                    reduceDistanceToLower = reduceDistanceToLower,
                                    reduceDistance        = reduceDistance,
                                    allAtoms              = allAtoms,
                                    ncores                = ncores)
            
            
            
            
            
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multiple_atomic_distances_coords( ndarray[C_INT32, ndim=1]      indexes not None,
                                      np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                      np.ndarray[C_FLOAT32, ndim=2] basis not None,
                                      bint                          isPBC,
                                      ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                      ndarray[C_INT32, ndim=1]      elementIndex not None,
                                      C_INT32                       numberOfElements,
                                      np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                                      np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                                      bint                          interMolecular = True,
                                      bint                          intraMolecular = True,
                                      bint                          countWithinLimits = True,
                                      bint                          reduceDistanceToUpper = False,
                                      bint                          reduceDistanceToLower = False,
                                      bint                          reduceDistance = False,
                                      bint                          allAtoms=True,
                                      C_INT32                       ncores = 1 ):    
    """
    Computes multiple atoms inter-molecular distances constraint given coordinates.
    
    :Arguments:
       #. indexes (int32 array): The atoms indexes array.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
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
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. dintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.

    """
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
        distances = pairs_distances_to_indexcoords( atomIndex = i, 
                                                    coords    = boxCoords,
                                                    basis     = basis,
                                                    isPBC     = isPBC,
                                                    allAtoms  = allAtoms,
                                                    ncores    = ncores)    
        # compute single atomic distances 
        single_atomic_distances_dists( atomIndex             = i, 
                                       distances             = distances,
                                       moleculeIndex         = moleculeIndex,
                                       elementIndex          = elementIndex,
                                       dintra                = dintra,
                                       dinter                = dinter,
                                       nintra                = nintra,
                                       ninter                = ninter,
                                       lowerLimit            = lowerLimit,
                                       upperLimit            = upperLimit,
                                       interMolecular        = interMolecular,
                                       intraMolecular        = intraMolecular,
                                       countWithinLimits     = countWithinLimits,
                                       reduceDistance        = reduceDistance,
                                       reduceDistanceToLower = reduceDistanceToLower,
                                       reduceDistanceToUpper = reduceDistanceToUpper,
                                       allAtoms              = allAtoms,
                                       ncores                = ncores )
    return nintra, dintra, ninter, dinter
    


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multiple_atomic_distances_dists( ndarray[C_INT32, ndim=1]      indexes not None,
                                     np.ndarray[C_FLOAT32, ndim=2] distances not None,
                                     ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                     ndarray[C_INT32, ndim=1]      elementIndex not None,
                                     C_INT32                       numberOfElements,
                                     np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                                     np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                                     bint                          interMolecular = True,
                                     bint                          intraMolecular = True,
                                     bint                          countWithinLimits = True,
                                     bint                          reduceDistanceToUpper = False,
                                     bint                          reduceDistanceToLower = False,
                                     bint                          reduceDistance = False,
                                     bint                          allAtoms=True,
                                     C_INT32                       ncores = 1 ):    
    """
    Computes multiple atoms inter-molecular distances constraint given distances.
    
    :Arguments:
       #. indexes (int32 array): The atoms indexes array.
       #. distances (float32 array): The distances array of the atoms with the rest of atoms.
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
       #. ncores (int32) [default=1]: The number of cores to use.
       
    :Returns:
       #. dintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.
    """
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
    for i from <C_INT32>0 <= i < <C_INT32>indexes.shape[0]:
        single_atomic_distances_dists( atomIndex             = indexes[i], 
                                       distances             = distances[:,i],
                                       moleculeIndex         = moleculeIndex,
                                       elementIndex          = elementIndex,
                                       dintra                = dintra,
                                       dinter                = dinter,
                                       nintra                = nintra,
                                       ninter                = ninter,
                                       lowerLimit            = lowerLimit,
                                       upperLimit            = upperLimit,
                                       interMolecular        = interMolecular,
                                       intraMolecular        = intraMolecular,
                                       countWithinLimits     = countWithinLimits,
                                       reduceDistance        = reduceDistance,
                                       reduceDistanceToLower = reduceDistanceToLower,
                                       reduceDistanceToUpper = reduceDistanceToUpper,
                                       allAtoms              = allAtoms,
                                       ncores                = ncores )
    return nintra, dintra, ninter, dinter


    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_atomic_distances_coords( np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                  np.ndarray[C_FLOAT32, ndim=2] basis not None,
                                  bint                          isPBC,
                                  ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                  ndarray[C_INT32, ndim=1]      elementIndex not None,
                                  C_INT32                       numberOfElements,
                                  np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                                  np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                                  bint                          interMolecular=True,
                                  bint                          intraMolecular=True,
                                  bint                          reduceDistanceToUpper=False,
                                  bint                          reduceDistanceToLower=False,
                                  bint                          reduceDistance=False,
                                  bint                          countWithinLimits=True,
                                  C_INT32                       ncores = 1):    
    """
    Computes all atoms inter-molecular distances constraint given coordinates.
    
    :Arguments:
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
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
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. dintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.
    """
    # get number of atoms
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange( <C_INT32>boxCoords.shape[0], dtype=NUMPY_INT32)
    # calculate histograms
    return multiple_atomic_distances_coords( indexes               = indexes,
                                             boxCoords             = boxCoords,
                                             basis                 = basis,
                                             isPBC                 = isPBC,
                                             moleculeIndex         = moleculeIndex,
                                             elementIndex          = elementIndex,
                                             numberOfElements      = numberOfElements,
                                             lowerLimit            = lowerLimit,
                                             upperLimit            = upperLimit,
                                             interMolecular        = interMolecular,
                                             intraMolecular        = intraMolecular,
                                             countWithinLimits     = countWithinLimits,
                                             reduceDistance        = reduceDistance,
                                             reduceDistanceToLower = reduceDistanceToLower,
                                             reduceDistanceToUpper = reduceDistanceToUpper,
                                             allAtoms              = False,
                                             ncores                = ncores)
                                           
   
 
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_atomic_distances_dists( np.ndarray[C_FLOAT32, ndim=2] distances not None,
                                 ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                 ndarray[C_INT32, ndim=1]      elementIndex not None,
                                 C_INT32                       numberOfElements,
                                 np.ndarray[C_FLOAT32, ndim=3] lowerLimit not None,
                                 np.ndarray[C_FLOAT32, ndim=3] upperLimit not None,
                                 bint                          interMolecular=True,
                                 bint                          intraMolecular=True,
                                 bint                          reduceDistanceToUpper=False,
                                 bint                          reduceDistanceToLower=False,
                                 bint                          reduceDistance=False,
                                 bint                          countWithinLimits=True):    
    """
    Computes all atoms inter-molecular distances constraint given distances.
    
    :Arguments:
       #. distances (float32 array): The distances array of the atoms with the rest of atoms.
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
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. dintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted distances.
       #. dinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted distances.
       #. nintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular counted elements.
       #. ninter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular counted elements.
    """
    # get number of atoms
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(<C_INT32>distances.shape[1], dtype=NUMPY_INT32)
    # calculate histograms
    return multiple_atomic_distances_dists( indexes               = indexes,
                                            distances             = distances,
                                            moleculeIndex         = moleculeIndex,
                                            elementIndex          = elementIndex,
                                            numberOfElements      = numberOfElements,
                                            lowerLimit            = lowerLimit,
                                            upperLimit            = upperLimit,
                                            interMolecular        = interMolecular,
                                            intraMolecular        = intraMolecular,
                                            countWithinLimits     = countWithinLimits,
                                            reduceDistance        = reduceDistance,
                                            reduceDistanceToLower = reduceDistanceToLower,
                                            reduceDistanceToUpper = reduceDistanceToUpper,
                                            allAtoms              = False )





@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _pair_elements_stats( C_INT32[:]     elementIndex,
                                C_INT32[:]     moleculeIndex,
                                C_INT32        numberOfElements,
                                C_INT32[:,:,:] nintra,
                                C_INT32[:,:,:] ninter):
    # declare variables
    cdef C_INT32 atomMoleculeIndex, inLoopMoleculeIndex, atomElementIndex, inLoopElementIndex    
    # double loops
    for i from <C_INT32>0 <= i < <C_INT32>elementIndex.shape[0]-1:
        atomElementIndex  =  elementIndex[i]
        atomMoleculeIndex =  moleculeIndex[i]
        for j from i+1 <= j < <C_INT32>elementIndex.shape[0]:
            inLoopMoleculeIndex = moleculeIndex[j]
            inLoopElementIndex  = elementIndex[j]
            # increment histograms
            if inLoopMoleculeIndex == atomMoleculeIndex:
                nintra[atomElementIndex,inLoopElementIndex,0] += INT32_ONE
            else:
                ninter[atomElementIndex,inLoopElementIndex,0] += INT32_ONE 



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pair_elements_stats( ndarray[C_INT32, ndim=1] elementIndex not None,
                         ndarray[C_INT32, ndim=1] moleculeIndex not None,
                         C_INT32                  numberOfElements):
    # declare variables
    cdef ndarray[C_INT32,    mode="c", ndim=3] nintra = np.zeros((numberOfElements,numberOfElements,1), dtype=NUMPY_INT32)
    cdef ndarray[C_INT32,    mode="c", ndim=3] ninter = np.zeros((numberOfElements,numberOfElements,1), dtype=NUMPY_INT32)
    # compute
    _pair_elements_stats(elementIndex     = elementIndex,
                         moleculeIndex    = moleculeIndex,
                         numberOfElements = numberOfElements,
                         nintra           = nintra,
                         ninter           = ninter)
    # return 
    return nintra, ninter
    
            
            
            
                                     