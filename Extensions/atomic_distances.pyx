"""
"""                 
from libc.math cimport sqrt, abs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from cython.parallel import prange

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
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_INT32   INT32_ZERO      = 0
cdef C_INT32   INT32_ONE       = 1


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x) nogil
    C_FLOAT32 ceil(C_FLOAT32 x) nogil
    C_FLOAT32 sqrt(C_FLOAT32 x) nogil

cdef inline C_FLOAT32 round(C_FLOAT32 num) nogil:
    return floor(num + HALF_BOX_LENGTH) if (num > FLOAT32_ZERO) else ceil(num - HALF_BOX_LENGTH)

    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def real_distances( C_INT32 atomIndex, 
                    ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                    ndarray[C_FLOAT32, ndim=2] basis not None,
                    ndarray[C_FLOAT32, ndim=1] output not None,
                    bint allAtoms = True):
    # declare variables
    cdef C_INT32 i, 
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # start index
    if allAtoms:
        startIndex = INT32_ZERO
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>boxCoords.shape[0]
    # loop
    for i from startIndex <= i < endIndex:
        ### TO BE VERIFIED WHETHER IT IS FASTER OR NOT TO KEEP OR REMVOVE THE IF CONDITION ###
        #if i == atomIndex:
        #    output[i] = FLOAT32_ZERO
        #    continue
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
        output[i] = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
    # return result
    return output    
                                           
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def parallel_real_distances( C_INT32 atomIndex, 
                             ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                             ndarray[C_FLOAT32, ndim=2] basis not None,
                             ndarray[C_FLOAT32, ndim=1] output not None,
                             bint allAtoms = True,
                             C_INT32 ncores = INT32_ONE):
    # declare variables
    cdef C_INT32 i, ci, startIndex, endIndex
    #cdef C_FLOAT32 box_dx, box_dy, box_dz
    #cdef C_FLOAT32 real_dx, real_dy, real_dz,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # start index
    if allAtoms:
        startIndex = INT32_ZERO
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>boxCoords.shape[0]
    # adjust ncores
    if ncores > endIndex-startIndex:
        ncores = endIndex-startIndex
    # create cores indexes range
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexesRange = np.linspace(startIndex, endIndex, ncores+1, dtype=NUMPY_INT32)
    # declare multi cores variables
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] real_dx = np.empty((ncores,), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] real_dy = np.empty((ncores,), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] real_dz = np.empty((ncores,), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] box_dx  = np.empty((ncores,), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] box_dy  = np.empty((ncores,), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] box_dz  = np.empty((ncores,), dtype=NUMPY_FLOAT32)
    # loop
    for ci in prange(ncores, nogil=True):
        #with gil:
        for i from indexesRange[ci] <= i < indexesRange[ci+1]:
            ### TO BE VERIFIED WHETHER IT IS FASTER OR NOT TO KEEP OR REMVOVE THE IF CONDITION ###
            #if i == atomIndex:
            #    output[i] = FLOAT32_ZERO
            #    continue
            # calculate difference
            box_dx[ci] = boxCoords[i,0]-atomBox_x
            box_dy[ci] = boxCoords[i,1]-atomBox_y
            box_dz[ci] = boxCoords[i,2]-atomBox_z
            box_dx[ci] -= round(box_dx[ci])
            box_dy[ci] -= round(box_dy[ci])
            box_dz[ci] -= round(box_dz[ci])
            # get real difference
            real_dx[ci] = box_dx[ci]*basis[0,0] + box_dy[ci]*basis[1,0] + box_dz[ci]*basis[2,0]
            real_dy[ci] = box_dx[ci]*basis[0,1] + box_dy[ci]*basis[1,1] + box_dz[ci]*basis[2,1]
            real_dz[ci] = box_dx[ci]*basis[0,2] + box_dy[ci]*basis[1,2] + box_dz[ci]*basis[2,2]
            # calculate distance         
            output[i] = <C_FLOAT32>sqrt(real_dx[ci]*real_dx[ci] + real_dy[ci]*real_dy[ci] + real_dz[ci]*real_dz[ci])
    # return result
    return output
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_pair_distribution_histograms( C_INT32 atomIndex, 
                                         ndarray[C_FLOAT32, ndim=1] distances not None,
                                         ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                         ndarray[C_INT32, ndim=1] elementIndex not None,
                                         ndarray[C_FLOAT32, ndim=3] hintra not None,
                                         ndarray[C_FLOAT32, ndim=3] hinter not None,
                                         C_FLOAT32 minDistance,
                                         C_FLOAT32 maxDistance,
                                         C_FLOAT32 bin, 
                                         bint allAtoms = True):
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_INT32 binIndex
    cdef C_INT32 atomMoleculeIndex, atomSymbolIndex
    cdef C_INT32 histSize
    cdef C_FLOAT32 float32Var, distance
    # cast arguments
    bin = <C_FLOAT32>bin
    minDistance = <C_FLOAT32>minDistance
    maxDistance = <C_FLOAT32>maxDistance
    # get histogram size
    histSize = <C_INT32>hintra.shape[2]
    # get atom molecule and symbol
    atomMoleculeIndex = moleculeIndex[atomIndex]
    atomSymbolIndex   = elementIndex[atomIndex]
    # start index
    if allAtoms:
        startIndex = <C_INT32>0
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>distances.shape[0]
    # loop
    for i from startIndex <= i < endIndex:
        if i == atomIndex: continue
        # get distance         
        distance = distances[i]
        # check limits
        if distance<minDistance:
            continue
        if distance>=maxDistance:
            continue
        # get index
        binIndex = <C_INT32>((distance-minDistance)/bin)
        # increment histograms
        if moleculeIndex[i] == atomMoleculeIndex:
            hintra[atomSymbolIndex,elementIndex[i],binIndex] += FLOAT32_ONE
        else:
            hinter[atomSymbolIndex,elementIndex[i],binIndex] += FLOAT32_ONE
   



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multiple_pair_distribution_histograms( ndarray[C_INT32, ndim=1] indexes not None,
                                           np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                           np.ndarray[C_FLOAT32, ndim=2] basis not None,
                                           ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                           ndarray[C_INT32, ndim=1] elementIndex not None,
                                           C_INT32 numberOfElements,
                                           C_FLOAT32 minDistance,
                                           C_FLOAT32 maxDistance,
                                           C_FLOAT32 bin,
                                           C_INT32 histSize,
                                           bint allAtoms=True):    
    # declare variables
    cdef C_INT32 i, ii
    # cast arguments
    bin         = <C_FLOAT32>bin
    minDistance = <C_FLOAT32>minDistance
    maxDistance = <C_FLOAT32>maxDistance
    histSize    = <C_INT32>histSize
    # create histograms
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] hintra = np.zeros((numberOfElements,numberOfElements,histSize), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] hinter = np.zeros((numberOfElements,numberOfElements,histSize), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] distances = np.empty((boxCoords.shape[0],), dtype=NUMPY_FLOAT32)

    # loop atoms
    for i in indexes:
        # compute distances
        distances = real_distances( atomIndex = i, 
                                    boxCoords = boxCoords,
                                    basis = basis,
                                    output=distances,
                                    allAtoms = allAtoms)
        # compute histogram           
        single_pair_distribution_histograms( atomIndex = i, 
                                             distances=distances,
                                             moleculeIndex = moleculeIndex,
                                             elementIndex = elementIndex,
                                             hintra = hintra,
                                             hinter = hinter,
                                             minDistance = minDistance,
                                             maxDistance = maxDistance,
                                             bin = bin,
                                             allAtoms = allAtoms )
    return hintra, hinter
    


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_pair_distribution_histograms( np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                       np.ndarray[C_FLOAT32, ndim=2] basis not None,
                                       ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                       ndarray[C_INT32, ndim=1] elementIndex not None,
                                       C_INT32 numberOfElements,
                                       C_FLOAT32 minDistance,
                                       C_FLOAT32 maxDistance,
                                       C_INT32 histSize,
                                       C_FLOAT32 bin):    
    # get number of atoms
    cdef numberOfAtoms = <C_INT32>boxCoords.shape[0]
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(numberOfAtoms, dtype=NUMPY_INT32)
    # calculate histograms
    return multiple_pair_distribution_histograms(indexes=indexes,
                                                 boxCoords = boxCoords,
                                                 basis = basis,
                                                 moleculeIndex = moleculeIndex,
                                                 elementIndex = elementIndex,
                                                 numberOfElements = numberOfElements,
                                                 minDistance = minDistance,
                                                 maxDistance = maxDistance,
                                                 histSize = histSize,
                                                 bin = bin,
                                                 allAtoms=False)
                                                 