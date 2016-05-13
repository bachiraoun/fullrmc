"""
This is a C compiled Cython generated module to calculate pair distribution histograms. It contains the following methods.
                   
**single_pair_distribution_histograms**: It calculates the pair distribution histograms of a single atom.
    
    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. hintra (float32 array): The (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. bin (bool): The histogram bin size.
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True
                                  
    :Returns:
       #. hintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.

                                              
**multiple_pair_distribution_histograms**: It calculates the pair distribution histograms of multiple atoms. 
    It creates the inter and intra-molecular distance histograms and calls 
    single_pair_distribution_histograms method for every desired atom index.

    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. histSize(int32): The histograms size.
       #. bin (bool): The histogram bin size.
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True
                     
    :Returns:
       #. hintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.

                                           
**full_pair_distribution_histograms**:  It calculates the pair distribution histograms of all atoms. 
    It calls multiple_distances method for all atoms.

    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. histSize(int32): The histograms size.
       #. bin (bool): The histogram bin size.
                     
    :Returns:
       #. hintra (float32 array): The created (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The created (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
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
cdef C_FLOAT32 FLOAT32_ZERO    = 0.0
cdef C_FLOAT32 FLOAT32_ONE     = 1.0
cdef C_INT32   INT32_ONE       = 1
cdef C_INT32   INT32_ZERO      = 0

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
def single_pair_distribution_histograms( C_INT32 atomIndex, 
                                         ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                         ndarray[C_FLOAT32, ndim=2] basis not None,
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
    cdef C_FLOAT32 float32Var
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz, distance,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # cast arguments
    bin = <C_FLOAT32>bin
    minDistance = <C_FLOAT32>minDistance
    maxDistance = <C_FLOAT32>maxDistance
    # get histogram size
    histSize = <C_INT32>hintra.shape[2]
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # get atom molecule and symbol
    atomMoleculeIndex = moleculeIndex[atomIndex]
    atomSymbolIndex   = elementIndex[atomIndex]
    # start index
    if allAtoms:
        startIndex = <C_INT32>0
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>boxCoords.shape[0]
    # loop
    for i from startIndex <= i < endIndex:
        if i == atomIndex: continue
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
    # loop atoms
    for i in indexes:
    #for i in prange(boxCoords.shape[0]-1):
        single_pair_distribution_histograms( atomIndex = i, 
                                             boxCoords = boxCoords,
                                             basis = basis,
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
                                           