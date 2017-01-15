"""
This is a C compiled module to compute pair distances histograms.
""" 
#from libc.math cimport sqrt, abs
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
cdef C_FLOAT32 FLOAT32_ZERO    = 0.0
cdef C_FLOAT32 FLOAT32_ONE     = 1.0
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_INT32   INT32_ZERO      = 0
cdef C_INT32   INT32_ONE       = 1



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _single_pairs_histograms( C_INT32          atomIndex, 
                                    C_INT32          atomSymbolIndex,
                                    C_INT32          atomMoleculeIndex,
                                    C_INT32          startIndex, 
                                    C_INT32          endIndex, 
                                    C_FLOAT32[:]     distances,
                                    C_INT32[:]       moleculeIndex,
                                    C_INT32[:]       elementIndex,
                                    C_FLOAT32[:,:,:] hintra,
                                    C_FLOAT32[:,:,:] hinter,
                                    C_FLOAT32        minDistance,
                                    C_FLOAT32        maxDistance,
                                    C_FLOAT32        bin,
                                    C_INT32          ncores = 1) nogil:
    cdef C_FLOAT32 distance
    cdef C_INT32 i, binIndex
    cdef C_INT32 num_threads = ncores
    for i in prange(startIndex, endIndex, INT32_ONE, nogil=True, schedule="static", num_threads=num_threads):
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
def single_pairs_histograms( C_INT32                    atomIndex, 
                             ndarray[C_FLOAT32, ndim=1] distances not None,
                             ndarray[C_INT32, ndim=1]   moleculeIndex not None,
                             ndarray[C_INT32, ndim=1]   elementIndex not None,
                             ndarray[C_FLOAT32, ndim=3] hintra not None,
                             ndarray[C_FLOAT32, ndim=3] hinter not None,
                             C_FLOAT32                  minDistance,
                             C_FLOAT32                  maxDistance,
                             C_FLOAT32                  bin, 
                             bint                       allAtoms = True,
                             C_INT32                    ncores = 1):
    """
    Computes the pair distribution histograms of a single atom given a distances array.
    
    :Arguments:
       #. atomIndex (int32): The index of the atom.
       #. distances (float32 array): The distances array.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. hintra (float32 array): The (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. bin (float32): The histogram bin size.
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True
       #. ncores (int32) [default=1]: The number of cores to use. 
                                  
    :Returns:
       #. hintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
    """
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_INT32 binIndex
    cdef C_INT32 atomMoleculeIndex, atomSymbolIndex
    cdef C_FLOAT32 float32Var, distance
    cdef C_INT32 num_threads = ncores
    # cast arguments
    bin = <C_FLOAT32>bin
    minDistance = <C_FLOAT32>minDistance
    maxDistance = <C_FLOAT32>maxDistance
    # get atom molecule and symbol
    atomMoleculeIndex = moleculeIndex[atomIndex]
    atomSymbolIndex   = elementIndex[atomIndex]
    # start index
    if allAtoms:
        startIndex = <C_INT32>0
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>distances.shape[0]
    # compute histograms
    _single_pairs_histograms( atomIndex         = atomIndex, 
                              atomSymbolIndex   = atomSymbolIndex,
                              atomMoleculeIndex = atomMoleculeIndex,
                              startIndex        = startIndex, 
                              endIndex          = endIndex, 
                              distances         = distances,
                              moleculeIndex     = moleculeIndex,
                              elementIndex      = elementIndex,
                              hintra            = hintra,
                              hinter            = hinter,
                              minDistance       = minDistance,
                              maxDistance       = maxDistance,
                              bin               = bin,
                              ncores            = ncores)

            

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multiple_pairs_histograms_coords( ndarray[C_INT32, ndim=1]      indexes not None,
                                      np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                      np.ndarray[C_FLOAT32, ndim=2] basis not None,
                                      bint                          isPBC,
                                      ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                      ndarray[C_INT32, ndim=1]      elementIndex not None,
                                      C_INT32                       numberOfElements,
                                      C_FLOAT32                     minDistance,
                                      C_FLOAT32                     maxDistance,
                                      C_FLOAT32                     bin,
                                      C_INT32                       histSize,
                                      bint                          allAtoms = True,
                                      C_INT32                       ncores = 1 ):    
    """
    Computes the pair distribution histograms of multiple atoms given atomic coordinates.
    
    :Arguments:
       #. indexes (int32 (k,3) numpy.ndarray): The atomic coordinates indexes array.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. bin (float32): The histogram bin size.
       #. histSize(int32): The histograms size.
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. hintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
    """
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
        # compute distances
        distances = pairs_distances_to_indexcoords( atomIndex = i, 
                                                    coords    = boxCoords,
                                                    basis     = basis,
                                                    isPBC     = isPBC,
                                                    allAtoms  = allAtoms,
                                                    ncores    = ncores)
        # compute histogram           
        single_pairs_histograms( atomIndex     = i, 
                                 distances     = distances,
                                 moleculeIndex = moleculeIndex,
                                 elementIndex  = elementIndex,
                                 hintra        = hintra,
                                 hinter        = hinter,
                                 minDistance   = minDistance,
                                 maxDistance   = maxDistance,
                                 bin           = bin,
                                 allAtoms      = allAtoms,
                                 ncores        = ncores )
    return hintra, hinter
    

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multiple_pairs_histograms_dists( ndarray[C_INT32, ndim=1]      indexes not None,
                                     np.ndarray[C_FLOAT32, ndim=2] distances not None,
                                     ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                     ndarray[C_INT32, ndim=1]      elementIndex not None,
                                     C_INT32                       numberOfElements,
                                     C_FLOAT32                     minDistance,
                                     C_FLOAT32                     maxDistance,
                                     C_FLOAT32                     bin,
                                     C_INT32                       histSize,
                                     bint                          allAtoms=True,
                                     C_INT32                       ncores = 1):    
    """
    Computes the pair distribution histograms of multiple atoms given atomic distances.
    
    :Arguments:
       #. indexes (int32 (k,3) numpy.ndarray): The atomic coordinates indexes array.
       #. distances (float32 array): The distances array.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. bin (float32): The histogram bin size.
       #. histSize(int32): The histograms size.
       #. allAtoms (bool): Perform the calculation over all the atoms. If False calculation starts from the given atomIndex. DEFAULT: True
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. hintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
    """
    # declare variables
    cdef C_INT32 i
    # cast arguments
    bin         = <C_FLOAT32>bin
    minDistance = <C_FLOAT32>minDistance
    maxDistance = <C_FLOAT32>maxDistance
    histSize    = <C_INT32>histSize
    # create histograms
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] hintra = np.zeros((numberOfElements,numberOfElements,histSize), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] hinter = np.zeros((numberOfElements,numberOfElements,histSize), dtype=NUMPY_FLOAT32)

    # loop
    for i from <C_INT32>0 <= i < <C_INT32>indexes.shape[0]:
        # compute histogram   
        single_pairs_histograms( atomIndex     = indexes[i], 
                                 distances     = distances[:,i],
                                 moleculeIndex = moleculeIndex,
                                 elementIndex  = elementIndex,
                                 hintra        = hintra,
                                 hinter        = hinter,
                                 minDistance   = minDistance,
                                 maxDistance   = maxDistance,
                                 bin           = bin,
                                 allAtoms      = allAtoms,
                                 ncores        = ncores )
    return hintra, hinter
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_pairs_histograms_coords( np.ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                  np.ndarray[C_FLOAT32, ndim=2] basis not None,
                                  bint                          isPBC,
                                  ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                  ndarray[C_INT32, ndim=1]      elementIndex not None,
                                  C_INT32                       numberOfElements,
                                  C_FLOAT32                     minDistance,
                                  C_FLOAT32                     maxDistance,
                                  C_FLOAT32                     bin,
                                  C_INT32                       histSize,
                                  C_INT32                       ncores = 1):    
    """
    Computes the pair distribution histograms of multiple atoms given atomic coordinates.
    
    :Arguments:
       #. boxCoords (float32 (n,3) numpy.ndarray): The atomic coordinates array.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. bin (float32): The histogram bin size.
       #. histSize(int32): The histograms size.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. hintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
    """
    # get number of atoms
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(<C_INT32>boxCoords.shape[0], dtype=NUMPY_INT32)
    # calculate histograms
    return multiple_pairs_histograms_coords(indexes          = indexes,
                                            boxCoords        = boxCoords,
                                            basis            = basis,
                                            isPBC            = isPBC,
                                            moleculeIndex    = moleculeIndex,
                                            elementIndex     = elementIndex,
                                            numberOfElements = numberOfElements,
                                            minDistance      = minDistance,
                                            maxDistance      = maxDistance,
                                            bin              = bin,
                                            histSize         = histSize,
                                            ncores           = ncores,
                                            allAtoms         = False)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_pairs_histograms_dists( np.ndarray[C_FLOAT32, ndim=2] distances not None,
                                 ndarray[C_INT32, ndim=1]      moleculeIndex not None,
                                 ndarray[C_INT32, ndim=1]      elementIndex not None,
                                 C_INT32                       numberOfElements,
                                 C_FLOAT32                     minDistance,
                                 C_FLOAT32                     maxDistance,
                                 C_FLOAT32                     bin,
                                 C_INT32                       histSize,
                                 C_INT32                       ncores = 1):    
    """
    Computes the pair distribution histograms of multiple atoms given atomic distances.
    
    :Arguments:
       #. distances (float32 array): The distances array.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. elementIndex (int32 array): The element's index array, assigning an element index for every atom.
       #. numberOfElements (int32): The number of elements in the system.
       #. minDistance (float32): The minimum distance to be counted in the histogram.
       #. maxDistance (float32): The maximum distance to be counted in the histogram.
       #. bin (float32): The histogram bin size.
       #. histSize(int32): The histograms size.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. hintra (float32 array): The updated (numberOfElements,numberOfElements,1) array for intra-molecular distances histograms.
       #. hinter (float32 array): The updated (numberOfElements,numberOfElements,1) array for inter-molecular distances histograms.
    """
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(<C_INT32>distances.shape[1], dtype=NUMPY_INT32)
    # calculate histograms
    return multiple_pairs_histograms_dists(indexes          = indexes,
                                           distances        = distances,
                                           moleculeIndex    = moleculeIndex,
                                           elementIndex     = elementIndex,
                                           numberOfElements = numberOfElements,
                                           minDistance      = minDistance,
                                           maxDistance      = maxDistance,
                                           histSize         = histSize,
                                           bin              = bin,
                                           allAtoms         = False,
                                           ncores           = ncores)





                                        