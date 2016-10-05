"""
This is a C compiled module to compute atomic bonds.
"""                      
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from cython.parallel import prange
from fullrmc.Core.pairs_distances import pairs_distances_to_point

# declare types
NUMPY_FLOAT32 = np.float32
NUMPY_INT32   = np.int32
ctypedef np.float32_t C_FLOAT32
ctypedef np.int32_t   C_INT32

# declare constants
cdef C_FLOAT32 FLOAT_ZERO  = 0.0
cdef C_FLOAT32 FLOAT_ONE   = 1.0
cdef C_INT32   INT32_ZERO  = 0
cdef C_INT32   INT32_ONE   = 1



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef C_FLOAT32 _single_atom_single_shell_dists( C_FLOAT32[:] distances,
                                                C_FLOAT32    lowerShell,
                                                C_FLOAT32    upperShell,
                                                C_INT32      ncores = 1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 coordNumber = FLOAT_ZERO
    cdef C_INT32 num_threads = ncores
    # loop
    for i in prange(INT32_ZERO, <C_INT32>distances.shape[0], INT32_ONE, nogil=True, schedule="static", num_threads=num_threads):
    #for i from 0 <= i < <C_INT32>distances.shape[0]:
        if lowerShell <= distances[i] <= upperShell:      
            coordNumber += FLOAT_ONE
            #with gil:
            #    print i, lowerShell, upperShell, distances[i]
    return coordNumber



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_single_shell_subdists( C_FLOAT32[:] distances,
                                       C_FLOAT32    lowerShell,
                                       C_FLOAT32    upperShell,
                                       C_INT32      ncores = 1):
    return _single_atom_single_shell_dists( distances  = distances,
                                            lowerShell = lowerShell,
                                            upperShell = upperShell,
                                            ncores     = ncores)


    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_single_shell_totdists( ndarray[C_FLOAT32, ndim=1] distances,
                                       C_INT32[:]                 shellIndexes,
                                       C_FLOAT32                  lowerShell,
                                       C_FLOAT32                  upperShell,
                                       C_INT32                    ncores = 1):
    # declare variables
    return _single_atom_single_shell_dists( distances  = distances[shellIndexes],
                                            lowerShell = lowerShell,
                                            upperShell = upperShell,
                                            ncores     =  ncores)
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_single_shell_coords( C_INT32                    coreIndex,
                                     C_INT32[:]                 shellIndexes,
                                     ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                     C_FLOAT32[:,:]             basis not None,
                                     bint                       isPBC,
                                     C_FLOAT32                  lowerShell,
                                     C_FLOAT32                  upperShell,
                                     C_INT32                    ncores = 1):
    # declare variables
    distances = pairs_distances_to_point( point  = boxCoords[ coreIndex ], 
                                          coords = boxCoords[ shellIndexes ],
                                          basis  = basis,
                                          isPBC  = isPBC,
                                          ncores = ncores)  
    # compute and return
    return single_atom_single_shell_subdists(distances  = distances,
                                             lowerShell = lowerShell,
                                             upperShell = upperShell,
                                             ncores     = ncores)    
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_multi_shells_totdists( ndarray[C_FLOAT32, ndim=1] distances,
                                       list                       shellsIndexes,
                                       C_FLOAT32[:]               lowerShells,
                                       C_FLOAT32[:]               upperShells,
                                       C_INT32                    ncores = 1):
    # declare variables
    cdef C_INT32 i
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] coordNumbers = np.zeros(len(shellsIndexes), dtype=NUMPY_FLOAT32)
    # compute coordination numbers
    for i from 0 <= i < <C_INT32>len(shellsIndexes):
        coordNumbers[i] = _single_atom_single_shell_dists( distances   = distances[shellsIndexes[i]],
                                                           lowerShell  = lowerShells[i],
                                                           upperShell  = upperShells[i],
                                                           ncores      = ncores)
    return coordNumbers
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_multi_shells_coords( C_INT32                    coreIndex,
                                     list                       shellsIndexes,
                                     ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                     C_FLOAT32[:,:]             basis not None,
                                     bint                       isPBC,
                                     C_FLOAT32[:]               lowerShells,
                                     C_FLOAT32[:]               upperShells,
                                     C_INT32                    ncores = 1):
    # declare variables
    cdef C_INT32 i
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] coordNumbers = np.zeros(len(shellsIndexes), dtype=NUMPY_FLOAT32)
    # compute coordination numbers
    for i from 0 <= i < <C_INT32>len(shellsIndexes):
        # declare variables
        distances = pairs_distances_to_point( point  = boxCoords[ coreIndex ], 
                                              coords = boxCoords[ shellsIndexes[i] ],
                                              basis  = basis,
                                              isPBC  = isPBC,
                                              ncores = ncores)      
        coordNumbers[i] = _single_atom_single_shell_dists( distances   = distances,
                                                           lowerShell  = lowerShells[i],
                                                           upperShell  = upperShells[i],
                                                           ncores      = ncores)
    return coordNumbers        
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_coord_number_totdists( C_INT32                    atomIndex,
                                       ndarray[C_FLOAT32, ndim=1] distances,
                                       list                       coresIndexes,
                                       list                       shellsIndexes,
                                       list                       lowerShells,
                                       list                       upperShells,
                                       list                       asCoreDefIdxs,
                                       list                       inShellDefIdxs,
                                       C_FLOAT32[:]               coordNumData,
                                       C_INT32                    ncores = 1):
    # declare variables
    cdef C_INT32 defIdx
    # compute coordination numbers as core
    for defIdx in asCoreDefIdxs[atomIndex]:
        coordNumber = single_atom_single_shell_totdists(distances    = distances,
                                                        shellIndexes = shellsIndexes[defIdx],
                                                        lowerShell   = lowerShells[defIdx] ,
                                                        upperShell   = upperShells[defIdx],
                                                        ncores       = ncores)
        coordNumData[defIdx] += coordNumber
    # compute coordination numbers in shell
    for defIdx in asCoreDefIdxs[atomIndex]:
        coordNumber = single_atom_single_shell_totdists(distances    = distances,
                                                        shellIndexes = coresIndexes[defIdx],
                                                        lowerShell   = lowerShells[defIdx] ,
                                                        upperShell   = upperShells[defIdx],
                                                        ncores       = ncores)
        coordNumData[defIdx] += coordNumber
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atom_coord_number_coords( C_INT32                    atomIndex,
                                     ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                     C_FLOAT32[:,:]             basis not None,
                                     bint                       isPBC,
                                     list                       coresIndexes,
                                     list                       shellsIndexes,
                                     list                       lowerShells,
                                     list                       upperShells,
                                     list                       asCoreDefIdxs,
                                     list                       inShellDefIdxs,
                                     C_FLOAT32[:]               coordNumData,
                                     C_INT32                    ncores = 1):
    # compute coordination numbers as core
    for defIdx in asCoreDefIdxs[atomIndex]:
        coordNumber = single_atom_single_shell_coords( coreIndex    = atomIndex,
                                                       shellIndexes = shellsIndexes[defIdx],
                                                       boxCoords    = boxCoords,
                                                       basis        = basis,
                                                       isPBC        = isPBC,
                                                       lowerShell   = lowerShells[defIdx],
                                                       upperShell   = upperShells[defIdx],
                                                       ncores       = ncores)
        coordNumData[defIdx] += coordNumber
    # compute coordination numbers in shell
    for defIdx in inShellDefIdxs[atomIndex]:
        coordNumber = single_atom_single_shell_coords( coreIndex    = atomIndex,
                                                       shellIndexes = coresIndexes[defIdx],
                                                       boxCoords    = boxCoords,
                                                       basis        = basis,
                                                       isPBC        = isPBC,
                                                       lowerShell   = lowerShells[defIdx],
                                                       upperShell   = upperShells[defIdx],
                                                       ncores       = ncores)
        coordNumData[defIdx] += coordNumber    
        
        
            
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multi_atoms_coord_number_totdists( C_INT32[:]   indexes,
                                       list         distances,
                                       list         coresIndexes,
                                       list         shellsIndexes,
                                       list         lowerShells,
                                       list         upperShells,
                                       list         asCoreDefIdxs,
                                       list         inShellDefIdxs,
                                       C_FLOAT32[:] coordNumData,
                                       C_INT32      ncores = 1):
    # declare variables
    cdef C_INT32 i
    for i from 0 <= i < <C_INT32>len(indexes):
        single_atom_coord_number_totdists( atomIndex      = indexes[i],
                                           distances      = distances[i],
                                           coresIndexes   = coresIndexes,
                                           shellsIndexes  = shellsIndexes,
                                           lowerShells    = lowerShells,
                                           upperShells    = upperShells,
                                           asCoreDefIdxs  = asCoreDefIdxs,
                                           inShellDefIdxs = inShellDefIdxs,
                                           coordNumData   = coordNumData,
                                           ncores         = ncores)
    
    

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def multi_atoms_coord_number_coords( C_INT32[:]                 indexes,
                                     ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                     C_FLOAT32[:,:]             basis not None,
                                     bint                       isPBC,
                                     list                       coresIndexes,
                                     list                       shellsIndexes,
                                     list                       lowerShells,
                                     list                       upperShells,
                                     list                       asCoreDefIdxs,
                                     list                       inShellDefIdxs,
                                     C_FLOAT32[:]               coordNumData,
                                     C_INT32                    ncores = 1):
    # declare variables
    cdef C_INT32 i
    for i from 0 <= i < <C_INT32>len(indexes):
        single_atom_coord_number_coords( atomIndex      = indexes[i],
                                         boxCoords      = boxCoords,
                                         basis          = basis,
                                         isPBC          = isPBC,
                                         coresIndexes   = coresIndexes,
                                         shellsIndexes  = shellsIndexes,
                                         lowerShells    = lowerShells,
                                         upperShells    = upperShells,
                                         asCoreDefIdxs  = asCoreDefIdxs,
                                         inShellDefIdxs = inShellDefIdxs,
                                         coordNumData   = coordNumData,
                                         ncores         = ncores)
    
    

    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def all_atoms_coord_number_totdists( ndarray[C_FLOAT32, ndim=2] distances,
                                     list                       coresIndexes,
                                     list                       shellsIndexes,
                                     list                       lowerShells,
                                     list                       upperShells,
                                     list                       asCoreDefIdxs,
                                     list                       inShellDefIdxs,
                                     C_FLOAT32[:]               coordNumData,
                                     C_INT32                    ncores = 1):
    # declare variables
    cdef C_INT32 i
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(<C_INT32>distances.shape[1], dtype=NUMPY_INT32)
    # run multiple atoms coordination number using distances
    multi_atoms_coord_number_totdists( indexes        = indexes,
                                       distances      = distances,
                                       coresIndexes   = coresIndexes,
                                       shellsIndexes  = shellsIndexes,
                                       lowerShells    = lowerShells,
                                       upperShells    = upperShells,
                                       asCoreDefIdxs  = asCoreDefIdxs,
                                       inShellDefIdxs = inShellDefIdxs,
                                       coordNumData   = coordNumData,
                                       ncores         = ncores)
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def all_atoms_coord_number_coords( ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                   C_FLOAT32[:,:]             basis not None,
                                   bint                       isPBC,
                                   list                       coresIndexes,
                                   list                       shellsIndexes,
                                   list                       lowerShells,
                                   list                       upperShells,
                                   list                       asCoreDefIdxs,
                                   list                       inShellDefIdxs,
                                   C_FLOAT32[:]               coordNumData,
                                   C_INT32                    ncores = 1):
    # declare variables
    cdef C_INT32 i
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(<C_INT32>boxCoords.shape[0], dtype=NUMPY_INT32)
    # run multiple atoms coordination number using coordinates
    multi_atoms_coord_number_coords( indexes        = indexes,
                                     boxCoords      = boxCoords,
                                     basis          = basis,
                                     isPBC          = isPBC,
                                     coresIndexes   = coresIndexes,
                                     shellsIndexes  = shellsIndexes,
                                     lowerShells    = lowerShells,
                                     upperShells    = upperShells,
                                     asCoreDefIdxs  = asCoreDefIdxs,
                                     inShellDefIdxs = inShellDefIdxs,
                                     coordNumData   = coordNumData,
                                     ncores         = ncores)


            
    
    
    
    