"""
This is a C compiled module to compute atomic pair distances.
"""
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
cdef C_INT32   INT32_ZERO      = 0
cdef C_INT32   INT32_ONE       = 1


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x) nogil
    C_FLOAT32 ceil(C_FLOAT32 x)  nogil
    C_FLOAT32 sqrt(C_FLOAT32 x)  nogil

    
cdef inline C_FLOAT32 round(C_FLOAT32 num) nogil:
    return floor(num + HALF_BOX_LENGTH) if (num > FLOAT32_ZERO) else ceil(num - HALF_BOX_LENGTH)


      
############################################################################################        
################################# C DIFFERENCE DEFINITIONS #################################

######## From To differences ########
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _from_to_boxpoint_realdifference_PBC( C_FLOAT32[:]   boxPointFrom,
                                                C_FLOAT32[:]   boxPointTo,
                                                C_FLOAT32[:]   difference,
                                                C_FLOAT32[:,:] basis):
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    # calculate difference
    diff_x = boxPointTo[0]-boxPointFrom[0]
    diff_y = boxPointTo[1]-boxPointFrom[1]
    diff_z = boxPointTo[2]-boxPointFrom[2]
    box_dx = diff_x-round(diff_x)
    box_dy = diff_y-round(diff_y)
    box_dz = diff_z-round(diff_z)
    # get real difference
    difference[0] = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
    difference[1] = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
    difference[2] = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]




@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _from_to_boxpoints_realdifferences_PBC( C_FLOAT32[:,:] boxPointsFrom,
                                                  C_FLOAT32[:,:] boxPointsTo,
                                                  C_FLOAT32[:,:] basis,
                                                  C_FLOAT32[:,:] differences,
                                                  C_INT32        ncores = 1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_INT32 num_threads = ncores
    # loop
    for i in prange(<C_INT32>boxPointsFrom.shape[0], nogil=True, schedule="static", num_threads=num_threads): 
    #for i from INT32_ZERO <= i < <C_INT32>boxPointsFrom.shape[0]:
        # calculate difference
        diff_x = boxPointsTo[i,0]-boxPointsFrom[i,0]
        diff_y = boxPointsTo[i,1]-boxPointsFrom[i,1]
        diff_z = boxPointsTo[i,2]-boxPointsFrom[i,2]
        box_dx = diff_x-round(diff_x)
        box_dy = diff_y-round(diff_y)
        box_dz = diff_z-round(diff_z)
        # get real difference
        differences[i,0] = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        differences[i,1] = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        differences[i,2] = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _from_to_realpoint_realdifference_PBC( C_FLOAT32[:] realPointFrom,
                                                 C_FLOAT32[:] realPointTo,
                                                 C_FLOAT32[:] difference):
    # calculate difference
    difference[0] = realPointTo[0]-realPointFrom[0]
    difference[1] = realPointTo[1]-realPointFrom[1]
    difference[2] = realPointTo[2]-realPointFrom[2]
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _from_to_realpoints_realdifferences_IBC( C_FLOAT32[:,:] realPointsFrom,
                                                   C_FLOAT32[:,:] realPointsTo,
                                                   C_FLOAT32[:,:] differences,
                                                   C_INT32        ncores = 1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_INT32 num_threads = ncores
    # loop
    for i in prange(INT32_ZERO, <C_INT32>realPointsFrom.shape[0], INT32_ONE, nogil=True, schedule="static", num_threads=num_threads):
    #for i from INT32_ZERO <= i < <C_INT32>boxPointsFrom.shape[0]:
        # calculate difference
        differences[i,0] = realPointsTo[i,0]-realPointsFrom[i,0]
        differences[i,1] = realPointsTo[i,1]-realPointsFrom[i,1]
        differences[i,2] = realPointsTo[i,2]-realPointsFrom[i,2]
        
        
        
######## point differences ########       
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _boxcoords_realdifferences_to_boxpoint_PBC( C_FLOAT32[:]   boxPoint,
                                                      C_FLOAT32[:,:] boxCoords,
                                                      C_FLOAT32[:,:] basis,
                                                      C_FLOAT32[:,:] differences,
                                                      C_INT32        ncores = 1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_INT32 num_threads = ncores
    # loop
    for i in prange(<C_INT32>boxCoords.shape[0], nogil=True, schedule="static", num_threads=num_threads): 
    #for i in range(<C_INT32>boxCoords.shape[0]):
        # calculate difference
        diff_x = boxPoint[0]-boxCoords[i,0]
        diff_y = boxPoint[1]-boxCoords[i,1]
        diff_z = boxPoint[2]-boxCoords[i,2]
        box_dx = diff_x-round(diff_x)
        box_dy = diff_y-round(diff_y)
        box_dz = diff_z-round(diff_z)
        # get real difference
        differences[i,0] = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        differences[i,1] = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        differences[i,2] = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _realcoords_realdifferences_to_realpoint_IBC( C_FLOAT32[:]   realPoint,
                                                        C_FLOAT32[:,:] realCoords,
                                                        C_FLOAT32[:,:] differences,
                                                        C_INT32        ncores=1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 atomReal_x, atomReal_y, atomReal_z
    cdef C_INT32 num_threads = ncores
    # get point coordinates
    atomReal_x = realPoint[0]
    atomReal_y = realPoint[1]
    atomReal_z = realPoint[2]    
    # loop
    for i in prange(<C_INT32>realCoords.shape[0], nogil=True, schedule="static", num_threads=num_threads): 
    #for i in range(<C_INT32>realCoords.shape[0]):
        # calculate difference
        differences[i,0] = atomReal_x-realCoords[i,0]
        differences[i,1] = atomReal_y-realCoords[i,1]
        differences[i,2] = atomReal_z-realCoords[i,2]
        
        
            
######## index differences ########
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)               
cdef void _boxcoords_realdifferences_to_indexcoords_PBC( C_INT32        atomIndex,
                                                         C_FLOAT32[:,:] boxCoords,
                                                         C_FLOAT32[:,:] basis,
                                                         C_FLOAT32[:,:] differences,
                                                         bint           allAtoms = True,
                                                         C_INT32        ncores = 1) nogil:                          
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    cdef C_INT32 num_threads = ncores
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
    for i in prange(startIndex, endIndex, INT32_ONE, nogil=True, schedule="static", num_threads=num_threads): 
    #for i from startIndex <= i < endIndex:
        # calculate difference
        diff_x = atomBox_x-boxCoords[i,0]
        diff_y = atomBox_y-boxCoords[i,1]
        diff_z = atomBox_z-boxCoords[i,2]
        box_dx = diff_x-round(diff_x)
        box_dy = diff_y-round(diff_y)
        box_dz = diff_z-round(diff_z)
        # get real difference
        differences[i,0] = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        differences[i,1] = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        differences[i,2] = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]                                                  
               
                                                    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _realcoords_realdifferences_to_indexcoords_IBC( C_INT32        atomIndex,
                                                          C_FLOAT32[:,:] realCoords,
                                                          C_FLOAT32[:,:] differences,
                                                          bint           allAtoms = True,
                                                          C_INT32        ncores=1) nogil:   
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_FLOAT32 atomReal_x, atomReal_y, atomReal_z
    cdef C_INT32 num_threads = ncores
    # get point coordinates
    atomReal_x = realCoords[atomIndex,0]
    atomReal_y = realCoords[atomIndex,1]
    atomReal_z = realCoords[atomIndex,2] 
    # start index
    if allAtoms:
        startIndex = INT32_ZERO
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>realCoords.shape[0]
    # loop
    for i in prange(startIndex, endIndex, INT32_ONE, nogil=True, schedule="static", num_threads=num_threads): 
    #for i from startIndex <= i < endIndex:
        # calculate difference
        differences[i,0] = atomReal_x - realCoords[i,0]
        differences[i,1] = atomReal_y - realCoords[i,1]
        differences[i,2] = atomReal_z - realCoords[i,2]

   

   
############################################################################################        
################################## C DISTANCE DEFINITIONS ##################################
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef C_FLOAT32 _boxcoords_realdistance_to_boxpoint_PBC( C_FLOAT32[:]   boxPoint1,
                                                        C_FLOAT32[:]   boxPoint2,
                                                        C_FLOAT32[:,:] basis) nogil:
    # declare variables
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz,
    # compute difference    
    diff_x = boxPoint1[0] - boxPoint2[0]
    diff_y = boxPoint1[1] - boxPoint2[1]
    diff_z = boxPoint1[2] - boxPoint2[2]
    box_dx = diff_x - round(diff_x)
    box_dy = diff_y - round(diff_y)
    box_dz = diff_z - round(diff_z)
    # get real difference
    real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
    real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
    real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
    # compute and return return distance
    return <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)

     
                
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _boxcoords_realdistances_to_boxpoint_PBC( C_FLOAT32[:]   boxPoint,
                                                    C_FLOAT32[:,:] boxCoords,
                                                    C_FLOAT32[:,:] basis,
                                                    C_FLOAT32[:]   distances,
                                                    C_INT32        ncores = 1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    cdef C_INT32 num_threads = ncores
    # get point coordinates
    atomBox_x = boxPoint[0]
    atomBox_y = boxPoint[1]
    atomBox_z = boxPoint[2]    
    # loop
    for i in prange(<C_INT32>boxCoords.shape[0], nogil=True, schedule="static", num_threads=num_threads): # Error compiling Cython file: Cannot read reduction variable in loop body
    #for i in range(<C_INT32>boxCoords.shape[0]):
        # calculate difference
        diff_x = atomBox_x - boxCoords[i,0]
        diff_y = atomBox_y - boxCoords[i,1]
        diff_z = atomBox_z - boxCoords[i,2]
        box_dx = diff_x - round(diff_x)
        box_dy = diff_y - round(diff_y)
        box_dz = diff_z - round(diff_z)
        # get real difference
        real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # calculate distance         
        distances[i] = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)               
cdef void _boxcoords_realdistances_to_indexcoords_PBC( C_INT32        atomIndex,
                                                       C_FLOAT32[:,:] boxCoords,
                                                       C_FLOAT32[:,:] basis,
                                                       C_FLOAT32[:]   distances,
                                                       bint           allAtoms = True,
                                                       C_INT32        ncores = 1) nogil:                          
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_FLOAT32 diff_x, diff_y, diff_z
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz,
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    cdef C_INT32 num_threads = ncores
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
    for i in prange(startIndex, endIndex, INT32_ONE, nogil=True, schedule="static", num_threads=num_threads): # Error compiling Cython file: Cannot read reduction variable in loop body
    #for i from startIndex <= i < endIndex:
        # calculate difference
        diff_x = atomBox_x - boxCoords[i,0]
        diff_y = atomBox_y - boxCoords[i,1]
        diff_z = atomBox_z - boxCoords[i,2]
        box_dx = diff_x - round(diff_x)
        box_dy = diff_y - round(diff_y)
        box_dz = diff_z - round(diff_z)
        # get real difference
        real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # calculate distance         
        distances[i] = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)                                                    
                                                    



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef C_FLOAT32 _realcoords_realdistance_to_realpoint_IBC( C_FLOAT32[:]   realPoint1,
                                                          C_FLOAT32[:]   realPoint2) nogil:
    # declare variables
    cdef C_FLOAT32 real_dx, real_dy, real_dz
    # loop
    real_dx = realPoint1[0]-realPoint2[0]
    real_dy = realPoint1[1]-realPoint2[1]
    real_dz = realPoint1[2]-realPoint2[2]
    # calculate and return distance         
    return <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)


                                     
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _realcoords_realdistances_to_realpoint_IBC( C_FLOAT32[:]   realPoint,
                                                      C_FLOAT32[:,:] realCoords,
                                                      C_FLOAT32[:]   distances,
                                                      C_INT32        ncores=1) nogil:
    # declare variables
    cdef C_INT32 i
    cdef C_FLOAT32 real_dx, real_dy, real_dz
    cdef C_FLOAT32 atomReal_x, atomReal_y, atomReal_z
    cdef C_INT32 num_threads = ncores
    # get point coordinates
    atomReal_x = realPoint[0]
    atomReal_y = realPoint[1]
    atomReal_z = realPoint[2]    
    # loop
    for i in prange(<C_INT32>realCoords.shape[0], nogil=True, schedule="static", num_threads=num_threads): 
    #for i in range(<C_INT32>realCoords.shape[0]):
        # calculate difference
        real_dx = realCoords[i,0]-atomReal_x
        real_dy = realCoords[i,1]-atomReal_y
        real_dz = realCoords[i,2]-atomReal_z
        # calculate distance         
        distances[i] = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)

        
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
cdef void _realcoords_realdistances_to_indexcoords_IBC( C_INT32        atomIndex,
                                                        C_FLOAT32[:,:] realCoords,
                                                        C_FLOAT32[:]   distances,
                                                        bint           allAtoms = True,
                                                        C_INT32        ncores=1) nogil:   
    # declare variables
    cdef C_INT32 i, startIndex, endIndex
    cdef C_FLOAT32 real_dx, real_dy, real_dz
    cdef C_FLOAT32 atomReal_x, atomReal_y, atomReal_z
    cdef C_INT32 num_threads = ncores
    # get point coordinates
    atomReal_x = realCoords[atomIndex,0]
    atomReal_y = realCoords[atomIndex,1]
    atomReal_z = realCoords[atomIndex,2] 
    # start index
    if allAtoms:
        startIndex = INT32_ZERO
    else:
        startIndex = <C_INT32>atomIndex
    endIndex = <C_INT32>realCoords.shape[0]
    # loop
    for i in prange(startIndex, endIndex, INT32_ONE, nogil=True, schedule="static", num_threads=num_threads): 
    #for i from startIndex <= i < endIndex:
        # calculate difference
        real_dx = atomReal_x - realCoords[i,0]
        real_dy = atomReal_y - realCoords[i,1]
        real_dz = atomReal_z - realCoords[i,2]
        # calculate distance         
        distances[i] = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
        
        
        
        
############################################################################################        
############################## PYTHON DIFFERENCE DEFINITIONS ###############################
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def from_to_points_differences( ndarray[C_FLOAT32, ndim=2] pointsFrom not None,
                                ndarray[C_FLOAT32, ndim=2] pointsTo not None,
                                C_FLOAT32[:,:]             basis not None,
                                bint                       isPBC, 
                                C_INT32                    ncores = 1):                       
    """
    Compute point to point vector difference between two atomic coordinates arrays taking 
    into account periodic or infinite boundary conditions. Difference is calculated as 
    the following:
    
    .. math::
            differences[i,:] = boundaryConditions( pointsTo[i,:] - pointsFrom[i,:] )
    
    :Arguments:
       #. pointsFrom (float32 (n,3) numpy.ndarray): The first atomic coordinates array of 
          the same shape as pointsTo.
       #. pointsTo (float32 (n,3) numpy.ndarray): The second atomic coordinates array of 
          the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. differences (float32 (n,3) numpy.ndarray): The computed differences array.   
    """
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] differences = np.empty((<C_INT32>pointsFrom.shape[0],3), dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        _from_to_boxpoints_realdifferences_PBC( boxPointsFrom = pointsFrom,
                                                boxPointsTo   = pointsTo,
                                                basis         = basis,
                                                differences   = differences,
                                                ncores        = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        _from_to_realpoints_realdifferences_IBC( realPointsFrom = pointsFrom,
                                                 realPointsTo   = pointsTo,
                                                 differences    = differences,
                                                 ncores         = ncores)
    # return differences
    return differences



# 
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pair_difference_to_point( ndarray[C_FLOAT32, ndim=1] point1  not None,
                              ndarray[C_FLOAT32, ndim=1] point2 not None,
                              C_FLOAT32[:,:]             basis not None,
                              bint                       isPBC, 
                              C_INT32                    ncores = 1):                        
    """
    Compute differences between one atomic coordinates arrays to a point coordinates  
    taking into account periodic or infinite boundary conditions. Difference is 
    calculated as the following:
    
    .. math::
            differences[i,:] = boundaryConditions( point[0,:] - coords[i,:] )
    
    :Arguments:
       #. point (float32 (1,3) numpy.ndarray): The atomic coordinates point.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. differences (float32 (n,3) numpy.ndarray): The computed differences array.   
    """
    # if periodic boundary conditions, coords must be in box
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] difference = np.empty((3), dtype=NUMPY_FLOAT32)
    if isPBC:
        _from_to_boxpoint_realdifference_PBC( boxPointFrom = point1,
                                              boxPointTo   = point2,
                                              difference   = difference,
                                              basis        = basis)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        _from_to_realpoint_realdifference_PBC( realPointFrom  = point1,
                                               realPointTo    = point2,
                                               difference     = difference)
    # return difference
    return difference



 
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_differences_to_point( ndarray[C_FLOAT32, ndim=1] point  not None,
                                ndarray[C_FLOAT32, ndim=2] coords not None,
                                C_FLOAT32[:,:]             basis not None,
                                bint                       isPBC, 
                                C_INT32                    ncores = 1):                        
    """
    Compute differences between one atomic coordinates arrays to a point coordinates  
    taking into account periodic or infinite boundary conditions. Difference is 
    calculated as the following:
    
    .. math::
            differences[i,:] = boundaryConditions( point[0,:] - coords[i,:] )
    
    :Arguments:
       #. point (float32 (1,3) numpy.ndarray): The atomic coordinates point.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. differences (float32 (n,3) numpy.ndarray): The computed differences array.   
    """
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] differences = np.empty((<C_INT32>coords.shape[0],3), dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        _boxcoords_realdifferences_to_boxpoint_PBC( boxPoint    = point,
                                                    boxCoords   = coords,
                                                    basis       = basis,
                                                    differences = differences,
                                                    ncores      = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        _realcoords_realdifferences_to_realpoint_IBC( realPoint   = point,
                                                      realCoords  = coords,
                                                      differences = differences,
                                                      ncores      = ncores)
    # return differences
    return differences


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_differences_to_indexcoords( C_INT32                    atomIndex,
                                      ndarray[C_FLOAT32, ndim=2] coords not None,
                                      C_FLOAT32[:,:]             basis not None,
                                      bint                       isPBC,
                                      bint                       allAtoms = True,
                                      C_INT32                    ncores = 1):                       
    """
    Compute differences between one atomic coordinates arrays to a point coordinates   
    given its index in the coordinates array and taking into account periodic or 
    infinite boundary conditions. Difference is calculated as the following:
    
    .. math::
            differences[i,:] = boundaryConditions( coords[atomIndex,:] - coords[i,:] )
    
    :Arguments:
       #. atomIndex (int32): The index of the atomic coordinates point.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. differences (float32 (n,3) numpy.ndarray): The computed differences array.   
    """
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] differences = np.empty((<C_INT32>coords.shape[0],3), dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        _boxcoords_realdifferences_to_indexcoords_PBC( atomIndex   = atomIndex,
                                                       boxCoords   = coords,
                                                       basis       = basis,
                                                       differences = differences,
                                                       allAtoms    = allAtoms,
                                                       ncores      = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        _realcoords_realdifferences_to_indexcoords_IBC( atomIndex   = atomIndex,
                                                        realCoords  = coords,
                                                        differences = differences,
                                                        allAtoms    = allAtoms,
                                                        ncores      = ncores)
    # return differences
    return differences    

    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_differences_to_multi_points( ndarray[C_FLOAT32, ndim=2] points not None,
                                       ndarray[C_FLOAT32, ndim=2] coords not None,
                                       C_FLOAT32[:,:]             basis not None,
                                       bint                       isPBC, 
                                       C_INT32                    ncores = 1):                       
    """
    Compute differences between one atomic coordinates arrays to a multiple points   
    coordinates taking into account periodic or infinite boundary conditions. 
    Difference is calculated as the following:
    
    .. math::
            differences[i,:,k] = boundaryConditions( points[k,:] - coords[i,:] )
    
    :Arguments:
       #. points (float32 (k,3) numpy.ndarray): The multiple atomic coordinates points.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. differences (float32 (n,3,k) numpy.ndarray): The computed differences array.   
    """
    cdef C_INT32 i
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] differences = np.empty( (<C_INT32>coords.shape[0], 
                                                                        3,
                                                                        <C_INT32>points.shape[1]), 
                                                                        dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        for i from INT32_ZERO <= i < <C_INT32>points.shape[1]:
            _boxcoords_realdifferences_to_boxpoint_PBC( boxPoint  = points[:,i],
                                                      boxCoords   = coords,
                                                      basis       = basis,
                                                      differences = differences[:,:,i],
                                                      ncores      = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        for i from INT32_ZERO <= i < <C_INT32>points.shape[1]:
            _realcoords_realdifferences_to_realpoint_IBC( realPoint   = points[:,i],
                                                          realCoords  = coords,
                                                          differences = differences[:,:,i],
                                                          ncores      = ncores)
    # return differences
    return differences    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_differences_to_multi_indexcoords( ndarray[C_INT32, ndim=1]   indexes,
                                            ndarray[C_FLOAT32, ndim=2] coords not None,
                                            C_FLOAT32[:,:]             basis not None,
                                            bint                       isPBC,
                                            bint                       allAtoms = True,
                                            C_INT32                    ncores = 1):                       
    """
    Compute differences between one atomic coordinates arrays to a points coordinates   
    given their indexes in the coordinates array and taking into account periodic or 
    infinite boundary conditions. Difference is calculated as the following:
    
    .. math::
            differences[i,:,k] = boundaryConditions( coords[indexes[k],:] - coords[i,:] )
    
    :Arguments:
       #. indexes (int32 (k,3) numpy.ndarray): The atomic coordinates indexes array.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. differences (float32 (n,3,k) numpy.ndarray): The computed differences array.   
    """
    cdef C_INT32 i
    cdef ndarray[C_FLOAT32,  mode="c", ndim=3] differences = np.empty( (<C_INT32>coords.shape[0], 
                                                                        3,
                                                                        <C_INT32>indexes.shape[0]), 
                                                                        dtype=NUMPY_FLOAT32)                                                                      
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        for i from INT32_ZERO <= i < <C_INT32>indexes.shape[0]:
            _boxcoords_realdifferences_to_indexcoords_PBC( atomIndex   = indexes[i],
                                                           boxCoords   = coords,
                                                           basis       = basis,
                                                           differences = differences[:,:,i],
                                                           allAtoms    = allAtoms,
                                                           ncores      = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        for i from INT32_ZERO <= i < <C_INT32>indexes.shape[0]:
            _realcoords_realdifferences_to_indexcoords_IBC( atomIndex   = indexes[i],
                                                            realCoords  = coords,
                                                            differences = differences[:,:,i],
                                                            allAtoms    = allAtoms,
                                                            ncores      = ncores)
    # return differences
    return differences 
    
    

    
############################################################################################        
############################### PYTHON DISTANCE DEFINITIONS ################################
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def point_to_point_distance( ndarray[C_FLOAT32, ndim=1] point1  not None,
                             ndarray[C_FLOAT32, ndim=1] point2 not None,
                             C_FLOAT32[:,:]             basis not None,
                             bint                       isPBC, 
                             C_INT32                    ncores = 1):                       
    
    """
    Compute distances between one atomic coordinates arrays to a point coordinates  
    taking into account periodic or infinite boundary conditions. Distances is 
    calculated as the following:
    
    .. math::
            distances[i] = \sqrt{ \sum_{d}^{3}{ boundaryConditions( point[0,d] - coords[i,d] )^{2}} }
    
    :Arguments:
       #. point (float32 (1,3) numpy.ndarray): The atomic coordinates point.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. distances (float32 (n,) numpy.ndarray): The computed distances array.   
    """
    # if periodic boundary conditions, coords must be in box
    if isPBC:
       return _boxcoords_realdistance_to_boxpoint_PBC( boxPoint1 = point1,
                                                       boxPoint2 = point2,
                                                       basis     = basis)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        return _realcoords_realdistance_to_realpoint_IBC( realPoint1  = point1,
                                                          realPoint2  = point2)



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_distances_to_point( ndarray[C_FLOAT32, ndim=1] point  not None,
                              ndarray[C_FLOAT32, ndim=2] coords not None,
                              C_FLOAT32[:,:]             basis not None,
                              bint                       isPBC, 
                              C_INT32                    ncores = 1):                       
    
    """
    Compute distances between one atomic coordinates arrays to a point coordinates  
    taking into account periodic or infinite boundary conditions. Distances is 
    calculated as the following:
    
    .. math::
            distances[i] = \sqrt{ \sum_{d}^{3}{ boundaryConditions( point[0,d] - coords[i,d] )^{2}} }
    
    :Arguments:
       #. point (float32 (1,3) numpy.ndarray): The atomic coordinates point.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. distances (float32 (n,) numpy.ndarray): The computed distances array.   
    """
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] distances = np.empty((<C_INT32>coords.shape[0],), dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        _boxcoords_realdistances_to_boxpoint_PBC( boxPoint  = point,
                                                  boxCoords = coords,
                                                  basis     = basis,
                                                  distances = distances,
                                                  ncores    = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        _realcoords_realdistances_to_realpoint_IBC( realPoint  = point,
                                                    realCoords = coords,
                                                    distances  = distances,
                                                    ncores     = ncores)
    # return distances
    return distances


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_distances_to_indexcoords( C_INT32                    atomIndex,
                                    ndarray[C_FLOAT32, ndim=2] coords not None,
                                    C_FLOAT32[:,:]             basis not None,
                                    bint                       isPBC,
                                    bint                       allAtoms = True,
                                    C_INT32                    ncores = 1):                       
    
    """
    Compute distances between one atomic coordinates arrays to a points coordinates   
    given their indexes in the coordinates array and taking into account periodic or 
    infinite boundary conditions. Distances is calculated as the following:
    
    .. math::
            distances[i] = \sqrt{ \sum_{d}^{3}{ boundaryConditions( coords[atomIndex[i],d] - coords[i,d] )^{2}} }
    
    :Arguments:
       #. point (float32 (1,3) numpy.ndarray): The atomic coordinates point.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. distances (float32 (n,) numpy.ndarray): The computed distances array.   
    """
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] distances = np.empty((<C_INT32>coords.shape[0],), dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        _boxcoords_realdistances_to_indexcoords_PBC( atomIndex = atomIndex,
                                                     boxCoords = coords,
                                                     basis     = basis,
                                                     distances = distances,
                                                     allAtoms  = allAtoms,
                                                     ncores    = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        _realcoords_realdistances_to_indexcoords_IBC( atomIndex  = atomIndex,
                                                      realCoords = coords,
                                                      distances  = distances,
                                                      allAtoms   = allAtoms,
                                                      ncores     = ncores)
    # return distances
    return distances    


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_distances_to_multi_points( ndarray[C_FLOAT32, ndim=2] points not None,
                                     ndarray[C_FLOAT32, ndim=2] coords not None,
                                     C_FLOAT32[:,:]             basis not None,
                                     bint                       isPBC, 
                                     C_INT32                    ncores = 1):                       
    """
    Compute distances between one atomic coordinates arrays to a multiple points   
    coordinates taking into account periodic or infinite boundary conditions. 
    Distances is calculated as the following:
    
    .. math::
            distances[i,k] = \sqrt{ \sum_{d}^{3}{ boundaryConditions( points[k,d] - coords[i,d] )^{2}} }
            
    :Arguments:
       #. points (float32 (k,3) numpy.ndarray): The multiple atomic coordinates points.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. distances (float32 (n,) numpy.ndarray): The computed distances array.   
    """
    cdef C_INT32 i
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] distances = np.empty( (<C_INT32>coords.shape[0], 
                                                                      <C_INT32>points.shape[1]), 
                                                                      dtype=NUMPY_FLOAT32)
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        for i from INT32_ZERO <= i < <C_INT32>points.shape[1]:
            _boxcoords_realdistances_to_boxpoint_PBC( boxPoint  = points[:,i],
                                                      boxCoords = coords,
                                                      basis     = basis,
                                                      distances = distances[:,i],
                                                      ncores    = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        for i from INT32_ZERO <= i < <C_INT32>points.shape[1]:
            _realcoords_realdistances_to_realpoint_IBC( realPoint  = points[:,i],
                                                        realCoords = coords,
                                                        distances  = distances[:,i],
                                                        ncores     = ncores)
    # return distances
    return distances    
    

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def pairs_distances_to_multi_indexcoords( ndarray[C_INT32, ndim=1]   indexes,
                                          ndarray[C_FLOAT32, ndim=2] coords not None,
                                          C_FLOAT32[:,:]             basis not None,
                                          bint                       isPBC,
                                          bint                       allAtoms = True,
                                          C_INT32                    ncores = 1):                       
    """
    Compute distances between one atomic coordinates arrays to a points coordinates   
    given their indexes in the coordinates array and taking into account periodic or 
    infinite boundary conditions. Distances is calculated as the following:
    
    .. math::
            distances[i,k] = \sqrt{ \sum_{d}^{3}{ boundaryConditions( coords[indexes[k],:]  - coords[i,d] )^{2}} }
            
    :Arguments:
       #. indexes (int32 (k,3) numpy.ndarray): The atomic coordinates indexes array.
       #. coords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. ncores (int32) [default=1]: The number of cores to use. 
       
    :Returns:
       #. distances (float32 (n,) numpy.ndarray): The computed distances array.   
    """
    cdef C_INT32 i
    cdef ndarray[C_FLOAT32,  mode="c", ndim=2] distances = np.empty( (<C_INT32>coords.shape[0], 
                                                                      <C_INT32>indexes.shape[0]), 
                                                                      dtype=NUMPY_FLOAT32)                                                                      
    # if periodic boundary conditions, coords must be in box
    if isPBC:
        for i from INT32_ZERO <= i < <C_INT32>indexes.shape[0]:
            _boxcoords_realdistances_to_indexcoords_PBC( atomIndex = indexes[i],
                                                         boxCoords = coords,
                                                         basis     = basis,
                                                         distances = distances[:,i],
                                                         allAtoms  = allAtoms,
                                                         ncores    = ncores)
    # if infinite boundary conditions coords must be in Cartesian normal space
    else:
        for i from INT32_ZERO <= i < <C_INT32>indexes.shape[0]:
            _realcoords_realdistances_to_indexcoords_IBC( atomIndex  = indexes[i],
                                                          realCoords = coords,
                                                          distances  = distances[:,i],
                                                          allAtoms   = allAtoms,
                                                          ncores     = ncores)
    # return distances
    return distances 



    