"""
This is a C compiled module to compute improper angles.
"""            
from libc.math cimport sqrt, fabs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from fullrmc.Core.pairs_distances import pairs_differences_to_point, from_to_points_differences

# declare types
NUMPY_FLOAT32 = np.float32
NUMPY_INT32   = np.int32
ctypedef np.float32_t C_FLOAT32
ctypedef np.int32_t   C_INT32

# declare constants
cdef C_FLOAT32 FLOAT_NEG_ONE   = -1.0
cdef C_FLOAT32 FLOAT_ZERO      = 0.0
cdef C_FLOAT32 FLOAT_ONE       = 1.0
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_FLOAT32 BOX_LENGTH      = 1.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5
cdef C_FLOAT32 PI              = 3.141592653589793
cdef C_FLOAT32 PI_2            = PI/2

 
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
def single_improper_angles_diffs( ndarray[C_FLOAT32, ndim=2]    improperVectors not None, 
                                  ndarray[C_FLOAT32, ndim=2]    oxVectors not None, 
                                  ndarray[C_FLOAT32, ndim=2]    oyVectors not None, 
                                  np.ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                                  np.ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                                  bint                          reduceAngleToUpper = False,
                                  bint                          reduceAngleToLower = False):
    """
    Computes the improper angles constraint between an improper atom and a plane atoms.
    The plane normal vector is calculated using the right-hand rule where (thumb=ox vector), 
    (index=oy vector) hence (oz=normal=second finger)
    
    :Arguments:
       #. improperVectors  (float32 array): The improper vectors array.
       #. oxVectors  (float32 array): The ox vectors array.
       #. oyVectors  (float32 array): The oy vectors array.
       #. lowerLimit (float32 array): The (oAtomIndex) array for lower limit or minimum bond length allowed.
       #. upperLimit (float32 array): The (oAtomIndex) array for upper limit or maximum bond length allowed.
       #. reduceAngleToUpper (bool): Whether to reduce angle found out of limits to the difference between the angle and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceAngleToLower (bool): Whether to reduce angle found out of limits to the difference between the angle and the lower limit. When True, this flag may lose its priority for reduceAngleToUpper if the later is True. DEFAULT: False
                  
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. angles: The calculated angles (rad).
          #. reducedAngles: The reduced angles (rad)
    """
    # declare variables
    cdef C_INT32 i, numberOfIndexes
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 vectorNorm, dot, angle, reducedAngle
    cdef C_FLOAT32 improperVector_x, improperVector_y, improperVector_z
    cdef C_FLOAT32 oxVector_x, oxVector_y, oxVector_z
    cdef C_FLOAT32 oyVector_x, oyVector_y, oyVector_z
    cdef C_FLOAT32 ozVector_x, ozVector_y, ozVector_z
    # get number of bonded indexes
    numberOfIndexes = <C_INT32>improperVectors.shape[0]
    # create angles and reducedAngles
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] angles        = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedAngles = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    # loop
    for i from 0 <= i < numberOfIndexes:
        ########################### normalize improper vector ###########################
        improperVector_x = improperVectors[i,0]
        improperVector_y = improperVectors[i,1]
        improperVector_z = improperVectors[i,2]
        vectorNorm = sqrt(improperVector_x*improperVector_x + improperVector_y*improperVector_y + improperVector_z*improperVector_z)
        if vectorNorm==0:
            raise Exception("Computing angle, improper vector found to have null length")
        improperVector_x /= vectorNorm
        improperVector_y /= vectorNorm
        improperVector_z /= vectorNorm
        ############################## normalize ox vector ##############################
        oxVector_x = oxVectors[i,0]
        oxVector_y = oxVectors[i,1]
        oxVector_z = oxVectors[i,2]
        vectorNorm = sqrt(oxVector_x*oxVector_x + oxVector_y*oxVector_y + oxVector_z*oxVector_z)
        if vectorNorm==0:
            raise Exception("Computing angle, ox vector found to have null length")
        oxVector_x /= vectorNorm
        oxVector_y /= vectorNorm
        oxVector_z /= vectorNorm
        ############################## normalize oy vector ##############################
        oyVector_x = oyVectors[i,0]
        oyVector_y = oyVectors[i,1]
        oyVector_z = oyVectors[i,2]
        vectorNorm = sqrt(oyVector_x*oyVector_x + oyVector_y*oyVector_y + oyVector_z*oyVector_z)
        if vectorNorm==0:
            raise Exception("Computing angle, oy vector found to have null length")
        oyVector_x /= vectorNorm
        oyVector_y /= vectorNorm
        oyVector_z /= vectorNorm
        ############################### compute oz vector ###############################
        # compute OZ vector as a×b= (a2b3−a3b2)i−(a1b3−a3b1)j+(a1b2−a2b1)k.
        ozVector_x =  oxVector_y*oyVector_z - oxVector_z*oyVector_y
        ozVector_y = -oxVector_x*oyVector_z + oxVector_z*oyVector_x
        ozVector_z =  oxVector_x*oyVector_y - oxVector_y*oyVector_x
        ################################ angle ################################
        # compute dot product
        dot = improperVector_x*ozVector_x + improperVector_y*ozVector_y + improperVector_z*ozVector_z
        # calculate angle
        angle = PI_2 - <C_FLOAT32>np.arccos( np.clip( dot ,-1, 1 ) ) # PI_2 - <C_FLOAT32>np.arccos( dot ) clipped for floating errors
        # compute reduced angle
        lower = lowerLimit[i]
        upper = upperLimit[i]
        if angle>=lower and angle<=upper:
            reducedAngle = FLOAT_ZERO     
        elif reduceAngleToUpper:
            reducedAngle = fabs(upper-angle)
        elif reduceAngleToLower:
            reducedAngle = fabs(lower-angle)
        else:
            if angle > (lower+upper)/FLOAT_TWO:
                reducedAngle = fabs(upper-angle)
            else:
                reducedAngle = fabs(lower-angle)
        # increment histograms
        angles[i]        = angle
        reducedAngles[i] = reducedAngle
    # return result
    return {"angles":angles ,"reducedAngles":reducedAngles}


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_improper_angles_coords( dict                       anglesDict not None, 
                                 ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                 ndarray[C_FLOAT32, ndim=2] basis not None,
                                 bint                       isPBC,
                                 bint                       reduceAngleToUpper = False,
                                 bint                       reduceAngleToLower = False,
                                 C_INT32                    ncores = 1):    
    """
    Computes the improper angles constraint between an improper atom and a plane atoms.
    The plane normal vector is calculated using the right-hand rule where (thumb=ox vector), 
    (index=oy vector) hence (oz=normal=second finger)
    
    :Arguments:
       #. anglesDict (python dictionary): The angles dictionary. Where keys are the improper atoms indexes 
          and values are dictionary of oAtomIndex array, xAtomIndex array, yAtomIndex array, 
          lowerLimit array, upperLimit array
       #. boxCoords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. reduceAngleToUpper (bool): Whether to reduce angle found out of limits to the difference between the angle and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceAngleToLower (bool): Whether to reduce angle found out of limits to the difference between the angle and the lower limit. When True, this flag may lose its priority for reduceAngleToUpper if the later is True. DEFAULT: False
       #. ncores (int32) [default=1]: The number of cores to use.
       
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. angles: The calculated angles (rad).
          #. reducedAngles: The reduced angles (rad)
    """
    
    anglesResult = {}
    for improperAtomIndex, angle in anglesDict.items():
        # improperVectors of dim (n oIndexes, 3)
        improperVectors = -pairs_differences_to_point( point  = boxCoords[ improperAtomIndex ], 
                                                       coords = boxCoords[ angle["oIndexes"] ],
                                                       basis  = basis,
                                                       isPBC  = isPBC,
                                                       ncores = ncores)  
        # oxVectors of dim (n oIndexes, 3)
        oxVectors = from_to_points_differences(  pointsFrom = boxCoords[ angle["oIndexes"] ],
                                                 pointsTo   = boxCoords[ angle["xIndexes"] ],
                                                 basis      = basis,
                                                 isPBC      = isPBC,
                                                 ncores     = ncores)     
        # oyVectors of dim (n oIndexes, 3)
        oyVectors = from_to_points_differences(  pointsFrom = boxCoords[ angle["oIndexes"] ],
                                                 pointsTo   = boxCoords[ angle["yIndexes"] ],
                                                 basis      = basis,
                                                 isPBC      = isPBC,
                                                 ncores     = ncores)     
        result = single_improper_angles_diffs( improperVectors    = improperVectors ,
                                               oxVectors          = oxVectors , 
                                               oyVectors          = oyVectors ,
                                               lowerLimit         = angle["lower"] ,
                                               upperLimit         = angle["upper"] ,
                                               reduceAngleToUpper = reduceAngleToUpper,
                                               reduceAngleToLower = reduceAngleToLower)                                  
        # update dictionary
        anglesResult[improperAtomIndex] = result
    return anglesResult













   
    