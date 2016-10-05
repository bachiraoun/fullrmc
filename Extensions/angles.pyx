"""
This is a C compiled module to compute bonded atoms angle.
"""      
from libc.math cimport sqrt, fabs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from fullrmc.Core.pairs_distances import pairs_differences_to_point

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
cdef C_INT32   INT_ZERO        = 0


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
def single_angles_diffs( ndarray[C_FLOAT32, ndim=2]    leftVectors not None, 
                         ndarray[C_FLOAT32, ndim=2]    rightVectors not None,
                         np.ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                         np.ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                         bint                          reduceAngleToUpper = False,
                         bint                          reduceAngleToLower = False):
    """
    Computes the angles constraint given bonded atoms vectors.
    
    :Arguments:
       #. leftVectors (float32 array): The left vectors array.
       #. rightVectors (float32 array): The right vectors array.
       #. lowerLimit (float32 array): The (numberOfLeftIndexes) array for lower limit or minimum bond length allowed.
       #. upperLimit (float32 array): The (numberOfLeftIndexes) array for upper limit or maximum bond length allowed.
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
    cdef C_FLOAT32 leftNorm, rightNorm, dot, angle, reducedAngle
    cdef C_FLOAT32 leftVector_x, leftVector_y, leftVector_z
    cdef C_FLOAT32 rightVector_x, rightVector_y, rightVector_z
    # get number of bonded indexes
    numberOfIndexes = <C_INT32>leftVectors.shape[0]
    # create angles and reducedAngles
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] angles        = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedAngles = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    # loop
    for i from INT_ZERO <= i < numberOfIndexes:
        # compute left vector norm
        leftVector_x = leftVectors[i,0]
        leftVector_y = leftVectors[i,1]
        leftVector_z = leftVectors[i,2]
        leftNorm     = sqrt(leftVector_x*leftVector_x + leftVector_y*leftVector_y + leftVector_z*leftVector_z)
        if leftNorm==0:
            raise Exception("Computing angle, left vector found to have null length")
        # compute right vector norm
        rightVector_x = rightVectors[i,0]
        rightVector_y = rightVectors[i,1]
        rightVector_z = rightVectors[i,2]
        rightNorm     = sqrt(rightVector_x*rightVector_x + rightVector_y*rightVector_y + rightVector_z*rightVector_z)
        if rightNorm==0:
            raise Exception("Computing angle, right vector found to have null length")
        # compute dot product
        dot = leftVector_x*rightVector_x + leftVector_y*rightVector_y + leftVector_z*rightVector_z
        # calculate angle
        dot  /= (leftNorm*rightNorm)
        angle = np.arccos( np.clip( dot ,-1, 1 ) )  # np.arccos( dot ) clip for floating errors
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
def full_angles_coords( dict                       anglesDict not None, 
                        ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                        ndarray[C_FLOAT32, ndim=2] basis not None,
                        bint                       isPBC,
                        bint                       reduceAngleToUpper = False,
                        bint                       reduceAngleToLower = False,
                        C_INT32                    ncores = 1):    
    """
    Computes the angles constraint given bonded atoms vectors.
    
    :Arguments:
       #. anglesDict (python dict): The angles dictionary. Where keys are central atoms indexes and values 
          are dictionary of leftIndexes array, rightIndexes array, lowerLimit array, upperLimit array
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
    for atomIndex, angle in anglesDict.items():
        leftVectors = pairs_differences_to_point( point  = boxCoords[ atomIndex ], 
                                                  coords = boxCoords[ angle["leftIndexes"] ],
                                                  basis  = basis,
                                                  isPBC  = isPBC,
                                                  ncores = ncores)  
        rightVectors = pairs_differences_to_point( point  = boxCoords[ atomIndex ], 
                                                   coords = boxCoords[ angle["rightIndexes"] ],
                                                   basis  = basis,
                                                   isPBC  = isPBC,
                                                   ncores = ncores)                                        
        result = single_angles_diffs( leftVectors        = leftVectors , 
                                      rightVectors       = rightVectors , 
                                      lowerLimit         = angle["lower"] ,
                                      upperLimit         = angle["upper"] ,
                                      reduceAngleToUpper = reduceAngleToUpper,
                                      reduceAngleToLower = reduceAngleToLower)
        # update dictionary
        anglesResult[atomIndex] = result
    return anglesResult




    