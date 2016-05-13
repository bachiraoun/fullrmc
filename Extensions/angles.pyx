"""
This is a C compiled Cython generated module to calculate angles constraints. 
It contains the following methods.

**single_angles**: It calculates the angles constraint of a single atom.
    :Arguments:
       #. centralAtomIndex (int32): The central atom index.
       #. leftIndexes (int32 array): The centralAtom's angles left atoms indexes.
       #. rightIndexes (int32 array): The centralAtom's angles right atoms indexes. number of leftIndexes must be equal to number of rightIndexes.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. lowerLimit (float32 array): The (numberOfLeftIndexes) array for lower limit or minimum bond length allowed.
       #. upperLimit (float32 array): The (numberOfLeftIndexes) array for upper limit or maximum bond length allowed.
       #. reduceAngleToUpper (bool): Whether to reduce angle found out of limits to the difference between the angle and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceAngleToLower (bool): Whether to reduce angle found out of limits to the difference between the angle and the lower limit. When True, this flag may lose its priority for reduceAngleToUpper if the later is True. DEFAULT: False
                  
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. angles: The calculated angles (rad).
          #. reducedAngles: The reduced angles (rad)
           
**full_angles**: It calculates the angles constraint of all atoms given a angles dictionary.
    :Arguments:
       #. angles (python dictionary): The angles dictionary. Where keys are central atoms indexes and values are dictionary of leftIndexes array, rightIndexes array, lowerLimit array, upperLimit array
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. reduceAngleToUpper (bool): Whether to reduce angle found out of limits to the difference between the angle and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceAngleToLower (bool): Whether to reduce angle found out of limits to the difference between the angle and the lower limit. When True, this flag may lose its priority for reduceAngleToUpper if the later is True. DEFAULT: False
                
    :Returns:
       #. result (python dictionary): where keys are central atoms indexes and values are dictionaries of exactly two keys as such.\n
          #. angles: The calculated bonds length
          #. reducedAngles: The reduced bonds length
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
def single_angles( C_INT32 centralAtomIndex,
                   ndarray[C_INT32, ndim=1] leftIndexes not None, 
                   ndarray[C_INT32, ndim=1] rightIndexes not None, 
                   ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                   ndarray[C_FLOAT32, ndim=2] basis not None,
                   np.ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                   np.ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                   bint reduceAngleToUpper = False,
                   bint reduceAngleToLower = False):
    # declare variables
    cdef C_INT32 i, numberOfIndexes, inLoopLeftAtomIndex, inLoopRightAtomIndex
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 sign_x, sign_y, sign_z
    cdef C_FLOAT32 vectorNorm, dot, angle, reducedAngle
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    cdef C_FLOAT32 leftVector_x, leftVector_y, leftVector_z
    cdef C_FLOAT32 rightVector_x, rightVector_y, rightVector_z
    # get number of bonded indexes
    numberOfIndexes = <C_INT32>len(leftIndexes)
    # create angles and reducedAngles
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] angles        = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedAngles = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    # get point coordinates
    atomBox_x = boxCoords[centralAtomIndex,0]
    atomBox_y = boxCoords[centralAtomIndex,1]
    atomBox_z = boxCoords[centralAtomIndex,2]
    # loop
    for i from INT_ZERO <= i < numberOfIndexes:
        # get inLoopAtomIndex
        inLoopLeftAtomIndex  = leftIndexes[i]
        inLoopRightAtomIndex = rightIndexes[i]
        # calculate left vector
        box_dx = boxCoords[inLoopLeftAtomIndex,0]-atomBox_x
        box_dy = boxCoords[inLoopLeftAtomIndex,1]-atomBox_y
        box_dz = boxCoords[inLoopLeftAtomIndex,2]-atomBox_z
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        leftVector_x = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        leftVector_y = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        leftVector_z = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # normalize left vector
        vectorNorm = sqrt(leftVector_x*leftVector_x + leftVector_y*leftVector_y + leftVector_z*leftVector_z)
        if vectorNorm==0:
            raise Exception("Computing angle, vector between %i and %i atoms is found to have null length"%(centralAtomIndex, inLoopLeftAtomIndex))
        leftVector_x /= vectorNorm
        leftVector_y /= vectorNorm
        leftVector_z /= vectorNorm
        # calculate right vector
        box_dx = boxCoords[inLoopRightAtomIndex,0]-atomBox_x
        box_dy = boxCoords[inLoopRightAtomIndex,1]-atomBox_y
        box_dz = boxCoords[inLoopRightAtomIndex,2]-atomBox_z
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        rightVector_x = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        rightVector_y = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        rightVector_z = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # normalize left vector
        vectorNorm = sqrt(leftVector_x*leftVector_x + leftVector_y*leftVector_y + leftVector_z*leftVector_z)
        if vectorNorm==0:
            raise Exception("Computing angle, vector between %i and %i atoms is found to have null length"%(centralAtomIndex, inLoopLeftAtomIndex))
        rightVector_x /= vectorNorm
        rightVector_y /= vectorNorm
        rightVector_z /= vectorNorm
        # compute dot product
        dot = leftVector_x*rightVector_x + leftVector_y*rightVector_y + leftVector_z*rightVector_z
        # calculate angle
        angle = np.arccos( np.clip( dot ,-1, 1 ) )        
        # compute reduced angle
        lower = lowerLimit[i]
        upper = upperLimit[i]
        if angle>=lower and angle<=upper:
            reducedAngle = FLOAT_ZERO     
        elif reduceAngleToUpper:
            reducedAngle = abs(upper-angle)
        elif reduceAngleToLower:
            reducedAngle = abs(lower-angle)
        else:
            if angle > (lower+upper)/FLOAT_TWO:
                reducedAngle = abs(upper-angle)
            else:
                reducedAngle = abs(lower-angle)
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
def full_angles( dict anglesDict not None, 
                 ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                 ndarray[C_FLOAT32, ndim=2] basis not None,
                 bint reduceAngleToUpper = False,
                 bint reduceAngleToLower = False):    
    anglesResult = {}
    for atomIndex, angle in anglesDict.items():
        result = single_angles( centralAtomIndex=atomIndex ,
                                leftIndexes=angle["leftIndexes"] , 
                                rightIndexes=angle["rightIndexes"] , 
                                boxCoords=boxCoords,
                                basis=basis ,
                                lowerLimit=angle["lower"] ,
                                upperLimit=angle["upper"] ,
                                reduceAngleToUpper = reduceAngleToUpper,
                                reduceAngleToLower = reduceAngleToLower)
        # update dictionary
        anglesResult[atomIndex] = result
    return anglesResult
    

                                           
   
    