"""
This is a C compiled Cython generated module to calculate angles constraints. 
It contains the following methods.

**single_improper_angles**: It calculates the improper angles constraint between an improper atom and a plane atoms.
   The plane normal vector is calculated using the right-hand rule where thumb=ox vector, index=oy vector hence oz=normal=second finger
    :Arguments:
       #. improperAtomIndex (int32): The atom index that must be in plane.
       #. oAtomIndex (int32 array): The first atom index to build the plane considered as origin.
       #. xAtomIndex (int32 array): The second atom index to build the plane considered as ox vector. number of oAtomIndex must be equal to number of xAtomIndex.
       #. yAtomIndex (int32 array): The third atom index to build the plane considered as oy vector. number of xAtomIndex must be equal to number of yAtomIndex.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. lowerLimit (float32 array): The (oAtomIndex) array for lower limit or minimum bond length allowed.
       #. upperLimit (float32 array): The (oAtomIndex) array for upper limit or maximum bond length allowed.
       #. reduceAngleToUpper (bool): Whether to reduce angle found out of limits to the difference between the angle and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceAngleToLower (bool): Whether to reduce angle found out of limits to the difference between the angle and the lower limit. When True, this flag may lose its priority for reduceAngleToUpper if the later is True. DEFAULT: False
                  
    :Returns:
       #. result (python dictionary): It has only two keys.\n
          #. angles: The calculated angles (rad).
          #. reducedAngles: The reduced angles (rad)
           
**full_improper_angles**: It calculates the improper angles constraint of all atoms given a angles dictionary.
    :Arguments:
       #. angles (python dictionary): The angles dictionary. Where keys are the improper atoms indexes and values are dictionary of oAtomIndex array, xAtomIndex array, yAtomIndex array, lowerLimit array, upperLimit array
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
def single_improper_angles( C_INT32 improperAtomIndex,
                            ndarray[C_INT32, ndim=1] oAtomIndex not None, 
                            ndarray[C_INT32, ndim=1] xAtomIndex not None, 
                            ndarray[C_INT32, ndim=1] yAtomIndex not None, 
                            ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                            ndarray[C_FLOAT32, ndim=2] basis not None,
                            np.ndarray[C_FLOAT32, ndim=1] lowerLimit not None,
                            np.ndarray[C_FLOAT32, ndim=1] upperLimit not None,
                            bint reduceAngleToUpper = False,
                            bint reduceAngleToLower = False):
    # declare variables
    cdef C_INT32 i, numberOfIndexes, inLoopOAtomIndex, inLoopXAtomIndex, inLoopYAtomIndex
    cdef C_FLOAT32 upper, lower
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 sign_x, sign_y, sign_z
    cdef C_FLOAT32 vectorNorm, dot, angle, reducedAngle
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    cdef C_FLOAT32 center_x, center_y, center_z
    cdef C_FLOAT32 improperVector_x, improperVector_y, improperVector_z
    cdef C_FLOAT32 oxVector_x, oxVector_y, oxVector_z
    cdef C_FLOAT32 oyVector_x, oyVector_y, oyVector_z
    cdef C_FLOAT32 ozVector_x, ozVector_y, ozVector_z
    # get number of bonded indexes
    numberOfIndexes = <C_INT32>len(oAtomIndex)
    # create angles and reducedAngles
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] angles        = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reducedAngles = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    # get point coordinates
    atomBox_x = boxCoords[improperAtomIndex,0]
    atomBox_y = boxCoords[improperAtomIndex,1]
    atomBox_z = boxCoords[improperAtomIndex,2]
    # loop
    for i from 0 <= i < numberOfIndexes:
        # get inLoopAtomIndex
        inLoopOAtomIndex = oAtomIndex[i]
        inLoopXAtomIndex = xAtomIndex[i]
        inLoopYAtomIndex = yAtomIndex[i]
        ########################### improper vector ###########################
        box_dx = boxCoords[i,0]-atomBox_x
        box_dy = boxCoords[i,1]-atomBox_y
        box_dz = boxCoords[i,2]-atomBox_z
        sign_x = FLOAT_ONE if box_dx>=0 else FLOAT_NEG_ONE
        sign_y = FLOAT_ONE if box_dy>=0 else FLOAT_NEG_ONE
        sign_z = FLOAT_ONE if box_dz>=0 else FLOAT_NEG_ONE
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        improperVector_x = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        improperVector_y = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        improperVector_z = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # normalize improper vector
        vectorNorm = sqrt(improperVector_x*improperVector_x + improperVector_y*improperVector_y + improperVector_z*improperVector_z)
        if vectorNorm==0:
            raise Exception("Computing improper angle, vector between %i and %i atoms is found to have null length"%(inLoopOAtomIndex, improperAtomIndex))
        improperVector_x /= vectorNorm
        improperVector_y /= vectorNorm
        improperVector_z /= vectorNorm
        ############################## ox vector ##############################
        box_dx = boxCoords[inLoopXAtomIndex,0] - boxCoords[inLoopOAtomIndex,0]
        box_dy = boxCoords[inLoopXAtomIndex,1] - boxCoords[inLoopOAtomIndex,0]
        box_dz = boxCoords[inLoopXAtomIndex,2] - boxCoords[inLoopOAtomIndex,0]
        sign_x = FLOAT_ONE if box_dx>=0 else FLOAT_NEG_ONE
        sign_y = FLOAT_ONE if box_dy>=0 else FLOAT_NEG_ONE
        sign_z = FLOAT_ONE if box_dz>=0 else FLOAT_NEG_ONE
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        oxVector_x = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        oxVector_y = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        oxVector_z = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # normalize improper vector
        vectorNorm = sqrt(oxVector_x*oxVector_x + oxVector_y*oxVector_y + oxVector_z*oxVector_z)
        if vectorNorm==0:
            raise Exception("Computing improper angle, vector between %i and %i atoms is found to have null length"%(inLoopOAtomIndex, inLoopXAtomIndex))
        oxVector_x /= vectorNorm
        oxVector_y /= vectorNorm
        oxVector_z /= vectorNorm
        ############################## oy vector ##############################
        box_dx = boxCoords[inLoopYAtomIndex,0] - boxCoords[inLoopOAtomIndex,0]
        box_dy = boxCoords[inLoopYAtomIndex,1] - boxCoords[inLoopOAtomIndex,0]
        box_dz = boxCoords[inLoopYAtomIndex,2] - boxCoords[inLoopOAtomIndex,0]
        sign_x = FLOAT_ONE if box_dx>=0 else FLOAT_NEG_ONE
        sign_y = FLOAT_ONE if box_dy>=0 else FLOAT_NEG_ONE
        sign_z = FLOAT_ONE if box_dz>=0 else FLOAT_NEG_ONE
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        oyVector_x = sign_x * (box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0])
        oyVector_y = sign_y * (box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1])
        oyVector_z = sign_z * (box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2])
        # normalize ox vector
        vectorNorm = sqrt(oyVector_x*oyVector_x + oyVector_y*oyVector_y + oyVector_z*oyVector_z)
        if vectorNorm==0:
            raise Exception("Computing improper angle, vector between %i and %i atoms is found to have null length"%(inLoopOAtomIndex, inLoopYAtomIndex))
        oyVector_x /= vectorNorm
        oyVector_y /= vectorNorm
        oyVector_z /= vectorNorm
        ############################## oz vector ##############################
        # compute OZ vector as a×b= (a2b3−a3b2)i−(a1b3−a3b1)j+(a1b2−a2b1)k.
        ozVector_x =  oxVector_y*oyVector_z - oxVector_z*oyVector_y
        ozVector_y = -oxVector_x*oyVector_z + oxVector_z*oyVector_x
        ozVector_z =  oxVector_x*oyVector_y - oxVector_y*oyVector_x
        ################################ angle ################################
        # compute dot product
        dot = improperVector_x*ozVector_x + improperVector_y*ozVector_y + improperVector_z*ozVector_z
        # calculate angle
        angle = PI_2 - <C_FLOAT32>np.arccos( np.clip( dot ,-1, 1 ) )        
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
def full_improper_angles( dict anglesDict not None, 
                          ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                          ndarray[C_FLOAT32, ndim=2] basis not None,
                          bint reduceAngleToUpper = False,
                          bint reduceAngleToLower = False):    
    anglesResult = {}
    for atomIndex, angle in anglesDict.items():
        result = single_improper_angles( improperAtomIndex=atomIndex ,
                                         oAtomIndex=angle["oIndexes"] , 
                                         xAtomIndex=angle["xIndexes"] , 
                                         yAtomIndex=angle["yIndexes"] , 
                                         boxCoords=boxCoords,
                                         basis=basis ,
                                         lowerLimit=angle["lower"] ,
                                         upperLimit=angle["upper"] ,
                                         reduceAngleToUpper = reduceAngleToUpper,
                                         reduceAngleToLower = reduceAngleToLower)
        # update dictionary
        anglesResult[atomIndex] = result
    return anglesResult
    

                                           
   
    