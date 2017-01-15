"""
This is a C compiled module to compute improper angles.
"""            
from libc.math cimport sqrt, fabs
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy cimport ndarray
from fullrmc.Core.pairs_distances import  pair_difference_to_point

# declare types
NUMPY_FLOAT32 = np.float32
NUMPY_INT32   = np.int32
ctypedef np.float32_t C_FLOAT32
ctypedef np.int32_t   C_INT32

# declare constants
cdef C_FLOAT32 FLOAT_ZERO      = 0.0
cdef C_FLOAT32 FLOAT_TWO       = 2.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5
cdef C_FLOAT32 PI              = 3.141592653589793
cdef C_FLOAT32 PI_DEG          = 180.
cdef C_FLOAT32 FULL_CIRCLE     = 360.
cdef C_FLOAT32 RAD_TO_DEG      = PI_DEG/PI
cdef C_INT32   INT_ONE         = 1

 
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
cdef _single_dihedral_angle( ndarray[C_FLOAT32, ndim=1]    b1 , 
                             ndarray[C_FLOAT32, ndim=1]    b2 , 
                             ndarray[C_FLOAT32, ndim=1]    b3 , 
                             C_FLOAT32                     lowerLimit1,
                             C_FLOAT32                     upperLimit1,
                             C_FLOAT32                     lowerLimit2,
                             C_FLOAT32                     upperLimit2,
                             C_FLOAT32                     lowerLimit3,
                             C_FLOAT32                     upperLimit3,
                             ndarray[C_FLOAT32, ndim=1]    angles ,
                             ndarray[C_FLOAT32, ndim=1]    reducedAngles ,
                             C_INT32                       index,
                             bint                          reduceAngleToUpper = False,
                             bint                          reduceAngleToLower = False):

    # declare variables
    cdef C_FLOAT32 n1_x, n1_y, n1_z
    cdef C_FLOAT32 n2_x, n2_y, n2_z
    cdef C_FLOAT32 m1_x, m1_y, m1_z
    cdef C_FLOAT32 var, norm, x, y, angle, reducedAngle    
    # normalize b1
    norm = sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2])
    if norm==0:
        raise Exception("dihedral b1 vector is found to be zero length")
    b1[0] /= norm
    b1[1] /= norm
    b1[2] /= norm
    # normalize b2
    norm = sqrt(b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2])
    if norm==0:
        raise Exception("dihedral b2 vector is found to be zero length")
    b2[0] /= norm
    b2[1] /= norm
    b2[2] /= norm
    # normalize b3
    norm = sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2])
    if norm==0:
        raise Exception("dihedral b3 vector is found to be zero length")
    b3[0] /= norm
    b3[1] /= norm
    b3[2] /= norm
    # compute n1 (b1 X b2)
    n1_x = b1[1] * b2[2] - b1[2] * b2[1]
    n1_y = b1[2] * b2[0] - b1[0] * b2[2]
    n1_z = b1[0] * b2[1] - b1[1] * b2[0]
    # compute n2 (b2 X b3)
    n2_x = b2[1] * b3[2] - b2[2] * b3[1]
    n2_y = b2[2] * b3[0] - b2[0] * b3[2]
    n2_z = b2[0] * b3[1] - b2[1] * b3[0]
    # compute m1 (n1 X b2) so n1,m1 and b2 form an orthogonal  frame
    m1_x = n1_y * b2[2] - n1_z * b2[1]
    m1_y = n1_z * b2[0] - n1_x * b2[2]
    m1_z = n1_x * b2[1] - n1_y * b2[0]
    # compute n2 in (n1,m1,b2) frame as x=n1.n2 and y=m1.n2 and z should be 0
    x = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z
    y = m1_x*n2_x + m1_y*n2_y + m1_z*n2_z
    # compute dihedral angle as atan2(y,x) and convert to degrees
    angle = <C_FLOAT32>np.arctan2( y,x )* RAD_TO_DEG
    # convert to between 0-360
    if angle<0:
        angle = FULL_CIRCLE+angle
    # compute reduced angle trying first shell
    if (upperLimit1<lowerLimit1) and angle>upperLimit1:
        reducedAngle = FLOAT_ZERO
    elif angle>=lowerLimit1 and angle<=upperLimit1:
        reducedAngle = FLOAT_ZERO
    else:
        if reduceAngleToUpper:
            reducedAngle = <C_FLOAT32>fabs(upperLimit1-angle)
        elif reduceAngleToLower:
            reducedAngle = <C_FLOAT32>fabs(lowerLimit1-angle)
        else:
            if angle > (lowerLimit1+upperLimit1)/FLOAT_TWO:
                reducedAngle = <C_FLOAT32>fabs(upperLimit1-angle)
            else:
                reducedAngle = <C_FLOAT32>fabs(lowerLimit1-angle)
    # compute reduced angle trying second shell
    if reducedAngle!=FLOAT_ZERO:
        if (upperLimit2<lowerLimit2) and angle>upperLimit2:
            reducedAngle = FLOAT_ZERO
        elif angle>=lowerLimit2 and angle<=upperLimit2:
            reducedAngle = FLOAT_ZERO
        else:
            if reduceAngleToUpper:
                var = <C_FLOAT32>fabs(upperLimit2-angle)
            elif reduceAngleToLower:
                var = <C_FLOAT32>fabs(lowerLimit2-angle)
            else:
                if angle > (lowerLimit2+upperLimit2)/FLOAT_TWO:
                    var = <C_FLOAT32>fabs(upperLimit2-angle)
                else:
                    var = <C_FLOAT32>fabs(lowerLimit2-angle)
            if var<reducedAngle:
                reducedAngle = var
    # compute reduced angle trying third and last shell
    if reducedAngle!=FLOAT_ZERO:
        if (upperLimit3<lowerLimit3) and angle>upperLimit3:
            reducedAngle = FLOAT_ZERO
        elif angle>=lowerLimit3 and angle<=upperLimit3:
            reducedAngle = FLOAT_ZERO
        else:
            if reduceAngleToUpper:
                var = <C_FLOAT32>fabs(upperLimit3-angle)
            elif reduceAngleToLower:
                var = <C_FLOAT32>fabs(lowerLimit3-angle)
            else:
                if angle > (lowerLimit3+upperLimit3)/FLOAT_TWO:
                    var = <C_FLOAT32>fabs(upperLimit3-angle)
                else:
                    var = <C_FLOAT32>fabs(lowerLimit3-angle)
            if var<reducedAngle:
                reducedAngle = var
    # set angles and reduced
    angles[index]        = angle
    reducedAngles[index] = reducedAngle


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def full_dihedral_angles_coords( ndarray[C_INT32, ndim=1]   indexes1 not None, 
                                 ndarray[C_INT32, ndim=1]   indexes2 not None,
                                 ndarray[C_INT32, ndim=1]   indexes3 not None,
                                 ndarray[C_INT32, ndim=1]   indexes4 not None,
                                 ndarray[C_FLOAT32, ndim=1] lowerLimit1 not None,
                                 ndarray[C_FLOAT32, ndim=1] upperLimit1 not None,
                                 ndarray[C_FLOAT32, ndim=1] lowerLimit2 not None,
                                 ndarray[C_FLOAT32, ndim=1] upperLimit2 not None,
                                 ndarray[C_FLOAT32, ndim=1] lowerLimit3 not None,
                                 ndarray[C_FLOAT32, ndim=1] upperLimit3 not None,
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
       #. indexes1 (int32 (n,) numpy.ndarray): Diherdral first atom indexes.
       #. indexes2 (int32 (n,) numpy.ndarray): Diherdral second atom indexes.
       #. indexes3 (int32 (n,) numpy.ndarray): Diherdral third atom indexes.
       #. indexes4 (int32 (n,) numpy.ndarray): Diherdral fourth atom indexes.
       #. lowerLimit1 (float32 (n,) numpy.ndarray): First shells lower limit.
       #. upperLimit1 (float32 (n,) numpy.ndarray): First shells upper limits.
       #. lowerLimit2 (float32 (n,) numpy.ndarray): Second shells lower limit.
       #. upperLimit2 (float32 (n,) numpy.ndarray): Second shells upper limits.
       #. lowerLimit3 (float32 (n,) numpy.ndarray): Third shells lower limit.
       #. upperLimit3 (float32 (n,) numpy.ndarray): Third shells upper limits.
       #. boxCoords (float32 (n,3) numpy.ndarray): The atomic coordinates array of the same shape as pointsFrom.
       #. basis (float32 (3,3) numpy.ndarray): The (3x3) boundary conditions box vectors.
       #. isPBC (bool): Whether it is a periodic boundary conditions or infinite.
       #. reduceAngleToUpper (bool): Whether to reduce angle found out of limits to the difference between the angle and the upper limit. When True, this flag has the higher priority. DEFAULT: False
       #. reduceAngleToLower (bool): Whether to reduce angle found out of limits to the difference between the angle and the lower limit. When True, this flag may lose its priority for reduceAngleToUpper if the later is True. DEFAULT: False
       #. ncores (int32) [default=1]: The number of cores to use.
       
    :Returns:
       #. angles: The calculated angles (rad).
       #. reducedAngles: The reduced angles (rad)
    """
    
    cdef C_INT32 i, numberOfIndexes
    cdef C_FLOAT32 angle, reducedAngle
    # get number of indexes
    numberOfIndexes = <C_INT32>indexes1.shape[0]
    # create abgles and reduced list
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] angles  = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] reduced = np.zeros((numberOfIndexes), dtype=NUMPY_FLOAT32) 

    # loop all angles
    for i from 0 <= i < numberOfIndexes:   
        b1 = pair_difference_to_point( point1 = boxCoords[indexes1[i],:], 
                                       point2 = boxCoords[indexes2[i],:], 
                                       basis  = basis,
                                       isPBC  = isPBC,
                                       ncores = INT_ONE) 
        b2 = pair_difference_to_point( point1 = boxCoords[indexes2[i],:], 
                                       point2 = boxCoords[indexes3[i],:], 
                                       basis  = basis,
                                       isPBC  = isPBC,
                                       ncores = INT_ONE) 
        b3 = pair_difference_to_point( point1 = boxCoords[indexes3[i],:], 
                                       point2 = boxCoords[indexes4[i],:], 
                                       basis  = basis,
                                       isPBC  = isPBC,
                                       ncores = INT_ONE)                                                                                                                      
        _single_dihedral_angle( b1                 = b1 ,
                                b2                 = b2 ,
                                b3                 = b3 ,
                                lowerLimit1        = lowerLimit1[i] ,
                                upperLimit1        = upperLimit1[i] ,
                                lowerLimit2        = lowerLimit2[i] ,
                                upperLimit2        = upperLimit2[i] ,
                                lowerLimit3        = lowerLimit3[i] ,
                                upperLimit3        = upperLimit3[i] ,
                                angles             = angles,
                                reducedAngles      = reduced,
                                index              = i ,
                                reduceAngleToUpper = reduceAngleToUpper,
                                reduceAngleToLower = reduceAngleToLower)  
    # return results
    return angles, reduced      

        



    
    
    
    
    
    
    
    
    
    
    
    

   
    