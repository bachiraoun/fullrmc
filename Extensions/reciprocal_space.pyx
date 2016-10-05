"""
This is a C compiled module to compute transformations from 
real to reciprocal space and vice versa.
"""
from libc.math cimport sin
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
cdef C_FLOAT32 FLOAT32_FOUR    = 4.0
cdef C_FLOAT32 FLOAT32_PI      = 3.1415927
cdef C_INT32   INT32_ONE       = 1
cdef C_INT32   INT32_ZERO      = 0


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x)
    C_FLOAT32 ceil(C_FLOAT32 x)

    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)    
def gr_to_sq(np.ndarray[C_FLOAT32, ndim=1] distances not None,
             np.ndarray[C_FLOAT32, ndim=1] gr not None,
             np.ndarray[C_FLOAT32, ndim=1] qrange not None,
             C_FLOAT32                     rho):
    """
    Transform pair correlation function g(r) to static structure factor S(q).
            
    :Arguments:
       #. distances (float32 (n,) numpy.ndarray): The g(r) bins positions in real space.
       #. gr (float32 (n,) numpy.ndarray): The pair correlation function g(r) data.
       #. qrange (float32 (m,) numpy.ndarray): The S(q) bins positions in reciprocal space.
       #. rho (float32) [default=1]: The number density of the system. 
       
    :Returns:
       #. sq (float32 (m,) numpy.ndarray): The static structure factor S(q) data.
    """
    cdef C_INT32 qidx, ridx, qdim, rdim
    cdef C_FLOAT32 dr, dq, q,r, fact
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] sq = np.ones((len(qrange),), dtype=NUMPY_FLOAT32)
    
    dr   = distances[1]-distances[0]
    qdim = <C_INT32>qrange.shape[0]
    rdim = <C_INT32>distances.shape[0]
    fact = FLOAT32_FOUR * FLOAT32_PI * rho
    # s(q) = 1+4*pi*Rho*INTEGRAL[r * g(r) * sin(qr)/q * dr]
    for qidx from INT32_ZERO <= qidx < qdim:
        q = qrange[qidx]
        for ridx from INT32_ZERO <= ridx < rdim:
            r = distances[ridx]
            sq[qidx] += fact * ( dr*r*(np.sin(q*r)/q)*(gr[ridx]-1) )
            #sq[qidx] += dr*r*(np.sin(q*r)/q)*(gr[ridx]-1) 
    return sq#FLOAT32_ONE + FLOAT32_FOUR*FLOAT32_PI*rho*sq
    
    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)    
def Gr_to_sq(np.ndarray[C_FLOAT32, ndim=1] distances not None,
             np.ndarray[C_FLOAT32, ndim=1] Gr not None,
             np.ndarray[C_FLOAT32, ndim=1] qrange not None):
    """
    Transform pair distribution function G(r) to static structure factor S(q).
            
    :Arguments:
       #. distances (float32 (n,) numpy.ndarray): The G(r) bins positions in real space.
       #. Gr (float32 (n,) numpy.ndarray): The pair correlation function Gr) data.
       #. qrange (float32 (m,) numpy.ndarray): The S(q) bins positions in reciprocal space.
       
    :Returns:
       #. sq (float32 (m,) numpy.ndarray): The static structure factor S(q) data.
    """
    cdef C_INT32 qidx, ridx, qdim, rdim
    cdef C_FLOAT32 dr, dq, q,r, fact
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] sq = np.ones((len(qrange),), dtype=NUMPY_FLOAT32)
    
    dr   = distances[1]-distances[0]
    qdim = <C_INT32>qrange.shape[0]
    rdim = <C_INT32>distances.shape[0]
    # s(q) = INTEGRAL[G(r) * sin(qr)/q * dr]
    for qidx from INT32_ZERO <= qidx < qdim:
        q = qrange[qidx]
        for ridx from INT32_ZERO <= ridx < rdim:
            r = distances[ridx]
            sq[qidx] +=  dr*(np.sin(q*r)/q)*(Gr[ridx]) 
    return sq
    
 
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def sq_to_Gr(np.ndarray[C_FLOAT32, ndim=1] qValues not None,
             np.ndarray[C_FLOAT32, ndim=1] rValues not None,
             np.ndarray[C_FLOAT32, ndim=1] sq not None,):
    """
    Transform static structure factor S(q) to pair distribution function G(r).
            
    :Arguments:
       #. qValues (float32 (m,) numpy.ndarray): The S(q) bins positions in reciprocal space.
       #. rValues (float32 (n,) numpy.ndarray): The G(r) bins positions in real space.
       #. sq (float32 (m,) numpy.ndarray): The static structure factor S(q) data.
       
    :Returns:
       #. Gr (float32 (n,) numpy.ndarray): The pair correlation function Gr) data.
       
    """
    cdef C_INT32 ridx, rdim
    cdef C_FLOAT32 dq
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] Gr = np.zeros((len(rValues),), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] sinqr_dq = np.zeros((len(rValues),), dtype=NUMPY_FLOAT32)
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] sq_1  = sq-1
    cdef ndarray[C_FLOAT32,  mode="c", ndim=1] qsq_1 = qValues*sq_1
    
    rdim = <C_INT32>rValues.shape[0]
    dq   = qValues[1]-qValues[0]
    for ridx from INT32_ZERO <= ridx < rdim:
        sinqr_dq = dq * sin(qValues*rValues[ridx])
        Gr[ridx] = (2./np.pi) * np.sum( qsq_1 * sinqr_dq )
    return Gr  
