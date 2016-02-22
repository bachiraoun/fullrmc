

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
cdef C_FLOAT32 FLOAT32_FOUR    = 4.0
cdef C_FLOAT32 FLOAT32_PI      = 3.1415927
cdef C_INT32   INT32_ONE       = 1
cdef C_INT32   INT32_ZERO      = 0


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x)
    C_FLOAT32 ceil(C_FLOAT32 x)
    C_FLOAT32 sqrt(C_FLOAT32 x)

    
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)    
def gr_to_sq(np.ndarray[C_FLOAT32, ndim=1] distances not None,
             np.ndarray[C_FLOAT32, ndim=1] gr not None,
             np.ndarray[C_FLOAT32, ndim=1] qrange not None,
             C_FLOAT32 rho):
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
    
        