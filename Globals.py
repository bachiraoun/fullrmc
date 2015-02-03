"""
This module provides all the global types, variables and some general methods that fullrmc needs.
"""
# standard libraries imports

# external libraries imports
import numpy as np

# data types definitions
INT_TYPE   = np.int32   # must be the integer type for the whole package
FLOAT_TYPE = np.float32 # must be the float type for the whole package

# floating precision
if FLOAT_TYPE is np.float32:                               
    PRECISION = FLOAT_TYPE(1e-5)  # Used to check precision of float32 loaded data 
elif FLOAT_TYPE is np.float64:
    PRECISION = FLOAT_TYPE(1e-10) # Used to check precision of float64 loaded data 
else:
    raise Exception("Unknown float type '%s'"%FLOAT_TYPE)
    
# Constants definitions
FLOAT_PLUS_INFINITY  = FLOAT_TYPE(np.finfo(FLOAT_TYPE).max) # +inf number for float type
FLOAT_MINUS_INFINITY = FLOAT_TYPE(np.finfo(FLOAT_TYPE).min) # -inf number for float type
INT_PLUS_INFINITY    = INT_TYPE(np.iinfo(np.int32).max)     # +inf number for integer type 
INT_MINUS_INFINITY   = INT_TYPE(np.iinfo(np.int32).min)     # -inf number for integer type 
PI                   = FLOAT_TYPE(np.pi)                    # pi the ratio of a circle's circumference to its diameter, set as constant for typing 

# Functions and methods
from random import random  as generate_random_float   # generates a random float number between 0 and 1
from random import randint as generate_random_integer # generates a random integer number between given lower and upper limits

        
        
        