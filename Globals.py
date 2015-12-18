"""
This module provides all the global types, variables and some general methods that fullrmc needs.
"""
# standard libraries imports
import sys
import os

# external libraries imports
import numpy as np
<<<<<<< HEAD
from pysimplelog import Logger as LOG
=======
from simplelogger import SimpleLogger
>>>>>>> 1218b7511b5ec4b0f951880d15321eb096f6e5a2

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


# Create LOGGER
class Logger(LOG):
    def __new__(cls, *args, **kwds):
        #Singleton interface for logger
        thisSingleton = cls.__dict__.get("__thisSingleton__")
        if thisSingleton is not None:
            return thisSingleton
        cls.__thisSingleton__ = thisSingleton = LOG.__new__(cls)
        return thisSingleton
        
    def __init__(self, *args, **kwargs):
        super(Logger, self).__init__(*args, **kwargs)
        # set logfile basename
        logFile = os.path.join(os.getcwd(), "fullrmc")
        self.set_log_file_basename(logFile)
        # set new log types
        self.add_log_type("argument fixed", name="FIXED", stdoutFlag=True,  fileFlag=True)
        self.add_log_type("move accepted",  name="INFO",  stdoutFlag=True,  fileFlag=True)
        self.add_log_type("move rejected",  name="INFO",  stdoutFlag=False, fileFlag=False)
        self.add_log_type("move not tried", name="INFO",  stdoutFlag=False, fileFlag=False)
        self.add_log_type("save engine",    name="INFO",  stdoutFlag=True,  fileFlag=True, level=sys.maxint)
        # set parameters
        self.__set_logger_params_from_file()
        
    def __set_logger_params_from_file(self):
        pass
        
LOGGER = Logger(name="fullrmc")  









