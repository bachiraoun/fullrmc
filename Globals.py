"""
This module provides all the global types, variables and some general methods that fullrmc needs.

Types:
======
#. INT_TYPE: The integer data type adopted by fullrmc.
#. FLOAT_TYPE: The floating data type adopted by fullrmc.

Variables:
==========
#. PRECISION: The floating data type precision variable.
#. FLOAT_PLUS_INFINITY: The floating data type maximum possible number.
#. FLOAT_MINUS_INFINITY: The floating data type minimum possible number.
#. INT_PLUS_INFINITY: The integer data type maximum possible number.
#. INT_MINUS_INFINITY: The integer data type minimum possible number.
#. PI: pi the ratio of a circle's circumference to its diameter.
"""
# standard libraries imports
from __future__ import print_function
import sys
import os

# external libraries imports
import numpy as np
from pysimplelog import SingleLogger as LOG


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
WATER_NUMBER_DENSITY = FLOAT_TYPE(.0333679)                 # This value is in A^-3

## get python 2 and 3 compatibilities
IS_PY3 = int(sys.version_info.major)==3
if IS_PY3:
    # THIS IS PYTHON 3
    str        = str
    long       = int
    unicode    = str
    bytes      = bytes
    basestring = str
    xrange     = range
    range      = lambda *args: list( xrange(*args) )
    maxint     = sys.maxsize
else:
    # THIS IS PYTHON 2
    assert int(sys.version_info.major)==2, "Only python 2.x.y or python 3.x.y are allowed. %s is used instead"%(sys.version.split()[0],)
    assert int(sys.version_info.minor)>=7, "Only minor version 7 is allowed in python 2.x.y distribution. %s is used instead"%(sys.version.split()[0],)
    str        = str
    unicode    = unicode
    bytes      = str
    long       = long
    basestring = basestring
    xrange     = xrange
    range      = range
    maxint     = sys.maxint


# Create LOGGER
class Logger(LOG):
    def custom_init(self):
        # set logfile basename
        logFile = os.path.join(os.getcwd(), "fullrmc")
        self.set_log_file_basename(logFile)
        # set new log types
        self.add_log_type("move not tried", name="INFO",           level=-15)
        self.add_log_type("move rejected",  name="INFO",           level=-10)
        self.add_log_type("move accepted",  name="INFO",           level= 15)
        self.add_log_type("engine saved",   name="INFO",           level= 17)
        self.add_log_type("argument fixed", name="FIXED",          level= 20)
        self.add_log_type("implement",      name="IMPLEMENTATION", level= 100)
        self.add_log_type("usage",          name="USAGE",          level= 1000)
        self.add_log_type("frame",          name="FRAME",          level= 1000)
        self.add_log_type("report",         name="REPORT ISSUE",   level= 1000)
        # set minimum level to 10
        self.set_minimum_level(10)
        # force error and critical logging no matter what logging level is
        self.force_log_type_flags(logType="error",    stdoutFlag=True, fileFlag=True)
        self.force_log_type_flags(logType="critical", stdoutFlag=True, fileFlag=True)
        #self.force_log_type_flags(logType="implement",stdoutFlag=True, fileFlag=True)
        #self.force_log_type_flags(logType="usage",    stdoutFlag=True, fileFlag=True)

    def fixed(self, message):
        """alias to message at fixed level"""
        self.log("argument fixed", message)

    def accepted(self, message):
        """alias to message at move accepted level"""
        self.log("move accepted", message)

    def rejected(self, message):
        """alias to message at move rejected level"""
        self.log("move rejected", message)

    def nottried(self, message):
        """alias to message at move not tried level"""
        self.log("move not tried", message)

    def saved(self, message):
        """alias to message at save engine level"""
        self.log("engine saved", message)

    def impl(self, message):
        """alias to message at implement engine level"""
        self.log("implement", message)

    def implement(self, message):
        """alias to message at usage engine level"""
        self.log("usage", message)

    def usage(self, message):
        """alias to message at usage engine level"""
        self.log("usage", message)

    def frame(self, message):
        """alias to message at setting engine level"""
        self.log("frame", message)

    def report(self, message):
        """alias to message at report engine level"""
        self.log("report", message)

# initialize Logger
LOGGER = Logger(name="fullrmc")
