# standard libraries imports
import os
import itertools

# external libraries imports
import wx
import numpy as np
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Engine import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )

# engine variables
engineFileName = "thf_engine.rmc"
engineFilePath = os.path.join(DIR_PATH, engineFileName)

# check Engine already saved
if engineFileName not in os.listdir(DIR_PATH):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineFilePath)
    PDF = ENGINE.constraints[0] 
    #ENGINE.set_boundary_conditions(48)
    #PDF.set_limits((None, None)); PDF.compute_data()
    #PDF.set_scale_factor(0.8); PDF.compute_data()
    PDF.plot()
    
    
 