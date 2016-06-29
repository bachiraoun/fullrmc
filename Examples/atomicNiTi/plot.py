# standard libraries imports
import os
import itertools

# external libraries imports
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Engine import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )

# engine variables
engineFileName = "engine.rmc"
engineFilePath = os.path.join(DIR_PATH, engineFileName)

# check Engine already saved
if engineFileName not in os.listdir(DIR_PATH):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineFilePath)
    GR     = ENGINE.constraints[0] 
    SQ     = ENGINE.constraints[1] 
    GR.plot(plt.figure().gca(), intra=False)
    SQ.plot(plt.figure().gca(), intra=False)
    plt.show()
    

