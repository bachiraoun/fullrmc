# standard libraries imports
import os


# fullrmc library imports
from fullrmc import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "system.rmc")

# load and visualize
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    ENGINE.visualize(foldIntoBox=True, representationParams='VDW 0.2 40')  
else:
    print mes
 
 

