# standard libraries imports
import os

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
    PDF.plot()
    
    
 