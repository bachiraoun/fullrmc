# standard libraries imports
import os

# external libraries imports

# fullrmc library imports
from fullrmc.Engine import Engine

# engine variables
enginePath = "engine.rmc"
    
# check Engine already saved
if enginePath not in os.listdir("."):
    exit()
else:
    ENGINE = Engine(pdb=None).load(enginePath)

# visualize    
ENGINE.visualize(foldIntoBox=True, representationParams='VDW 0.1 20')     
    
    
    
