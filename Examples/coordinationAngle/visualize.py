# standard libraries imports
import os

# external libraries imports

# pdbParser imports

# fullrmc library imports
from fullrmc.Engine import Engine


# engine variables
engineSavePath = "engine.rmc"
    
# check Engine already saved
if engineSavePath not in os.listdir("."):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)
    ENGINE.visualize()     
    
    
    
