# standard libraries imports
import os

# external libraries imports

# fullrmc library imports
from fullrmc.Engine import Engine

# engine variables
engineSavePath = "thf_engine.rmc"
    
# check Engine already saved
if engineSavePath not in os.listdir("."):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)
    # visualize    
    ENGINE.visualize( boxToCenter=True)     
    
    
    
