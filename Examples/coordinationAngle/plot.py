# standard libraries imports
import os

# external libraries imports
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Engine import Engine


# engine variables
engineSavePath = "engine.rmc"
    
# check Engine already saved
if engineSavePath not in os.listdir("."):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)
    PDF = ENGINE.constraints[0] 
    PDF.plot()
    

    
