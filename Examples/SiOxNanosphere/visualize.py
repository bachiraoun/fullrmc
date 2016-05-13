## imports
import os
from fullrmc.Engine import Engine

# engine
engineSavePath       = "SiOx.rmc"

if engineSavePath in os.listdir(os.path.dirname( os.path.realpath(__file__) )):
    ENGINE = Engine(pdb=None).load(engineSavePath)
    ENGINE.visualize(boxWidth=0,  representationParams="VDW 0.7 100")