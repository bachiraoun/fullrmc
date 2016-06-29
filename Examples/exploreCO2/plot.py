## imports
import os
from fullrmc.Engine import Engine

# engine
engineSavePath  = "CO2.rmc"

if engineSavePath in os.listdir(os.path.dirname( os.path.realpath(__file__) )):
    ENGINE = Engine(pdb=None).load(engineSavePath)
    ENGINE.constraints[0].plot()
