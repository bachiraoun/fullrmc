## imports
import os
from fullrmc.Engine import Engine
import matplotlib.pyplot as plt

# engine
engineSavePath       = "SiOx.rmc"

if engineSavePath in os.listdir(os.path.dirname( os.path.realpath(__file__) )):
    ENGINE = Engine(pdb=None).load(engineSavePath)
    ENGINE.constraints[0].plot(inter=False, intra=False, shapeFunc=True, legendLoc='upper left')
