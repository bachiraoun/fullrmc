# standard libraries imports
import os

# import matplotlib
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "system.rmc")

# load and plot
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    GR     = ENGINE.constraints[0] 
    GR.plot(intra=False, inter=True)
else:
    print mes
 
 
