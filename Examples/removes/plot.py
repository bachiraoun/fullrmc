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
    PDF, EMD = ENGINE.constraints
    EMD.plot(show=False)
    PDF.plot(intra=False, show=True)
else:
    print mes
 
 
