# standard libraries imports
import os

# fullrmc library imports
from fullrmc import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "SiOx.rmc")

# load
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    ENGINE.constraints[0].plot(inter=False, intra=False, shapeFunc=True, legendLoc='upper left')
else:
    print mes
 
