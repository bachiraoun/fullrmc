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
    ENGINE.set_used_frame('size_distribution/7')
    ENGINE.visualize(boxWidth=0,  representationParams="VDW 0.4 100")
else:
    print mes
