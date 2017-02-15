# standard libraries imports
import os

# fullrmc library imports
from fullrmc import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "engine.rmc")

# load
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    GR, SQ, CN, MD = ENGINE.constraints
    CN.compute_data()
    GR.plot(intra=False,show=False)
    SQ.plot(intra=False,show=False)
    CN.plot(show=False)
    MD.plot(show=True)
else:
    print mes
 
 
