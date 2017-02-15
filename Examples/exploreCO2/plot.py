# standard libraries imports
import os

# fullrmc library imports
from fullrmc import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "CO2.rmc")

# load
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    PDF, IMD, B, BA = ENGINE.constraints
    IMD.plot(show=False)
    B.plot(show=False)
    BA.plot(show=False)
    PDF.plot(show=True)
else:
    print mes