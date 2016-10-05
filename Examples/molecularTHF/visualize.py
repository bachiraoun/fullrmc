# standard libraries imports
import os

# fullrmc library imports
from fullrmc.Engine import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "thf_engine.rmc")

# load
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    ENGINE.visualize( boxToCenter=True)  
else:
    print mes
  
    
    
    
