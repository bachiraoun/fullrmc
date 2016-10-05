# standard libraries imports
import os

# import matplotlib
import matplotlib.pyplot as plt

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
    GR     = ENGINE.constraints[0] 
    SQ     = ENGINE.constraints[1] 
    for frame in ['no_constraints', 'with_vdw', 'all_constraints']:
        ENGINE.set_used_frame(frame)
        GR.plot(plt.figure().gca(), intra=False)
        SQ.plot(plt.figure().gca(), intra=False)
    plt.show()
else:
    print mes
 
 
