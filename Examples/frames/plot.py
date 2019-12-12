# standard libraries imports
from __future__ import print_function
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
    for frame in list(ENGINE.frames):
        ENGINE.set_used_frame(frame)
        GR, SQ, ACN, EMD = ENGINE.constraints
        GR.plot(ax=plt.figure().gca(), intra=False, show=False)
        SQ.plot(ax=plt.figure().gca(), intra=False, show=False)
        if ACN.used:
            ACN.plot(ax=plt.figure().gca(), show=False)
        if ACN.used:
            EMD.plot(ax=plt.figure().gca(), show=False)
    plt.show()
else:
    print(mes)
