# standard libraries imports
import os

# fullrmc library imports
from fullrmc import Engine

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
engineFilePath = os.path.join(DIR_PATH, "thf_engine.rmc")

# load
ENGINE = Engine(path=None)
result, mes = ENGINE.is_engine(engineFilePath, mes=True)
if result:
    ENGINE = ENGINE.load(engineFilePath)
    PDF, EMD, B, BA, IA = ENGINE.constraints
    # plot all constraints
    PDF.plot(show=False)
    EMD.plot(show=False)
    B.plot(lineWidth=2, nbins=20,  split='element', show=False)
    BA.plot(lineWidth=2, nbins=20, split='element', show=False)
    IA.plot(lineWidth=2, nbins=20, split='element', show=True )
else:
    print mes
 