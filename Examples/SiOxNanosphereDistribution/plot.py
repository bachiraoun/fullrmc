# standard libraries imports
from __future__ import print_function
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
    PDF, EMD, ACN = ENGINE.constraints

    PDF.plot(frame='size_distribution', multiplot=True, intra=False, shapeFunc=False,
             xlabelParams=True, ylabelParams=True, legendParams=False,
             titleFormat = "@{frame} [${multiframeWeight:.3f}\\%]$ ($Std.Err.={standardError:.3f}$)",
             show=False)

    PDF.plot(frame='size_distribution', multiplot=False, intra=False, shapeFunc=False,
             xlabelParams=True, ylabelParams=True, legendParams={'loc':'lower right', 'fontsize':8, "frameon":False},
             titleFormat = "@{frame} [${multiframeWeight:.3f}\\%]$ ($Std.Err.={standardError:.3f}$)",
             show=False)

    PDF.plot_multiframe_weights('size_distribution')

else:
    print(mes)
