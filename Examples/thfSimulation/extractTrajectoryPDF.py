# standard libraries imports
import time
import os
import itertools
import cPickle as pickle

# external libraries imports
import wx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint


def plot(output, figName, imgpath, show=True, save=True):
    # create figure
    FIG = plt.figure(figsize=(20,11.25))
    FIG.patch.set_facecolor('white')
    grid = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[2, 1, 1])
    grid.update(left=0.0, right=0.95, wspace=0.05)
    
    imagAx  = plt.subplot(grid[:, 0])
    totalAx = plt.subplot(grid[0, 1:])
    lowRAx  = plt.subplot(grid[1, 1])
    highRAx = plt.subplot(grid[1, 2])
    
    # plot image
    image = mpimg.imread(imgpath)
    imagAx.imshow(image)
    
    # set axis ticks
    imagAx.set_axis_off()
    highRAx.get_yaxis().set_ticks([])
    lowRAx.get_yaxis().set_ticks([])
    
    # Create plotting styles
    colors = ["b",'g','r','c','m','y']
    markers = ["",'.','+','^','|']
    INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], colors)]
    INTRA_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTRA_STYLES)]
    INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], colors)]
    INTER_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTER_STYLES)]

    # plot
    experimentalPDF = output["observed"]
    experimentalDistances = output["observedR"]
    shellsCenter = output["computedR"] 

    totalAx.plot(experimentalDistances, experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
    lowRAx.plot(experimentalDistances, experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
    highRAx.plot(experimentalDistances, experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
    totalAx.plot(shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
    lowRAx.plot(shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
    highRAx.plot(shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
     
    intraStyleIndex = 0
    interStyleIndex = 0
    for key, val in output.items():
        if key in ("pdf_total", "pdf"):
            continue
        elif "intra" in key:
            totalAx.plot(shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("pdf_")[1] )
            lowRAx.plot(shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("pdf_")[1] )
            highRAx.plot(shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("pdf_")[1] )
            intraStyleIndex+=1
        elif "inter" in key:
            totalAx.plot(shellsCenter, val, INTER_STYLES[interStyleIndex], markevery=5, label=key.split("pdf_")[1] )
            lowRAx.plot(shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("pdf_")[1] )
            highRAx.plot(shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("pdf_")[1] )
            interStyleIndex+=1
            
    # set legend        
    totalAx.legend(ncol=2, frameon=False, fontsize=20)
    
    # set chisquare
    imagAx.text(.5,0.95,"$\chi^2=%.6f$"%output['chiSquare'], fontsize=35,
                horizontalalignment='center',
                transform=imagAx.transAxes)
            
    # remove y ticks labels
    lowRAx.set_yticklabels([])
    highRAx.set_yticklabels([])
    
    # set x limits
    lowRAx.relim()
    highRAx.relim()
    lowRAx.set_xlim([0.5,2.6])
    highRAx.set_xlim([3,16])
    
    # set y limits
    lowRAx.set_ylim(top=6.5)
    highRAx.set_ylim(top=1.5)
    
    # show plot
    if show: plt.show()  
    if save: FIG.savefig(figName)
    plt.close(FIG)
    
    
# set logger file flag
LOGGER.set_log_to_file_flag(False)
LOGGER.set_log_to_stdout_flag(False)

# engine variables
expDataPath = "thf_pdf.exp"
pdfDataPath = "pdf.data"

# pdbFiles
pdbFiles  = [fn for fn in os.listdir("pdbFiles") if ".pdb" in fn]
generated = [int(fn.split("_")[0]) for fn in pdbFiles]
pdbFiles = [os.path.join("pdbFiles",pdbFiles[idx]) for idx in  np.argsort(generated)]

if os.path.isfile(pdfDataPath): 
    data = pickle.load( open( pdfDataPath, "rb" ) )
else:
    data = {}

# compute pdfs
dataAdded = False
for idx in range(len(pdbFiles)):
    fname = pdbFiles[idx]
    if data.has_key(fname): continue
    dataAdded = True
    print "loading frame %i out of %i"%(idx, len(pdbFiles))
    # create constraints
    PDF = PairDistributionConstraint(engine=None, experimentalData=expDataPath, weighting="atomicNumber")
    # create engine
    ENGINE = Engine(pdb=fname, constraints=[PDF])
    ENGINE.run(numberOfSteps=0)
    # get pdf
    output = PDF.get_constraint_value()
    output["observed"]  =  PDF.experimentalPDF
    output["observedR"] =  PDF.experimentalDistances
    output["computedR"] =  PDF.shellsCenter
    output["chiSquare"] =  ENGINE.chiSquare
    data[fname] = output
    
if dataAdded: pickle.dump( data, open( pdfDataPath, "wb" ) )


# plot data
idx = 0
for key in pdbFiles:
    path, figName = os.path.split(key)
    figName = os.path.join("pdfFigures",figName.split(".pdb")[0]+".png")
    imgpath = str(idx).rjust(5,"0")
    imgpath = os.path.join("pdfFigures","snapshot."+imgpath+".png")
    plot(data[key], figName=figName, imgpath=imgpath, show=False, save=True)
    idx += 1
