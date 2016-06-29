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


def plot(output, figName, imgpath, zoom=True, show=True, save=True):
    # create figure
    FIG = plt.figure(figsize=(20,11.25))
    FIG.patch.set_facecolor('white')
    grid = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.9])
    grid.update(left=0.0, right=0.95, top=0.85, wspace=0.1)
    
    imagAx = plt.subplot(grid[:, 0])
    plotAx = plt.subplot(grid[0, 1])
    # add zoomAx
    if zoom:
        zoomAx = plt.axes([.725, .45, .2, .2])
        zoomAx.set_xlim([12,20])
        zoomAx.set_ylim([-0.5,0.65])
        plt.setp(zoomAx, xticks=range(12,21,2), yticks=np.arange(-0.4,0.61, 0.2))
        
    # plot image
    image = mpimg.imread(imgpath)
    imagAx.imshow(image)
    
    # set axis ticks
    imagAx.set_axis_off()
    
    # Create plotting styles
    colors = ["b",'g','r','c','m','y']
    markers = ["",'.','+','^','|']
    STYLE = [r[0] + r[1]for r in itertools.product(['-'], colors)]
    STYLE = [r[0] + r[1]for r in itertools.product(markers, STYLE)]

    # plot
    experimentalPDF = output["observed"]
    experimentalDistances = output["observedR"]
    shellsCenter = output["computedR"] 
    # plot partials
    styleIndex = 0
    for key, val in output.items():
        if key in ("pdf_total", "pdf"):
            continue
        elif "inter" in key:
            plotAx.plot(shellsCenter, val, STYLE[styleIndex], markevery=5, linewidth=3.0, label=key.split('rdf_inter_')[1] )
            if zoom:
                zoomAx.plot(shellsCenter, val, STYLE[styleIndex], markevery=5, linewidth=3.0, label=key.split('rdf_inter_')[1] )
            styleIndex+=1
    # plot experimental and total
    plotAx.plot(experimentalDistances, experimentalPDF, 'ro', label="observed", markersize=12, markevery=1 )
    plotAx.plot(shellsCenter, output["pdf"], 'k', linewidth=4.0,  markevery=25, label="total" )
    if zoom:
        zoomAx.plot(experimentalDistances, experimentalPDF, 'ro', label="observed", markersize=12, markevery=1 )
        zoomAx.plot(shellsCenter, output["pdf"], 'k', linewidth=4.0,  markevery=25, label="total" )
    # set legend        
    plotAx.legend(ncol=2, frameon=False, fontsize=30)
    
    # set chisquare
    imagAx.text(.5,0.95,"$\chi^2=%.6f$"%output['chiSquare'], fontsize=45,
                horizontalalignment='center',
                transform=imagAx.transAxes)
    
    # set labels
    plotAx.set_xlabel("$r (\AA)$", size=40)
    plotAx.set_ylabel("$g(r)$", size=40, labelpad=-10)
    #[tick.label.set_fontsize(20) for tick in plotAx.xaxis.get_major_ticks()]
    #[tick.label.set_fontsize(20) for tick in plotAx.yaxis.get_major_ticks()]
    plotAx.tick_params(axis='both', which='major', width=3, size=20, labelsize=30)
    [i.set_linewidth(3) for i in plotAx.spines.itervalues()]
    
    # show plot 
    if save: FIG.savefig(figName)
    if show: plt.show() 
    plt.close(FIG)
    
    
# set logger file flag
LOGGER.set_log_to_file_flag(False)
LOGGER.set_log_to_stdout_flag(False)

# engine variables
expDataPath = "experimental.gr"
pdfDataPath = "pdf.data"

# pdbFiles
pdbFiles  = [fn for fn in os.listdir("pdbFiles") if ".pdb" in fn]
generated = [int(item.split('.pdb')[0]) for item in os.listdir("pdbFiles")]
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
    print "loading frame %i out of %i --> %s"%(idx, len(pdbFiles), fname)
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
    output["chiSquare"] =  PDF.squaredDeviations
    data[fname] = output
    
if dataAdded: pickle.dump( data, open( pdfDataPath, "wb" ) )


# plot data
idx = 0
for key in pdbFiles:
    #key = pdbFiles[-1]
    print key
    path, figName = os.path.split(key)
    figName = os.path.join("pdfFigures",figName.split(".pdb")[0]+".png")
    imgpath = str(idx).rjust(5,"0")
    imgpath = os.path.join("snapshots","."+imgpath+".bmp")
    plot(data[key], figName=figName, imgpath=imgpath, zoom=idx>=209, show=False, save=True)
    idx += 1
    #exit()
