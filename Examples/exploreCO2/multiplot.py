# standard libraries imports
import os
import itertools

# external libraries imports
import wx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# fullrmc library imports
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairCorrelationConstraints import PairDistributionConstraint


# Create plotting styles
#styles  = ['-','--','-.',':']
colors = ["b",'g','r','c','m','y']
markers = ["",'.','+','^','|']
INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], colors)]
INTRA_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTRA_STYLES)]
INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], colors)]
INTER_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTER_STYLES)]

enginePath = "CO2.rmc"

# create engine
ENGINE = Engine(pdb=None).load(enginePath)
PDF_CONSTRAINT = ENGINE.constraints[0]


def create_figure(PDF):
    # get output
    output = PDF.get_constraint_value()
    # create figure
    FIG = plt.figure()
    FIG.patch.set_facecolor('white')
    grid = gridspec.GridSpec(nrows=2, ncols=2)
    grid.update(left=0.05, right=0.95, wspace=0.05)
    totalAx = plt.subplot(grid[0, :])
    lowRAx  = plt.subplot(grid[1, 0])
    highRAx = plt.subplot(grid[1, 1])
    # set axis ticks
    highRAx.get_yaxis().set_ticks([])
    lowRAx.get_yaxis().set_ticks([])
    # plot experimental
    totalAx.plot(PDF.experimentalDistances, ENGINE.constraints[0].experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
    lowRAx.plot(PDF.experimentalDistances, ENGINE.constraints[0].experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
    highRAx.plot(PDF.experimentalDistances, ENGINE.constraints[0].experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
    # plot partials
    intraStyleIndex = 0
    interStyleIndex = 0
    for key, val in output.items():
        if key in ("pdf_total", "pdf"):
            continue
        elif "intra" in key:
            totalAx.plot(PDF.shellCenters, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
            lowRAx.plot(PDF.shellCenters, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
            highRAx.plot(PDF.shellCenters, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
            intraStyleIndex+=1
        elif "inter" in key:
            totalAx.plot(PDF.shellCenters, val, INTER_STYLES[interStyleIndex], markevery=5, label=key.split("rdf_")[1] )
            lowRAx.plot(PDF.shellCenters, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
            highRAx.plot(PDF.shellCenters, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
            interStyleIndex+=1
    # plot totals
    totalAx.plot(PDF.shellCenters, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
    lowRAx.plot(PDF.shellCenters, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
    highRAx.plot(PDF.shellCenters, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )  
    # set legend        
    totalAx.legend(ncol=2, frameon=False, fontsize=12)#, loc=(1.05,-1.25))
    # remove y ticks labels
    lowRAx.set_yticklabels([])
    highRAx.set_yticklabels([])
    # set x limits
    lowRAx.relim()
    highRAx.relim()
    lowRAx.set_xlim([0,1.4])
    highRAx.set_xlim([1.4,5])
    totalAx.set_xlim([0,10])
    # set y limits
    lowRAx.set_ylim(top=4.9)
    highRAx.set_ylim(top=1.5)
    # set title
    if ENGINE.totalStandardError is not None:  plt.suptitle("$StdErr=%.6f$"%ENGINE.totalStandardError, fontsize=20)
    return FIG, totalAx, lowRAx, highRAx

def add_low_r_ax_annotations(ax):
    # annotate lowRAx
    ax.annotate('', xy=(0.67, 1.1), xytext=(1.35, -0.45), ha="right",
                      arrowprops=dict(facecolor='black', 
                                      arrowstyle="simple", 
                                      fc="w", ec="k",
                                      connectionstyle="arc3,rad=1.7") )
    ax.text(0.45, 3.6,'explore allows tunneling\n and going through energy\n barriers from a peak \nto another.',
                horizontalalignment='center',
                verticalalignment='center')

def add_total_ax_annotations(ax):                
    ax.annotate('', xy=(0.725, 1.1), xytext=(3.5, 2), ha="right",
                         arrowprops=dict(facecolor='black', 
                                         arrowstyle="-|>", 
                                         fc="w", ec="k",
                                         connectionstyle="arc3,rad=-0.2") )
    
    ax.annotate('', xy=(1.1, 4), xytext=(3.5, 3.5), ha="right",
                         arrowprops=dict(facecolor='black', 
                                         arrowstyle="-|>", 
                                         fc="w", ec="k",
                                         connectionstyle="arc3,rad=0.2") )
                                         
    ax.text(3.5, 2.75,'Bonding electron density\n polarizability measured by X-rays',
                 horizontalalignment='center',
                 verticalalignment='center')
                 
# get output
FIG, totalAx, lowRAx, highRAx = create_figure(PDF_CONSTRAINT)  
add_low_r_ax_annotations(lowRAx)
add_total_ax_annotations(totalAx)

                
# show
plt.show()  
            
        

    
