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

# create figure
FIG = plt.figure()
FIG.patch.set_facecolor('white')
grid = gridspec.GridSpec(nrows=2, ncols=2)
grid.update(left=0.1, bottom=0.1, right=0.95, wspace=0.05)

totalAx = plt.subplot(grid[0, :])
lowRAx  = plt.subplot(grid[1, 0])
highRAx = plt.subplot(grid[1, 1])


# set axis ticks
highRAx.get_yaxis().set_ticks([])
lowRAx.get_yaxis().set_ticks([])

# Create plotting styles
colors = ["b",'g','r','c','m','y']
markers = ["",'.','+','^','|']
INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], colors)]
INTRA_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTRA_STYLES)]
INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], colors)]
INTER_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTER_STYLES)]

# engine variables
engineSavePath = "thf_engine.rmc"
    
# check Engine already saved
if engineSavePath not in os.listdir("."):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)
    PDF = ENGINE.constraints[0] 
    #ENGINE.set_pdb("pdbFiles\\518300_atoms.pdb"); PDF.compute_data()
    #ENGINE.set_pdb("pdbFiles\\0_original.pdb"); PDF.compute_data()
    #PDF.set_scale_factor(1.05); PDF.compute_data()
    #PDF.set_limits((None, None)); PDF.compute_data()


# plot
output = PDF.get_constraint_value()
totalAx.plot(PDF.experimentalDistances, ENGINE.constraints[0].experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
lowRAx.plot(PDF.experimentalDistances, ENGINE.constraints[0].experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
highRAx.plot(PDF.experimentalDistances, ENGINE.constraints[0].experimentalPDF, 'ro', label="observed", markersize=7.5, markevery=1 )
totalAx.plot(PDF.shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
lowRAx.plot(PDF.shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
highRAx.plot(PDF.shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
 
#intraStyleIndex = 0
#interStyleIndex = 0
#for key, val in output.items():
#    if key in ("pdf_total", "pdf"):
#        continue
#    elif "intra" in key:
#        totalAx.plot(PDF.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
#        lowRAx.plot(PDF.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
#        highRAx.plot(PDF.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
#        intraStyleIndex+=1
#    elif "inter" in key:
#        totalAx.plot(PDF.shellsCenter, val, INTER_STYLES[interStyleIndex], markevery=5, label=key.split("rdf_")[1] )
#        lowRAx.plot(PDF.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
#        highRAx.plot(PDF.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key.split("rdf_")[1] )
#        interStyleIndex+=1
#        
# set legend        
totalAx.legend(ncol=2, frameon=False, fontsize=20)#, loc=(1.05,-1.25))
        
# remove y ticks labels
lowRAx.set_yticklabels([])
highRAx.set_yticklabels([])

# set x limits
lowRAx.relim()
highRAx.relim()
lowRAx.set_xlim([0.5,2.6])
highRAx.set_xlim([2.7,16])

# set y limits
lowRAx.set_ylim(top=8)
highRAx.set_ylim(top=1.5)

# add labels
plt.gcf().text(0.05,0.5, 'g(r)',
     horizontalalignment='center',
     verticalalignment='center',
     rotation=90, fontsize=20)
plt.gcf().text(0.55,0.05, 'r($\AA$)',
     horizontalalignment='center',
     verticalalignment='center', fontsize=20)
#plt.gcf().text(0.975,0.025, 'c)',
#     horizontalalignment='center',
#     verticalalignment='center', fontsize=20)
# show plot
#plt.suptitle("$\chi^2=%.6f$"%ENGINE.chiSquare, fontsize=20)

plt.show()  
 

    
    
