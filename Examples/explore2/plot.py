# standard libraries imports
import os
import itertools

# external libraries imports
import wx
import numpy as np
import matplotlib.pyplot as plt

# pdbParser imports

# fullrmc library imports
from fullrmc.Engine import Engine


# Create plotting styles
#styles  = ['-','--','-.',':']
colors = ["b",'g','r','c','m','y']
markers = ["",'.','+','^','|']
INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], colors)]
INTRA_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTRA_STYLES)]
INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], colors)]
INTER_STYLES = [r[0] + r[1]for r in itertools.product(markers, INTER_STYLES)]

# engine variables
engineSavePath = "CO2.rmc"
    
# check Engine already saved
if engineSavePath not in os.listdir("."):
    exit()
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)
    PDF = ENGINE.constraints[0] 
#print PDF.scaleFactor; PDF.set_scale_factor(1.); print PDF.scaleFactor
#PDF.set_limits((None,None)); ENGINE.initialize_used_constraints()
#print ENGINE.boundaryConditions.get_vectors(); ENGINE.set_boundary_conditions(55) ;print ENGINE.boundaryConditions.get_vectors(); ENGINE.initialize_used_constraints();PDF = ENGINE.constraints[0] 
#ENGINE.set_pdb("pdbFiles/192624660_atoms.pdb"); ENGINE.constraints[0].set_limits((None,None)); ENGINE.initialize_used_constraints();PDF = ENGINE.constraints[0]


# get output
output = PDF.get_constraint_value()
# plot experimental
plt.plot(PDF.experimentalDistances,PDF.experimentalPDF, 'ro', label="experimental", markersize=7.5, markevery=1 )
# plot partials
intraStyleIndex = 0
interStyleIndex = 0
for key, val in output.items():
    if key in ("pdf_total", "rdf"):
        continue
    elif "intra" in key:
        plt.plot(PDF.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key )
        intraStyleIndex+=1
    elif "inter" in key:
        plt.plot(PDF.shellsCenter, val, INTER_STYLES[interStyleIndex], markevery=5, label=key )
        interStyleIndex+=1
# plot total
plt.plot(PDF.shellsCenter, output["pdf_total"], 'k', linewidth=3.0,  markevery=25, label="total" )
        
plt.legend(frameon=False, ncol=2)
if ENGINE.chiSquare is not None: plt.title("$\chi^2=%.6f$"%ENGINE.chiSquare)

plt.show()    
    
    
    
    
