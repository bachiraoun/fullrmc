# standard libraries imports
import os
import itertools

# external libraries imports
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairCorrelationConstraints import PairDistributionConstraint


ENGINE =  Engine(pdb='system.pdb', constraints=None)
PDF_CONSTRAINT = PairDistributionConstraint(engine=None, experimentalData="experimental.gr", weighting="atomicNumber")
ENGINE.add_constraints([PDF_CONSTRAINT]) 

# Create plotting styles
colors = ["b",'g','r','c','m','y']
markers = ["",'.','+','^','|']
STYLE = [r[0] + r[1]for r in itertools.product(['-'], colors)]
STYLE = [r[0] + r[1]for r in itertools.product(markers, INTER_STYLES)]

def plot(PDF, figName, imgpath, show=False, save=True):
    # plot
    output = PDF.get_constraint_value()
    plt.plot(PDF.experimentalDistances,PDF.experimentalPDF, 'ro', label="experimental", markersize=7.5, markevery=1 )
    plt.plot(PDF.shellsCenter, output["pdf"], 'k', linewidth=3.0,  markevery=25, label="total" )
    
    styleIndex = 0
    for key, val in output.items():
        if key in ("pdf_total", "pdf"):
            continue
        elif "inter" in key:
            plt.plot(PDF.shellsCenter, val, STYLE[styleIndex], markevery=5, label=key.split('rdf_inter_')[1] )
            styleIndex+=1
    plt.legend(frameon=False, ncol=1)
    # set labels
    plt.title("$\\chi^{2}=%.6f$"%PDF.squaredDeviations, size=20)
    plt.xlabel("$r (\AA)$", size=20)
    plt.ylabel("$g(r)$", size=20)
    # show plot
    if save: plt.savefig(figName)
    if show: plt.show()  
    plt.close()
    
    
for num in sorted( [int(item.split('.pdb')[0]) for item in os.listdir("pdbFiles")] ):
   print str(num)+".pdb"
   pdbPath = os.path.join("pdbFiles", str(num)+".pdb")
   figName = os.path.join("pdfFigures",str(num)+".png")
   imgpath = "pdfFigures"
   ENGINE.set_pdb(pdbPath)
   PDF_CONSTRAINT.compute_data()
   plot(PDF_CONSTRAINT, figName, imgpath, show=False, save=True)
## plot data
#idx = 0
#for key in pdbFiles:
#    path, figName = os.path.split(key)
#    figName = os.path.join("pdfFigures",figName.split(".pdb")[0]+".png")
#    imgpath = str(idx).rjust(5,"0")
#    imgpath = os.path.join("pdfFigures","snapshot."+imgpath+".png")
#    plot(data[key], figName=figName, imgpath=imgpath, show=False, save=True)
#    idx += 1
