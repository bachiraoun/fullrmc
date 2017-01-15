##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER, FLOAT_TYPE
from fullrmc.Engine import Engine
from fullrmc.Core.Collection import rebin
from fullrmc.Constraints.AtomicCoordinationConstraints import AtomicCoordinationNumberConstraint
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.StructureFactorConstraints import ReducedStructureFactorConstraint
from fullrmc.Generators.Swaps import SwapPositionsGenerator


##########################################################################################
#####################################  CREATE ENGINE  ####################################
# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
# files name
engineFileName = "engine.rmc"
grFileName     = "experimental.gr"
sqFileName     = "experimental.fq"
pdbFileName    = "system.pdb" 
# engine variables
grExpPath      = os.path.join(DIR_PATH, grFileName)
sqExpPath      = os.path.join(DIR_PATH, sqFileName)
pdbPath        = os.path.join(DIR_PATH, pdbFileName)
engineFilePath = os.path.join(DIR_PATH, engineFileName)
# set some useful flags
EXPORT_PDB  = False    
FRESH_START = True

# check Engine exists, if not build it otherwise load it.
ENGINE = Engine(path=None)
if not ENGINE.is_engine(engineFilePath) or FRESH_START:
   # create engine
    ENGINE = Engine(path=engineFilePath, freshStart=True)
    ENGINE.set_pdb(pdbFileName)
    # add G(r) constraint
    PDF_CONSTRAINT = PairDistributionConstraint(experimentalData=grExpPath, weighting="atomicNumber")
    ENGINE.add_constraints([PDF_CONSTRAINT]) 
    # Rebin S(Q) experimental data and build constraint
    Sq = np.transpose( rebin(np.loadtxt(sqExpPath) , bin=0.05) ).astype(FLOAT_TYPE)
    RSF_CONSTRAINT = ReducedStructureFactorConstraint(experimentalData=Sq, weighting="atomicNumber")
    ENGINE.add_constraints([RSF_CONSTRAINT])
    # add coordination number constraint and set to un-used
    ACN_CONSTRAINT = AtomicCoordinationNumberConstraint()
    ENGINE.add_constraints([ACN_CONSTRAINT]) 
    ACN_CONSTRAINT.set_used(False)
    # add inter-molecular distance constraint
    EMD_CONSTRAINT = InterMolecularDistanceConstraint(defaultDistance=2.2, flexible=True)
    ENGINE.add_constraints([EMD_CONSTRAINT]) 
    # save engine
    ENGINE.save()
else:
    ENGINE = ENGINE.load(engineFilePath)
    # unpack constraints before fitting in case tweaking is needed
    PDF_CONSTRAINT, RSF_CONSTRAINT, ACN_CONSTRAINT, EMD_CONSTRAINT = ENGINE.constraints

##########################################################################################
#####################################  DIFFERENT RUNS  ################################### 
def run_normal(nsteps, saveFrequency, engineFilePath, exportPdb=True):
    ENGINE.set_groups_as_atoms()
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=saveFrequency)
    if exportPdb:
        ENGINE.export_pdb( os.path.join(DIR_PATH, "pdbFiles","%i.pdb"%(ENGINE.generated)) )

def run_swap(nsteps, saveFrequency, engineFilePath, exportPdb=True):
    # activate coordination number. Play with definition and set_used to True
    ACN_CONSTRAINT.set_coordination_number_definition( [ ('ti','ti',2.5, 3.5, 4, 8), 
                                                         ('ti','ni',2.2, 3.1, 6, 10),
                                                         ('ni','ni',2.5, 3.5, 4, 8), 
                                                         ('ni','ti',2.2, 3.1, 6, 10) ] )
    ACN_CONSTRAINT.set_used(False)
    # reset groups
    ENGINE.set_groups_as_atoms()
    #### set swap generators Ni-->Ti and Ti-->Ni ###
    allElements = ENGINE.allElements
    niSwapList = [[idx] for idx in range(len(allElements)) if allElements[idx]=='ni']
    tiSwapList = [[idx] for idx in range(len(allElements)) if allElements[idx]=='ti']
    # create swap generator
    toNiSG = SwapPositionsGenerator(swapList=niSwapList)
    toTiSG = SwapPositionsGenerator(swapList=tiSwapList)
    # set swap generator to groups
    for g in ENGINE.groups:
        if allElements[g.indexes[0]]=='ni':
            g.set_move_generator(toTiSG)
        elif allElements[g.indexes[0]]=='ti':
            g.set_move_generator(toNiSG)
    # run
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=saveFrequency)
    if exportPdb:
        ENGINE.export_pdb( os.path.join(DIR_PATH, "pdbFiles","%i.pdb"%(ENGINE.generated)) )

##########################################################################################
#####################################  RUN SIMULATION  ###################################
# export first pdb
if EXPORT_PDB:
    ENGINE.export_pdb( os.path.join("pdbFiles","%i.pdb"%(ENGINE.generated)) )
    
# run normal 10 times for 10000 step each time
for _ in range(10):
    run_normal(nsteps=10000, saveFrequency=10000, engineFilePath=engineFilePath, exportPdb=EXPORT_PDB)
    
# start fitting scale factors each 10 accepted moves
PDF_CONSTRAINT.set_adjust_scale_factor((10, 0.8, 1.2)) 
RSF_CONSTRAINT.set_adjust_scale_factor((10, 0.8, 1.2))

# run normal 100 times for 9000 step each time 
for _ in range(50):
    run_normal(nsteps=10000, saveFrequency=10000, engineFilePath=engineFilePath, exportPdb=EXPORT_PDB)
    
# run swaping 100 times for 1000 step each time
ACN_CONSTRAINT.set_used(False)  
for _ in range(100):
    run_swap(nsteps=1000, saveFrequency=1000, engineFilePath=engineFilePath, exportPdb=EXPORT_PDB)


# run normal 100 times for 9000 step each time 
for _ in range(10):
    run_normal(nsteps=10000, saveFrequency=10000, engineFilePath=engineFilePath, exportPdb=EXPORT_PDB)

    
##########################################################################################
##################################  PLOT PDF CONSTRAINT  #################################
import matplotlib.pyplot as plt
PDF_CONSTRAINT.plot(plt.figure().gca(), intra=False)
RSF_CONSTRAINT.plot(plt.figure().gca(), intra=False)
plt.show()
