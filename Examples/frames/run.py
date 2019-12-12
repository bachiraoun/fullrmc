##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

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
try:
    DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
except:
    DIR_PATH = ''


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
FRESH_START = False

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
def run_normal(nsteps, saveFrequency, engineFilePath):
    ENGINE.set_groups_as_atoms()
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=saveFrequency, restartPdb=None)

def run_swap(nsteps, saveFrequency, engineFilePath):
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
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=saveFrequency, restartPdb=None)


##########################################################################################
###################################  RUN no_constraints  #################################
# rename first frame set by default to '0'
if 'no_constraints' not in ENGINE.frames:
    ENGINE.rename_frame('0', 'no_constraints')
ENGINE.set_used_frame('no_constraints')
PDF_CONSTRAINT, RSF_CONSTRAINT, ACN_CONSTRAINT, EMD_CONSTRAINT = ENGINE.constraints
ACN_CONSTRAINT.set_used(False)
EMD_CONSTRAINT.set_used(False)
run_normal(nsteps=1000000, saveFrequency=1000, engineFilePath=engineFilePath)
run_swap(nsteps=100000, saveFrequency=1000, engineFilePath=engineFilePath)
run_normal(nsteps=10000, saveFrequency=1000, engineFilePath=engineFilePath)


##########################################################################################
######################################  RUN with_vdw  ####################################
if 'with_vdw' not in ENGINE.frames:
    ENGINE.add_frames('with_vdw')
    ENGINE.set_used_frame('with_vdw')
    ENGINE.reinit_frame('with_vdw')
ENGINE.set_used_frame('with_vdw')
PDF_CONSTRAINT, RSF_CONSTRAINT, ACN_CONSTRAINT, EMD_CONSTRAINT = ENGINE.constraints
ACN_CONSTRAINT.set_used(False)
EMD_CONSTRAINT.set_used(True)
run_normal(nsteps=1000000, saveFrequency=1000, engineFilePath=engineFilePath)
run_swap(nsteps=100000, saveFrequency=1000, engineFilePath=engineFilePath)
run_normal(nsteps=10000, saveFrequency=1000, engineFilePath=engineFilePath)

##########################################################################################
###################################  RUN all_constraints  ################################
if 'all_constraints' not in ENGINE.frames:
    ENGINE.add_frames('all_constraints')
    ENGINE.set_used_frame('all_constraints')
    ENGINE.reinit_frame('all_constraints')
ENGINE.set_used_frame('all_constraints')
PDF_CONSTRAINT, RSF_CONSTRAINT, ACN_CONSTRAINT, EMD_CONSTRAINT = ENGINE.constraints
ACN_CONSTRAINT.set_used(True)
EMD_CONSTRAINT.set_used(True)
ACN_CONSTRAINT.set_coordination_number_definition( [ ('ti','ti',2.5, 3.5, 4, 8),
                                                     ('ti','ni',2.2, 3.1, 6, 10),
                                                     ('ni','ni',2.5, 3.5, 4, 8),
                                                     ('ni','ti',2.2, 3.1, 6, 10) ] )
run_normal(nsteps=1000000, saveFrequency=1000, engineFilePath=engineFilePath)
run_swap(nsteps=100000, saveFrequency=1000, engineFilePath=engineFilePath)
run_normal(nsteps=10000, saveFrequency=1000, engineFilePath=engineFilePath)


##########################################################################################
###################################### CALL plot.py ######################################
os.system("%s %s"%(sys.executable, os.path.join(DIR_PATH, 'plot.py')))
