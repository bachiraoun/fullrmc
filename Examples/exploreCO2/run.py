##########################################################################################
#############################  IMPORTING USEFUL DEFINITIONS  #############################
# standard distribution imports
import os

# external libraries imports

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairCorrelationConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
from fullrmc.Generators.Translations import TranslationGenerator
from fullrmc.Generators.Rotations import RotationGenerator

##########################################################################################
#####################################  CREATE ENGINE  ####################################
# file names
expDataPath = "Xrays.gr"
pdbPath = "CO2.pdb"
enginePath = "CO2.rmc"
# initialize engine
ENGINE = Engine(pdb=pdbPath, constraints=None)
PDF_CONSTRAINT = PairDistributionConstraint(engine=None, experimentalData=expDataPath, weighting="atomicNumber")
IMD_CONSTRAINT = InterMolecularDistanceConstraint(engine=None, defaultDistance=1.4)
B_CONSTRAINT   = BondConstraint(engine=None)
BA_CONSTRAINT  = BondsAngleConstraint(engine=None)
# add constraints
ENGINE.add_constraints([PDF_CONSTRAINT, IMD_CONSTRAINT, B_CONSTRAINT, BA_CONSTRAINT])
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"CO2": [('C' ,'O1' , 0.52, 1.4),
                                                                  ('C' ,'O2' , 0.52, 1.4)] })
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"CO2": [ ('C' ,'O1' ,'O2' , 170, 180)] })
# set all constraints as used. Used value is True by default! 
# Now you know you can deactivate any constraint at any time though.
PDF_CONSTRAINT.set_used(True)
IMD_CONSTRAINT.set_used(True)
B_CONSTRAINT.set_used(True)
BA_CONSTRAINT.set_used(True)
# initialize constraints data
ENGINE.initialize_used_constraints()


##########################################################################################
#####################################  DIFFERENT RUNS  ################################### 
def run_atoms(ENGINE, rang=None, recur=None, xyzFrequency=500):
    ENGINE.set_groups(None)  
    # set selector
    if recur is None: recur = 10
    ENGINE.set_group_selector(RandomSelector(ENGINE))
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    if rang is None: rang=20
    for stepIdx in range(rang):
        LOGGER.info("Running 'atoms' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, savePath=enginePath, saveFrequency=nsteps, xyzFrequency=xyzFrequency, xyzPath="atomsTraj.xyz")

def run_molecules(ENGINE, rang=5, recur=100, refine=False, explore=True, exportPdb=False, xyzFrequency=500):
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator( MoveGeneratorCollector(collection=[TranslationGenerator(amplitude=0.2),RotationGenerator(amplitude=2)],randomize=True) ) for g in ENGINE.groups]
    # number of steps
    nsteps = 20*len(ENGINE.groups)
    # set selector
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    for stepIdx in range(rang):
        LOGGER.info("Running 'molecules' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=enginePath, xyzFrequency=xyzFrequency, xyzPath="moleculeTraj.xyz")
         
def run_recurring_atoms(ENGINE, rang=None, recur=None, explore=True, refine=False, xyzFrequency=500):
    ENGINE.set_groups(None)  
    # set selector
    if recur is None: recur = 10
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    if rang is None: rang=20
    for stepIdx in range(rang):
        LOGGER.info("Running 'explore' mode step %i"%(stepIdx))
        if explore:
            ENGINE.run(numberOfSteps=nsteps, savePath=enginePath, saveFrequency=nsteps, xyzFrequency=xyzFrequency, xyzPath="exploreTraj.xyz")
        elif refine:            
            ENGINE.run(numberOfSteps=nsteps, savePath=enginePath, saveFrequency=nsteps, xyzFrequency=xyzFrequency, xyzPath="refineTraj.xyz")
        else:
            ENGINE.run(numberOfSteps=nsteps, savePath=enginePath, saveFrequency=nsteps, xyzFrequency=xyzFrequency, xyzPath="recurTraj.xyz")
            
##########################################################################################
####################################  RUN SIMULATION  ####################################
# remove all .xyz trajectory files
files = [f for f in os.listdir(".") if os.path.isfile(f) and ".xyz" in f]
[os.remove(fname) for fname in files]
# run atoms
run_atoms(ENGINE, rang=4, xyzFrequency=None)
run_molecules(ENGINE, xyzFrequency=None)
run_recurring_atoms(ENGINE, rang=8, explore=True, refine=False, xyzFrequency=None) 
run_recurring_atoms(ENGINE, rang=4, explore=False, refine=True, xyzFrequency=None) 

##########################################################################################
##################################  PLOT PDF CONSTRAINT  #################################
PDF_CONSTRAINT.plot()
    
    
    