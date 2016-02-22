import itertools
import os

# external libraries imports
import numpy as np
from pdbParser.pdbParser import pdbParser

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairCorrelationConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Generators.Translations import TranslationGenerator


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
# initialize constraints data
PDF_CONSTRAINT.set_used(True)
IMD_CONSTRAINT.set_used(True)
B_CONSTRAINT.set_used(True)
BA_CONSTRAINT.set_used(True)
ENGINE.initialize_used_constraints()
#ENGINE = Engine(pdb=None).load(enginePath)
        
# ############ RUN ATOMS ############ #    
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

# ############ EXPLORE ATOMS ############ #          
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
            
# tweak constraints
PDF_CONSTRAINT = ENGINE.constraints[0]
IM_CONSTRAINT  = ENGINE.constraints[1]
B_CONSTRAINT   = ENGINE.constraints[2]

# remove all .xyz trajectory files
files = [f for f in os.listdir(".") if os.path.isfile(f) and ".xyz" in f]
[os.remove(fname) for fname in files]

# run atoms
run_atoms(ENGINE, rang=4)
run_recurring_atoms(ENGINE, rang=8, explore=True, refine=False) 
run_recurring_atoms(ENGINE, rang=4, explore=False, refine=True) 

    