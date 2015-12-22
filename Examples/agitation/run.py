# standard libraries imports
import os

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Generators.Agitations import DistanceAgitationGenerator, AngleAgitationGenerator
from fullrmc.Selectors.OrderedSelectors import DefinedOrderSelector
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint

# pdbParser imports
from pdbParser import pdbParser
from pdbParser.Utilities.Geometry import translate, get_center


# shut down logging
LOGGER.set_log_file_basename("fullrmc")
LOGGER.set_minimum_level(30)

pdbPath = "water.pdb" 
pdb = pdbParser(pdbPath)
translate(pdb.indexes, pdb, -get_center(pdb.indexes, pdb))
pdb.export_pdb(pdbPath)
ENGINE = Engine(pdb=pdb, constraints=None)

# add constraints
B_CONSTRAINT  = BondConstraint(engine=None)
BA_CONSTRAINT = BondsAngleConstraint(engine=None)
ENGINE.add_constraints([B_CONSTRAINT, BA_CONSTRAINT]) 
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"TIP": [('OH2' ,'H1' , 0.8, 1.2),
                                                                  ('OH2' ,'H2' , 0.8, 1.2)] })
ENGINE.add_constraints([BA_CONSTRAINT])
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"TIP": [ ('OH2'  ,'H1' ,'H2' , 85, 130)] })


# set ordered selector
ENGINE.set_group_selector(DefinedOrderSelector(ENGINE, order=None) )

# agitate bonds
#ENGINE.set_groups_as_molecules() 
#groups = [g for g in ENGINE.groups]
groups = []
for idx in range(0,ENGINE.pdb.numberOfAtoms,3):
    groups.append(np.array([idx, idx+1]))
    groups.append(np.array([idx, idx+2]))
ENGINE.set_groups(groups)   
[g.set_move_generator(DistanceAgitationGenerator(amplitude=0.2,agitate=(False,True))) for g in ENGINE.groups]
# set ordered selector
ENGINE.set_group_selector(DefinedOrderSelector(ENGINE, order=None) )
# run engine
xyzPath="bonds.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
nsteps = 250*len(ENGINE.groups)
xyzFrequency = len(ENGINE.groups)
ENGINE.groupSelector.set_order(range(len(ENGINE.groups)))
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)


# agitate angles
ENGINE.set_groups_as_molecules() 
[g.set_move_generator(AngleAgitationGenerator(amplitude=5)) for g in ENGINE.groups]
# run engine
xyzPath="angles.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
nsteps = 250*len(ENGINE.groups)
xyzFrequency = len(ENGINE.groups)
ENGINE.groupSelector.set_order(range(len(ENGINE.groups)))
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)


# agitate all
ENGINE.set_groups_as_molecules() 
groups = [g for g in ENGINE.groups]
for idx in range(0,ENGINE.pdb.numberOfAtoms,3):
    groups.append(np.array([idx, idx+1]))
    groups.append(np.array([idx, idx+2]))
ENGINE.set_groups(groups)  
# set move generator
for g in ENGINE.groups:
    if len(g)==2:
        g.set_move_generator(DistanceAgitationGenerator(amplitude=0.2,agitate=(False,True)))
    elif len(g)==3:
        g.set_move_generator(AngleAgitationGenerator(amplitude=5))
    else:
        raise 
# run engine
xyzPath="all.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
nsteps = 250*len(ENGINE.groups)
xyzFrequency = len(ENGINE.groups)
ENGINE.groupSelector.set_order(range(len(ENGINE.groups)))
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)


# visualize
ENGINE.visualize()    
    
 
    
 






