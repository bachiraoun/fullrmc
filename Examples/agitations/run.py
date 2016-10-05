##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Generators.Agitations import DistanceAgitationGenerator, AngleAgitationGenerator
from fullrmc.Selectors.OrderedSelectors import DefinedOrderSelector
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint


##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
#####################################  CREATE ENGINE  ####################################
pdbPath = "waterBox.pdb"
ENGINE = Engine(path=None)
ENGINE.set_pdb( pdbPath  )

# add constraints
B_CONSTRAINT  = BondConstraint()
BA_CONSTRAINT = BondsAngleConstraint()
ENGINE.add_constraints([B_CONSTRAINT, BA_CONSTRAINT]) 
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"TIP": [('OH2' ,'H1' , 0.8, 1.1),
                                                                  ('OH2' ,'H2' , 0.8, 1.1)] })
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"TIP": [ ('OH2'  ,'H1' ,'H2' , 85, 120)] })

##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def agitate_bonds():
    print "Agitate bonds"
    # agitate bonds
    groups = []
    for idx in range(0,ENGINE.pdb.numberOfAtoms,3):
        groups.append(np.array([idx, idx+1]))
        groups.append(np.array([idx, idx+2]))
    ENGINE.set_groups(groups) 
    [g.set_move_generator(DistanceAgitationGenerator(amplitude=0.1,agitate=(False,True))) for g in ENGINE.groups]
    # set ordered selector
    ENGINE.set_group_selector(DefinedOrderSelector(ENGINE, order=None) )
    # run engine
    xyzPath="bonds.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    nsteps = 250*len(ENGINE.groups)
    xyzFrequency = len(ENGINE.groups)
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath, restartPdb=None)

def agitate_angles():
    print "Agitate angles"
    # agitate angles
    ENGINE.set_groups_as_molecules() 
    [g.set_move_generator(AngleAgitationGenerator(amplitude=5)) for g in ENGINE.groups]
    # set ordered selector
    ENGINE.set_group_selector(DefinedOrderSelector(ENGINE, order=None) )
    # run engine
    xyzPath="angles.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    nsteps = 250*len(ENGINE.groups)
    xyzFrequency = len(ENGINE.groups)
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath, restartPdb=None)

def agitate_both():
    print "Agitate both"
    # agitate all
    ENGINE.set_groups_as_molecules() 
    groups = [g for g in ENGINE.groups]
    for idx in range(0,ENGINE.pdb.numberOfAtoms,3):
        groups.append(np.array([idx, idx+1]))
        groups.append(np.array([idx, idx+2]))
    ENGINE.add_groups(groups)  
    # set move generator
    for g in ENGINE.groups:
        if len(g)==2:
            g.set_move_generator(DistanceAgitationGenerator(amplitude=0.1, agitate=(False,True)))
        elif len(g)==3:
            g.set_move_generator(AngleAgitationGenerator(amplitude=5))
        else:
            raise 
    # set ordered selector
    ENGINE.set_group_selector(DefinedOrderSelector(ENGINE, order=None) )
    # set ordered selector
    ENGINE.set_group_selector(DefinedOrderSelector(ENGINE, order=None) )
    # run engine
    xyzPath="both.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    nsteps = 250*len(ENGINE.groups)
    xyzFrequency = len(ENGINE.groups)
    # run engine
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath, restartPdb=None)

 
##########################################################################################
#####################################  RUN SIMULATION  ###################################
agitate_bonds()
agitate_angles()
agitate_both()


##########################################################################################
#################################  VISUALIZE SIMULATION  #################################
ENGINE.set_pdb(pdbPath)
ENGINE.visualize( commands = ["bonds.xyz", "angles.xyz", "both.xyz"],
                  boxToCenter=True, boxWidth=1, 
                  representationParams='CPK 1.0 0.2 50 50')    
    
 






