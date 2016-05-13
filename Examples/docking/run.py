##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Generators.Rotations import RotationGenerator
from fullrmc.Generators.Translations import TranslationTowardsCenterGenerator


##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
#####################################  CREATE ENGINE  ####################################
pdbPath = "system.pdb" 
ENGINE = Engine(pdb=pdbPath, constraints=None)

# add inter-molecular distance constraint
EMD_CONSTRAINT = InterMolecularDistanceConstraint(engine=ENGINE, defaultDistance=1.75)
ENGINE.add_constraints([EMD_CONSTRAINT]) 

##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def move_towards():
    # set only one molecule group
    ENGINE.set_groups_as_molecules() 
    secMolIdxs = ENGINE.groups[1].indexes 
    ENGINE.set_groups( ENGINE.groups[0] )
    # set move generator
    for g in ENGINE.groups:
        t = TranslationTowardsCenterGenerator(center={'indexes': secMolIdxs},amplitude=0.15, angle=90)
        r = RotationGenerator(amplitude=10)
        mg = MoveGeneratorCollector(collection=[t,r], randomize=True, weights=[(0,1),(1,5)])
        g.set_move_generator(mg)
    # set runtime parameters    
    nsteps = 1000
    xyzFrequency = 1
    # run engine
    xyzPath="trajectory.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

 

##########################################################################################
#####################################  RUN SIMULATION  ###################################
move_towards()


##########################################################################################
##################################  VISUALIZE SIMULATION  ################################
ENGINE.set_pdb(pdbPath)
ENGINE.visualize( commands = ["trajectory.xyz"], 
                  boxWidth=0, 
                  representationParams='CPK 1.0 0.2 50 50')    
    
 





 
    
 






