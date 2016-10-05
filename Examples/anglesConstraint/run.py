##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint


##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
#####################################  CREATE ENGINE  ####################################
pdbPath = "system.pdb" 
ENGINE = Engine(path=None)
ENGINE.set_pdb(pdbPath)

# add constraints
B_CONSTRAINT  = BondConstraint()
BA_CONSTRAINT = BondsAngleConstraint()
ENGINE.add_constraints([B_CONSTRAINT, BA_CONSTRAINT]) 
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"TIP": [('OH2' ,'H1' , 0.8, 1.1),
                                                                  ('OH2' ,'H2' , 0.8, 1.1)] })
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"TIP": [ ('OH2'  ,'H1' ,'H2' , 80, 120)] })

# set TranslationGenerator move generators amplitude
[g.moveGenerator.set_amplitude(0.025) for g in ENGINE.groups]


##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def run_normal(nsteps, xyzPath):
    BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"TIP": [ ('OH2'  ,'H1' ,'H2' , 80, 120)] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)

def run_squeezed(nsteps, xyzPath):
    BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"TIP": [ ('OH2'  ,'H1' ,'H2' , 30, 40)] })                                
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)

def run_streched(nsteps, xyzPath):
    BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"TIP": [ ('OH2'  ,'H1' ,'H2' , 160, 170)] })                                
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)
    
    
##########################################################################################
#####################################  RUN SIMULATION  ###################################
xyzPath ="trajectory.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
run_normal(2000,   xyzPath)
run_squeezed(2000, xyzPath)
run_streched(2000, xyzPath)


##########################################################################################
##################################  VISUALIZE SIMULATION  ################################
ENGINE.set_pdb(pdbPath)
ENGINE.visualize( commands = ["trajectory.xyz"], 
                  boxWidth=0, bgColor="white",
                  representationParams='CPK 1.0 0.2 50 50',
                  otherParams=["label add Angles 0/1 0/0 0/2",
                               "label textsize 1.5",
                               "label textthickness 2",
                               "color Labels Angles black",] )    
    
 





 
    
 






