

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
from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint


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
IA_CONSTRAINT  = ImproperAngleConstraint()
ENGINE.add_constraints([B_CONSTRAINT, BA_CONSTRAINT, IA_CONSTRAINT]) 
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"PFT": [('Xe' ,'F1' , 1.9, 2.1),
                                                                  ('Xe' ,'F2' , 1.9, 2.1),
                                                                  ('Xe' ,'F3' , 1.9, 2.1),
                                                                  ('Xe' ,'F4' , 1.9, 2.1),
                                                                  ('Xe' ,'F5' , 1.9, 2.1),] })
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"PFT": [ ('Xe' ,'F1' ,'F2' , 60 , 80),
                                                                      ('Xe' ,'F2' ,'F3' , 60 , 80),
                                                                      ('Xe' ,'F3' ,'F4' , 60 , 80),
                                                                      ('Xe' ,'F4' ,'F5' , 60 , 80),
                                                                      ('Xe' ,'F5' ,'F1' , 60 , 80),] })

IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"PFT": [ ('F4','F2','F1','F3', -2, 2),
                                                                      ('F5','F2','F1','F3', -2, 2),
                                                                      ('Xe','F2','F1','F3', -2, 2) ] })

# set TranslationGenerator move generators amplitude
[g.moveGenerator.set_amplitude(0.1) for g in ENGINE.groups]


##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def run_normal(nsteps, xyzPath):
    IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"PFT": [ ('F4','F2','F1','F3', -2, 2),
                                                                          ('F5','F2','F1','F3', -2, 2),
                                                                          ('Xe','F2','F1','F3', -2, 2) ] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)

def run_loosen_1(nsteps, xyzPath):
    IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"PFT": [ ('F4','F2','F1','F3', -60, -50),
                                                                          ('F5','F2','F1','F3', -60, -50),
                                                                          ('Xe','F2','F1','F3', -60, -50) ] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)

def run_loosen_2(nsteps, xyzPath):
    IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"PFT": [ ('F4','F2','F1','F3', 50, 60),
                                                                          ('F5','F2','F1','F3', 50, 60),
                                                                          ('Xe','F2','F1','F3', 50, 60) ] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)
                                                                          

    
##########################################################################################
#####################################  RUN SIMULATION  ###################################
xyzPath ="trajectory.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
run_normal(1000,   xyzPath)
run_loosen_1(1000, xyzPath)
run_loosen_2(1000, xyzPath)
run_normal(1000,   xyzPath)

##########################################################################################
##################################  VISUALIZE SIMULATION  ################################
ENGINE.set_pdb(pdbPath)
ENGINE.visualize( commands = ["trajectory.xyz"], 
                  boxWidth=0, bgColor="white",
                  representationParams='CPK 1.0 0.2 50 50',
                  otherParams = ["label add Atoms 0/0",
                                 "label add Atoms 0/1", 
                                 "label add Atoms 0/2",
                                 "label add Atoms 0/3",
                                 "label add Atoms 0/4",
                                 "label add Atoms 0/5",
                                 "label textsize 1.5",
                                 "label textthickness 2",
                                 "color Labels Atoms black"] )    