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
from fullrmc.Constraints.DihedralAngleConstraints import DihedralAngleConstraint


##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
#####################################  CREATE ENGINE  ####################################
pdbPath = "system.pdb" 
ENGINE = Engine(path=None)
ENGINE.set_pdb(pdbPath)

# add constraints
B_CONSTRAINT   = BondConstraint()
BA_CONSTRAINT  = BondsAngleConstraint()
DA_CONSTRAINT  = DihedralAngleConstraint()
ENGINE.add_constraints([B_CONSTRAINT, BA_CONSTRAINT, DA_CONSTRAINT]) 
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"BUT": [# C-C bonds
                                                                  ('C1' ,'C2' , 1.33, 1.73),
                                                                  ('C2' ,'C3' , 1.33, 1.73),
                                                                  ('C3' ,'C4' , 1.33, 1.73),
                                                                  # C-H3 bonds
                                                                  ('C1' ,'H11', 1.01, 1.21),
                                                                  ('C1' ,'H12', 1.01, 1.21),
                                                                  ('C1' ,'H13', 1.01, 1.21),
                                                                  ('C4' ,'H41', 1.01, 1.21),
                                                                  ('C4' ,'H42', 1.01, 1.21),
                                                                  ('C4' ,'H43', 1.01, 1.21),
                                                                  # C-H2 bonds
                                                                  ('C2' ,'H21', 0.99, 1.19),
                                                                  ('C2' ,'H22', 0.99, 1.19),
                                                                  ('C3' ,'H31', 0.99, 1.19),
                                                                  ('C3' ,'H32', 0.99, 1.19),
                                                                  # VDW bonds
                                                                  ('C2' ,'H11', 2.0, 100),
                                                                  ('C2' ,'H12', 2.0, 100),
                                                                  ('C2' ,'H13', 2.0, 100),
                                                                  ('C3' ,'H41', 2.0, 100),
                                                                  ('C3' ,'H42', 2.0, 100),
                                                                  ('C3' ,'H43', 2.0, 100),
                                                                  ('C2' ,'H31', 2.0, 100),
                                                                  ('C2' ,'H32', 2.0, 100),
                                                                  ('C3' ,'H21', 2.0, 100),
                                                                  ('C3' ,'H22', 2.0, 100),
                                                                  ('H11','H21', 2.0, 100),
                                                                  ('H11','H22', 2.0, 100),
                                                                  ('H12','H21', 2.0, 100),
                                                                  ('H12','H22', 2.0, 100),
                                                                  ('H13','H21', 2.0, 100),
                                                                  ('H13','H22', 2.0, 100),
                                                                  ('H41','H31', 2.0, 100),
                                                                  ('H41','H32', 2.0, 100),
                                                                  ('H42','H31', 2.0, 100),
                                                                  ('H42','H32', 2.0, 100),
                                                                  ('H43','H31', 2.0, 100),
                                                                  ('H43','H32', 2.0, 100),
                                                                  ('H31','H21', 2.0, 100),
                                                                  ('H31','H22', 2.0, 100),
                                                                  ('H32','H21', 2.0, 100),
                                                                  ('H32','H22', 2.0, 100),] })
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"BUT": [ # C-C-C angles
                                                                      ('C2' ,'C1' ,'C3' , 99 , 119),
                                                                      ('C3' ,'C1' ,'C4' , 99 , 119),
                                                                      # C-H3 angles
                                                                      ('C1' ,'H11','H12', 99 , 119),
                                                                      ('C1' ,'H11','H13', 99 , 119),
                                                                      ('C1' ,'H12','H13', 99 , 119),
                                                                      ('C4' ,'H41','H42', 99 , 119),
                                                                      ('C4' ,'H41','H43', 99 , 119),
                                                                      ('C4' ,'H42','H43', 99 , 119),
                                                                      # C-H2 angles
                                                                      ('C2' ,'H21','H22', 99 , 119),
                                                                      ('C3' ,'H31','H32', 99 , 119),
                                                                      # C-C-H angles
                                                                      ('C1' ,'C2' ,'H11', 101, 121),
                                                                      ('C1' ,'C2' ,'H12', 101, 121),
                                                                      ('C1' ,'C2' ,'H13', 101, 121),
                                                                      ('C4' ,'C3' ,'H41', 101, 121),
                                                                      ('C4' ,'C3' ,'H42', 101, 121),
                                                                      ('C4' ,'C3' ,'H43', 101, 121),
                                                                      ('C2' ,'C1' ,'H21', 97 , 117),
                                                                      ('C2' ,'C3' ,'H22', 97 , 117),
                                                                      ('C3' ,'C2' ,'H31', 97 , 117),
                                                                      ('C3' ,'C4' ,'H32', 97 , 117)] })
DA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 30,90, 150,210, 270,330)] })


# set TranslationGenerator move generators amplitude
ENGINE.set_groups_as_atoms()
[g.moveGenerator.set_amplitude(0.1) for g in ENGINE.groups]


##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def run_first_shell(nsteps, xyzPath, xyzFreq=100):
    DA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 150,160, 150,160, 150,160) ] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=xyzFreq, xyzPath=xyzPath, restartPdb=None)

def run_second_shell(nsteps, xyzPath, xyzFreq=100):
    DA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 10,30, 10,30, 10,30) ] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=xyzFreq, xyzPath=xyzPath, restartPdb=None)

def run_third_shell(nsteps, xyzPath, xyzFreq=100):
    DA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 90,100, 90,100, 90,100) ] })
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=xyzFreq, xyzPath=xyzPath, restartPdb=None)

##########################################################################################
#####################################  RUN SIMULATION  ###################################
xyzPath ="trajectory.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
run_first_shell(100000,   xyzPath)
run_second_shell(100000,  xyzPath)
run_third_shell(100000,   xyzPath)


##########################################################################################
##################################  VISUALIZE SIMULATION  ################################
ENGINE.set_pdb(pdbPath)
# VMD dihedral angle calculation is between -180 and 180 and direction is arbitrary.
# Therefore dihedral angle label can be anything as angle, -angle, 360+angle and 
# 360-angle
ENGINE.visualize(commands = ["trajectory.xyz"], 
                 boxWidth=0, bgColor="white",
                 representationParams='CPK 1.0 0.2 50 50',
                 otherParams = ["axes location off",
                                "label add Dihedrals 0/0 0/1 0/2 0/3",
                                "label textsize 1.5",
                                "label textthickness 2",
                                "color Labels Dihedrals black"] ) 


