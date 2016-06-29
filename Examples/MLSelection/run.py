##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

# external libraries imports
import wx
import numpy as np
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint
from fullrmc.Generators.Translations import TranslationGenerator
from fullrmc.Selectors.RandomSelectors import RandomSelector, SmartRandomSelector


##########################################################################################
###################################  SHUT DOWN LOGGING  ##################################
LOGGER.set_log_to_stdout_flag(False)
# set very high logging level and force 'move accepted'.
LOGGER.set_minimum_level(sys.maxint)
LOGGER.force_log_type_flags("move accepted", stdoutFlag=False, fileFlag=True)

##########################################################################################
#####################################  CREATE ENGINE  ####################################
# set log files name and turn off stout logging
normalSelLog = "normal"
MLSelLog     = "ML"

# runtime arguments
numberOfSteps = 2000000

# set engine variables
pdbPath = "thf.pdb"
enginePath = "engine.rmc"

# create engine
ENGINE = Engine(pdb=pdbPath, constraints=None)
# initialize constraints
B_CONSTRAINT   = BondConstraint(engine=None)
BA_CONSTRAINT  = BondsAngleConstraint(engine=None)
IA_CONSTRAINT  = ImproperAngleConstraint(engine=None)
# add constraints
ENGINE.add_constraints([B_CONSTRAINT])
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"THF": [('O' ,'C1' , 1.20, 1.70),
                                                                  ('O' ,'C4' , 1.20, 1.70),
                                                                  ('C1','C2' , 1.25, 1.90),
                                                                  ('C2','C3' , 1.25, 1.90),
                                                                  ('C3','C4' , 1.25, 1.90),
                                                                  ('C1','H11', 0.88, 1.16),('C1','H12', 0.88, 1.16),
                                                                  ('C2','H21', 0.88, 1.16),('C2','H22', 0.88, 1.16),
                                                                  ('C3','H31', 0.88, 1.16),('C3','H32', 0.88, 1.16),
                                                                  ('C4','H41', 0.88, 1.16),('C4','H42', 0.88, 1.16)] })
ENGINE.add_constraints([BA_CONSTRAINT])
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"THF": [ ('O'  ,'C1' ,'C4' , 105, 125),
                                                                      ('C1' ,'O'  ,'C2' , 100, 120),
                                                                      ('C4' ,'O'  ,'C3' , 100, 120),
                                                                      ('C2' ,'C1' ,'C3' , 95 , 115),
                                                                      ('C3' ,'C2' ,'C4' , 95 , 115),
                                                                      # H-C-H angle
                                                                      ('C1' ,'H11','H12', 98 , 118),
                                                                      ('C2' ,'H21','H22', 98 , 118),
                                                                      ('C3' ,'H31','H32', 98 , 118),
                                                                      ('C4' ,'H41','H42', 98 , 118),
                                                                      # H-C-O angle
                                                                      ('C1' ,'H11','O'  , 100, 120),
                                                                      ('C1' ,'H12','O'  , 100, 120),
                                                                      ('C4' ,'H41','O'  , 100, 120),
                                                                      ('C4' ,'H42','O'  , 100, 120),                                                                           
                                                                      # H-C-C
                                                                      ('C1' ,'H11','C2' , 103, 123),
                                                                      ('C1' ,'H12','C2' , 103, 123),
                                                                      ('C2' ,'H21','C1' , 103, 123),
                                                                      ('C2' ,'H21','C3' , 103, 123),
                                                                      ('C2' ,'H22','C1' , 103, 123),
                                                                      ('C2' ,'H22','C3' , 103, 123),
                                                                      ('C3' ,'H31','C2' , 103, 123),
                                                                      ('C3' ,'H31','C4' , 103, 123),
                                                                      ('C3' ,'H32','C2' , 103, 123),
                                                                      ('C3' ,'H32','C4' , 103, 123),
                                                                      ('C4' ,'H41','C3' , 103, 123),
                                                                      ('C4' ,'H42','C3' , 103, 123) ] })
ENGINE.add_constraints([IA_CONSTRAINT])
IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"THF": [ ('C2','O','C1','C4', -15, 15),
                                                                      ('C3','O','C1','C4', -15, 15) ] })
# initialize constraints data
ENGINE.initialize_used_constraints()
ENGINE.save(enginePath)

##########################################################################################
################################  CREATE MOVES GENERATORS  ###############################
# set all move generators to Translation with a maximum amplutide of 0.3 A
[g.set_move_generator(TranslationGenerator(amplitude=0.3)) for g in ENGINE.groups]
# set randomly 25% of the translation amplitude to 10A
[g.set_move_generator(TranslationGenerator(amplitude=10.)) for g in ENGINE.groups if np.random.random()>0.25]


##########################################################################################
#####################################  DIFFERENT RUNS  ################################### 
#################### run normal selector ####################
def normal_run(engine=ENGINE):
    LOGGER.force_log("info", "normal selection %i started... DON'T INTERRUPT"%numberOfSteps, stdout=True, file=False)
    # delete existing log files
    normalLogs = [fn for fn in next(os.walk("."))[2] if ".log" in fn and normalSelLog in fn]
    [os.remove(l) for l in normalLogs]
    # set log file name
    LOGGER.set_log_file_basename(normalSelLog)
    # load fresh engine
    engine = engine.load(enginePath)
    # set random group selector
    engine.set_group_selector(RandomSelector(engine))
    # run engine
    engine.run(numberOfSteps=numberOfSteps, saveFrequency=2*numberOfSteps)
    LOGGER.force_log("info", "normal selection finished", stdout=True, file=False)
    
############### run machine learning selector ###############
def ML_run(engine=ENGINE):
    LOGGER.force_log("info", "machine learning selection %i started... DON'T INTERRUPT"%numberOfSteps, stdout=True, file=False)
    # delete existing log files
    MLLogs = [fn for fn in next(os.walk("."))[2] if ".log" in fn and MLSelLog in fn]
    [os.remove(l) for l in MLLogs]
    # set log file name
    LOGGER.set_log_file_basename(MLSelLog)
    # load fresh engine
    engine = engine.load(enginePath)
    # set smart group selector
    engine.set_group_selector(SmartRandomSelector(engine))
    # run engine
    engine.run(numberOfSteps=numberOfSteps, saveFrequency=2*numberOfSteps)
    LOGGER.force_log("info", "machine learning selection finished", stdout=True, file=False)

##########################################################################################
#####################################  RUN SIMULATION  ###################################
normal_run()
ML_run()
LOGGER.flush()

##########################################################################################
################################  READ LOGGING FILES DATA ################################
# machine learning logs
MLSelLog = sorted([fn for fn in next(os.walk("."))[2] if ".log" in fn and MLSelLog in fn])
mlGenerated = []
mlAccepted  = []
for log in MLSelLog:
    fd = open(log,'r')
    mlLines = fd.readlines()
    fd.close()
    mlGenerated.extend([float(l.split("Generated:")[1].split("-")[0]) for l in mlLines])
    mlAccepted.extend([float(l.split("Accepted:")[1].split("(")[0]) for l in mlLines])    
mlGenerated = np.array(mlGenerated)
mlAccepted = np.array(mlAccepted)
mlAccepted = 100.*mlAccepted/mlGenerated
np.savetxt(X=np.transpose([mlGenerated,mlAccepted]), 
           fname="MLSelection.dat", 
           fmt='%.3f', delimiter="   ",
           header="Generated    Accepted(%)")
# normal logs
normalLogs = sorted([fn for fn in next(os.walk("."))[2] if ".log" in fn and normalSelLog in fn])
nGenerated = []
nAccepted  = []
for log in normalLogs:
    fd = open(log,'r')
    mlLines = fd.readlines()
    fd.close()
    nGenerated.extend([float(l.split("Generated:")[1].split("-")[0]) for l in mlLines])
    nAccepted.extend([float(l.split("Accepted:")[1].split("(")[0]) for l in mlLines])    
nGenerated = np.array(nGenerated)
nAccepted = np.array(nAccepted)
nAccepted = 100.*nAccepted/nGenerated
np.savetxt(X=np.transpose([nGenerated,nAccepted]), 
           fname="traditionalSelection.dat", 
           fmt='%.3f', delimiter="   ",
           header="Generated    Accepted(%)")
           

##########################################################################################
##################################  PLOT LOGGING DATA ####################################    
plt.plot(mlGenerated, mlAccepted, 'black',linewidth=3, label="machine learning selection")
plt.plot(nGenerated, nAccepted, 'red', linewidth=3, label="traditional selection")
plt.xlabel("Generated moves")
plt.ylabel("Accepted moves (%)")
plt.legend(frameon=False, loc="upper left")
plt.show()

  
    





