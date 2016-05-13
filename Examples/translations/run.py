##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

# external libraries imports

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Generators.Translations import TranslationGenerator, TranslationAlongSymmetryAxisGenerator
from fullrmc.Core.Collection import get_principal_axis


##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
#####################################  CREATE ENGINE  ####################################
pdbPath = "molecule.pdb"
ENGINE = Engine(pdb=pdbPath, constraints=None)

# set groups as the whole molecule
ENGINE.set_groups_as_molecules()   

nsteps = 500
xyzFrequency = 1


##########################################################################################
#####################################  DIFFERENT RUNS  ###################################
def along_axis_0():
    # run engine translation along axis 0
    xyzPath="along0.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    _,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
    print "Translation along symmetry axis 0: ",X
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(amplitude=0.5, axis=0)) for g in ENGINE.groups]    
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

def along_axis_1():
    # run engine translation along axis 1
    xyzPath="along1.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    _,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
    print "Translation along symmetry axis 1: ", Y
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(amplitude=0.5, axis=1)) for g in ENGINE.groups]    
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

def along_axis_2():    
    # run engine translation along axis 2
    xyzPath="along2.xyz"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    _,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
    print "Translation along symmetry axis 2: ", Z
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(amplitude=0.5, axis=2)) for g in ENGINE.groups]    
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

def random():    
    # run engine random translations
    xyzPath="random.xyz"
    print "Random translation"
    if os.path.isfile(xyzPath): os.remove(xyzPath)
    [g.set_move_generator(TranslationGenerator(amplitude=0.5)) for g in ENGINE.groups]
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)
 
 
##########################################################################################
#####################################  RUN SIMULATION  ###################################
along_axis_0()
along_axis_1()
along_axis_2()
random() 


##########################################################################################
##################################  VISUALIZE SIMULATION  ################################
ENGINE.set_pdb(pdbPath)
ENGINE.visualize( commands = ["along0.xyz", "along1.xyz", "along2.xyz", "random.xyz"],
                  boxWidth=0, representationParams='CPK 1.0 0.2 50 50')    
    
 






