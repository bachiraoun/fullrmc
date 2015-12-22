# standard libraries imports
import os

# external libraries imports

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Generators.Rotations import RotationGenerator, RotationAboutAxisGenerator, RotationAboutSymmetryAxisGenerator
from fullrmc.Core.Collection import get_principal_axis

# shut down logging
LOGGER.set_log_file_basename("fullrmc")
LOGGER.set_minimum_level(30)

pdbPath = "thf_single_molecule.pdb" 
ENGINE = Engine(pdb=pdbPath, constraints=None)

# set groups as the whole molecule
ENGINE.set_groups_as_molecules()   

nsteps = 500
xyzFrequency = 1



# run engine rotation about axis 0
xyzPath="about0.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Rotation about symmetry axis 0: ",X
[g.set_move_generator(RotationAboutSymmetryAxisGenerator(amplitude=10, axis=0)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

# run engine rotation about axis 1
xyzPath="about1.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Rotation about symmetry axis 1: ", Y
[g.set_move_generator(RotationAboutSymmetryAxisGenerator(amplitude=10, axis=1)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

# run engine rotation about axis 2
xyzPath="about2.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Rotation about symmetry axis 2: ", Z
[g.set_move_generator(RotationAboutSymmetryAxisGenerator(amplitude=10, axis=2)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

# run engine random rotations
xyzPath="random.xyz"
print "Random rotation"
if os.path.isfile(xyzPath): os.remove(xyzPath)
[g.set_move_generator(RotationGenerator(amplitude=10)) for g in ENGINE.groups]
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)
 
# run engine rotation about defined axis
xyzPath="aboutAxis.xyz"
axis=(1,1,1)
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Rotation about user defined axis: ",axis
[g.set_move_generator(RotationAboutAxisGenerator(amplitude=10, axis=axis)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)
 
ENGINE.visualize()    
    
 






