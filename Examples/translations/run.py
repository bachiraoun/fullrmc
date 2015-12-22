# standard libraries imports
import os

# external libraries imports

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Generators.Translations import TranslationGenerator, TranslationAlongSymmetryAxisGenerator
from fullrmc.Core.Collection import get_principal_axis

# shut down logging
LOGGER.set_log_file_basename("fullrmc")
LOGGER.set_minimum_level(30)

pdbPath = "nagma_single_molecule.pdb" 
ENGINE = Engine(pdb=pdbPath, constraints=None)

# set groups as the whole molecule
ENGINE.set_groups_as_molecules()   

nsteps = 500
xyzFrequency = 1



# run engine translation along axis 0
xyzPath="along0.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Translation along symmetry axis 0: ",X
[g.set_move_generator(TranslationAlongSymmetryAxisGenerator(amplitude=0.5, axis=0)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

# run engine translation along axis 1
xyzPath="along1.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Translation along symmetry axis 1: ", Y
[g.set_move_generator(TranslationAlongSymmetryAxisGenerator(amplitude=0.5, axis=1)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

# run engine translation along axis 2
xyzPath="along2.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
_,_,_,_,X,Y,Z =get_principal_axis(ENGINE.realCoordinates)
print "Translation along symmetry axis 2: ", Z
[g.set_move_generator(TranslationAlongSymmetryAxisGenerator(amplitude=0.5, axis=2)) for g in ENGINE.groups]    
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)

# run engine random rotations
xyzPath="random.xyz"
print "Random translation"
if os.path.isfile(xyzPath): os.remove(xyzPath)
[g.set_move_generator(TranslationGenerator(amplitude=0.5)) for g in ENGINE.groups]
ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, xyzFrequency=xyzFrequency, xyzPath=xyzPath)
 

ENGINE.visualize()    
    
 






