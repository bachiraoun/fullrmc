##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER, FLOAT_TYPE
from fullrmc.Engine import Engine
from fullrmc.Core.Collection import generate_random_float, get_rotation_matrix, rotate
from fullrmc.Generators.Translations import TranslationGenerator
from fullrmc.Constraints.AtomicCoordinationConstraints import AtomicCoordinationNumberConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint


##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
##############################  XY TRANSLATION GENERATOR  ################################
class XYTranslationGenerator(TranslationGenerator):
    def transform_coordinates(self, coordinates, argument=None):
        # generate random vector and ensure it is not zero
        vector    = np.array(1-2*np.random.random(3), dtype=FLOAT_TYPE)
        vector[2] = 0
        norm      = np.linalg.norm(vector) 
        if norm == 0:
            while norm == 0:
                vector    = np.array(1-2*np.random.random(3), dtype=FLOAT_TYPE)
                vector[2] = 0
                norm      = np.linalg.norm(vector)  
        # normalize vector
        vector /= FLOAT_TYPE( norm )
        # compute baseVector
        baseVector = FLOAT_TYPE(vector*self.amplitude[0])
        # amplify vector
        maxAmp  = FLOAT_TYPE(self.amplitude[1]-self.amplitude[0])
        vector *= FLOAT_TYPE(generate_random_float()*maxAmp)
        vector += baseVector
        # translate and return
        return coordinates+vector
        
        
##########################################################################################
#####################################  CREATE ENGINE  ####################################
pdbPath = "system.pdb" 
ENGINE = Engine(path=None)
ENGINE.set_pdb(pdbPath)
# add constraints
ACN_CONSTRAINT = AtomicCoordinationNumberConstraint()
ENGINE.add_constraints([ACN_CONSTRAINT]) 
ACN_CONSTRAINT.set_coordination_number_definition( [ ('Al','Cl',1.5, 2.5, 2, 2),
                                                     ('Al','S', 2.5, 3.0, 2, 2)] )
# add inter-molecular distance constraint
EMD_CONSTRAINT = InterMolecularDistanceConstraint(defaultDistance=1.0)
ENGINE.add_constraints([EMD_CONSTRAINT]) 
# set TranslationGenerator move generators amplitude
ENGINE.set_groups([[idx] for idx, el in enumerate( ENGINE.allElements ) if el != 'Al'])
[g.set_move_generator(XYTranslationGenerator(g, amplitude=0.1)) for g in ENGINE.groups]


##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def run_normal(nsteps, xyzPath):
    ACN_CONSTRAINT.set_coordination_number_definition( [ ('Al','Cl',1.5, 2.5, 2, 2),
                                                         ('Al','S', 2.5, 3.0, 2, 2)] )
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps*2, xyzFrequency=1, xyzPath=xyzPath, restartPdb=None)


##########################################################################################
#####################################  RUN SIMULATION  ###################################
xyzPath ="trajectory.xyz"
if os.path.isfile(xyzPath): os.remove(xyzPath)
run_normal(10000,    xyzPath)


##########################################################################################
##################################  VISUALIZE SIMULATION  ################################
# create otherParams
otherParams = ["draw material Opaque"]
otherParams.append("draw color red")
otherParams.append( "graphics top cylinder {10 10 9.95} {10 10 10.05} radius 1.5 resolution 100 filled no" )
otherParams.append("draw color blue")
otherParams.append( "graphics top cylinder {10 10 9.95} {10 10 10.05} radius 2.5 resolution 100 filled no" )
otherParams.append("draw color green")
otherParams.append( "graphics top cylinder {10 10 9.95} {10 10 10.05} radius 3.0 resolution 100 filled no" )
otherParams.append("label add Bonds 0/0 0/1")
otherParams.append("label add Bonds 0/0 0/2")
otherParams.append("label add Bonds 0/0 0/3")
otherParams.append("label add Bonds 0/0 0/4")
otherParams.append("label textsize 1.25")
otherParams.append("label textthickness 2")
otherParams.append("color Labels Bonds black")
otherParams.append("axes location off")
otherParams.append("light 0 on")
otherParams.append("light 0 pos {20 20 0}")
otherParams.append("light 1 on")
otherParams.append("light 0 pos {-20 -20 0}")
otherParams.append("light 2 on")
otherParams.append("light 0 pos {-20 20 0}")
otherParams.append("light 3 on")
otherParams.append("light 0 pos {20 -20 0}")
otherParams.append("scale set 0.3")  
otherParams.append("display resize 800 800" )  

ENGINE.set_pdb(pdbPath)          
ENGINE.visualize( commands = ["trajectory.xyz"], 
                  boxWidth = 0, bgColor="white",
                  representationParams = 'VDW 0.1 100',
                  otherParams = otherParams)   
    
 





 
    
 






