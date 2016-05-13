##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# imports
import os
import numpy as np
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.CoordinationNumberConstraints import AtomicCoordinationNumberConstraint
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.OrderedSelectors import DirectionalOrderSelector

  
##########################################################################################
#####################################  CREATE ENGINE  ####################################
experimentalDataPath = "SiOx.gr"
structurePdbPath     = "SiOx.pdb"
engineSavePath       = "SiOx.rmc"

# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
if engineSavePath not in os.listdir(DIR_PATH):
    ## create and initialize engine
    ENGINE = Engine(pdb=structurePdbPath, constraints=None)
    # create and add pair distribution constraint to the engine
    PDF_CONSTRAINT = PairDistributionConstraint(engine=None, experimentalData=experimentalDataPath, weighting="atomicNumber")
    # shape function parameters
    params = {'rmin':0., 'rmax':None, 'dr':0.5,
              'qmin':0.0001, 'qmax':0.6, 'dq':0.005,
              'updateFreq':1000}
    PDF_CONSTRAINT.set_shape_function_parameters(params)
    ENGINE.add_constraints([PDF_CONSTRAINT])
    # Intermolecular constraint
    EMD_CONSTRAINT = InterMolecularDistanceConstraint(engine=None)
    ENGINE.add_constraints([EMD_CONSTRAINT])
    EMD_CONSTRAINT.set_type_definition("element")
    EMD_CONSTRAINT.set_pairs_distance([('Si','Si',1.75), ('o','o',1.10), ('Si','o',1.30)])
    # coordination number constraint
    #ACNC_CONSTRAINT = AtomicCoordinationNumberConstraint(engine=None)
    #ENGINE.add_constraints([ACNC_CONSTRAINT])
    #ACNC_CONSTRAINT.set_coordination_number_definition( coordNumDef={"Si": [('Si',1.8,2.6,3,5)]})
    # initialize constraints
    ENGINE.initialize_used_constraints()
    # set number density
    ENGINE.set_number_density(.0125)
    # save engine
    ENGINE.save(engineSavePath)
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)

# get constraints
PDF_CONSTRAINT, EMD_CONSTRAINT = ENGINE.constraints

##########################################################################################
#####################################  DIFFERENT RUNS  ###################################
def run_normal_rmc(nsteps=150000, saveFreq=10000):
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=saveFreq, savePath=engineSavePath)

def run_explore(recur=50, saveFreq=10000):
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=False, explore=True)
    ENGINE.set_group_selector(gs)
     # number of steps
    nsteps = recur*len(ENGINE.groups)
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        
def expand_nanoparticule(recur=50, explore=True, refine=False): 
    # create expansion selector and explore creating shell layer
    center = np.sum(ENGINE.realCoordinates, axis=0)/ENGINE.realCoordinates.shape[0]
    GS  = DirectionalOrderSelector(ENGINE, center=center, expand=True, adjustMoveGenerators=True)
    RGS = RecursiveGroupSelector(GS, recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(RGS)
    nsteps = ENGINE.numberOfAtoms*recur
    # run rmc
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps/10, savePath=engineSavePath)

##########################################################################################
#####################################  RUN SIMULATION  ###################################
# run normal rmc
run_normal_rmc(nsteps=100000)
run_explore()
run_normal_rmc(nsteps=100000)
run_explore()
# expand nanoparticule
expand_nanoparticule()
expand_nanoparticule()
expand_nanoparticule()
expand_nanoparticule()
    
##########################################################################################
##################################  PLOT PDF CONSTRAINT  #################################
PDF_CONSTRAINT.plot(intra=False)


