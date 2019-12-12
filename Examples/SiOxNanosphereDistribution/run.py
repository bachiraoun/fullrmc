# standard library imports
from __future__ import print_function
import os, sys

# external libraries imports

# fullrmc imports
from fullrmc.Globals import FLOAT_TYPE, LOGGER
from fullrmc.Engine import Engine
from fullrmc.Core.Constraint import ExperimentalConstraint
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.AtomicCoordinationConstraints import AtomicCoordinationNumberConstraint
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.OrderedSelectors import DirectionalOrderSelector
from fullrmc.Core.Group import EmptyGroup
from fullrmc import MultiframeUtils

##########################################################################################
#####################################  CREATE ENGINE  ####################################
# dirname
try:
    DIR_PATH = os.path.dirname( os.path.realpath(__file__) )
except:
    DIR_PATH = ''


# files name
grFileName     = "SiOx.gr"
pdbFileName    = "SiOx.pdb"
engineFileName = "SiOx.rmc"
multiframe     = 'size_distribution'
numberOfFrames = 10
FRESH_START    = False
# engine variables
grExpPath      = os.path.join(DIR_PATH, grFileName)
pdbPath        = os.path.join(DIR_PATH, pdbFileName)
engineFilePath = os.path.join(DIR_PATH, engineFileName)


ENGINE = Engine(path=None)
if not ENGINE.is_engine(engineFilePath) or FRESH_START:
   # create engine
    ENGINE = Engine(path=engineFilePath, freshStart=True)
    ENGINE.set_pdb(pdbPath)
    # create and add pair distribution constraint to the engine
    PDF_CONSTRAINT = PairDistributionConstraint(experimentalData=grExpPath, weighting="atomicNumber")
    # shape function parameters
    params = {'rmin':0., 'rmax':None, 'dr':0.5,
              'qmin':0.0001, 'qmax':0.6, 'dq':0.005,
              'updateFreq':1000}
    PDF_CONSTRAINT.set_shape_function_parameters(params)
    ENGINE.add_constraints([PDF_CONSTRAINT])
    # Intermolecular constraint
    EMD_CONSTRAINT = InterMolecularDistanceConstraint()
    ENGINE.add_constraints([EMD_CONSTRAINT])
    EMD_CONSTRAINT.set_type_definition("element")
    EMD_CONSTRAINT.set_pairs_distance([('Si','Si',1.75), ('O','O',1.10), ('Si','O',1.30)])
    # coordination number constraint
    ACNC_CONSTRAINT = AtomicCoordinationNumberConstraint()
    ENGINE.add_constraints([ACNC_CONSTRAINT])
    ACNC_CONSTRAINT.set_coordination_number_definition( coordNumDef=[('Si','Si',1.8,2.8,3,6), ])
    # initialize constraints
    _ = ENGINE.initialize_used_constraints()
    # set number density
    ENGINE.set_number_density(.0125)
    # save engine
    ENGINE.save()
else:
    ENGINE = ENGINE.load(engineFilePath)
    # get constraints
    PDF_CONSTRAINT, EMD_CONSTRAINT, ACNC_CONSTRAINT = ENGINE.constraints



# add multiframe
if not ENGINE.is_frame(multiframe):
    ENGINE.add_frame({'name':multiframe, 'frames_name':numberOfFrames})
    for sf in ENGINE.frames[multiframe]['frames_name']:
        ENGINE.set_used_frame(os.path.join(multiframe, sf))
        ENGINE.set_pdb(os.path.join(DIR_PATH, 'multiframe_structure_%s.pdb'%sf))
        _usedConstraints, _constraints, _rigidConstraints = ENGINE.initialize_used_constraints(sortConstraints=True)
        if not len(_usedConstraints):
            LOGGER.warn("@%s No constraints are used. Configuration will be randomized"%ENGINE.usedFrame)
        # runtime initialize group selector
        ENGINE._Engine__groupSelector._runtime_initialize()
        # runtime initialize constraints
        _=[c._runtime_initialize() for c in _usedConstraints]
        LOGGER.info("@%s Stochastic engine files are created"%ENGINE.usedFrame)


################################################################################
###################### CREATE SOFTGRID WORKERS MANAGEMENT ######################
WM = MultiframeUtils.WorkersManagement()
WM.start(engine=ENGINE, multiframe='size_distribution', orchestrator=None)


################################################################################
######################## OPTIMIZE OXYGEN ATOMS POSITION ########################
for sf in ENGINE.frames[multiframe]['frames_name']:
    fname = os.path.join(multiframe, sf)
    ENGINE.set_used_frame(fname)
    # set groups as oxygen atoms only
    elements = ENGINE.get_original_data('allElements')
    groups   = [[idx] for idx, el in enumerate(elements) if el=='O']
    ENGINE.set_groups(groups)
    LOGGER.info("@%s Setting oxygen atoms groups (%i)"%(fname, len(groups),))

# run all frames independantly to optimize and fix oxygen atom positions
WM.run_independant(nCycle=10, numberOfSteps=10000, saveFrequency=1, cycleTimeout=3600)




################################################################################
######################### OPTIMIZE ALL ATOMS POSITION ##########################
for sf in ENGINE.frames[multiframe]['frames_name']:
    fname = os.path.join(multiframe, sf)
    ENGINE.set_used_frame(fname)
    # set groups as oxygen atoms only
    ENGINE.set_groups_as_atoms()
    LOGGER.info("@%s Setting groups as atoms (%i)"%(fname, len(ENGINE.groups)))

# run all frames independantly to optimize and fix oxygen atom positions
WM.run_independant(nCycle=50, numberOfSteps=10000, saveFrequency=1, cycleTimeout=3600)


################################################################################
################## MOVE ALL ATOMS IN ALL FRAMES SYNCHRONOUSLY ##################
for sf in ENGINE.frames[multiframe]['frames_name']:
    fname = os.path.join(multiframe, sf)
    ENGINE.set_used_frame(fname)
    # set groups as all atoms
    ENGINE.set_groups_as_atoms()
    LOGGER.info("@%s Setting groups as atoms (%i)"%(fname, len(ENGINE.groups)))

# run all frames as a mixture system of all subframes
WM.run_dependant(nCycle=10000, firstNAccepted=1, numberOfSteps=100, subframesWeight=None, normalize=1, driftTolerance=1)
WM.run_dependant(nCycle=10000, firstNAccepted=1, numberOfSteps=100, subframesWeight=10,   normalize=1, driftTolerance=1)


################################################################################
################## MOVE ALL ATOMS IN ALL FRAMES SYNCHRONOUSLY ##################
RECUR = 200
for sf in ENGINE.frames[multiframe]['frames_name']:
    fname = os.path.join(multiframe, sf)
    ENGINE.set_used_frame(fname)
    # set groups as all atoms
    ENGINE.set_groups_as_atoms()
    # create recurring random selector to explore
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=RECUR, refine=False, explore=True)
    ENGINE.set_group_selector(gs)
    LOGGER.info("@%s Setting groups as atoms (%i)"%(fname, len(ENGINE.groups)))

# run all frames as a mixture system of all subframes
WM.run_dependant(nCycle=10000, firstNAccepted=1, numberOfSteps=RECUR, saveFrequency=10, cycleTimeout=900)



################################################################################
################################  CALL plot.py  ################################
os.system("%s %s"%(sys.executable, os.path.join(DIR_PATH, 'plot.py')))
