##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER, FLOAT_TYPE
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.PairCorrelationConstraints import PairCorrelationConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint
from fullrmc.Core.Collection import convert_Gr_to_gr
from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Selectors.OrderedSelectors import DefinedOrderSelector
from fullrmc.Generators.Translations import TranslationGenerator, TranslationAlongSymmetryAxisGenerator
from fullrmc.Generators.Rotations import RotationGenerator, RotationAboutSymmetryAxisGenerator
from fullrmc.Generators.Agitations import DistanceAgitationGenerator, AngleAgitationGenerator


##########################################################################################
#####################################  CREATE ENGINE  ####################################
# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )

# engine file names
engineFileName = "thf_engine.rmc"
expFileName    = "thf_pdf.exp"
pdbFileName    = "thf.pdb" 
freshStart     = False

# engine variables
expPath        = os.path.join(DIR_PATH, expFileName)
pdbPath        = os.path.join(DIR_PATH, pdbFileName)
engineFilePath = os.path.join(DIR_PATH, engineFileName)
ENGINE = Engine(path=None)
if freshStart or not ENGINE.is_engine(engineFilePath):
    # create engine
    ENGINE = Engine(path=engineFilePath, freshStart=True)
    ENGINE.set_pdb(pdbFileName)
    ## create experimental constraints
    #PDF_CONSTRAINT = PairDistributionConstraint(experimentalData=expPath, weighting="atomicNumber")
    _,_,_, gr = convert_Gr_to_gr(np.loadtxt(expPath), minIndex=[4,5,6])
    dataWeights = np.ones(gr.shape[0])
    dataWeights[:np.nonzero(gr[:,1]>0)[0][0]] = 0  
    PDF_CONSTRAINT = PairCorrelationConstraint(experimentalData=gr.astype(FLOAT_TYPE), weighting="atomicNumber", dataWeights=dataWeights)
    # create and define molecular constraints
    EMD_CONSTRAINT = InterMolecularDistanceConstraint(defaultDistance=1.5)
    B_CONSTRAINT   = BondConstraint()
    BA_CONSTRAINT  = BondsAngleConstraint()
    IA_CONSTRAINT  = ImproperAngleConstraint()
    # add constraints to engine
    ENGINE.add_constraints([PDF_CONSTRAINT, EMD_CONSTRAINT, B_CONSTRAINT, BA_CONSTRAINT, IA_CONSTRAINT])
    # initialize constraints definitions
    B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"THF": [('O' ,'C1' , 1.29, 1.70),
                                                                      ('O' ,'C4' , 1.29, 1.70),
                                                                      ('C1','C2' , 1.29, 1.70),
                                                                      ('C2','C3' , 1.29, 1.70),
                                                                      ('C3','C4' , 1.29, 1.70),
                                                                      ('C1','H11', 0.58, 1.15),('C1','H12', 0.58, 1.15),
                                                                      ('C2','H21', 0.58, 1.15),('C2','H22', 0.58, 1.15),
                                                                      ('C3','H31', 0.58, 1.15),('C3','H32', 0.58, 1.15),
                                                                      ('C4','H41', 0.58, 1.15),('C4','H42', 0.58, 1.15)] })
    BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"THF": [ ('O'  ,'C1' ,'C4' , 95 , 135),
                                                                          ('C1' ,'O'  ,'C2' , 95 , 135),
                                                                          ('C4' ,'O'  ,'C3' , 95 , 135),
                                                                          ('C2' ,'C1' ,'C3' , 90 , 120),
                                                                          ('C3' ,'C2' ,'C4' , 90 , 120),
                                                                          # H-C-H angle
                                                                          ('C1' ,'H11','H12', 95 , 125),
                                                                          ('C2' ,'H21','H22', 95 , 125),
                                                                          ('C3' ,'H31','H32', 95 , 125),
                                                                          ('C4' ,'H41','H42', 95 , 125),
                                                                          # H-C-O angle
                                                                          ('C1' ,'H11','O'  , 100, 120),
                                                                          ('C1' ,'H12','O'  , 100, 120),
                                                                          ('C4' ,'H41','O'  , 100, 120),
                                                                          ('C4' ,'H42','O'  , 100, 120),                                                                           
                                                                          # H-C-C
                                                                          ('C1' ,'H11','C2' , 80, 123),
                                                                          ('C1' ,'H12','C2' , 80, 123),
                                                                          ('C2' ,'H21','C1' , 80, 123),
                                                                          ('C2' ,'H21','C3' , 80, 123),
                                                                          ('C2' ,'H22','C1' , 80, 123),
                                                                          ('C2' ,'H22','C3' , 80, 123),
                                                                          ('C3' ,'H31','C2' , 80, 123),
                                                                          ('C3' ,'H31','C4' , 80, 123),
                                                                          ('C3' ,'H32','C2' , 80, 123),
                                                                          ('C3' ,'H32','C4' , 80, 123),
                                                                          ('C4' ,'H41','C3' , 80, 123),
                                                                          ('C4' ,'H42','C3' , 80, 123) ] })
    IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"THF": [ ('C2','O','C1','C4', -15, 15),
                                                                          ('C3','O','C1','C4', -15, 15) ] })
    # save engine
    ENGINE.save()
else:
    ENGINE = ENGINE.load(engineFilePath)
    PDF_CONSTRAINT, EMD_CONSTRAINT, B_CONSTRAINT, BA_CONSTRAINT, IA_CONSTRAINT = ENGINE.constraints
 
 
##########################################################################################
#####################################  DIFFERENT RUNS  ###################################    
# ############ RUN C-H BONDS ############ #
def bonds_CH(ENGINE, rang=10, recur=10, refine=False, explore=True): 
    groups = []
    for idx in range(0,ENGINE.pdb.numberOfAtoms, 13):
        groups.append( np.array([idx+1 ,idx+2 ], dtype=np.int32) ) # C1-H11
        groups.append( np.array([idx+1 ,idx+3 ], dtype=np.int32) ) # C1-H12
        groups.append( np.array([idx+4 ,idx+5 ], dtype=np.int32) ) # C2-H21
        groups.append( np.array([idx+4 ,idx+6 ], dtype=np.int32) ) # C2-H22
        groups.append( np.array([idx+7 ,idx+8 ], dtype=np.int32) ) # C3-H31
        groups.append( np.array([idx+7 ,idx+9 ], dtype=np.int32) ) # C3-H32
        groups.append( np.array([idx+10,idx+11], dtype=np.int32) ) # C4-H41
        groups.append( np.array([idx+10,idx+12], dtype=np.int32) ) # C4-H42
    ENGINE.set_groups(groups)
    [g.set_move_generator(DistanceAgitationGenerator(amplitude=0.2,agitate=(True,True))) for g in ENGINE.groups]
    # set selector
    if refine or explore:
        gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
        ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'bonds_CH' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN H-C-H ANGLES ############ #   
def angles_HCH(ENGINE, rang=5, recur=10, refine=False, explore=True):  
    groups = []
    for idx in range(0,ENGINE.pdb.numberOfAtoms, 13):
        groups.append( np.array([idx+1 ,idx+2, idx+3 ], dtype=np.int32) ) # H11-C1-H12
        groups.append( np.array([idx+4 ,idx+5, idx+6 ], dtype=np.int32) ) # H21-C2-H22
        groups.append( np.array([idx+7 ,idx+8, idx+9 ], dtype=np.int32) ) # H31-C3-H32
        groups.append( np.array([idx+10,idx+11,idx+12], dtype=np.int32) ) # H41-C4-H42
    ENGINE.set_groups(groups)   
    [g.set_move_generator(AngleAgitationGenerator(amplitude=5)) for g in ENGINE.groups] 
    # set selector
    if refine or explore:
        gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
        ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'angles_HCH' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN ATOMS ############ #    
def atoms_type(ENGINE, type='C', rang=30, recur=20, refine=False, explore=True):
    allElements = ENGINE.allElements
    groups = []
    for idx, el in enumerate(allElements):
        if el == type:
            groups.append( [idx] )
    ENGINE.set_groups(groups)
    # set selector
    if refine or explore:
        gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
        ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'atoms' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
              
# ############ RUN ATOMS ############ #    
def atoms(ENGINE, rang=30, recur=20, refine=False, explore=True):
    ENGINE.set_groups_as_atoms()  
    # set selector
    if refine or explore:
        gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
        ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'atoms' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN ROTATION ABOUT SYMM AXIS 0 ############ #
def about0(ENGINE, rang=5, recur=100, refine=True, explore=False):  
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator(RotationAboutSymmetryAxisGenerator(axis=0, amplitude=180)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order ), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'about0' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN ROTATION ABOUT SYMM AXIS 1 ############ #
def about1(ENGINE, rang=5, recur=10, refine=True, explore=False):  
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator(RotationAboutSymmetryAxisGenerator(axis=1, amplitude=180)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order ), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'about1' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
         
# ############ RUN ROTATION ABOUT SYMM AXIS 2 ############ #
def about2(ENGINE, rang=5, recur=100, refine=True, explore=False): 
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator(RotationAboutSymmetryAxisGenerator(axis=2, amplitude=180)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order ), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'about2' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN TRANSLATION ALONG SYMM AXIS 0 ############ #
def along0(ENGINE, rang=5, recur=100, refine=False, explore=True):  
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(axis=0, amplitude=0.1)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order ), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'along0' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN TRANSLATION ALONG SYMM AXIS 1 ############ #
def along1(ENGINE, rang=5, recur=100, refine=False, explore=True):  
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(axis=1, amplitude=0.1)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    recur = 200
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'along1' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN TRANSLATION ALONG SYMM AXIS 2 ############ # 
def along2(ENGINE, rang=5, recur=100, refine=False, explore=True):   
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(axis=2, amplitude=0.1)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order ), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'along2' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        
# ############ RUN MOLECULES ############ #
def molecules(ENGINE, rang=5, recur=100, refine=False, explore=True):
    ENGINE.set_groups_as_molecules()
    [g.set_move_generator( MoveGeneratorCollector(collection=[TranslationGenerator(amplitude=0.2),RotationGenerator(amplitude=2)],randomize=True) ) for g in ENGINE.groups]
    # number of steps
    nsteps = 20*len(ENGINE.groups)
    # set selector
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    for stepIdx in range(rang):
        LOGGER.info("Running 'molecules' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)

# ############ SHRINK SYSTEM ############ #   
def shrink(ENGINE, newDim):
    ENGINE.set_groups_as_molecules()  
    [g.set_move_generator( MoveGeneratorCollector(collection=[TranslationGenerator(amplitude=0.2),RotationGenerator(amplitude=5)],randomize=True) ) for g in ENGINE.groups]
    # get groups order    
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    # change boundary conditions
    bcFrom = str([list(bc) for bc in ENGINE.boundaryConditions.get_vectors()] )
    ENGINE.set_boundary_conditions(newDim)
    bcTo   = str([list(bc) for bc in ENGINE.boundaryConditions.get_vectors()] )
    LOGGER.info("boundary conditions changed from %s to %s"%(bcFrom,bcTo))
    # set selector
    recur = 200
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order ), recur=recur, refine=True)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(10):
        LOGGER.info("Running 'shrink' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps)
        fname = "shrink_"+str(newDim).replace(".","p")

##########################################################################################
#####################################  RUN SIMULATION  ###################################
atoms(ENGINE, explore=True, refine=False)
atoms(ENGINE, explore=True, refine=False)
# set short limits
PDF_CONSTRAINT.set_limits((None,5))
# fit bonds
bonds_CH(ENGINE)
# fit angles
angles_HCH(ENGINE)
# reset limits to normal
PDF_CONSTRAINT.set_limits((None,None))
# fit atoms
atoms(ENGINE, explore=False, refine=False)
atoms_type(ENGINE, explore=False, refine=False)
# fit molecules
molecules(ENGINE)

# refine scaling factor
atoms(ENGINE, explore=True, refine=False)
atoms_type(ENGINE, explore=True, refine=False)
PDF_CONSTRAINT.set_limits((None,None))
atoms(ENGINE, explore=False, refine=False)
about0(ENGINE)
about1(ENGINE)
about2(ENGINE)
along0(ENGINE)
along1(ENGINE)
along2(ENGINE)
molecules(ENGINE)
atoms(ENGINE, explore=True, refine=False)
# allow scale adjustment
PDF_CONSTRAINT.set_adjust_scale_factor((10, 0.8, 1.2)) 
about0(ENGINE)
about1(ENGINE)
about2(ENGINE)
along0(ENGINE)
along1(ENGINE)
along2(ENGINE)
molecules(ENGINE)
atoms(ENGINE, explore=True, refine=False)


##########################################################################################
####################################  PLOT CONSTRAINTS  ##################################
PDF_CONSTRAINT.plot(show=False)
EMD_CONSTRAINT.plot(show=False)
B_CONSTRAINT.plot(lineWidth=2,  nbins=20,  split='element', show=False)
BA_CONSTRAINT.plot(lineWidth=2, nbins=20,  split='element', show=False)
IA_CONSTRAINT.plot(lineWidth=2, nbins=20,  split='element', show=True)



