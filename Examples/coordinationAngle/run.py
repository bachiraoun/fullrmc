# standard libraries imports
import os

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint, AtomicCoordinationAngleConstraint
from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint
from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Selectors.OrderedSelectors import DefinedOrderSelector
from fullrmc.Generators.Translations import TranslationGenerator, TranslationAlongSymmetryAxisGenerator
from fullrmc.Generators.Rotations import RotationGenerator, RotationAboutSymmetryAxisGenerator
from fullrmc.Generators.Agitations import DistanceAgitationGenerator, AngleAgitationGenerator

# engine variables
expDataPath = "cos_pdf.exp"
pdbPath = "cos40pc_den1p9.pdb"
engineSavePath = "engine.rmc"
    
# check Engine already saved
if engineSavePath not in os.listdir("."):
    CONSTRUCT = True
else:
    CONSTRUCT = False
  
# construct and save engine or load engine
if CONSTRUCT or True:
    # initialize engine
    ENGINE = Engine(pdb=pdbPath, constraints=None)
    # create constraints
    PDF_CONSTRAINT = PairDistributionConstraint(engine=None, experimentalData=expDataPath, weighting="atomicNumber", adjustScaleFactor=(0, 0.8, 1.2))
    ACA_CONSTRAINT = AtomicCoordinationAngleConstraint(engine=None, typeDefinition='name')
    B_CONSTRAINT   = BondConstraint(engine=None)
    BA_CONSTRAINT  = BondsAngleConstraint(engine=None)
    IA_CONSTRAINT  = ImproperAngleConstraint(engine=None)
    # add constraints to engine
    ENGINE.add_constraints([PDF_CONSTRAINT,  ACA_CONSTRAINT, B_CONSTRAINT, BA_CONSTRAINT, IA_CONSTRAINT])
    # initialize constraints definitions
    #ACA_CONSTRAINT.set_coordination_angle_definition( {"S":  [ (('S',1.73, 2.40), ('S',1.73, 2.40), (110.,130.)), ] } )
    ACA_CONSTRAINT.set_coordination_angle_definition( {"S":  [ (('S',2.73, 4.40), ('S',2.73, 4.40), (110.,130.)), ] } )
    B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"COS": [('S1' ,'S2' , 1.73, 2.40),
                                                                      ('S2' ,'S3' , 1.73, 2.40),
                                                                      ('S3' ,'S4' , 1.73, 2.40),
                                                                      ('S4' ,'S5' , 1.73, 2.40),
                                                                      ('S5' ,'S6' , 1.73, 2.40),
                                                                      ('S6' ,'S7' , 1.73, 2.40),
                                                                      ('S7' ,'S8' , 1.73, 2.40),
                                                                      ('S8' ,'S1' , 1.73, 2.40)] })
    BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"COS": [ ('S1' ,'S2' ,'S8' , 96, 116),
                                                                          ('S2' ,'S1' ,'S3' , 96, 116),
                                                                          ('S3' ,'S2' ,'S4' , 96, 116),
                                                                          ('S4' ,'S3' ,'S5' , 96, 116),
                                                                          ('S5' ,'S4' ,'S6' , 96, 116),
                                                                          ('S6' ,'S5' ,'S7' , 96, 116),
                                                                          ('S7' ,'S6' ,'S8' , 96, 116),
                                                                          ('S8' ,'S1' ,'S7' , 96, 116)] })
    IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"COS": [ ('S1','S3','S5','S7', -15, 15),
                                                                          ('S2','S4','S6','S8', -15, 15)] })
    # initialize constraints data
    PDF_CONSTRAINT.set_used(True)
    ACA_CONSTRAINT.set_used(True)
    B_CONSTRAINT.set_used(True)
    BA_CONSTRAINT.set_used(True)
    IA_CONSTRAINT.set_used(True)
    #ENGINE.initialize_used_constraints()
    # save engine
    ENGINE.save(engineSavePath)
else:
    ENGINE = Engine(pdb=None).load(engineSavePath)
    


ACA_CONSTRAINT = ENGINE.constraints[1]
#print ACA_CONSTRAINT.coordAngleDefinition
ACA_CONSTRAINT.compute_data()
exit()



# ############ RUN ATOMS ############ #    
def all_atoms(ENGINE, rang=10, recur=10, exportPdb=False):
    ENGINE.set_groups_as_atoms()  
    # set selector
    ENGINE.set_group_selector(RandomSelector(ENGINE))
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'all atoms' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_all_atoms_%s.pdb"%(ENGINE.generated,R)) )    

# ############ RUN ATOMS ############ #    
def sulfur_atoms(ENGINE, rang=10, recur=100, refine=False, explore=True , exportPdb=False):
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)==1]
    ENGINE.set_groups(groups) 
    # set selector
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'sulfur atoms' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_sulfur_atoms_%s.pdb"%(ENGINE.generated,R)) )    

            
# ############ RUN ROTATION ABOUT SYMM AXIS 0 ############ #
def about0(ENGINE, rang=10, recur=50, refine=True, explore=False, exportPdb=False):  
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
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
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_about0_%s.pdb"%(ENGINE.generated,R)) )  
 
# ############ RUN ROTATION ABOUT SYMM AXIS 1 ############ #
def about1(ENGINE, rang=10, recur=50, refine=True, explore=False, exportPdb=False):  
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
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
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_about1_%s.pdb"%(ENGINE.generated,R)) )   
        
# ############ RUN ROTATION ABOUT SYMM AXIS 2 ############ #
def about2(ENGINE, rang=10, recur=50, refine=True, explore=False, exportPdb=False): 
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
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
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_about2_%s.pdb"%(ENGINE.generated,R)) )  

# ############ RUN TRANSLATION ALONG SYMM AXIS 0 ############ #
def along0(ENGINE, rang=10, recur=50, refine=True, explore=False, exportPdb=False):  
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(axis=0, amplitude=0.5)) for g in ENGINE.groups]
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
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_along0_%s.pdb"%(ENGINE.generated,R)) )   

# ############ RUN TRANSLATION ALONG SYMM AXIS 1 ############ #
def along1(ENGINE, rang=10, recur=50, refine=True, explore=False, exportPdb=False):  
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(axis=1, amplitude=0.5)) for g in ENGINE.groups]
    # set selector
    centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
    distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
    order     = np.argsort(distances)
    recur = recur
    gs = RecursiveGroupSelector(DefinedOrderSelector(ENGINE, order = order), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    # number of steps
    nsteps = recur*len(ENGINE.groups)
    for stepIdx in range(rang):
        LOGGER.info("Running 'along1' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_along1_%s.pdb"%(ENGINE.generated,R)) ) 

# ############ RUN TRANSLATION ALONG SYMM AXIS 2 ############ # 
def along2(ENGINE, rang=10, recur=50, refine=True, explore=False, exportPdb=False):   
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
    [g.set_move_generator(TranslationAlongSymmetryAxisGenerator(axis=2, amplitude=0.5)) for g in ENGINE.groups]
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
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_along2_%s.pdb"%(ENGINE.generated,R)) ) 

# ############ RUN MOLECULES ############ #
def molecules(ENGINE, rang=10, recur=5, refine=True, explore=True, exportPdb=False):
    ENGINE.set_groups_as_molecules()
    groups = [g for g in ENGINE.groups if len(g)>1]
    ENGINE.set_groups(groups)
    [g.set_move_generator( MoveGeneratorCollector(collection=[TranslationGenerator(amplitude=0.2),RotationGenerator(amplitude=2)],randomize=True) ) for g in ENGINE.groups]
    # number of steps
    nsteps = 20*len(ENGINE.groups)
    # set selector
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=refine, explore=explore)
    ENGINE.set_group_selector(gs)
    for stepIdx in range(rang):
        LOGGER.info("Running 'molecules' mode step %i"%(stepIdx))
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=nsteps, savePath=engineSavePath)
        R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
        if exportPdb:
            ENGINE.export_pdb( os.path.join("pdbFiles","%i_molecules_%s.pdb"%(ENGINE.generated,R)) )

# tweak constraints
PDF_CONSTRAINT = ENGINE.constraints[0]
#PDF_CONSTRAINT.set_scale_factor(1.4)
#print B_CONSTRAINT.bondsMap; exit()

# set window function
#windowFunction = np.sinc(0.3*np.arange(-5,5,0.1)).astype(np.float32)
#PDF_CONSTRAINT.set_window_function(windowFunction)
                                                                      

# ############ RUN ENGINE ############ #
R = "%.2f-%.2f"%(PDF_CONSTRAINT.shellsCenter[0], PDF_CONSTRAINT.shellsCenter[-1])
#ENGINE.export_pdb( os.path.join("pdbFiles","%i_original_%s.pdb"%(ENGINE.generated,R)) )
# run atoms
all_atoms(ENGINE)
sulfur_atoms(ENGINE)
# set adjust scale factor
PDF_CONSTRAINT.set_adjust_scale_factor((10, 0.8, 1.2))
# run atoms
all_atoms(ENGINE)
sulfur_atoms(ENGINE)
# run about
about0(ENGINE)
about1(ENGINE)
about2(ENGINE)
# run along
along0(ENGINE)
along1(ENGINE)
along2(ENGINE)
# run molecules
molecules(ENGINE)
# run atoms
all_atoms(ENGINE)
sulfur_atoms(ENGINE)

