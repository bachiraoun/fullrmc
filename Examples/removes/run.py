#  ####################################################################################  #
#  ########################### IMPORTING USEFUL DEFINITIONS ###########################  #
## standard library imports
import os

## external libraries imports
import numpy as np
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.Modify import set_records_attribute_values

## fullrmc imports
from fullrmc.Engine import Engine
from fullrmc.Core.Group import EmptyGroup
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.AtomicCoordinationConstraints import AtomicCoordinationNumberConstraint
from fullrmc.Generators.Translations import TranslationGenerator, TranslationAlongSymmetryAxisGenerator
from fullrmc.Generators.Swaps import SwapPositionsGenerator


#  ####################################################################################  #
#  ############################# DECLARE USEFUL VARIABLES #############################  #
experimentalDataPath = "pdf.exp"
structurePdbPath     = "system.pdb"
engineSavePath       = "system.rmc"
FRESH_START          = False

#  ####################################################################################  #
#  ################################### CREATE ENGINE ##################################  #
ENGINE = Engine(path=None)
if not ENGINE.is_engine(engineSavePath) or FRESH_START:
    ENGINE = Engine(path=engineSavePath, freshStart=True)
    ENGINE.set_pdb(structurePdbPath)
    ## create and add pair distribution constraint
    PDF_CONSTRAINT = PairDistributionConstraint(experimentalData=experimentalDataPath, weighting="atomicNumber")
    ENGINE.add_constraints([PDF_CONSTRAINT])
    ## create and add intermolecular distances constraint
    EMD_CONSTRAINT = InterMolecularDistanceConstraint()
    ENGINE.add_constraints([EMD_CONSTRAINT])
    EMD_CONSTRAINT.set_type_definition("element")
    EMD_CONSTRAINT.set_pairs_distance([('Co','Co',2.00),('Co','O' ,1.7),('Co','Li',2.00),('Co','Mn',2.00),('Co','Ni',2.00),
                                       ('Mn','Mn',2.00),('Mn','O' ,1.7),('Mn','Li',2.00),('Mn','Ni',2.00),
                                       ('Ni','Ni',2.00),('Ni','O' ,1.7),('Ni','Li',2.00),
                                       ('O' ,'O' ,1.20),('O' ,'Li',1.8),
                                       ('Li','Li',2.40),])
    ## create and add coordination number constraint
    ACNC_CONSTRAINT = AtomicCoordinationNumberConstraint(rejectProbability=0.7)
    ENGINE.add_constraints([ACNC_CONSTRAINT])
    ACNC_CONSTRAINT.set_coordination_number_definition( coordNumDef=[('Co','O',1.7,3.0,4,7), 
                                                                     ('Mn','O',1.7,3.0,4,7), 
                                                                     ('Ni','O',1.7,3.0,4,7)])
    ## save engine
    ENGINE.save()
else:
    ENGINE = ENGINE.load(engineSavePath)
    ## unpack constraints before fitting in case tweaking is needed
    PDF_CONSTRAINT, EMD_CONSTRAINT, ACNC_CONSTRAINT = ENGINE.constraints


#  ####################################################################################  #
#  ############################### DEFINE DIFFERENT RUNS ##############################  #
def normal_run(numberOfSteps=100000, saveFrequency=10000):
    ## reset groups as atoms
    ENGINE.set_groups_as_atoms()
    ## run engine
    ENGINE.run(numberOfSteps=numberOfSteps, saveFrequency=saveFrequency)
    
def swaps_run(numberOfSteps=100000, saveFrequency=10000):
    ## reset groups as atoms
    ENGINE.set_groups_as_atoms()
    ## build swap lists
    ALL_ELEMENTS = ENGINE.get_original_data('allElements')
    liSwaps = [[idx] for idx in xrange(len(ALL_ELEMENTS)) if ALL_ELEMENTS[idx]=='li' or ALL_ELEMENTS[idx]=='Li']
    meSwaps = [[idx] for idx in xrange(len(ALL_ELEMENTS)) if ALL_ELEMENTS[idx] in ('co','ni','mn') or ALL_ELEMENTS[idx] in ('Co','Ni','Mn')]
    ## set swap generators
    for g in ENGINE.groups:
        idx   = g.indexes[0]
        elIdx = ALL_ELEMENTS[idx]
        if elIdx in ('li','Li'):
            SPG=SwapPositionsGenerator(swapList=meSwaps)
            g.set_move_generator(SPG)
        elif elIdx in ('co','Co','mn','Mn','ni','Ni'):
            SPG=SwapPositionsGenerator(swapList=liSwaps)
            g.set_move_generator(SPG)
        else:
            g.set_move_generator( TranslationGenerator(amplitude=0.05) )
    ## run engine
    ENGINE.run(numberOfSteps=numberOfSteps, saveFrequency=saveFrequency)

def removes_run(numberOfSteps=100, saveFrequency=100):
    ## compute indexes lists
    ALL_ELEMENTS = ENGINE.get_original_data('allElements')
    oIndexes  = [idx for idx in xrange(len(ALL_ELEMENTS)) if ALL_ELEMENTS[idx]=='o'  or ALL_ELEMENTS[idx]=='O']
    liIndexes = [idx for idx in xrange(len(ALL_ELEMENTS)) if ALL_ELEMENTS[idx]=='li' or ALL_ELEMENTS[idx]=='Li']
    meIndexes = [idx for idx in xrange(len(ALL_ELEMENTS)) if ALL_ELEMENTS[idx] in ('co','ni','mn') or ALL_ELEMENTS[idx] in ('Co','Ni','Mn')]
    ## create empty group to remove oxygen. 
    ## By default EmptyGroup move generator is AtomsRemoveGenerator with its atomsList 
    ## None which means it will remove any atom from system.
    RO  = EmptyGroup()
    RO.moveGenerator.set_maximum_collected(20)
    RO.moveGenerator.set_atoms_list(oIndexes)
    ## create empty group to remove lithium
    RLi = EmptyGroup()
    RLi.moveGenerator.set_maximum_collected(500)
    RLi.moveGenerator.set_atoms_list(liIndexes)
    ## create empty group to remove cobalt
    RCo = EmptyGroup()
    RCo.moveGenerator.set_maximum_collected(500)
    RCo.moveGenerator.set_atoms_list(meIndexes)
    ## set groups to engine
    ENGINE.set_groups([RO,RLi,RCo])
    ## run engine
    ENGINE.run(numberOfSteps=numberOfSteps, saveFrequency=saveFrequency)
    
    
#  ####################################################################################  #
#  ################################## RUN SIMULATION ##################################  #
## run normal 
normal_run(numberOfSteps=100000, saveFrequency=50000)
PDF_CONSTRAINT.set_adjust_scale_factor((10,0.7,1.3)) 
normal_run(numberOfSteps=250000, saveFrequency=50000)

## add first swapping frame
if not ENGINE.is_frame('swap_1'):
    ENGINE.add_frame('swap_1')
## use swap_1 frame and run swapping
ENGINE.set_used_frame('swap_1')
swaps_run(numberOfSteps=200000, saveFrequency=50000)
normal_run(numberOfSteps=250000, saveFrequency=50000)

## add first removing frame
if not ENGINE.is_frame('removes_2'):
    ENGINE.add_frame('removes_2')
## use removes_2 frame and run some removes
ENGINE.set_used_frame('removes_2')
for _ in range(20):
    # remove as little as possible then try to refine
    removes_run(numberOfSteps=100, saveFrequency=100)
    normal_run(numberOfSteps=100000, saveFrequency=10000)
    swaps_run(numberOfSteps=50000, saveFrequency=10000)




