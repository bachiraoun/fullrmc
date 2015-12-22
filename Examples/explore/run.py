# standard libraries imports
import time
import os
import itertools

# external libraries imports
import numpy as np
from pdbParser.pdbParser import pdbParser

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Generators.Translations import TranslationGenerator
from fullrmc.Generators.Rotations import RotationGenerator
from fullrmc.Core.Collection import generate_random_float
from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
from fullrmc.Core.GroupSelector import RecursiveGroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector
from fullrmc.Core.transform_coordinates import transform_coordinates


class BiasedEngine(Engine):
    """
    This is a biased engine that fakes the computation of biasedChiSquare
    just for the purpose of this example
    """
    def __init__(self, improveProbability=0.01, initialChiSquare=1000000, *args, **kwargs):
        super(BiasedEngine, self).__init__(*args, **kwargs)
        self.__improveProbability = improveProbability
        self.biasedChiSquare = initialChiSquare
    
    def compute_chi_square(self, constraints, current=True):
        if generate_random_float() <= self.__improveProbability:
            newChiSquare = self.biasedChiSquare-generate_random_float()*1e-3
            if newChiSquare<=0:
                raise Exception("biasedChiSquare reached 0. restart using bigger initialChiSquare")
        else:
            newChiSquare = self.biasedChiSquare+generate_random_float()*1e-3    
        return newChiSquare
    
    def run(self, numberOfSteps=100000, saveFrequency=1000, savePath="restart", 
                  xyzFrequency=None, xyzPath="trajectory.xyz"):
        """
        This is an exact copy of engine run method with slight changes marked with #-->
        to make two trajectories, one of the real system and another the explored space.
        new code is marked with #<--
        all leading variables double scores __ removed.
        """
        # get arguments
        #-->_numberOfSteps            = self.__runtime_get_number_of_steps(numberOfSteps)
        #-->_saveFrequency, _savePath = self.__runtime_get_save_engine(saveFrequency, savePath)
        #-->_xyzFrequency, _xyzPath   = self.__runtime_get_save_xyz(xyzFrequency, xyzPath)
        _numberOfSteps = numberOfSteps #<--
        _saveFrequency = 2*numberOfSteps #<--
        _savePath = savePath  #<--
        # create xyz file
        #-->if _xyzFrequency is not None:
        #-->    _xyzfd = open(_xyzPath, 'a')
        _xyzfd = open("trajectory.xyz", 'a')#<--
        # get and initialize used constraints
        _usedConstraints, _constraints, _enhanceOnlyConstraints = self.initialize_used_constraints()
        if not len(_usedConstraints):
            LOGGER.warn("No constraints are used. Configuration will be randomize")
        # compute biasedChiSquare
        self.biasedChiSquare = self.compute_chi_square(_constraints, current=True)
        # initialize useful arguments
        _engineStartTime    = time.time()
        _lastSavedChiSquare = self.biasedChiSquare
        _coordsBeforeMove   = None
        _moveTried          = False
        # initialize group selector
        self.groupSelector._runtime_initialize()
        
        self.__realCoords = self.realCoordinates  #<--
        self.__boxCoords  = self.boxCoordinates  #<--
        #   #####################################################################################   #
        #   #################################### RUN ENGINE #####################################   #
        LOGGER.info("Engine started %i steps, biasedChiSquare is: %.6f"%(_numberOfSteps, self.biasedChiSquare) )
        self.__generated = 0 #<--
        self.__tried = 0     #<--
        self.__accepted=0    #<--
        for step in xrange(_numberOfSteps):
            # increment generated
            self.__generated += 1
            # get group
            self.__lastSelectedGroupIndex = self.groupSelector.select_index()
            group = self.groups[self.__lastSelectedGroupIndex]
            # get atoms indexes
            groupAtomsIndexes = group.indexes
            # get move generator
            groupMoveGenerator = group.moveGenerator
            # get group atoms coordinates before applying move 
            if _coordsBeforeMove is None or not self.groupSelector.isRecurring:
                _coordsBeforeMove = np.array(self.realCoordinates[groupAtomsIndexes], dtype=self.realCoordinates.dtype)
            elif self.groupSelector.explore:
                if _moveTried:
                    _coordsBeforeMove = movedRealCoordinates
            elif not self.groupSelector.refine:
                _coordsBeforeMove = np.array(self.realCoordinates[groupAtomsIndexes], dtype=self.realCoordinates.dtype)
            # compute moved coordinates
            movedRealCoordinates = groupMoveGenerator.move(_coordsBeforeMove)
            movedBoxCoordinates  = transform_coordinates(transMatrix=self.reciprocalBasisVectors , coords=movedRealCoordinates)
            ########################### compute enhanceOnlyConstraints ############################
            rejectMove = False
            for c in _enhanceOnlyConstraints:
                # compute before move
                c.compute_before_move(indexes = groupAtomsIndexes)
                # compute after move
                c.compute_after_move(indexes = groupAtomsIndexes, movedBoxCoordinates=movedBoxCoordinates)
                # get rejectMove
                rejectMove = c.should_step_get_rejected(c.afterMoveChiSquare)
                if rejectMove:
                    break
            _moveTried = not rejectMove
            ############################## reject move before trying ##############################
            if rejectMove:
                # enhanceOnlyConstraints reject move
                for c in _enhanceOnlyConstraints:
                    c.reject_move(indexes=groupAtomsIndexes)
                # log generated move rejected before getting tried
                LOGGER.log("move not tried","Generated move %i is not tried"%self.tried)
            ###################################### try move #######################################
            else:
                self.__tried += 1
                for c in _constraints:
                    # compute before move
                    c.compute_before_move(indexes = groupAtomsIndexes)
                    # compute after move
                    c.compute_after_move(indexes = groupAtomsIndexes, movedBoxCoordinates=movedBoxCoordinates)
            ################################ compute new biasedChiSquare ################################
                newChiSquare = self.compute_chi_square(_constraints, current=False)
                #if len(_constraints) and (newChiSquare >= self.biasedChiSquare):
                if newChiSquare > self.biasedChiSquare:
                    if generate_random_float() > self.tolerance:
                        rejectMove = True
                    else:
                        self.tolerated += 1
                        self.biasedChiSquare  = newChiSquare
                else:
                    self.biasedChiSquare = newChiSquare
            ################################## reject tried move ##################################
            if rejectMove:
                # set selector move rejected
                self.groupSelector.move_rejected(self.__lastSelectedGroupIndex)
                if _moveTried:
                    # constraints reject move
                    for c in _constraints:
                        c.reject_move(indexes=groupAtomsIndexes)
                    # log tried move rejected
                    LOGGER.log("move rejected","Tried move %i is rejected"%self.__generated)
            ##################################### accept move #####################################
            else:
                self.__accepted  += 1
                # set selector move accepted
                self.groupSelector.move_accepted(self.__lastSelectedGroupIndex)
                # constraints reject move
                for c in _usedConstraints:
                    c.accept_move(indexes=groupAtomsIndexes)
                # set new coordinates
                self.__realCoords[groupAtomsIndexes] = movedRealCoordinates
                self.__boxCoords[groupAtomsIndexes]  = movedBoxCoordinates
                # log new successful move
                triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
                acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
                LOGGER.log("move accepted","Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - biasedChiSquare:%.6f" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.biasedChiSquare))
            ##################################### save engine #####################################
            if _saveFrequency is not None:
                if not(step+1)%_saveFrequency:
                    if _lastSavedChiSquare==self.biasedChiSquare:
                        LOGGER.info("Save engine omitted because no improvement made since last save.")
                    else:
                        # update state
                        self.state  = time.time()
                        for c in _usedConstraints:
                           #c.increment_tried()
                           c.set_state(self.state)
                        # save engine
                        _lastSavedChiSquare = self.biasedChiSquare
                        self.save(_savePath)
            ############################### dump coords to xyz file ###############################
            #-->if _xyzFrequency is not None:
            #-->    if not(step+1)%_xyzFrequency:
            #-->        _xyzfd.write("%s\n"%self.__pdb.numberOfAtoms)
            #-->        triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
            #-->        acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
            #-->        _xyzfd.write("Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - biasedChiSquare:%.6f\n" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.biasedChiSquare))
            #-->        frame = [self.allNames[idx]+ " " + "%10.5f"%self.__realCoords[idx][0] + " %10.5f"%self.__realCoords[idx][1] + " %10.5f"%self.__realCoords[idx][2] + "\n" for idx in self.__pdb.xindexes]
            #-->        _xyzfd.write("".join(frame)) 
            triedRatio    = 100.*(float(self.__tried)/float(self.__generated)) #<--
            acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated)) #<--
            _xyzfd.write("%s\n"%(len(groupAtomsIndexes)*2) ) #<--
            _xyzfd.write("Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - biasedChiSquare:%.6f\n" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.biasedChiSquare)) #<--
            frame = [self.allNames[idx]+ " " + "%10.5f"%self.realCoordinates[idx][0] + " %10.5f"%self.realCoordinates[idx][1] + " %10.5f"%self.realCoordinates[idx][2] + "\n" for idx in groupAtomsIndexes] #<--
            frame.extend([self.allNames[idx]+ " " + "%10.5f"%_coordsBeforeMove[idx][0] + " %10.5f"%_coordsBeforeMove[idx][1] + " %10.5f"%_coordsBeforeMove[idx][2] + "\n" for idx in range(_coordsBeforeMove.shape[0])]) #<--
            _xyzfd.write("".join(frame)) #<--
        #   #####################################################################################   #
        #   ################################# FINISH ENGINE RUN #################################   #        
        #-->LOGGER.info("Engine finishes executing all '%i' steps in %s" % (_numberOfSteps, get_elapsed_time(_engineStartTime, format="%d(days) %d:%d:%d")))
        # close .xyz file
        #-->if _xyzFrequency is not None:
        #-->    _xyzfd.close()
        _xyzfd.close() #<--
  
# engine variables
pdbPath = "nagma_in_water.pdb"
engineSavePath = "thf_engine.rmc"

# initialize engine
ENGINE = BiasedEngine(pdb=pdbPath, constraints=None)
# create constraints
EMD_CONSTRAINT = InterMolecularDistanceConstraint(engine=None)
# add constraints to engine
ENGINE.add_constraints([EMD_CONSTRAINT])
# set thf molecule the only group
ENGINE.set_groups_as_molecules()
ENGINE.set_groups(ENGINE.groups[0])

# ############ RUN ############ #
def explore(ENGINE, nsteps=1000, rang=3):
    # set move generator
    [g.set_move_generator( MoveGeneratorCollector(collection=[TranslationGenerator(amplitude=0.5),RotationGenerator(amplitude=5)],randomize=True) ) for g in ENGINE.groups]
    # set selector
    recur = nsteps
    gs = RecursiveGroupSelector(RandomSelector(ENGINE), recur=recur, refine=False, explore=True)
    # run
    for step in range(rang):
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps)
        #ENGINE.export_pdb( os.path.join("pdbFiles","step_%i.pdb"%step) )    

        
# ############ RUN ENGINE ############ #
files = [f for f in os.listdir(".") if os.path.isfile(f) and ".xyz" in f]
# remove all .xyz trajectory files
[os.remove(fname) for fname in files]
explore(ENGINE, 1000)
ENGINE.visualize()

