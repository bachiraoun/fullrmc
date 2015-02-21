"""
AngleConstraints contains classes for all constraints related angles between atoms.

.. inheritance-diagram:: fullrmc.Constraints.AngleConstraints
    :parts: 1
"""

# standard libraries imports
import itertools

# external libraries imports
import numpy as np
from timeit import default_timer as timer

# fullrmc imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, SingularConstraint, EnhanceOnlyConstraint
from fullrmc.Core.angles import full_angles

class BondsAngleConstraint(EnhanceOnlyConstraint, SingularConstraint):
    """
    Its controls the angle between 3 defined atoms.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. anglesMap (list): The angles map definition.
           Every item must be a list of five items.
           
            #. First item: The central atom index.
            #. Second item: The index of the left atom forming the angle (interchangeable with the right atom).
            #. Third item: The index of the right atom forming the angle (interchangeable with the left atom).
            #. Fourth item: The minimum lower limit or the minimum angle allowed in rad.
            #. Fifth item: The maximum upper limit or the maximum angle allowed in rad.
        #. rejectProbability (None, numpy.ndarray): rejection probability numpy.array.
           If None, rejectProbability will be automatically generated to 1 for all step where chiSquare increase.
    """
    def __init__(self, engine, anglesMap=None, rejectProbability=None):
        # initialize constraint
        EnhanceOnlyConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability)
        # set bonds map
        self.set_angles(anglesMap)
        
    @property
    def anglesMap(self):
        """ Get angles map."""
        return self.__anglesMap
    
    @property
    def angles(self):
        """ Get angles dictionary."""
        return self.__angles
    
    @property    
    def atomsLUAD(self):
        """ Get look up angles dictionary, connecting every atom's index to a central atom angles definition of angles attribute."""
        return self.__atomsLUAD
        
    @property
    def chiSquare(self):
        """ Get constraint's current chi square."""
        if self.data is None:
            return None
        else: 
            return self.compute_chi_square(data = self.data)
            
    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed","update boundary conditions",):
            self.__initialize_constraint__()        
        
    def should_step_get_rejected(self, chiSquare):
        """
        Overloads 'EnhanceOnlyConstraint' should_step_get_rejected method.
        It computes whether to accept or reject a move based on before and after move calculation and not chiSquare.
        If any of activeAtomsDataBeforeMove or activeAtomsDataAfterMove is None an Exception will get raised.
        
        :Parameters:
            #. chiSquare (number): not used in this case
        
        :Return:
            #. result (boolean): True to reject step, False to accept
        """
        if self.activeAtomsDataBeforeMove is None or self.activeAtomsDataAfterMove is None:
            raise Exception(log.LocalLogger("fullrmc").logger.error("must compute data before and after group move"))
        reject = False
        for index in self.activeAtomsDataBeforeMove.keys():
            before = self.activeAtomsDataBeforeMove[index]["reducedAngles"]
            after  = self.activeAtomsDataAfterMove[index]["reducedAngles"]
            if np.any((after-before)>PRECISION):
                reject = True
                break
        #print before, after, np.any(after>before), reject
        return reject
        
    def set_angles(self, anglesMap):
        """ 
        Sets the angles dictionary by parsing the anglesMap list.
        
        :Parameters:
            #. anglesMap (list): The angles map definition.
               Every item must be a list of five items.
               
               #. First item: The central atom index.
               #. Second item: The index of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The index of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed in rad.
               #. Fifth item: The maximum upper limit or the maximum angle allowed in rad.
        """
        map = []
        if self.engine is not None:
            if anglesMap is not None:
                assert isinstance(anglesMap, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("anglesMap must be None or a list")
                for angle in anglesMap:
                    assert isinstance(angle, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("anglesMap items must be lists")
                    angle = list(angle)
                    assert len(angle)==5, log.LocalLogger("fullrmc").logger.error("anglesMap items must be lists of 5 items each")
                    centralIdx, leftIdx, rightIdx, lower, upper = angle
                    assert is_integer(centralIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of first item must be an integer")
                    centralIdx = INT_TYPE(centralIdx)
                    assert is_integer(leftIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of second item must be an integer")
                    leftIdx = INT_TYPE(leftIdx)
                    assert is_integer(rightIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of third item must be an integer")
                    rightIdx = INT_TYPE(rightIdx)
                    assert centralIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists first item must be positive")
                    assert leftIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists second item must be positive")
                    assert rightIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists third item must be positive")
                    assert centralIdx!=leftIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists first and second items can't be the same")
                    assert centralIdx!=rightIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists first and third items can't be the same")
                    assert leftIdx!=rightIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists second and third items can't be the same")
                    assert is_number(lower), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of third item must be a number")
                    lower = FLOAT_TYPE(lower)
                    assert is_number(upper), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of fourth item must be a number")
                    upper = FLOAT_TYPE(upper)
                    assert lower>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fourth item must be positive")
                    assert upper>lower, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fourth item must be smaller than the fifth item")
                    assert upper<=PI, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be smaller or equal to %.10f"%PI)
                    map.append((centralIdx, leftIdx, rightIdx, lower, upper))  
        # set anglesMap definition
        self.__anglesMap = map     
        # create bonds list of indexes arrays
        self.__angles = {}
        self.__atomsLUAD = {}
        if self.engine is not None:
            # parse bondsMap
            for angle in self.__anglesMap:
                centralIdx, leftIdx, rightIdx, lower, upper = angle
                assert centralIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                assert leftIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                assert rightIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                # create atoms look up angles dictionary
                if not self.__atomsLUAD.has_key(centralIdx):
                    self.__atomsLUAD[centralIdx] = []
                if not self.__atomsLUAD.has_key(leftIdx):
                    self.__atomsLUAD[leftIdx] = []
                if not self.__atomsLUAD.has_key(rightIdx):
                    self.__atomsLUAD[rightIdx] = []
                # create angles
                if not self.__angles.has_key(centralIdx):
                    self.__angles[centralIdx] = {"leftIndexes":[],"rightIndexes":[],"lower":[],"upper":[]}
                # check for redundancy and append
                elif leftIdx in self.__angles[centralIdx]["leftIndexes"]:
                    index = self.__angles[centralIdx]["leftIndexes"].index(leftIdx)
                    if rightIdx == self.__angles[centralIdx]["rightIndexes"][index]:
                        log.LocalLogger("fullrmc").logger.warn("Angle definition for central atom index '%i' and interchangeable left an right '%i' and '%i' is  already defined. New angle limits [%.3f,%.3f] ignored and old angle limits [%.3f,%.3f] kept."%(centralIdx, leftIdx, rightIdx, lower, upper, self.__angles[centralIdx]["lower"][index], self.__angles[centralIdx]["upper"][index]))
                        continue
                elif leftIdx in self.__angles[centralIdx]["rightIndexes"]:
                    index = self.__angles[centralIdx]["rightIndexes"].index(leftIdx)
                    if rightIdx == self.__angles[centralIdx]["leftIndexes"][index]:
                        log.LocalLogger("fullrmc").logger.warn("Angle definition for central atom index '%i' and interchangeable left an right '%i' and '%i' is  already defined. New angle limits [%.3f,%.3f] ignored and old angle limits [%.3f,%.3f] kept."%(centralIdx, leftIdx, rightIdx, lower, upper, self.__angles[centralIdx]["lower"][index], self.__angles[centralIdx]["upper"][index]))
                        continue
                # add angle definition
                self.__angles[centralIdx]["leftIndexes"].append(leftIdx)
                self.__angles[centralIdx]["rightIndexes"].append(rightIdx)
                self.__angles[centralIdx]["lower"].append(lower)
                self.__angles[centralIdx]["upper"].append(upper)
                self.__atomsLUAD[centralIdx].append(centralIdx)
                self.__atomsLUAD[leftIdx].append(centralIdx)
                self.__atomsLUAD[rightIdx].append(centralIdx)
            # finalize angles
            for idx in self.engine.pdb.xindexes:
                angles = self.__angles.get(idx, {"leftIndexes":[],"rightIndexes":[],"lower":[],"upper":[]} )
                self.__angles[INT_TYPE(idx)] =  {"leftIndexes": np.array(angles["leftIndexes"], dtype = INT_TYPE), 
                                                 "rightIndexes": np.array(angles["rightIndexes"], dtype = INT_TYPE),
                                                 "lower"  : np.array(angles["lower"]  , dtype = FLOAT_TYPE),
                                                 "upper"  : np.array(angles["upper"]  , dtype = FLOAT_TYPE) }
                lut = self.__atomsLUAD.get(idx, [] )
                self.__atomsLUAD[INT_TYPE(idx)] = sorted(set(lut))
        # reset constraint
        self.__initialize_constraint__()
    
    def creates_angles_by_definition(self, anglesDefinition):
        """ 
        Creates anglesMap using angles definition.
        Calls set_angles(anglesMap) and generates angles attribute.
        
        :Parameters:
            #. anglesDefinition (dict): The angles definition. 
               Every key must be a molecule name (residue name in pdb file). 
               Every key value must be a list of angles definitions. 
               Every angle definition is a list of five items where:
               
               #. First item: The name of the central atom forming the angle.
               #. Second item: The name of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The name of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed in degrees.
               #. Fifth item: The maximum upper limit or the maximum angle allowed in degrees.
        
        ::
        
            e.g. (Carbon tetrachloride):  anglesDefinition={"CCL4": [('C','CL1','CL2' , 105, 115),
                                                                     ('C','CL2','CL3' , 105, 115),
                                                                     ('C','CL3','CL4' , 105, 115),                                      
                                                                     ('C','CL4','CL1' , 105, 115) ] }
                                                                 
        """
        if self.engine is None:
            raise Exception(log.LocalLogger("fullrmc").logger.error("Engine is not defined. Can't create angles"))
        assert isinstance(anglesDefinition, dict), log.LocalLogger("fullrmc").logger.error("anglesDefinition must be a dictionary")
        # check map definition
        existingMoleculesNames = sorted(set(self.engine.moleculesNames))
        anglesDef = {}
        for mol, angles in anglesDefinition.items():
            if mol not in existingMoleculesNames:
                log.LocalLogger("fullrmc").logger.warn("Molecule name '%s' in anglesDefinition is not recognized, angles definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(angles, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("mapDefinition molecule angles must be a list")
            angles = list(angles)
            molAnglesMap = []
            for angle in angles:
                assert isinstance(angle, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("mapDefinition angles must be a list")
                angle = list(angle)
                assert len(angle)==5
                centralAt, leftAt, rightAt, lower, upper = angle
                assert is_number(lower)
                lower = FLOAT_TYPE(lower)
                assert is_number(upper)
                upper = FLOAT_TYPE(upper)
                assert lower>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fourth item must be positive")
                assert upper>lower, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fourth item must be smaller than the fifth item")
                assert upper<=180, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be smaller or equal to 180")
                lower *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
                upper *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
                # check for redundancy
                append = True
                for b in molAnglesMap:
                    if (b[0]==centralAt) and ( (b[1]==leftAt and b[2]==rightAt) or (b[1]==rightAt and b[2]==leftAt) ):
                        log.LocalLogger("fullrmc").logger.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
                        append = False
                        break
                if append:
                    molAnglesMap.append((centralAt, leftAt, rightAt, lower, upper))
            # create bondDef for molecule mol 
            anglesDef[mol] = molAnglesMap
        # create mols dictionary
        mols = {}
        for idx in self.engine.pdb.xindexes:
            molName = self.engine.moleculesNames[idx]
            if not molName in anglesDef.keys():    
                continue
            molIdx = self.engine.moleculesIndexes[idx]
            if not mols.has_key(molIdx):
                mols[molIdx] = {"name":molName, "indexes":[], "names":[]}
            mols[molIdx]["indexes"].append(idx)
            mols[molIdx]["names"].append(self.engine.allNames[idx])
        # get anglesMap
        anglesMap = []         
        for val in mols.values():
            indexes = val["indexes"]
            names   = val["names"]
            # get definition for this molecule
            thisDef = anglesDef[val["name"]]
            for angle in thisDef:
                centralIdx = indexes[ names.index(angle[0]) ]
                leftIdx    = indexes[ names.index(angle[1]) ]
                rightIdx   = indexes[ names.index(angle[2]) ]
                lower      = angle[3]
                upper      = angle[4]
                anglesMap.append((centralIdx, leftIdx, rightIdx, lower, upper))
        # create angles
        self.set_angles(anglesMap=anglesMap)
    
    def compute_chi_square(self, data):
        """ 
        Computes the chi square of data not satisfying constraint conditions. 
        
        :Parameters:
            #. data (numpy.array): The constraint value data to compute chiSquare.
            
        :Returns:
            #. chiSquare (number): The calculated chiSquare multiplied by the contribution factor of the constraint.
        """
        chiSquare = 0
        for idx, angle in data.items():
            chiSquare +=  np.sum(angle["reducedAngles"]**2)
        FLOAT_TYPE( chiSquare )

    def get_constraint_value(self):
        """
        Computes all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        return self.data
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        # get angles dictionary slice
        anglesIndexes = []
        for idx in self.engine.pdb.indexes:
            anglesIndexes.extend( self.__atomsLUAD[idx] )
        anglesDict = {}
        for idx in set(anglesIndexes):
            anglesDict[idx] = self.__angles[idx] 
        # compute data before move
        dataDict = full_angles( anglesDict=anglesDict ,
                                boxCoords=self.engine.boxCoordinates,
                                basis=self.engine.basisVectors ,
                                reduceAngleToUpper = False,
                                reduceAngleToLower = False)
        self.set_data( dataDict )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set chiSquare
        #self.set_chi_square( self.compute_chi_square(data = self.__data) )
        
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
        """
        # get angles dictionary slice
        anglesIndexes = []
        for idx in indexes:
            anglesIndexes.extend( self.__atomsLUAD[idx] )
        anglesDict = {}
        for idx in set(anglesIndexes):
            anglesDict[idx] = self.angles[idx] 
        # compute data before move
        dataDict = full_angles( anglesDict=anglesDict ,
                                boxCoords=self.engine.boxCoordinates,
                                basis=self.engine.basisVectors ,
                                reduceAngleToUpper = False,
                                reduceAngleToLower = False)
        # set data before move
        self.set_active_atoms_data_before_move( dataDict )
        self.set_active_atoms_data_after_move(None)
        
    def compute_after_move(self, indexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        # get angles dictionary slice
        anglesIndexes = []
        for idx in indexes:
            anglesIndexes.extend( self.__atomsLUAD[idx] )
        anglesDict = {}
        for idx in set(anglesIndexes):
            anglesDict[idx] = self.__angles[idx] 
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[indexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[indexes] = movedBoxCoordinates
        # compute data before move
        dataDict = full_angles( anglesDict=anglesDict ,
                                boxCoords=self.engine.boxCoordinates,
                                basis=self.engine.basisVectors ,
                                reduceAngleToUpper = False,
                                reduceAngleToLower = False)
        # set data after move
        self.set_active_atoms_data_after_move( dataDict )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
  
    def accept_move(self, indexes):
        """ 
        Accept move.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
        """
        # get indexes
        anglesIndexes = []
        for idx in indexes:
            anglesIndexes.extend( self.__atomsLUAD[idx] )
        for idx in set(anglesIndexes):
            self.data[idx]["angles"]        = self.activeAtomsDataAfterMove[idx]["angles"]
            self.data[idx]["reducedAngles"] = self.activeAtomsDataAfterMove[idx]["reducedAngles"]
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)

    def reject_move(self, indexes):
        """ 
        Reject move.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)



    
    
            