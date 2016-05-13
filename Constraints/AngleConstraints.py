"""
AngleConstraints contains classes for all constraints related angles between atoms.

.. inheritance-diagram:: fullrmc.Constraints.AngleConstraints
    :parts: 1
"""

# standard libraries imports
import itertools
import copy

# external libraries imports
import numpy as np
from timeit import default_timer as timer

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.angles import full_angles


class BondsAngleConstraint(RigidConstraint, SingularConstraint):
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
        #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
    
    .. code-block:: python
    
        ## Methane (CH4) molecule sketch
        ## 
        ##              H4
        ##              |
        ##              |
        ##           _- C -_
        ##        H1-  /    -_
        ##            /       H3
        ##           H2
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # create and add constraint
        BAC = BondsAngleConstraint(engine=None)
        ENGINE.add_constraints(BAC)
        
        # define intra-molecular angles 
        BAC.create_angles_by_definition( anglesDefinition={"CH4": [ ('C','H1','H2', 100, 120),
                                                                    ('C','H2','H3', 100, 120),
                                                                    ('C','H3','H4', 100, 120),
                                                                    ('C','H4','H1', 100, 120) ]} )
                                                                          
        
        
    """
    def __init__(self, engine, anglesMap=None, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability)
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
    def standardError(self):
        """ Get constraint's current standard error."""
        if self.data is None:
            return None
        else: 
            return self.compute_standard_error(data = self.data)
            
    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed","update boundary conditions",):
            self.reset_constraint()        
        
    def should_step_get_rejected(self, standardError):
        """
        Overloads 'RigidConstraint' should_step_get_rejected method.
        It computes whether to accept or reject a move based on before and after move calculation and not standardError.
        If any of activeAtomsDataBeforeMove or activeAtomsDataAfterMove is None an Exception will get raised.
        
        :Parameters:
            #. standardError (number): not used in this case
        
        :Return:
            #. result (boolean): True to reject step, False to accept
        """
        if self.activeAtomsDataBeforeMove is None or self.activeAtomsDataAfterMove is None:
            raise Exception(LOGGER.error("must compute data before and after group move"))
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
                assert isinstance(anglesMap, (list, set, tuple)), LOGGER.error("anglesMap must be None or a list")
                for angle in anglesMap:
                    assert isinstance(angle, (list, set, tuple)), LOGGER.error("anglesMap items must be lists")
                    angle = list(angle)
                    assert len(angle)==5, LOGGER.error("anglesMap items must be lists of 5 items each")
                    centralIdx, leftIdx, rightIdx, lower, upper = angle
                    assert is_integer(centralIdx), LOGGER.error("anglesMap items lists of first item must be an integer")
                    centralIdx = INT_TYPE(centralIdx)
                    assert is_integer(leftIdx), LOGGER.error("anglesMap items lists of second item must be an integer")
                    leftIdx = INT_TYPE(leftIdx)
                    assert is_integer(rightIdx), LOGGER.error("anglesMap items lists of third item must be an integer")
                    rightIdx = INT_TYPE(rightIdx)
                    assert centralIdx>=0, LOGGER.error("anglesMap items lists first item must be positive")
                    assert leftIdx>=0, LOGGER.error("anglesMap items lists second item must be positive")
                    assert rightIdx>=0, LOGGER.error("anglesMap items lists third item must be positive")
                    assert centralIdx!=leftIdx, LOGGER.error("bondsMap items lists first and second items can't be the same")
                    assert centralIdx!=rightIdx, LOGGER.error("bondsMap items lists first and third items can't be the same")
                    assert leftIdx!=rightIdx, LOGGER.error("bondsMap items lists second and third items can't be the same")
                    assert is_number(lower), LOGGER.error("anglesMap items lists of third item must be a number")
                    lower = FLOAT_TYPE(lower)
                    assert is_number(upper), LOGGER.error("anglesMap items lists of fourth item must be a number")
                    upper = FLOAT_TYPE(upper)
                    assert lower>=0, LOGGER.error("anglesMap items lists fourth item must be positive")
                    assert upper>lower, LOGGER.error("anglesMap items lists fourth item must be smaller than the fifth item")
                    assert upper<=PI, LOGGER.error("anglesMap items lists fifth item must be smaller or equal to %.10f"%PI)
                    map.append((centralIdx, leftIdx, rightIdx, lower, upper))  
        # set anglesMap definition
        self.__anglesMap = map     
        # create bonds list of indexes arrays
        self.__angles = {}
        self.__atomsLUAD = {}
        if self.engine is not None:
            # parse bondsMap
            for angle in self.__anglesMap:
                self.add_angle(angle)
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
        self.reset_constraint()
    
    def add_angle(self, angle):
        """
        Add a single angle to the list of constraint angles.
        
        :Parameters:
            #. angle (list): The bond list of five items.\n
               #. First item: The central atom index.
               #. Second item: The index of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The index of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed in rad.
               #. Fifth item: The maximum upper limit or the maximum angle allowed in rad.
        """
        centralIdx, leftIdx, rightIdx, lower, upper = angle
        assert centralIdx<len(self.engine.pdb), LOGGER.error("angle atom index must be smaller than maximum number of atoms")
        assert leftIdx<len(self.engine.pdb), LOGGER.error("angle atom index must be smaller than maximum number of atoms")
        assert rightIdx<len(self.engine.pdb), LOGGER.error("angle atom index must be smaller than maximum number of atoms")
        centralIdx = INT_TYPE(centralIdx)
        leftIdx    = INT_TYPE(leftIdx)
        rightIdx   = INT_TYPE(rightIdx)
        # create atoms look up angles dictionary
        if not self.__atomsLUAD.has_key(centralIdx):
            self.__atomsLUAD[centralIdx] = []
        if not self.__atomsLUAD.has_key(leftIdx):
            self.__atomsLUAD[leftIdx] = []
        if not self.__atomsLUAD.has_key(rightIdx):
            self.__atomsLUAD[rightIdx] = []
        # create angles
        if not self.__angles.has_key(centralIdx):
            centralIdxToArray = False
            self.__angles[centralIdx] = {"leftIndexes":[],"rightIndexes":[],"lower":[],"upper":[]}
        else:
            centralIdxToArray = not isinstance(self.__angles[centralIdx]["leftIndexes"], list)
            self.__angles[centralIdx] = {"leftIndexes"  :list(self.__angles[centralIdx]["leftIndexes"]),
                                         "rightIndexes" :list(self.__angles[centralIdx]["rightIndexes"]),
                                         "lower"        :list(self.__angles[centralIdx]["lower"]),
                                         "upper"        :list(self.__angles[centralIdx]["upper"]) }
        # check for redundancy and append
        ignoreFlag=False
        if leftIdx in self.__angles[centralIdx]["leftIndexes"]:
            index = self.__angles[centralIdx]["leftIndexes"].index(leftIdx)
            if rightIdx == self.__angles[centralIdx]["rightIndexes"][index]:
                LOGGER.warn("Angle definition for central atom index '%i' and interchangeable left an right '%i' and '%i' is  already defined. New angle limits [%.3f,%.3f] are ignored and old angle limits [%.3f,%.3f] are kept."%(centralIdx, leftIdx, rightIdx, lower, upper, self.__angles[centralIdx]["lower"][index], self.__angles[centralIdx]["upper"][index]))
                ignoreFlag=True
        elif leftIdx in self.__angles[centralIdx]["rightIndexes"]:
            index = self.__angles[centralIdx]["rightIndexes"].index(leftIdx)
            if rightIdx == self.__angles[centralIdx]["leftIndexes"][index]:
                LOGGER.warn("Angle definition for central atom index '%i' and interchangeable left an right '%i' and '%i' is  already defined. New angle limits [%.3f,%.3f] are ignored and old angle limits [%.3f,%.3f] are kept."%(centralIdx, leftIdx, rightIdx, lower, upper, self.__angles[centralIdx]["lower"][index], self.__angles[centralIdx]["upper"][index]))
                ignoreFlag=True
        # add angle definition
        if not ignoreFlag:
            self.__angles[centralIdx]["leftIndexes"].append(leftIdx)
            self.__angles[centralIdx]["rightIndexes"].append(rightIdx)
            self.__angles[centralIdx]["lower"].append(lower)
            self.__angles[centralIdx]["upper"].append(upper)
            self.__atomsLUAD[centralIdx].append(centralIdx)
            self.__atomsLUAD[leftIdx].append(centralIdx)
            self.__atomsLUAD[rightIdx].append(centralIdx)
        if centralIdxToArray:
            angles = self.__angles.get(centralIdxToArray, {"leftIndexes":[],"rightIndexes":[],"lower":[],"upper":[]} )
            self.__angles[centralIdxToArray] =  {"leftIndexes"  : np.array(angles["leftIndexes"], dtype = INT_TYPE), 
                                                 "rightIndexes" : np.array(angles["rightIndexes"], dtype = INT_TYPE),
                                                 "lower"        : np.array(angles["lower"]  , dtype = FLOAT_TYPE),
                                                 "upper"        : np.array(angles["upper"]  , dtype = FLOAT_TYPE) }    
        # sort lookup tables
        lut = self.__atomsLUAD.get(centralIdx, [] )
        self.__atomsLUAD[centralIdx] = sorted(set(lut))
        lut = self.__atomsLUAD.get(leftIdx, [] )
        self.__atomsLUAD[leftIdx] = sorted(set(lut))
        lut = self.__atomsLUAD.get(rightIdx, [] )
        self.__atomsLUAD[rightIdx] = sorted(set(lut))

    def create_angles_by_definition(self, anglesDefinition):
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
            raise Exception(LOGGER.error("Engine is not defined. Can't create angles"))
        assert isinstance(anglesDefinition, dict), LOGGER.error("anglesDefinition must be a dictionary")
        # check map definition
        existingMoleculesNames = sorted(set(self.engine.moleculesNames))
        anglesDef = {}
        for mol, angles in anglesDefinition.items():
            if mol not in existingMoleculesNames:
                LOGGER.warn("Molecule name '%s' in anglesDefinition is not recognized, angles definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(angles, (list, set, tuple)), LOGGER.error("mapDefinition molecule angles must be a list")
            angles = list(angles)
            molAnglesMap = []
            for angle in angles:
                assert isinstance(angle, (list, set, tuple)), LOGGER.error("mapDefinition angles must be a list")
                angle = list(angle)
                assert len(angle)==5
                centralAt, leftAt, rightAt, lower, upper = angle
                assert is_number(lower)
                lower = FLOAT_TYPE(lower)
                assert is_number(upper)
                upper = FLOAT_TYPE(upper)
                assert lower>=0, LOGGER.error("anglesMap items lists fourth item must be positive")
                assert upper>lower, LOGGER.error("anglesMap items lists fourth item must be smaller than the fifth item")
                assert upper<=180, LOGGER.error("anglesMap items lists fifth item must be smaller or equal to 180")
                lower *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
                upper *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
                # check for redundancy
                append = True
                for b in molAnglesMap:
                    if (b[0]==centralAt) and ( (b[1]==leftAt and b[2]==rightAt) or (b[1]==rightAt and b[2]==leftAt) ):
                        LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
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
    
    def compute_standard_error(self, data):
        """ 
        Compute the standard error (StdErr) of data not satisfying constraint conditions. 
        
        .. math::
            StdErr = \\sum \\limits_{i}^{C} 
            ( \\theta_{i} - \\theta_{i}^{min} ) ^{2} 
            \\int_{0}^{\\theta_{i}^{min}} \\delta(\\theta-\\theta_{i}) d \\theta
            +
            ( \\theta_{i} - \\theta_{i}^{max} ) ^{2} 
            \\int_{\\theta_{i}^{max}}^{\\pi} \\delta(\\theta-\\theta_{i}) d \\theta
                               
        Where:\n
        :math:`C` is the total number of defined angles constraints. \n
        :math:`\\theta_{i}^{min}` is the angle constraint lower limit set for constraint i. \n
        :math:`\\theta_{i}^{max}` is the angle constraint upper limit set for constraint i. \n
        :math:`\\theta_{i}` is the angle computed for constraint i. \n
        :math:`\\delta` is the Dirac delta function. \n
        :math:`\\int_{0}^{\\theta_{i}^{min}} \\delta(\\theta-\\theta_{i}) d \\theta` 
        is equal to 1 if :math:`0 \\leqslant \\theta_{i} \\leqslant \\theta_{i}^{min}` and 0 elsewhere.\n
        :math:`\\int_{\\theta_{i}^{max}}^{\\pi} \\delta(\\theta-\\theta_{i}) d \\theta` 
        is equal to 1 if :math:`\\theta_{i}^{max} \\leqslant \\theta_{i} \\leqslant \\pi` and 0 elsewhere.\n

        :Parameters:
            #. data (numpy.array): The constraint value data to compute standardError.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        standardError = 0
        for idx, angle in data.items():
            standardError +=  np.sum(angle["reducedAngles"]**2)
        return FLOAT_TYPE( standardError )

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
        dataDict = full_angles( anglesDict         = anglesDict ,
                                boxCoords          = self.engine.boxCoordinates,
                                basis              = self.engine.basisVectors ,
                                reduceAngleToUpper = False,
                                reduceAngleToLower = False)
        self.set_data( dataDict )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        #self.set_standard_error( self.compute_standard_error(data = dataDict) )
        
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
        dataDict = full_angles( anglesDict         = anglesDict ,
                                boxCoords          = self.engine.boxCoordinates,
                                basis              = self.engine.basisVectors ,
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
        dataDict = full_angles( anglesDict         = anglesDict ,
                                boxCoords          = self.engine.boxCoordinates,
                                basis              = self.engine.basisVectors ,
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



    
    
class AtomicCoordinationAngleConstraint(RigidConstraint, SingularConstraint):
    """
    It's a quasi-rigid constraint that controls the inter-molecular coordination angle between atoms. 
    The maximum standard error is defined as the sum of minimum number of neighbours of all atoms
    in the system.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. defaultDistance (number): The minimum distance allowed set by default for all atoms type.
        #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules about how to differentiate between atoms and how to parse pairsLimits.   
        #. coordAngleDef (None, dict): The coordination number definition. It must be a dictionary where keys are
               atoms type. Atoms type can be 'element' or 'name' as set by set_type_definition method.\n
               Every key value is the atom central type and every value is a list of definitions. 
               Every definition is a tuple or list of 3 items.\n
               #. tuple of 3 items (first neighbour atom type, lower limit coordination distance, upper limit coordination distance)
               #. tuple of 3 items (second neighbour atom type, lower limit coordination distance, upper limit coordination distance) 
               #. tuple of 2 items (lower limit coordination angle, upper limit coordination angle)
        #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
    """
    def __init__(self, engine, typeDefinition="element", coordAngleDef=None, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability)
        # set type definition
        self.__coordAngleDefinition = coordAngleDef
        self.set_type_definition(typeDefinition)

    @property
    def typeDefinition(self):
        """Get types definition."""    
        return self.__typeDefinition
        
    @property
    def coordAngleDefinition(self):
        """Get coordination number definition dictionary"""
        return self.__coordAngleDefinition
        
    def listen(self, message, argument=None):
        """   
        listen to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed", "update molecules indexes"):
            self.set_type_definition(typeDefinition=self.__typeDefinition, coordAngleDef = self.__coordAngleDefinition)
        elif message in("update boundary conditions",):
            self.reset_constraint()        
    
    def set_type_definition(self, typeDefinition, coordAngleDef=None):
        """
        Sets the atoms typing definition.
        
        :Parameters:
            #. typeDefinition (string): Can be either 'element' or 'name'. 
               It sets the rules about how to differentiate between atoms and how to parse pairsLimits.   
            #. coordAngleDef (None, dict): The coordination number definition. It must be a dictionary where keys are
               atoms type. Atoms type can be 'element' or 'name' as set by set_type_definition method.\n
               Every key value is the atom central type and every value is a list of definitions. 
               Every definition is a tuple or list of 3 items.\n
               #. tuple of 3 items (first neighbour atom type, lower limit coordination distance, upper limit coordination distance)
               #. tuple of 3 items (second neighbour atom type, lower limit coordination distance, upper limit coordination distance) 
               #. tuple of 2 items (lower limit coordination angle, upper limit coordination angle)
        """
        # set typeDefinition
        assert typeDefinition in ("name", "element"), LOGGER.error("typeDefinition must be either 'name' or 'element'")
        if self.engine is None:
            self.__types                = []
            self.__typesLUT             = {}
            self.__typesInverseLUT      = {}
            self.__allTypes             = []
            self.__numberOfTypes        = []
            self.__typesIndexes         = []
            self.__numberOfAtomsPerType = []
        elif typeDefinition == "name":
            self.__types                = self.engine.names
            self.__typesLUT             = dict(zip(self.__types,range(len(self.__types))))
            self.__typesInverseLUT      = {v: k for k, v in self.__typesLUT.items()}
            self.__allTypes             = self.engine.allNames
            self.__numberOfTypes        = self.engine.numberOfNames
            self.__typesIndexes         = self.engine.namesIndexes
            self.__numberOfAtomsPerType = self.engine.numberOfAtomsPerName  
        elif typeDefinition == "element":
            self.__types                = self.engine.elements
            self.__typesLUT             = dict(zip(self.__types,range(len(self.__types))))
            self.__typesInverseLUT      = {v: k for k, v in self.__typesLUT.items()}
            self.__allTypes             = self.engine.allElements
            self.__numberOfTypes        = self.engine.numberOfElements
            self.__typesIndexes         = self.engine.elementsIndexes
            self.__numberOfAtomsPerType = self.engine.numberOfAtomsPerElement
        # get type indexes LUT
        self.__typeIndexesLUT = dict([(idx,[]) for idx in self.__typesLUT.values()])
        for idx in range(len(self.__typesIndexes)):
            self.__typeIndexesLUT[self.__typesIndexes[idx]].append(idx)
        for k,v in self.__typeIndexesLUT.items():
            self.__typeIndexesLUT[k] = np.array(v, dtype=INT_TYPE)
        #for typeIdx in 
        self.__typeDefinition = typeDefinition
        # set coordination number definition
        if coordAngleDef is None:
            coordAngleDef = self.__coordAngleDefinition
        self.set_coordination_angle_definition(coordAngleDef)
        
    def set_coordination_angle_definition(self, coordAngleDef):
        """
        Set the coordination angle definition.

        :Parameters:
            #. coordAngleDef (None, dict): The coordination number definition. It must be a dictionary where keys are
               atoms type. Atoms type can be 'element' or 'name' as set by set_type_definition method.\n
               Every key value is the atom central type and every value is a list of definitions. 
               Every definition is a tuple or list of 3 items.\n
               #. tuple of 3 items (first neighbour atom type, lower limit coordination distance, upper limit coordination distance)
               #. tuple of 3 items (second neighbour atom type, lower limit coordination distance, upper limit coordination distance) 
               #. tuple of 2 items (lower limit coordination angle, upper limit coordination angle)
               
               ::

                   e.g. {"C":  [ (('C',1.4,1.6.), ('C',1.4,1.6.), (100.,120.)), ...], "O": ... }      
                   
        """
        if self.engine is None:
            self.__coordAngleDefinition = coordAngleDef
            self.__coordAngData         = None
            return
        elif coordAngleDef is None:
            coordAngleDef = {}
        else:
            assert isinstance(coordAngleDef, dict), LOGGER.error("coordAngleDef must be a dictionary")
            for key, caValues in coordAngleDef.items():
                keyValues = []
                assert key in self.__types, LOGGER.error("coordAngleDef key '%s' is not a valid type %s."%(key, self.__types))
                assert isinstance(caValues, (list, set, tuple)), LOGGER.error("Coordination angle key '%s' definition value must be a list."%key)
                for caIt in caValues:
                    assert isinstance(caIt, (list, set, tuple)), LOGGER.error("Coordination angle key '%s' definition values must be a list of lists."%key)    
                    caIt = list(caIt)
                    assert len(caIt)==3, LOGGER.error("Coordination angle key '%s' definition values must be a list of lists of length 3 each."%key)    
                    first, second, angLim = caIt
                    # check first coordination neighbour
                    assert isinstance(first, (list, set, tuple)), LOGGER.error("Coordination angle key '%s'. Definition first value must be a list"%key)     
                    first = list(first)
                    assert len(first)==3, LOGGER.error("Coordination angle key '%s'. Definition first value must be a list of three items"%key)
                    assert first[0] in self.__types, LOGGER.error("Coordination angle key '%s'. Definition first value must be a list which first item must be a valid type %s"%(key, self.__types))
                    assert is_number(first[1]), LOGGER.error("Coordination angle key '%s'. Definition first value must be a list which second item must be a number"%(key))       
                    first[1] = FLOAT_TYPE(first[1])
                    assert first[1]>0, LOGGER.error("Coordination angle key '%s'. Definition first value must be a list which second item must be a positive number"%(key))  
                    assert is_number(first[2]), LOGGER.error("Coordination angle key '%s'. Definition first value must be a list which third item must be a number"%(key))       
                    first[2] = FLOAT_TYPE(first[2])
                    assert first[2]>=first[1], LOGGER.error("Coordination angle key '%s'. Definition first value must be a list which third item must be a number bigger than the second item"%(key))       
                    # check second coordination neighbour
                    assert isinstance(second, (list, set, tuple)), LOGGER.error("Coordination angle key '%s'. Definition second value must be a list"%key)     
                    second = list(second)
                    assert len(second)==3, LOGGER.error("Coordination angle key '%s'. Definition second value must be a list of three items"%key)
                    assert second[0] in self.__types, LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which first item must be a valid type %s"%(key, self.__types))
                    assert is_number(second[1]), LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which second item must be a number"%(key))       
                    second[1] = FLOAT_TYPE(second[1])
                    assert second[1]>0, LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which second item must be a positive number"%(key))  
                    assert is_number(second[2]), LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which third item must be a number"%(key))       
                    second[2] = FLOAT_TYPE(second[2])
                    assert second[2]>=second[1], LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which third item must be a number bigger than the second item"%(key))       
                    # check angles limit
                    assert isinstance(angLim, (list, set, tuple)), LOGGER.error("Coordination angle key '%s'. Definition third value must be a list"%key)     
                    angLim = list(angLim)
                    assert is_number(angLim[0]), LOGGER.error("Coordination angle key '%s'. Definition third value must be a list which first item must be a number"%(key))       
                    angLim[0] = FLOAT_TYPE(angLim[0])
                    assert angLim[0]>=0, LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which first item must be a positive number"%(key))  
                    assert is_number(angLim[1]), LOGGER.error("Coordination angle key '%s'. Definition third value must be a list which second item must be a number"%(key))       
                    angLim[1] = FLOAT_TYPE(angLim[1])
                    assert angLim[0]<=angLim[1], LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which first item must be smaller than the second one"%(key))  
                    assert angLim[1]<=180, LOGGER.error("Coordination angle key '%s'. Definition second value must be a list which second item must be smaller than 180"%(key))  
                    # append definition
                    if (first, second, angLim) in keyValues:
                        LOGGER.warn("Coordination angle key '%s'. Definition %s is redundant."%(key, (first, second, angLim)))
                    else:
                        keyValues.append( (first, second, angLim) )
                # update definition 
                coordAngleDef[key] = keyValues
        # set coordination angle dictionary definition
        self.__coordAngleDefinition = coordAngleDef
        # set coordinationNumberPerType
        typeData                     = {}
        typeData['firstTypeIdx']     = []
        typeData['firstLowerLimit']  = []
        typeData['firstUpperLimit']  = []
        typeData['secondTypeIdx']    = []
        typeData['secondLowerLimit'] = []
        typeData['secondUpperLimit'] = []
        typeData['angleLowerLimit']  = []
        typeData['angleUpperLimit']  = []
        self.__typesCoordAngDef      = {}
        # initialize typesCoordNumDef
        for typeName in self.__types:
            typeIdx  = self.__typesLUT[typeName] 
            self.__typesCoordAngDef[typeIdx] = copy.deepcopy(typeData)
        for typeName, caValues in self.__coordAngleDefinition.items():
            typeIdx  = self.__typesLUT[typeName] 
            data = self.__typesCoordAngDef[typeIdx]
            for first, second, angLim in caValues:
                typeData['firstTypeIdx'].append( self.__typesLUT[first[0]] )
                typeData['firstLowerLimit'].append( first[1] )
                typeData['firstUpperLimit'].append( first[2] )
                typeData['secondTypeIdx'].append( self.__typesLUT[second[0]] )
                typeData['secondLowerLimit'].append( second[1] )
                typeData['secondUpperLimit'].append( second[2] )
                typeData['angleLowerLimit'].append( angLim[0] )
                typeData['angleUpperLimit'].append( angLim[1] )
            # set data 
            self.__typesCoordAngDef[typeIdx]['firstTypeIdx']     = np.array(data['firstTypeIdx'], dtype=INT_TYPE)
            self.__typesCoordAngDef[typeIdx]['firstLowerLimit']  = np.array(data['firstLowerLimit'], dtype=FLOAT_TYPE)
            self.__typesCoordAngDef[typeIdx]['firstUpperLimit']  = np.array(data['firstUpperLimit'], dtype=FLOAT_TYPE)
            self.__typesCoordAngDef[typeIdx]['secondTypeIdx']    = np.array(data['secondTypeIdx'], dtype=INT_TYPE)
            self.__typesCoordAngDef[typeIdx]['secondLowerLimit'] = np.array(data['secondLowerLimit'], dtype=FLOAT_TYPE)
            self.__typesCoordAngDef[typeIdx]['secondUpperLimit'] = np.array(data['secondUpperLimit'], dtype=FLOAT_TYPE)
            self.__typesCoordAngDef[typeIdx]['angleLowerLimit']  = np.array(data['angleLowerLimit'], dtype=FLOAT_TYPE)
            self.__typesCoordAngDef[typeIdx]['angleUpperLimit']  = np.array(data['angleUpperLimit'], dtype=FLOAT_TYPE)
        
        # set coordination number data
        self.__coordAngData = []
        maxSquaredDev = 0
        for idx in range(len(self.__allTypes)):
            typeName = self.__allTypes[idx]
            typeIdx  = self.__typesLUT[typeName]
            typeDef  = self.__typesCoordAngDef.get(typeName, None)
            data     = {}
            if not len(self.__typesCoordAngDef[typeIdx]['firstTypeIdx']):
                data['firstNeighs']  = {}
                data['secondNeighs'] = {}
                data['deviations']   = np.array([0], dtype=INT_TYPE)
            else:
                numberOfEntries = len(self.__typesCoordAngDef[typeIdx]['firstTypeIdx'])
                data['firstNeighs']  = {}
                data['secondNeighs'] = {}
                data['deviations']   = np.zeros((numberOfEntries,), dtype=INT_TYPE)
            data['neighbouring']      = {}
            data['standardError'] = np.sum(data['deviations']**2)
            #maxSquaredDev += data['standardError']
            self.__coordAngData.append(data)
        # set maximum squared deviation as the sum of all atoms atomDeviations
        #self._set_maximum_standard_error(maxSquaredDev)
        
     
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        self.__coordAngData = full_atomic_coordination_angle( boxCoords       = self.engine.boxCoordinates,
                                                              basis           = self.engine.basisVectors,
                                                              moleculeIndex   = self.engine.moleculesIndexes,
                                                              typesIndex      = self.__typesIndexes,
                                                              typesDefinition = self.__typesCoordAngDef,
                                                              typeIndexesLUT  = self.__typeIndexesLUT,
                                                              coordAngData    = self.__coordAngData)
        ## update data
        #self.set_data( self.__coordAngData )
        #self.set_active_atoms_data_before_move(None)
        #self.set_active_atoms_data_after_move(None)
        ## set standardError
        #SD = self.compute_standard_error(data = self.__coordAngData)
        #self.set_standard_error(SD) 




        