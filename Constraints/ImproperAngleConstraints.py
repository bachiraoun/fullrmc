"""
ImproperAngleConstraints contains classes for all constraints related improper angles between atoms.

.. inheritance-diagram:: fullrmc.Constraints.ImproperAngleConstraints
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
from fullrmc.Core.improper_angles import full_improper_angles

class ImproperAngleConstraint(EnhanceOnlyConstraint, SingularConstraint):
    """
    Its controls the improper angle between 4 defined atoms. It is mainly used to keep atoms in the plane.
    The improper angle is the defined between a first improper atom and the plane formed of the three other atoms.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. anglesMap (list): The angles map definition.
               Every item must be a list of five items.
               
               #. First item: The improper atom index that must be in the plane.
               #. Second item: The index of the atom 'O' considered the origin of the plane.
               #. Third item: The index of the atom 'x' used to calculated 'Ox' vector.
               #. Fourth item: The index of the atom 'y' used to calculated 'Oy' vector.
               #. Fifth item: The minimum lower limit or the minimum angle allowed in rad.
               #. Sixth item: The maximum upper limit or the maximum angle allowed in rad.
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
        return self.self.__anglesMap
    
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
        return reject
        
    def set_angles(self, anglesMap):
        """ 
        Sets the angles dictionary by parsing the anglesMap list.
        
        :Parameters:
            #. anglesMap (list): The angles map definition.
               Every item must be a list of five items.\n
               #. First item: The improper atom index that must be in the plane.
               #. Second item: The index of the atom 'O' considered the origin of the plane.
               #. Third item: The index of the atom 'x' used to calculated 'Ox' vector.
               #. Fourth item: The index of the atom 'y' used to calculated 'Oy' vector.
               #. Fifth item: The minimum lower limit or the minimum angle allowed in rad.
               #. Sixth item: The maximum upper limit or the maximum angle allowed in rad.
        """
        map = []
        if self.engine is not None:
            if anglesMap is not None:
                assert isinstance(anglesMap, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("anglesMap must be None or a list")
                for angle in anglesMap:
                    assert isinstance(angle, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("anglesMap items must be lists")
                    angle = list(angle)
                    assert len(angle)==6, log.LocalLogger("fullrmc").logger.error("anglesMap items must be lists of 6 items each")
                    improperIdx, oIdx, xIdx, yIdx, lower, upper = angle
                    assert is_integer(improperIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of first item must be an integer")
                    improperIdx = INT_TYPE(improperIdx)
                    assert is_integer(oIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of second item must be an integer")
                    oIdx = INT_TYPE(oIdx)
                    assert is_integer(xIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of third item must be an integer")
                    xIdx = INT_TYPE(xIdx)
                    assert is_integer(yIdx), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of fourth item must be an integer")
                    yIdx = INT_TYPE(yIdx)
                    assert improperIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists first item must be positive")
                    assert oIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists second item must be positive")
                    assert xIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists third item must be positive")
                    assert yIdx>=0, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fourth item must be positive")
                    assert improperIdx!=oIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists first and second items can't be the same")
                    assert improperIdx!=xIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists first and third items can't be the same")
                    assert improperIdx!=yIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists first and fourth items can't be the same")
                    assert oIdx!=xIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists second and third items can't be the same")
                    assert oIdx!=yIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists second and fourth items can't be the same")
                    assert xIdx!=yIdx, log.LocalLogger("fullrmc").logger.error("bondsMap items lists third and fourth items can't be the same")
                    assert is_number(lower), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of fifth item must be a number")
                    lower = FLOAT_TYPE(lower)
                    assert is_number(upper), log.LocalLogger("fullrmc").logger.error("anglesMap items lists of sixth item must be a number")
                    upper = FLOAT_TYPE(upper)
                    assert lower>=FLOAT_TYPE(-PI/2), log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be bigger than %10f"%FLOAT_TYPE(-PI/2))
                    assert upper>lower, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be smaller than the sixth item")
                    assert upper<=FLOAT_TYPE(PI/2), log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be smaller or equal to %.10f"%FLOAT_TYPE(PI/2))
                    map.append((improperIdx, oIdx, xIdx, yIdx, lower, upper))  
        # set anglesMap definition
        self.__anglesMap = map      
        # create bonds list of indexes arrays
        self.__angles = {}
        self.__atomsLUAD = {}
        if self.engine is not None:
            # parse bondsMap
            for angle in self.__anglesMap:
                improperIdx, oIdx, xIdx, yIdx, lower, upper = angle
                assert improperIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                assert oIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                assert xIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                assert yIdx<len(self.engine.pdb), log.LocalLogger("fullrmc").logger.error("angle atom index must be smaller than maximum number of atoms")
                # create atoms look up angles dictionary
                if not self.__atomsLUAD.has_key(improperIdx):
                    self.__atomsLUAD[improperIdx] = []
                if not self.__atomsLUAD.has_key(oIdx):
                    self.__atomsLUAD[oIdx] = []
                if not self.__atomsLUAD.has_key(xIdx):
                    self.__atomsLUAD[xIdx] = []
                if not self.__atomsLUAD.has_key(yIdx):
                    self.__atomsLUAD[yIdx] = []
                # create improper angles
                if not self.__angles.has_key(improperIdx):
                    self.__angles[improperIdx] = {"oIndexes":[],"xIndexes":[],"yIndexes":[],"lower":[],"upper":[]}
                # check for redundancy and append
                elif oIdx in self.__angles[improperIdx]["oIndexes"]:
                    index = self.__angles[improperIdx]["oIndexes"].index(oIndexes)
                    if sorted(oIdx,xIdx,yIdx) == sorted(self.__angles[improperIdx]["oIndexes"][index],self.__angles[improperIdx]["xIndexes"][index],self.__angles[improperIdx]["yIndexes"][index]):
                        log.LocalLogger("fullrmc").logger.warn("Improper angle definition for improper atom index '%i' and (O,x,y) atoms indexes (%i,%i,%i)  already defined. New angle limits [%.3f,%.3f] ignored and old angle limits [%.3f,%.3f] kept."%(improperIdx, oIdx, xIdx, yIdx, lower, upper, self.__angles[improperIdx]["lower"][index], self.__angles[improperIdx]["upper"][index]))
                        continue
                # add improper angle definition
                self.__angles[improperIdx]["oIndexes"].append(oIdx)
                self.__angles[improperIdx]["xIndexes"].append(xIdx)
                self.__angles[improperIdx]["yIndexes"].append(yIdx)
                self.__angles[improperIdx]["lower"].append(lower)
                self.__angles[improperIdx]["upper"].append(upper)
                self.__atomsLUAD[improperIdx].append(improperIdx)
                self.__atomsLUAD[oIdx].append(improperIdx)
                self.__atomsLUAD[xIdx].append(improperIdx)
                self.__atomsLUAD[yIdx].append(improperIdx)
            # finalize improper angles
            for idx in self.engine.pdb.xindexes:
                angles = self.__angles.get(idx, {"oIndexes":[],"xIndexes":[],"yIndexes":[],"lower":[],"upper":[]} )
                self.__angles[INT_TYPE(idx)] =  {"oIndexes": np.array(angles["oIndexes"], dtype = INT_TYPE), 
                                                 "xIndexes": np.array(angles["xIndexes"], dtype = INT_TYPE),
                                                 "yIndexes": np.array(angles["yIndexes"], dtype = INT_TYPE),
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
               
               #. First item: The name of the improper atom that must be in the plane.
               #. Second item: The name of the atom 'O' considered the origin of the plane.
               #. Third item: The name of the atom 'x' used to calculated 'Ox' vector.
               #. Fourth item: The name of the atom 'y' used to calculated 'Oy' vector.
               #. Fifth item: The minimum lower limit or the minimum angle allowed in degrees.
               #. Sixth item: The maximum upper limit or the maximum angle allowed in degrees.
        
        ::
        
            e.g. (Benzene):  anglesDefinition={"BENZ": [('C3','C1','C2','C6', -10, 10),
                                                        ('C4','C1','C2','C6', -10, 10),
                                                        ('C5','C1','C2','C6', -10, 10) ] }
                                                  
        """
        if self.engine is None:
            raise Exception("Engine is not defined. Can't create angles")
        assert isinstance(anglesDefinition, dict), "anglesDefinition must be a dictionary"
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
                assert len(angle)==6
                improperAt, oAt, xAt, yAt, lower, upper = angle
                assert is_number(lower)
                lower = FLOAT_TYPE(lower)
                assert is_number(upper)
                upper = FLOAT_TYPE(upper)
                assert lower>=-90, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be bigger or equal to -90 deg.")
                assert upper>lower, log.LocalLogger("fullrmc").logger.error("anglesMap items lists fifth item must be smaller than the sixth item")
                assert upper<=90, log.LocalLogger("fullrmc").logger.error("anglesMap items lists sixth item must be smaller or equal to 90")
                lower *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
                upper *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
                # check for redundancy
                append = True
                for b in molAnglesMap:
                    if (b[0]==improperAt):
                        if sorted(oAt,xAt,yAt) == sorted(b[1],b[2],b[3]):
                            log.LocalLogger("fullrmc").logger.warn("Improper angle definition for improper atom index '%i' and (O,x,y) atoms indexes (%i,%i,%i)  already defined. New angle limits [%.3f,%.3f] ignored and old angle limits [%.3f,%.3f] kept."%(improperIdx, oIdx, xIdx, yIdx, lower, upper, self.__angles[improperIdx]["lower"][index], self.__angles[improperIdx]["upper"][index]))
                            append = False
                            break
                if append:
                    molAnglesMap.append((improperAt, oAt, xAt, yAt, lower, upper))
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
                improperIdx = indexes[ names.index(angle[0]) ]
                oIdx        = indexes[ names.index(angle[1]) ]
                xIdx        = indexes[ names.index(angle[2]) ]
                yIdx        = indexes[ names.index(angle[3]) ]
                lower       = angle[4]
                upper       = angle[5]
                anglesMap.append((improperIdx, oIdx, xIdx, yIdx, lower, upper))
        # create angles
        self.set_angles(anglesMap=anglesMap)
    
    def compute_chi_square(self, data):
        """ 
        Compute the chi square of data not satisfying constraint conditions. 
        
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
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
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
            anglesDict[idx] = self.angles[idx] 
        # compute data before move
        dataDict = full_improper_angles( anglesDict=anglesDict ,
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
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # get angles dictionary slice
        anglesIndexes = []
        for idx in indexes:
            anglesIndexes.extend( self.__atomsLUAD[idx] )
        anglesDict = {}
        for idx in set(anglesIndexes):
            anglesDict[idx] = self.angles[idx] 
        # compute data before move
        dataDict = full_improper_angles( anglesDict=anglesDict ,
                                         boxCoords=self.engine.boxCoordinates,
                                         basis=self.engine.basisVectors ,
                                         reduceAngleToUpper = False,
                                         reduceAngleToLower = False)
        # set data before move
        self.set_active_atoms_data_before_move( dataDict )
        self.set_active_atoms_data_after_move(None)
        
    def compute_after_move(self, indexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed
        
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
        dataDict = full_improper_angles( anglesDict=anglesDict ,
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
        Accept move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
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
        Reject move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)



    
    
            