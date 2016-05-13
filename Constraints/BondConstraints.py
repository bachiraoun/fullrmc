"""
BondConstraints contains classes for all constraints related bonds between atoms.

.. inheritance-diagram:: fullrmc.Constraints.BondConstraints
    :parts: 1
"""

# standard libraries imports
import itertools

# external libraries imports
import numpy as np
from timeit import default_timer as timer

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.bonds import full_bonds

class BondConstraint(RigidConstraint, SingularConstraint):
    """
    Its controls the bond between 2 defined atoms.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. bondsMap (list): The bonds map definition.
               Every item must be a list of four items.\n
               #. First item is the first atom index.
               #. Second item the second atom index forming the bond, 
               #. Third item: The lower limit or the minimum bond length allowed.
               #. Fourth item: The upper limit or the maximum bond length allowed.
        #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
    
    .. code-block:: python
    
        ## Water (H2O) molecule sketch
        ## 
        ##              O
        ##            /   \      
        ##         /   H2O   \  
        ##       H1           H2 
 
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.BondConstraints import BondConstraint
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # create and add constraint
        BC = BondConstraint(engine=None)
        ENGINE.add_constraints(BC)
        
        # define intra-molecular bonds 
        BC.create_angles_by_definition( bondsDefinition={"H2O": [ ('O','H1', 0.88, 1.02),
                                                                  ('O','H2', 0.88, 1.02) ]} )
    
    """
    
    def __init__(self, engine, bondsMap=None, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability)
        # set bonds map
        self.set_bonds(bondsMap)
        
    @property
    def bondsMap(self):
        """ Get bonds map"""
        return self.__bondsMap
    
    @property
    def bonds(self):
        """ Get bonds dictionary"""
        return self.__bonds
        
    @property
    def standardError(self):
        """Get constraint's current standard error."""
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
            before = self.activeAtomsDataBeforeMove[index]["reducedDistances"]
            after  = self.activeAtomsDataAfterMove[index]["reducedDistances"]
            if np.any((after-before)>PRECISION):
                reject = True
                break
        return reject
        
    def set_bonds(self, bondsMap):
        """ 
        Sets the bonds dictionary by parsing the bondsMap list.
        
        :Parameters:
            #. bondsMap (list): The bonds map definition.
               Every item must be a list of four items.\n
               #. First item is the first atom index.
               #. Second item the second atom index forming the bond, 
               #. Third item: The lower limit or the minimum bond length allowed.
               #. Fourth item: The upper limit or the maximum bond length allowed.
        """
        map = []
        if self.engine is not None:
            if bondsMap is not None:
                assert isinstance(bondsMap, (list, set, tuple)), LOGGER.error("bondsMap must be None or a list")
                for bond in bondsMap:
                    assert isinstance(bond, (list, set, tuple)), LOGGER.error("bondsMap items must be lists")
                    bond = list(bond)
                    assert len(bond)==4, LOGGER.error("bondsMap items must be lists of 4 items each")
                    idx1, idx2, lower, upper = bond
                    assert is_integer(idx1), LOGGER.error("bondsMap items lists first item must be an integer")
                    idx1 = INT_TYPE(idx1)
                    assert is_integer(idx2), LOGGER.error("bondsMap items lists second item must be an integer")
                    idx2 = INT_TYPE(idx2)
                    assert idx1>=0, LOGGER.error("bondsMap items lists first item must be positive")
                    assert idx2>=0, LOGGER.error("bondsMap items lists second item must be positive")
                    assert idx1!=idx2, LOGGER.error("bondsMap items lists first and second items can't be the same")
                    assert is_number(lower), LOGGER.error("bondsMap items lists of third item must be a number")
                    lower = FLOAT_TYPE(lower)
                    assert is_number(upper), LOGGER.error("bondsMap items lists of fourth item must be a number")
                    upper = FLOAT_TYPE(upper)
                    assert lower>=0, LOGGER.error("bondsMap items lists third item must be positive")
                    assert upper>lower, LOGGER.error("bondsMap items lists third item must be smaller than the fourth item")
                    map.append((idx1, idx2, lower, upper))  
        # set bondsMap definition
        self.__bondsMap = map      
        # create bonds list of indexes arrays
        self.__bonds = {}
        if self.engine is not None:
            # parse bondsMap
            for bond in self.__bondsMap:
                self.add_bond(bond)
            # finalize bonds
            for idx in self.engine.pdb.xindexes:
                bonds = self.__bonds.get(idx, {"indexes":[],"lower":[],"upper":[]} )
                self.__bonds[INT_TYPE(idx)] =  {"indexes": np.array(bonds["indexes"], dtype = INT_TYPE)  ,
                                                "lower"  : np.array(bonds["lower"]  , dtype = FLOAT_TYPE),
                                                "upper"  : np.array(bonds["upper"]  , dtype = FLOAT_TYPE) }
        # reset constraint
        self.reset_constraint()
    
    def add_bond(self, bond):
        """
        Add a single bond to the list of constraint bonds.
        
        :Parameters:
            #. bond (list): The bond list of four items.\n
               #. First item is the first atom index.
               #. Second item the second atom index forming the bond, 
               #. Third item: The lower limit or the minimum bond length allowed.
               #. Fourth item: The upper limit or the maximum bond length allowed.
        """
        idx1, idx2, lower, upper = bond
        assert idx1<len(self.engine.pdb), LOGGER.error("bond atom index must be smaller than maximum number of atoms")
        assert idx2<len(self.engine.pdb), LOGGER.error("bond atom index must be smaller than maximum number of atoms")
        idx1 = INT_TYPE(idx1)
        idx2 = INT_TYPE(idx2)
        # create bonds
        if not self.__bonds.has_key(idx1):
            idx1ToArray = False
            self.__bonds[idx1] = {"indexes":[],"lower":[],"upper":[]}
        else:
            idx1ToArray = not isinstance(self.__bonds[idx1]["indexes"], list)
            self.__bonds[idx1] = {"indexes":list(self.__bonds[idx1]["indexes"]),
                                  "lower"  :list(self.__bonds[idx1]["lower"]),
                                  "upper"  :list(self.__bonds[idx1]["upper"]) }
        if not self.__bonds.has_key(idx2):
            idx2ToArray = False
            self.__bonds[idx2] = {"indexes":[],"lower":[],"upper":[]}
        else:
            idx2ToArray = not isinstance(self.__bonds[idx2]["indexes"], list)
            self.__bonds[idx2] = {"indexes":list(self.__bonds[idx2]["indexes"]),
                                  "lower"  :list(self.__bonds[idx2]["lower"]),
                                  "upper"  :list(self.__bonds[idx2]["upper"]) }
        # check for redundancy and append
        if idx2 in self.__bonds[idx1]["indexes"]:
            index = self.__bonds[idx1]["indexes"].index(idx2)
            LOGGER.warn("Atom index '%i' is already defined in atom '%i' bonds list. New bond limits [%.3f,%.3f] are ignored and old bond limits [%.3f,%.3f] are kept. "%(idx2, idx1, lower, upper, self.__bonds[idx1]["lower"][indexes], self.__bonds[idx1]["upper"][indexes]))
        else:
            self.__bonds[idx1]["indexes"].append(idx2)
            self.__bonds[idx1]["lower"].append(lower)
            self.__bonds[idx1]["upper"].append(upper)
        if idx1 in self.__bonds[idx2]["indexes"]:
            index = self.__bonds[idx2]["indexes"].index(idx1)
            LOGGER.warn("Atom index '%i' is already defined in atom '%i' bonds list. New bond limits [%.3f,%.3f] are ignored and old bond limits [%.3f,%.3f] are kept. "%(idx1, idx2, lower, upper, self.__bonds[idx2]["lower"][indexes], self.__bonds[idx1]["upper"][indexes]))
        else:
            self.__bonds[idx2]["indexes"].append(idx1)
            self.__bonds[idx2]["lower"].append(lower)
            self.__bonds[idx2]["upper"].append(upper)
        # make array
        if idx1ToArray:
            bonds = self.__bonds.get(idx1, {"indexes":[],"lower":[],"upper":[]} )
            self.__bonds[idx1] =  {"indexes": np.array(bonds["indexes"], dtype = INT_TYPE)  ,
                                    "lower"  : np.array(bonds["lower"]  , dtype = FLOAT_TYPE),
                                    "upper"  : np.array(bonds["upper"]  , dtype = FLOAT_TYPE) }
        if idx2ToArray:
            bonds = self.__bonds.get(idx2, {"indexes":[],"lower":[],"upper":[]} )
            self.__bonds[idx2] =  {"indexes": np.array(bonds["indexes"], dtype = INT_TYPE)  ,
                                    "lower"  : np.array(bonds["lower"]  , dtype = FLOAT_TYPE),
                                    "upper"  : np.array(bonds["upper"]  , dtype = FLOAT_TYPE) }
                                                         
    def create_bonds_by_definition(self, bondsDefinition):
        """ 
        Creates bondsMap using bonds definition.
        Calls set_bonds(bondsMap) and generates bonds attribute.
        
        :Parameters:
            #. bondsDefinition (dict): The bonds definition. 
               Every key must be a molecule name (residue name in pdb file). 
               Every key value must be a list of bonds definitions. 
               Every bond definition is a list of four items where:
               
               #. First item: The name of the first atom forming the bond.
               #. Second item: The name of the second atom forming the bond.
               #. Third item: The lower limit or the minimum bond length allowed.
               #. Fourth item: The upper limit or the maximum bond length allowed.
               
           ::
           
                e.g. (Carbon tetrachloride):  bondsDefinition={"CCL4": [('C','CL1' , 1.55, 1.95),
                                                                        ('C','CL2' , 1.55, 1.95),
                                                                        ('C','CL3' , 1.55, 1.95),                                       
                                                                        ('C','CL4' , 1.55, 1.95) ] }
                                                                    
        """
        if self.engine is None:
            raise Exception(LOGGER.error("engine is not defined. Can't create bonds"))
        assert isinstance(bondsDefinition, dict), LOGGER.error("bondsDefinition must be a dictionary")
        # check map definition
        existingMoleculesNames = sorted(set(self.engine.moleculesNames))
        bondsDef = {}
        for mol, bonds in bondsDefinition.items():
            if mol not in existingMoleculesNames:
                LOGGER.warn("Molecule name '%s' in bondsDefinition is not recognized, bonds definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(bonds, (list, set, tuple)), LOGGER.error("mapDefinition molecule bonds must be a list")
            bonds = list(bonds)
            molBondsMap = []
            for bond in bonds:
                assert isinstance(bond, (list, set, tuple)), LOGGER.error("mapDefinition bonds must be a list")
                bond = list(bond)
                assert len(bond)==4, LOGGER.error("mapDefinition bonds list must be of length 4")
                at1, at2, lower, upper = bond
                assert is_number(lower), LOGGER.error("mapDefinition bonds list third item must be a number")
                lower = FLOAT_TYPE(lower)
                assert is_number(upper), LOGGER.error("mapDefinition bonds list fourth item must be a number")
                upper = FLOAT_TYPE(upper)
                assert lower>=0, LOGGER.error("mapDefinition bonds list third item must be bigger than 0")
                assert upper>lower, LOGGER.error("mapDefinition bonds list fourth item must be bigger than the third item")
                # check for redundancy
                append = True
                for b in molBondsMap:
                    if (b[0]==at1 and b[1]==at2) or (b[1]==at1 and b[0]==at2):
                        LOGGER.warn("Redundant definition for bondsDefinition found. The later '%s' is ignored"%str(b))
                        append = False
                        break
                if append:
                    molBondsMap.append((at1, at2, lower, upper))
            # create bondDef for molecule mol 
            bondsDef[mol] = molBondsMap
        # create mols dictionary
        mols = {}
        for idx in self.engine.pdb.xindexes:
            molName = self.engine.moleculesNames[idx]
            if not molName in bondsDef.keys():    
                continue
            molIdx = self.engine.moleculesIndexes[idx]
            if not mols.has_key(molIdx):
                mols[molIdx] = {"name":molName, "indexes":[], "names":[]}
            mols[molIdx]["indexes"].append(idx)
            mols[molIdx]["names"].append(self.engine.allNames[idx])
        # get bondsMap
        bondsMap = []         
        for val in mols.values():
            indexes = val["indexes"]
            names   = val["names"]
            # get definition for this molecule
            thisDef = bondsDef[val["name"]]
            for bond in thisDef:
                idx1  = indexes[ names.index(bond[0]) ]
                idx2  = indexes[ names.index(bond[1]) ]
                lower = bond[2]
                upper = bond[3]
                bondsMap.append((idx1, idx2, lower, upper))
        # create bonds
        self.set_bonds(bondsMap=bondsMap)
    
    def compute_standard_error(self, data):
        """ 
        Compute the standard error (StdErr) of data not satisfying constraint conditions. 
        
        .. math::
            StdErr = \\sum \\limits_{i}^{C} 
            ( \\beta_{i} - \\beta_{i}^{min} ) ^{2} 
            \\int_{0}^{\\beta_{i}^{min}} \\delta(\\beta-\\beta_{i}) d \\beta
            +
            ( \\beta_{i} - \\beta_{i}^{max} ) ^{2} 
            \\int_{\\beta_{i}^{max}}^{\\infty} \\delta(\\beta-\\beta_{i}) d \\beta
                               
        Where:\n
        :math:`C` is the total number of defined bonds constraints. \n
        :math:`\\beta_{i}^{min}` is the bond constraint lower limit set for constraint i. \n
        :math:`\\beta_{i}^{max}` is the bond constraint upper limit set for constraint i. \n
        :math:`\\beta_{i}` is the bond length computed for constraint i. \n
        :math:`\\delta` is the Dirac delta function. \n
        :math:`\\int_{0}^{\\beta_{i}^{min}} \\delta(\\beta-\\beta_{i}) d \\beta` 
        is equal to 1 if :math:`0 \\leqslant \\beta_{i} \\leqslant \\beta_{i}^{min}` and 0 elsewhere.\n
        :math:`\\int_{\\beta_{i}^{max}}^{\\pi} \\delta(\\beta-\\beta_{i}) d \\beta` 
        is equal to 1 if :math:`\\beta_{i}^{max} \\leqslant \\beta_{i} \\leqslant \\infty` and 0 elsewhere.\n

        :Parameters:
            #. data (numpy.array): The constraint value data to compute standardError.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        standardError = 0
        for idx, bond in data.items():
            standardError +=  np.sum(bond["reducedDistances"]**2)
        return FLOAT_TYPE( standardError )

    def get_constraint_value(self):
        """
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        return self.data
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        dataDict = full_bonds(bonds                 = self.__bonds, 
                              boxCoords             = self.engine.boxCoordinates,
                              basis                 = self.engine.basisVectors,
                              reduceDistanceToUpper = False,
                              reduceDistanceToLower = False)
        self.set_data( dataDict )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        #self.set_standard_error( self.compute_standard_error(data = self.__data) )
        
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # get bonds dictionary slice
        bondsDict = {}
        for idx in indexes:
            bondsDict[idx] = self.__bonds[idx]
        # compute data before move
        dataDict = full_bonds(bonds                 = bondsDict, 
                              boxCoords             = self.engine.boxCoordinates,
                              basis                 = self.engine.basisVectors,
                              reduceDistanceToUpper = False,
                              reduceDistanceToLower = False)
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
        # get bonds dictionary slice
        
        bondsDict = {}
        for idx in indexes:
            bondsDict[idx] = self.__bonds[idx]
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[indexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[indexes] = movedBoxCoordinates
        # compute data after move
        dataDict = full_bonds(bonds                 = bondsDict, 
                              boxCoords             = self.engine.boxCoordinates,
                              basis                 = self.engine.basisVectors,
                              reduceDistanceToUpper = False,
                              reduceDistanceToLower = False)
        self.set_active_atoms_data_after_move( dataDict )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute standardError after move
        #self.set_after_move_standard_error( self.compute_standard_error(data = dataDict ) )
  
    def accept_move(self, indexes):
        """ 
        Accept move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        for idx in indexes:
            self.data[idx]["bondsLength"]      = self.activeAtomsDataAfterMove[idx]["bondsLength"]
            self.data[idx]["reducedDistances"] = self.activeAtomsDataAfterMove[idx]["reducedDistances"]
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



    
    
            