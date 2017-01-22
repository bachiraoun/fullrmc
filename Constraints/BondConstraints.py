"""
BondConstraints contains classes for all constraints related bonds between atoms.

.. inheritance-diagram:: fullrmc.Constraints.BondConstraints
    :parts: 1
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, raise_if_collected, reset_if_collected_out_of_date
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.bonds import full_bonds_coords




class BondConstraint(RigidConstraint, SingularConstraint):
    """
    Controls the bond length defined between 2 defined atoms.
    
    +--------------------------------------------------------------------------------+
    |.. figure:: bondSketch.png                                                      |
    |   :width: 237px                                                                |
    |   :height: 200px                                                               |
    |   :align: center                                                               |
    |                                                                                |
    |   Bond sketch defined between two atoms.                                       |  
    +--------------------------------------------------------------------------------+
    
    .. raw:: html

        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/GxmJae9h78E" 
        frameborder="0" allowfullscreen>
        </iframe>
        
        
    :Parameters:
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
        ENGINE = Engine(path='my_engine.rmc')
        
        # set pdb file
        ENGINE.set_pdb('system.pdb')
        
        # create and add constraint
        BC = BondConstraint()
        ENGINE.add_constraints(BC)
        
        # define intra-molecular bonds 
        BC.create_bonds_by_definition( bondsDefinition={"H2O": [ ('O','H1', 0.88, 1.02),
                                                                 ('O','H2', 0.88, 1.02) ]} )
    
        
    """
    
    def __init__(self, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, rejectProbability=rejectProbability)
        # set atomsCollector datakeys
        self._atomsCollector.set_data_keys( ['map',] )
        # init bonds data
        self.__bondsList = [[],[],[],[]]      
        self.__bonds     = {}
        # set computation cost
        self.set_computation_cost(1.0)
        # create dump flag
        self.__dumpBonds = True
         # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_BondConstraint__bondsList',
                           '_BondConstraint__bonds',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA', tuple(RUNTIME_DATA) )
        
    @property
    def bondsList(self):
        """ List of defined bonds"""
        return self.__bondsList
    
    @property
    def bonds(self):
        """ Get bonds dictionary map of every and each atom"""
        return self.__bonds

    def _on_collector_reset(self):
        self._atomsCollector._randomData = set([])

    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine set","update boundary conditions",):
            # set bonds and reset constraint
            self.set_bonds(self.__bondsList, tform=False)     
            # reset constraint is called in set_bonds
    
    #@raise_if_collected
    def set_bonds(self, bondsList, tform=True):
        """ 
        Sets bonds dictionary by parsing the bondsList list.
        
        :Parameters:
            #. bondsList (None, list): The bonds map definition. If None is given no
               bonds are defined. Otherwise it can be of any of the following two forms:
               
                   tuples format: every item must be a list or tuple of four items.\n
                   #. First item:  the first atom index.
                   #. Second item: the second atom index forming the bond. 
                   #. Third item: The lower limit or the minimum bond length allowed.
                   #. Fourth item: The upper limit or the maximum bond length allowed.
                   
                   four vectors format: List of exaclty four lists or numpy.arrays or vectors of the same length.\n
                   #. First item: list contains the first atom indexes.
                   #. Second item: list contains the second atom indexes forming the bond. 
                   #. Third item: list containing the lower limit or the minimum bond length allowed.
                   #. Fourth item: list containing the upper limit or the maximum bond length allowed.
            #. tform (boolean): set whether given bondsList follows tuples format, If not 
               then it must follow the four vectors one.
        """
        # check if bondsList is given
        if bondsList is None:
            bondsList = [[],[],[],[]]
            tform     = False 
        elif len(bondsList) == 4 and len(bondsList[0]) == 0:
            tform     = False 
        if self.engine is None:
            self.__bondsList = bondsList      
            self.__bonds     = {}
        else:
            NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
            # set bonds and bondsList definition
            oldBondsList = self.__bondsList
            oldBonds     = self.__bonds
            self.__bondsList = [np.array([], dtype=INT_TYPE),
                                np.array([], dtype=INT_TYPE),
                                np.array([], dtype=FLOAT_TYPE),
                                np.array([], dtype=FLOAT_TYPE)]      
            self.__bonds     = {}
            self.__dumpBonds = False
            # build bonds
            try: 
                if tform:
                    for bond in bondsList:
                        self.add_bond(bond)
                else:
                    for idx in xrange(len(bondsList[0])):
                        bond = [bondsList[0][idx], bondsList[1][idx], bondsList[2][idx], bondsList[3][idx]]
                        self.add_bond(bond)
            except Exception as e:
                self.__dumpBonds = True
                self.__bondsList = oldBondsList
                self.__bonds     = oldBonds
                LOGGER.error(e)
                import traceback
                raise Exception(traceback.format_exc())
            self.__dumpBonds = True
            # finalize bonds
            for idx in xrange(NUMBER_OF_ATOMS):
                self.__bonds[INT_TYPE(idx)] = self.__bonds.get(INT_TYPE(idx), {"indexes":[],"map":[]} )
        # dump to repository
        self._dump_to_repository({'_BondConstraint__bondsList':self.__bondsList,
                                  '_BondConstraint__bonds'    :self.__bonds})
        # reset constraint
        self.reset_constraint()
                                                                 
    #@raise_if_collected
    def add_bond(self, bond):
        """
        Add a single bond to the list of constraint bonds.
        
        :Parameters:
            #. bond (list): The bond list of four items.\n
               #. First item: the first atom index.
               #. Second item: the second atom index forming the bond. 
               #. Third item: The lower limit or the minimum bond length allowed.
               #. Fourth item: The upper limit or the maximum bond length allowed.
        """
        assert self.engine is not None, LOGGER.error("setting a bond is not allowed unless engine is defined.")
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        assert isinstance(bond, (list, set, tuple)), LOGGER.error("bond items must be lists")
        assert len(bond)==4, LOGGER.error("bond items must be lists of 4 items each")
        idx1, idx2, lower, upper = bond
        assert is_integer(idx1), LOGGER.error("bondsList items lists first item must be an integer")
        idx1 = INT_TYPE(idx1)
        assert is_integer(idx2), LOGGER.error("bondsList items lists second item must be an integer")
        idx2 = INT_TYPE(idx2)
        assert idx1<NUMBER_OF_ATOMS, LOGGER.error("bond atom index must be smaller than maximum number of atoms")
        assert idx2<NUMBER_OF_ATOMS, LOGGER.error("bond atom index must be smaller than maximum number of atoms")
        assert idx1>=0, LOGGER.error("bond first item must be positive")
        assert idx2>=0, LOGGER.error("bond second item must be positive")
        assert idx1!=idx2, LOGGER.error("bond first and second items can't be the same")
        assert is_number(lower), LOGGER.error("bond third item must be a number")
        lower = FLOAT_TYPE(lower)
        assert is_number(upper), LOGGER.error("bond fourth item must be a number")
        upper = FLOAT_TYPE(upper)
        assert lower>=0, LOGGER.error("bond third item must be positive")
        assert upper>lower, LOGGER.error("bond third item must be smaller than the fourth item")
        # create bonds
        if not self.__bonds.has_key(idx1):
            bondsIdx1 = {"indexes":[],"map":[]} 
        else:
            bondsIdx1 = {"indexes":self.__bonds[idx1]["indexes"], 
                         "map"    :self.__bonds[idx1]["map"] }
        if not self.__bonds.has_key(INT_TYPE(idx2)):
            bondsIdx2 = {"indexes":[],"map":[]} 
        else:
            bondsIdx2 = {"indexes":self.__bonds[idx2]["indexes"], 
                         "map"    :self.__bonds[idx2]["map"] }
        # set bond 
        if idx2 in bondsIdx1["indexes"]:
            assert idx1 in bondsIdx2["indexes"], LOOGER.error("mismatched bonds between atom '%s' and '%s'"%(idx1,idx2))
            at2InAt1 = bondsIdx1["indexes"].index(idx2)
            at1InAt2 = bondsIdx2["indexes"].index(idx1) 
            assert bondsIdx1["map"][at2InAt1] == bondsIdx2["map"][at1InAt2], LOOGER.error("bonded atoms '%s' and '%s' point to different defintions"%(idx1,idx2)) 
            setPos = bondsIdx1["map"][at2InAt1]
            LOGGER.warn("Bond between atom index '%i' and '%i' is already defined. New bond limits [%.3f,%.3f] will replace old bond limits [%.3f,%.3f]. "%(idx2, idx1, lower, upper, self.__bondsList[2][setPos], self.__bondsList[3][setPos]))  
            self.__bondsList[0][setPos] = idx1
            self.__bondsList[1][setPos] = idx2
            self.__bondsList[2][setPos] = lower
            self.__bondsList[3][setPos] = upper
        else:
            bondsIdx1["map"].append( len(self.__bondsList[0]) )
            bondsIdx2["map"].append( len(self.__bondsList[0]) ) 
            bondsIdx1["indexes"].append(idx2) 
            bondsIdx2["indexes"].append(idx1)
            self.__bondsList[0] = np.append(self.__bondsList[0],idx1)
            self.__bondsList[1] = np.append(self.__bondsList[1],idx2)
            self.__bondsList[2] = np.append(self.__bondsList[2],lower)
            self.__bondsList[3] = np.append(self.__bondsList[3],upper)
        self.__bonds[idx1] = bondsIdx1
        self.__bonds[idx2] = bondsIdx2
        # dump to repository
        if self.__dumpBonds:
            self._dump_to_repository({'_BondConstraint__bondsList' :self.__bondsList,
                                      '_BondConstraint__bonds'     :self.__bonds})
            # reset constraint
            self.reset_constraint()

    #@raise_if_collected
    def create_bonds_by_definition(self, bondsDefinition):
        """ 
        Creates bondsList using bonds definition.
        Calls set_bonds(bondsList) and generates bonds attribute.
        
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
            raise Exception(LOGGER.error("engine is not defined. Can't create bonds by definition"))
            return
        if bondsDefinition is None:
            bondsDefinition = {}
        assert isinstance(bondsDefinition, dict), LOGGER.error("bondsDefinition must be a dictionary")
        ALL_NAMES         = self.engine.get_original_data("allNames")
        NUMBER_OF_ATOMS   = self.engine.get_original_data("numberOfAtoms")
        MOLECULES_NAMES   = self.engine.get_original_data("moleculesNames")
        MOLECULES_INDEXES = self.engine.get_original_data("moleculesIndexes")
        # check map definition
        existingMoleculesNames = sorted(set(MOLECULES_NAMES))
        bondsDef = {}
        for mol, bonds in bondsDefinition.items():
            if mol not in existingMoleculesNames:
                LOGGER.warn("Molecule name '%s' in bondsDefinition is not recognized, bonds definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(bonds, (list, set, tuple)), LOGGER.error("mapDefinition molecule bonds must be a list")
            bonds = list(bonds)
            molbondsList = []
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
                for b in molbondsList:
                    if (b[0]==at1 and b[1]==at2) or (b[1]==at1 and b[0]==at2):
                        LOGGER.warn("Redundant definition for bondsDefinition found. The later '%s' is ignored"%str(b))
                        append = False
                        break
                if append:
                    molbondsList.append((at1, at2, lower, upper))
            # create bondDef for molecule mol 
            bondsDef[mol] = molbondsList
        # create mols dictionary
        mols = {}
        for idx in xrange(NUMBER_OF_ATOMS):
            molName = MOLECULES_NAMES[idx]
            if not molName in bondsDef.keys():    
                continue
            molIdx = MOLECULES_INDEXES[idx]
            if not mols.has_key(molIdx):
                mols[molIdx] = {"name":molName, "indexes":[], "names":[]}
            mols[molIdx]["indexes"].append(idx)
            mols[molIdx]["names"].append(ALL_NAMES[idx])
        # get bondsList
        bondsList = []         
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
                bondsList.append((idx1, idx2, lower, upper))
        # create bonds
        self.set_bonds(bondsList=bondsList)
    
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
            #. data (object): The constraint value data to compute standardError.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        return FLOAT_TYPE( np.sum(data["reducedLengths"]**2) )

    def get_constraint_value(self):
        """
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        return self.data
      
    @reset_if_collected_out_of_date
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        if len(self._atomsCollector):
            bondsData   = np.zeros(self.__bondsList[0].shape[0], dtype=FLOAT_TYPE)
            reducedData = np.zeros(self.__bondsList[0].shape[0], dtype=FLOAT_TYPE)
            bondsIndexes = set(set(range(self.__bondsList[0].shape[0])))
            bondsIndexes = list( bondsIndexes-self._atomsCollector._randomData )
            idx1 = self._atomsCollector.get_relative_indexes(self.__bondsList[0][bondsIndexes])
            idx2 = self._atomsCollector.get_relative_indexes(self.__bondsList[1][bondsIndexes])
            lowerLimit = self.__bondsList[2][bondsIndexes]
            upperLimit = self.__bondsList[3][bondsIndexes]
        else:
            idx1 = self._atomsCollector.get_relative_indexes(self.__bondsList[0])
            idx2 = self._atomsCollector.get_relative_indexes(self.__bondsList[1])
            lowerLimit = self.__bondsList[2]
            upperLimit = self.__bondsList[3]
        # compute
        bonds, reduced = full_bonds_coords(idx1                  = idx1,
                                           idx2                  = idx2,
                                           lowerLimit            = lowerLimit, 
                                           upperLimit            = upperLimit, 
                                           boxCoords             = self.engine.boxCoordinates,
                                           basis                 = self.engine.basisVectors,
                                           isPBC                 = self.engine.isPBC,
                                           reduceDistanceToUpper = False,
                                           reduceDistanceToLower = False,
                                           ncores                = INT_TYPE(1))                       
        # create full length data
        if len(self._atomsCollector):
            bondsData[bondsIndexes]   = bonds
            reducedData[bondsIndexes] = reduced
            bonds   = bondsData
            reduced = reducedData
        self.set_data( {"bondsLength":bonds, "reducedLengths":reduced} )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        self.set_standard_error( self.compute_standard_error(data = self.data) )
        # set original data
        if self.originalData is None:
            self._set_original_data(self.data)

    def compute_before_move(self, realIndexes, relativeIndexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # get bonds indexes
        bondsIndexes = []
        #for idx in relativeIndexes:
        for idx in realIndexes:
            bondsIndexes.extend( self.__bonds[idx]['map'] )
        #bondsIndexes = list(set(bondsIndexes))
        # remove collected bonds
        bondsIndexes = list( set(bondsIndexes)-set(self._atomsCollector._randomData) )
        # compute data before move
        if len(bondsIndexes):
            bonds, reduced = full_bonds_coords(idx1                  = self._atomsCollector.get_relative_indexes(self.__bondsList[0][bondsIndexes]), 
                                               idx2                  = self._atomsCollector.get_relative_indexes(self.__bondsList[1][bondsIndexes]), 
                                               lowerLimit            = self.__bondsList[2][bondsIndexes], 
                                               upperLimit            = self.__bondsList[3][bondsIndexes], 
                                               boxCoords             = self.engine.boxCoordinates,
                                               basis                 = self.engine.basisVectors,
                                               isPBC                 = self.engine.isPBC,
                                               reduceDistanceToUpper = False,
                                               reduceDistanceToLower = False,
                                               ncores                = INT_TYPE(1)) 
        else:
            bonds   = None
            reduced = None
        # set data before move
        self.set_active_atoms_data_before_move( {"bondsIndexes":bondsIndexes, "bondsLength":bonds, "reducedLengths":reduced} )
        self.set_active_atoms_data_after_move(None)
        
    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        bondsIndexes = self.activeAtomsDataBeforeMove["bondsIndexes"]
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[relativeIndexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[relativeIndexes] = movedBoxCoordinates
        # compute data after move
        if len(bondsIndexes):
            bonds, reduced = full_bonds_coords(idx1                  = self._atomsCollector.get_relative_indexes(self.__bondsList[0][bondsIndexes]),#self.__bondsList[0][bondsIndexes], 
                                               idx2                  = self._atomsCollector.get_relative_indexes(self.__bondsList[1][bondsIndexes]),#self.__bondsList[1][bondsIndexes], 
                                               lowerLimit            = self.__bondsList[2][bondsIndexes], 
                                               upperLimit            = self.__bondsList[3][bondsIndexes], 
                                               boxCoords             = self.engine.boxCoordinates,
                                               basis                 = self.engine.basisVectors,
                                               isPBC                 = self.engine.isPBC,
                                               reduceDistanceToUpper = False,
                                               reduceDistanceToLower = False,
                                               ncores                = INT_TYPE(1)) 
        else:
            bonds   = None
            reduced = None
        # set active data after move
        self.set_active_atoms_data_after_move( {"bondsLength":bonds, "reducedLengths":reduced} )
        # reset coordinates
        self.engine.boxCoordinates[relativeIndexes] = boxData
        # compute standardError after move
        if bonds is None:
            self.set_after_move_standard_error( self.standardError )
        else:
            # bondsIndexes is a fancy slicing, RL is a copy not a view.
            RL = self.data["reducedLengths"][bondsIndexes]
            self.data["reducedLengths"][bondsIndexes] += reduced-self.activeAtomsDataBeforeMove["reducedLengths"]
            self.set_after_move_standard_error( self.compute_standard_error(data = self.data) )
            self.data["reducedLengths"][bondsIndexes] = RL
  
    def accept_move(self, realIndexes, relativeIndexes):
        """ 
        Accept move
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # get bonds indexes
        bondsIndexes = self.activeAtomsDataBeforeMove["bondsIndexes"]
        if len(bondsIndexes):
            # set new data
            data = self.data
            data["bondsLength"][bondsIndexes]    += self.activeAtomsDataAfterMove["bondsLength"]-self.activeAtomsDataBeforeMove["bondsLength"]
            data["reducedLengths"][bondsIndexes] += self.activeAtomsDataAfterMove["reducedLengths"]-self.activeAtomsDataBeforeMove["reducedLengths"]
            self.set_data( data )
            # update standardError
            self.set_standard_error( self.afterMoveStandardError )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)

    def reject_move(self, realIndexes, relativeIndexes):
        """ 
        Reject move
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
    
    def compute_as_if_amputated(self, realIndex, relativeIndex):
        """ 
        Compute and return constraint's data and standard error as if atom given its 
        its was amputated.
        
        :Parameters:
            #. realIndex (numpy.ndarray): atom index as a numpy array of a single element.
        """
        pass
        
    def accept_amputation(self, realIndex, relativeIndex):
        """ 
        Accept amputation of atom and sets constraints data and standard error accordingly.
        
        :Parameters:
            #. realIndex (numpy.ndarray): atom index as a numpy array of a single element.

        """
        # MAYBE WE DON"T NEED TO CHANGE DATA AND SE. BECAUSE THIS MIGHT BE A PROBLEM 
        # WHEN IMPLEMENTING ATOMS RELEASING. MAYBE WE NEED TO COLLECT DATA INSTEAD, REMOVE
        # AND ADD UPON RELEASE
        # get all involved data
        bondsIndexes = []
        for idx in realIndex:
            bondsIndexes.extend( self.__bonds[idx]['map'] )
        bondsIndexes = list(set(bondsIndexes))
        if len(bondsIndexes):
            # set new data
            data = self.data
            data["bondsLength"][bondsIndexes]    = 0
            data["reducedLengths"][bondsIndexes] = 0
            self.set_data( data )
            # update standardError
            SE = self.compute_standard_error(data = self.get_constraint_value()) 
            self.set_standard_error( SE )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
    
    def reject_amputation(self, realIndex, relativeIndex):
        """ 
        Reject amputation of atom.
        
        :Parameters:
            #. realIndex (numpy.ndarray): atom index as a numpy array of a single element.
        """
        pass
           
    def _on_collector_collect_atom(self, realIndex):
        # get bond indexes
        BI = self.__bonds[realIndex]['map']
        # append all mapped bonds to collector's random data
        self._atomsCollector._randomData = self._atomsCollector._randomData.union( set(BI) )
        #print realIndex, BI, self._atomsCollector._randomData
        # collect atom bondIndexes
        self._atomsCollector.collect(realIndex, dataDict={'map':BI})  
    
    def plot(self, ax=None, nbins=50, subplots=True, 
                   wspace=0.3, hspace=0.3,
                   histtype='bar', lineWidth=None, lineColor=None,
                   xlabel=True, xlabelSize=16,
                   ylabel=True, ylabelSize=16,
                   legend=True, legendCols=1, legendLoc='best',
                   title=True, titleStdErr=True, usedFrame=True,):
        """ 
        Plot bonds constraint distribution histogram.
        
        :Parameters:
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If ax is given, the figure won't be rendered and drawn and subplots 
               parameters will be omitted. If None is given a new plot figure will be 
               created and the figue will be rendered and drawn.
            #. nbins (int): number of bins in histogram.
            #. subplots (boolean): Whether to add plot constraint on multiple axes.
            #. wspace (float): The amount of width reserved for blank space between 
               subplots, expressed as a fraction of the average axis width.
            #. hspace (float): The amount of height reserved for white space between 
               subplots, expressed as a fraction of the average axis height.
            #. histtype (string): the histogram type. optional among
                ['bar', 'barstacked', 'step', 'stepfilled']
            #. lineWidth (None, integer): bars contour line width. If None then default
               value will be given.
            #. lineColor (None, integer): bars contour line color. If None then default
               value will be given.
            #. xlabel (boolean): Whether to create x label.
            #. xlabelSize (number): The x label font size.
            #. ylabel (boolean): Whether to create y label.
            #. ylabelSize (number): The y label font size.
            #. legend (boolean): Whether to create the legend or not
            #. legendCols (integer): Legend number of columns.
            #. legendLoc (string): The legend location. Anything among
               'right', 'center left', 'upper right', 'lower right', 'best', 'center', 
               'lower left', 'center right', 'upper left', 'upper center', 'lower center'
               is accepted.
            #. title (boolean): Whether to create the title or not
            #. usedFrame(boolean): Whether to show used frame name.
            #. titleStdErr (boolean): Whether to show constraint standard error value in title.
        
        :Returns:
            #. axes (matplotlib Axes, List): The matplotlib axes or a list of axes.
        """
        # get constraint value
        output = self.get_constraint_value()
        if output is None:
            LOGGER.warn("%s constraint data are not computed."%(self.__class__.__name__))
            return
        # compute categories 
        categories = {}
        lower = self.__bondsList[2]
        upper = self.__bondsList[3]
        for idx in xrange(self.__bondsList[0].shape[0]):
            if self._atomsCollector.is_collected(idx):
                continue
            l = lower[idx]
            u = upper[idx]
            k = (l,u)
            L = categories.get(k, [])
            L.append(idx)
            categories[k] = L
        ncategories = len(categories.keys())
        # import matplotlib
        import matplotlib.pyplot as plt
        # get axes
        if ax is None:
            if subplots and ncategories>1:
                x = np.ceil(np.sqrt(ncategories))
                y = np.ceil(ncategories/x)
                _, N_AXES = plt.subplots(int(x), int(y) )
                N_AXES = N_AXES.flatten()
                plt.subplots_adjust(wspace=wspace, hspace=hspace)
                FIG = N_AXES[0].get_figure()
            else:
                AXES = plt.gca()
                FIG = AXES.get_figure()
                subplots = False
        else:
            AXES = ax  
            FIG = AXES.get_figure()
            subplots = False 
        # start plotting
        COLORS = ["b",'g','r','c','y','m']
        if subplots:
            for idx, key in enumerate(categories.keys()): 
                L,U  = key
                COL  = COLORS[idx%len(COLORS)]
                AXES = N_AXES[idx]
                idxs = categories[key]
                data = self.data["bondsLength"][idxs]
                # plot histogram
                D, _, P = AXES.hist(x=data, bins=nbins, 
                                    color=COL, label="(%.2f,%.2f)"%(L,U),
                                    histtype=histtype)
                # vertical lines
                Y = max(D)
                AXES.plot([L,L],[0,Y], linewidth=1.0, color='k', linestyle='--')
                AXES.plot([U,U],[0,Y], linewidth=1.0, color='k', linestyle='--')
                # legend
                if legend:
                    AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
                # set axis labels
                if xlabel:
                    AXES.set_xlabel("$r(\AA)$", size=xlabelSize)
                if ylabel:
                    AXES.set_ylabel("$number$"  , size=ylabelSize)
                if lineWidth is not None:
                    [p.set_linewidth(lineWidth) for p in P]
                if lineColor is not None:
                    [p.set_edgecolor(lineColor) for p in P]
        else:
            for idx, key in enumerate(categories.keys()): 
                L,U  = key
                COL  = COLORS[idx%len(COLORS)]
                idxs = categories[key]
                data = self.data["bondsLength"][idxs]
                # plot histogram
                D, _, P = AXES.hist(x=data, bins=nbins, 
                                    color=COL, label="(%.2f,%.2f)"%(L,U),
                                    histtype=histtype)
                # vertical lines
                Y = max(D)
                AXES.plot([L,L],[0,Y], linewidth=1.0, color='k', linestyle='--')
                AXES.plot([U,U],[0,Y], linewidth=1.0, color='k', linestyle='--')
                if lineWidth is not None:
                    [p.set_linewidth(lineWidth) for p in P]
                if lineColor is not None:
                    [p.set_edgecolor(lineColor) for p in P]
            # legend
            if legend:
                AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
            # set axis labels
            if xlabel:
                AXES.set_xlabel("$r(\AA)$", size=xlabelSize)
            if ylabel:
                AXES.set_ylabel("$number$"  , size=ylabelSize)
            
            
        # set title
        if title:
            if usedFrame:
                t = '$frame: %s$ : '%self.engine.usedFrame.replace('_','\_')
            else:
                t = ''
            if titleStdErr and self.standardError is not None:
                t += "$std$ $error=%.6f$ "%(self.standardError)
            if len(t):
                FIG.suptitle(t, fontsize=14)
        
        # set background color
        FIG.patch.set_facecolor('white')
        #show
        if ax is None:
            plt.show()
        # return axes
        if subplots:
            return N_AXES
        else:
            return AXES   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
            