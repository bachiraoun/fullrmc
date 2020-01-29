"""
BondConstraints contains classes for all constraints related to bond length
between atoms.

.. inheritance-diagram:: fullrmc.Constraints.BondConstraints
    :parts: 1
"""
# standard libraries imports
from __future__ import print_function
import copy, re

# external libraries imports
import numpy as np

# fullrmc imports
from ..Globals import INT_TYPE, FLOAT_TYPE, PI, LOGGER
from ..Globals import str, long, unicode, bytes, basestring, range, xrange, maxint
from ..Core.Collection import is_number, is_integer, raise_if_collected, reset_if_collected_out_of_date
from ..Core.Collection import get_caller_frames
from ..Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from ..Core.bonds import full_bonds_coords




class BondConstraint(RigidConstraint, SingularConstraint):
    """
    Controls the bond's length defined between two atoms.

    +--------------------------------------------------------------------------+
    |.. figure:: bondSketch.png                                                |
    |   :width: 237px                                                          |
    |   :height: 200px                                                         |
    |   :align: center                                                         |
    |                                                                          |
    |   Bond sketch defined between two atoms.                                 |
    +--------------------------------------------------------------------------+

    .. raw:: html

        <iframe width="560" height="315"
        src="https://www.youtube.com/embed/GxmJae9h78E"
        frameborder="0" allowfullscreen>
        </iframe>


    +------------------------------------------------------------------------------+
    |.. figure:: bond_constraint_plot_method.png                                   |
    |   :width: 530px                                                              |
    |   :height: 400px                                                             |
    |   :align: left                                                               |
    +------------------------------------------------------------------------------+


    :Parameters:
        #. rejectProbability (Number): rejecting probability of all steps
           where standardError increases. It must be between 0 and 1 where 1
           means rejecting all steps where standardError increases and 0 means
           accepting all steps regardless whether standardError increases or not.

    .. code-block:: python

        ## Water (H2O) molecule sketch
        ##
        ##              O
        ##            /   \\
        ##         /   H2O   \\
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
        self.__bondsDefinition = None
        self.__bondsList       = [[],[],[],[]]
        self.__bonds           = {}
        # set computation cost
        self.set_computation_cost(1.0)
        # create dump flag
        self.__dumpBonds = True
         # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_BondConstraint__bondsDefinition',
                           '_BondConstraint__bondsList',
                           '_BondConstraint__bonds',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA', tuple(RUNTIME_DATA) )

    def _codify_update__(self, name='constraint', addDependencies=True):
        dependencies = []
        code         = []
        if addDependencies:
            code.extend(dependencies)
        code.append("{name}.set_used({val})".format(name=name, val=self.used))
        code.append("{name}.set_reject_probability({val})".format(name=name, val=self.rejectProbability))
        if self.__bondsDefinition is not None:
            code.append("{name}.create_bonds_by_definition({val})".format(name=name, val=self.bondsDefinition))
        else:
            bonds = self.bondsList
            code.append("bonds = {val}".format(val=bonds))
            code.append("{name}.set_bonds(bonds)".format(name=name, val=bonds))
        # return
        return dependencies, '\n'.join(code)


    def _codify__(self, engine, name='constraint', addDependencies=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        dependencies = 'from fullrmc.Constraints import BondConstraints'
        code         = []
        if addDependencies:
            code.append(dependencies)
        code.append("{name} = BondConstraints.BondConstraint\
(rejectProbability={rejectProbability})".format(name=name, rejectProbability=self.rejectProbability))
        code.append("{engine}.add_constraints([{name}])".format(engine=engine, name=name))
        if self.__bondsDefinition is not None:
            code.append("{name}.create_bonds_by_definition({bonds})".
            format(name=name, bonds=self.__bondsDefinition))
        elif len(self.__bondsList[0]):
            code.append("{name}.set_bonds({bonds})".
            format(name=name, bonds=self.__bondsList))
        # return
        return [dependencies], '\n'.join(code)

    @property
    def bondsList(self):
        """ List of defined bonds"""
        return self.__bondsList

    @property
    def bondsDefinition(self):
        """bonds definition copy if bonds are defined as such"""
        return copy.deepcopy(self.__bondsDefinition)

    @property
    def bonds(self):
        """ Bonds dictionary map of every and each atom"""
        return self.__bonds

    def _on_collector_reset(self):
        self._atomsCollector._randomData = set([])

    def listen(self, message, argument=None):
        """
        Listens to any message sent from the Broadcaster.

        :Parameters:
            #. message (object): Any python object to send to constraint's
               listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in ("engine set","update pdb","update molecules indexes","update elements indexes","update names indexes"):
            # set bonds and reset constraint
            if self.__bondsDefinition is not None:
                self.create_bonds_by_definition(self.__bondsDefinition)
            else:
                self.set_bonds(self.__bondsList, tform=False)
            # reset constraint is called in set_bonds
        elif message in ("update boundary conditions",):
            # reset constraint
            self.reset_constraint()


    #@raise_if_collected
    def set_bonds(self, bondsList, tform=True):
        """
        Sets bonds dictionary by parsing bondsList list.

        :Parameters:
            #. bondsList (None, list): Bonds definition list. If None is given
               no bonds are defined. Otherwise it can be of any of the
               following two forms:

                   tuples format: every item must be a list or tuple of four
                   items.\n
                   #. First atom index forming the bond.
                   #. Second atom index forming the bond.
                   #. Lower limit or the minimum bond length
                      allowed.
                   #. Upper limit or the maximum bond length
                      allowed.

                   four vectors format: List of exaclty four lists or
                   numpy.arrays of the same length.\n
                   #. List contains the first atoms index
                      forming the bond.
                   #. List contains the second atoms index
                      forming the bond.
                   #. List containing the lower limit or
                      the minimum bond length allowed.
                   #. List containing the upper limit or
                      the maximum bond length allowed.

            #. tform (boolean): set whether given bondsList follows tuples
               format, If not then it must follow the four vectors one.
        """
        assert self.engine is not None, LOGGER.error("setting bonds is not allowed unless engine is defined.")
        assert isinstance(bondsList, (list,set,tuple)), "bondsList must be a list"
        # convert to list of tuples
        if not tform:
            assert len(bondsList) == 4, LOGGER.error("non tuple form bondsList must be a list of four items")
            assert all([isinstance(i, (list,tuple,np.ndarray)) for i in bondsList]), LOGGER.error("non tuple form bondsList must be a list of list or tuple or numpy.ndarray")
            assert all([len(i)==len(bondsList[0]) for i in bondsList]), LOGGER.error("bondsList items list length mismatch")
            bondsList = zip(*bondsList)
        # get number of atoms
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        # loop bonds
        tempB  = {}
        bondsL = [[],[],[],[]]
        bonds  = {}
        for b in bondsList:
            assert isinstance(b, (list, set, tuple)), LOGGER.error("bonds item must be a list")
            assert len(b) == 4, LOGGER.error("bondsList items lists must have 4 items")
            idx1,idx2,lower,upper = b
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
            assert (idx1,idx2) not in tempB, LOGGER.error("redundant definition of bonds between atoms '%i' and '%i'"%(idx1,idx2))
            assert (idx2,idx1) not in tempB, LOGGER.error("redundant definition of bonds between atoms '%i' and '%i'"%(idx1,idx2))
            tempB[(idx1,idx2)] = True
            # create bonds
            if not idx1 in bonds:
                bondsIdx1 = {"indexes":[],"map":[]}
            else:
                bondsIdx1 = {"indexes":bonds[idx1]["indexes"],
                             "map"    :bonds[idx1]["map"] }
            if not idx2 in bonds:
                bondsIdx2 = {"indexes":[],"map":[]}
            else:
                bondsIdx2 = {"indexes":bonds[idx2]["indexes"],
                             "map"    :bonds[idx2]["map"] }
            # set bond
            if idx2 in bondsIdx1["indexes"]:
                at2InAt1 = bondsIdx1["indexes"].index(idx2)
                at1InAt2 = bondsIdx2["indexes"].index(idx1)
                setPos   = bondsIdx1["map"][at2InAt1]
                bondsL[0][setPos] = idx1
                bondsL[1][setPos] = idx2
                bondsL[2][setPos] = lower
                bondsL[3][setPos] = upper
            else:
                bondsIdx1["map"].append( len(bondsL[0]) )
                bondsIdx2["map"].append( len(bondsL[0]) )
                bondsIdx1["indexes"].append(idx2)
                bondsIdx2["indexes"].append(idx1)
                bondsL[0].append(idx1)
                bondsL[1].append(idx2)
                bondsL[2].append(lower)
                bondsL[3].append(upper)
            bonds[idx1] = bondsIdx1
            bonds[idx2] = bondsIdx2
        # finalize bonds
        for idx in xrange(NUMBER_OF_ATOMS):
            bonds[INT_TYPE(idx)] = bonds.get(INT_TYPE(idx), {"indexes":[],"map":[]} )
        # set attributes
        self.__bondsDefinition = None
        self.__bonds           = bonds
        self.__bondsList       = [np.array(bondsL[0], dtype=INT_TYPE),
                                  np.array(bondsL[1], dtype=INT_TYPE),
                                  np.array(bondsL[2], dtype=FLOAT_TYPE),
                                  np.array(bondsL[3], dtype=FLOAT_TYPE),]
        # dump to repository
        if self.__dumpBonds:
            self._dump_to_repository({'_BondConstraint__bondsDefinition' :self.__bondsDefinition,
                                      '_BondConstraint__bondsList'       :self.__bondsList,
                                      '_BondConstraint__bonds'           :self.__bonds})
            # reset constraint
            self.reset_constraint()

    #@raise_if_collected
    def create_bonds_by_definition(self, bondsDefinition):
        """
        Creates bondsList using bonds definition.
        Calls set_bonds(bondsList) and generates bonds attribute.

        :Parameters:
            #. bondsDefinition (dict): The bonds definition.
               Every key must be a molecule's name.
               Every key value must be a list of bonds definitions.
               Every bond definition is a list of four items where:

               #. Name of the first atom forming the bond.
               #. Name of the second atom forming the bond.
               #. Lower limit or the minimum bond length allowed.
               #. Upper limit or the maximum bond length allowed.

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
        ALL_NAMES       = self.engine.get_original_data("allNames")
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        MOLECULES_NAME  = self.engine.get_original_data("moleculesName")
        MOLECULES_INDEX = self.engine.get_original_data("moleculesIndex")
        # check map definition
        existingmoleculesName = sorted(set(MOLECULES_NAME))
        bondsDef = {}
        for mol in bondsDefinition:
            bonds = bondsDefinition[mol]
            if mol not in existingmoleculesName:
                LOGGER.usage("Molecule name '%s' in bondsDefinition is not recognized, bonds definition for this particular molecule is omitted"%str(mol))
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
            molName = MOLECULES_NAME[idx]
            if not molName in bondsDef:
                continue
            molIdx = MOLECULES_INDEX[idx]
            if not molIdx in mols:
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
        self.__dumpBonds = False
        try:
            self.set_bonds(bondsList=bondsList)
        except Exception as err:
            self.__dumpBonds = True
            raise Exception(err)
        else:
            self.__dumpBonds = True
            self.__bondsDefinition = bondsDefinition
            self._dump_to_repository({'_BondConstraint__bondsDefinition' :self.__bondsDefinition,
                                      '_BondConstraint__bondsList'       :self.__bondsList,
                                      '_BondConstraint__bonds'           :self.__bonds})
            # reset constraint
            self.reset_constraint()


    def compute_standard_error(self, data):
        """
        Compute the standard error (StdErr) of data not satisfying constraint
        conditions.

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
            #. data (object): Data to compute standardError.

        :Returns:
            #. standardError (number): The calculated standardError of the
            given.
        """
        return FLOAT_TYPE( np.sum(data["reducedLengths"]**2) )

    def get_constraint_value(self):
        """
        Get constraint's data.

        :Returns:
            #. data (numpy.array): The constraint value data
        """
        return self.data

    @reset_if_collected_out_of_date
    def compute_data(self, update=True):
        """ Compute constraint's data.

        :Parameters:
            #. update (boolean): whether to update constraint data and
               standard error with new computation. If data is computed and
               updated by another thread or process while the stochastic
               engine is running, this might lead to a state alteration of
               the constraint which will lead to a no additional accepted
               moves in the run

        :Returns:
            #. data (dict): constraint data dictionary
            #. standardError (float): constraint standard error
        """
        if len(self._atomsCollector):
            bondsData   = np.zeros(self.__bondsList[0].shape[0], dtype=FLOAT_TYPE)
            reducedData = np.zeros(self.__bondsList[0].shape[0], dtype=FLOAT_TYPE)
            #bondsIndexes = set(set(range(self.__bondsList[0].shape[0])))
            bondsIndexes = set(range(self.__bondsList[0].shape[0]))
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
        # create data and compute standard error
        data     = {"bondsLength":bonds, "reducedLengths":reduced}
        stdError = self.compute_standard_error(data = data)
        # update
        if update:
            self.set_data( data )
            self.set_active_atoms_data_before_move(None)
            self.set_active_atoms_data_after_move(None)
            # set standardError
            self.set_standard_error( stdError )
            # set original data
            if self.originalData is None:
                self._set_original_data(self.data)
        # return
        return data, stdError

    def compute_before_move(self, realIndexes, relativeIndexes):
        """
        Compute constraint's data before move is executed.

        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms index the move will
               be applied to.
            #. relativeIndexes (numpy.ndarray): Not used here.
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
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Group atoms relative index
               the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new
               coordinates.
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
        # increment tried
        self.increment_tried()

    def accept_move(self, realIndexes, relativeIndexes):
        """
        Accept move

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
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
        # increment accepted
        self.increment_accepted()

    def reject_move(self, realIndexes, relativeIndexes):
        """
        Reject move

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)

    def compute_as_if_amputated(self, realIndex, relativeIndex):
        """
        Compute and return constraint's data and standard error as if atom given its
        its was amputated.

        :Parameters:
            #. realIndex (numpy.ndarray): Not used here.
            #. relativeIndex (numpy.ndarray): Not used here.
        """
        pass

    def accept_amputation(self, realIndex, relativeIndex):
        """
        Accept amputation of atom and sets constraints data and standard error accordingly.

        :Parameters:
            #. realIndex (numpy.ndarray): Atom's index as a numpy array
               of a single element.
            #. relativeIndex (numpy.ndarray): Not used here.
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
            #. realIndex (numpy.ndarray): Not used here.
            #. relativeIndex (numpy.ndarray): Not used here.
        """
        pass

    def _on_collector_collect_atom(self, realIndex):
        # get bond indexes
        BI = self.__bonds[realIndex]['map']
        # append all mapped bonds to collector's random data
        self._atomsCollector._randomData = self._atomsCollector._randomData.union( set(BI) )
        #print(realIndex, BI, self._atomsCollector._randomData)
        # collect atom bondIndexes
        self._atomsCollector.collect(realIndex, dataDict={'map':BI})


    def _plot(self,frameIndex, propertiesLUT,
                   spacing,numberOfTicks,nbins,splitBy,
                   ax, barsRelativeWidth,limitsParams,
                   legendParams,titleParams,
                   xticksParams, yticksParams,
                   xlabelParams, ylabelParams,
                   stackHorizontal, gridParams,
                   colorCodeXticksLabels, *args, **kwargs):

        # get needed data
        frame                = propertiesLUT['frames-name'][frameIndex]
        data                 = propertiesLUT['frames-data'][frameIndex]
        standardError        = propertiesLUT['frames-standard_error'][frameIndex]
        numberOfRemovedAtoms = propertiesLUT['frames-number_of_removed_atoms'][frameIndex]
        # import matplotlib
        import matplotlib.pyplot as plt
        # compute categories
        if splitBy == 'name':
            splitBy = self.engine.get_original_data("allNames", frame=frame)
        elif splitBy == 'element':
            splitBy = self.engine.get_original_data("allElements", frame=frame)
        else:
            splitBy = None
        # check for angles
        if not len(self.__bondsList[0]):
            LOGGER.warn("@{frm} no bonds found. It's even not defined or no atoms where found in definition.".format(frm=frame))
            return
        # build categories
        atom1 = self.__bondsList[0]
        atom2 = self.__bondsList[1]
        lower = self.__bondsList[2]
        upper = self.__bondsList[3]
        categories = {}
        for idx in xrange(self.__bondsList[0].shape[0]):
            #if self._atomsCollector.is_collected(idx):
            #    continue
            if self._atomsCollector.is_collected(atom1[idx]):
                continue
            if self._atomsCollector.is_collected(atom2[idx]):
                continue
            if splitBy is not None:
                a1 = splitBy[ atom1[idx] ]
                a2 = splitBy[ atom2[idx] ]
            else:
                a1 = a2 = ''
            l  = lower[idx]
            u  = upper[idx]
            k  = (a1,a2,l,u)
            L  = categories.get(k, [])
            L.append(idx)
            categories[k] = L
        ncategories = len(categories)
        # start plotting
        COLORS  = ["b",'g','r','c','y','m']
        catKeys = sorted(categories, key=lambda x:x[2])
        shifts  = [0]
        xticks   = []
        xticksL  = []
        yticks   = []
        yticksL  = []
        ticksCol = []
        for idx, key in enumerate(catKeys):
            a1,a2, L,U  = key
            label  = "%s%s%s(%.2f,%.2f)"%(a1,'-'*(len(a1)>0),a2,L,U)
            col    = COLORS[idx%len(COLORS)]
            idxs   = categories[key]
            catd   = data["bondsLength"][idxs]
            dmin   = np.min(catd)
            dmax   = np.max(catd)
            # append xticks labels
            dmint = dmaxt = []
            if dmin<L:
                dmint = [dmin]
            if dmax>U:
                dmaxt = [dmax]
            xticksL.extend( dmint + list(np.linspace(start=L,stop=U,num=numberOfTicks, endpoint=True)) + dmaxt )
            # rescale histogram
            resc = dsh = 0
            if stackHorizontal:
                resc   = min(L,np.min(catd)) - spacing # rescale to origin + spacing
                catd  -= resc - shifts[-1] # shift to stack to the right of the last histogram
                dmin   = np.min(catd)
                dmax   = np.max(catd)
                dmint = dmaxt = []
                L     -= resc - shifts[-1]
                U     -= resc - shifts[-1]
                dsh    = shifts[-1]
            # append ticks positions
            if len(dmint):
                dmint = [dmin-resc+dsh]
            if len(dmaxt):
                dmaxt = [dmax-resc+dsh]
            xticks.extend( dmint + list(np.linspace(start=L,stop=U,num=numberOfTicks, endpoint=True)) + dmaxt )
            # append shifts
            if stackHorizontal:
                shifts.append(max(dmax,U))
                bottom = None
            else:
                bottom = shifts[-1]
            # get data limits
            #bins = _get_bins(dmin=dmin, dmax=dmax, boundaries=[L,U], nbins=nbins)
            bins  = list(np.linspace(start=min(dmin,L),stop=max(dmax,U),num=nbins, endpoint=True))
            D, _, P = ax.hist(x=catd, bins=bins,rwidth=barsRelativeWidth,
                              bottom=bottom, color=col, label=label,
                              histtype='bar')
            # vertical lines
            lmp = limitsParams
            if lmp.get('color',None) is None:
                lmp = copy.deepcopy(lmp)
                lmp['color'] = col
            Y = max(D)
            B = 0 if stackHorizontal else shifts[-1]
            ax.plot([L,L],[B,B+Y+0.1*Y], **lmp)
            ax.plot([U,U],[B,B+Y+0.1*Y], **lmp)
            if not stackHorizontal:
                shifts.append(shifts[-1]+Y+0.1*Y)
                yticks.append(bottom+Y/2)
                yticksL.append(Y/2)
                yticks.append(B+Y)
                yticksL.append(Y)
            # adapt ticks color
            ticksCol.extend([col]*(len(xticksL)-len(ticksCol)))
        # update ticks
        ax.set_xticks(xticks)
        ax.set_xticklabels( ['%.2f'%t for t in xticksL], **xticksParams)
        if not stackHorizontal:
            ax.set_yticks(yticks)
            ax.set_yticklabels( ['%i'%t for t in yticksL], **yticksParams)
        else:
            ax.set_yticklabels( ['%i'%t for t in ax.get_yticks()], **yticksParams)
        if colorCodeXticksLabels:
            for ticklabel, tickcolor in zip(ax.get_xticklabels(), ticksCol):
                ticklabel.set_color(tickcolor)
        # plot legend
        if legendParams is not None:
            ax.legend(**legendParams)
        # grid parameters
        if gridParams is not None:
            gp = copy.deepcopy(gridParams)
            axis = gp.pop('axis', 'both')
            if axis is None:
                axis = 'x' if stackHorizontal else 'y'
            ax.grid(axis=axis, **gp)
        # set axis labels
        ax.set_xlabel(**xlabelParams)
        ax.set_ylabel(**ylabelParams)
        # set title
        if titleParams is not None:
            title = copy.deepcopy(titleParams)
            label = title.pop('label',"").format(frame=frame,standardError=standardError, numberOfRemovedAtoms=numberOfRemovedAtoms,used=self.used)
            ax.set_title(label=label, **title)



    def plot(self, spacing=0.1, numberOfTicks=3, nbins=20, barsRelativeWidth=0.95,
                   splitBy=None, stackHorizontal=True, colorCodeXticksLabels=True,
                   xlabelParams={'xlabel':'$r(\\AA)$', 'size':10},
                   ylabelParams={'ylabel':'number', 'size':10},
                   limitsParams={'linewidth':1.0, 'color':None, 'linestyle':'--'},
                   **kwargs):

         """
         Alias to Constraint.plot with additional parameters

         :Additional/Adjusted Parameters:
             #. spacing (float): spacing between definitions histgrams
             #. numberOfTicks (integer): number of ticks per definition histogram
             #. nbins (integer): number of bins per definition histogram
             #. barsRelativeWidth (float): histogram bar relative width >0 and <1
             #. splitBy (None, string): Split definition histograms by atom
                element, name or merely distance. accepts None, 'element', 'name'
             #. stackHorizontal (boolean): whether to stack definition plots
                horizontally or veritcally
             #. colorCodeXticksLabels (boolean): whether to color code x ticks
                per definition color
             #. xlabelParams (None, dict): modified matplotlib.axes.Axes.set_xlabel
                parameters.
             #. ylabelParams (None, dict): modified matplotlib.axes.Axes.set_ylabel
                parameters.
             #. titleParams (None, dict): axes title parameters
         """
         return super(BondConstraint, self).plot(spacing=spacing, nbins=nbins,
                                                 numberOfTicks=numberOfTicks,
                                                 splitBy=splitBy,
                                                 stackHorizontal=stackHorizontal,
                                                 colorCodeXticksLabels=colorCodeXticksLabels,
                                                 barsRelativeWidth=barsRelativeWidth,
                                                 limitsParams=limitsParams,
                                                 xlabelParams=xlabelParams,
                                                 ylabelParams=ylabelParams,
                                                 **kwargs)

    def _constraint_copy_needs_lut(self):
        return {'_BondConstraint__bondsDefinition'  :'_BondConstraint__bondsDefinition',
                '_BondConstraint__bondsList'        :'_BondConstraint__bondsList',
                '_BondConstraint__bonds'            :'_BondConstraint__bonds',
                '_Constraint__used'                 :'_Constraint__used',
                '_Constraint__data'                 :'_Constraint__data',
                '_Constraint__standardError'        :'_Constraint__standardError',
                '_Constraint__state'                :'_Constraint__state',
                '_Engine__state'                    :'_Engine__state',
                '_Engine__boxCoordinates'           :'_Engine__boxCoordinates',
                '_Engine__basisVectors'             :'_Engine__basisVectors',
                '_Engine__isPBC'                    :'_Engine__isPBC',
                '_Engine__moleculesIndex'           :'_Engine__moleculesIndex',
                '_Engine__elementsIndex'            :'_Engine__elementsIndex',
                '_Engine__numberOfAtomsPerElement'  :'_Engine__numberOfAtomsPerElement',
                '_Engine__elements'                 :'_Engine__elements',
                '_Engine__numberDensity'            :'_Engine__numberDensity',
                '_Engine__volume'                   :'_Engine__volume',
                '_atomsCollector'                   :'_atomsCollector',
                ('engine','_atomsCollector')        :'_atomsCollector',
               }


    def _get_export(self, frameIndex, propertiesLUT, format='%s'):
        # create data, metadata and header
        frame = propertiesLUT['frames-name'][frameIndex]
        data  = propertiesLUT['frames-data'][frameIndex]
        # compute categories
        names    = self.engine.get_original_data("allNames", frame=frame)
        elements = self.engine.get_original_data("allElements", frame=frame)
        atom1    = self.__bondsList[0]
        atom2    = self.__bondsList[1]
        lower    = self.__bondsList[2]
        upper    = self.__bondsList[3]
        consData = data["bondsLength"]
        header = ['atom_1_index',   'atom_2_index',
                  'atom_1_element', 'atom_2_element',
                  'atom_1_name',     'atom_2_name',
                  'lower_limit', 'upper_limit', 'value']
        data = []
        for idx in xrange(self.__bondsList[0].shape[0]):
            #if self._atomsCollector.is_collected(idx):
            #    continue
            if self._atomsCollector.is_collected(atom1[idx]):
                continue
            if self._atomsCollector.is_collected(atom2[idx]):
                continue
            data.append([str(atom1[idx]),str(atom2[idx]),
                             elements[atom1[idx]],elements[atom2[idx]],
                             names[atom1[idx]],names[atom2[idx]],
                             format%lower[idx], format%upper[idx],
                             format%consData[idx]] )
        # save
        return header, data
