"""
AngleConstraints contains classes for all constraints related angles
between atoms.

.. inheritance-diagram:: fullrmc.Constraints.AngleConstraints
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
from ..Core.Collection import is_number, raise_if_collected, reset_if_collected_out_of_date
from ..Core.Collection import get_caller_frames
from ..Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from ..Core.angles import full_angles_coords




class BondsAngleConstraint(RigidConstraint, SingularConstraint):
    """
    Controls angle defined between 3 defined atoms, a first atom called central
    and the remain two called left and right.


    +-------------------------------------------------------------------------+
    |.. figure:: angleSketch.png                                              |
    |   :width: 308px                                                         |
    |   :height: 200px                                                        |
    |   :align: center                                                        |
    |                                                                         |
    |   Angle sketch defined between three atoms.                             |
    +-------------------------------------------------------------------------+

    .. raw:: html

        <iframe width="560" height="315"
        src="https://www.youtube.com/embed/ezBbbO9IVig"
        frameborder="0" allowfullscreen>
        </iframe>


    +----------------------------------------------------------------------+
    |.. figure:: bonds_angle_constraint_plot_method.png                    |
    |   :width: 530px                                                      |
    |   :height: 400px                                                     |
    |   :align: left                                                       |
    +----------------------------------------------------------------------+


    :Parameters:
        #. rejectProbability (Number): Rejecting probability of all steps where
           standardError increases. It must be between 0 and 1 where 1 means
           rejecting all steps where standardError increases and 0 means
           accepting all steps regardless whether standardError increases or
           not.

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
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb('system.pdb')

        # create and add constraint
        BAC = BondsAngleConstraint()
        ENGINE.add_constraints(BAC)

        # define intra-molecular angles
        BAC.create_angles_by_definition( anglesDefinition={"CH4": [ ('C','H1','H2', 100, 120),
                                                                    ('C','H2','H3', 100, 120),
                                                                    ('C','H3','H4', 100, 120),
                                                                    ('C','H4','H1', 100, 120) ]} )


    """
    def __init__(self, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, rejectProbability=rejectProbability)
        # set atomsColletor data keys
        self._atomsCollector.set_data_keys( ['centralMap','otherMap'] )
        # init angles data
        self.__anglesDefinition = None
        self.__anglesList       = [[],[],[],[],[]]
        self.__angles           = {}
        # set computation cost
        self.set_computation_cost(2.0)
        # create dump flag
        self.__dumpAngles = True
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_BondsAngleConstraint__anglesDefinition',
                           '_BondsAngleConstraint__anglesList',
                           '_BondsAngleConstraint__angles',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',  tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA',tuple(RUNTIME_DATA) )

    def _codify_update__(self, name='constraint', addDependencies=True):
        dependencies = []
        code         = []
        if addDependencies:
            code.extend(dependencies)
        code.append("{name}.set_used({val})".format(name=name, val=self.used))
        code.append("{name}.set_reject_probability({val})".format(name=name, val=self.rejectProbability))
        if self.anglesDefinition is not None:
            code.append("{name}.create_angles_by_definition({val})".format(name=name, val=self.anglesDefinition))
        else:
            angles = self.anglesList
            angles = [angles[0],angles[1],angles[2], FLOAT_TYPE(180)*angles[3]/PI,FLOAT_TYPE(180)*angles[4]/PI]
            code.append("angles = {val}".format(val=angles))
            code.append("{name}.set_angles(angles)".format(name=name, val=angles))
        # return
        return dependencies, '\n'.join(code)

    def _codify__(self, engine, name='constraint', addDependencies=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        dependencies = ['from fullrmc.Constraints import AngleConstraints']
        code         = []
        if addDependencies:
            code.extend(dependencies)
        code.append("{name} = AngleConstraints.BondsAngleConstraint\
(rejectProbability={rejectProbability})".format(name=name, rejectProbability=self.rejectProbability))
        code.append("{engine}.add_constraints([{name}])".format(engine=engine, name=name))
        if self.__anglesDefinition is not None:
            code.append("{name}.create_angles_by_definition({angles})".
            format(name=name, angles=self.__anglesDefinition))
        elif len(self.__anglesList[0]):
            angles = self.__anglesList
            angles = [angles[0],angles[1],angles[2], FLOAT_TYPE(180.)*angles[3]/PI, FLOAT_TYPE(180.)*angles[4]/PI]
            code.append("{name}.set_angles({angles})".
            format(name=name, angles=angles))
        # return
        return dependencies, '\n'.join(code)

    @property
    def anglesList(self):
        """ Defined angles list."""
        return self.__anglesList

    @property
    def anglesDefinition(self):
        """angles definition copy if angles are defined as such"""
        return copy.deepcopy(self.__anglesDefinition)

    @property
    def angles(self):
        """ Angles dictionary of every and each atom."""
        return self.__angles

    def _on_collector_reset(self):
        self._atomsCollector._randomData = set([])

    def listen(self, message, argument=None):
        """
        Listen to any message sent from the Broadcaster.

        :Parameters:
            #. message (object): Any python object to send to constraint's
               listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in ("engine set","update pdb","update molecules indexes","update elements indexes","update names indexes"):
            if self.__anglesDefinition is not None:
                self.create_angles_by_definition(self.__anglesDefinition)
            else:
                # set angles and reset constraint
                AL = [ self.__anglesList[0],self.__anglesList[1],self.__anglesList[2],
                       [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[3]],
                       [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[4]] ]
                self.set_angles( AL, tform=False )
        elif message in ("update boundary conditions",):
            # reset constraint
            self.reset_constraint()


    def set_angles(self, anglesList, tform=True):
        """
        Sets the angles dictionary by parsing the anglesList. All angles are in
        degrees.

        :Parameters:
            #. anglesList (list): The angles list definition that can be
               given in two different formats.

               tuples format:  every item must be a list of five items.\n
               #. Central atom index.
               #. Index of the left atom forming the angle
                  (interchangeable with the right atom).
               #. Index of the right atom forming the angle
                  (interchangeable with the left atom).
               #. Minimum lower limit or the minimum angle
                  allowed in degrees which later will be converted to rad.
               #. Maximum upper limit or the maximum angle
                  allowed in degrees which later will be converted to rad.

               five vectors format:  List of exaclty five lists or
               numpy.arrays or vectors of the same length.\n
               #. List containing central atom indexes.
               #. List containing the index of the left atom
                  forming the angle (interchangeable with the right atom).
               #. List containing the index of the right atom
                  forming the angle (interchangeable with the left atom).
               #. List containing the minimum lower limit or the
                  minimum angle allowed in degrees which later will be
                  converted to rad.
               #. List containing the maximum upper limit or the
                  maximum angle allowed in degrees which later will be
                  converted to rad.

           #. tform (boolean): set whether given anglesList follows tuples
              format, If False, then it must follow the five vectors one.
        """
        assert self.engine is not None, LOGGER.error("setting angles is not allowed unless engine is defined.")
        assert isinstance(anglesList, (list,set,tuple)), "anglesList must be a list"
        # convert to list of tuples
        if not tform:
            assert len(anglesList) == 5, LOGGER.error("non tuple form anglesList must be a list of 5 items")
            assert all([isinstance(i, (list,tuple,np.ndarray)) for i in anglesList]), LOGGER.error("non tuple form anglesList must be a list of list or tuple or numpy.ndarray")
            assert all([len(i)==len(anglesList[0]) for i in anglesList]), LOGGER.error("anglesList items list length mismatch")
            anglesList = zip(*anglesList)
        # get number of atoms
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        # loop angles
        anglesL = [[],[],[],[],[]]
        angles  = {}
        tempA   = {}
        for a in anglesList:
            assert isinstance(a, (list, set, tuple)), LOGGER.error("angle item must be list")
            assert len(a) == 5, LOGGER.error("angle items must be lists of 5 items each")
            centralIdx, leftIdx, rightIdx, lower, upper = a
            assert centralIdx<NUMBER_OF_ATOMS, LOGGER.error("angle atom index must be smaller than maximum number of atoms")
            assert leftIdx<NUMBER_OF_ATOMS, LOGGER.error("angle atom index must be smaller than maximum number of atoms")
            assert rightIdx<NUMBER_OF_ATOMS, LOGGER.error("angle atom index must be smaller than maximum number of atoms")
            centralIdx = INT_TYPE(centralIdx)
            leftIdx    = INT_TYPE(leftIdx)
            rightIdx   = INT_TYPE(rightIdx)
            assert is_number(lower)
            lower = FLOAT_TYPE(lower)
            assert is_number(upper)
            upper = FLOAT_TYPE(upper)
            assert lower>=0, LOGGER.error("angle items lists fourth item must be positive")
            assert upper>lower, LOGGER.error("angle items lists fourth item must be smaller than the fifth item")
            assert upper<=180, LOGGER.error("angle items lists fifth item must be smaller or equal to 180")
            lower *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
            upper *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
            # check for redundancy
            assert (centralIdx,leftIdx,rightIdx) not in tempA, LOGGER.error("redundant definition of angles between central atom '%i' and left atom '%i' and right atom '%i'"%(centralIdx,leftIdx,rightIdx))
            assert (centralIdx,rightIdx,leftIdx) not in tempA, LOGGER.error("redundant definition of angles between central atom '%i' and left atom '%i' and right atom '%i'"%(centralIdx,leftIdx,rightIdx))
            tempA[(centralIdx,leftIdx,rightIdx)] = True
            # create central angle
            if not centralIdx in angles:
                anglesCentral = {"left":[],"right":[],"centralMap":[],"otherMap":[]}
            else:
                anglesCentral = {"left"       :angles[centralIdx]["left"],
                                 "right"      :angles[centralIdx]["right"],
                                 "centralMap" :angles[centralIdx]["centralMap"],
                                 "otherMap"   :angles[centralIdx]["otherMap"] }
            # create left angle
            if not leftIdx in angles:
                anglesLeft = {"left":[],"right":[],"centralMap":[],"otherMap":[]}
            else:
                anglesLeft = {"left"       :angles[leftIdx]["left"],
                              "right"      :angles[leftIdx]["right"],
                              "centralMap" :angles[leftIdx]["centralMap"],
                              "otherMap"   :angles[leftIdx]["otherMap"] }
            # create right angle
            if not rightIdx in angles:
                anglesRight = {"left":[],"right":[],"centralMap":[],"otherMap":[]}
            else:
                anglesRight = {"left"       :angles[rightIdx]["left"],
                               "right"      :angles[rightIdx]["right"],
                               "centralMap" :angles[rightIdx]["centralMap"],
                               "otherMap"   :angles[rightIdx]["otherMap"] }
            # set angle
            anglesCentral['left'].append(leftIdx)
            anglesCentral['right'].append(rightIdx)
            anglesCentral['centralMap'].append( len(anglesL[0]) )
            anglesLeft['otherMap'].append( len(anglesL[0]) )
            anglesRight['otherMap'].append( len(anglesL[0]) )
            anglesL[0].append(centralIdx)
            anglesL[1].append(leftIdx)
            anglesL[2].append(rightIdx)
            anglesL[3].append(lower)
            anglesL[4].append(upper)
            angles[centralIdx] = anglesCentral
            angles[leftIdx]    = anglesLeft
            angles[rightIdx]   = anglesRight
        # finalize angles
        for idx in xrange(NUMBER_OF_ATOMS):
            angles[INT_TYPE(idx)] = angles.get(INT_TYPE(idx),  {"left":[],"right":[],"centralMap":[],"otherMap":[]}  )
        # set angles
        self.__angles           = angles
        self.__anglesList       = [np.array(anglesL[0], dtype=INT_TYPE),
                                   np.array(anglesL[1], dtype=INT_TYPE),
                                   np.array(anglesL[2], dtype=INT_TYPE),
                                   np.array(anglesL[3], dtype=FLOAT_TYPE),
                                   np.array(anglesL[4], dtype=FLOAT_TYPE),]
        self.__anglesDefinition = None
        # dump to repository
        if self.__dumpAngles:
            self._dump_to_repository({'_BondsAngleConstraint__anglesDefinition':self.__anglesDefinition,
                                      '_BondsAngleConstraint__anglesList'      :self.__anglesList,
                                      '_BondsAngleConstraint__angles'          :self.__angles})
            # reset constraint
            self.reset_constraint()


    #@raise_if_collected
    def create_angles_by_definition(self, anglesDefinition):
        """
        Creates anglesList using angles definition.
        Calls set_angles(anglesList) and generates angles attribute.

        :Parameters:
            #. anglesDefinition (dict): Angles definition dictionary.
               Every key must be a molecule's name.
               Every key value must be a list of angles definitions.
               Every angle definition is a list of five items where:

               #. Name of the central atom forming the angle.
               #. Name of the left atom forming the angle
                  (interchangeable with the right atom).
               #. Name of the right atom forming the angle
                  (interchangeable with the left atom).
               #. Minimum lower limit or the minimum angle
                  allowed in degrees which later will be converted to rad.
               #. Maximum upper limit or the maximum angle
                  allowed in degrees which later will be converted to rad.

        ::

            e.g. (Carbon tetrachloride):  anglesDefinition={"CCL4": [('C','CL1','CL2' , 105, 115),
                                                                     ('C','CL2','CL3' , 105, 115),
                                                                     ('C','CL3','CL4' , 105, 115),
                                                                     ('C','CL4','CL1' , 105, 115) ] }

        """
        if self.engine is None:
            raise Exception(LOGGER.error("Engine is not defined. Can't create angles by definition"))
        assert isinstance(anglesDefinition, dict), LOGGER.error("anglesDefinition must be a dictionary")
        ALL_NAMES       = self.engine.get_original_data("allNames")
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        MOLECULES_NAME  = self.engine.get_original_data("moleculesName")
        MOLECULES_INDEX = self.engine.get_original_data("moleculesIndex")
        # check map definition
        existingmoleculesName = sorted(set(MOLECULES_NAME))
        anglesDef = {}
        for mol in anglesDefinition:
            angles = anglesDefinition[mol]
            if mol not in existingmoleculesName:
                LOGGER.usage("Molecule name '%s' in anglesDefinition is not recognized, angles definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(angles, (list, set, tuple)), LOGGER.error("mapDefinition molecule angles must be a list")
            angles = list(angles)
            molAnglesList = []
            for angle in angles:
                assert isinstance(angle, (list, set, tuple)), LOGGER.error("mapDefinition angles must be a list")
                angle = list(angle)
                assert len(angle)==5
                centralAt, leftAt, rightAt, lower, upper = angle
                # check for redundancy
                append = True
                for b in molAnglesList:
                    if (b[0]==centralAt) and ( (b[1]==leftAt and b[2]==rightAt) or (b[1]==rightAt and b[2]==leftAt) ):
                        LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
                        append = False
                        break
                if append:
                    molAnglesList.append((centralAt, leftAt, rightAt, lower, upper))
            # create bondDef for molecule mol
            anglesDef[mol] = molAnglesList
        # create mols dictionary
        mols = {}
        for idx in xrange(NUMBER_OF_ATOMS):
            molName = MOLECULES_NAME[idx]
            if not molName in anglesDef:
                continue
            molIdx = MOLECULES_INDEX[idx]
            if not molIdx in mols:
                mols[molIdx] = {"name":molName, "indexes":[], "names":[]}
            mols[molIdx]["indexes"].append(idx)
            mols[molIdx]["names"].append(ALL_NAMES[idx])
        # get anglesList
        anglesList = []
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
                anglesList.append((centralIdx, leftIdx, rightIdx, lower, upper))
        # create angles
        self.__dumpAngles = False
        try:
            self.set_angles(anglesList=anglesList)
        except Exception as err:
            self.__dumpAngles = True
            raise Exception(err)
        else:
            self.__dumpAngles = True
            self.__anglesDefinition = anglesDefinition
            self._dump_to_repository({'_BondsAngleConstraint__anglesDefinition':self.__anglesDefinition,
                                      '_BondsAngleConstraint__anglesList'      :self.__anglesList,
                                      '_BondsAngleConstraint__angles'          :self.__angles})
            # reset constraint
            self.reset_constraint()

    def compute_standard_error(self, data):
        """
        Compute the standard error (StdErr) of data not satisfying constraint
        conditions.

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
            #. data (numpy.array): Constraint's data to compute standardError.

        :Returns:
            #. standardError (number): The calculated standardError of the
               given data.
        """
        return FLOAT_TYPE( np.sum(data["reducedAngles"]**2) )

    def get_constraint_value(self):
        """
        Get partial Mean Pair Distances (MPD) below the defined
        minimum distance.

        :Returns:
            #. MPD (dictionary): MPD dictionary, where keys are the
               element wise intra and inter molecular MPDs and values are
               the computed MPDs.
        """
        if self.data is None:
            LOGGER.warn("%s data must be computed first using 'compute_data' method."%(self.__class__.__name__))
            return {}
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
            anglesData  = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            reducedData = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            #anglesIndex = set(set(range(self.__anglesList[0].shape[0])))
            anglesIndex = set(range(self.__anglesList[0].shape[0]))
            anglesIndex = list( anglesIndex-self._atomsCollector._randomData )
            central = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndex])
            left    = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndex])
            right   = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndex])
            lowerLimit = self.__anglesList[3][anglesIndex]
            upperLimit = self.__anglesList[4][anglesIndex]
        else:
            central = self._atomsCollector.get_relative_indexes(self.__anglesList[0])
            left    = self._atomsCollector.get_relative_indexes(self.__anglesList[1])
            right   = self._atomsCollector.get_relative_indexes(self.__anglesList[2])
            lowerLimit = self.__anglesList[3]
            upperLimit = self.__anglesList[4]
        # compute data
        angles, reduced = full_angles_coords(central            = central,
                                             left               = left,
                                             right              = right,
                                             lowerLimit         = lowerLimit,
                                             upperLimit         = upperLimit,
                                             boxCoords          = self.engine.boxCoordinates,
                                             basis              = self.engine.basisVectors,
                                             isPBC              = self.engine.isPBC,
                                             reduceAngleToUpper = False,
                                             reduceAngleToLower = False,
                                             ncores             = INT_TYPE(1))
        # create full length data
        if len(self._atomsCollector):
            anglesData[anglesIndex]  = angles
            reducedData[anglesIndex] = reduced
            angles  = anglesData
            reduced = reducedData
        # create data and compute standard error
        data     = {"angles":angles, "reducedAngles":reduced}
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
        Compute constraint before move is executed.

        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms index the move will
               be applied to.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        # get angles indexes
        anglesIndex = []
        #for idx in relativeIndexes:
        for idx in realIndexes:
            anglesIndex.extend( self.__angles[idx]['centralMap'] )
            anglesIndex.extend( self.__angles[idx]['otherMap'] )
        anglesIndex = list( set(anglesIndex)-self._atomsCollector._randomData )
        # compute data before move
        if len(anglesIndex):
            angles, reduced =  full_angles_coords( central            = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndex]),
                                                   left               = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndex]),
                                                   right              = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndex]),
                                                   lowerLimit         = self.__anglesList[3][anglesIndex],
                                                   upperLimit         = self.__anglesList[4][anglesIndex],
                                                   boxCoords          = self.engine.boxCoordinates,
                                                   basis              = self.engine.basisVectors ,
                                                   isPBC              = self.engine.isPBC,
                                                   reduceAngleToUpper = False,
                                                   reduceAngleToLower = False,
                                                   ncores             = INT_TYPE(1))
        else:
            angles  = None
            reduced = None
        # set data before move
        self.set_active_atoms_data_before_move( {"anglesIndex":anglesIndex, "angles":angles, "reducedAngles":reduced} )
        self.set_active_atoms_data_after_move(None)

    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """
        Compute constraint after move is executed.

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Group atoms relative index
               the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        # get angles indexes
        anglesIndex = self.activeAtomsDataBeforeMove["anglesIndex"]
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[relativeIndexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[relativeIndexes] = movedBoxCoordinates
        # compute data before move
        if len(anglesIndex):
            angles, reduced =  full_angles_coords( central            = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndex]),
                                                   left               = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndex]),
                                                   right              = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndex]),
                                                   lowerLimit         = self.__anglesList[3][anglesIndex],
                                                   upperLimit         = self.__anglesList[4][anglesIndex],
                                                   boxCoords          = self.engine.boxCoordinates,
                                                   basis              = self.engine.basisVectors ,
                                                   isPBC              = self.engine.isPBC,
                                                   reduceAngleToUpper = False,
                                                   reduceAngleToLower = False,
                                                   ncores             = INT_TYPE(1))
        else:
            angles  = None
            reduced = None
        # set active data after move
        self.set_active_atoms_data_after_move( {"angles":angles, "reducedAngles":reduced} )
        # reset coordinates
        self.engine.boxCoordinates[relativeIndexes] = boxData
        # compute standardError after move
        if angles is None:
            self.set_after_move_standard_error( self.standardError )
        else:
            # anglesIndex is a fancy slicing, RL is a copy not a view.
            RL = self.data["reducedAngles"][anglesIndex]
            self.data["reducedAngles"][anglesIndex] += reduced-self.activeAtomsDataBeforeMove["reducedAngles"]
            self.set_after_move_standard_error( self.compute_standard_error(data = self.data) )
            self.data["reducedAngles"][anglesIndex] = RL
        # increment tried
        self.increment_tried()

    def accept_move(self, realIndexes, relativeIndexes):
        """
        Accept move.

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        # get indexes
        anglesIndex = self.activeAtomsDataBeforeMove["anglesIndex"]
        if len(anglesIndex):
            # set new data
            data = self.data
            data["angles"][anglesIndex]        += self.activeAtomsDataAfterMove["angles"]-self.activeAtomsDataBeforeMove["angles"]
            data["reducedAngles"][anglesIndex] += self.activeAtomsDataAfterMove["reducedAngles"]-self.activeAtomsDataBeforeMove["reducedAngles"]
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
        Reject move.

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)

    def accept_amputation(self, realIndex, relativeIndex):
        """
        Accept amputation of atom and set constraint's data and
        standard error accordingly.

        :Parameters:
            #. realIndex (numpy.ndarray): Atom's index as a numpy array
               of a single element.
            #. relativeIndex (numpy.ndarray): Not used here.
        """
        # MAYBE WE DON"T NEED TO CHANGE DATA AND SE. BECAUSE THIS MIGHT BE A PROBLEM
        # WHEN IMPLEMENTING ATOMS RELEASING. MAYBE WE NEED TO COLLECT DATA INSTEAD, REMOVE
        # AND ADD UPON RELEASE
        # get all involved data
        anglesIndex = []
        for idx in realIndex:
            anglesIndex.extend( self.__angles[idx]['centralMap'] )
            anglesIndex.extend( self.__angles[idx]['otherMap'] )
        anglesIndex = list(set(anglesIndex))
        if len(anglesIndex):
            # set new data
            data = self.data
            data["angles"][anglesIndex]        = 0
            data["reducedAngles"][anglesIndex] = 0
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
        # get angle indexes
        AI = self.__angles[realIndex]['centralMap'] + self.__angles[realIndex]['otherMap']
        # append all mapped bonds to collector's random data
        self._atomsCollector._randomData = self._atomsCollector._randomData.union( set(AI) )
        # collect atom anglesIndex
        self._atomsCollector.collect(realIndex, dataDict={'centralMap':self.__angles[realIndex]['centralMap'],
                                                          'otherMap':  self.__angles[realIndex]['otherMap']})

    def _plot(self,frameIndex, propertiesLUT,
                   spacing,numberOfTicks,nbins,splitBy,
                   ax, barsRelativeWidth,limitsParams,
                   legendParams,titleParams,
                   xticksParams, yticksParams,
                   xlabelParams, ylabelParams,
                   gridParams, stackHorizontal,
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
        if not len(self.__anglesList[0]):
            LOGGER.warn("@{frm} no angles found. It's even not defined or no atoms where found in definition.".format(frm=frame))
            return
        # build categories
        atom2 = self.__anglesList[0]
        atom1 = self.__anglesList[1]
        atom3 = self.__anglesList[2]
        lower = self.__anglesList[3]
        upper = self.__anglesList[4]
        categories = {}
        for idx in xrange(self.__anglesList[0].shape[0]):
            #if self._atomsCollector.is_collected(idx):
            #    continue
            if self._atomsCollector.is_collected(atom1[idx]):
                continue
            if self._atomsCollector.is_collected(atom2[idx]):
                continue
            if self._atomsCollector.is_collected(atom3[idx]):
                continue
            if splitBy is not None:
                a1 = splitBy[ atom1[idx] ]
                a2 = splitBy[ atom2[idx] ]
                a3 = splitBy[ atom3[idx] ]
            else:
                a1 = a2 = a3 = ''
            l = lower[idx]
            u = upper[idx]
            k = (a1,a2,a3,l,u)
            L = categories.get(k, [])
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
            a1,a2,a3, L,U  = key
            L     = L*180./np.pi
            U     = U*180./np.pi
            label = "%s%s%s%s%s(%.2f,%.2f)"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,L,U)
            col   = COLORS[idx%len(COLORS)]
            idxs  = categories[key]
            catd  = data["angles"][idxs]*180./np.pi
            dmin  = np.min(catd)
            dmax  = np.max(catd)
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
            # append xticks positions
            if len(dmint):
                dmint = [dmin-resc+dsh]
            if len(dmaxt):
                dmaxt = [dmax-resc+dsh]
            xticks.extend( dmint + list(np.linspace(start=L,stop=U,num=numberOfTicks, endpoint=True)) + dmaxt )
            # append shifts
            if stackHorizontal:
                shifts.append(max(dmax,U))
                bottom = 0
            else:
                bottom = shifts[-1]
            # get data limits
            #bins = _get_bins(dmin=dmin, dmax=dmax, boundaries=[L,U], nbins=nbins)
            bins  = list(np.linspace(start=min(dmin,L),stop=max(dmax,U),num=nbins, endpoint=True))
            D, _, P = ax.hist(x=catd, bins=bins,rwidth=barsRelativeWidth,
                              color=col, label=label,
                              bottom=bottom, histtype='bar')
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


    def plot(self, spacing=2, numberOfTicks=2, nbins=20, barsRelativeWidth=0.95,
                   splitBy=None, stackHorizontal=True, colorCodeXticksLabels=True,
                   xlabelParams={'xlabel':'$deg.$', 'size':10},
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
         return super(BondsAngleConstraint, self).plot(spacing=spacing, nbins=nbins,
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
        return {'_BondsAngleConstraint__anglesDefinition':'_BondsAngleConstraint__anglesDefinition',
                '_BondsAngleConstraint__anglesList'      :'_BondsAngleConstraint__anglesList',
                '_BondsAngleConstraint__angles'          :'_BondsAngleConstraint__angles',
                '_Constraint__used'                      :'_Constraint__used',
                '_Constraint__data'                      :'_Constraint__data',
                '_Constraint__standardError'             :'_Constraint__standardError',
                '_Constraint__state'                     :'_Constraint__state',
                '_Engine__state'                         :'_Engine__state',
                '_Engine__boxCoordinates'                :'_Engine__boxCoordinates',
                '_Engine__basisVectors'                  :'_Engine__basisVectors',
                '_Engine__isPBC'                         :'_Engine__isPBC',
                '_Engine__moleculesIndex'                :'_Engine__moleculesIndex',
                '_Engine__elementsIndex'                 :'_Engine__elementsIndex',
                '_Engine__numberOfAtomsPerElement'       :'_Engine__numberOfAtomsPerElement',
                '_Engine__elements'                      :'_Engine__elements',
                '_Engine__numberDensity'                 :'_Engine__numberDensity',
                '_Engine__volume'                        :'_Engine__volume',
                '_atomsCollector'                        :'_atomsCollector',
                ('engine','_atomsCollector')             :'_atomsCollector',
               }

    def _get_export(self, frameIndex, propertiesLUT, format='%s'):
        # create data, metadata and header
        frame = propertiesLUT['frames-name'][frameIndex]
        data  = propertiesLUT['frames-data'][frameIndex]
        # compute categories
        names    = self.engine.get_original_data("allNames", frame=frame)
        elements = self.engine.get_original_data("allElements", frame=frame)
        atom2 = self.__anglesList[0]
        atom1 = self.__anglesList[1]
        atom3 = self.__anglesList[2]
        lower = self.__anglesList[3]*180./np.pi
        upper = self.__anglesList[4]*180./np.pi
        consData = data["angles"]*180./np.pi
        header = ['atom_1_index',   'atom_2_index', 'atom_3_index',
                  'atom_1_element', 'atom_2_element', 'atom_3_element',
                  'atom_1_name',     'atom_2_name', 'atom_3_name',
                  'lower_limit', 'upper_limit', 'value']
        data = []
        for idx in xrange(self.__anglesList[0].shape[0]):
            #if self._atomsCollector.is_collected(idx):
            #    continue
            if self._atomsCollector.is_collected(atom1[idx]):
                continue
            if self._atomsCollector.is_collected(atom2[idx]):
                continue
            if self._atomsCollector.is_collected(atom3[idx]):
                continue
            data.append([str(atom1[idx]),str(atom2[idx]),str(atom3[idx]),
                             elements[atom1[idx]],elements[atom2[idx]],elements[atom3[idx]],
                             names[atom1[idx]],names[atom2[idx]],names[atom3[idx]],
                             format%lower[idx], format%upper[idx],
                             format%consData[idx]] )
        # save
        return header, data
