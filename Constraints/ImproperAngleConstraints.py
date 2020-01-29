"""
ImproperAngleConstraints contains classes for all constraints related to
improper angles between atoms.

.. inheritance-diagram:: fullrmc.Constraints.ImproperAngleConstraints
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
from ..Core.improper_angles import full_improper_angles_coords




class ImproperAngleConstraint(RigidConstraint, SingularConstraint):
    """
    Controls the improper angle formed with 4 defined atoms. It's mainly used
    to keep the improper atom in the plane defined with three other atoms.
    The improper vector is defined as the vector from the first atom of the
    plane to the improper atom. Therefore the improper angle is defined between
    the improper vector and the plane.


    +--------------------------------------------------------------------------+
    |.. figure:: improperSketch.png                                            |
    |   :width: 269px                                                          |
    |   :height: 200px                                                         |
    |   :align: center                                                         |
    |                                                                          |
    |   Improper angle sketch defined between four atoms.                      |
    +--------------------------------------------------------------------------+


     .. raw:: html

        <iframe width="560" height="315"
        src="https://www.youtube.com/embed/qVATE-9cIBg"
        frameborder="0" allowfullscreen>
        </iframe>

    :Parameters:
        #. rejectProbability (Number): rejecting probability of all steps
           where standardError increases. It must be between 0 and 1 where 1
           means rejecting all steps where standardError increases and 0 means
           accepting all steps regardless whether standardError increases or
           not.

    .. code-block:: python

        ## Tetrahydrofuran (THF) molecule sketch
        ##
        ##              O
        ##   H41      /   \\      H11
        ##      \\  /         \\  /
        ## H42-- C4    THF     C1 --H12
        ##        \\ MOLECULE  /
        ##         \\         /
        ##   H31-- C3-------C2 --H21
        ##        /          \\
        ##     H32            H22
        ##

        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint

        # create engine
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb('system.pdb')

        # create and add constraint
        IAC = ImproperAngleConstraint()
        ENGINE.add_constraints(IAC)

        # define intra-molecular improper angles
        IAC.create_angles_by_definition( anglesDefinition={"THF": [ ('C2','O','C1','C4', -15, 15),
                                                                    ('C3','O','C1','C4', -15, 15) ] })
    """

    def __init__(self, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, rejectProbability=rejectProbability)
        # set atomsCollector data keys
        self._atomsCollector.set_data_keys( ['improperMap','otherMap'] )
        # init angles data
        self.__anglesDefinition = None
        self.__anglesList       = [[],[],[],[],[],[]]
        self.__angles           = {}
        # set computation cost
        self.set_computation_cost(3.0)
        # create dump flag
        self.__dumpAngles = True
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_ImproperAngleConstraint__anglesDefinition',
                           '_ImproperAngleConstraint__anglesList',
                           '_ImproperAngleConstraint__angles',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA) )
        object.__setattr__(self, 'RUNTIME_DATA', tuple(RUNTIME_DATA) )

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
            angles = [angles[0],angles[1],angles[2],angles[3], FLOAT_TYPE(180)*angles[4]/PI,FLOAT_TYPE(180)*angles[5]/PI]
            code.append("angles = {val}".format(val=angles))
            code.append("{name}.set_angles(angles)".format(name=name, val=angles))
        # return
        return dependencies, '\n'.join(code)


    def _codify__(self, engine, name='constraint', addDependencies=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        dependencies = 'from fullrmc.Constraints import ImproperAngleConstraints'
        code         = []
        if addDependencies:
            code.append(dependencies)
        code.append("{name} = ImproperAngleConstraints.ImproperAngleConstraint\
(rejectProbability={rejectProbability})".format(name=name, rejectProbability=self.rejectProbability))
        code.append("{engine}.add_constraints([{name}])".format(engine=engine, name=name))
        if self.__anglesDefinition is not None:
            code.append("{name}.create_angles_by_definition({angles})".
            format(name=name, angles=self.__anglesDefinition))
        elif len(self.__anglesList[0]):
            angles = constraint.anglesList
            angles = [angles[0],angles[1],angles[2],angles[3], FLOAT_TYPE(180.)*angles[4]/PI, FLOAT_TYPE(180.)*angles[5]/PI]
            code.append("{name}.set_angles({angles})".
            format(name=name, angles=angles))
        # return
        return [dependencies], '\n'.join(code)


    @property
    def anglesList(self):
        """ Get improper angles list."""
        return self.__anglesList

    @property
    def anglesDefinition(self):
        """angles definition copy if improper angles are defined as such"""
        return copy.deepcopy(self.__anglesDefinition)

    @property
    def angles(self):
        """ Get angles dictionary for every and each atom."""
        return self.__angles

    def _on_collector_reset(self):
        self._atomsCollector._randomData = set([])

    def listen(self, message, argument=None):
        """
        Listens to any message sent from the Broadcaster.

        :Parameters:
            #. message (object): Any python object to send to constraint's
               listen method.
            #. argument (object): Any type of argument to pass to the
               listeners.
        """
        if message in ("engine set","update pdb","update molecules indexes","update elements indexes","update names indexes"):
            if self.__anglesDefinition is not None:
                self.create_angles_by_definition(self.__anglesDefinition)
            else:
                # set angles and reset constraint
                AL = [ self.__anglesList[0],self.__anglesList[1],
                       self.__anglesList[2],self.__anglesList[3],
                       [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[4]],
                       [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[5]] ]
                self.set_angles(anglesList=AL, tform=False)
        elif message in ("update boundary conditions",):
            # reset constraint
            self.reset_constraint()


    def set_angles(self, anglesList, tform=True):
        """
        Set angles dictionary by parsing anglesList list.

        :Parameters:
            #. anglesList (list): Angles list definition.

               tuples format: every item must be a list of five items.\n
               #. Improper atom index that must be in the plane.
               #. Index of atom 'O' considered the origin of the plane.
               #. Index of atom 'x' used to calculated 'Ox' vector.
               #. Index of atom 'y' used to calculated 'Oy' vector.
               #. Minimum lower limit or minimum angle allowed
                  in degrees which later will be converted to rad.
               #. Maximum upper limit or maximum angle allowed
                  in degrees which later will be converted to rad.

               six vectors format: every item must be a list of five items.\n
               #. List containing improper atoms index that must be in the
                  plane.
               #. List containing index of atoms 'O' considered the origin
                  of the plane.
               #. List containing index of atoms 'x' used to calculated
                  'Ox' vector.
               #. List containing index of atom 'y' used to calculated 'Oy'
                  vector.
               #. List containing minimum lower limit or minimum angle allowed
                  in degrees which later will be converted to rad.
               #. List containing maximum upper limit or maximum angle allowed
                  in degrees which later will be converted to rad.

           #. tform (boolean): Whether given anglesList follows tuples format,
              If not then it must follow the six vectors one.
        """
        assert self.engine is not None, LOGGER.error("setting angles is not allowed unless engine is defined.")
        assert isinstance(anglesList, (list,set,tuple)), "anglesList must be a list"
        # convert to list of tuples
        if not tform:
            assert len(anglesList) == 6, LOGGER.error("non tuple form anglesList must be a list of 6 items")
            assert all([isinstance(i, (list,tuple,np.ndarray)) for i in anglesList]), LOGGER.error("non tuple form anglesList must be a list of list or tuple or numpy.ndarray")
            assert all([len(i)==len(anglesList[0]) for i in anglesList]), LOGGER.error("anglesList items list length mismatch")
            anglesList = zip(*anglesList)
        # get number of atoms
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        # loop angles
        anglesL = [[],[],[],[],[],[]]
        angles  = {}
        tempA   = {}
        for a in anglesList:
            assert isinstance(a, (list, set, tuple)), LOGGER.error("anglesList items must be lists")
            assert len(a)==6, LOGGER.error("anglesList items must be lists of 6 items each")
            improperIdx, oIdx, xIdx, yIdx, lower, upper = a
            assert is_integer(improperIdx), LOGGER.error("angle first item must be an integer")
            improperIdx = INT_TYPE(improperIdx)
            assert is_integer(oIdx), LOGGER.error("angle second item must be an integer")
            oIdx = INT_TYPE(oIdx)
            assert is_integer(xIdx), LOGGER.error("angle third item must be an integer")
            xIdx = INT_TYPE(xIdx)
            assert is_integer(yIdx), LOGGER.error("angle fourth item must be an integer")
            yIdx = INT_TYPE(yIdx)
            assert improperIdx>=0, LOGGER.error("angle first item must be positive")
            assert improperIdx<NUMBER_OF_ATOMS, LOGGER.error("angle first item atom index must be smaller than maximum number of atoms")
            assert oIdx>=0, LOGGER.error("angle second item must be positive")
            assert oIdx<NUMBER_OF_ATOMS, LOGGER.error("angle second item atom index must be smaller than maximum number of atoms")
            assert xIdx>=0, LOGGER.error("angle third item must be positive")
            assert xIdx<NUMBER_OF_ATOMS, LOGGER.error("angle third item atom index must be smaller than maximum number of atoms")
            assert yIdx>=0, LOGGER.error("angle fourth item must be positive")
            assert yIdx<NUMBER_OF_ATOMS, LOGGER.error("angle atom index must be smaller than maximum number of atoms")
            assert improperIdx!=oIdx, LOGGER.error("angle second items can't be the same")
            assert improperIdx!=xIdx, LOGGER.error("angle third items can't be the same")
            assert improperIdx!=yIdx, LOGGER.error("angle fourth items can't be the same")
            assert oIdx!=xIdx, LOGGER.error("angle second and third items can't be the same")
            assert oIdx!=yIdx, LOGGER.error("angle second and fourth items can't be the same")
            assert xIdx!=yIdx, LOGGER.error("angle third and fourth items can't be the same")
            assert is_number(lower), LOGGER.error("angle fifth item must be a number")
            lower = FLOAT_TYPE(lower)
            assert is_number(upper), LOGGER.error("angle sixth item must be a number")
            upper = FLOAT_TYPE(upper)
            assert lower>=-90, LOGGER.error("angle fifth item must be bigger or equal to -90 deg.")
            assert upper>lower, LOGGER.error("angle fifth item must be smaller than the sixth item")
            assert upper<=90, LOGGER.error("angle sixth item must be smaller or equal to 90")
            lower *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
            upper *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
            # check for redundancy
            plane   = [oIdx, xIdx, yIdx]
            impDef  = tuple([improperIdx] + sorted(plane))
            assert impDef not in tempA, LOGGER.error("Redundant definition for improper angle between improper atom '%s' and plane %s"%(improperIdx,plane))
            tempA[impDef] = True
            # create improper angle
            if not improperIdx in angles:
                anglesImproper = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]}
            else:
                anglesImproper = {"oIdx"        :angles[improperIdx]["oIdx"],
                                  "xIdx"        :angles[improperIdx]["xIdx"],
                                  "yIdx"        :angles[improperIdx]["yIdx"],
                                  "improperMap" :angles[improperIdx]["improperMap"],
                                  "otherMap"    :angles[improperIdx]["otherMap"] }
            # create anglesO angle
            if not oIdx in angles:
                anglesO = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]}
            else:
                anglesO = {"oIdx"        :angles[oIdx]["oIdx"],
                           "xIdx"        :angles[oIdx]["xIdx"],
                           "yIdx"        :angles[oIdx]["yIdx"],
                           "improperMap" :angles[oIdx]["improperMap"],
                           "otherMap"    :angles[oIdx]["otherMap"] }
            # create anglesX angle
            if not xIdx in angles:
                anglesX = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]}
            else:
                anglesX = {"oIdx"        :angles[xIdx]["oIdx"],
                           "xIdx"        :angles[xIdx]["xIdx"],
                           "yIdx"        :angles[xIdx]["yIdx"],
                           "improperMap" :angles[xIdx]["improperMap"],
                           "otherMap"    :angles[xIdx]["otherMap"] }
            # create anglesY angle
            if not yIdx in angles:
                anglesY = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]}
            else:
                anglesY = {"oIdx"        :angles[yIdx]["oIdx"],
                           "xIdx"        :angles[yIdx]["xIdx"],
                           "yIdx"        :angles[yIdx]["yIdx"],
                           "improperMap" :angles[yIdx]["improperMap"],
                           "otherMap"    :angles[yIdx]["otherMap"] }
            # set improper angle
            anglesImproper["oIdx"].append(oIdx)
            anglesImproper["xIdx"].append(xIdx)
            anglesImproper["yIdx"].append(yIdx)
            anglesImproper["improperMap"].append( len(anglesL[0]) )
            anglesO["otherMap"].append( len(anglesL[0]) )
            anglesX["otherMap"].append( len(anglesL[0]) )
            anglesY["otherMap"].append( len(anglesL[0]) )
            anglesL[0].append(improperIdx)
            anglesL[1].append(oIdx)
            anglesL[2].append(xIdx)
            anglesL[3].append(yIdx)
            anglesL[4].append(lower)
            anglesL[5].append(upper)
        # finalize angles
        for idx in xrange(NUMBER_OF_ATOMS):
            angles[INT_TYPE(idx)] = angles.get(INT_TYPE(idx), {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]}  )
        # set angles
        self.__angles           = angles
        self.__anglesList       = [np.array(anglesL[0], dtype=INT_TYPE),
                                   np.array(anglesL[1], dtype=INT_TYPE),
                                   np.array(anglesL[2], dtype=INT_TYPE),
                                   np.array(anglesL[3], dtype=INT_TYPE),
                                   np.array(anglesL[4], dtype=FLOAT_TYPE),
                                   np.array(anglesL[5], dtype=FLOAT_TYPE)]
        self.__anglesDefinition = None
        # dump to repository
        if self.__dumpAngles:
            self._dump_to_repository({'_DihedralAngleConstraint__anglesDefinition':self.__anglesDefinition,
                                      '_DihedralAngleConstraint__anglesList'      :self.__anglesList,
                                      '_DihedralAngleConstraint__angles'          :self.__angles})
            # reset constraint
            self.reset_constraint()


    #@raise_if_collected
    def create_angles_by_definition(self, anglesDefinition):
        """
        Creates anglesList using angles definition. This calls set_angles(anglesMap)
        and generates angles attribute. All angles are in degrees.

        :Parameters:
            #. anglesDefinition (dict): Angles definition.
               Every key must be a molecule name.
               Every key value must be a list of angles definitions.
               Every angle definition is a list of five items where:

               #. Name of improper atom that must be in the plane.
               #. Name of atom 'O' considered the origin of the plane.
               #. Name of atom 'x' used to calculated 'Ox' vector.
               #. Name of atom 'y' used to calculated 'Oy' vector.
               #. Minimum lower limit or minimum angle allowed
                  in degrees which later will be converted to rad.
               #. Maximum upper limit or maximum angle allowed
                  in degrees which later will be converted to rad.

        ::

            e.g. (Benzene):  anglesDefinition={"BENZ": [('C3','C1','C2','C6', -10, 10),
                                                        ('C4','C1','C2','C6', -10, 10),
                                                        ('C5','C1','C2','C6', -10, 10) ] }

        """
        if self.engine is None:
            raise Exception("Engine is not defined. Can't create impoper angles by definition")
        assert isinstance(anglesDefinition, dict), "anglesDefinition must be a dictionary"
        # check map definition
        ALL_NAMES       = self.engine.get_original_data("allNames")
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        MOLECULES_NAME  = self.engine.get_original_data("moleculesName")
        MOLECULES_INDEX = self.engine.get_original_data("moleculesIndex")
        existingMoleculesName = sorted(set(MOLECULES_NAME))
        anglesDef = {}
        for mol in anglesDefinition:
            angles = anglesDefinition[mol]
            if mol not in existingMoleculesName:
                LOGGER.usage("Molecule name '%s' in anglesDefinition is not recognized, angles definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(angles, (list, set, tuple)), LOGGER.error("mapDefinition molecule angles must be a list")
            angles = list(angles)
            molAnglesList = []
            tempA         = {}
            for angle in angles:
                assert isinstance(angle, (list, set, tuple)), LOGGER.error("mapDefinition angles must be a list")
                angle = list(angle)
                assert len(angle)==6
                improperAt, oAt, xAt, yAt, lower, upper = angle
                # check for redundancy
                plane   = [oAt, xAt, yAt]
                impDef  = tuple([improperAt] + sorted(plane))
                assert impDef not in tempA, LOGGER.error("Redundant definition for improper angle between improper atom '%s' and plane %s"%(improperIdx,plane))
                tempA[impDef] = True
                molAnglesList.append((improperAt, oAt, xAt, yAt, lower, upper))
                ## check for redundancy
                #append = True
                #for b in molAnglesList:
                #    if (b[0]==improperAt):
                #        if sorted([oAt,xAt,yAt]) == sorted([b[1],b[2],b[3]]):
                #            LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
                #            append = False
                #            break
                #if append:
                #    molAnglesList.append((improperAt, oAt, xAt, yAt, lower, upper))
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
                improperIdx = indexes[ names.index(angle[0]) ]
                oIdx        = indexes[ names.index(angle[1]) ]
                xIdx        = indexes[ names.index(angle[2]) ]
                yIdx        = indexes[ names.index(angle[3]) ]
                lower       = angle[4]
                upper       = angle[5]
                anglesList.append((improperIdx, oIdx, xIdx, yIdx, lower, upper))
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
            self._dump_to_repository({'_ImproperAngleConstraint__anglesDefinition':self.__anglesDefinition,
                                      '_ImproperAngleConstraint__anglesList'      :self.__anglesList,
                                      '_ImproperAngleConstraint__angles'          :self.__angles})
            # reset constraint
            self.reset_constraint()

    def compute_standard_error(self, data):
        """
        Compute the standard error (StdErr) of data not satisfying constraint's
        conditions.

        .. math::
            StdErr = \\sum \\limits_{i}^{C}
            ( \\theta_{i} - \\theta_{i}^{min} ) ^{2}
            \\int_{0}^{\\theta_{i}^{min}} \\delta(\\theta-\\theta_{i}) d \\theta
            +
            ( \\theta_{i} - \\theta_{i}^{max} ) ^{2}
            \\int_{\\theta_{i}^{max}}^{\\pi} \\delta(\\theta-\\theta_{i}) d \\theta

        Where:\n
        :math:`C` is the total number of defined improper angles constraints. \n
        :math:`\\theta_{i}^{min}` is the improper angle constraint lower limit set for constraint i. \n
        :math:`\\theta_{i}^{max}` is the improper angle constraint upper limit set for constraint i. \n
        :math:`\\theta_{i}` is the improper angle computed for constraint i. \n
        :math:`\\delta` is the Dirac delta function. \n
        :math:`\\int_{0}^{\\theta_{i}^{min}} \\delta(\\theta-\\theta_{i}) d \\theta`
        is equal to 1 if :math:`0 \\leqslant \\theta_{i} \\leqslant \\theta_{i}^{min}` and 0 elsewhere.\n
        :math:`\\int_{\\theta_{i}^{max}}^{\\pi} \\delta(\\theta-\\theta_{i}) d \\theta`
        is equal to 1 if :math:`\\theta_{i}^{max} \\leqslant \\theta_{i} \\leqslant \\pi` and 0 elsewhere.\n

        :Parameters:
            #. data (numpy.array): data to compute standardError.

        :Returns:
            #. standardError (number): computed standardError of given data.
        """
        return FLOAT_TYPE( np.sum(data["reducedAngles"]**2) )

    def get_constraint_value(self):
        """
        get constraint's data value.

        :Returns:
            #. data (dictionary): constraint data.
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
            anglesData    = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            reducedData   = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            #anglesIndexes = set(set(range(self.__anglesList[0].shape[0])))
            anglesIndexes = set(range(self.__anglesList[0].shape[0]))
            anglesIndexes = list( anglesIndexes-self._atomsCollector._randomData )
            improperIdxs = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes])
            oIdxs        = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes])
            xIdxs        = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes])
            yIdxs        = self._atomsCollector.get_relative_indexes(self.__anglesList[3][anglesIndexes])
            lowerLimit = self.__anglesList[4][anglesIndexes]
            upperLimit = self.__anglesList[5][anglesIndexes]
        else:
            improperIdxs = self._atomsCollector.get_relative_indexes(self.__anglesList[0])
            oIdxs        = self._atomsCollector.get_relative_indexes(self.__anglesList[1])
            xIdxs        = self._atomsCollector.get_relative_indexes(self.__anglesList[2])
            yIdxs        = self._atomsCollector.get_relative_indexes(self.__anglesList[3])
            lowerLimit = self.__anglesList[4]
            upperLimit = self.__anglesList[5]
        # compute data
        angles, reduced =  full_improper_angles_coords( improperIdxs       = improperIdxs,
                                                        oIdxs              = oIdxs,
                                                        xIdxs              = xIdxs,
                                                        yIdxs              = yIdxs,
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
            anglesData[anglesIndexes]  = angles
            reducedData[anglesIndexes] = reduced
            angles  = anglesData
            reduced = reducedData
        # create data and compute standard error
        data     = {"angles":angles, "reducedAngles":reduced}
        stdError = self.compute_standard_error(data = data)
        # update
        if update:
            self.set_data(data)
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
        # get angles indexes
        anglesIndexes = []
        #for idx in relativeIndexes:
        for idx in realIndexes:
            anglesIndexes.extend( self.__angles[idx]['improperMap'] )
            anglesIndexes.extend( self.__angles[idx]['otherMap'] )
        #anglesIndexes = list(set(anglesIndexes))
        anglesIndexes = list( set(anglesIndexes)-set(self._atomsCollector._randomData) )
        # compute data before move
        if len(anglesIndexes):
            angles, reduced =  full_improper_angles_coords( improperIdxs       = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes]),
                                                            oIdxs              = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes]),
                                                            xIdxs              = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes]),
                                                            yIdxs              = self._atomsCollector.get_relative_indexes(self.__anglesList[3][anglesIndexes]),
                                                            lowerLimit         = self.__anglesList[4][anglesIndexes],
                                                            upperLimit         = self.__anglesList[5][anglesIndexes],
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
        self.set_active_atoms_data_before_move( {"anglesIndexes":anglesIndexes, "angles":angles, "reducedAngles":reduced} )
        self.set_active_atoms_data_after_move(None)

    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """
        Compute constraint's data after move is executed.

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Group atoms relative index the
               move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new
               coordinates.
        """
        # get angles indexes
        anglesIndexes = self.activeAtomsDataBeforeMove["anglesIndexes"]
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[relativeIndexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[relativeIndexes] = movedBoxCoordinates
        # compute data before move
        if len(anglesIndexes):
            angles, reduced =  full_improper_angles_coords( improperIdxs       = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes]),
                                                            oIdxs              = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes]),
                                                            xIdxs              = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes]),
                                                            yIdxs              = self._atomsCollector.get_relative_indexes(self.__anglesList[3][anglesIndexes]),
                                                            lowerLimit         = self.__anglesList[4][anglesIndexes],
                                                            upperLimit         = self.__anglesList[5][anglesIndexes],
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
            # anglesIndexes is a fancy slicing, RL is a copy not a view.
            RL = self.data["reducedAngles"][anglesIndexes]
            self.data["reducedAngles"][anglesIndexes] += reduced-self.activeAtomsDataBeforeMove["reducedAngles"]
            self.set_after_move_standard_error( self.compute_standard_error(data = self.data) )
            self.data["reducedAngles"][anglesIndexes] = RL
        # increment tried
        self.increment_tried()

    def accept_move(self, realIndexes, relativeIndexes):
        """
        Accept move

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        # get indexes
        anglesIndexes = self.activeAtomsDataBeforeMove["anglesIndexes"]
        if len(anglesIndexes):
            # set new data
            data = self.data
            data["angles"][anglesIndexes]        += self.activeAtomsDataAfterMove["angles"]-self.activeAtomsDataBeforeMove["angles"]
            data["reducedAngles"][anglesIndexes] += self.activeAtomsDataAfterMove["reducedAngles"]-self.activeAtomsDataBeforeMove["reducedAngles"]
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

    def accept_amputation(self, realIndex, relativeIndex):
        """
        Accept amputation of atom and sets constraint's data and standard
        error accordingly.

        :Parameters:
            #. realIndex (numpy.ndarray): Atom's index as a numpy array
               of a single element.
            #. relativeIndex (numpy.ndarray): Not used here.
        """
        # MAYBE WE DON"T NEED TO CHANGE DATA AND SE. BECAUSE THIS MIGHT BE A PROBLEM
        # WHEN IMPLEMENTING ATOMS RELEASING. MAYBE WE NEED TO COLLECT DATA INSTEAD, REMOVE
        # AND ADD UPON RELEASE
        # get all involved data
        anglesIndexes = []
        for idx in realIndex:
            anglesIndexes.extend( self.__angles[idx]['improperMap'] )
            anglesIndexes.extend( self.__angles[idx]['otherMap'] )
        anglesIndexes = list(set(anglesIndexes))
        if len(anglesIndexes):
            # set new data
            data = self.data
            data["angles"][anglesIndexes]        = 0
            data["reducedAngles"][anglesIndexes] = 0
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
        AI = self.__angles[realIndex]['improperMap'] + self.__angles[realIndex]['otherMap']
        # append all mapped bonds to collector's random data
        self._atomsCollector._randomData = self._atomsCollector._randomData.union( set(AI) )
        # collect atom anglesIndexes
        self._atomsCollector.collect(realIndex, dataDict={'improperMap':self.__angles[realIndex]['improperMap'],
                                                          'otherMap'   :self.__angles[realIndex]['otherMap']})


    def _plot(self,frameIndex, propertiesLUT,
                   spacing,numberOfTicks,nbins,splitBy,
                   ax, barsRelativeWidth,limitsParams,
                   legendParams,titleParams,
                   xticksParams, yticksParams,
                   xlabelParams, ylabelParams,
                   gridParams,stackHorizontal,
                   colorCodeXticksLabels,*args, **kwargs):

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
        categories = {}
        atom2 = self.__anglesList[0]
        atom1 = self.__anglesList[1]
        atom3 = self.__anglesList[2]
        atom4 = self.__anglesList[3]
        lower = self.__anglesList[4]
        upper = self.__anglesList[5]
        for idx in xrange(self.__anglesList[0].shape[0]):
            #if self._atomsCollector.is_collected(idx):
            #    continue
            if self._atomsCollector.is_collected(atom1[idx]):
                continue
            if self._atomsCollector.is_collected(atom2[idx]):
                continue
            if self._atomsCollector.is_collected(atom3[idx]):
                continue
            if self._atomsCollector.is_collected(atom4[idx]):
                continue
            if splitBy is not None:
                a1 = splitBy[ atom1[idx] ]
                a2 = splitBy[ atom2[idx] ]
                a3 = splitBy[ atom3[idx] ]
                a4 = splitBy[ atom4[idx] ]
            else:
                a1 = a2 = a3 = a4 = ''
            l = lower[idx]
            u = upper[idx]
            k = (a1,a2,a3,a4,l,u)
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
            a1,a2,a3,a4, L,U  = key
            L  = L*180./np.pi
            U  = U*180./np.pi
            LU = "(%.2f,%.2f)"%(L,U)
            label = "%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LU)
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
         return super(ImproperAngleConstraint, self).plot(spacing=spacing, nbins=nbins,
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
        return {'_ImproperAngleConstraint__anglesDefinition':'_ImproperAngleConstraint__anglesDefinition',
                '_ImproperAngleConstraint__anglesList'      :'_ImproperAngleConstraint__anglesList',
                '_ImproperAngleConstraint__angles'          :'_ImproperAngleConstraint__angles',
                '_Constraint__used'                         :'_Constraint__used',
                '_Constraint__data'                         :'_Constraint__data',
                '_Constraint__standardError'                :'_Constraint__standardError',
                '_Constraint__state'                        :'_Constraint__state',
                '_Engine__state'                            :'_Engine__state',
                '_Engine__boxCoordinates'                   :'_Engine__boxCoordinates',
                '_Engine__basisVectors'                     :'_Engine__basisVectors',
                '_Engine__isPBC'                            :'_Engine__isPBC',
                '_Engine__moleculesIndex'                   :'_Engine__moleculesIndex',
                '_Engine__elementsIndex'                    :'_Engine__elementsIndex',
                '_Engine__numberOfAtomsPerElement'          :'_Engine__numberOfAtomsPerElement',
                '_Engine__elements'                         :'_Engine__elements',
                '_Engine__numberDensity'                    :'_Engine__numberDensity',
                '_Engine__volume'                           :'_Engine__volume',
                '_atomsCollector'                           :'_atomsCollector',
                ('engine','_atomsCollector')                :'_atomsCollector',
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
        atom4 = self.__anglesList[3]
        lower = self.__anglesList[4]*180./np.pi
        upper = self.__anglesList[5]*180./np.pi
        consData = data["angles"]*180./np.pi
        header = ['atom_1_index', 'atom_2_index', 'atom_3_index', 'atom_4_index',
                  'atom_1_element', 'atom_2_element', 'atom_3_element','atom_4_element',
                  'atom_1_name', 'atom_2_name', 'atom_3_name', 'atom_4_name',
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
            if self._atomsCollector.is_collected(atom4[idx]):
                continue
            data.append([str(atom1[idx]),str(atom2[idx]),str(atom3[idx]),str(atom4[idx]),
                             elements[atom1[idx]],elements[atom2[idx]],elements[atom3[idx]],elements[atom4[idx]],
                             names[atom1[idx]],names[atom2[idx]],names[atom3[idx]],names[atom4[idx]],
                             format%lower[idx], format%upper[idx],
                             format%consData[idx]] )
        # save
        return header, data
