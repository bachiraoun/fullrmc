"""
ImproperAngleConstraints contains classes for all constraint's related to
improper angles between bonded atoms.

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
from ..Core.dihedral_angles import full_dihedral_angles_coords




class DihedralAngleConstraint(RigidConstraint, SingularConstraint):
    """
    Dihedral angle is defined between two intersecting planes formed
    with defined atoms. Dihedral angle constraint can control up to three
    angle shells at the same times.


    +---------------------------------------------------------------------------+
    |.. figure:: dihedralSketch.png                                             |
    |   :width: 312px                                                           |
    |   :height: 200px                                                          |
    |   :align: center                                                          |
    |                                                                           |
    |   Dihedral angle sketch defined between two planes formed with four atoms.|
    +---------------------------------------------------------------------------+


     .. raw:: html

        <iframe width="560" height="315"
        src="https://www.youtube.com/embed/1wUSYcNygd4"
        frameborder="0" allowfullscreen>
        </iframe>


    :Parameters:
        #. rejectProbability (Number): rejecting probability of all steps
           where standardError increases. It must be between 0 and 1 where 1
           means rejecting all steps where standardError increases and 0
           means accepting all steps regardless whether standardError increases or not.

    .. code-block:: python

        ## Butane (BUT) molecule sketch
        ##
        ##       H13  H22  H32  H43
        ##        |    |    |    |
        ## H11---C1---C2---C3---C4---H41
        ##        |    |    |    |
        ##       H12  H21  H31  H42
        ##

        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.DihedralAngleConstraints import DihedralAngleConstraint

        # create engine
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb('system.pdb')

        # create and add constraint
        DAC = DihedralAngleConstraint()
        ENGINE.add_constraints(DAC)

        # define intra-molecular dihedral angles
        DAC.create_angles_by_definition( anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 40,80, 100,140, 290,330), ] })


    """

    def __init__(self, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, rejectProbability=rejectProbability)
        # set atomsCollector data keys
        self._atomsCollector.set_data_keys( ['dihedralMap','otherMap'] )
        # init angles data
        self.__anglesDefinition = None
        self.__anglesList       = [[],[],[],[],[],[],[],[],[],[]]
        self.__angles           = {}
        # set computation cost
        self.set_computation_cost(3.0)
        # create dump flag
        self.__dumpAngles = True
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_DihedralAngleConstraint__anglesDefinition',
                           '_DihedralAngleConstraint__anglesList',
                           '_DihedralAngleConstraint__angles',] )
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
            code.append("angles = {val}".format(val=angles))
            code.append("{name}.set_angles(angles)".format(name=name, val=angles))
        # return
        return dependencies, '\n'.join(code)


    def _codify__(self, engine, name='constraint', addDependencies=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        dependencies = 'from fullrmc.Constraints import DihedralAngleConstraints'
        code         = []
        if addDependencies:
            code.append(dependencies)
        code.append("{name} = DihedralAngleConstraints.DihedralAngleConstraint\
(rejectProbability={rejectProbability})".format(name=name, rejectProbability=self.rejectProbability))
        code.append("{engine}.add_constraints([{name}])".format(engine=engine, name=name))
        if self.__anglesDefinition is not None:
            code.append("{name}.create_angles_by_definition({angles})".
            format(name=name, angles=self.__anglesDefinition))
        elif len(self.__anglesList[0]):
            code.append("{name}.set_angles({angles})".
            format(name=name, angles=self.__anglesList))
        # return
        return [dependencies], '\n'.join(code)

    @property
    def anglesList(self):
        """ Improper angles list."""
        return self.__anglesList

    @property
    def anglesDefinition(self):
        """angles definition copy if dihedral angles are defined as such"""
        return copy.deepcopy(self.__anglesDefinition)

    @property
    def angles(self):
        """ Angles dictionary for every and each atom."""
        return self.__angles

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
            if self.__anglesDefinition is not None:
                self.create_angles_by_definition(self.__anglesDefinition)
            else:
                self.set_angles(anglesList=self.__anglesList, tform=False)
        elif message in ("update boundary conditions",):
            # reset constraint
            self.reset_constraint()


    def set_angles(self, anglesList, tform=True):
        """
        Sets the angles dictionary by parsing the anglesList list.
        All angles are in degrees. Dihedral angle can control up to three
        angle shells at the same times defined using three different lower
        and upper angle bounds simulating three different dihedral
        potential energy minimums. Dihedral angles are defined from 0 to
        360 degrees. Shell's lower and  upper bound defines a dihedral angle
        clockwise. Therefore in order to take into considerations the limits
        at 0 and 360 degrees, lower bound is allowed to be higher than the
        higher bound.\n
        e.g. (50, 100) is a dihedral shell defined in the angle range between 50 and
        100 degrees. But (100, 50) dihedral shell is defined between 100 to 360 degrees
        and wraps the range from 0 to 100. (50, 100) and (100, 50) are complementary
        and cover the whole range from 0 to 360 deg. \n

        :Parameters:
            #. anglesList (list): The angles list definition.

               tuples format: every item must be a list of ten items.\n
               #. First atom index of the first plane.
               #. Second atom index of the first plane and
                  first atom index of the second plane.
               #. Third atom index of the first plane and second
                  atom index of the second plane.
               #. Fourth atom index of the second plane.
               #. Minimum lower limit of the first shell or
                  minimum angle allowed in degrees which later will be
                  converted to rad.
               #. Maximum upper limit of the first shell or
                  maximum angle allowed in degrees which later will be
                  converted to rad.
               #. Minimum lower limit of the second shell or
                  minimum angle allowed in degrees which later will be
                  converted to rad.
               #. Maximum upper limit of the second shell or
                  maximum angle allowed in degrees which later will be
                  converted to rad.
               #. Minimum lower limit of the third shell or
                  minimum angle allowed in degrees which later will be
                  converted to rad.
               #. Maximum upper limit of the third shell or
                  maximum angle allowed in degrees.\n\n

               ten vectors format: every item must be a list of five items.\n
               #. List containing first atoms index of the first plane.
               #. List containing second atoms index of the
                  first plane and first atoms index of the second plane.
               #. List containing third atoms indexes of the
                  first plane and second atoms index of the second plane.
               #. List containing fourth atoms index of the second plane.
               #. List containing minimum lower limit of the first shell
                  or minimum angle allowed in degrees which later will be
                  converted to rad.
               #. List containing maximum upper limit of the first shell
                  or maximum angle allowed in degrees which later will be
                  converted to rad.
               #. List containing minimum lower limit of the second
                  shell or minimum angle allowed in degrees which later will be
                  converted to rad.
               #. List containing maximum upper limit of the second
                  shell or maximum angle allowed in degrees which later will be
                  converted to rad.
               #. List containing minimum lower limit of the third
                  shell or minimum angle allowed in degrees which later will be
                  converted to rad.
               #. List containing maximum upper limit of the third shell
                  or maximum angle allowed in degrees which later will be
                  converted to rad.

           #. tform (boolean): set whether given anglesList follows tuples format, If not
              then it must follow the ten vectors one.

        **N.B.** Defining three shells boundaries is mandatory. In case fewer than three
        shells is needed, it suffices to repeat one of the shells boundaries.\n
        e.g. ('C1','C2','C3','C4', 40,80, 100,140, 40,80), in the herein definition the
        last shell is a repetition of the first which means only two shells are defined.
        """
        assert self.engine is not None, LOGGER.error("setting angles is not allowed unless engine is defined.")
        assert isinstance(anglesList, (list,set,tuple)), "anglesList must be a list"
        # convert to list of tuples
        if not tform:
            assert len(anglesList) == 10, LOGGER.error("non tuple form anglesList must be a list of 10 items")
            assert all([isinstance(i, (list,tuple,np.ndarray)) for i in anglesList]), LOGGER.error("non tuple form anglesList must be a list of list or tuple or numpy.ndarray")
            assert all([len(i)==len(anglesList[0]) for i in anglesList]), LOGGER.error("anglesList items list length mismatch")
            anglesList = zip(*anglesList)
        # get number of atoms
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        # loop angles
        anglesL = [[],[],[],[],[],[],[],[],[],[]]
        angles  = {}
        tempA   = {}
        for a in anglesList:
            assert isinstance(a, (list, set, tuple)), LOGGER.error("anglesList items must be lists")
            assert len(a)==10, LOGGER.error("anglesList items must be lists of 10 items each")
            idx1, idx2, idx3, idx4, lower1, upper1, lower2, upper2, lower3, upper3 = a
            assert is_integer(idx1), LOGGER.error("angle first item must be an integer")
            idx1 = INT_TYPE(idx1)
            assert is_integer(idx2), LOGGER.error("angle second item must be an integer")
            idx2 = INT_TYPE(idx2)
            assert is_integer(idx3), LOGGER.error("angle third item must be an integer")
            idx3 = INT_TYPE(idx3)
            assert is_integer(idx4), LOGGER.error("angle fourth item must be an integer")
            idx4 = INT_TYPE(idx4)
            assert idx1>=0, LOGGER.error("angle first item must be positive")
            assert idx1<NUMBER_OF_ATOMS, LOGGER.error("angle first item atom index must be smaller than maximum number of atoms")
            assert idx2>=0, LOGGER.error("angle second item must be positive")
            assert idx2<NUMBER_OF_ATOMS, LOGGER.error("angle second item atom index must be smaller than maximum number of atoms")
            assert idx3>=0, LOGGER.error("angle third item must be positive")
            assert idx3<NUMBER_OF_ATOMS, LOGGER.error("angle third item atom index must be smaller than maximum number of atoms")
            assert idx4>=0, LOGGER.error("angle fourth item must be positive")
            assert idx4<NUMBER_OF_ATOMS, LOGGER.error("angle atom index must be smaller than maximum number of atoms")
            assert idx1!=idx2, LOGGER.error("angle second items can't be the same")
            assert idx1!=idx3, LOGGER.error("angle third items can't be the same")
            assert idx1!=idx4, LOGGER.error("angle fourth items can't be the same")
            assert idx2!=idx3, LOGGER.error("angle second and third items can't be the same")
            assert idx2!=idx4, LOGGER.error("angle second and fourth items can't be the same")
            assert idx3!=idx4, LOGGER.error("angle third and fourth items can't be the same")
            assert is_number(lower1), LOGGER.error("angle fifth item must be a number")
            lower1 = FLOAT_TYPE(lower1)
            assert is_number(upper1), LOGGER.error("angle sixth item must be a number")
            upper1 = FLOAT_TYPE(upper1)
            assert lower1>=0, LOGGER.error("angle fifth item must be bigger or equal to 0 deg.")
            assert lower1<=360, LOGGER.error("angle fifth item must be smaller or equal to 360 deg.")
            assert upper1>=0, LOGGER.error("angle sixth item must be bigger or equal to 0 deg.")
            assert upper1<=360, LOGGER.error("angle sixth item must be smaller or equal to 360 deg.")
            assert is_number(lower2), LOGGER.error("angle seventh item must be a number")
            lower2 = FLOAT_TYPE(lower2)
            assert is_number(upper2), LOGGER.error("angle eights item must be a number")
            upper2 = FLOAT_TYPE(upper2)
            assert lower2>=0, LOGGER.error("angle seventh item must be bigger or equal to 0 deg.")
            assert lower2<=360, LOGGER.error("angle seventh item must be smaller or equal to 360 deg.")
            assert upper2>=0, LOGGER.error("angle eightth item must be bigger or equal to 0 deg.")
            assert upper2<=360, LOGGER.error("angle eightth item must be smaller or equal to 360 deg.")
            assert is_number(lower3), LOGGER.error("angle nineth item must be a number")
            lower3 = FLOAT_TYPE(lower3)
            assert is_number(upper3), LOGGER.error("angle tenth item must be a number")
            upper3 = FLOAT_TYPE(upper3)
            assert lower3>=0, LOGGER.error("angle nineth item must be bigger or equal to 0 deg.")
            assert lower3<=360, LOGGER.error("angle nineth item must be smaller or equal to 360 deg.")
            assert upper3>=0, LOGGER.error("angle tenth item must be bigger or equal to 0 deg.")
            assert upper3<=360, LOGGER.error("angle tenth item must be smaller or equal to 360 deg.")
            # check for redundancy
            plane0  = [idx1, idx2, idx3]
            plane1  = [idx2, idx3, idx4]
            splane0 = tuple(sorted(plane0))
            splane1 = tuple(sorted(plane1))
            assert (splane0,splane1) not in tempA, LOGGER.error("Redundant definition for dihedral angle definition between planes %s and %s"%(plane0,plane1))
            assert (splane1,splane0) not in tempA, LOGGER.error("Redundant definition for dihedral angle definition between planes %s and %s"%(plane0,plane1))
            tempA[(splane0,splane1)] = True
            tempA[(splane1,splane0)] = True
            # create dihedral angles1
            if not idx1 in self.__angles:
                angles1 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]}
            else:
                angles1 = {"idx2"        :self.__angles[idx1]["idx2"],
                           "idx3"        :self.__angles[idx1]["idx3"],
                           "idx4"        :self.__angles[idx1]["idx4"],
                           "dihedralMap" :self.__angles[idx1]["dihedralMap"],
                           "otherMap"    :self.__angles[idx1]["otherMap"] }
            # create dihedral angle2
            if not idx2 in self.__angles:
                angles2 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]}
            else:
                angles2 = {"idx2"        :self.__angles[idx2]["idx2"],
                           "idx3"        :self.__angles[idx2]["idx3"],
                           "idx4"        :self.__angles[idx2]["idx4"],
                           "dihedralMap" :self.__angles[idx2]["dihedralMap"],
                           "otherMap"    :self.__angles[idx2]["otherMap"] }
            # create dihedral angle3
            if not idx3 in self.__angles:
                angles3 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]}
            else:
                angles3 = {"idx2"        :self.__angles[idx3]["idx2"],
                           "idx3"        :self.__angles[idx3]["idx3"],
                           "idx4"        :self.__angles[idx3]["idx4"],
                           "dihedralMap" :self.__angles[idx3]["dihedralMap"],
                           "otherMap"    :self.__angles[idx3]["otherMap"] }
            # create dihedral angle4
            if not idx4 in self.__angles:
                angles4 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]}
            else:
                angles4 = {"idx2"        :self.__angles[idx4]["idx2"],
                           "idx3"        :self.__angles[idx4]["idx3"],
                           "idx4"        :self.__angles[idx4]["idx4"],
                           "dihedralMap" :self.__angles[idx4]["dihedralMap"],
                           "otherMap"    :self.__angles[idx4]["otherMap"] }
            # set dihedral angle
            angles1["idx2"].append(idx2)
            angles1["idx3"].append(idx3)
            angles1["idx4"].append(idx4)
            angles1["dihedralMap"].append( len(anglesL[0]) )
            angles2["otherMap"].append( len(anglesL[0]) )
            angles3["otherMap"].append( len(anglesL[0]) )
            angles4["otherMap"].append( len(anglesL[0]) )
            anglesL[0].append(idx1)
            anglesL[1].append(idx2)
            anglesL[2].append(idx3)
            anglesL[3].append(idx4)
            anglesL[4].append(lower1)
            anglesL[5].append(upper1)
            anglesL[6].append(lower2)
            anglesL[7].append(upper2)
            anglesL[8].append(lower3)
            anglesL[9].append(upper3)
            angles[idx1] = angles1
            angles[idx2] = angles2
            angles[idx3] = angles3
            angles[idx4] = angles4
        # finalize angles
        for idx in xrange(NUMBER_OF_ATOMS):
            angles[INT_TYPE(idx)] = angles.get(INT_TYPE(idx), {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]}  )
        # set angles
        self.__angles           = angles
        self.__anglesList       = [np.array(anglesL[0], dtype=INT_TYPE),
                                   np.array(anglesL[1], dtype=INT_TYPE),
                                   np.array(anglesL[2], dtype=INT_TYPE),
                                   np.array(anglesL[3], dtype=INT_TYPE),
                                   np.array(anglesL[4], dtype=FLOAT_TYPE),
                                   np.array(anglesL[5], dtype=FLOAT_TYPE),
                                   np.array(anglesL[6], dtype=FLOAT_TYPE),
                                   np.array(anglesL[7], dtype=FLOAT_TYPE),
                                   np.array(anglesL[8], dtype=FLOAT_TYPE),
                                   np.array(anglesL[9], dtype=FLOAT_TYPE),]
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
        Creates anglesList using angles definition.
        Calls set_angles(anglesMap) and generates angles attribute.

        :Parameters:
            #. anglesDefinition (dict): The angles definition.
               Every key must be a molecule name (residue name in pdb file).
               Every key value must be a list of angles definitions.
               Every angle definition is a list of ten items where:

               #. Name of the first dihedral atom.
               #. Name of the second dihedral atom of the first plane
                  and the first atom of the second plane.
               #. Name of the third dihedral atom of the first plane
                  and the second  atom of the second plane.
               #. Name of the fourth dihderal atom of the second plane.
               #. Minimum lower limit of the first shell or the minimum
                  angle allowed in degrees which later will be
                  converted to rad.
               #. Maximum upper limit of the first or the maximum
                  angle allowed in degrees which later will be
                  converted to rad.
               #. Minimum lower limit of the second shell or the minimum
                  angle allowed in degrees which later will be
                  converted to rad.
               #. Maximum upper limit of the second or the maximum
                  angle allowed in degrees which later will be
                  converted to rad.
               #. Minimum lower limit of the third shell or the minimum
                  angle allowed in degrees which later will be
                  converted to rad.
               #. Maximum upper limit of the third or the maximum
                  angle allowed in degrees which later will be
                  converted to rad.

        ::

            e.g. (Butane):  anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 40,80, 100,140, 290,330), ] }


        """
        if self.engine is None:
            raise Exception("Engine is not defined. Can't create dihedral angles by definition")
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
                assert len(angle)==10, LOGGER.error("angles definition must be of length 10")
                at1, at2, at3, at4, lower1, upper1, lower2, upper2, lower3, upper3 = angle
                # check for redundancy
                plane0  = [at1, at2, at3]
                plane1  = [at2, at3, at4]
                splane0 = tuple(sorted(plane0))
                splane1 = tuple(sorted(plane1))
                assert (splane0,splane1) not in tempA, LOGGER.error("Redundant definition for dihedral angle definition between planes %s and %s"%(plane0,plane1))
                assert (splane1,splane0) not in tempA, LOGGER.error("Redundant definition for dihedral angle definition between planes %s and %s"%(plane0,plane1))
                tempA[(splane0,splane1)] = True
                tempA[(splane1,splane0)] = True
                molAnglesList.append((at1, at2, at3, at4, lower1, upper1, lower2, upper2, lower3, upper3))
                #for b in molAnglesList:
                #    if sa == sorted([b[0],b[1],b[2],b[3]]):
                #        LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(angle))
                #        append = False
                #        break
                #    if (b[0]==at1): 2019-12-20
                #        if sorted([at2,at3,at4]) == sorted([b[1],b[2],b[3]]):
                #            LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
                #            append = False
                #            break
                #if append:
                #    molAnglesList.append((at1, at2, at3, at4, lower1, upper1, lower2, upper2, lower3, upper3))
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
                idx1   = indexes[ names.index(angle[0]) ]
                idx2   = indexes[ names.index(angle[1]) ]
                idx3   = indexes[ names.index(angle[2]) ]
                idx4   = indexes[ names.index(angle[3]) ]
                lower1 = angle[4]
                upper1 = angle[5]
                lower2 = angle[6]
                upper2 = angle[7]
                lower3 = angle[8]
                upper3 = angle[9]
                anglesList.append( (idx1, idx2, idx3, idx4, lower1, upper1, lower2, upper2, lower3, upper3) )
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
            self._dump_to_repository({'_DihedralAngleConstraint__anglesDefinition':self.__anglesDefinition,
                                      '_DihedralAngleConstraint__anglesList'      :self.__anglesList,
                                      '_DihedralAngleConstraint__angles'          :self.__angles})
            # reset constraint
            self.reset_constraint()

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
            #. data (numpy.array): The constraint value data to compute standardError.

        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        return FLOAT_TYPE( np.sum(data["reducedAngles"]**2) )

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
            anglesData    = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            reducedData   = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            #anglesIndexes = set(set(range(self.__anglesList[0].shape[0])))
            anglesIndexes = set(range(self.__anglesList[0].shape[0]))
            anglesIndexes = list( anglesIndexes-self._atomsCollector._randomData )

            indexes1 = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes])
            indexes2 = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes])
            indexes3 = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes])
            indexes4 = self._atomsCollector.get_relative_indexes(self.__anglesList[3][anglesIndexes])
            lowerLimit1 = self.__anglesList[4][anglesIndexes]
            upperLimit1 = self.__anglesList[5][anglesIndexes]
            lowerLimit2 = self.__anglesList[6][anglesIndexes]
            upperLimit2 = self.__anglesList[7][anglesIndexes]
            lowerLimit3 = self.__anglesList[8][anglesIndexes]
            upperLimit3 = self.__anglesList[9][anglesIndexes]
        else:
            indexes1 = self._atomsCollector.get_relative_indexes(self.__anglesList[0])
            indexes2 = self._atomsCollector.get_relative_indexes(self.__anglesList[1])
            indexes3 = self._atomsCollector.get_relative_indexes(self.__anglesList[2])
            indexes4 = self._atomsCollector.get_relative_indexes(self.__anglesList[3])
            lowerLimit1 = self.__anglesList[4]
            upperLimit1 = self.__anglesList[5]
            lowerLimit2 = self.__anglesList[6]
            upperLimit2 = self.__anglesList[7]
            lowerLimit3 = self.__anglesList[8]
            upperLimit3 = self.__anglesList[9]
        # compute data
        angles, reduced =  full_dihedral_angles_coords( indexes1           = indexes1,
                                                        indexes2           = indexes2,
                                                        indexes3           = indexes3,
                                                        indexes4           = indexes4,
                                                        lowerLimit1        = lowerLimit1,
                                                        upperLimit1        = upperLimit1,
                                                        lowerLimit2        = lowerLimit2,
                                                        upperLimit2        = upperLimit2,
                                                        lowerLimit3        = lowerLimit3,
                                                        upperLimit3        = upperLimit3,
                                                        boxCoords          = self.engine.boxCoordinates,
                                                        basis              = self.engine.basisVectors ,
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
        # get angles indexes
        anglesIndexes = []
        #for idx in relativeIndexes:
        for idx in realIndexes:
            anglesIndexes.extend( self.__angles[idx]['dihedralMap'] )
            anglesIndexes.extend( self.__angles[idx]['otherMap'] )
        #anglesIndexes = list(set(anglesIndexes))
        anglesIndexes = list( set(anglesIndexes)-set(self._atomsCollector._randomData) )
        # compute data before move
        if len(anglesIndexes):
            angles, reduced =  full_dihedral_angles_coords( indexes1           = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes]),
                                                            indexes2           = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes]),
                                                            indexes3           = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes]),
                                                            indexes4           = self._atomsCollector.get_relative_indexes(self.__anglesList[3][anglesIndexes]),
                                                            lowerLimit1        = self.__anglesList[4][anglesIndexes],
                                                            upperLimit1        = self.__anglesList[5][anglesIndexes],
                                                            lowerLimit2        = self.__anglesList[6][anglesIndexes],
                                                            upperLimit2        = self.__anglesList[7][anglesIndexes],
                                                            lowerLimit3        = self.__anglesList[8][anglesIndexes],
                                                            upperLimit3        = self.__anglesList[9][anglesIndexes],
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
            #. relativeIndexes (numpy.ndarray): Group atoms relative index
               the move will be applied to.
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
            angles, reduced =  full_dihedral_angles_coords( indexes1           = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes]),
                                                            indexes2           = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes]),
                                                            indexes3           = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes]),
                                                            indexes4           = self._atomsCollector.get_relative_indexes(self.__anglesList[3][anglesIndexes]),
                                                            lowerLimit1        = self.__anglesList[4][anglesIndexes],
                                                            upperLimit1        = self.__anglesList[5][anglesIndexes],
                                                            lowerLimit2        = self.__anglesList[6][anglesIndexes],
                                                            upperLimit2        = self.__anglesList[7][anglesIndexes],
                                                            lowerLimit3        = self.__anglesList[8][anglesIndexes],
                                                            upperLimit3        = self.__anglesList[9][anglesIndexes],
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
        Accept move.

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
        Accept amputation of atom and set constraint's data and standard
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
            anglesIndexes.extend( self.__angles[idx]['dihedralMap'] )
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
        AI = self.__angles[realIndex]['dihedralMap'] + self.__angles[realIndex]['otherMap']
        # append all mapped bonds to collector's random data
        self._atomsCollector._randomData = self._atomsCollector._randomData.union( set(AI) )
        # collect atom anglesIndexes
        self._atomsCollector.collect(realIndex, dataDict={'dihedralMap':self.__angles[realIndex]['dihedralMap'],
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
        atom2  = self.__anglesList[0]
        atom1  = self.__anglesList[1]
        atom3  = self.__anglesList[2]
        atom4  = self.__anglesList[3]
        lower1 = self.__anglesList[4]
        upper1 = self.__anglesList[5]
        lower2 = self.__anglesList[6]
        upper2 = self.__anglesList[7]
        lower3 = self.__anglesList[8]
        upper3 = self.__anglesList[9]
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
            l1 = lower1[idx]
            u1 = upper1[idx]
            l2 = lower2[idx]
            u2 = upper2[idx]
            l3 = lower3[idx]
            u3 = upper3[idx]
            k = (a1,a2,a3,a4,l1,u1,l2,u2,l3,u3)
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
            a1,a2,a3,a4, L1,U1, L2,U2, L3,U3  = key
            LU = sorted(set( [(L1,U1),(L2,U2),(L3,U3)] ))
            LA = " ".join( ["(%.2f,%.2f)"%(l,u)  for l,u in LU] )
            L  = min(L1,L2,L3,U1,U2,U3)
            U  = max(L1,L2,L3,U1,U2,U3)
            label = "%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LA)
            col   = COLORS[idx%len(COLORS)]
            idxs  = categories[key]
            catd  = data["angles"][idxs]
            dmin  = np.min(catd)
            dmax  = np.max(catd)
            # append xticks labels
            dmint1 = dmaxt1 = []
            dmint2 = dmaxt2 = []
            dmint3 = dmaxt3 = []
            if dmin<L1:
                dmint1 = [dmin]
            if dmax>U1:
                dmaxt1 = [dmax]
            if dmin<L2:
                dmint2 = [dmin]
            if dmax>U2:
                dmaxt2 = [dmax]
            if dmin<L3:
                dmint3 = [dmin]
            if dmax>U3:
                dmaxt3 = [dmax]
            xticksL.extend( dmint1 + list(np.linspace(start=L1,stop=U1,num=numberOfTicks, endpoint=True)) + dmaxt1 )
            xticksL.extend( dmint2 + list(np.linspace(start=L2,stop=U2,num=numberOfTicks, endpoint=True)) + dmaxt2 )
            xticksL.extend( dmint3 + list(np.linspace(start=L3,stop=U3,num=numberOfTicks, endpoint=True)) + dmaxt3 )
            # rescale histogram
            resc = dsh = 0
            if stackHorizontal:
                resc   = min(L,np.min(catd)) - spacing # rescale to origin + spacing
                catd  -= resc - shifts[-1] # shift to stack to the right of the last histogram
                dmin   = np.min(catd)
                dmax   = np.max(catd)
                dmint = dmaxt = []
                L1     -= resc - shifts[-1]
                U1     -= resc - shifts[-1]
                L2     -= resc - shifts[-1]
                U2     -= resc - shifts[-1]
                L3     -= resc - shifts[-1]
                U3     -= resc - shifts[-1]
                dsh    = shifts[-1]
                LU     = [(L1,U1),(L2,U2),(L3,U3)]
            # append xticks positions
            if len(dmint1):
                dmint1 = [dmin-resc+shifts[-1]]
            if len(dmaxt1):
                dmaxt1 = [dmax-resc+shifts[-1]]
            if len(dmint2):
                dmint2 = [dmin-resc+shifts[-1]]
            if len(dmaxt2):
                dmaxt2 = [dmax-resc+shifts[-1]]
            if len(dmint3):
                dmint3 = [dmin-resc+shifts[-1]]
            if len(dmaxt3):
                dmaxt3 = [dmax-resc+shifts[-1]]
            xticks.extend( dmint1 + list(np.linspace(start=L1,stop=U1, num=numberOfTicks, endpoint=True)) + dmaxt1 )
            xticks.extend( dmint2 + list(np.linspace(start=L2,stop=U2, num=numberOfTicks, endpoint=True)) + dmaxt2 )
            xticks.extend( dmint3 + list(np.linspace(start=L3,stop=U3, num=numberOfTicks, endpoint=True)) + dmaxt3 )
            # append shifts
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
            lmp = copy.deepcopy(limitsParams)
            linestyle = lmp.pop('linestyle',None)
            if lmp.get('color',None) is None:
                lmp['color'] = col
            Y = max(D)
            B = 0 if stackHorizontal else shifts[-1]
            for idx, (l,u) in enumerate(LU):
                ls = linestyle
                if ls is None:
                    ls = ['--','-.',':'][idx]
                ax.plot([l,l],[B,B+Y+0.1*Y], linestyle=ls, **lmp)
                ax.plot([u,u],[B,B+Y+0.1*Y], linestyle=ls, **lmp)
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
        if stackHorizontal:
            ax.set_yticklabels( ['%i'%t for t in ax.get_yticks()], **yticksParams)
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels( ['%i'%t for t in yticksL], **yticksParams)
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


    def plot(self, spacing=2, numberOfTicks=3, nbins=20, barsRelativeWidth=0.95,
                   splitBy=None, stackHorizontal=False, colorCodeXticksLabels=True,
                   xlabelParams={'xlabel':'$deg.$', 'size':10},
                   ylabelParams={'ylabel':'number', 'size':10},
                   limitsParams={'linewidth':1.0, 'color':None, 'linestyle':None},
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
         return super(DihedralAngleConstraint, self).plot(spacing=spacing, nbins=nbins,
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
        return {'_DihedralAngleConstraint__anglesDefinition':'_DihedralAngleConstraint__anglesDefinition',
                '_DihedralAngleConstraint__anglesList'      :'_DihedralAngleConstraint__anglesList',
                '_DihedralAngleConstraint__angles'          :'_DihedralAngleConstraint__angles',
                '_Constraint__data'                         :'_Constraint__data',
                '_Constraint__standardError'                :'_Constraint__standardError',
                '_Constraint__state'                        :'_Constraint__state',
                '_Constraint__used'                         :'_Constraint__used',
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
        atom2  = self.__anglesList[0]
        atom1  = self.__anglesList[1]
        atom3  = self.__anglesList[2]
        atom4  = self.__anglesList[3]
        lower1 = self.__anglesList[4]
        upper1 = self.__anglesList[5]
        lower2 = self.__anglesList[6]
        upper2 = self.__anglesList[7]
        lower3 = self.__anglesList[8]
        upper3 = self.__anglesList[9]
        consData = data["angles"]
        header = ['atom_1_index', 'atom_2_index', 'atom_3_index', 'atom_4_index',
                  'atom_1_element', 'atom_2_element', 'atom_3_element','atom_4_element',
                  'atom_1_name', 'atom_2_name', 'atom_3_name', 'atom_4_name',
                  'shell_1_lower_limit', 'shell_1_upper_limit',
                  'shell_2_lower_limit', 'shell_2_upper_limit',
                  'shell_3_lower_limit', 'shell_3_upper_limit', 'value']
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
                             format%lower1[idx], format%upper1[idx],
                             format%lower2[idx], format%upper2[idx],
                             format%lower3[idx], format%upper3[idx],
                             format%consData[idx]] )
        # save
        return header, data
