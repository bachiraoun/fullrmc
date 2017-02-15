"""
ImproperAngleConstraints contains classes for all constraints related improper angles between atoms.

.. inheritance-diagram:: fullrmc.Constraints.ImproperAngleConstraints
    :parts: 1
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, raise_if_collected, reset_if_collected_out_of_date
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.dihedral_angles import full_dihedral_angles_coords




class DihedralAngleConstraint(RigidConstraint, SingularConstraint):
    """
    Dihedral angle is defined between two intersecting planes formed with four defined 
    atoms. Dihedral angle constraint can control 3 angle shells at the same times.
    
    +--------------------------------------------------------------------------------+
    |.. figure:: dihedralSketch.png                                                  |
    |   :width: 312px                                                                |
    |   :height: 200px                                                               |
    |   :align: center                                                               |
    |                                                                                |
    |   Dihedral angle sketch defined between two planes formed with four atoms.     |  
    +--------------------------------------------------------------------------------+
    
    :Parameters:
        #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
    
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
        self.__anglesList = [[],[],[],[],[],[],[],[],[],[]]      
        self.__angles     = {}
        # set computation cost
        self.set_computation_cost(3.0)
        # create dump flag
        self.__dumpAngles = True
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_DihedralAngleConstraint__anglesList',
                           '_DihedralAngleConstraint__angles',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA) )
        object.__setattr__(self, 'RUNTIME_DATA', tuple(RUNTIME_DATA) )
        
    @property
    def anglesList(self):
        """ Get improper angles list."""
        return self.__anglesList
    
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
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine set","update boundary conditions",):     
            self.set_angles(anglesList=self.__anglesList, tform=False)   
            # reset constraint is called in set_angles
   
    #@raise_if_collected
    def set_angles(self, anglesList, tform=True):
        """ 
        Sets the angles dictionary by parsing the anglesList list. All angles are in 
        degrees. Dihedral angle can control 3 angle shells at the same times defined using 
        three different lower and upper angle bounds simulating three different dihedral 
        potential energy minimums. Dihedral angles are defined from 0 to 360 degrees. 
        Shell's lower and  upper bound defines a dihedral angle clockwise. Therefore in 
        order to take into considerations the limits at 0 and 360 degrees, lower bound 
        is allowed to be higher than the higher bound.\n
        e.g. (50, 100) is a dihedral shell defined in the angle range between 50 and 
        100 degrees. But (100, 50) dihedral shell is defined between 100 to 360 degrees
        and wraps the range from 0 to 100. (50, 100) and (100, 50) are complementary 
        and cover the whole range from 0 to 360 deg. \n
        
        :Parameters:
            #. anglesList (list): The angles list definition.
              
               tuples format: every item must be a list of ten items.\n
               #. First item: The first atom index of the first plane.
               #. Second item: The second atom index of the first plane and the first 
                  atom index of the second plane.
               #. Third item: The third atom index of the first plane and the second 
                  atom index of the second plane.
               #. Fourth item: The fourth atom index of the second plane.
               #. Fifth item: The minimum lower limit of the first shell or the minimum 
                  angle allowed in degrees.
               #. Sixth item: The maximum upper limit of the first shell or the minimum 
                  angle allowed in degrees.
               #. Seventh item: The minimum lower limit of the second shell or the minimum 
                  angle allowed in degrees.
               #. Eights item: The maximum upper limit of the second shell or the minimum 
                  angle allowed in degrees.
               #. Nineth item: The minimum lower limit of the third shell or the minimum 
                  angle allowed in degrees.
               #. Tenth item: The maximum upper limit of the third shell or the minimum 
                  angle allowed in degrees.\n\n
                
               ten vectors format: every item must be a list of five items.\n
               #. First item: List containing the first atom indexes of the first plane.
               #. Second item: List containing indexes of the second atom index of the 
                  first plane and the first atom index of the second plane.
               #. Third item: List containing indexes of the third atom index of the 
                  first plane and the second atom index of the second plane.
               #. Fourth item: List containing indexes of the fourth atom index of the 
                  second plane.
               #. Fifth item: List containing the minimum lower limit of the first shell 
                  or the minimum angle allowed in degrees which later will be converted 
                  to rad.
               #. Sixth item: List containing the maximum upper limit of the first shell 
                  or the minimum angle allowed in degrees which later will be converted 
                  to rad.
               #. Seventh item: List containing the minimum lower limit of the second 
                  shell or the minimum angle allowed in degrees which later will be 
                  converted to rad.
               #. Eights item: List containing the maximum upper limit of the second 
                  shell or the minimum angle allowed in degrees which later will be 
                  converted to rad.
               #. Nineth item: List containing the minimum lower limit of the third 
                  shell or the minimum angle allowed in degrees which later will be 
                  converted to rad.
               #. Tenth item: List containing the maximum upper limit of the third shell 
                  or the minimum angle allowed in degrees which later will be converted 
                  to rad.
                  
           #. tform (boolean): set whether given anglesList follows tuples format, If not 
              then it must follow the ten vectors one.
        
        **N.B.** Defining three shells boundaries is mandatory. In case fewer than three
        shells is needed, it suffices to repeat one of the shells boundaries.\n
        e.g. ('C1','C2','C3','C4', 40,80, 100,140, 40,80), in the herein definition the 
        last shell is a repetition of the first which means only two shells are defined.
        """
        # check if bondsList is given
        if anglesList is None:
            anglesList = [[],[],[],[],[],[],[],[],[],[]]
            tform      = False 
        elif len(anglesList) == 10 and len(anglesList[0]) == 0:
            tform     = False 
        if self.engine is None:
            self.__anglesList = anglesList      
            self.__angles     = {}
        else:
            NUMBER_OF_ATOMS   = self.engine.get_original_data("numberOfAtoms")
            oldAnglesList = self.__anglesList
            oldAngles     = self.__angles
            self.__anglesList = [np.array([], dtype=INT_TYPE),
                                 np.array([], dtype=INT_TYPE),
                                 np.array([], dtype=INT_TYPE),
                                 np.array([], dtype=INT_TYPE),
                                 np.array([], dtype=FLOAT_TYPE),
                                 np.array([], dtype=FLOAT_TYPE),
                                 np.array([], dtype=FLOAT_TYPE),
                                 np.array([], dtype=FLOAT_TYPE),
                                 np.array([], dtype=FLOAT_TYPE),
                                 np.array([], dtype=FLOAT_TYPE)]      
            self.__angles     = {}
            self.__dumpAngles = False
            # build angles
            try: 
                if tform:
                    for angle in anglesList:
                        self.add_angle(angle)
                else:
                    for idx in xrange(len(anglesList[0])):
                        angle = [anglesList[0][idx], anglesList[1][idx], 
                                 anglesList[2][idx], anglesList[3][idx], 
                                 anglesList[4][idx], anglesList[5][idx],
                                 anglesList[6][idx], anglesList[7][idx],
                                 anglesList[8][idx], anglesList[9][idx]]
                        self.add_angle(angle)
            except Exception as e:
                self.__dumpAngles = True
                self.__anglesList = oldAnglesList
                self.__angles     = oldAngles
                LOGGER.error(e)
                import traceback
                raise Exception(traceback.format_exc())
            self.__dumpAngles = True
            # finalize angles
            for idx in xrange(NUMBER_OF_ATOMS):
                self.__angles[INT_TYPE(idx)] = self.__angles.get(INT_TYPE(idx), {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]}  )
        # dump to repository
        self._dump_to_repository({'_DihedralAngleConstraint__anglesList' :self.__anglesList,
                                  '_DihedralAngleConstraint__angles'     :self.__angles})
        # reset constraint
        self.reset_constraint()
    
    #@raise_if_collected
    def add_angle(self, angle):
        """
        Add a single angle to the list of constraint angles. All angles are in degrees.
        
        :Parameters:
            #. angle (list): The angle list of ten items.\n
               #. First item: The first atom index of the first plane.
               #. Second item: The second atom index of the first plane and the first 
                  atom index of the second plane.
               #. Third item: The third atom index of the first plane and the second 
                  atom index of the second plane.
               #. Fourth item: The fourth atom index of the second plane.
               #. Fifth item: The minimum lower limit of the first shell or the minimum 
                  angle allowed in degrees.
               #. Sixth item: The maximum upper limit of the first shell or the minimum 
                  angle allowed in degrees.
               #. seventh item: The minimum lower limit of the second shell or the minimum 
                  angle allowed in degrees.
               #. eights item: The maximum upper limit of the second shell or the minimum 
                  angle allowed in degrees.
               #. nineth item: The minimum lower limit of the third shell or the minimum 
                  angle allowed in degrees.
               #. tenth item: The maximum upper limit of the third shell or the minimum 
                  angle allowed in degrees.
        """
        assert self.engine is not None, LOGGER.error("setting an angle is not allowed unless engine is defined.")
        NUMBER_OF_ATOMS   = self.engine.get_original_data("numberOfAtoms")
        assert isinstance(angle, (list, set, tuple)), LOGGER.error("anglesList items must be lists")
        assert len(angle)==10, LOGGER.error("anglesList items must be lists of 10 items each")
        idx1, idx2, idx3, idx4, lower1, upper1, lower2, upper2, lower3, upper3 = angle
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
        #lower1 *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
        #upper1 *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
        assert is_number(lower2), LOGGER.error("angle seventh item must be a number")
        lower2 = FLOAT_TYPE(lower2)
        assert is_number(upper2), LOGGER.error("angle eights item must be a number")
        upper2 = FLOAT_TYPE(upper2)
        assert lower2>=0, LOGGER.error("angle seventh item must be bigger or equal to 0 deg.")
        assert lower2<=360, LOGGER.error("angle seventh item must be smaller or equal to 360 deg.")
        assert upper2>=0, LOGGER.error("angle eightth item must be bigger or equal to 0 deg.")
        assert upper2<=360, LOGGER.error("angle eightth item must be smaller or equal to 360 deg.")
        #lower2 *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
        #upper2 *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
        assert is_number(lower3), LOGGER.error("angle nineth item must be a number")
        lower3 = FLOAT_TYPE(lower3)
        assert is_number(upper3), LOGGER.error("angle tenth item must be a number")
        upper3 = FLOAT_TYPE(upper3)
        assert lower3>=0, LOGGER.error("angle nineth item must be bigger or equal to 0 deg.")
        assert lower3<=360, LOGGER.error("angle nineth item must be smaller or equal to 360 deg.")
        assert upper3>=0, LOGGER.error("angle tenth item must be bigger or equal to 0 deg.")
        assert upper3<=360, LOGGER.error("angle tenth item must be smaller or equal to 360 deg.")
        #lower3 *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
        #upper3 *= FLOAT_TYPE( PI/FLOAT_TYPE(180.) )
        # create dihedral angles1
        if not self.__angles.has_key(idx1):
            angles1 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]} 
        else:
            angles1 = {"idx2"        :self.__angles[idx1]["idx2"], 
                       "idx3"        :self.__angles[idx1]["idx3"], 
                       "idx4"        :self.__angles[idx1]["idx4"], 
                       "dihedralMap" :self.__angles[idx1]["dihedralMap"], 
                       "otherMap"    :self.__angles[idx1]["otherMap"] }          
        # create dihedral angle2
        if not self.__angles.has_key(idx2):
            angles2 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]} 
        else:
            angles2 = {"idx2"        :self.__angles[idx2]["idx2"], 
                       "idx3"        :self.__angles[idx2]["idx3"], 
                       "idx4"        :self.__angles[idx2]["idx4"], 
                       "dihedralMap" :self.__angles[idx2]["dihedralMap"], 
                       "otherMap"    :self.__angles[idx2]["otherMap"] }  
        # create dihedral angle3
        if not self.__angles.has_key(idx3):
            angles3 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]} 
        else:
            angles3 = {"idx2"        :self.__angles[idx3]["idx2"], 
                       "idx3"        :self.__angles[idx3]["idx3"], 
                       "idx4"        :self.__angles[idx3]["idx4"], 
                       "dihedralMap" :self.__angles[idx3]["dihedralMap"], 
                       "otherMap"    :self.__angles[idx3]["otherMap"] }  
        # create dihedral angle4
        if not self.__angles.has_key(idx4):
            angles4 = {"idx2":[],"idx3":[],"idx4":[],"dihedralMap":[],"otherMap":[]} 
        else:
            angles4 = {"idx2"        :self.__angles[idx4]["idx2"], 
                       "idx3"        :self.__angles[idx4]["idx3"], 
                       "idx4"        :self.__angles[idx4]["idx4"], 
                       "dihedralMap" :self.__angles[idx4]["dihedralMap"], 
                       "otherMap"    :self.__angles[idx4]["otherMap"] }  
        # check for re-defining
        setPos = idx2Pos = idx3Pos = idx4Pos = None               
        if idx2 in angles1["idx2"] and idx3 in angles1["idx3"] and idx4 in angles1["idx4"]:
            idx2Pos = angles1["idx2"].index(idx2)
            idx3Pos = angles1["idx3"].index(idx3)
            idx4Pos = angles1["idx4"].index(idx4)
        elif idx2 in angles1["idx2"] and idx4 in angles1["idx3"] and idx3 in angles1["idx4"]:
            idx2Pos = angles1["idx2"].index(idx2)
            idx3Pos = angles1["idx4"].index(idx3)
            idx4Pos = angles1["idx3"].index(idx4)
        elif idx3 in angles1["idx2"] and idx2 in angles1["idx3"] and idx4 in angles1["idx4"]:
            idx2Pos = angles1["idx3"].index(idx2)
            idx3Pos = angles1["idx2"].index(idx3)
            idx4Pos = angles1["idx4"].index(idx4)
        elif idx4 in angles1["idx2"] and idx3 in angles1["idx3"] and idx2 in angles1["idx4"]:
            idx2Pos = angles1["idx4"].index(idx2)
            idx3Pos = angles1["idx3"].index(idx3)
            idx4Pos = angles1["idx2"].index(idx4)
        if idx2Pos is not None and (idx2Pos==idx3Pos) and (idx2Pos==idx4Pos):
            LOGGER.warn("Dihedral angle definition for atom1 index '%i' and atom2 '%i' and atom3 '%i' and atom4 '%i' is  already defined. New shells' limits [(%.3f,%.3f),(%.3f,%.3f),(%.3f,%.3f)] are set."%(idx1, idx2, idx3, idx4, lower1, upper1, lower2, upper2, lower3, upper3))
            setPos = angles1["dihedralMap"][idx2Pos]
        # set angle
        if setPos is None:
            angles1["idx2"].append(idx2)        
            angles1["idx3"].append(idx3)                
            angles1["idx4"].append(idx4)        
            angles1["dihedralMap"].append( len(self.__anglesList[0]) )
            angles2["otherMap"].append( len(self.__anglesList[0]) )
            angles3["otherMap"].append( len(self.__anglesList[0]) )
            angles4["otherMap"].append( len(self.__anglesList[0]) )
            self.__anglesList[0] = np.append(self.__anglesList[0],idx1)
            self.__anglesList[1] = np.append(self.__anglesList[1],idx2)
            self.__anglesList[2] = np.append(self.__anglesList[2],idx3)
            self.__anglesList[3] = np.append(self.__anglesList[3],idx4)
            self.__anglesList[4] = np.append(self.__anglesList[4],lower1)
            self.__anglesList[5] = np.append(self.__anglesList[5],upper1)
            self.__anglesList[6] = np.append(self.__anglesList[6],lower2)
            self.__anglesList[7] = np.append(self.__anglesList[7],upper2)
            self.__anglesList[8] = np.append(self.__anglesList[8],lower3)
            self.__anglesList[9] = np.append(self.__anglesList[9],upper3) 
        else:
            assert self.__anglesList[0][setPos] == idx1, LOGGER.error("mismatched dihedral angle atom '%s' and '%s'"%(self.__anglesList[0][setPos],idx1))
            assert sorted([idx2, idx3, idx4]) == sorted([self.__anglesList[1][idx2],self.__anglesList[2][idx3],self.__anglesList[3][idx4]]), LOGGER.error("mismatched dihedral angle atom2, atom3, atom4 at atom1 '%s'"%(idx1))
            self.__anglesList[1][setPos] = idx2
            self.__anglesList[2][setPos] = idx3
            self.__anglesList[3][setPos] = idx4
            self.__anglesList[4][setPos] = lower1
            self.__anglesList[5][setPos] = upper1
            self.__anglesList[6][setPos] = lower2
            self.__anglesList[7][setPos] = upper2
            self.__anglesList[8][setPos] = lower3
            self.__anglesList[9][setPos] = upper3
        self.__angles[idx1] = angles1
        self.__angles[idx2] = angles2
        self.__angles[idx3] = angles3
        self.__angles[idx4] = angles4
        # dump to repository
        if self.__dumpAngles:
            self._dump_to_repository({'_DihedralAngleConstraint__anglesList' :self.__anglesList,
                                      '_DihedralAngleConstraint__angles'     :self.__angles})
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
               
               #. First item: The name of the first dihedral atom.
               #. Second item: The name of the second dihedral atom of the first plane 
                  and the first atom of the second plane.
               #. Third item: The name of the third dihedral atom of the first plane 
                  and the second  atom of the second plane.
               #. Fourth item: The name of the fourth dihderal atom of the second plane.
               #. Fifth item: The minimum lower limit of the first shell or the minimum 
                  angle allowed in degrees.
               #. Sixth item: The maximum upper limit of the first or the maximum 
                  angle allowed in degrees.   
               #. Seventh item: The minimum lower limit of the second shell or the minimum 
                  angle allowed in degrees.
               #. Eightth item: The maximum upper limit of the second or the maximum 
                  angle allowed in degrees.
               #. Nineth item: The minimum lower limit of the third shell or the minimum 
                  angle allowed in degrees.
               #. Tenth item: The maximum upper limit of the third or the maximum 
                  angle allowed in degrees.
        
        ::
        
            e.g. (Butane):  anglesDefinition={"BUT": [ ('C1','C2','C3','C4', 40,80, 100,140, 290,330), ] }
                                                                
                                                  
        """
        if self.engine is None:
            raise Exception("Engine is not defined. Can't create dihedral angles by definition")
        assert isinstance(anglesDefinition, dict), "anglesDefinition must be a dictionary"
        # check map definition
        ALL_NAMES         = self.engine.get_original_data("allNames")
        NUMBER_OF_ATOMS   = self.engine.get_original_data("numberOfAtoms")
        MOLECULES_NAMES   = self.engine.get_original_data("moleculesNames")
        MOLECULES_INDEXES = self.engine.get_original_data("moleculesIndexes")
        existingMoleculesNames = sorted(set(MOLECULES_NAMES))
        anglesDef = {}
        for mol, angles in anglesDefinition.items():
            if mol not in existingMoleculesNames:
                log.LocalLogger("fullrmc").logger.warn("Molecule name '%s' in anglesDefinition is not recognized, angles definition for this particular molecule is omitted"%str(mol))
                continue
            assert isinstance(angles, (list, set, tuple)), LOGGER.error("mapDefinition molecule angles must be a list")
            angles = list(angles)
            molAnglesList = []
            for angle in angles:
                assert isinstance(angle, (list, set, tuple)), LOGGER.error("mapDefinition angles must be a list")
                angle = list(angle)
                assert len(angle)==10
                at1, at2, at3, at4, lower1, upper1, lower2, upper2, lower3, upper3 = angle
                # check for redundancy
                append = True
                for b in molAnglesList:
                    if (b[0]==at1):
                        if sorted([at2,at3,at4]) == sorted([b[1],b[2],b[3]]):
                            LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
                            append = False
                            break
                if append:
                    molAnglesList.append((at1, at2, at3, at4, lower1, upper1, lower2, upper2, lower3, upper3))
            # create bondDef for molecule mol 
            anglesDef[mol] = molAnglesList
        # create mols dictionary
        mols = {}
        for idx in xrange(NUMBER_OF_ATOMS):
            molName = MOLECULES_NAMES[idx]
            if not molName in anglesDef.keys():    
                continue
            molIdx = MOLECULES_INDEXES[idx]
            if not mols.has_key(molIdx):
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
        self.set_angles(anglesList=anglesList)
    
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
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        return self.data
        
    @reset_if_collected_out_of_date
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        if len(self._atomsCollector):
            anglesData    = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            reducedData   = np.zeros(self.__anglesList[0].shape[0], dtype=FLOAT_TYPE)
            anglesIndexes = set(set(range(self.__anglesList[0].shape[0])))
            anglesIndexes  = list( anglesIndexes-self._atomsCollector._randomData )
            
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
        # set data.     
        self.set_data( {"angles":angles, "reducedAngles":reduced} )
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
        Compute constraint after move is executed
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
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
  
    def accept_move(self, realIndexes, relativeIndexes):
        """ 
        Accept move
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
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

    def reject_move(self, realIndexes, relativeIndexes):
        """ 
        Reject move
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
    
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
            #. realIndex (numpy.ndarray): atom index as a numpy array of a single element.
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

        
    def plot(self, ax=None, nbins=20, subplots=True, split=None,
                   wspace=0.3, hspace=0.3,
                   histtype='bar', lineWidth=None, lineColor=None,
                   xlabel=True, xlabelSize=16,
                   ylabel=True, ylabelSize=16,
                   legend=True, legendCols=1, legendLoc='best',
                   title=True, titleStdErr=True, titleAtRem=True,
                   titleUsedFrame=True, show=True):
        """ 
        Plot dihedral angles constraint distribution histogram.
        
        :Parameters:
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If ax is given,  subplots parameters will be omitted. 
               If None is given a new plot figure will be created.
            #. nbins (int): number of bins in histogram.
            #. subplots (boolean): Whether to add plot constraint on multiple axes.
            #. split (None, 'name', 'element'): To split plots into histogram per atom 
               names, elements in addition to lower and upper bounds. If None histograms 
               will be built from lower and upper bounds only.
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
            #. title (boolean): Whether to create the title or not.
            #. titleStdErr (boolean): Whether to show constraint standard error value in title.
            #. titleAtRem (boolean): Whether to show engine's number of removed atoms.
            #. titleUsedFrame(boolean): Whether to show used frame name in title.
            #. show (boolean): Whether to render and show figure before returning.
        
        :Returns:
            #. figure (matplotlib Figure): matplotlib used figure.
            #. axes (matplotlib Axes, List): matplotlib axes or a list of axes.
        """
        def _get_bins(dmin, dmax, boundaries, nbins):
            # create bins
            delta = float(dmax-dmin)/float(nbins-1)
            bins  = range(nbins)
            bins  = [b*delta for b in bins]
            bins  = [b+dmin for b in bins]
            # check boundaries
            bidx = 0
            for b in sorted(boundaries):
                for i in range(bidx, len(bins)-1):
                    bidx = i
                    # exact match with boundary
                    if b==bins[bidx]:
                        break
                    # boundary between two bins, move closest bin to boundary
                    if bins[bidx] < b < bins[bidx+1]:
                        if b-bins[bidx] > bins[bidx+1]-b:
                            bins[bidx+1] = b
                        else:
                            bins[bidx]   = b
                        break
            # return bins
            return bins
        # get constraint value
        output = self.get_constraint_value()
        if output is None:
            LOGGER.warn("%s constraint data are not computed."%(self.__class__.__name__))
            return
       # compute categories 
        if split == 'name':
            splitV = self.engine.get_original_data("allNames")
        elif split == 'element':
            splitV = self.engine.get_original_data("allElements")
        else:
            splitV = None
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
            if self._atomsCollector.is_collected(idx):
                continue
            if splitV is not None:
                a1 = splitV[ atom1[idx] ]
                a2 = splitV[ atom2[idx] ]
                a3 = splitV[ atom3[idx] ]
                a4 = splitV[ atom4[idx] ]
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
        ncategories = len(categories.keys())
        # import matplotlib
        import matplotlib.pyplot as plt
        # get axes
        if ax is None:
            if subplots and ncategories>1:
                x = np.ceil(np.sqrt(ncategories))
                y = np.ceil(ncategories/x)
                FIG, N_AXES = plt.subplots(int(x), int(y) )
                N_AXES = FIG.flatten()
                FIG.subplots_adjust(wspace=wspace, hspace=hspace)
                [N_AXES[i].axis('off') for i in range(ncategories,len(N_AXES))]
            else:
                FIG  = plt.figure()
                AXES = FIG.gca()
                subplots = False 
        else:
            AXES = ax  
            FIG = AXES.get_figure()
            subplots = False 
        # start plotting
        COLORS = ["b",'g','r','c','y','m']
        if subplots:
            for idx, key in enumerate(categories.keys()): 
                a1,a2,a3,a4, L1,U1, L2,U2, L3,U3  = key
                LU = sorted(set( [(L1,U1),(L2,U2),(L3,U3)] ))
                LA = " ".join( ["(%.2f,%.2f)"%(l,u)  for l,u in LU] )
                label = "%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LA)
                COL  = COLORS[idx%len(COLORS)]
                AXES = N_AXES[idx]
                idxs = categories[key]
                data = self.data["angles"][idxs]
                # get data limits
                mn = np.min(data)
                mx = np.max(data)
                # get bins
                BINS = _get_bins(dmin=mn, dmax=mx, boundaries=[L1,U1,L2,U2,L3,U3], nbins=nbins)
                # plot histogram
                D, _, P = AXES.hist(x=data, bins=BINS, 
                                    color=COL, label=label,
                                    histtype=histtype)
                # vertical lines
                Y = max(D)
                for idx, (l,u) in enumerate(LU):
                    AXES.plot([l,l],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle=['--','-.',':'][idx])
                    AXES.plot([u,u],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle=['--','-.',':'][idx])
                # legend
                if legend:
                    AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
                # set axis labels
                if xlabel:
                    AXES.set_xlabel("$deg.$", size=xlabelSize)
                if ylabel:
                    AXES.set_ylabel("$number$"  , size=ylabelSize)
                if lineWidth is not None:
                    [p.set_linewidth(lineWidth) for p in P]
                if lineColor is not None:
                    [p.set_edgecolor(lineColor) for p in P]
                # update limits
                AXES.set_xmargin(0.1)
                AXES.autoscale() 
        else:
            for idx, key in enumerate(categories.keys()): 
                a1,a2,a3,a4, L1,U1, L2,U2, L3,U3  = key
                LU = sorted(set( [(L1,U1),(L2,U2),(L3,U3)] ))
                LA = " ".join( ["(%.2f,%.2f)"%(l,u)  for l,u in LU] )
                label = "%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LA)
                COL  = COLORS[idx%len(COLORS)]
                idxs = categories[key]
                data = self.data["angles"][idxs]
                # get data limits
                mn = np.min(data)
                mx = np.max(data)
                # get bins
                BINS = _get_bins(dmin=mn, dmax=mx, boundaries=[L1,U1,L2,U2,L3,U3], nbins=nbins)
                # plot histogram
                D, _, P = AXES.hist(x=data, bins=BINS, 
                                    color=COL, label=label,
                                    histtype=histtype)
                # vertical lines
                Y = max(D)
                for idx, (l,u) in enumerate(LU):
                    AXES.plot([l,l],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle=['--','-.',':'][idx])
                    AXES.plot([u,u],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle=['--','-.',':'][idx])
                if lineWidth is not None:
                    [p.set_linewidth(lineWidth) for p in P]
                if lineColor is not None:
                    [p.set_edgecolor(lineColor) for p in P]
            # update limits
            AXES.set_xmargin(0.1)
            AXES.autoscale()
            # legend
            if legend:
                AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
            # set axis labels
            if xlabel:
                AXES.set_xlabel("$deg.$", size=xlabelSize)
            if ylabel:
                AXES.set_ylabel("$number$"  , size=ylabelSize)
        # set title
        if title:
            FIG.canvas.set_window_title('Dihedral Angle Constraint')
            if titleUsedFrame:
                t = '$frame: %s$ : '%self.engine.usedFrame.replace('_','\_')
            else:
                t = ''
            if titleAtRem:
                t += "$%i$ $rem.$ $at.$ - "%(len(self.engine._atomsCollector))
            if titleStdErr and self.standardError is not None:
                t += "$std$ $error=%.6f$ "%(self.standardError)
            if len(t):
                FIG.suptitle(t, fontsize=14)
        
        # set background color
        FIG.patch.set_facecolor('white')
        #show
        if show:
            plt.show()
        # return axes
        if subplots:
            return FIG, N_AXES
        else:
            return FIG, AXES  
    
    def export(self, fname, delimiter='     ', comments='# ', split=None):
        """
        Export pair distribution constraint.
        
        :Parameters:
            #. fname (path): full file name and path.
            #. delimiter (string): String or character separating columns.
            #. comments (string): String that will be prepended to the header.
            #. split (None, 'name', 'element'): To split output into per atom names,
               elements in addition to lower and upper bounds. If None output 
               will be built from lower and upper bounds only.
        """
        # get constraint value
        output = self.get_constraint_value()
        if not len(output):
            LOGGER.warn("%s constraint data are not computed."%(self.__class__.__name__))
            return
        # compute categories 
        if split == 'name':
            splitV = self.engine.get_original_data("allNames")
        elif split == 'element':
            splitV = self.engine.get_original_data("allElements")
        else:
            splitV = None
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
            if self._atomsCollector.is_collected(idx):
                continue
            if splitV is not None:
                a1 = splitV[ atom1[idx] ]
                a2 = splitV[ atom2[idx] ]
                a3 = splitV[ atom3[idx] ]
                a4 = splitV[ atom4[idx] ]
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
        ncategories = len(categories.keys())
        # create data
        for idx, key in enumerate(categories.keys()): 
            idxs = categories[key]
            data = self.data["angles"][idxs]
            categories[key] = [str(d) for d in data]
        # adjust data size
        maxSize = max( [len(v) for v in categories.values()] )
        for key, data in categories.items():
            add =  maxSize-len(data)
            if add > 0:
                categories[key] = data + ['']*add
        # start creating header and data
        sortCa = sorted( categories.keys() )
        header = []
        for key in sortCa:
            a1,a2,a3,a4, L1,U1, L2,U2, L3,U3  = key
            LU = sorted(set( [(L1,U1),(L2,U2),(L3,U3)] ))
            LA = " ".join( ["(%.2f,%.2f)"%(l,u)  for l,u in LU] )
            header.append( ("%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LA)).replace(' ','') )
        data   = [categories[key] for key in sortCa]
        # save
        data = np.transpose(data)
        np.savetxt(fname     = fname, 
                   X         = data, 
                   fmt       = '%s', 
                   delimiter = delimiter, 
                   header    = " ".join(header),
                   comments  = comments)       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        







        
        
        

    
    

    
    
            