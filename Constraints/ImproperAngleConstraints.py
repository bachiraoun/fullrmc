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
from fullrmc.Core.improper_angles import full_improper_angles_coords




class ImproperAngleConstraint(RigidConstraint, SingularConstraint):
    """
    Controls the improper angle formed with 4 defined atoms. It's mainly used to 
    keep the improper atom in the plane defined with three other atoms. 
    The improper vector is defined as the vector from the first atom of the plane to
    the improper atom. Therefore the improper angle is defined between the improper
    vector and the plane.
    
    +--------------------------------------------------------------------------------+
    |.. figure:: improperSketch.png                                                  |
    |   :width: 269px                                                                |
    |   :height: 200px                                                               |
    |   :align: center                                                               |
    |                                                                                |
    |   Improper angle sketch defined between four atoms.                            |  
    +--------------------------------------------------------------------------------+
    
     .. raw:: html

        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/qVATE-9cIBg" 
        frameborder="0" allowfullscreen>
        </iframe> 
        
        
    :Parameters:
        #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
    
    .. code-block:: python
        
        ## Tetrahydrofuran (THF) molecule sketch
        ## 
        ##              O
        ##   H41      /   \      H11
        ##      \  /         \  /
        ## H42-- C4    THF    C1 --H12
        ##        \  MOLECULE /
        ##         \         /
        ##   H31-- C3-------C2 --H21
        ##        /         \\
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
        self.__anglesList = [[],[],[],[],[],[]]      
        self.__angles     = {}
        # set computation cost
        self.set_computation_cost(3.0)
        # create dump flag
        self.__dumpAngles = True
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_ImproperAngleConstraint__anglesList',
                           '_ImproperAngleConstraint__angles',] )
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
            # set angles and reset constraint
            AL = [ self.__anglesList[0],self.__anglesList[1],
                   self.__anglesList[2],self.__anglesList[3],
                   [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[4]],
                   [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[5]] ]
            self.set_angles(anglesList=AL, tform=False) 
            # reset constraint is called in set_angles 
      
    #@raise_if_collected
    def set_angles(self, anglesList, tform=True):
        """ 
        Sets the angles dictionary by parsing the anglesList list.
        
        :Parameters:
            #. anglesList (list): The angles list definition.
              
               tuples format: every item must be a list of five items.\n
               #. First item: The improper atom index that must be in the plane.
               #. Second item: The index of the atom 'O' considered the origin of the plane.
               #. Third item: The index of the atom 'x' used to calculated 'Ox' vector.
               #. Fourth item: The index of the atom 'y' used to calculated 'Oy' vector.
               #. Fifth item: The minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Sixth item: The maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
                
               six vectors format: every item must be a list of five items.\n
               #. First item: List containing improper atom indexes that must be in the plane.
               #. Second item: List containing indexes of the atom 'O' considered the origin of the plane.
               #. Third item: List containing indexes of the atom 'x' used to calculated 'Ox' vector.
               #. Fourth item: List containing indexes of the atom 'y' used to calculated 'Oy' vector.
               #. Fifth item: List containing the minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Sixth item: List containing the maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
                  
           #. tform (boolean): set whether given anglesList follows tuples format, If not 
              then it must follow the six vectors one.
        """
        # check if bondsList is given
        if anglesList is None:
            anglesList = [[],[],[],[],[],[]]
            tform      = False 
        elif len(anglesList) == 6 and len(anglesList[0]) == 0:
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
                        angle = [anglesList[0][idx], anglesList[1][idx], anglesList[2][idx], anglesList[3][idx], anglesList[4][idx], anglesList[5][idx]]
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
                self.__angles[INT_TYPE(idx)] = self.__angles.get(INT_TYPE(idx), {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]}  )
        # dump to repository
        self._dump_to_repository({'_ImproperAngleConstraint__anglesList' :self.__anglesList,
                                  '_ImproperAngleConstraint__angles'     :self.__angles})
        # reset constraint
        self.reset_constraint()
    
    #@raise_if_collected
    def add_angle(self, angle):
        """
        Add a single angle to the list of constraint angles. All angles are in degrees.
        
        :Parameters:
            #. angle (list): The angle list of five items.\n
               #. First item: The central atom index.
               #. Second item: The index of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The index of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Fifth item: The maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
               #. Sixth item: The maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
        """
        assert self.engine is not None, LOGGER.error("setting an angle is not allowed unless engine is defined.")
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        assert isinstance(angle, (list, set, tuple)), LOGGER.error("anglesList items must be lists")
        assert len(angle)==6, LOGGER.error("anglesList items must be lists of 6 items each")
        improperIdx, oIdx, xIdx, yIdx, lower, upper = angle
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
        # create improper angle
        if not self.__angles.has_key(improperIdx):
            anglesImproper = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]} 
        else:
            anglesImproper = {"oIdx"        :self.__angles[improperIdx]["oIdx"], 
                              "xIdx"        :self.__angles[improperIdx]["xIdx"], 
                              "yIdx"        :self.__angles[improperIdx]["yIdx"], 
                              "improperMap" :self.__angles[improperIdx]["improperMap"], 
                              "otherMap"    :self.__angles[improperIdx]["otherMap"] }          
        # create anglesO angle
        if not self.__angles.has_key(oIdx):
            anglesO = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]} 
        else:
            anglesO = {"oIdx"        :self.__angles[oIdx]["oIdx"], 
                       "xIdx"        :self.__angles[oIdx]["xIdx"], 
                       "yIdx"        :self.__angles[oIdx]["yIdx"], 
                       "improperMap" :self.__angles[oIdx]["improperMap"], 
                       "otherMap"    :self.__angles[oIdx]["otherMap"] }     
        # create anglesX angle
        if not self.__angles.has_key(xIdx):
            anglesX = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]} 
        else:
            anglesX = {"oIdx"        :self.__angles[xIdx]["oIdx"], 
                       "xIdx"        :self.__angles[xIdx]["xIdx"], 
                       "yIdx"        :self.__angles[xIdx]["yIdx"], 
                       "improperMap" :self.__angles[xIdx]["improperMap"], 
                       "otherMap"    :self.__angles[xIdx]["otherMap"] }
        # create anglesY angle
        if not self.__angles.has_key(yIdx):
            anglesY = {"oIdx":[],"xIdx":[],"yIdx":[],"improperMap":[],"otherMap":[]} 
        else:
            anglesY = {"oIdx"        :self.__angles[yIdx]["oIdx"], 
                       "xIdx"        :self.__angles[yIdx]["xIdx"], 
                       "yIdx"        :self.__angles[yIdx]["yIdx"], 
                       "improperMap" :self.__angles[yIdx]["improperMap"], 
                       "otherMap"    :self.__angles[yIdx]["otherMap"] }
        # check for re-defining
        setPos = oPos = xPos = yPos = None
        if oIdx in anglesImproper["oIdx"] and xIdx in anglesImproper["xIdx"] and yIdx in anglesImproper["yIdx"]:
            oPos = anglesImproper["oIdx"].index(oIdx)
            xPos = anglesImproper["xIdx"].index(xIdx)
            yPos = anglesImproper["yIdx"].index(yIdx)
        elif oIdx in anglesImproper["oIdx"] and yIdx in anglesImproper["xIdx"] and xIdx in anglesImproper["yIdx"]:
            oPos = anglesImproper["oIdx"].index(oIdx)
            xPos = anglesImproper["yIdx"].index(xIdx)
            yPos = anglesImproper["xIdx"].index(yIdx)
        elif xIdx in anglesImproper["oIdx"] and oIdx in anglesImproper["xIdx"] and yIdx in anglesImproper["yIdx"]:
            oPos = anglesImproper["xIdx"].index(oIdx)
            xPos = anglesImproper["oIdx"].index(xIdx)
            yPos = anglesImproper["yIdx"].index(yIdx)
        elif yIdx in anglesImproper["oIdx"] and xIdx in anglesImproper["xIdx"] and oIdx in anglesImproper["yIdx"]:
            oPos = anglesImproper["yIdx"].index(oIdx)
            xPos = anglesImproper["xIdx"].index(xIdx)
            yPos = anglesImproper["oIdx"].index(yIdx)
        if oPos is not None and (oPos==xPos) and (oPos==yPos):
            LOGGER.warn("Angle definition for improper atom index '%i' and O '%i' and X '%i' and Y '%i' atoms is  already defined. New angle limits [%.3f,%.3f] are set."%(improperIdx, oIdx, xIdx, yIdx, lower, upper))
            setPos = anglesImproper["improperMap"][oPos]
        # set angle
        if setPos is None:
            anglesImproper["oIdx"].append(oIdx)        
            anglesImproper["xIdx"].append(xIdx)                
            anglesImproper["yIdx"].append(yIdx)        
            anglesImproper["improperMap"].append( len(self.__anglesList[0]) )
            anglesO["otherMap"].append( len(self.__anglesList[0]) )
            anglesX["otherMap"].append( len(self.__anglesList[0]) )
            anglesY["otherMap"].append( len(self.__anglesList[0]) )
            self.__anglesList[0] = np.append(self.__anglesList[0],improperIdx)
            self.__anglesList[1] = np.append(self.__anglesList[1],oIdx)
            self.__anglesList[2] = np.append(self.__anglesList[2],xIdx)
            self.__anglesList[3] = np.append(self.__anglesList[3],yIdx)
            self.__anglesList[4] = np.append(self.__anglesList[4],lower)
            self.__anglesList[5] = np.append(self.__anglesList[5],upper)
            
        else:
            assert self.__anglesList[0][setPos] == improperIdx, LOGGER.error("mismatched angles improper atom '%s' and '%s'"%(self.__anglesList[0][setPos],improperIdx))
            assert sorted([oIdx, xIdx, yIdx]) == sorted([self.__anglesList[1][setPos],self.__anglesList[2][setPos],self.__anglesList[3][setPos]]), LOGGER.error("mismatched angles O, Y and Y at improper atom '%s'"%(improperIdx))
            self.__anglesList[1][setPos] = oIdx
            self.__anglesList[2][setPos] = xIdx
            self.__anglesList[3][setPos] = yIdx
            self.__anglesList[4][setPos] = lower
            self.__anglesList[5][setPos] = upper
        self.__angles[improperIdx] = anglesImproper
        self.__angles[oIdx]        = anglesO
        self.__angles[xIdx]        = anglesX
        self.__angles[yIdx]        = anglesY
        # dump to repository
        if self.__dumpAngles:
            self._dump_to_repository({'_ImproperAngleConstraint__anglesList' :self.__anglesList,
                                      '_ImproperAngleConstraint__angles'     :self.__angles})
            # reset constraint
            self.reset_constraint() 
    
    #@raise_if_collected
    def create_angles_by_definition(self, anglesDefinition):
        """ 
        Creates anglesList using angles definition. This calls set_angles(anglesMap) 
        and generates angles attribute. All angles are in degrees.
        
        :Parameters:
            #. anglesDefinition (dict): The angles definition. 
               Every key must be a molecule name (residue name in pdb file). 
               Every key value must be a list of angles definitions. 
               Every angle definition is a list of five items where:
               
               #. First item: The name of the improper atom that must be in the plane.
               #. Second item: The name of the atom 'O' considered the origin of the plane.
               #. Third item: The name of the atom 'x' used to calculated 'Ox' vector.
               #. Fourth item: The name of the atom 'y' used to calculated 'Oy' vector.
               #. Fifth item: The minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Sixth item: The maximum upper limit or the maximum angle allowed 
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
                assert len(angle)==6
                improperAt, oAt, xAt, yAt, lower, upper = angle
                # check for redundancy
                append = True
                for b in molAnglesList:
                    if (b[0]==improperAt):
                        if sorted([oAt,xAt,yAt]) == sorted([b[1],b[2],b[3]]):
                            LOGGER.warn("Redundant definition for anglesDefinition found. The later '%s' is ignored"%str(b))
                            append = False
                            break
                if append:
                    molAnglesList.append((improperAt, oAt, xAt, yAt, lower, upper))
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
                improperIdx = indexes[ names.index(angle[0]) ]
                oIdx        = indexes[ names.index(angle[1]) ]
                xIdx        = indexes[ names.index(angle[2]) ]
                yIdx        = indexes[ names.index(angle[3]) ]
                lower       = angle[4]
                upper       = angle[5]
                anglesList.append((improperIdx, oIdx, xIdx, yIdx, lower, upper))
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
            #. realIndexes (numpy.ndarray): Group atoms relative indexes the move 
               will be applied to
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
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
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
            #. realIndexes (numpy.ndarray): atom index as a numpy array of a single 
               element.
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
 
        
        
    def plot(self, ax=None, nbins=50, subplots=True, split=None, 
                   wspace=0.3, hspace=0.3,
                   histtype='bar', lineWidth=None, lineColor=None,
                   xlabel=True, xlabelSize=16,
                   ylabel=True, ylabelSize=16,
                   legend=True, legendCols=1, legendLoc='best',
                   title=True, titleStdErr=True, titleAtRem=True,
                   titleUsedFrame=True, show=True):
        """ 
        Plot improper angles constraint distribution histogram.
        
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
            
        +------------------------------------------------------------------------------+ 
        |.. figure:: improper_angle_constraint_plot_method.png                         | 
        |   :width: 530px                                                              | 
        |   :height: 400px                                                             |
        |   :align: left                                                               | 
        +------------------------------------------------------------------------------+
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
        atom2 = self.__anglesList[0]
        atom1 = self.__anglesList[1]
        atom3 = self.__anglesList[2]
        atom4 = self.__anglesList[3]
        lower = self.__anglesList[4]
        upper = self.__anglesList[5]
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
            l = lower[idx]
            u = upper[idx]
            k = (a1,a2,a3,a4,l,u)
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
                N_AXES = N_AXES.flatten()
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
                a1,a2,a3,a4, L,U  = key
                L  = L*180./np.pi
                U  = U*180./np.pi
                LU = "(%.2f,%.2f)"%(L,U)
                label = "%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LU)
                COL  = COLORS[idx%len(COLORS)]
                AXES = N_AXES[idx]
                idxs = categories[key]
                data = self.data["angles"][idxs]*180./np.pi
                # get data limits
                mn = np.min(data)
                mx = np.max(data)
                # get bins
                BINS = _get_bins(dmin=mn, dmax=mx, boundaries=[L,U], nbins=nbins)
                # plot histogram
                D, _, P = AXES.hist(x=data, bins=BINS, 
                                    color=COL, label=label,
                                    histtype=histtype)
                # vertical lines
                Y = max(D)
                AXES.plot([L,L],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle='--')
                AXES.plot([U,U],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle='--')
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
                a1,a2,a3,a4, L,U  = key
                L  = L*180./np.pi
                U  = U*180./np.pi
                LU = "(%.2f,%.2f)"%(L,U)
                label = "%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LU)
                COL  = COLORS[idx%len(COLORS)]
                idxs = categories[key]
                data = self.data["angles"][idxs]*180./np.pi
                # get data limits
                mn = np.min(data)
                mx = np.max(data)
                # get bins
                BINS = _get_bins(dmin=mn, dmax=mx, boundaries=[L,U], nbins=nbins)
                # plot histogram
                D, _, P = AXES.hist(x=data, bins=BINS, 
                                    color=COL, label=label,
                                    histtype=histtype)
                # vertical lines
                Y = max(D)
                AXES.plot([L,L],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle='--')
                AXES.plot([U,U],[0,Y+0.1*Y], linewidth=1.0, color='k', linestyle='--')
                if lineWidth is not None:
                    [p.set_linewidth(lineWidth) for p in P]
                if lineColor is not None:
                    [p.set_edgecolor(lineColor) for p in P]
            # legend
            if legend:
                AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
            # set axis labels
            if xlabel:
                AXES.set_xlabel("$deg.$", size=xlabelSize)
            if ylabel:
                AXES.set_ylabel("$number$"  , size=ylabelSize)
            # update limits
            AXES.set_xmargin(0.1)
            AXES.autoscale()
                
        # set title
        if title:
            FIG.canvas.set_window_title('Improper Angle Constraint')
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
        atom2 = self.__anglesList[0]
        atom1 = self.__anglesList[1]
        atom3 = self.__anglesList[2]
        atom4 = self.__anglesList[3]
        lower = self.__anglesList[4]
        upper = self.__anglesList[5]
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
            l = lower[idx]
            u = upper[idx]
            k = (a1,a2,a3,a4,l,u)
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
            a1,a2,a3,a4, L,U   = key
            L  = L*180./np.pi
            U  = U*180./np.pi
            LU = "(%.2f,%.2f)"%(L,U)
            header.append( ("%s%s%s%s%s%s%s%s"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,'-'*(len(a1)>0),a4,LU)).replace(' ','') )
        data   = [categories[key] for key in sortCa]
        # save
        data = np.transpose(data)
        np.savetxt(fname     = fname, 
                   X         = data, 
                   fmt       = '%s', 
                   delimiter = delimiter, 
                   header    = " ".join(header),
                   comments  = comments)   
        
        
        
        
        
        

    
    

    
    
            