"""
AngleConstraints contains classes for all constraints related angles between atoms.

.. inheritance-diagram:: fullrmc.Constraints.AngleConstraints
    :parts: 1
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, raise_if_collected, reset_if_collected_out_of_date
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.angles import full_angles_coords




class BondsAngleConstraint(RigidConstraint, SingularConstraint):
    """
    Controls the angle defined between 3 defined atoms, a first atom called central
    and the remain two called left and right.
    
    
    +--------------------------------------------------------------------------------+
    |.. figure:: angleSketch.png                                                     |
    |   :width: 308px                                                                |
    |   :height: 200px                                                               |
    |   :align: center                                                               |
    |                                                                                |
    |   Angle sketch defined between three atoms.                                    |  
    +--------------------------------------------------------------------------------+
    
    .. raw:: html

        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/ezBbbO9IVig" 
        frameborder="0" allowfullscreen>
        </iframe>
        
    
    :Parameters:
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
    def __init__(self, anglesMap=None, rejectProbability=1):
        # initialize constraint
        RigidConstraint.__init__(self, rejectProbability=rejectProbability)
        # set atomsColletor data keys
        self._atomsCollector.set_data_keys( ['centralMap','otherMap'] )
        # init angles data
        self.__anglesList = [[],[],[],[],[]]      
        self.__angles     = {}
        # set computation cost
        self.set_computation_cost(2.0)
        # create dump flag
        self.__dumpAngles = True
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_BondsAngleConstraint__anglesList',
                           '_BondsAngleConstraint__angles',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',  tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA',tuple(RUNTIME_DATA) )
        
    @property
    def anglesList(self):
        """ Get defined angles list."""
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
            AL = [ self.__anglesList[0],self.__anglesList[1],self.__anglesList[2],
                   [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[3]],
                   [a*FLOAT_TYPE(180.)/PI for a in self.__anglesList[4]] ]
            self.set_angles( AL, tform=False )
            # reset constraint is called in set_angles
    
    #@raise_if_collected
    def set_angles(self, anglesList, tform=True):
        """ 
        Sets the angles dictionary by parsing the anglesMap list. All angles are in 
        degrees.
        
        :Parameters:
            #. anglesList (list): The angles list definition.
              
               tuples format:  every item must be a list of five items.\n
               #. First item: The central atom index.
               #. Second item: The index of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The index of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Fifth item: The maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
                  
               five vectors format:  List of exaclty five lists or numpy.arrays or vectors of the same length.\n
               #. First item: List containing central atom indexes.
               #. Second item: List containing the index of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: List containing the index of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: List containing the minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Fifth item: List containing the maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
            
           #. tform (boolean): set whether given anglesList follows tuples format, If not 
              then it must follow the four vectors one.
        """
        # check if bondsList is given
        if anglesList is None:
            bondsList = [[],[],[],[],[]]
            tform     = False 
        elif len(anglesList) == 5 and len(anglesList[0]) == 0:
            tform     = False 
        if self.engine is None:
            self.__anglesList = anglesList      
            self.__angles     = {}
        else:
            NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
            oldAnglesList = self.__anglesList
            oldAngles     = self.__angles
            self.__anglesList = [np.array([], dtype=INT_TYPE),
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
                        angle = [anglesList[0][idx], anglesList[1][idx], anglesList[2][idx], anglesList[3][idx], anglesList[4][idx]]
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
                self.__angles[INT_TYPE(idx)] = self.__angles.get(INT_TYPE(idx),  {"left":[],"right":[],"centralMap":[],"otherMap":[]}  )
        # dump to repository
        self._dump_to_repository({'_BondsAngleConstraint__anglesList' :self.__anglesList,
                                  '_BondsAngleConstraint__angles'     :self.__angles})
        # reset constraint
        self.reset_constraint()

    #@raise_if_collected
    def add_angle(self, angle):
        """
        Add a single angle to the list of constraint angles. All angles are in degrees.
        
        :Parameters:
            #. angle (list): The bond list of five items.\n
               #. First item: The central atom index.
               #. Second item: The index of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The index of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Fifth item: The maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
        """
        assert self.engine is not None, LOGGER.error("setting an angle is not allowed unless engine is defined.")
        NUMBER_OF_ATOMS = self.engine.get_original_data("numberOfAtoms")
        assert isinstance(angle, (list, set, tuple)), LOGGER.error("angle items must be lists")
        assert len(angle)==5, LOGGER.error("angle items must be lists of 5 items each")
        centralIdx, leftIdx, rightIdx, lower, upper = angle
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
        # create central angle
        if not self.__angles.has_key(centralIdx):
            anglesCentral = {"left":[],"right":[],"centralMap":[],"otherMap":[]} 
        else:
            anglesCentral = {"left"       :self.__angles[centralIdx]["left"], 
                             "right"      :self.__angles[centralIdx]["right"], 
                             "centralMap" :self.__angles[centralIdx]["centralMap"], 
                             "otherMap"   :self.__angles[centralIdx]["otherMap"] }
        # create left angle
        if not self.__angles.has_key(leftIdx):
            anglesLeft = {"left":[],"right":[],"centralMap":[],"otherMap":[]} 
        else:
            anglesLeft = {"left"       :self.__angles[leftIdx]["left"], 
                          "right"      :self.__angles[leftIdx]["right"], 
                          "centralMap" :self.__angles[leftIdx]["centralMap"], 
                          "otherMap"   :self.__angles[leftIdx]["otherMap"] }               
        # create right angle
        if not self.__angles.has_key(rightIdx):
            anglesRight = {"left":[],"right":[],"centralMap":[],"otherMap":[]} 
        else:
            anglesRight = {"left"       :self.__angles[rightIdx]["left"], 
                           "right"      :self.__angles[rightIdx]["right"], 
                           "centralMap" :self.__angles[rightIdx]["centralMap"], 
                           "otherMap"   :self.__angles[rightIdx]["otherMap"] }
                             
        # check for re-defining
        setPos = ileft = iright = None
        if leftIdx in anglesCentral["left"] and rightIdx in anglesCentral["right"]:
            ileft  = anglesCentral["left"].index(leftIdx)
            iright = anglesCentral["right"].index(rightIdx)
        elif leftIdx in anglesCentral["right"] and rightIdx in anglesCentral["left"]:
            ileft  = anglesCentral["right"].index(leftIdx)
            iright = anglesCentral["left"].index(rightIdx)
        if (ileft is not None) and (ileft == iright):
            LOGGER.warn("Angle definition for central atom index '%i' and interchangeable left '%i' atom and right '%i' atom is  already defined. New angle limits [%.3f,%.3f] are set."%(centralIdx, leftIdx, rightIdx, lower, upper))
            setPos = anglesCentral["centralMap"][ileft]
        # set angle
        if setPos is None:
            anglesCentral['left'].append(leftIdx)
            anglesCentral['right'].append(rightIdx)
            anglesCentral['centralMap'].append( len(self.__anglesList[0]) )
            anglesLeft['otherMap'].append( len(self.__anglesList[0]) )
            anglesRight['otherMap'].append( len(self.__anglesList[0]) )
            self.__anglesList[0] = np.append(self.__anglesList[0],centralIdx)
            self.__anglesList[1] = np.append(self.__anglesList[1],leftIdx)
            self.__anglesList[2] = np.append(self.__anglesList[2],rightIdx)
            self.__anglesList[3] = np.append(self.__anglesList[3],lower)
            self.__anglesList[4] = np.append(self.__anglesList[4],upper)
        else:
            assert self.__anglesList[0][setPos] == centralIdx, LOOGER.error("mismatched angles central atom '%s' and '%s'"%(elf.__anglesList[0][setPos],centralIdx))
            assert sorted([leftIdx,rightIdx]) == sorted([self.__anglesList[1][setPos],self.__anglesList[2][setPos]]), LOOGER.error("mismatched angles left and right at central atom '%s' and '%s'"%(elf.__anglesList[0][setPos],centralIdx))
            self.__anglesList[3][setPos] = lower
            self.__anglesList[4][setPos] = upper
        self.__angles[centralIdx] = anglesCentral
        self.__angles[leftIdx]    = anglesLeft
        self.__angles[rightIdx]   = anglesRight
        # dump to repository
        if self.__dumpAngles:
            self._dump_to_repository({'_BondsAngleConstraint__anglesList' :self.__anglesList,
                                      '_BondsAngleConstraint__angles'     :self.__angles})
            # reset constraint
            self.reset_constraint()
    
    #@raise_if_collected
    def create_angles_by_definition(self, anglesDefinition):
        """ 
        Creates anglesList using angles definition.
        Calls set_angles(anglesList) and generates angles attribute.
        
        :Parameters:
            #. anglesDefinition (dict): The angles definition. 
               Every key must be a molecule name (residue name in pdb file). 
               Every key value must be a list of angles definitions. 
               Every angle definition is a list of five items where:
               
               #. First item: The name of the central atom forming the angle.
               #. Second item: The name of the left atom forming the angle (interchangeable with the right atom).
               #. Third item: The name of the right atom forming the angle (interchangeable with the left atom).
               #. Fourth item: The minimum lower limit or the minimum angle allowed 
                  in degrees which later will be converted to rad.
               #. Fifth item: The maximum upper limit or the maximum angle allowed 
                  in degrees which later will be converted to rad.
        
        ::
        
            e.g. (Carbon tetrachloride):  anglesDefinition={"CCL4": [('C','CL1','CL2' , 105, 115),
                                                                     ('C','CL2','CL3' , 105, 115),
                                                                     ('C','CL3','CL4' , 105, 115),                                      
                                                                     ('C','CL4','CL1' , 105, 115) ] }
                                                                 
        """
        if self.engine is None:
            raise Exception(LOGGER.error("Engine is not defined. Can't create angles by definition"))
        assert isinstance(anglesDefinition, dict), LOGGER.error("anglesDefinition must be a dictionary")
        ALL_NAMES         = self.engine.get_original_data("allNames")
        NUMBER_OF_ATOMS   = self.engine.get_original_data("numberOfAtoms")
        MOLECULES_NAMES   = self.engine.get_original_data("moleculesNames")
        MOLECULES_INDEXES = self.engine.get_original_data("moleculesIndexes")
        # check map definition
        existingMoleculesNames = sorted(set(MOLECULES_NAMES))
        anglesDef = {}
        for mol, angles in anglesDefinition.items():
            if mol not in existingMoleculesNames:
                LOGGER.warn("Molecule name '%s' in anglesDefinition is not recognized, angles definition for this particular molecule is omitted"%str(mol))
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
                centralIdx = indexes[ names.index(angle[0]) ]
                leftIdx    = indexes[ names.index(angle[1]) ]
                rightIdx   = indexes[ names.index(angle[2]) ]
                lower      = angle[3]
                upper      = angle[4]
                anglesList.append((centralIdx, leftIdx, rightIdx, lower, upper))
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
        return FLOAT_TYPE( np.sum(data["reducedAngles"]**2) )

    def get_constraint_value(self):
        """
        Computes all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
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
            central = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes])
            left    = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes])
            right   = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes])
            lowerLimit = self.__anglesList[3][anglesIndexes]
            upperLimit = self.__anglesList[4][anglesIndexes]
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
            anglesData[anglesIndexes]  = angles
            reducedData[anglesIndexes] = reduced
            angles  = anglesData
            reduced = reducedData
        # set data
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
        Compute constraint before move is executed.
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to.
        """
        # get angles indexes
        anglesIndexes = []
        #for idx in relativeIndexes:
        for idx in realIndexes:
            anglesIndexes.extend( self.__angles[idx]['centralMap'] )
            anglesIndexes.extend( self.__angles[idx]['otherMap'] )
        anglesIndexes = list( set(anglesIndexes)-self._atomsCollector._randomData )
        # compute data before move
        if len(anglesIndexes):
            angles, reduced =  full_angles_coords( central            = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes]), 
                                                   left               = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes]), 
                                                   right              = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes]), 
                                                   lowerLimit         = self.__anglesList[3][anglesIndexes], 
                                                   upperLimit         = self.__anglesList[4][anglesIndexes], 
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
        Compute constraint after move is executed.
        
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
            angles, reduced =  full_angles_coords( central            = self._atomsCollector.get_relative_indexes(self.__anglesList[0][anglesIndexes]), 
                                                   left               = self._atomsCollector.get_relative_indexes(self.__anglesList[1][anglesIndexes]), 
                                                   right              = self._atomsCollector.get_relative_indexes(self.__anglesList[2][anglesIndexes]), 
                                                   lowerLimit         = self.__anglesList[3][anglesIndexes], 
                                                   upperLimit         = self.__anglesList[4][anglesIndexes], 
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
        Accept move.
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to.
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
        Reject move.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
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
            anglesIndexes.extend( self.__angles[idx]['centralMap'] )
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
            #. index (numpy.ndarray): atom index as a numpy array of a single element.
        """
        pass
    
    def _on_collector_collect_atom(self, realIndex):
        # get angle indexes
        AI = self.__angles[realIndex]['centralMap'] + self.__angles[realIndex]['otherMap']
        # append all mapped bonds to collector's random data
        self._atomsCollector._randomData = self._atomsCollector._randomData.union( set(AI) )
        # collect atom anglesIndexes
        self._atomsCollector.collect(realIndex, dataDict={'centralMap':self.__angles[realIndex]['centralMap'],
                                                          'otherMap':  self.__angles[realIndex]['otherMap']})
    
    def plot(self, ax=None, nbins=25, subplots=True, split=None,
                   wspace=0.3, hspace=0.3,
                   histtype='bar', lineWidth=None, lineColor=None,
                   xlabel=True, xlabelSize=16,
                   ylabel=True, ylabelSize=16,
                   legend=True, legendCols=1, legendLoc='best',
                   title=True, titleStdErr=True, titleAtRem=True,
                   titleUsedFrame=True, show=True):
        """ 
        Plot angles constraint distribution histogram.
        
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
        |.. figure:: bonds_angle_constraint_plot_method.png                            | 
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
        # import matplotlib
        import matplotlib.pyplot as plt
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
        lower = self.__anglesList[3]
        upper = self.__anglesList[4]
        for idx in xrange(self.__anglesList[0].shape[0]):
            if self._atomsCollector.is_collected(idx):
                continue
            if splitV is not None:
                a1 = splitV[ atom1[idx] ]
                a2 = splitV[ atom2[idx] ]
                a3 = splitV[ atom3[idx] ]
            else:
                a1 = a2 = a3 = ''
            l = lower[idx]
            u = upper[idx]
            k = (a1,a2,a3,l,u)
            L = categories.get(k, [])
            L.append(idx)
            categories[k] = L
        ncategories = len(categories.keys())
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
                a1,a2,a3, L,U  = key
                L = L*180./np.pi
                U = U*180./np.pi
                label = "%s%s%s%s%s(%.2f,%.2f)"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,L,U)
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
                a1,a2,a3, L,U  = key
                L = L*180./np.pi
                U = U*180./np.pi
                label = "%s%s%s%s%s(%.2f,%.2f)"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,L,U)
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
            FIG.canvas.set_window_title('Bonds Angle Constraint')
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
        lower = self.__anglesList[3]
        upper = self.__anglesList[4]
        for idx in xrange(self.__anglesList[0].shape[0]):
            if self._atomsCollector.is_collected(idx):
                continue
            if splitV is not None:
                a1 = splitV[ atom1[idx] ]
                a2 = splitV[ atom2[idx] ]
                a3 = splitV[ atom3[idx] ]
            else:
                a1 = a2 = a3 = ''
            l = lower[idx]
            u = upper[idx]
            k = (a1,a2,a3,l,u)
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
        header = [("%s%s%s%s%s(%.2f,%.2f)"%(a1,'-'*(len(a1)>0),a2,'-'*(len(a1)>0),a3,L,U)).replace(" ","_") for a1,a2,a3,L,U in sortCa]
        data   = [categories[key] for key in sortCa]
        # save
        data = np.transpose(data)
        np.savetxt(fname     = fname, 
                   X         = data, 
                   fmt       = '%s', 
                   delimiter = delimiter, 
                   header    = " ".join(header),
                   comments  = comments)
        
        
        
        
        
        
        
        














    
    





        