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
        
        
        
        
        
        
        
        














    
    





        