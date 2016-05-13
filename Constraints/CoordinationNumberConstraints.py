"""
CoordinationNumberConstraints contains classes for all constraints related to coordination numbers in shells around atoms.

.. inheritance-diagram:: fullrmc.Constraints.CoordinationNumberConstraints
    :parts: 1
    
"""

# standard libraries imports
import itertools
import copy

# external libraries imports
import numpy as np
from timeit import default_timer as timer

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint, QuasiRigidConstraint
from fullrmc.Core.atomic_coordination_number import single_atomic_coordination_number, multiple_atomic_coordination_number, full_atomic_coordination_number, atom_coordination_number_data


class AtomicCoordinationNumberConstraint(QuasiRigidConstraint, SingularConstraint):
    """
    It's a quasi-rigid constraint that controls the inter-molecular coordination number between atoms. 
    The maximum standard error is defined as the sum of minimum number of neighbours of all atoms
    in the system.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. defaultDistance (number): The minimum distance allowed set by default for all atoms type.
        #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules about how to differentiate between atoms and how to parse pairsLimits.   
        #. coordNumDef (None, dict): The coordination number definition. It must be a dictionary where keys are
               atoms type. Atoms type can be 'element' or 'name' as set by set_type_definition method.\n
               Every key value is a list of definitions. Every definition is a tuple or list of 5 items.\n
               #. the neighbouring atom type.
               #. the lower limit of the coordination shell.
               #. the upper limit of the coordination shell.
               #. the minimum number of neighbours in the shell.
               #. the maximum number of neighbours in the shell.
        #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
        #. thresholdRatio(Number): The threshold of satisfied data, above which the constraint become free.
           It must be between 0 and 1 where 1 means all data must be satisfied and therefore the constraint
           behave like a RigidConstraint and 0 means none of the data must be satisfied and therefore the
           constraint becomes always free and useless.
    """
    def __init__(self, engine, typeDefinition="element", coordNumDef=None, rejectProbability=1, thresholdRatio=0.8):
        # initialize constraint
        QuasiRigidConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability, thresholdRatio=thresholdRatio)
        # set type definition
        self.__coordNumDefinition = coordNumDef
        self.set_type_definition(typeDefinition)

    @property
    def typeDefinition(self):
        """Get types definition."""    
        return self.__typeDefinition
        
    @property
    def coordNumDefinition(self):
        """Get coordination number definition dictionary"""
        return self.__coordNumDefinition
        
    def listen(self, message, argument=None):
        """   
        listen to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed", "update molecules indexes"):
            self.set_type_definition(typeDefinition=self.__typeDefinition, coordNumDef = self.__coordNumDefinition)
        elif message in("update boundary conditions",):
            self.reset_constraint()        
    
    def set_type_definition(self, typeDefinition, coordNumDef=None):
        """
        Sets the atoms typing definition.
        
        :Parameters:
            #. typeDefinition (string): Can be either 'element' or 'name'. 
               It sets the rules about how to differentiate between atoms and how to parse pairsLimits.   
            #. coordNumDef (None, dict): The coordination number definition. It must be a dictionary where keys are
               atoms type. Atoms type can be 'element' or 'name' as set by set_type_definition method.\n
               Every key value is a list of definitions. Every definition is a tuple or list of 5 items.\n
               #. the neighbouring atom type.
               #. the lower limit of the coordination shell.
               #. the upper limit of the coordination shell.
               #. the minimum number of neighbours in the shell.
               #. the maximum number of neighbours in the shell.
        """
        # set typeDefinition
        assert typeDefinition in ("name", "element"), LOGGER.error("typeDefinition must be either 'name' or 'element'")
        if self.engine is None:
            self.__types                = []
            self.__typesLUT             = {}
            self.__typesInverseLUT      = {}
            self.__allTypes             = []
            self.__numberOfTypes        = []
            self.__typesIndexes         = []
            self.__numberOfAtomsPerType = []
        elif typeDefinition == "name":
            self.__types                = self.engine.names
            self.__typesLUT             = dict(zip(self.__types,range(len(self.__types))))
            self.__typesInverseLUT      = {v: k for k, v in self.__typesLUT.items()}
            self.__allTypes             = self.engine.allNames
            self.__numberOfTypes        = self.engine.numberOfNames
            self.__typesIndexes         = self.engine.namesIndexes
            self.__numberOfAtomsPerType = self.engine.numberOfAtomsPerName  
        elif typeDefinition == "element":
            self.__types                = self.engine.elements
            self.__typesLUT             = dict(zip(self.__types,range(len(self.__types))))
            self.__typesInverseLUT      = {v: k for k, v in self.__typesLUT.items()}
            self.__allTypes             = self.engine.allElements
            self.__numberOfTypes        = self.engine.numberOfElements
            self.__typesIndexes         = self.engine.elementsIndexes
            self.__numberOfAtomsPerType = self.engine.numberOfAtomsPerElement
        # get type indexes LUT
        self.__typeIndexesLUT = dict([(idx,[]) for idx in self.__typesLUT.values()])
        for idx in range(len(self.__typesIndexes)):
            self.__typeIndexesLUT[self.__typesIndexes[idx]].append(idx)
        for k,v in self.__typeIndexesLUT.items():
            self.__typeIndexesLUT[k] = np.array(v, dtype=INT_TYPE)
        #for typeIdx in 
        self.__typeDefinition = typeDefinition
        # set coordination number definition
        if coordNumDef is None:
            coordNumDef = self.__coordNumDefinition
        self.set_coordination_number_definition(coordNumDef)
        
    def set_coordination_number_definition(self, coordNumDef):
        """
        Set the coordination number definition.

        :Parameters:
            #. coordNumDef (None, dict): The coordination number definition. It must be a dictionary where keys are
               atoms type. Atoms type can be 'element' or 'name' as set by set_type_definition method.\n
               Every key value is a list of definitions. Every definition is a tuple or list of 5 items.\n
               #. the neighbouring atom type.
               #. the lower limit of the coordination shell.
               #. the upper limit of the coordination shell.
               #. the minimum number of neighbours in the shell.
               #. the maximum number of neighbours in the shell.
               
               ::

                   e.g. {"H":  [ ('C',1.,2.,3,3), ('H',2.,4.5,5,6), ('H',3.5, 5,6,7), ...], "O": ... }      
                   
        """
        if self.engine is None:
            self.__coordNumDefinition = coordNumDef
            self.__coordNumData       = None
            return
        elif coordNumDef is None:
            coordNumDef = {}
        else:
            assert isinstance(coordNumDef, dict), LOGGER.error("coordNumDef must be a dictionary")
            for key, cnValues in coordNumDef.items():
                keyValues = []
                assert key in self.__types, LOGGER.error("coordNumDef key '%s' is not a valid type %s."%(key, self.__types))
                assert isinstance(cnValues, (list, set, tuple)), LOGGER.error("Coordination number key '%s' definition value must be a list."%key)
                for cnEl in cnValues:
                    assert isinstance(cnEl, (list, set, tuple)), LOGGER.error("Coordination number key '%s' definition values must be a list of lists."%key)    
                    cnEl = list(cnEl)
                    assert len(cnEl)==5, LOGGER.error("Coordination number key '%s' definition values must be a list of lists of length 5 each."%key)    
                    # check every and each item of the definition
                    atomType, minDis, maxDis, minNum, maxNum = cnEl
                    assert atomType in self.__types, LOGGER.error("Coordination number key '%s'. Definition first value must be a valid type %s."%(key, self.__types))
                    assert is_number(minDis), LOGGER.error("Coordination number key '%s'. Definition second value must be a number."%key)       
                    minDis = FLOAT_TYPE(minDis)
                    assert minDis>=0, LOGGER.error("Coordination number key '%s'. Definition second value must be a positive number."%key)       
                    assert is_number(maxDis), LOGGER.error("Coordination number key '%s'. Definition third value must be a number."%key)       
                    maxDis = FLOAT_TYPE(maxDis)
                    assert maxDis>minDis, LOGGER.error("Coordination number key '%s'. Definition third value must be bigger than the third."%key)       
                    assert is_integer(minNum), LOGGER.error("Coordination number key '%s'. Definition fourth value must be an integer bigger than 0. %s is given"%(key,minNum))       
                    minNum = INT_TYPE(minNum)
                    assert minNum>=0, LOGGER.error("Coordination number key '%s'. Definition fourth value must be a positive integer."%key)       
                    assert is_integer(maxNum), LOGGER.error("Coordination number key '%s'. Definition fifth value must be an integer bigger than 0. %s is given"%(key,maxNum))       
                    maxNum = INT_TYPE(maxNum)
                    assert maxNum>=minNum, LOGGER.error("Coordination number key '%s'. Definition fifth value must be bigger or equal to the fourth."%key)             
                    #assert is_number(weight), LOGGER.error("Coordination number key '%s'. Definition sixth value must be a number."%key)       
                    #weight = FLOAT_TYPE(weight)
                    #assert weight>0, LOGGER.error("Coordination number key '%s'. Definition second value must be a positive number."%key)       

                    # append definition
                    if (atomType, minDis, maxDis, minNum, maxNum) in keyValues:
                        LOGGER.warn("Coordination number key '%s'. Definition %s is redundant."%(key, (atomType, minDis, maxDis, minNum, maxNum)))
                    else:
                        keyValues.append( (atomType, minDis, maxDis, minNum, maxNum) )
                # update definition 
                coordNumDef[key] = keyValues
        # set coordination number dictionary definition
        self.__coordNumDefinition = coordNumDef
        # set coordinationNumberPerType
        typeData                 = {}
        typeData['lowerLimit']   = []
        typeData['upperLimit']   = []
        typeData['minNumOfNeig'] = []
        typeData['maxNumOfNeig'] = []
        typeData['neigTypeIdx']  = []
        self.__typesCoordNumDef  = {}
        # initialize typesCoordNumDef
        for typeName in self.__types:
            typeIdx  = self.__typesLUT[typeName] 
            self.__typesCoordNumDef[typeIdx] = copy.deepcopy(typeData)
        for typeName, cnValues in self.__coordNumDefinition.items():
            typeIdx  = self.__typesLUT[typeName] 
            data = self.__typesCoordNumDef[typeIdx]
            for atomType, minDis, maxDis, minNum, maxNum in cnValues:
                data['lowerLimit'].append(minDis)
                data['upperLimit'].append(maxDis)
                data['minNumOfNeig'].append(minNum)
                data['maxNumOfNeig'].append(maxNum)
                data['neigTypeIdx'].append(self.__typesLUT[atomType] )    
            # sort minimums and add data to coordNumPerType
            indexes = np.argsort(data['lowerLimit'])
            self.__typesCoordNumDef[typeIdx] = {}
            self.__typesCoordNumDef[typeIdx]['lowerLimit']   = np.array([data['lowerLimit'][idx]   for idx in indexes], dtype=FLOAT_TYPE)
            self.__typesCoordNumDef[typeIdx]['upperLimit']   = np.array([data['upperLimit'][idx]   for idx in indexes], dtype=FLOAT_TYPE)
            self.__typesCoordNumDef[typeIdx]['minNumOfNeig'] = np.array([data['minNumOfNeig'][idx] for idx in indexes], dtype=INT_TYPE)
            self.__typesCoordNumDef[typeIdx]['maxNumOfNeig'] = np.array([data['maxNumOfNeig'][idx] for idx in indexes], dtype=INT_TYPE)
            self.__typesCoordNumDef[typeIdx]['neigTypeIdx']  = np.array([data['neigTypeIdx'][idx]  for idx in indexes], dtype=INT_TYPE)
        # set coordination number data
        self.__coordNumData = []
        maxSquaredDev = 0
        for idx in range(len(self.__allTypes)):
            typeName = self.__allTypes[idx]
            typeIdx  = self.__typesLUT[typeName]
            typeDef  = self.__typesCoordNumDef.get(typeName, None)
            data     = {}
            if not len(self.__typesCoordNumDef[typeIdx]['neigTypeIdx']):
                data['neighbours'] = {}
                data['deviations'] = np.array([0], dtype=INT_TYPE)
            else:
                numberOfEntries = len(self.__typesCoordNumDef[typeIdx]['lowerLimit'])
                data['neighbours'] = {}
                data['deviations'] = -np.array(self.__typesCoordNumDef[typeIdx]['minNumOfNeig'], dtype=INT_TYPE)
            data['neighbouring']      = {}
            data['standardError'] = np.sum(data['deviations']**2)
            maxSquaredDev += data['standardError']
            self.__coordNumData.append(data)
        # set maximum squared deviation as the sum of all atoms atomDeviations
        self._set_maximum_standard_error(maxSquaredDev)
                    
    def compute_standard_error(self, data):
        """ 
        Compute the standard error (StdErr) of data not satisfying constraint conditions. 
        
        .. math::
            StdErr = \\sum \\limits_{i}^{A} \\sum \\limits_{s}^{S} 
            (D_{i,s})^{2}  \n

            D_{i,s}=\\begin{cases}
              n_{i,s}-N_{min,i,s}, & \\text{if $n_{i,s}<N_{min,i,s}$}.\\\\
              n_{i,s}-N_{max,i,s}, & \\text{if $n_{i,s}>N_{max,i,s}$}.\\\\
              0                  , & \\text{if $`N_{min,i,s}<=n_{i,s}<=N_{max,i,s}$}
            \\end{cases}
                
        Where:\n
        :math:`A`           is the total number of atoms in the system. \n
        :math:`S`           is the total number of shells defined for atom i. \n
        :math:`D_{i,s}`     is the deviations of the coordination number of atom i in shell s. \n
        :math:`n_{i,s}`     is the number of neighbours of atom i in shell s. \n
        :math:`N_{min,i,s}` is the defined minimum number of neighbours around atom i in shell s. \n
        :math:`N_{max,i,s}` is the defined maximum number of neighbours around atom i in shell s. \n
         

        :Parameters:
            #. data (numpy.array): The constraint value data to compute standardError.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        StdErr = 0
        for cn in data:
            StdErr += cn['standardError']
        return StdErr
        
    def get_constraint_value(self):
        """
        Gets squared deviation per shell definition
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        if self.data is None:
            log.LocalLogger("fullrmc").logger.warn("data must be computed first using 'compute_data' method.")
            return {}
        # initialize constraint data
        numericData = {}
        namesData   = {}
        for k1,v1 in self.__typesCoordNumDef.items():
            numericData[k1] = {}
            namesData[self.__typesInverseLUT[k1]]= {}
            for k2 in v1['neigTypeIdx']:
                numericData[k1][k2] = 0
                namesData[self.__typesInverseLUT[k1]][self.__typesInverseLUT[k2]] = 0
        # compute constraints data
        data = self.data
        for idx1 in range(len(self.data)): 
            t1 = self.__typesIndexes[idx1]
            for idx2 in range(len(self.__typesCoordNumDef[t1]['neigTypeIdx'])):
                t2 = self.__typesCoordNumDef[t1]['neigTypeIdx'][idx2]
                numericData[t1][t2] += abs(data[idx1]['deviations'][idx2])
                namesData[self.__typesInverseLUT[t1]][self.__typesInverseLUT[t2]] += abs(data[idx1]['deviations'][idx2])        
        # return data
        return numericData, namesData   
         
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        self.__coordNumData = full_atomic_coordination_number( boxCoords       = self.engine.boxCoordinates,
                                                               basis           = self.engine.basisVectors,
                                                               moleculeIndex   = self.engine.moleculesIndexes,
                                                               typesIndex      = self.__typesIndexes,
                                                               typesDefinition = self.__typesCoordNumDef,
                                                               typeIndexesLUT  = self.__typeIndexesLUT,
                                                               coordNumData    = self.__coordNumData)
        # update data
        self.set_data( self.__coordNumData )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        stdErr = self.compute_standard_error(data = self.__coordNumData)
        self.set_standard_error(stdErr)

    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        pass
           
    def compute_after_move(self, indexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[indexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[indexes] = movedBoxCoordinates
        # compute data after move
        neighboursOfAtom     = []
        neighboursShellIdx   = []
        neighbours           = [] 
        affectedAtomsIndexes = []
        for idx in indexes:
            data = atom_coordination_number_data( atomIndex       = idx,
                                                  boxCoords       = self.engine.boxCoordinates,
                                                  basis           = self.engine.basisVectors,
                                                  moleculeIndex   = self.engine.moleculesIndexes,
                                                  typesIndex      = self.__typesIndexes,
                                                  typesDefinition = self.__typesCoordNumDef)
            neighboursOfAtom.extend(data[0])
            neighboursShellIdx.extend(data[1])
            neighbours.extend(data[2]) 
            # append all affected atoms
            affectedAtomsIndexes.append(idx)
            affectedAtomsIndexes.extend(self.__coordNumData[idx]['neighbours'].keys())
            affectedAtomsIndexes.extend(self.__coordNumData[idx]['neighbouring'].keys())
            affectedAtomsIndexes.extend(data[0])   
            affectedAtomsIndexes.extend(data[2])   
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # get affected atoms indexes
        affectedAtomsIndexes = tuple(set(affectedAtomsIndexes))
        # affected atoms data
        affectedCoordNumData = [self.__coordNumData[idx] for idx in affectedAtomsIndexes]
        # set active atoms before move data
        self.set_active_atoms_data_before_move( {'affectedIndexes':affectedAtomsIndexes, 'data':copy.deepcopy(affectedCoordNumData)} ) 
        # remove atom from all coordination number data
        for atomIndex in indexes:
            # remove neighbours
            for atIdx, shIdx in self.__coordNumData[atomIndex]['neighbours'].items():
                self.__coordNumData[atIdx]['neighbouring'].pop(atomIndex)
            # remove neighbouring
            for atIdx, shIdx in self.__coordNumData[atomIndex]['neighbouring'].items():
                self.__coordNumData[atIdx]['neighbours'].pop(atomIndex)
            # reset atom coordination number data
            self.__coordNumData[atomIndex]['neighbours']   = {}
            self.__coordNumData[atomIndex]['neighbouring'] = {}             
        # append neighbours and neighbouring
        for idx in xrange(len(neighboursOfAtom)):
            atIdx    = neighboursOfAtom[idx]
            shellIdx = neighboursShellIdx[idx]
            neigIdx  = neighbours[idx]
            self.__coordNumData[atIdx]['neighbours'][neigIdx]   = shellIdx
            self.__coordNumData[neigIdx]['neighbouring'][atIdx] = shellIdx
        # update standard error
        for atIdx in affectedAtomsIndexes:
            atomTypeIndex  = self.__typesIndexes[atIdx]
            atomCnDef      = self.__typesCoordNumDef[atomTypeIndex]
            atomShellsType = atomCnDef['neigTypeIdx']
            atomNeighbours = self.__coordNumData[atIdx]['neighbours']
            # compute standard error
            deviations = np.zeros(len(atomCnDef['minNumOfNeig']), dtype=INT_TYPE)
            for shellIdx in atomNeighbours.values():
                deviations[shellIdx] += 1
            for shellIdx in range(len(atomCnDef['minNumOfNeig'])):
                if deviations[shellIdx] < atomCnDef['minNumOfNeig'][shellIdx]:
                    deviations[shellIdx] = deviations[shellIdx]-atomCnDef['minNumOfNeig'][shellIdx]
                elif deviations[shellIdx] > atomCnDef['maxNumOfNeig'][shellIdx]:
                    deviations[shellIdx] = deviations[shellIdx]-atomCnDef['maxNumOfNeig'][shellIdx]
                else:
                    deviations[shellIdx] = 0
            self.__coordNumData[atIdx]['deviations']  = deviations      
            self.__coordNumData[atIdx]['standardError'] = np.sum(self.__coordNumData[atIdx]['deviations']**2)   
        # set active atoms after move data
        self.set_active_atoms_data_after_move( {'affectedIndexes':affectedAtomsIndexes, 'data':affectedCoordNumData} )
        # set before move standard error
        BM = self.compute_standard_error(data = self.activeAtomsDataBeforeMove['data'])
        AM = self.compute_standard_error(data = affectedCoordNumData)
        SD = self.standardError-BM+AM
        self.set_after_move_standard_error( SD )

    def accept_move(self, indexes):
        """ 
        Accept move.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_standard_error(self.afterMoveStandardError)
        self.set_after_move_standard_error( None )
        
    def reject_move(self, indexes):
        """ 
        Reject move.
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # check if constraints data before are already computed
        if self.activeAtomsDataBeforeMove is None:
            return
        # reset data
        affectedIndexes = self.activeAtomsDataBeforeMove['affectedIndexes']
        afterMoveData   = self.activeAtomsDataBeforeMove['data']
        for idx1, idx2 in zip(affectedIndexes, range(len(affectedIndexes))):
            self.__coordNumData[idx1] = afterMoveData[idx2]
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_after_move_standard_error( None )


    
    
            