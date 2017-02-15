"""
DistanceConstraints contains classes for all constraints related distances between atoms.

.. inheritance-diagram:: fullrmc.Constraints.DistanceConstraints
    :parts: 1
"""

# standard libraries imports
import itertools

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, raise_if_collected, reset_if_collected_out_of_date
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.atomic_distances import multiple_atomic_distances_coords, full_atomic_distances_coords, pair_elements_stats

class _DistanceConstraint(RigidConstraint, SingularConstraint):
    """
    This is the design class for all distance related constraints. 
    It contains all common methods and definitions but it's forbiden to be instantiated.
    """
    def __init__(self, defaultDistance, typeDefinition, 
                       pairsDistanceDefinition, flexible, rejectProbability):
        # disable instantiation
        assert not self.__class__.__name__ == "_DistanceConstraint", LOGGER.error("Instantiating '_DistanceConstraint' is forbidden.")
        # initialize constraint
        RigidConstraint.__init__(self, rejectProbability=rejectProbability)
        # set atoms collector data keys
        self._atomsCollector.set_data_keys( ('typeIndexes', 'allTypes') )  
        # set defaultDistance
        self.set_default_distance(defaultDistance)
        # set flexible
        self.set_flexible(flexible)
        # set type definition
        self.__pairsDistanceDefinition = None
        self.set_type_definition(typeDefinition, pairsDistanceDefinition)
        # set computation cost
        self.set_computation_cost(4.0)
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_DistanceConstraint__defaultDistance',
                           '_DistanceConstraint__pairsDistanceDefinition',
                           '_DistanceConstraint__pairsDistance',
                           '_DistanceConstraint__flexible',
                           '_DistanceConstraint__lowerLimitArray',
                           '_DistanceConstraint__upperLimitArray',
                           '_DistanceConstraint__typePairs',
                           '_DistanceConstraint__typePairsIndexes',
                           '_DistanceConstraint__typeDefinition',
                           '_DistanceConstraint__types',
                           '_DistanceConstraint__allTypes',
                           '_DistanceConstraint__numberOfTypes',
                           '_DistanceConstraint__typeIndexes',
                           '_DistanceConstraint__numberOfAtomsPerType',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( ['_DistanceConstraint__typeIndexes',
                              '_DistanceConstraint__allTypes',
                              '_DistanceConstraint__numberOfAtomsPerType'] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA', tuple(RUNTIME_DATA) )
       
    @property
    def defaultDistance(self):
        """ Gets the experimental data distances minimum. """
        return self.__defaultDistance
             
    @property
    def pairsDistanceDefinition(self):
        """ Get elements pairs """
        return self.__pairsDistanceDefinition
    
    @property
    def pairsDistance(self):
        """ Get elements pairs """
        return self.__pairsDistance
    
    @property
    def flexible(self):
        """ Get flexible flag """
        return self.__flexible
        
    @property
    def lowerLimitArray(self):
        """ 
        Get lowerLimitArray used in distances calculation.
        for InterMolecularDistanceConstraint it's always a numpy.zeros array        
        """
        return self.__lowerLimitArray
    
    @property
    def upperLimitArray(self):
        """ 
        Get upperLimitArray used in distances calculation.
        for InterMolecularDistanceConstraint it's the minimum distance allowed between pair of intermolecular atoms 
        """
        return self.__upperLimitArray
        
    @property
    def typePairs(self):
        """ Atoms type pairs sorted list"""
        return self.__typePairs
        
    @property
    def typeDefinition(self):
        """ atoms type definition. """
        return self.__typeDefinition
        
    @property
    def types(self):
        """ Defined atom types set. """
        return self.__types
    
    @property
    def allTypes(self):
        """ All atom types. """
        return self.__allTypes 
        
    @property
    def numberOfTypes(self):
        """ Number of defined atom types in the configuration. """
        return self.__numberOfTypes
        
    @property
    def typeIndexes(self):
        """ Type indexes list. """
        return self.__typeIndexes
    
    @property
    def typePairsIndexes(self):
        """numpy array look up for pairs types indexes."""
        return self.__typePairsIndexes
        
    @property
    def numberOfAtomsPerType(self):
        """ Get number of atoms per type dict. """
        return self.__numberOfAtomsPerType
        
    def _on_collector_reset(self):
        pass
        
    def listen(self, message, argument=None):
        """   
        listen to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine set", "update molecules indexes"):
            self.set_type_definition(self.__typeDefinition, self.__pairsDistanceDefinition)
            # reset constraint is called in set_paris_distance
        elif message in("update boundary conditions",):
            self.reset_constraint()
                
    def set_flexible(self, flexible):
        """ 
        Sets flexible flag.
        
        :Parameters:
            #. flexible (boolean): Whether to allow atoms to break constraints definition 
               under the condition of decreasing total standardError. If flexible is 
               set to False, atoms will never be allowed to cross from above to below 
               minimum allowed distance. Even if the later will decrease some other 
               unsatisfying atoms distances, and therefore the total standardError 
               of the constraint. 
        """
        assert isinstance(flexible, bool), LOGGER.error("flexible must be boolean")
        self.__flexible = flexible
        # dump to repository
        self._dump_to_repository({'_DistanceConstraint__flexible' :self.__flexible})
        
    def set_default_distance(self, defaultDistance):
        """ 
        Sets the default intermolecular minimum distance. 
        
        :Parameters:
            #. defaultDistance (number): The default minimum distance.
        """
        assert is_number(defaultDistance), LOGGER.error("defaultDistance must be a number")
        defaultDistance = FLOAT_TYPE(defaultDistance)
        assert defaultDistance>=0, LOGGER.error("defaultDistance must be positive")
        self.__defaultDistance = defaultDistance
        # dump to repository
        self._dump_to_repository({'_DistanceConstraint__defaultDistance' :self.__defaultDistance})
           
    def set_type_definition(self, typeDefinition, pairsDistanceDefinition=None):
        """ 
        Its an alias to set_pairs_distance used when typeDefinition needs to be re-defined.
        
        :Parameters:
            #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules about how to differentiate between atoms and how to parse pairsLimits.
            #. pairsDistanceDefinition (None, list, set, tuple): The minimum distance set to every pair of elements. 
               If None is given the already defined pairsDistanceDefinition will be used and passed to set_pairs_distance method.
        """
        # set typeDefinition
        assert typeDefinition in ("name", "element"), LOGGER.error("typeDefinition must be either 'name' or 'element'")
        if self.engine is None:
            types                = None
            allTypes             = None
            numberOfTypes        = None
            typeIndexes          = None
            numberOfAtomsPerType = None
        elif typeDefinition == "name":
            # copying because after loading and deserializing engine, pointer to
            # original data is lost and this will generate ambiguity in atoms collection 
            types                = self.engine.get_original_data("names")
            allTypes             = self.engine.get_original_data("allNames")
            numberOfTypes        = len(types)
            typeIndexes          = self.engine.get_original_data("namesIndexes")
            numberOfAtomsPerType = self.engine.get_original_data("numberOfAtomsPerName")
        elif typeDefinition == "element":
            types                = self.engine.get_original_data("elements")
            allTypes             = self.engine.get_original_data("allElements")
            numberOfTypes        = len(types)
            typeIndexes          = self.engine.get_original_data("elementsIndexes")
            numberOfAtomsPerType = self.engine.get_original_data("numberOfAtomsPerElement")
        # set type definition
        self.__typeDefinition       = typeDefinition
        self.__types                = types
        self.__allTypes             = allTypes
        self.__numberOfTypes        = numberOfTypes
        self.__typeIndexes          = typeIndexes 
        self.__numberOfAtomsPerType = numberOfAtomsPerType
        if self.__types is None:
            self.__typePairs        = None
            self.__typePairsIndexes = [[],[]]
        else:
            self.__typePairs = sorted(itertools.combinations_with_replacement(self.__types,2))
            self.__typePairsIndexes = [[],[]]
            for pair in self.__typePairs:
                idi = self.__types.index(pair[0])
                idj = self.__types.index(pair[1])
                self.__typePairsIndexes[0].append(idi)
                self.__typePairsIndexes[1].append(idj)
            self.__typePairsIndexes = np.transpose(self.__typePairsIndexes).astype(INT_TYPE)
        # dump to repository
        self._dump_to_repository({'_DistanceConstraint__typeDefinition'      :self.__typeDefinition,
                                  '_DistanceConstraint__types'               :self.__types,
                                  '_DistanceConstraint__allTypes'            :self.__allTypes,
                                  '_DistanceConstraint__numberOfTypes'       :self.__numberOfTypes,
                                  '_DistanceConstraint__typeIndexes'         :self.__typeIndexes,
                                  '_DistanceConstraint__numberOfAtomsPerType':self.__numberOfAtomsPerType,
                                  '_DistanceConstraint__typePairs'           :self.__typePairs,
                                  '_DistanceConstraint__typePairsIndexes'    :self.__typePairsIndexes})
        # set pair distance
        if pairsDistanceDefinition is None:
            pairsDistanceDefinition = self.__pairsDistanceDefinition
        self.set_pairs_distance(pairsDistanceDefinition)
    
    #@raise_if_collected
    def set_pairs_distance(self, pairsDistanceDefinition):
        """ 
        Sets the pairs intermolecular minimum distance. 
        
        :Parameters:
            #. pairsDistanceDefinition (None, list, set, tuple): The minimum distance set to every pair of elements. 
               A list of tuples must be given, all missing pairs will get automatically assigned the given defaultMinimumDistance value.
               First defined elements pair distance will cancel all redundant. 
               If None is given all pairs will be automatically generated and assigned the given defaultMinimumDistance value 
               
               ::
               
                   e.g. [('h','h',1.5), ('h','c',2.015), ...] 
               
        """
        if self.engine is None:
            newPairsDistance = pairsDistanceDefinition
        elif pairsDistanceDefinition is None:
            newPairsDistance = {}
            for el1 in self.__types:
                newPairsDistance[el1] = {}
                for el2 in self.__types:
                    newPairsDistance[el1][el2] = self.__defaultDistance
        else:
            newPairsDistance = {}
            assert isinstance(pairsDistanceDefinition, (list, set, tuple)), LOGGER.error("pairsDistanceDefinition must be a list")
            for pair in pairsDistanceDefinition:
                assert isinstance(pair, (list, set, tuple)), LOGGER.error("pairsDistanceDefinition list items must be lists as well")
                pair = list(pair)
                assert len(pair)==3, LOGGER.error("pairsDistanceDefinition list pair item list must have three items")
                if pair[0] not in self.__types:
                    LOGGER.warn("pairsDistanceDefinition list pair item '%s' is not a valid engine type '%s', definition item omitted"%(pair[0], self.__typeDefinition) )
                    continue
                if pair[1] not in self.__types: 
                    LOGGER.warn("pairsDistanceDefinition list pair item '%s' is not a valid engine type '%s', definition item omitted"%(pair[1], self.__typeDefinition) )
                    continue
                # create elements keys
                if not newPairsDistance.has_key(pair[0]):
                    newPairsDistance[pair[0]] = {}
                if not newPairsDistance.has_key(pair[1]):
                    newPairsDistance[pair[1]] = {}
                assert is_number(pair[2]), LOGGER.error("pairsDistanceDefinition list pair item list third item must be a number")
                distance = FLOAT_TYPE(pair[2])
                assert distance>=0, LOGGER.error("pairsDistanceDefinition list pair item list third item must be bigger than 0")
                # set minimum distance
                if newPairsDistance[pair[0]].has_key(pair[1]):
                    LOGGER.warn("types pair ('%s','%s') distance definition is redundant, '%s' is omitted"%(pair[0], pair[1], pair))
                else:
                    newPairsDistance[pair[0]][pair[1]] = distance
                if newPairsDistance[pair[1]].has_key(pair[0]) and pair[0]!=pair[1]:
                    LOGGER.warn("types pair ('%s','%s') distance definition is redundant, '%s' is omitted"%(pair[1], pair[0], pair))
                else:
                    newPairsDistance[pair[1]][pair[0]] = distance
            # complete not defined distances
            for el1 in self.__types:
                if not newPairsDistance.has_key(el1):
                    newPairsDistance[el1] = {}
                for el2 in self.__types:
                    if not newPairsDistance.has_key(el2):
                        newPairsDistance[el2] = {}
                    if not newPairsDistance[el1].has_key(el2): 
                        if newPairsDistance[el2].has_key(el1):
                            newPairsDistance[el1][el2] = newPairsDistance[el2][el1]
                        else:
                            LOGGER.warn("types pair ('%s','%s') distance definition is not defined and therefore it is set to the default distance '%s'"%(el1, el2, self.__defaultDistance))
                            newPairsDistance[el1][el2] = self.__defaultDistance
                            newPairsDistance[el2][el1] = self.__defaultDistance
                    assert newPairsDistance[el1][el2] == newPairsDistance[el2][el1], LOGGER.error("types '%s', and '%s' pair distance definitions are in conflict. (%s,%s, %s) and (%s,%s, %s)"%(el1,el2, el1,el2,newPairsDistance[el1][el2], el2,el1, newPairsDistance[el2][el1]))       
        # set new pairsDistance value
        self.__pairsDistanceDefinition = pairsDistanceDefinition
        self.__pairsDistance           = newPairsDistance
        #if self.__pairsDistance is not None:
        if self.engine is not None:
            self.__lowerLimitArray = np.zeros((self.__numberOfTypes, self.__numberOfTypes, 1), dtype=FLOAT_TYPE) 
            self.__upperLimitArray = np.zeros((self.__numberOfTypes, self.__numberOfTypes, 1), dtype=FLOAT_TYPE) 
            for idx1 in range(self.__numberOfTypes):
                el1 = self.__types[idx1]
                for idx2 in range(self.__numberOfTypes): 
                    el2  = self.__types[idx2]
                    dist = self.__pairsDistance[el1][el2]
                    self.__upperLimitArray[idx1,idx2,0] = FLOAT_TYPE(dist)
                    self.__upperLimitArray[idx2,idx1,0] = FLOAT_TYPE(dist)
        else:
            self.__lowerLimitArray = None     
            self.__upperLimitArray = None   
        # dump to repository
        self._dump_to_repository({'_DistanceConstraint__pairsDistanceDefinition': self.__pairsDistanceDefinition,
                                  '_DistanceConstraint__pairsDistance'          : self.__pairsDistance,
                                  '_DistanceConstraint__lowerLimitArray'        : self.__lowerLimitArray,
                                  '_DistanceConstraint__upperLimitArray'        : self.__upperLimitArray})  
        # reset constraint
        self.reset_constraint() # ADDED 2017-JAN-08
    
    def should_step_get_rejected(self, standardError):
        """
        Given a standardError, return whether to keep or reject new standardError 
        according to the constraint rejectProbability.
        In addition, if flexible flag is set to True, total number of atoms not satisfying 
        constraints definition must be decreasing or at least remain the same.
        
        :Parameters:
            #. standardError (number): The standardError to compare with the Constraint standardError
        
        :Returns:
            #. result (boolean): True to reject step, False to accept
        """
        if self.__flexible:
            # compute if step should get rejected as a RigidConstraint
            return super(_DistanceConstraint, self).should_step_get_rejected(standardError)
        else:
            cond = self.activeAtomsDataAfterMove["number"]>self.activeAtomsDataBeforeMove["number"]
            if np.any(cond):
                return True
            return False
    
    def _compute_standard_error(self, distances):          
        return FLOAT_TYPE( np.sum(distances) )
        
    def compute_standard_error(self, data):
        """ 
        Compute the standard error (stdErr) of data not satisfying constraint conditions. 
        
        .. math::
            stdErr = \\sum \\limits_{i}^{N} \\sum \\limits_{i+1}^{N} 
            \\left| d_{ij}-D_{ij}) \\right| 
            \\int_{0}^{D_{ij}} \\delta(x-d_{ij}) dx 
                  
        Where:\n
        :math:`N` is the total number of atoms in the system. \n
        :math:`D_{ij}` is the distance constraint set for atoms pair (i,j). \n
        :math:`d_{ij}` is the distance between atom i and atom j. \n
        :math:`\\delta` is the Dirac delta function. \n
        :math:`\\int_{0}^{D_{ij}} \\delta(x-d_{ij}) dx` 
        is equal to 1 if :math:`0 \\leqslant d_{ij} \\leqslant D_{ij}` and 0 elsewhere.\n 
    
        :Parameters:
            #. data (dict): The constraint value data returned using 
               get_constraint_value.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        standardError = np.sum(data.values())
        return FLOAT_TYPE(standardError)
    
    def _get_constraint_value(self):
        if self.data is None:
            LOGGER.warn("data must be computed first using 'compute_data' method.")
            return np.array([], dtype=FLOAT_TYPE)
        # compute distances
        idi = self.__typePairsIndexes[:,0]
        idj = self.__typePairsIndexes[:,1]
        numbers   = (self.data["number"][idi,idj] + self.data["number"][idj,idi]).reshape(-1)
        distances = (self.data["distanceSum"][idi,idj] + self.data["distanceSum"][idj,idi]).reshape(-1)
        nonZero   = np.where(numbers)
        distances[nonZero] /= numbers[nonZero]
        return distances
        
    def get_constraint_value(self):
        """
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        if self.data is None:
            LOGGER.warn("data must be computed first using 'compute_data' method.")
            return {}
        # compute distances
        distances = self._get_constraint_value()
        # compute output
        output = {}
        for idx, pair in enumerate(self.__typePairs):
            key = "md_%s-%s"% pair
            output[key] = distances[idx]
        # return
        return output
    
    def _on_collector_collect_atom(self, realIndex):
        # get relative index
        relativeIndex = self._atomsCollector.get_relative_index(realIndex)
        # create dataDict
        dataDict = {}
        dataDict['typeIndexes'] = self.typeIndexes[relativeIndex]
        dataDict['allTypes']    = self.allTypes[relativeIndex]
        # reduce all indexes above relativeIndex in typeIndexes
        # delete data
        self.__typeIndexes = np.delete(self.__typeIndexes, relativeIndex, axis=0)
        self.__allTypes    = np.delete(self.__allTypes,     relativeIndex, axis=0)
        self.__numberOfAtomsPerType[dataDict['allTypes']] -= 1
        # collect atom
        self._atomsCollector.collect(realIndex, dataDict=dataDict)
            
    def _on_collector_release_atom(self, realIndex):
        pass
        
        
class _MolecularDistanceConstraint(_DistanceConstraint):  
    """
    This is the design class for all molecular distance related constraints. 
    It contains all common methods and definitions but it's forbiden to be instantiated.
    """ 
    def __init__(self, *args, **kwargs):
        assert not self.__class__.__name__ == "_MolecularDistanceConstraint", LOGGER.error("Instantiating '_MolecularDistanceConstraint' is forbidden.")
        # check customizable flags
        assert self._interMolecular or self._intraMolecular, LOGGER.error("In _MolecularDistanceConstraint either '_interMolecular' or '_intraMolecular' must be set to True")
        assert not (self._interMolecular and self._intraMolecular), LOGGER.error("In _MolecularDistanceConstraint both '_interMolecular' and '_intraMolecular' can't be set to True")
        # initialize constraint
        super(_MolecularDistanceConstraint, self).__init__(*args, **kwargs) 
        # creating fixed flags
        object.__setattr__(self, '_countWithinLimits', True)
        object.__setattr__(self, '_reduceDistance', False)
        object.__setattr__(self, '_reduceDistanceToUpper', True)
        object.__setattr__(self, '_reduceDistanceToLower', False)
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_MolecularDistanceConstraint__typesStats', ] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA)   )
    
    
    def __setattr__(self, name, value):
        if name in ('_interMolecular','_intraMolecular','_countWithinLimits','_reduceDistance','_reduceDistanceToUpper','_reduceDistanceToLower'):
            raise Exception( LOGGER.error("setting '%'s is forbiden"%name) )
        object.__setattr__(self, name, value)
        
    
    def set_type_definition(self, typeDefinition, pairsDistanceDefinition=None):
        """ 
        Its an alias to set_pairs_distance used when typeDefinition needs to be re-defined.
        
        :Parameters:
            #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules about how to differentiate between atoms and how to parse pairsLimits.
            #. pairsDistanceDefinition (None, list, set, tuple): The minimum distance set to every pair of elements. 
               If None is given the already defined pairsDistanceDefinition will be used and passed to set_pairs_distance method.
        """
        super(_MolecularDistanceConstraint, self).set_type_definition(typeDefinition          = typeDefinition, 
                                                                      pairsDistanceDefinition = pairsDistanceDefinition) 
        # create stats
        stats = None
        if self.types is None:
            stats = None
        elif self._interMolecular:
            _, stats = pair_elements_stats(elementIndex     = self.typeIndexes,
                                           numberOfElements = self.numberOfTypes,
                                           moleculeIndex    = self.engine.moleculesIndexes)
        else:
            stats, _ = pair_elements_stats(elementIndex     = self.typeIndexes,
                                           numberOfElements = self.numberOfTypes,
                                           moleculeIndex    = self.engine.moleculesIndexes)
        # set types stats
        self.__typesStats = stats
        # dump to repository
        self._dump_to_repository({'_MolecularDistanceConstraint__typesStats' :self.__typesStats})

    @property
    def typesStats(self):
        """numpy array of number of found pairs in system"""
        return self.__typesStats
        
    @reset_if_collected_out_of_date
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        # compute data
        nintra,dintra, ninter,dinter = \
        full_atomic_distances_coords( boxCoords             = self.engine.boxCoordinates,
                                      basis                 = self.engine.basisVectors,
                                      isPBC                 = self.engine.isPBC,
                                      moleculeIndex         = self.engine.moleculesIndexes,
                                      elementIndex          = self.typeIndexes,
                                      numberOfElements      = self.numberOfTypes,
                                      lowerLimit            = self.lowerLimitArray,
                                      upperLimit            = self.upperLimitArray,
                                      interMolecular        = self._interMolecular,
                                      intraMolecular        = self._intraMolecular,
                                      reduceDistance        = self._reduceDistance,
                                      reduceDistanceToUpper = self._reduceDistanceToUpper,
                                      reduceDistanceToLower = self._reduceDistanceToLower,
                                      countWithinLimits     = self._countWithinLimits,
                                      ncores                = self.engine._runtime_ncores) 
        if self._interMolecular:
            number      = ninter
            distanceSum = dinter
        else:
            number      = nintra
            distanceSum = dintra
        # update data
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        self.set_standard_error( self._compute_standard_error(distances = self._get_constraint_value()) )
        # set original data
        if self.originalData is None:
            self._set_original_data(self.data)
    
    def compute_before_move(self, realIndexes, relativeIndexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        nintraM,dintraM, ninterM,dinterM = \
        multiple_atomic_distances_coords( indexes               = relativeIndexes,
                                          boxCoords             = self.engine.boxCoordinates,
                                          basis                 = self.engine.basisVectors,
                                          isPBC                 = self.engine.isPBC,
                                          moleculeIndex         = self.engine.moleculesIndexes,
                                          elementIndex          = self.typeIndexes,
                                          numberOfElements      = self.numberOfTypes,
                                          lowerLimit            = self.lowerLimitArray,
                                          upperLimit            = self.upperLimitArray,
                                          allAtoms              = True,
                                          countWithinLimits     = self._countWithinLimits,
                                          reduceDistance        = self._reduceDistance,
                                          reduceDistanceToUpper = self._reduceDistanceToUpper,
                                          reduceDistanceToLower = self._reduceDistanceToLower,
                                          interMolecular        = self._interMolecular,
                                          intraMolecular        = self._intraMolecular,
                                          ncores                = self.engine._runtime_ncores)    
        nintraF,dintraF, ninterF,dinterF = \
        full_atomic_distances_coords( boxCoords             = self.engine.boxCoordinates[relativeIndexes],
                                      basis                 = self.engine.basisVectors,
                                      isPBC                 = self.engine.isPBC,
                                      moleculeIndex         = self.engine.moleculesIndexes[relativeIndexes],
                                      elementIndex          = self.typeIndexes[relativeIndexes],
                                      numberOfElements      = self.numberOfTypes,
                                      lowerLimit            = self.lowerLimitArray,
                                      upperLimit            = self.upperLimitArray,
                                      interMolecular        = self._interMolecular,
                                      intraMolecular        = self._intraMolecular,
                                      reduceDistance        = self._reduceDistance,
                                      reduceDistanceToUpper = self._reduceDistanceToUpper,
                                      reduceDistanceToLower = self._reduceDistanceToLower,
                                      countWithinLimits     = self._countWithinLimits,
                                      ncores                = self.engine._runtime_ncores)
        # check if inter or intra
        if self._interMolecular:
            numberM      = ninterM
            distanceSumM = dinterM
            numberF      = ninterF
            distanceSumF = dinterF
        else:
            numberM      = nintraM
            distanceSumM = dintraM
            numberF      = nintraF
            distanceSumF = dintraF
        # set active atoms data
        self.set_active_atoms_data_before_move( {"number":numberM-numberF, "distanceSum":distanceSumM-distanceSumF} )
        self.set_active_atoms_data_after_move(None)
    
    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[relativeIndexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[relativeIndexes] = movedBoxCoordinates
        # calculate pair distribution function
        nintraM,dintraM, ninterM,dinterM = \
        multiple_atomic_distances_coords( indexes               = relativeIndexes,
                                          boxCoords             = self.engine.boxCoordinates,
                                          basis                 = self.engine.basisVectors,
                                          isPBC                 = self.engine.isPBC,
                                          moleculeIndex         = self.engine.moleculesIndexes,
                                          elementIndex          = self.typeIndexes,
                                          numberOfElements      = self.numberOfTypes,
                                          lowerLimit            = self.lowerLimitArray,
                                          upperLimit            = self.upperLimitArray,
                                          allAtoms              = True,
                                          countWithinLimits     = self._countWithinLimits,
                                          reduceDistance        = self._reduceDistance,
                                          reduceDistanceToUpper = self._reduceDistanceToUpper,
                                          reduceDistanceToLower = self._reduceDistanceToLower,
                                          interMolecular        = self._interMolecular,
                                          intraMolecular        = self._intraMolecular,
                                          ncores                = self.engine._runtime_ncores)               
        nintraF,dintraF, ninterF,dinterF = \
        full_atomic_distances_coords( boxCoords             = self.engine.boxCoordinates[relativeIndexes],
                                      basis                 = self.engine.basisVectors,
                                      isPBC                 = self.engine.isPBC,
                                      moleculeIndex         = self.engine.moleculesIndexes[relativeIndexes],
                                      elementIndex          = self.typeIndexes[relativeIndexes],
                                      numberOfElements      = self.numberOfTypes,
                                      lowerLimit            = self.lowerLimitArray,
                                      upperLimit            = self.upperLimitArray,
                                      interMolecular        = self._interMolecular,
                                      intraMolecular        = self._intraMolecular,
                                      reduceDistance        = self._reduceDistance,
                                      reduceDistanceToUpper = self._reduceDistanceToUpper,
                                      reduceDistanceToLower = self._reduceDistanceToLower,
                                      countWithinLimits     = self._countWithinLimits,
                                      ncores                = self.engine._runtime_ncores)
        # check if inter or intra
        if self._interMolecular:
            numberM      = ninterM
            distanceSumM = dinterM
            numberF      = ninterF
            distanceSumF = dinterF
        else:
            numberM      = nintraM
            distanceSumM = dintraM
            numberF      = nintraF
            distanceSumF = dintraF
        # set active atoms data
        self.set_active_atoms_data_after_move( {"number":numberM-numberF, "distanceSum":distanceSumM-distanceSumF} )
        # reset coordinates
        self.engine.boxCoordinates[relativeIndexes] = boxData
        # compute standardError after move
        number = self.data["number"]-self.activeAtomsDataBeforeMove["number"]+self.activeAtomsDataAfterMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]+self.activeAtomsDataAfterMove["distanceSum"]
        data = self.data
        # change temporarily data attribute
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        self.set_after_move_standard_error( self._compute_standard_error(distances = self._get_constraint_value()) )
        # change back data attribute
        self.set_data( data )
    
    def accept_move(self, realIndexes, relativeIndexes):
        """ 
        Accept move
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        number = self.data["number"]-self.activeAtomsDataBeforeMove["number"]+self.activeAtomsDataAfterMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]+self.activeAtomsDataAfterMove["distanceSum"]
        # change permanently data attribute
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_standard_error( self.afterMoveStandardError )
        self.set_after_move_standard_error( None )
    
    def reject_move(self, realIndexes, relativeIndexes):
        """ 
        Reject move
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_after_move_standard_error( None )

    def compute_as_if_amputated(self, realIndex, relativeIndex):
        """ 
        Compute and return constraint's data and standard error as if atom given its 
        its was amputated.
        
        :Parameters:
            #. index (numpy.ndarray): atom index as a numpy array of a single element.
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
        self.compute_before_move(realIndexes=realIndex, relativeIndexes=relativeIndex)
        #self.compute_before_move(indexes = np.array([index], dtype=INT_TYPE) )
        # change permanently data attribute
        number      = self.data["number"]-self.activeAtomsDataBeforeMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        # update standardError
        SE = self._compute_standard_error(distances = self._get_constraint_value()) 
        self.set_standard_error( SE )
    
    def reject_amputation(self, realIndex, relativeIndex):
        """ 
        Reject amputation of atom.
        
        :Parameters:
            #. realIndex (numpy.ndarray): atom index as a numpy array of a single element.
        """
        pass
    
    def plot(self, ax=None, width=0.6,
                   correctColor = '#0066ff',
                   erroneousColor = '#d62728',
                   xlabel=True, xlabelSize=16,
                   ylabel=True, ylabelSize=16,
                   legend=True, legendCols=1, legendLoc='best', legendText=('correct','erroneous'),
                   title=True, titleStdErr=True, titleAtRem=True,
                   titleUsedFrame=True, show=True):
        """ 
        Plot pair distribution constraint.
        
        :Parameters:
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If None is given a new plot figure will be created.
            #. width (number): Bars width, must be >0 and <=1
            #. correctColor (color): correct data bars color.
            #. erroneousColor (color): erroneous data bars color.
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
            #. legendText (list): The legend of number of correct pairs satisfying
               constraint and number of erroneous pairs unsatisfying constraint
               condition.
            #. title (boolean): Whether to create the title or not
            #. titleStdErr (boolean): Whether to show constraint standard error value in title.
            #. titleAtRem (boolean): Whether to show engine's number of removed atoms.
            #. titleUsedFrame(boolean): Whether to show used frame name in title.
            #. show (boolean): Whether to render and show figure before returning.
        
        :Returns:
            #. figure (matplotlib Figure): matplotlib used figure.
            #. axes (matplotlib Axes): matplotlib used axes.

        +------------------------------------------------------------------------------+ 
        |.. figure:: molecular_distances_constraint_plot_method.png                    | 
        |   :width: 530px                                                              | 
        |   :height: 400px                                                             |
        |   :align: left                                                               | 
        +------------------------------------------------------------------------------+
        """
        # get constraint value
        if self.data is None:
            LOGGER.warn("%s constraint data are not computed."%(self.__class__.__name__))
            return
        # check width
        assert 0<width<=1, LOGGER.error("width must be a number between 0 and 1")
        # import matplotlib
        import matplotlib.pyplot as plt
        # get axes
        if ax is None:  
            FIG  = plt.figure()
            AXES = plt.gca()
        else:
            AXES = ax   
            FIG  = AXES.get_figure()
        # get numbers and differences
        idi = self.typePairsIndexes[:,0]
        idj = self.typePairsIndexes[:,1]
        numbers = (self.data["number"][idi,idj] + self.data["number"][idj,idi]).reshape(-1).astype(FLOAT_TYPE)
        stats   = (self.typesStats[idi,idj] + self.typesStats[idj,idi]).reshape(-1).astype(FLOAT_TYPE)
        diff    = stats-numbers
        # plot bars  
        ind   = np.arange(1,len(numbers)+1)  # the x locations for the groups
        p1 = AXES.bar(ind, diff, width, color=correctColor, label=legendText[0])
        p2 = AXES.bar(ind, numbers, width, bottom=diff, color=erroneousColor, label=legendText[1])
        AXES.set_xlim(0,len(numbers)+1.5)
        AXES.set_ylim(0, max(stats)+0.15*max(stats))
        # set ticks
        plt.xticks(ind+width/2., ["%s-%s"%(el1,el2) for el1,el2 in self.typePairs])
        # ratio labels
        data = self._get_constraint_value()
        for d, rect in zip(data, AXES.patches):
            height = rect.get_height()
            t = AXES.text(x     = rect.get_x() + rect.get_width()/2, 
                          y     = height + 5, 
                          s     = " "+str(d), 
                          color = 'black',
                          rotation = 90,
                          horizontalalignment = 'center', 
                          verticalalignment   = 'bottom')
        # set axis labels
        if xlabel:
            AXES.set_xlabel("Type pairs", size=xlabelSize)
        if ylabel:
            AXES.set_ylabel("Number of pairs"  , size=ylabelSize)
        # set title
        if title:
            FIG.canvas.set_window_title('Molecular Distances Constraint')
            if titleUsedFrame:
                t = '$frame: %s$ : '%self.engine.usedFrame.replace('_','\_')
            else:
                t = ''
            if titleAtRem:
                t += "$%i$ $rem.$ $at.$ - "%(len(self.engine._atomsCollector))
            if titleStdErr and self.standardError is not None:
                t += "$std$ $error=%.6f$ "%(self.standardError)
            if len(t):
                AXES.set_title(t)  
        # set background color
        FIG.patch.set_facecolor('white')
        # plot legend
        if legend:
            AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
        #show
        if show:
            plt.show()
        # return axes
        return FIG, AXES
    
    def export(self, fname, delimiter='     ', comments='# '):
        """
        Export pair distribution constraint.
        
        :Parameters:
            #. fname (path): full file name and path.
            #. delimiter (string): String or character separating columns.
            #. comments (string): String that will be prepended to the header.
        """
        # get constraint value
        output = self.get_constraint_value()
        if not len(output):
            LOGGER.warn("%s constraint data are not computed."%(self.__class__.__name__))
            return
        # get numbers and differences
        idi = self.typePairsIndexes[:,0]
        idj = self.typePairsIndexes[:,1]
        numbers = (self.data["number"][idi,idj] + self.data["number"][idj,idi]).reshape(-1).astype(FLOAT_TYPE)
        stats   = (self.typesStats[idi,idj] + self.typesStats[idj,idi]).reshape(-1).astype(FLOAT_TYPE)
        diff    = stats-numbers
        # set data as strings
        stats   = [str(i) for i in stats]
        diff    = [str(i) for i in diff]
        numbers = [str(i) for i in numbers]
        pairs   = ["%s-%s"%(el1,el2) for el1,el2 in self.typePairs]
        stdErr  = self._get_constraint_value()
        # start creating header and data
        header = ["description","num_pairs","correct","erroneous","standard_error"]
        data   = [pairs,stats,diff,numbers,stdErr]
        # save
        data = np.transpose(data)
        np.savetxt(fname     = fname, 
                   X         = data, 
                   fmt       = '%s', 
                   delimiter = delimiter, 
                   header    = " ".join(header),
                   comments  = comments)
        
        
class InterMolecularDistanceConstraint(_MolecularDistanceConstraint):
    """
    Its controls the inter-molecular distances between atoms.
    
    :Parameters:
        #. defaultDistance (number): The minimum distance allowed set by default for all 
           atoms type.
        #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules 
           about how to differentiate between atoms and how to parse pairsLimits.
        #. pairsDistanceDefinition (None, list, set, tuple): The minimum distance set to 
           every pair of elements. 
           A list of tuples must be given, all missing pairs will get automatically 
           assigned the given defaultMinimumDistance value.
           First defined elements pair distance will cancel all redundant. 
           If None is given all pairs will be automatically generated and assigned the 
           given defaultMinimumDistance value .
           ::
           
               e.g. [('h','h',1.5), ('h','c',2.015), ...] 
           
        #. flexible (boolean): Whether to allow atoms to break constraint's definition 
           under the condition of decreasing total standardError of the constraint. 
           If flexible is set to False, atoms will never be allowed to cross from above 
           to below minimum allowed distance. Even if the later will decrease some other 
           unsatisfying atoms distances, and therefore the total standardError of the 
           constraint. 
        #. rejectProbability (Number): rejecting probability of all steps where 
           standardError increases. It must be between 0 and 1 where 1 means 
           rejecting all steps where standardError increases and 0 means accepting 
           all steps regardless whether standardError increases or not.
    
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
        
        # create engine 
        ENGINE = Engine(path='my_engine.rmc')
        
        # set pdb file
        ENGINE.set_pdb('system.pdb')
        
        # create and add constraint
        EMD = InterMolecularDistanceConstraint()
        ENGINE.add_constraints(EMD)
        
        # create definition
        EMD.set_pairs_distance([('Si','Si',1.75), ('O','O',1.10), ('Si','O',1.30)])
    
    
    .. autoattribute:: defaultDistance
    .. autoattribute:: pairsDistanceDefinition
    .. autoattribute:: pairsDistance
    .. autoattribute:: flexible
    .. autoattribute:: lowerLimitArray
    .. autoattribute:: upperLimitArray
    .. autoattribute:: typePairs
    .. autoattribute:: typeDefinition
    .. autoattribute:: types
    .. autoattribute:: allTypes
    .. autoattribute:: numberOfTypes    
    .. autoattribute:: typeIndexes
    .. autoattribute:: numberOfAtomsPerType
    .. automethod:: listen
    .. automethod:: set_flexible
    .. automethod:: set_default_distance
    .. automethod:: set_type_definition
    .. automethod:: set_pairs_distance
    .. automethod:: should_step_get_rejected    
    .. automethod:: compute_standard_error
    .. automethod:: get_constraint_value
    .. automethod:: compute_data
    .. automethod:: compute_before_move
    .. automethod:: compute_after_move
    .. automethod:: accept_move
    .. automethod:: reject_move
    .. automethod:: compute_as_if_amputated    
    .. automethod:: accept_amputation
    .. automethod:: reject_amputation
    .. automethod:: plot
    .. automethod:: export
    """
    def __init__(self, defaultDistance=1.5, typeDefinition='element', 
                       pairsDistanceDefinition=None, flexible=True, rejectProbability=1):
        # creating customizable flags
        object.__setattr__(self, '_interMolecular', True)
        object.__setattr__(self, '_intraMolecular', False)
        # initialize constraint
        super(InterMolecularDistanceConstraint, self).__init__(defaultDistance         = defaultDistance, 
                                                               typeDefinition          = typeDefinition,
                                                               pairsDistanceDefinition = pairsDistanceDefinition,
                                                               flexible                = flexible,
                                                               rejectProbability       = rejectProbability) 



class IntraMolecularDistanceConstraint(_MolecularDistanceConstraint):
    """
    .. py:class::IntraMolecularDistanceConstraint

    Its controls the intra-molecular distances between atoms.
    
    :Parameters:
        #. defaultDistance (number): The minimum distance allowed set by default for all 
           atoms type.
        #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules 
           about how to differentiate between atoms and how to parse pairsLimits.
        #. pairsDistanceDefinition (None, list, set, tuple): The minimum distance set to 
           every pair of elements. 
           A list of tuples must be given, all missing pairs will get automatically 
           assigned the given defaultMinimumDistance value.
           First defined elements pair distance will cancel all redundant. 
           If None is given all pairs will be automatically generated and assigned the 
           given defaultMinimumDistance value .
           ::
           
               e.g. [('h','h',1.5), ('h','c',2.015), ...] 
           
        #. flexible (boolean): Whether to allow atoms to break constraint's definition 
           under the condition of decreasing total standardError of the constraint. 
           If flexible is set to False, atoms will never be allowed to cross from above 
           to below minimum allowed distance. Even if the later will decrease some other 
           unsatisfying atoms distances, and therefore the total standardError of the 
           constraint. 
        #. rejectProbability (Number): rejecting probability of all steps where 
           standardError increases. It must be between 0 and 1 where 1 means 
           rejecting all steps where standardError increases and 0 means accepting 
           all steps regardless whether standardError increases or not.
    
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.DistanceConstraints import IntraMolecularDistanceConstraint
        
        # create engine 
        ENGINE = Engine(path='my_engine.rmc')
        
        # set pdb file
        ENGINE.set_pdb('system.pdb')
        
        # create and add constraint
        EMD = IntraMolecularDistanceConstraint()
        ENGINE.add_constraints(EMD)
        
        # create definition
        EMD.set_pairs_distance([('Si','Si',1.75), ('O','O',1.10), ('Si','O',1.30)])
    

    .. autoattribute:: defaultDistance
    .. autoattribute:: pairsDistanceDefinition
    .. autoattribute:: pairsDistance
    .. autoattribute:: flexible
    .. autoattribute:: lowerLimitArray
    .. autoattribute:: upperLimitArray
    .. autoattribute:: typePairs
    .. autoattribute:: typeDefinition
    .. autoattribute:: types
    .. autoattribute:: allTypes
    .. autoattribute:: numberOfTypes    
    .. autoattribute:: typeIndexes
    .. autoattribute:: numberOfAtomsPerType
    .. automethod:: listen
    .. automethod:: set_flexible
    .. automethod:: set_default_distance
    .. automethod:: set_type_definition
    .. automethod:: set_pairs_distance
    .. automethod:: should_step_get_rejected    
    .. automethod:: compute_standard_error
    .. automethod:: get_constraint_value
    .. automethod:: compute_data
    .. automethod:: compute_before_move
    .. automethod:: compute_after_move
    .. automethod:: accept_move
    .. automethod:: reject_move
    .. automethod:: compute_as_if_amputated    
    .. automethod:: accept_amputation
    .. automethod:: reject_amputation
    .. automethod:: plot
    .. automethod:: export
    """
    def __init__(self, defaultDistance=1.5, typeDefinition='name', 
                       pairsDistanceDefinition=None, flexible=True, rejectProbability=1):
        # creating customizable flags
        object.__setattr__(self, '_interMolecular', False)
        object.__setattr__(self, '_intraMolecular', True)
        # initialize constraint
        super(IntraMolecularDistanceConstraint, self).__init__(defaultDistance         = defaultDistance, 
                                                               typeDefinition          = typeDefinition,
                                                               pairsDistanceDefinition = pairsDistanceDefinition,
                                                               flexible                = flexible,
                                                               rejectProbability       = rejectProbability) 
    


    
    
            