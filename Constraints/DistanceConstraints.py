"""
.. inheritance-diagram:: fullrmc.Constraints.DistanceConstraints
    :parts: 2 
"""

# standard libraries imports
import itertools
import warnings

# external libraries imports
import numpy as np
from timeit import default_timer as timer

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, FLOAT_PLUS_INFINITY
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, SingularConstraint, EnhanceOnlyConstraint
from fullrmc.Core.distances import multiple_distances, full_distances


class InterMolecularDistanceConstraint(EnhanceOnlyConstraint, SingularConstraint):
    """
    Its controls the inter-molecular distances between atoms.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. defaultDistance (number): The minimum distance allowed set by default for all atoms type.
        #. pairsDistance (None, list, set, tuple): The minimum distance set to every pair of elements. 
           A list of tuples must be given, all missing pairs will get automatically assigned the given defaultMinimumDistance value.
           First defined elements pair distance will cancel all redundant. 
           If None is given all pairs will be automatically generated and assigned the given defaultMinimumDistance value .e.g. [('h','h',1.5), ('h','c',2.015), ...] 
        #. rejectProbability (None, numpy.ndarray): rejection probability numpy.array.
           If None, rejectProbability will be automatically generated to 1 for all step where chiSquare increase.

    """
    def __init__(self, engine, defaultDistance=1.5, pairsDistance=None, rejectProbability=None):
        # initialize constraint
        EnhanceOnlyConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability)
        # set defaultDistance
        self.set_default_distance(defaultDistance)
        # set pairsDistance
        self.set_pairs_distance(pairsDistance)
                  
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
    def elementsPairs(self):
        """ Get elements pairs """
        return self.__elementsPairs
        
    def listen(self, message, argument=None):
        """   
        listen to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. arguments (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed", "update molecules indexes"):
            if self.engine is not None:
                self.__elementsPairs = sorted(itertools.combinations_with_replacement(self.engine.elements,2))
                self.set_pairs_distance(self.__pairsDistanceDefinition)
            else:
                self.__elementsPairs = None
                self.set_pairs_distance(self.__pairsDistanceDefinition)
        elif message in("update boundary conditions",):
            self.__initialize_constraint__()        
                
    def set_default_distance(self, defaultDistance):
        """ 
        Sets the default intermolecular minimum distance. 
        
        :Parameters:
            #. defaultDistance (number): The default minimum distance.
        """
        assert is_number(defaultDistance), "defaultDistance must be a number"
        defaultDistance = FLOAT_TYPE(defaultDistance)
        assert defaultDistance>=0, "defaultDistance must be positive"
        self.__defaultDistance = defaultDistance
        
    def set_pairs_distance(self, pairsDistance):
        """ 
        Sets the pairs intermolecular minimum distance. 
        
        :Parameters:
            #. pairsDistance (None, list, set, tuple): The minimum distance set to every pair of elements. 
              A list of tuples must be given, all missing pairs will get automatically assigned the given defaultMinimumDistance value.
              First defined elements pair distance will cancel all redundant. 
              If None is given all pairs will be automatically generated and assigned the given defaultMinimumDistance value .e.g. [('h','h',1.5), ('h','c',2.015), ...] 
        """
        if self.engine is None:
            newPairsDistance = None
        elif pairsDistance is None:
            newPairsDistance = {}
            for el1 in self.engine.elements:
                newPairsDistance[el1] = {}
                for el2 in self.engine.elements:
                    newPairsDistance[el1][el2] = self.__defaultDistance
        else:
            newPairsDistance = {}
            assert isinstance(pairsDistance, (list, set, tuple)), "pairsDistance must be a list"
            for pair in pairsDistance:
                assert isinstance(pair, (list, set, tuple)), "pairsDistance list items must be lists as well"
                pair = list(pair)
                assert len(pair)==3, "pairsDistance list pair item list must have three items"
                if pair[0] not in self.engine.elements:
                    warnings.warn("pairsDistance list pair item '%s' is not a valid engine element, definition item omitted"%pair[0])
                    continue
                if pair[1] not in self.engine.elements: 
                    warnings.warn("pairsDistance list pair item '%s' is not a valid engine element, definition item omitted"%pair[1])
                    continue
                # create elements keys
                if not newPairsDistance.has_key(pair[0]):
                    newPairsDistance[pair[0]] = {}
                if not newPairsDistance.has_key(pair[1]):
                    newPairsDistance[pair[1]] = {}
                assert is_number(pair[2]), "pairsDistance list pair item list third item must be a number"
                distance = FLOAT_TYPE(pair[2])
                assert distance>=0, "pairsDistance list pair item list third item must be bigger than 0"
                # set minimum distance
                if newPairsDistancepair[0].has_key(pair[1]):
                    warnings.warn("elements pair ('%s','%s') distance definition is redundant, '%s' is omitted"%(pair[0], pair[1], pair))
                else:
                    newPairsDistance[pair[0]][pair[1]] = distance
                if newPairsDistancepair[1].has_key(pair[0]):
                    warnings.warn("elements pair ('%s','%s') distance definition is redundant, '%s' is omitted"%(pair[1], pair[0], pair))
                else:
                    newPairsDistance[pair[1]][pair[0]] = distance
        # set new pairsDistance value
        self.__pairsDistanceDefinition = pairsDistance
        self.__pairsDistance = newPairsDistance
        if self.__pairsDistance is not None:
            self.__lowerLimitArray = np.zeros((self.engine.numberOfElements, self.engine.numberOfElements, 1), dtype=FLOAT_TYPE) 
            self.__upperLimitArray = np.zeros((self.engine.numberOfElements, self.engine.numberOfElements, 1), dtype=FLOAT_TYPE) 
            for idx1 in range(self.engine.numberOfElements):
                el1 = self.engine.elements[idx1]
                for idx2 in range(self.engine.numberOfElements): 
                    el1  = self.engine.elements[idx1]
                    dist = self.__pairsDistance[el1][el2]
                    assert dist == self.__pairsDistance[el2][el1], "pairsDistance must be symmetric"
                    self.__upperLimitArray[idx1,idx2,0] = FLOAT_TYPE(dist)
                    self.__upperLimitArray[idx2,idx1,0] = FLOAT_TYPE(dist)
        else:
            self.__lowerLimitArray = None     
            self.__upperLimitArray = None                 
            
    def compute_chi_square(self, data):
        """ 
        Compute the chi square of data not satisfying constraint conditions. 
        
        :Parameters:
            #. data (numpy.array): The constraint value data to compute chiSquare.
            
        :Returns:
            #. chiSquare (number): The calculated chiSquare multiplied by the contribution factor of the constraint.
        """
        # compute difference
        chiSquare = 0.0
        number = 0
        for k, val in data.items():
            if val < PRECISION:
                continue
            number += 1
            el1, el2 = k.split("intermd_")[1].split("-")
            dist = self.__pairsDistance[el1][el2]
            diff = dist-val
            assert diff>0 , "difference must be positive. %.6f is found for val:%.6f and minimumDistance: %.6f. Try recomputing constraint data using 'compute_data' method"%(diff, val, dist)
            assert diff<=dist, "difference must be smaller than minimum distance. %.6f is found for val:%.6f and minimumDistance: %.6f .Try recomputing constraint data using 'compute_data' method"%(diff, val, dist)
            # normalize to make it between 0 and 1
            chiSquare += (diff/dist)**2
        # normalize
        #if number:
        #    chiSquare /= number
        return FLOAT_TYPE(chiSquare)
        
    def get_constraint_value(self):
        """
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        if self.data is None:
            warnings.warn("data must be computed first using 'compute_data' method.")
            return {}
        output = {}
        for pair in self.__elementsPairs:
            output["intermd_%s-%s" % pair] = FLOAT_TYPE(0.0)
        for pair in self.__elementsPairs:
            # get index of element
            idi = self.engine.elements.index(pair[0])
            idj = self.engine.elements.index(pair[1])
            # get mean value
            number = self.data["number"][idi,idj][0] + self.data["number"][idj,idi][0]
            distanceSum = self.data["distanceSum"][idi,idj][0] + self.data["distanceSum"][idj,idi][0]
            if number != 0:
                output["intermd_%s-%s" % pair] += FLOAT_TYPE(distanceSum/number) 
        return output    
        
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        _,_,number,distanceSum = full_distances( boxCoords=self.engine.boxCoordinates,
                                                 basis=self.engine.basisVectors,
                                                 moleculeIndex = self.engine.moleculesIndexes,
                                                 elementIndex = self.engine.elementsIndexes,
                                                 numberOfElements = self.engine.numberOfElements,
                                                 lowerLimit=self.__lowerLimitArray,
                                                 upperLimit=self.__upperLimitArray,
                                                 interMolecular = True,
                                                 intraMolecular = False,
                                                 countWithinLimits = True)
        # update data
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set chiSquare
        self.set_chi_square( self.compute_chi_square(data = self.get_constraint_value()) )
    
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        
        _,_,numberM,distanceSumM = multiple_distances( indexes = indexes,
                                                       boxCoords=self.engine.boxCoordinates,
                                                       basis=self.engine.basisVectors,
                                                       moleculeIndex = self.engine.moleculesIndexes,
                                                       elementIndex = self.engine.elementsIndexes,
                                                       numberOfElements = self.engine.numberOfElements,
                                                       lowerLimit=self.__lowerLimitArray,
                                                       upperLimit=self.__upperLimitArray,
                                                       allAtoms = True,
                                                       countWithinLimits = True,
                                                       interMolecular = True,
                                                       intraMolecular = False)
        _,_,numberF,distanceSumF = full_distances( boxCoords=self.engine.boxCoordinates[indexes],
                                                   basis=self.engine.basisVectors,
                                                   moleculeIndex = self.engine.moleculesIndexes[indexes],
                                                   elementIndex = self.engine.elementsIndexes[indexes],
                                                   numberOfElements = self.engine.numberOfElements,
                                                   lowerLimit=self.__lowerLimitArray,
                                                   upperLimit=self.__upperLimitArray,
                                                   countWithinLimits = True,
                                                   interMolecular = True,
                                                   intraMolecular = False)
        self.set_active_atoms_data_before_move( {"number":numberM-numberF, "distanceSum":distanceSumM-distanceSumF} )
        self.set_active_atoms_data_after_move(None)
    
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
        # calculate pair distribution function
        _,_,numberM,distanceSumM = multiple_distances( indexes = indexes,
                                                       boxCoords=self.engine.boxCoordinates,
                                                       basis=self.engine.basisVectors,
                                                       moleculeIndex = self.engine.moleculesIndexes,
                                                       elementIndex = self.engine.elementsIndexes,
                                                       numberOfElements = self.engine.numberOfElements,
                                                       lowerLimit=self.__lowerLimitArray,
                                                       upperLimit=self.__upperLimitArray,
                                                       allAtoms = True,
                                                       countWithinLimits = True,
                                                       interMolecular = True,
                                                       intraMolecular = False)
        _,_,numberF,distanceSumF = full_distances( boxCoords=self.engine.boxCoordinates[indexes],
                                                   basis=self.engine.basisVectors,
                                                   moleculeIndex = self.engine.moleculesIndexes[indexes],
                                                   elementIndex = self.engine.elementsIndexes[indexes],
                                                   numberOfElements = self.engine.numberOfElements,
                                                   lowerLimit=self.__lowerLimitArray,
                                                   upperLimit=self.__upperLimitArray,
                                                   countWithinLimits = True,
                                                   interMolecular = True,
                                                   intraMolecular = False)
        self.set_active_atoms_data_after_move( {"number":numberM-numberF, "distanceSum":distanceSumM-distanceSumF} )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute chiSquare after move
        number = self.data["number"]-self.activeAtomsDataBeforeMove["number"]+self.activeAtomsDataAfterMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]+self.activeAtomsDataAfterMove["distanceSum"]
        data = self.data
        # change temporarily data attribute
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        self.set_after_move_chi_square( self.compute_chi_square(data = self.get_constraint_value()) )
        # change back data attribute
        self.set_data( data )
    
    def accept_move(self, indexes):
        """ 
        Accept move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        number = self.data["number"]-self.activeAtomsDataBeforeMove["number"]+self.activeAtomsDataAfterMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]+self.activeAtomsDataAfterMove["distanceSum"]
        # change permanently data attribute
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update chiSquare
        self.set_chi_square( self.afterMoveChiSquare )
        self.set_after_move_chi_square( None )
    
    def reject_move(self, indexes):
        """ 
        Reject move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update chiSquare
        self.set_after_move_chi_square( None )


class IntraMolecularDistanceConstraint(EnhanceOnlyConstraint, SingularConstraint):
    """
    Its controls the intra-molecular distances between atoms.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. defaultMinDistance (number): The minimum distance allowed set by default for all atoms type.
        #. typeDefinition (string): Can be either 'element' or 'name'. It sets the rules about how to differentiate between atoms and how to parse pairsLimits
        #. pairsLimitsDefinition (None, list, set, tuple): The lower and upper limits distance set to every pair of elements. 
           A list of tuples must be given, all missing pairs will get automatically assigned the given defaultMinDistance to infinity value.
           First defined elements pair distance will cancel all redundant. 
           If None is given all pairs will be automatically generated and assigned the given defaultMinDistance to infinity value .e.g. [('c','h',0.9, 1.2), ...] 
        #. rejectProbability (None, numpy.ndarray): rejection probability numpy.array.
           If None, rejectProbability will be automatically generated to 1 for all step where chiSquare increase.
        #. mode (string): Defines the way chiSquare is calculated. In such constraints the definition of chiSquare
           can be confusing because many parameters play different roles in this type of calculation. The number of
           unsatisfied constraint conditions is an important parameter and the reduced unsatisfied distances is another one.
           Choosing a mode of calculation puts more weight and importance on a parameter. Allowed modes are:
            #. distance (Default): chiSquare is simply calculated as the square summation of the reduced out of limits distances.
               This mode ensures minimizing the global unsatisfied distance of the constraint while additional atom-pairs unsatisfying constraint conditions can be created.
            #. number: chiSquare is calculated such as the number of non-satisfied constraints must decrease 
               from one step to another while the square summation of the reduced out of limits distances might increase.             
    """
    def __init__(self, engine, defaultMinDistance=0.67, typeDefinition="name", pairsLimitsDefinition=None, rejectProbability=None, mode="distance"):
        # create modes
        self.__chiSquareModes = {"distance":"__distance_chi_square__",
                                 "number"  :"__number_chi_square__"}
        # initialize constraint
        EnhanceOnlyConstraint.__init__(self, engine=engine, rejectProbability=rejectProbability)
        # set defaultDistance
        self.set_default_minimum_distance(defaultMinDistance)
        # set pairs type definition and pairsLimitsDefinition
        self.__pairsLimitsDefinition = None
        self.set_type_definition(typeDefinition, pairsLimitsDefinition)
        self.set_mode(mode)
    
    @property
    def typeDefinition(self):
        """Get types definition."""    
        return self.__typeDefinition
        
    @property
    def existingChiSquareModes(self):
        """ Get list of defined chiSquare modes of calculation"""
        return self.__chiSquareModes.keys()
        
    @property
    def mode(self):
        """ Get the mode of chiSquare calculation. """
        return self.__mode
        
    @property
    def types(self):
        """ Get the defined types set. """
        return self.__types
    
    @property
    def allTypes(self):
        """ Get all atoms types. """
        return self.__allTypes 
        
    @property
    def numberOfTypes(self):
        """ Get the number of defined types in the configuration. """
        if self.__types is not None:
            return len(self.__types)
        else:
            return None
        
    @property
    def defaultMinDistance(self):
        """ Gets the experimental data distances minimum. """
        return self.__defaultMinDistance
             
    @property
    def pairsLimitsDefinition(self):
        """ Get the pairs limits definition as defined by user"""
        return self.__pairsLimitsDefinition    
        
    @property
    def pairsLimits(self):
        """ Get the normalized pairs limits definition"""
        return self.__pairsLimits 

    @property
    def typesPairs(self):
        """ Get the list of pairs of types"""
        return self.__typesPairs        
    
    @property
    def typesIndexes(self):
        """ Get defined types indexes"""
        return self.__typesIndexes  
        
    @property
    def numberOfAtomsPerType(self):
        """ Get the number of atoms per type dictionary."""
        return self.__numberOfAtomsPerType
        
    @property
    def lowerLimitArray(self):
        """ Get lowerLimitArray used in distances calculation. """
        return self.__lowerLimitArray
    
    @property
    def upperLimitArray(self):
        """ Get upperLimitArray used in distances calculation. """
        return self.__upperLimitArray
        
    def listen(self, message, argument=None):
        """   
        listen to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. arguments (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed", "update molecules indexes"):
            self.set_type_definition(typeDefinition=self.__typeDefinition)

        elif message in("update boundary conditions",):
            self.__initialize_constraint__()        
                
    def set_mode(self, mode):
        """ 
        Sets the chiSquare mode of calculation. 
        
        :Parameters:
            #. mode (object): The mode of calculation
        """
        assert mode in self.__chiSquareModes.keys(), "allowed modes are %s"%self.__chiSquareModes.keys()
        self.__mode = mode
        # reinitialize constraint
        self.__initialize_constraint__()
        
    def set_default_minimum_distance(self, defaultMinDistance):
        """ 
        Sets the default intermolecular minimum distance. 
        
        :Parameters:
            #. defaultMinDistance (number): The default minimum distance.
        """
        assert is_number(defaultMinDistance), "defaultMinDistance must be a number"
        defaultMinDistance = FLOAT_TYPE(defaultMinDistance)
        assert defaultMinDistance>=0, "defaultMinDistance must be positive"
        self.__defaultMinDistance = defaultMinDistance
    
    def set_type_definition(self, typeDefinition, pairsLimitsDefinition=None):
        """
        Its an alias to set_pairs_limits with pairsLimits argument passed as the already defined pairsLimitsDefinitions
        """
        # set typeDefinition
        assert typeDefinition in ("name", "element"), "typeDefinition must be either 'name' or 'element'"
        if self.engine is None:
            self.__types                = None
            self.__allTypes             = None
            self.__numberOfTypes        = None
            self.__typesIndexes         = None
            self.__numberOfAtomsPerType = None
        elif typeDefinition == "name":
            self.__types                = self.engine.names
            self.__allTypes             = self.engine.allNames
            self.__numberOfTypes        = self.engine.numberOfNames
            self.__typesIndexes         = self.engine.namesIndexes
            self.__numberOfAtomsPerType = self.engine.numberOfAtomsPerName
        elif typeDefinition == "element":
            self.__types                = self.engine.elements
            self.__allTypes             = self.engine.allElements
            self.__numberOfTypes        = self.engine.numberOfElements
            self.__typesIndexes         = self.engine.elementsIndexes
            self.__numberOfAtomsPerType = self.engine.numberOfAtomsPerElement
        self.__typeDefinition = typeDefinition
        # check pdb atoms
        if self.engine is not None:
            lastMolIdx = None
            lut = {}
            for idx in range(len(self.__allTypes)):
                molIdx = self.engine.moleculesIndexes[idx]
                name   = self.__allTypes[idx]
                if lastMolIdx != molIdx:
                    lut = {}
                    lastMolIdx = molIdx
                if lut.has_key(name):
                    raise Exception( "molecule index '%i' is found to have the same atom %s '%s', This is not allowed for '%s' constraint"%(lastMolIdx, self.__typeDefinition, name, self.__class__.__name__) )
                else:
                    lut[name] = 1
        # set pairs limits
        if pairsLimitsDefinition is None:
            pairsLimitsDefinition = self.__pairsLimitsDefinition
        self.set_pairs_limits(pairsLimitsDefinition)
        
    def set_pairs_limits(self, pairsLimitsDefinition):
        """ 
        Sets the pairs intermolecular minimum distance. 
        
        :Parameters:
            #. pairsLimitsDefinition (None, list, set, tuple): The lower and upper limits distance set to every pair of elements. 
              A list of tuples must be given, all missing pairs will get automatically assigned the given defaultMinDistance to infinity value.
              First defined elements pair distance will cancel all redundant. 
              If None is given all pairs will be automatically generated and assigned the given defaultMinDistance to infinity value .e.g. [('c','h',0.9, 1.2), ...] 
        """
        # set types pairs
        if self.__types is not None:
            self.__typesPairs = sorted(itertools.combinations_with_replacement(self.__types,2))
        else:
            self.__typesPairs = None
        # get pairs limits
        pairsLimitsDict = {}
        if self.engine is None:
            pairsLimitsDict = None
        elif pairsLimitsDefinition is not None:
            pairsLimitsDict = {}
            assert isinstance(pairsLimitsDefinition, (list, set, tuple)), "pairsLimitsDefinition must be a list"
            for pair in pairsLimitsDefinition:
                assert isinstance(pair, (list, set, tuple)), "pairsLimitsDefinition list items must be lists as well"
                pair = list(pair)
                assert len(pair)==4, "pairsLimitsDefinition list pair item list must have four items"
                if pair[0] not in self.__types:
                    warning.warn("pairsLimitsDefinition list pair item '%s' is not a valid type, definition item omitted"%pair[0])
                    continue
                if pair[1] not in self.__types: 
                    warning.warn("pairsLimitsDefinition list pair item '%s' is not a valid type, definition item omitted"%pair[1])
                    continue
                # create type keys
                if not pairsLimitsDict.has_key(pair[0]):
                    pairsLimitsDict[pair[0]] = {}
                if not pairsLimitsDict.has_key(pair[1]):
                    pairsLimitsDict[pair[1]] = {}
                assert is_number(pair[2]), "pairsLimitsDefinition list pair item list third item must be a number"
                lower = FLOAT_TYPE(pair[2])
                assert is_number(pair[3]), "pairsLimitsDefinition list pair item list foruth item must be a number"
                upper = FLOAT_TYPE(pair[3])
                assert lower>=0, "pairsLimitsDefinition list pair item list third item must be bigger than 0"
                assert upper>lower, "pairsLimitsDefinition list pair item list fourth item must be bigger than the third item"
                # set minimum distance
                if pairsLimitsDict[pair[0]].has_key(pair[1]):
                    warnings.warn("elements pair ('%s','%s') distance definition is redundant, '%s' is omitted"%(pair[0], pair[1], pair))
                else:
                    pairsLimitsDict[pair[0]][pair[1]] = (lower, upper)
                if pairsLimitsDict[pair[1]].has_key(pair[0]):
                    warnings.warn("elements pair ('%s','%s') distance definition is redundant, '%s' is omitted"%(pair[1], pair[0], pair))
                else:
                    pairsLimitsDict[pair[1]][pair[0]] = (lower, upper)
        # complete pairsLimitsDict to all elements
        if self.engine is not None:
            for el1 in self.__types:
                if not pairsLimitsDict.has_key(el1):
                    pairsLimitsDict[el1] = {}
                for el2 in self.__types:
                    if not pairsLimitsDict[el1].has_key(el2):
                        pairsLimitsDict[el1][el2] = (self.__defaultMinDistance, FLOAT_PLUS_INFINITY)
        # set new pairsLimitsDefinition value
        self.__pairsLimitsDefinition = pairsLimitsDefinition
        self.__pairsLimits = pairsLimitsDict
        # set limits arrays
        if self.__pairsLimits is not None:
            self.__lowerLimitArray = np.zeros((self.__numberOfTypes, self.__numberOfTypes, 1), dtype=FLOAT_TYPE) 
            self.__upperLimitArray = np.zeros((self.__numberOfTypes, self.__numberOfTypes, 1), dtype=FLOAT_TYPE) 
            for idx1 in range(self.__numberOfTypes):
                el1 = self.__types[idx1]
                for idx2 in range(self.__numberOfTypes): 
                    el2  = self.__types[idx2]
                    l,u = self.__pairsLimits[el1][el2]
                    sl,su = self.__pairsLimits[el2][el1]
                    assert l==sl and u==su, "pairsDistance must be symmetric"
                    self.__lowerLimitArray[idx1,idx2,0] = FLOAT_TYPE(l)
                    self.__upperLimitArray[idx1,idx2,0] = FLOAT_TYPE(u)
                    self.__lowerLimitArray[idx2,idx1,0] = FLOAT_TYPE(l)
                    self.__upperLimitArray[idx2,idx1,0] = FLOAT_TYPE(u)
        else:
            self.__lowerLimitArray = None  
            self.__upperLimitArray = None            
            
    def __distance_chi_square__(self, data):
        """
        calculates the squared total distanceSum
        """
        return FLOAT_TYPE( np.sum(data["distanceSum"]**2) )
    
    def __number_chi_square__(self, data):
        """
        chiSquare = SUM_j(SUM_i( meanDistance * (1.0/Nj_total) * (Ni_total/(2*Ni_total-Ni_unsatisfied))**2 ))
        where meanDistance = SUM_ji( reducedDistance )**2 / Ni_total
        """
        chiSquare = 0
        for idx1 in range(len(self.__types)):
            Nj_total = FLOAT_TYPE(self.__numberOfAtomsPerType[self.__types[idx1]])
            if Nj_total == 0: continue
            for idx2 in range(idx1,len(self.__types)):
                Ni_total       = FLOAT_TYPE(self.__numberOfAtomsPerType[self.__types[idx2]])
                if Ni_total == 0: continue
                Ni_unsatisfied = FLOAT_TYPE(data["number"][idx1][idx2][0])
                if Ni_unsatisfied == 0: continue
                meanDistance   = FLOAT_TYPE(data["distanceSum"][idx1][idx2][0]**2)/Ni_unsatisfied
                chiSquare     += meanDistance * (1.0/Nj_total) * (Ni_total/(2*Ni_total-Ni_unsatisfied))**2
        return FLOAT_TYPE(chiSquare)
    
    def compute_chi_square(self, data):
        """ 
        Compute the chi square of data not satisfying constraint conditions. 
        
        :Parameters:
            #. data (numpy.array): The constraint value data to compute chiSquare.
            
        :Returns:
            #. chiSquare (number): The calculated chiSquare multiplied by the contribution factor of the constraint.
        """
        return getattr(self, self.__chiSquareModes[self.__mode])(data)

    def get_constraint_value(self):
        """
        Compute all partial Mean Pair Distances (MPD) below the defined minimum distance. 
        
        :Returns:
            #. MPD (dictionary): The MPD dictionary, where keys are the element wise intra and inter molecular MPDs and values are the computed MPDs.
        """
        if self.data is None:
            warnings.warn("data must be computed first using 'compute_data' method.")
            return {}
        output = {}
        for pair in self.__typesPairs:
            output["intramd_%s-%s" % pair] = FLOAT_TYPE(0.0)
        for pair in self.__typesPairs:
            # get index of element
            idi = self.__types.index(pair[0])
            idj = self.__types.index(pair[1])
            # get mean value
            number = self.data["number"][idi,idj][0] + self.data["number"][idj,idi][0]
            distanceSum = self.data["distanceSum"][idi,idj][0] + self.data["distanceSum"][idj,idi][0]
            if number != 0:
                output["intramd_%s-%s" % pair] += FLOAT_TYPE(distanceSum/number) 
        return output    
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        number,distanceSum,_,_ = full_distances( boxCoords=self.engine.boxCoordinates,
                                                 basis=self.engine.basisVectors,
                                                 moleculeIndex = self.engine.moleculesIndexes,
                                                 elementIndex = self.__typesIndexes,
                                                 numberOfElements = self.__numberOfTypes,
                                                 lowerLimit=self.__lowerLimitArray,
                                                 upperLimit=self.__upperLimitArray,
                                                 countWithinLimits = False,
                                                 reduceDistance = True,
                                                 reduceDistanceToUpper = False,
                                                 reduceDistanceToLower = False,
                                                 interMolecular = False,
                                                 intraMolecular = True)
        # update data
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set chiSquare
        self.set_chi_square( self.compute_chi_square(data = self.data) )
    
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        numberM,distanceSumM,_,_ = multiple_distances( indexes = indexes,
                                                       boxCoords=self.engine.boxCoordinates,
                                                       basis=self.engine.basisVectors,
                                                       moleculeIndex = self.engine.moleculesIndexes,
                                                       elementIndex = self.__typesIndexes,
                                                       numberOfElements = self.__numberOfTypes,
                                                       lowerLimit=self.__lowerLimitArray,
                                                       upperLimit=self.__upperLimitArray,
                                                       allAtoms = True,
                                                       countWithinLimits = False,
                                                       reduceDistance = True,
                                                       reduceDistanceToUpper = False,
                                                       reduceDistanceToLower = False,
                                                       interMolecular = False,
                                                       intraMolecular = True)
        numberF,distanceSumF,_,_ = full_distances( boxCoords=self.engine.boxCoordinates[indexes],
                                                   basis=self.engine.basisVectors,
                                                   moleculeIndex = self.engine.moleculesIndexes[indexes],
                                                   elementIndex = self.__typesIndexes[indexes],
                                                   numberOfElements = self.__numberOfTypes,
                                                   lowerLimit=self.__lowerLimitArray,
                                                   upperLimit=self.__upperLimitArray,
                                                   countWithinLimits = False,
                                                   reduceDistance = True,
                                                   reduceDistanceToUpper = False,
                                                   reduceDistanceToLower = False,
                                                   interMolecular = False,
                                                   intraMolecular = True)
        self.set_active_atoms_data_before_move( {"number":numberM-numberF, "distanceSum":distanceSumM-distanceSumF} )
        self.set_active_atoms_data_after_move(None)
    
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
        # calculate pair distribution function
        numberM,distanceSumM,_,_ = multiple_distances( indexes = indexes,
                                                       boxCoords=self.engine.boxCoordinates,
                                                       basis=self.engine.basisVectors,
                                                       moleculeIndex = self.engine.moleculesIndexes,
                                                       elementIndex = self.__typesIndexes,
                                                       numberOfElements = self.__numberOfTypes,
                                                       lowerLimit=self.__lowerLimitArray,
                                                       upperLimit=self.__upperLimitArray,
                                                       allAtoms = True,
                                                       countWithinLimits = False,
                                                       reduceDistance = True,
                                                       reduceDistanceToUpper = False,
                                                       reduceDistanceToLower = False,
                                                       interMolecular = False,
                                                       intraMolecular = True)
        numberF,distanceSumF,_,_ = full_distances( boxCoords=self.engine.boxCoordinates[indexes],
                                                   basis=self.engine.basisVectors,
                                                   moleculeIndex = self.engine.moleculesIndexes[indexes],
                                                   elementIndex = self.__typesIndexes[indexes],
                                                   numberOfElements = self.__numberOfTypes,
                                                   lowerLimit=self.__lowerLimitArray,
                                                   upperLimit=self.__upperLimitArray,
                                                   countWithinLimits = False,
                                                   reduceDistance = True,
                                                   reduceDistanceToUpper = False,
                                                   reduceDistanceToLower = False,
                                                   interMolecular = False,
                                                   intraMolecular = True)
        self.set_active_atoms_data_after_move( {"number":numberM-numberF, "distanceSum":distanceSumM-distanceSumF} )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute chiSquare after move
        number = self.data["number"]-self.activeAtomsDataBeforeMove["number"]+self.activeAtomsDataAfterMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]+self.activeAtomsDataAfterMove["distanceSum"]
        data = self.data
        # change temporarily data attribute
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        self.set_after_move_chi_square( self.compute_chi_square(data = {"number":number, "distanceSum":distanceSum}) )
        # change back data attribute
        self.set_data( data )
    
    def accept_move(self, indexes):
        """ 
        Accept move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        number = self.data["number"]-self.activeAtomsDataBeforeMove["number"]+self.activeAtomsDataAfterMove["number"]
        distanceSum = self.data["distanceSum"]-self.activeAtomsDataBeforeMove["distanceSum"]+self.activeAtomsDataAfterMove["distanceSum"]
        # change permanently data attribute
        self.set_data( {"number":number, "distanceSum":distanceSum} )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update chiSquare
        self.set_chi_square( self.afterMoveChiSquare )
        self.set_after_move_chi_square( None )
    
    def reject_move(self, indexes):
        """ 
        Reject move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update chiSquare
        self.set_after_move_chi_square( None )


    
    
    


    
    
            