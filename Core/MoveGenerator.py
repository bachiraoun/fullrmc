"""
MoveGenerator contains parent classes for all move generators.
A MoveGenerator sub-class is used at the Engine runtime to generate moves upon selected groups.
Every group has its own MoveGenerator class and definitions, therefore it has become possible
to fully customize how a group of atoms should move.

.. inheritance-diagram:: fullrmc.Core.MoveGenerator
    :parts: 1
"""

# standard libraries imports
import inspect
from random import randint, shuffle

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, generate_random_float


class MoveGenerator(object):
    """ 
    It is the parent class for all moves generators.
    This class can't be instantiated but its sub-classes might be.
    
    :Parameters:
        #. group (None, Group): The group instance.
    """
    def __init__(self, group=None):
        self.set_group(group)
    
    @property
    def group(self):
        """ Get the group instance."""
        return self.__group
    
    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. arguments (object): Any python object.
        """
        pass
        
    def set_group(self, group):
        """
        Set the MoveGenerator group.
        
        :Parameters:
            #. group (None, Group): group instance. 
        """
        if group is not None:
            from fullrmc.Core.Group import Group
            assert isinstance(group, Group), LOGGER.error("group must be a fullrmc Group instance")
            valid, message = self.check_group(group)
            if not valid:
                raise Exception( LOGGER.error("%s"%message) )
        self.__group = group
    
    def check_group(self, group):
        """
        Checks the generator's group. 
        This method must be overloaded in all MoveGenerator sub-classes.
        
        :Parameters:
            #. group (Group): the Group instance
        """
        raise Exception(LOGGER.error("MovesGenerator '%s' method must be overloaded"%inspect.stack()[0][3]))
        
    def transform_coordinates(self, coordinates, argument=None):
        """
        Transform coordinates. This method is called in every move.
        This method must be overloaded in all MoveGenerator sub-classes.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the move.
            #. argument (object): Any other argument needed to perform the move.
               In General it's not needed.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the move.
        """
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def move(self, coordinates):
        """
        Moves coordinates. 
        This method must NOT be overloaded in MoveGenerator sub-classes.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the transformation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the transformation.
        """
        return self.transform_coordinates(coordinates=coordinates)


class SwapGenerator(MoveGenerator):
    """ 
    It is a particular move generator that instead of generating a 
    move upon a group of atoms, it will exchange the group atom positions
    with other atoms from swapList. 
    Because the swapList can be big, swapGenerator can be assigned to
    multiple groups at the same time under the condition of all groups
    having the same length.\n
    
    During engine runtime, whenever a swap generator is encountered,
    all sophisticated selection recurrence modes such as (refining, exploring) 
    will be reduced to simple recurrence.\n

    This class can't be instantiated but its sub-classes might be.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. swapLength (Integer): The swap length that defines the length of the group 
           and the length of the every swap sub-list in swapList.
        #. swapList (None, List): The list of atoms.\n
           If None is given, no swapping or exchanging will be performed.\n
           If List is given, it must contain lists of atom indexes where every 
           sub-list must have the same number of atoms as the group.
    """
    def __init__(self, group=None, swapLength=1, swapList=None):
        super(SwapGenerator, self).__init__(group=group) 
        # set swap length
        self.set_swap_length(swapLength)
        # set swap list
        self.set_swap_list(swapList)
        #  initialize swapping variables
        self.__groupAtomsIndexes = None
        self.__swapAtomsIndexes  = None        
    
    @property
    def swapLength(self):
        """ Get swap length."""
        return self.__swapLength 
    
    @property
    def swapList(self):
        """ Get swap list."""
        return self.__swapList
        
    @property
    def groupAtomsIndexes (self):
        """ Get last selected group atoms indexes."""
        return self.__groupAtomsIndexes 
    
    @property
    def swapAtomsIndexes(self):
        """ Get last swap atoms indexes."""
        return self.__swapAtomsIndexes
        
    def set_swap_length(self, swapLength):
        """
        Set swap length. it will reset swaplist automatically.
    
        :Parameters:
            #. swapLength (Integer): The swap length that defines the length of the group 
               and the length of the every swap sub-list in swapList.
        """   
        assert is_integer(swapLength), LOGGER.error("swapLength must be an integer")
        swapLength = INT_TYPE(swapLength)
        assert swapLength>0, LOGGER.error("swapLength must be bigger than 0")
        self.__swapLength = swapLength
        self.__swapList   = ()
        
    def set_group(self, group):
        """
        Set the MoveGenerator group.
        
        :Parameters:
            #. group (None, Group): group instance. 
        """
        MoveGenerator.set_group(self, group)
        if self.group is not None:
            assert len(self.group) == self.__swapLength, LOGGER.error("SwapGenerator groups length must be equal to swapLength.")

    def set_swap_list(self, swapList):
        """
        Set the swap-list to exchange atom positions from.
        
        :Parameters: 
            #. swapList (None, List): The list of atoms.\n 
               If None is given, no swapping or exchanging will be performed.\n
               If List is given, it must contain lists of atom indexes where every 
               sub-list length must be equal to swapLength.
        """
        if swapList is None:
            self.__swapList = ()
        else:
            SL = []
            assert isinstance(swapList, (list,tuple)), LOGGER.error("swapList must be a list")
            for sl in swapList:
                assert isinstance(sl, (list,tuple)), LOGGER.error("swapList items must be a list")
                subSL = []
                for num in sl:
                    assert is_integer(num), LOGGER.error("swapList sub-list items must be integers")
                    num = INT_TYPE(num)
                    assert num>=0, LOGGER.error("swapList sub-list items must be positive")
                    subSL.append(num)
                assert len(set(subSL))==len(subSL), LOGGER.error("swapList items must not have any redundancy")
                assert len(subSL) == self.__swapLength, LOGGER.error("swapList item length must be equal to swapLength")
                SL.append(np.array(subSL, dtype=INT_TYPE))
            self.__swapList = tuple(SL)
    
    def append_to_swap_list(self, subList):
        """
        append a sub list to swap list
        
        :Parameters: 
            #. subList (List): The sub-list of atom indexes to append to swapList.
        """
        assert isinstance(subList, (list,tuple)), LOGGER.error("subList must be a list")
        subSL = []
        for num in subList:
            assert is_integer(num), LOGGER.error("subList items must be integers")
            num = INT_TYPE(num)
            assert num>=0, LOGGER.error("subList items must be positive")
            subSL.append(num)
        assert len(set(subSL))==len(subSL), LOGGER.error("swapList items must not have any redundancy")
        assert len(subSL) == self.__swapLength, LOGGER.error("swapList item length must be equal to swapLength")
        # append
        self.__swapList = list(self.__swapList)
        self.__swapList.append(subSL)
        self.__swapList = tuple(self.__swapList)
    
    def get_ready_for_move(self, groupAtomsIndexes):  
        """
        Set the swap generator ready to perform a move. Unlike a normal move generator,
        swap generators will affect not only the selected atoms but other atoms as well.
        Therefore at engine runtime, selected atoms will be extended to all affected 
        atoms by the swap.\n
        This method is called automatically upon engine runtime
        to ensure that all affect atoms with the swap are updated.
        
        :Parameters: 
            #. groupAtomsIndexes (numpy.ndarray): The atoms indexes to swap.
        
        :Returns: 
            #. indexes (numpy.ndarray): All the atoms involved in the swap move 
               including the given groupAtomsIndexes.
        """
        self.__groupAtomsIndexes = groupAtomsIndexes
        self.__swapAtomsIndexes  = self.swapList[ randint(0,len(self.swapList)-1) ]
        return np.concatenate( (self.__groupAtomsIndexes,self.__swapAtomsIndexes) )
        
        
        
class PathGenerator(MoveGenerator):
    """ 
    PathGenerator is a MoveGenerator sub-class where moves definitions are pre-stored in a path 
    and get pulled out at every move step.
    This class can't be instantiated but its sub-classes might be.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. path (None, list): The list of moves.
        #. randomize (boolean): Whether to pull moves randomly from path or pull moves in order at every step.
    """
    
    def __init__(self, group=None, path=None, randomize=False):
        super(PathGenerator, self).__init__(group=group) 
        # set path
        self.set_path(path)
        # set randomize
        self.set_randomize(randomize)
        # initialize flags
        self.__initialize_path_generator__()
    
    def __initialize_path_generator__(self):
        self.__step = 0
        
    @property
    def step(self):
        """ Get the current step number."""
        return self.__step
        
    @property
    def path(self):
        """ Get the path list of moves."""
        return self.__path
        
    @property
    def randomize(self):
        """ Get randomize flag."""
        return self.__randomize
    
    def check_path(self, path):
        """
        Checks the generator's path.
        This method must be overloaded in all PathGenerator sub-classes.
        
        :Parameters:
            #. path (list): The list of moves.
        """ 
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
    
    def normalize_path(self, path):
        """
        Normalizes all path moves. 
        It is called automatically upon set_path method is called.
        This method can be overloaded in all MoveGenerator sub-classes.
        
        :Parameters:
            #. path (list): The list of moves.
        
        :Returns:
            #. path (list): The list of moves.
        """
        return list(path)
        
    def set_path(self, path):
        """
        Sets the moves path.
        
        :Parameters:
            #. path (list): The list of moves.
        """
        valid, message = self.check_path(path)
        if not valid:
            raise Exception(message)
        # normalize path
        self.__path = self.normalize_path( path )
        # reset generator
        self.__initialize_path_generator__()
        
    def set_randomize(self, randomize):
        """
        Sets whether to randomize moves selection.
        
        :Parameters:
            #. randomize (boolean): Whether to pull moves randomly from path or pull moves in order at every step.
        """
        assert isinstance(randomize, bool), LOGGER.error("randomize must be boolean")
        self.__randomize = randomize
        
    def move(self, coordinates):
        """
        Moves coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the transformation
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the transformation
        """
        if self.__randomize:
            move = self.__path[ randint(0,len(self.__path)-1) ]
        else:
            move = self.__path[self.__step]
            self.__step = (self.__step+1)%len(self.__path)
        # perform the move
        return self.transform_coordinates(coordinates, argument=move)    

    
class MoveGeneratorCombinator(MoveGenerator):
    """ 
    MoveGeneratorCombinator combines all moves of a list of MoveGenerators and applies it at once.
    
    :Parameters:
        #. group (None, Group): The constraint RMC engine.
        #. combination (list): The list of MoveGenerator instances.
        #. shuffle (boolean): Whether to shuffle generator instances at every move or to combine moves in the list order.
    
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Core.MoveGenerator import MoveGeneratorCombinator
        from fullrmc.Generators.Translations import TranslationGenerator
        from fullrmc.Generators.Rotations import RotationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        
        ##### Define each group move generator as a combination of a translation and a rotation. #####
        # create recursive group selector. Recurrence is set to 20 with explore flag set to True.
        # shuffle is set to True which means that at every selection the order of move generation
        # is random. At one step a translation is performed prior to rotation and in another step
        # the rotation is performed at first.
        # selected from the collector.
        for g in ENGINE.groups:
            # create translation generator
            TMG = TranslationGenerator(amplitude=0.2)
            # create rotation generator only when group length is bigger than 1.
            if len(g)>1:
                RMG = RotationGenerator(amplitude=2)
                MG  = MoveGeneratorCombinator(collection=[TMG,RMG],shuffle=True)
            else:
                MG  = MoveGeneratorCombinator(collection=[TMG],shuffle=True)
            g.set_move_generator( MG )
    """
    
    def __init__(self, group=None, combination=None, shuffle=False):
        # set combination
        self.__combination = []
        # initialize
        super(MoveGeneratorCombinator, self).__init__(group=group) 
        # set path
        self.set_combination(path)
        # set randomize
        self.set_shuffle(shuffle)
        
    @property
    def shuffle(self):
        """ Get shuffle flag."""
        return self.__shuffle
        
    @property
    def combination(self):
        """ Get the combination list of MoveGenerator instances."""
        return self.__combination
        
    def check_group(self, group):
        """
        Checks the generator's group.
        This methods always returns True because normally all combination MoveGenerator instances groups are checked.
        This method must NOT be overloaded unless needed.
        
        :Parameters:
            #. group (Group): the Group instance
        """
        return True, ""
    
    def set_group(self, group):
        """
        Set the MoveGenerator group.
        
        :Parameters:
            #. group (None, Group): group instance. 
        """
        MoveGenerator.set_group(self, group)
        for mg in self.__combination:
            mg.set_group(group)
         
    def set_combination(self, combination):
        """
        Sets the generators combination list.
        
        :Parameters:
            #. combination (list): The list of MoveGenerator instances.
        """
        assert isinstance(combination, (list,set,tuple)), LOGGER.error("combination must be a list")
        combination = list(combination)
        for c in combination:
            assert isinstance(c, MoveGenerator), LOGGER.error("every item in combination list must be a MoveGenerator instance")
            c.set_group(self.group)
        self.__combination = combination
        
    def set_shuffle(self, shuffle):
        """
        Sets whether to shuffle moves generator.
        
        :Parameters:
            #. shuffle (boolean): Whether to shuffle generator instances at every move or to combine moves in the list order.
        """
        assert isinstance(shuffle, bool), LOGGER.error("shuffle must be boolean")
        self.__shuffle = shuffle

    def move(self, coordinates):
        """
        Moves coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the transformation
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the transformation
        """
        indexes = range(len(self.__combination))
        if self.__shuffle:
            shuffle( indexes )
        # create the move combination
        for idx in indexes:
            coordinates = self.__combination[idx].move(coordinates)
        return coordinates
    
        
class MoveGeneratorCollector(MoveGenerator):
    """ 
    MoveGeneratorCollector collects MoveGenerators instances and applies the move of one instance at every step.
    
    :Parameters:
        #. group (None, Group): The constraint RMC engine.
        #. collection (list): The list of MoveGenerator instances.
        #. randomize (boolean): Whether to pull MoveGenerator instance randomly from collection list or in order.     
        #. weights (None, list): Generators selections Weights list. 
           It must be None for equivalent weighting or list of (generatorIndex, weight) tuples. 
           If randomize is False, weights list is ignored upon generator selection from collection.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
        from fullrmc.Generators.Translations import TranslationGenerator
        from fullrmc.Generators.Rotations import RotationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        
        ##### Define each group move generator as a combination of a translation and a rotation. #####
        # create recursive group selector. Recurrence is set to 20 with explore flag set to True.
        # randomize is set to True which means that at every selection a generator is randomly
        # selected from the collector.
        for g in ENGINE.groups:
            # create translation generator
            TMG = TranslationGenerator(amplitude=0.2)
            # create rotation generator only when group length is bigger than 1.
            if len(g)>1:
                RMG = RotationGenerator(amplitude=2)
                MG  = MoveGeneratorCollector(collection=[TMG,RMG],randomize=True)
            else:
                MG  = MoveGeneratorCollector(collection=[TMG],randomize=True)
            g.set_move_generator( MG )
            
    """
    def __init__(self, group=None, collection=None, randomize=True, weights=None):
        # set collection
        self.__collection = []
        # initialize
        super(MoveGeneratorCollector, self).__init__(group=group) 
        # set path
        self.set_collection(collection)
        # set randomize
        self.set_randomize(randomize)
        # set weights
        self.set_weights(weights)
        # initialize flags
        self.__initialize_generator()
    
    def __initialize_generator(self):
        self.__step = 0
    
    def __check_single_weight(self, w):
        """Checks a single group weight tuple format"""
        assert isinstance(w, (list,set,tuple)),LOGGER.error("weights list items must be tuples")
        assert len(w)==2, LOGGER.error("weights list tuples must have exactly 2 items")
        idx  = w[0]
        wgt = w[1]
        assert is_integer(idx), LOGGER.error("weights list tuples first item must be an integer")
        idx = INT_TYPE(idx)
        assert idx>=0, LOGGER.error("weights list tuples first item must be positive")
        assert idx<len(self.__collection), LOGGER.error("weights list tuples first item must be smaller than the number of generators in collection")
        assert is_number(wgt), LOGGER.error("weights list tuples second item must be an integer")
        wgt = FLOAT_TYPE(wgt)
        assert wgt>0, LOGGER.error("weights list tuples first item must be bigger than 0")
        # all True return idx and weight
        return idx, wgt  
        
    @property
    def randomize(self):
        """ Get randomize flag."""
        return self.__randomize
        
    @property
    def collection(self):
        """ Get the list of MoveGenerator instances."""
        return self.__collection
        
    @property
    def generatorsWeight(self):
        """ Generators selection weights list."""
        return self.__generatorsWeight
        
    @property    
    def selectionScheme(self):
        return self.__selectionScheme
    
    def set_group(self, group):
        """
        Set the MoveGenerator group.
        
        :Parameters:
            #. group (None, Group): group instance. 
        """
        MoveGenerator.set_group(self, group)
        for mg in self.__collection:
            mg.set_group(group)
            
    def check_group(self, group):
        """
        Checks the generator's group.
        This methods always returns True because normally all collection MoveGenerator instances groups are checked.
        This method must NOT be overloaded unless needed.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, ""
        
    def set_collection(self, collection):
        """
        Sets the generators instances collection list.
        
        :Parameters:
            #. collection (list): The list of move generator instance.
        """
        assert isinstance(collection, (list,set,tuple)), LOGGER.error("collection must be a list")
        collection = list(collection)
        for c in collection:
            assert isinstance(c, MoveGenerator), LOGGER.error("every item in collection list must be a MoveGenerator instance")
            c.set_group(self.group)
        self.__collection = collection
        # reset generator
        self.__initialize_generator()
        
    def set_randomize(self, randomize):
        """
        Sets whether to randomize MoveGenerator instance selection from collection list.
        
        :Parameters:
            #. randomize (boolean): Whether to pull MoveGenerator instance randomly from collection list or in order.
        """
        assert isinstance(randomize, bool), LOGGER.error("randomize must be boolean")
        self.__randomize = randomize

    def set_weights(self, weights): 
        """
        Set groups selection weighting scheme.
        
        :Parameters:
            #. weights (None, list): Generators selections Weights list. 
               It must be None for equivalent weighting or list of (generatorIndex, weight) tuples. 
               If randomize is False, weights list is ignored upon generator selection from collection.
        """  
        generatorsWeight = np.ones(len(self.__collection), dtype=FLOAT_TYPE)      
        if weights is not None:
            assert isinstance(weights, (list,set,tuple)),LOGGER.error("weights must be a list")
            for w in weights:
                idx, wgt = self.__check_single_weight(w)
                # update groups weight
                generatorsWeight[idx] = wgt
        # set groups weight
        self.__generatorsWeight = generatorsWeight
        # create selection histogram
        self.set_selection_scheme()
    
    def set_selection_scheme(self):
        """ Sets selection scheme. """
        cumsumWeights = np.cumsum(self.__generatorsWeight, dtype=FLOAT_TYPE)
        self.__selectionScheme = cumsumWeights/cumsumWeights[-1]
        
    def move(self, coordinates):
        """
        Moves coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the transformation
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the transformation
        """
        if self.__randomize:
            index = INT_TYPE( np.searchsorted(self.__selectionScheme, generate_random_float()) )
            moveGenerator = self.__collection[ index ]
        else:
            moveGenerator = self.__collection[self.__step]
            self.__step   = (self.__step+1)%len(self.__collection)
        # perform the move
        return moveGenerator.move(coordinates) 
        
                
        
        