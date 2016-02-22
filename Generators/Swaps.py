"""
Swaps contains all swap or atoms position exchange MoveGenerator classes.

.. inheritance-diagram:: fullrmc.Generators.Swaps
    :parts: 1 
                                                                                          
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_number, is_integer
from fullrmc.Core.MoveGenerator import  MoveGenerator, SwapGenerator


class SwapPositionsGenerator(SwapGenerator):
    """
    Generates positional swapping between atoms of the selected group and other atoms
    randomly selected from swapList.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. swapLength (Integer): The swap length that defines the length of the group 
           and the length of the every swap sub-list in swapList.
        #. swapList (None, List): The list of atoms.\n
           If None is given, no swapping or exchanging will be performed.\n
           If List is given, it must contain lists of atoms where every 
           sub-list must have the same number of atoms as the group.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Swaps import SwapPositionsGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        ##### set swap moves between Lithium and Manganese atoms in Li2MnO3 system #####
        # reset engine groups to atoms to insure atomic grouping of all the system's atoms
        ENGINE.set_groups_as_atoms()
        # get all elements list 
        elements = ENGINE.allElements
        # create list of lithium atoms indexes
        liIndexes = [[idx] for idx in xrange(len(elements)) if elements[idx]=='li']
        # create list of manganese atoms indexes
        mnIndexes = [[idx] for idx in xrange(len(elements)) if elements[idx]=='mn']
        # create swap generator to lithium atoms
        swapWithLi = SwapPositionsGenerator(swapList=liIndexes)
        # create swap generator to manganese atoms
        swapWithMn = SwapPositionsGenerator(swapList=mnIndexes)
        # set swap generator to groups
        for g in ENGINE.groups:
            # get group's atom index
            idx = g.indexes[0]
            # set swap to manganese for lithium atoms
            if elements[idx]=='li':
                g.set_move_generator(swapWithMn)
            # set swap to lithium for manganese atoms
            elif elements[idx]=='mn':
                g.set_move_generator(swapWithLi)
            # the rest are oxygen atoms. Default RandomTranslation generator are kept.
                                                                      
    """
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, ""
       
    def set_swap_length(self, swapLength):
        """
        Set swap length. it will reset swaplist automatically.
    
        :Parameters:
            #. swapLength (Integer): The swap length that defines the length of the group 
               and the length of the every swap sub-list in swapList.
        """   
        super(SwapPositionsGenerator, self).set_swap_length(swapLength=swapLength)
        self.__swapArray = np.empty( (self.swapLength,3), dtype=FLOAT_TYPE )
        
    def transform_coordinates(self, coordinates, argument=None):
        """
        Transform coordinates by swapping. This method is called in every move.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the swapping.
            #. argument (object): Any other argument needed to perform the move.
               In General it's not needed.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the move.
        """
        # swap coordinates
        self.__swapArray[:,:] = coordinates[:self.swapLength ,:]
        coordinates[:self.swapLength ,:] = coordinates[self.swapLength :,:]
        coordinates[self.swapLength :,:] = self.__swapArray[:,:]
        # return
        return coordinates


class SwapCentersGenerator(SwapGenerator):
    """
    Computes geometric center of the selected group, and swaps its atoms
    by translation to the atoms geometric center of the other atoms which
    are randomly selected from swapList and vice-versa. 
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. swapList (None, List): The list of atoms.\n
           If None is given, no swapping or exchanging will be performed.\n
           If List is given, it must contain lists of atom indexes.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Swaps import SwapCentersGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        ##### set swap moves between first 10 molecular groups of a system #####
        # reset engine groups to molecules
        ENGINE.set_groups_as_molecules()
        # set swap generator to the first 10 groups
        GROUPS = [ENGINE.groups[idx] for idx in range(10)]
        for gidx, group in enumerate(GROUPS):
            swapList = [g.indexes for idx,g in enumerate(GROUPS) if idx!=gidx]
            swapGen  = SwapCentersGenerator(swapList=swapList)
            group.set_move_generator(swapGen)    
    """  
    def __init__(self, group=None, swapList=None):
        super(SwapCentersGenerator, self).__init__(group=group, swapLength=None, swapList=swapList ) 

    @property
    def swapLength(self):
        """ Get swap length. In this Case it is always None as 
        swapLength is not required for this generator."""
        return self.__swapLength 
    
    @property
    def swapList(self):
        """ Get swap list."""
        return self.__swapList
        
    def set_swap_length(self, swapLength):
        """
        Set swap length. The swap length that defines the length of the group 
        and the length of the every swap sub-list in swapList. 
        It will automatically be set to None as SwapCentersGenerator 
        does not require a fixed length.
    
        :Parameters:
            #. swapLength (None): The swap length.
        """   
        self.__swapLength = None
        self.__swapList   = ()
        
    def set_swap_list(self, swapList):
        """
        Set the swap-list to swap groups centers.
        
        :Parameters: 
            #. swapList (None, List): The list of atoms.\n 
               If None is given, no swapping or exchanging will be performed.\n
               If List is given, it must contain lists of atom indexes.
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
                SL.append(np.array(subSL, dtype=INT_TYPE))
            self.__swapList = tuple(SL)
            
    def set_group(self, group):
        """
        Set the MoveGenerator group.
        
        :Parameters:
            #. group (None, Group): group instance. 
        """
        MoveGenerator.set_group(self, group)
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, ""
       
    def transform_coordinates(self, coordinates, argument=None):
        """
        Transform coordinates by swapping. This method is called in every move.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the swapping.
            #. argument (object): Any other argument needed to perform the move.
               In General it's not needed.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the move.
        """
        # get translation vector
        swapLength    = len(self.groupAtomsIndexes)
        swapsOfCenter = np.mean(coordinates[:swapLength,:], axis=0)
        swapsToCenter = np.mean(coordinates[swapLength :,:], axis=0)
        direction     = swapsToCenter-swapsOfCenter
        # swap by translation
        coordinates[: swapLength,:] += direction
        coordinates[swapLength :,:] -= direction
        # return
        return coordinates
        
        
        