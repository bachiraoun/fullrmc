"""
Swaps contains all swap or atoms position exchange MoveGenerator classes.

.. inheritance-diagram:: fullrmc.Generators.Swaps
    :parts: 1 
                                                                                          
"""

# standard libraries imports

# external libraries imports

# fullrmc imports
from fullrmc.Core.MoveGenerator import  SwapGenerator


class SwapPositionsGenerator(SwapGenerator):
    """
    Generates swapping between atoms of the selected group and other atoms
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
       

    


    