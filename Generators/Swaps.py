"""
Translations contains all swap MoveGenerator classes.

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
    """
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, ""
       

    


    