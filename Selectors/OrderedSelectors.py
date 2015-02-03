"""
.. inheritance-diagram:: fullrmc.Selectors.OrderedSelectors
    :parts: 2
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE
from fullrmc.Core.Collection import is_integer, is_number
from fullrmc.Core.GroupSelector import GroupSelector


class DefinedOrderSelector(GroupSelector):
    """
    DefinedOrderSelector is a group selector with a defined order of selection.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. order (None, list, set, tuple, numpy.ndarray): The selector order of groups.
           If None, order is set automatically to all groups indexes list.
    """
    
    def __init__(self, engine, recur=None, order=None):
        # initialize GroupSelector
        super(DefinedOrderSelector, self).__init__(engine=engine, recur=recur)
        # set order
        self.set_order(order)
        # initialize selector
        self.__initialize_selector__()
        
    def __initialize_selector__(self):
        self.__index = 0
        
    @property
    def order(self):
        """ Get a list copy of the order of selection."""
        return list(self.__order)
        
    def set_order(self, order):
        """
        Set selector groups order
        
        :Parameters:
            #. order (None, list, set, tuple, numpy.ndarray): The selector order of groups
        """
        newOrder = []
        if order is None:
            newOrder = range(len(self.engine.groups))
        else:
            assert isinstance(order, (list, set, tuple, np.ndarray)), "order must a instance among list, set, tuple or numpy.ndarray"
            if isinstance(order, np.ndarray):
                assert len(order.shape)==1,"order numpy.ndarray must have one dimension"
            order = list(order)
            assert len(order)>0, "order can't be empty"
            for idx in order:
                assert is_integer(idx), "order indexes must be integers"
                idx = int(idx)
                assert idx>=0, "order indexes must be positive"
                assert idx<len(self.engine.groups), "order indexes must be smaller than engine's number of groups"
                newOrder.append(idx)
        # set order
        self.__order = np.array(newOrder, dtype=INT_TYPE)
        # re-initialize selector
        self.__initialize_selector__()
        
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        # get group index
        groupIndex = self.__order[self.__index]
        # update order index 
        self.__index = (self.__index+1)%len(self.__order)
        # return group index
        return groupIndex 
        
        
        
        
        
        
        
        
        