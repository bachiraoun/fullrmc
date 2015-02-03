"""
.. inheritance-diagram:: fullrmc.Selectors.RandomSelectors
    :parts: 2 
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, generate_random_float, generate_random_integer
from fullrmc.Core.Collection import is_integer, is_number
from fullrmc.Core.GroupSelector import GroupSelector


class RandomSelector(GroupSelector):
    """
    RandomSelector generates indexes randomly for engine group selection
    """
    
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE(generate_random_integer(0,len(self.engine.groups)-1))
        

   
class weightedRandomSelector(RandomSelector):
    """
    weightedRandomSelector generates indexes randomly following groups weighting scheme.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. weights (None, list): Weights list. It must be None for equivalent weighting or list of (groupIndex, weight) tuples
    """
    
    def __init__(self, engine, recur=None, weights=None):
        # initialize GroupSelector
        super(DefinedOrderSelector, self).__init__(engine=engine, recur=recur)
        # set weights
        self.set_weights(weights)
        
    def set_weights(self, weights): 
        """
        Set groups selection weighting
        
        :Parameters:
            #. weights (None, list): Weights list. It must be None for equivalent weighting or list of (groupIndex, weight) tuples 
        """  
        groupsWeight = np.ones(len(self.engine.groups), dtype=FLOAT_TYPE)      
        if weights is not None:
            for w in weights:
                assert isinstance(w, (list,set,tuple)),"weights list items must be tuples"
                assert len(w)==2, "weights list tuples must have exactly 2 items"
                idx  = w[0]
                wgt = w[1]
                assert is_integer(idx), "weights list tuples first item must be an integer"
                idx = INT_TYPE(idx)
                assert idx>=0, "weights list tuples first item must be positive"
                assert idx<len(self.engine.groups), "weights list tuples first item must be smaller than engine's number of groups"
                assert is_number(wgt), "weights list tuples second item must be an integer"
                wgt = FLOAT_TYPE(wgt)
                assert wgt>=0, "weights list tuples first item must be positive"
                # update groups weight
                groupsWeight[idx] = wgt
        # set groups weight
        self.__groupsWeight = groupsWeight
        # create selection histogram
        self.update_selection_scheme()
        
    def set_group_weight(self, groupWeight):
        """
        Set a single group weight.
        
        :Parameters:
            #. groupWeight (list, set, tuple): Group weight list composed of groupIndex as first element and groupWeight as second.
        """
        assert isinstance(groupWeight, (list,set,tuple)),"groupWeight must be a tuple"
        assert len(groupWeight)==2, "groupWeight must have exactly 2 items"
        idx  = groupWeight[0]
        wgt = groupWeight[1]
        assert is_integer(idx), "groupWeight first item must be an integer"
        idx = INT_TYPE(idx)
        assert idx>=0, "groupWeight first item must be positive"
        assert idx<len(self.engine.groups), "groupWeight first item must be smaller than engine's number of groups"
        assert is_number(wgt), "groupWeight second item must be an integer"
        wgt = FLOAT_TYPE(wgt)
        assert wgt>=0, "groupWeight first item must be positive"
        # update groups weight
        self.__groupsWeight[idx] = wgt
        # create selection histogram
        self.update_selection_scheme()
    
    def update_selection_scheme(self):
        cumsumWeights = np.cumsum(self.__groupsWeight, dtype=FLOAT_TYPE)
        self.__selectionScheme = cumsumWeights/cumsumWeights[-1]
        
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE( np.sortedsearch(self.__selectionScheme, generate_random_float()) )

    
    
            
            
            