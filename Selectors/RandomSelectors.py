"""
RandomSelectors contains GroupSelector classes of random order of selections.

.. inheritance-diagram:: fullrmc.Selectors.RandomSelectors
    :parts: 1
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, generate_random_float, generate_random_integer, LOGGER
from fullrmc.Core.Collection import is_integer, is_number
from fullrmc.Core.GroupSelector import GroupSelector


class RandomSelector(GroupSelector):
    """
    RandomSelector generates indexes randomly for engine group selection.
    """
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE(generate_random_integer(0,len(self.engine.groups)-1))
        

class WeightedRandomSelector(RandomSelector):
    """
    WeightedRandomSelector generates indexes randomly following groups weighting scheme.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. recur (None, integer): Set number of times to recur.
           If None, recur is equivalent to 0.
           Recurrence property is only used when the selector instance is wrapped with a RecursiveGroupSelector.
        #. weights (None, list): Weights list. It must be None for equivalent weighting or list of (groupIndex, weight) tuples.
    """
    
    def __init__(self, engine, recur=None, weights=None):
        # initialize GroupSelector
        super(WeightedRandomSelector, self).__init__(engine=engine, recur=recur)
        # set weights
        self.set_weights(weights)
        
    def __check_single_weight(self, w):
        """Checks a single group weight tuple format"""
        assert isinstance(w, (list,set,tuple)),LOGGER.error("weights list items must be tuples")
        assert len(w)==2, LOGGER.error("weights list tuples must have exactly 2 items")
        idx  = w[0]
        wgt = w[1]
        assert is_integer(idx), LOGGER.error("weights list tuples first item must be an integer")
        idx = INT_TYPE(idx)
        assert idx>=0, LOGGER.error("weights list tuples first item must be positive")
        assert idx<len(self.engine.groups), LOGGER.error("weights list tuples first item must be smaller than engine's number of groups")
        assert is_number(wgt), LOGGER.error("weights list tuples second item must be an integer")
        wgt = FLOAT_TYPE(wgt)
        assert wgt>0, LOGGER.error("weights list tuples first item must be bigger than 0")
        # all True return idx and weight
        return idx, wgt  
    
    @property    
    def groupsWeight(self):
        return self.__groupsWeight
        
    @property    
    def selectionScheme(self):
        return self.__selectionScheme
        
    def set_weights(self, weights): 
        """
        Set groups selection weighting scheme.
        
        :Parameters:
            #. weights (None, list): Weights list. It must be None for equivalent weighting or list of (groupIndex, weight) tuples. 
        """  
        groupsWeight = np.ones(len(self.engine.groups), dtype=FLOAT_TYPE)      
        if weights is not None:
            assert isinstance(weights, (list,set,tuple)),LOGGER.error("weights must be a list")
            for w in weights:
                idx, wgt = self.__check_single_weight(w)
                # update groups weight
                groupsWeight[idx] = wgt
        # set groups weight
        self.__groupsWeight = groupsWeight
        # create selection histogram
        self.set_selection_scheme()
        
    def set_group_weight(self, groupWeight):
        """
        Set a single group weight.
        
        :Parameters:
            #. groupWeight (list, set, tuple): Group weight list composed of groupIndex as first element and groupWeight as second.
        """
        idx, wgt = self.__check_single_weight(groupWeight)
        # update groups weight
        self.__groupsWeight[idx] = wgt
        # create selection histogram
        self.set_selection_scheme()
    
    def set_selection_scheme(self):
        """ Sets selection scheme. """
        cumsumWeights = np.cumsum(self.__groupsWeight, dtype=FLOAT_TYPE)
        self.__selectionScheme = cumsumWeights/cumsumWeights[-1]
        
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE( np.searchsorted(self.__selectionScheme, generate_random_float()) )

    
    
class DynamicalWeightsRandomSelector(WeightedRandomSelector):   
    """
    DynamicalWeightsRandomSelector generates indexes randomly following groups weighting scheme.
    Weighted scheme dynamically updates at engine runtime, making use of an intelligent machine 
    learning algorithm.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. recur (None, integer): Set number of times to recur.
           If None, recur is equivalent to 0.
           Recurrence property is only used when the selector instance is wrapped with a RecursiveGroupSelector.
        #. weights (None, list): Weights list. It must be None for equivalent weighting or list of (groupIndex, weight) tuples.
        #. factor (Number): The weight increase of every group when a step get accepted.
           Must be a positive number.
        #. reduce (bool): Whether to reduce by factor a group's weight when a move is rejected.
           Reduction by factor will be performed only if group weight remains positive.
    """
    
    def __init__(self, engine, recur=None, weights=None, factor=1, reduce=True):
        # initialize GroupSelector
        super(DynamicalWeightsRandomSelector, self).__init__(engine=engine, recur=recur, weights=weights)
        # set weights factor
        self.set_factor(factor)
        # set reduce flag
        self.set_reduce(reduce)
        
    @property
    def factor(self):
        """The weight factor."""
        return self.__factor
        
    @property
    def reduce(self):
        """The reduce flag."""
        return self.__reduce
           
    @property    
    def selectionScheme(self):
        return self.__selectionScheme
        
    def set_factor(self, factor):
        """
        Set the weight factor.
    
        :Parameters:
            #. factor (Number): The weight increase of every group when a step get accepted.
               Must be a positive number.
        """
        assert is_number(factor), LOGGER.error("factor must be a number")
        factor = FLOAT_TYPE(factor)
        assert factor>=0, LOGGER.error("factor must be positive")
        self.__factor = factor
            
    def set_reduce(self, reduce):
        """
        Set reduce flag.
    
        :Parameters:
            #. reduce (bool): Whether to reduce by factor a group's weight when a move is rejected.
        """
        assert isinstance(reduce, bool), LOGGER.error("reduce must be a boolean")
        self.__reduce = reduce
             
    def set_selection_scheme(self):
        """ Sets selection scheme. """
        self.__selectionScheme  = np.cumsum(self.groupsWeight, dtype=FLOAT_TYPE)
         
    def move_accepted(self, index):
        """
        This method is called by the engine when a move generated on a group is accepted.
        This method is empty must be overloaded when needed.
        
        :Parameters:
            #. index (integer): the selected group index in engine groups list
        """
        self.__selectionScheme[index:] += self.__factor
    
    def move_rejected(self, index):
        """
        This method is called by the engine when a move generated on a group is rejected.
        This method is empty must be overloaded when needed.
        
        :Parameters:
            #. index (integer): the selected group index in engine groups list
        """
        if not reduce:
            return
        if index == 0:
            if  self.__selectionScheme[index] - self.__factor > 0:
                self.__selectionScheme[index:] -= self.__factor  
        elif self.__selectionScheme[index] - self.__factor > self.__selectionScheme[index-1]:
            self.__selectionScheme[index:] -= self.__factor  
                  
    
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE( np.searchsorted(self.__selectionScheme, generate_random_float()*self.__selectionScheme[-1]) )
        
        
        

        
            