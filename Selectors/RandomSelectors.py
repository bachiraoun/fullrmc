"""
RandomSelectors contains GroupSelector classes of random order of selections.

.. inheritance-diagram:: fullrmc.Selectors.RandomSelectors
    :parts: 1

+------------------------------------------------------+------------------------------------------------------+
| Machine learning on group selection is shown herein. Groups are set to single atom where only random        |
| translation moves generators with different amplitudes are used allowing moves to be accepted in different  |
| ratios. In those two examples :class:`SmartRandomSelector` allowing machine learning upon group             |
| selection. No experimental constraints are used but only inter-molecular distances, intra-molecular bonds,  |
| angles and improper angles constraints are used to keep the integrity of the system and molecules.          |
| As one can see, group selection machine learning is very effective allowing consequent improvement on the   |
| accepted moves. Still, fast convergence of the system and the ratio of accepted moves is highly correlated  |
| with the move generator assigned to the groups.                                                             |
+------------------------------------------------------+------------------------------------------------------+ 
|.. figure:: machineLearningSelectionAmp0p3.png        |.. figure:: machineLearningSelectionAmp0p25.png       |
|   :width: 375px                                      |   :width: 375px                                      |
|   :height: 300px                                     |   :height: 300px                                     |
|   :align: left                                       |   :align: left                                       |
|                                                      |                                                      |
|   25% of assigned moves generators amplitude is set  |   25% of assigned moves generators amplitude is set  |
|   to 10A allowing very few moves on those groups     |   to 10A allowing very few moves on those groups     |
|   to be accepted and the rest of moves generators    |   to be accepted and the rest of moves generators    |
|   amplitudes is set to 0.3A.                         |   amplitudes is set to 0.25A.                        |
|                                                      |                                                      |
+------------------------------------------------------+------------------------------------------------------+    
   
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_integer, is_number, generate_random_float, generate_random_integer
from fullrmc.Core.GroupSelector import GroupSelector


class RandomSelector(GroupSelector):
    """
    RandomSelector generates indexes randomly for engine group selection.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector fullrmc engine instance.
        
        
    .. code-block:: python
        
        # import external libraries
        import numpy as np
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Selectors.RandomSelectors import RandomSelector
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups generators as needed ...
        
        # set group selector as random selection from all defined groups.
        ENGINE.set_group_selector( RandomSelector(ENGINE) )  
        
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
        #. weights (None, list): Weights list. It must be None for equivalent weighting or list of (groupIndex, weight) tuples.
    
    .. code-block:: python
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Selectors.RandomSelectors import WeightedRandomSelector
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups generators as needed ...
        
        # set group selector as random selection but with double likelihood to
        # selecting the first and the last group.
        weights = [1 for _ in ENGINE.pdb.indexes]
        weights[0] = weights[-1] = 2 
        ENGINE.set_group_selector( WeightedRandomSelector(ENGINE, weights) )  
        
    """
    def __init__(self, engine, weights=None):
        # initialize GroupSelector
        super(WeightedRandomSelector, self).__init__(engine=engine)
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
    
    def _set_selection_scheme(self):
        """ Sets selection scheme. """
        cumsumWeights = np.cumsum(self.__weights, dtype=FLOAT_TYPE)
        self.__selectionScheme = cumsumWeights/cumsumWeights[-1]
        
    @property    
    def weights(self):
        """Groups weight of selection as initialized."""
        return self.__weights
        
    @property    
    def groupsWeight(self):
        """Groups weight of selection at current state."""
        groupsWeight = np.copy(self.selectionScheme)
        if len(self.selectionScheme) > 1:
            groupsWeight[1:] -= self.selectionScheme[:-1]
        return groupsWeight
        
    @property    
    def selectionScheme(self):
        """Groups selection scheme used upon group selection."""
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
        self.__weights = groupsWeight
        # create selection histogram
        self._set_selection_scheme()
        
    def set_group_weight(self, groupWeight):
        """
        Set a single group weight.
        
        :Parameters:
            #. groupWeight (list, set, tuple): Group weight list composed of groupIndex as first element and groupWeight as second.
        """
        idx, wgt = self.__check_single_weight(groupWeight)
        # update groups weight
        self.__weights[idx] = wgt
        # create selection histogram
        self._set_selection_scheme()
        
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE( np.searchsorted(self.__selectionScheme, generate_random_float()) )

    
    
class SmartRandomSelector(WeightedRandomSelector):   
    """
    SmartRandomSelector is a random group selector fed with machine learning algorithm.
    The indexes generation is biased and it evolves throughout the simulation towards 
    selecting groups with more successful moves history.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. weights (None, list): Weights list fed as initial biasing scheme. 
           It must be None for equivalent weighting or list of (groupIndex, weight) tuples.
        #. biasFactor (Number): The biasing factor of every group when a step get accepted.
           Must be a positive number.
        #. unbiasFactor(None, Number): Whether to un-bias a group's weight when a move is rejected.
           If None, un-biasing is turned off.
           Un-biasing will be performed only if group weight remains positive.
    
    .. code-block:: python
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Selectors.RandomSelectors import SmartRandomSelector
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups generators as needed ...
        
        # set group selector as random smart selection that will adjust its
        # weighting scheme to improve the chances of moves getting accepted.        
        ENGINE.set_group_selector( SmartRandomSelector(ENGINE) )  
        
    """
    
    def __init__(self, engine, weights=None, biasFactor=1, unbiasFactor=None):
        # initialize GroupSelector
        super(SmartRandomSelector, self).__init__(engine=engine, weights=weights)
        # set bias factor
        self.set_bias_factor(biasFactor)
        # set un-bias factor
        self.set_unbias_factor(unbiasFactor)
    
    def _set_selection_scheme(self):
        """ Sets selection scheme. """
        self.__selectionScheme = np.cumsum(self.weights, dtype=FLOAT_TYPE)
        
    @property
    def biasFactor(self):
        """The biasing factor."""
        return self.__biasFactor
        
    @property
    def unbiasFactor(self):
        """The unbiasing factor."""
        return self.__unbiasFactor
           
    @property    
    def selectionScheme(self):
        """Groups selection scheme used upon group selection."""
        return self.__selectionScheme
        
    def set_bias_factor(self, biasFactor):
        """
        Set the biasing factor.
    
        :Parameters:
            #. biasFactor (Number): The biasing factor of every group when a step get accepted.
               Must be a positive number.
        """
        assert is_number(biasFactor), LOGGER.error("biasFactor must be a number")
        biasFactor = FLOAT_TYPE(biasFactor)
        assert biasFactor>=0, LOGGER.error("biasFactor must be positive")
        self.__biasFactor = biasFactor
            
    def set_unbias_factor(self, unbiasFactor):
        """
        Set the unbiasing factor.
    
        :Parameters:
            #. unbiasFactor(None, Number): Whether to unbias a group's weight when a move is rejected.
               If None, unbiasing is turned off.
               Unbiasing will be performed only if group weight remains positive.
        """
        if unbiasFactor is not None:
            assert is_number(unbiasFactor), LOGGER.error("unbiasFactor must be a number")
            unbiasFactor = FLOAT_TYPE(unbiasFactor)
            assert unbiasFactor>=0, LOGGER.error("unbiasFactor must be positive")
        self.__unbiasFactor = unbiasFactor
         
    def move_accepted(self, index):
        """
        This method is called by the engine when a move generated on a group is accepted.
        This method is empty must be overloaded when needed.
        
        :Parameters:
            #. index (integer): the selected group index in engine groups list
        """
        self.__selectionScheme[index:] += self.__biasFactor
    
    def move_rejected(self, index):
        """
        This method is called by the engine when a move generated on a group is rejected.
        This method is empty must be overloaded when needed.
        
        :Parameters:
            #. index (integer): the selected group index in engine groups list
        """
        if self.__unbiasFactor is None:
            return
        if index == 0:
            if  self.__selectionScheme[index] - self.__unbiasFactor > 0:
                self.__selectionScheme[index:] -= self.__unbiasFactor  
        elif self.__selectionScheme[index] - self.__unbiasFactor > self.__selectionScheme[index-1]:
            self.__selectionScheme[index:] -= self.__unbiasFactor  
                  
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        return INT_TYPE( np.searchsorted(self.__selectionScheme, generate_random_float()*self.__selectionScheme[-1]) )
        
        
        

        
            