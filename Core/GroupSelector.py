"""
GroupSelector contains parent classes for all group selectors.
A GroupSelector is used at the Engine runtime to select groups upon which a move will be applied.
Therefore it has become possible to fully customize the selection of groups of atoms and to choose 
when and how frequently a group can be chosen to perform a move upon.

.. inheritance-diagram:: fullrmc.Core.GroupSelector
    :parts: 1
"""

# standard libraries imports
import inspect

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_integer
from fullrmc.Core.Group import Group
from fullrmc.Core.MoveGenerator import PathGenerator


class GroupSelector(object):
    """ 
    GroupSelector is the parent class that selects groups to perform moves upon engine runtime.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector fullrmc engine instance.
    """
    
    def __init__(self, engine=None):
        # set engine
        self.set_engine(engine)

    def _runtime_initialize(self):
        """   
        This method is called by the engine at the runtime to initialize the group selector if needed.
        """
        pass
        
    @property
    def engine(self):
        """ Get the engine instance."""
        return self.__engine
    
    @property
    def refine(self):
        """ 
        Get refine flag value. It will always return False because
        refine is a property of RecursiveGroupSelector instances only. 
        """
        return False
    
    @property
    def explore(self):
        """ 
        Get explore flag value. It will always return False because
        explore is a property of RecursiveGroupSelector instances only. 
        """
        return False
        
    @property
    def willSelect(self):
        """ 
        Get whether next step a new selection is occur or still the same group is going to be selected again. 
        It will always return True because recurrence is a property of RecursiveGroupSelector instances only.
        """
        return True
    
    @property
    def willRecur(self):
        """ 
        Get whether next step the same group will be returned.
        It will always return False because this is a property of RecursiveGroupSelector instances only.
        """
        return False
    
    @property
    def willRefine(self):
        """ 
        Get whether selection is recurring and refine flag is True.
        It will always return False because recurrence is a property of RecursiveGroupSelector instances only.
        """
        return False
    
    @property
    def willExplore(self):
        """ 
        Get whether selection is recurring and explore flag is True.
        It will always return False because recurrence is a property of RecursiveGroupSelector instances only.
        """
        return False
        
    @property
    def isNewSelection(self):
        """ 
        Get whether the last step a new selection was made. 
        It will always return True because recurrence is a property of RecursiveGroupSelector instances only.
        """
        return True
        
    @property
    def isRecurring(self):
        """ 
        Get whether the last step the same group was returned.
        It will always return False because this is a property of RecursiveGroupSelector instances only.
        """
        return False
        
    @property
    def isRefining(self):
        """ 
        Get whether selection is recurring and refine flag is True.
        It will always return False because recurrence is a property of RecursiveGroupSelector instances only.
        """
        return False
    
    @property
    def isExploring(self):
        """ 
        Get whether selection is recurring and explore flag is True.
        It will always return False because recurrence is a property of RecursiveGroupSelector instances only.
        """
        return False
        
    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. arguments (object): Any type of argument to pass to the listeners.
        """
        pass
        
    def set_engine(self, engine):
        """
        Sets the selector fullrmc engine instance.
        
        :Parameters:
            #. engine (None, fullrmc.Engine): The selector fullrmc engine.
        """
        if engine is not None:
            from fullrmc.Engine import Engine
            assert isinstance(engine, Engine), LOGGER.error("engine must be None or fullrmc Engine instance")
        self.__engine = engine
        
    def select_index(self):
        """
        This method must be overloaded in every GroupSelector sub-class
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
    
    def move_accepted(self, index):
        """
        This method is called by the engine when a move generated on a group is accepted.
        This method is empty must be overloaded when needed.
        
        :Parameters:
            #. index (integer): the selected group index in engine groups list
        """
        pass
    
    def move_rejected(self, index):
        """
        This method is called by the engine when a move generated on a group is rejected.
        This method is empty must be overloaded when needed.
        
        :Parameters:
            #. index (integer): the selected group index in engine groups list
        """
        pass
        

class RecursiveGroupSelector(GroupSelector):
    """
    RecursiveSelector is the only selector that can use the recursive property on a selection.
    It is used as a wrapper around a GroupSelector instance.
    
    :Parameters:
        #. selector (fullrmc.Core.GroupSelector.GroupSelector): The selector instance to wrap.
        #. recur (integer): Set number of times to recur. 
           It must be a positive integer.
        #. override (boolean): Override temporary recur value.
           recur value will be overridden only when selected group move generator is a PathGenerator instance.
           In this particular case, recur value will be temporary changed to the number of moves stored in the PathGenerator.
           If selected group move generator is not a PathGenerator instance, 
           recur value will take back its original value.
        #. refine (boolean): Its an engine flag that is used to refine the position of a group until 
           recurrence expires and a new group is selected. Refinement is done by applying moves upon 
           the selected group always from its initial position at the time it was selected until 
           recurrence expires, then the best position is kept.
        #. explore (boolean): Its an engine flag that is used to make a group explore the space around it
           until recurrence expires and a new group is selected. Exploring is done by applying moves upon
           the selected group starting from its initial position and evolving in a trajectory like way 
           until recurrence expires, then the best position is kept.
    
    **NB**: refine and explore flags can't both be set to True at the same time. 
    When this happens refine flag gets automatically switched to False.
    The usage of those flags is very important because they allow groups of atoms to go out
    of local minima in the energy surface. The way RMC works is
    by always minimizing the total energy of the system (error) using gradient descent method.
    The use of those flags allows the system to go up hill in the energy surface searching
    for other lower minimas, while always conserving the lowest energy state found and not
    changing the system structure until a better structure with smaller error is found.

    .. raw:: html
         
        <p>
        The following video compares the Reverse Monte Carlo traditional fitting mode
        with fullrmc's recursive selection one with explore flag set to True. 
        From a potential point of view, exploring allows to cross forbidden unlikely 
        energy barriers and going out of local minimas.
        </p>
        <iframe width= "560" height="315" 
        src="https://www.youtube.com/embed/24Rd2EZ2vVo?rel=0"  
        frameborder="0" allowfullscreen>
        </iframe>  
        
        <p>
        The following video is an example of refining the position of a molecule using
        RecursiveGroupSelector and setting refine flag to True.
        The molecule is always refined from its original position towards a new one generated by the move generator.         
        </p>
        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/tFYHOD00j_o?rel=0" 
        frameborder="0" allowfullscreen>
        </iframe>
        
        <p>
        The following video is an example of exploring the space of a molecule using
        RecursiveGroupSelector and setting explore flag to True.
        The molecule explores the allowed space by wandering via its move generator and only
        moves enhancing the structure are stored.         
        </p>
        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/wafgTfHNpvs?rel=0" 
        frameborder="0" allowfullscreen>
        </iframe>    
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Core.GroupSelector import RecursiveGroupSelector
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        ##### Wrap engine group selector with a recursive group selector. #####
        # create recursive group selector. Recurrence is set to 20 with explore flag set to True.
        RGS = RecursiveGroupSelector(ENGINE.groupSelector, recur=20, refine=False, explore=True)
        ENGINE.set_group_selector(RGS)
    """
    
    def __init__(self, selector, recur=10, override=True, refine=False, explore=False):
        # set selector instance
        assert isinstance(selector, GroupSelector),LOGGER.error("selector must be a fullrmc GroupSelector instance")
        assert not isinstance(selector, RecursiveGroupSelector),LOGGER.error("selector can't not be a RecursiveGroupSelector instance")
        self.__selector = selector
        # initialize selector
        super(RecursiveGroupSelector, self).__init__(engine=self.__selector.engine) 
        # change engine selector to
        if self.__selector.engine is not None:
            self.__selector.engine.set_group_selector(self)
        # set recur value
        self.set_recur(recur)
        # set override
        self.set_override(override)
        # set refine
        self.set_refine(refine)
        # set explore
        self.set_explore(explore)
        # initialize
        self.__initialize_selector__()

    def __initialize_selector__(self):
        # initialize last selected index
        self.__lastSelectedIndex = None
        self.__position = self.__recur
    
    def _runtime_initialize(self):
        """   
        This method is called by the engine at the runtime to initialize the group selector if needed.
        """
        self.__selector._runtime_initialize()
        
    @property
    def selector(self):
        """ Get the wrapped selector instance."""
        return self.__selector
        
    @property
    def lastSelectedIndex(self):
        """ Get the last selected group index."""
        return self.__lastSelectedIndex
    
    @property
    def willSelect(self):
        """ Get whether next step a new selection is occur or still the same group is going to be selected again. """
        return self.__position >= self.__recur
    
    @property
    def willRecur(self):
        """ Get whether next step the same group will be returned. """
        return not self.willSelect
    
    @property
    def willRefine(self):
        """ Get whether next step the same group will be returned and refine flag is True."""
        return self.isRecurring and self.__refine
    
    @property
    def willExplore(self):
        """ Get whether next step the same group will be returned and explore flag is True."""
        return self.isRecurring and self.__explore
        
    @property
    def isNewSelection(self):
        """ Get whether this last step a new selection was made. """
        return self.__position <= 1
        
    @property
    def isRecurring(self):
        """ Get whether this last step the same group was returned. """
        return not self.isNewSelection
    
    @property
    def isRefining(self):
        """ Get whether this last step the same group was returned and refine flag is True."""
        return self.isRecurring and self.__refine
    
    @property
    def isExploring(self):
        """ Get whether this last step the same group was returned and explore flag is True."""
        return self.isRecurring and self.__explore
          
    @property
    def override(self):
        """ Get override flag value. """
        return self.__override
    
    @property
    def refine(self):
        """ Get refine flag value. """
        return self.__refine
    
    @property
    def explore(self):
        """ Get explore flag value. """
        return self.__explore
        
    @property
    def currentRecur(self):
        """ Get the current recur value which is selected group dependant when override flag is True."""
        return self.__recur
    
    @property
    def recur(self):
        """ 
        Get current recur value. 
        The set recur value can change during engine runtime if override flag is True.
        To get the recur value as set by set_recur method recurAsSet must be used.
        """
        return self.__recur
    
    @property
    def recurAsSet(self):
        """ Get recur value as set but set_recur method."""
        return self.__recurAsSet
        
    @property
    def position(self):
        """ Get the position of the selector in the path."""
        return self.__position
        
    @property
    def engine(self):
        """ Get the wrapped selector engine instance."""
        return self.__selector.engine
        
    def set_engine(self, engine):
        """
        Sets the wrapped selector fullrmc engine instance.
        
        :Parameters:
            #. engine (None, fullrmc.Engine): The selector fullrmc engine.
        """
        self.__selector.set_engine(engine)
    
    def set_recur(self, recur):
        """
        Sets the recur value.
        
        :Parameters:
            #. recur (integer): Set the recur value.
               It must be a positive integer.
        """
        assert is_integer(recur), LOGGER.error("recur must be an integer")
        recur = INT_TYPE(recur)
        assert recur>=0, LOGGER.error("recur must be positive")
        self.__recur      = recur
        self.__recurAsSet = recur
        
    def set_override(self, override):
        """
        Select override value.
        
        :Parameters:
            #. override (boolean): Override selector recur value only when selected group move generator is a PathGenerator instance.
               Overridden recur value is temporary and totally selected group dependant.
               If selected group move generator is not a PathGenerator instance, recur value will take back selector's recur value.
        """
        assert isinstance(override, bool), LOGGER.error("override must be a boolean")
        self.__override = override
    
    def set_refine(self, refine):
        """
        Select override value.
        
        :Parameters:
            #. refine (boolean): Its an engine flag that is used to refine the position of a group until 
               recurrence expires and a new group is selected. Refinement is done by applying moves upon 
               the selected group always from its initial position at the time it was selected until 
               recurrence expires, then the best position is kept.
        """
        assert isinstance(refine, bool), LOGGER.error("refine must be a boolean")
        self.__refine = refine
     
    def set_explore(self, explore):
        """
        Select override value.
        
        :Parameters:
            #. explore (boolean): Its an engine flag that is used to make a group explore the space around it
               until recurrence expires and a new group is selected. Exploring is done by applying moves upon
               the selected group starting from its initial position and evolving in a trajectory like way 
               until recurrence expires, then the best position is kept.
        """
        assert isinstance(explore, bool), LOGGER.error("explore must be a boolean")
        self.__explore = explore
        if self.__refine and self.__explore:
            LOGGER.log("argument fixed", "refine and explore flags are not allowed both True. Conflict is resolved by setting refine flag to False")
            self.__refine = False
            
    def select_index(self):
        """
        Select new index.
        
        :Returns:
            #. index (integer): the selected group index in engine groups list
        """
        # select new group
        if self.willSelect:
            self.__lastSelectedIndex = self.__selector.select_index()
            # reset count
            self.__position = 0
            # reset recur
            if self.__override:
                groupGenerator = self.engine.groups[self.__lastSelectedIndex].moveGenerator
                if isinstance(groupGenerator, PathGenerator):
                    self.__recur = len(groupGenerator.path)
            else:
                self.__recur = self.__recurAsSet
        # move position by 1
        self.__position += 1
        # return selection
        return self.__lastSelectedIndex
            

        
        
        
    
    
           
           
            