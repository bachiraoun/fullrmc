"""
OrderedSelectors contains GroupSelector classes of defined order of selection.

.. inheritance-diagram:: fullrmc.Selectors.OrderedSelectors
    :parts: 1
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports=
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_integer, is_number
from fullrmc.Core.GroupSelector import GroupSelector


class DefinedOrderSelector(GroupSelector):
    """
    DefinedOrderSelector is a group selector with a defined order of selection.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. order (None, list, set, tuple, numpy.ndarray): The selector order of groups.
           If None, order is set automatically to all groups indexes list.
    
    .. code-block:: python
        
        # import external libraries
        import numpy as np
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Selectors.OrderedSelectors import DefinedOrderSelector
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups generators as needed ...
        
        ##### set the order of selection from closest to the origin to the further. #####
        # compute groups centers
        centers   = [np.sum(ENGINE.realCoordinates[g.indexes], axis=0)/len(g) for g in ENGINE.groups]
        # compute distances to origin
        distances = [np.sqrt(np.add.reduce(c**2)) for c in centers]
        # compute increasing order
        order     = np.argsort(distances)
        # set group selector
        ENGINE.set_group_selector( DefinedOrderSelector(ENGINE, order = order) )         
    
    """
    def __init__(self, engine, order=None):
        # initialize GroupSelector
        super(DefinedOrderSelector, self).__init__(engine=engine)
        # set order
        self.set_order(order)
        # initialize selector
        self.__initialize_selector__()
           
    def __initialize_selector__(self):
        if self.__order is None:
            self.__index = None
        else:
            self.__index = 0
        
    def _runtime_initialize(self):
        """   
        Automatically sets the selector order at the engine runtime.
        """
        assert self.engine is not None, LOGGER.error("engine must be set prior to calling _runtime_initialize")
        if self.__order is None:
            self.__order = np.array(range(len(self.engine.groups)), dtype=INT_TYPE)
            self.__initialize_selector__()
        
    @property
    def order(self):
        """ List copy of the order of selection."""
        if self.__order is None:
            order = None
        else:
            order = list(self.__order)
        return order
    
    @property
    def index(self):
        """The current selection index."""
        return self.__index
        
    def set_order(self, order):
        """
        Set selector groups order.
        
        :Parameters:
            #. order (None, list, set, tuple, numpy.ndarray): The selector order of groups.
        """
        if order is None:
            newOrder = None
        else:
            assert isinstance(order, (list, set, tuple, np.ndarray)), LOGGER.error("order must a instance among list, set, tuple or numpy.ndarray")
            if isinstance(order, np.ndarray):
                assert len(order.shape)==1, LOGGER.error("order numpy.ndarray must have one dimension")
            order = list(order)
            assert len(order)>0, LOGGER.error("order can't be empty")
            newOrder = []
            for idx in order:
                assert is_integer(idx), LOGGER.error("order indexes must be integers")
                idx = int(idx)
                assert idx>=0, LOGGER.error("order indexes must be positive")
                assert idx<len(self.engine.groups), LOGGER.error("order indexes must be smaller than engine's number of groups")
                newOrder.append(idx)
            newOrder = np.array(newOrder, dtype=INT_TYPE)
        # set order
        self.__order = newOrder
        # re-initialize selector
        self.__initialize_selector__()
        
    def select_index(self):
        """
        Select index.
        
        :Returns:
            #. index (integer): The selected group index in engine groups list.
        """
        # get group index
        groupIndex = self.__order[self.__index]
        # update order index 
        self.__index = (self.__index+1)%len(self.__order)
        # return group index
        return groupIndex 
        
        
class DirectionalOrderSelector(DefinedOrderSelector):        
    """
    DirectionalOrderSelector is a group selector with a defined order of selection.
    The order of selection is computed automatically at engine runtime by computing
    Groups distance to center, and setting the order from the further to the closest 
    if expand argument is True or from the closest to the further if expand is False. 
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The selector RMC engine.
        #. center (None, list, set, tuple, numpy.ndarray): The center of expansion.
           If None, the center is automatically set to the origin (0,0,0).
        #. expand (bool): Whether to set the order from the the further to the closest 
           or from the closest to the further if it is set to False. 
        #. adjustMoveGenerators (bool): If set to True, all groups move generator instances will
           be changed automatically at engine runtime to a MoveGeneratorCollector of 
           TranslationTowardsCenterGenerator and a randomRotation (for only more than 2 atoms groups). 
           Generators parameters can be given by generatorsParams. It is advisable to 
           set this flag to True in order to take advantage of an automatic and intelligent directional moves.
        #. generatorsParams (None, dict): The automatically created moves generators parameters.
           If None is given, default parameters are used. If a dictionary is given, only two keys are allowed.
           'TG' key is for TranslationTowardsCenterGenerator parameters and 'RG' key is
           for RotationGenerator parameters. TranslationTowardsCenterGenerator amplitude parameter
           is not the same for all groups but intelligently allowing certain groups to move more than
           others according to damping parameter.
           
           **Parameters are the following:**\n
           * TG_amp = generatorsParams['TG']['amplitude']: Used for TranslationTowardsCenterGenerator amplitude parameters.
           * TG_ang = generatorsParams['TG']['angle']: Used as TranslationTowardsCenterGenerator angle parameters.
           * TG_dam = generatorsParams['TG']['damping']: Also used for TranslationTowardsCenterGenerator amplitude parameters.
           * RG_ang = generatorsParams['RG']['amplitude']: Used as RotationGenerator angle parameters.
           
           **Parameters are used as the following:**\n
           * TG = TranslationTowardsCenterGenerator(center={"fixed":center}, amplitude=AMPLITUDE, angle=TG_ang)\n
             Where TG_amp < AMPLITUDE < TG_amp.TG_dam
           * RG = RotationGenerator(amplitude=RG_ang)         
           * MoveGeneratorCollector(collection=[TG,RG], randomize=True)
           
           **NB: The parameters are not checked for errors until engine runtime.**           
    
    .. raw:: html

        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/6nsNJrOhLu4?rel=0" 
        frameborder="0" allowfullscreen>
        </iframe>
       
    
    .. code-block:: python
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Selectors.OrderedSelectors import DirectionalOrderSelector
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups generators as needed ...
        
        # Set the order of selection from further to the closest to a (1,1,1).
        # Automatically adjust the groups move generators allowing modulation of amplitudes.
        ENGINE.set_group_selector( DirectionalOrderSelector(ENGINE, 
                                                            center = (1,1,1),
                                                            adjustMoveGenerators=True) )         
        
    """
    def __init__(self, engine, center=None, expand=True,
                       adjustMoveGenerators=False,
                       generatorsParams={"TG":{"amplitude":0.1, "damping":0.1, "angle":90},
                                         "RG":{"amplitude":10}}):
        # initialize GroupSelector
        super(DirectionalOrderSelector, self).__init__(engine=engine, order=None)
        # set center
        self.set_center(center)
        # set expand
        self.set_expand(expand)  
        # set expand
        self.set_adjust_move_generators(adjustMoveGenerators)  
        # set expand
        self.set_generators_parameters(generatorsParams)          
        
    def _runtime_initialize(self):
        """   
        Automatically sets the selector order at the engine runtime.
        """
        diffs = np.array([(np.sum(self.engine.realCoordinates[g.indexes], axis=0)/len(g))-self.__center for g in self.engine.groups], dtype=FLOAT_TYPE)
        dists = np.array([np.sqrt(np.add.reduce(diff**2)) for diff in diffs])
        order = np.argsort(dists).astype(INT_TYPE)
        if self.__expand:
            order = [o for o in reversed(order)]
        # set order
        self.set_order(order)
        # set groups move generators
        if self.__adjustMoveGenerators:
            from fullrmc.Core.MoveGenerator import MoveGeneratorCollector
            from fullrmc.Generators.Rotations import RotationGenerator
            from fullrmc.Generators.Translations import TranslationTowardsCenterGenerator
            TG_amp  = self.__generatorsParams['TG']['amplitude']
            TG_ang  = self.__generatorsParams['TG']['angle']
            TG_dam  = self.__generatorsParams['TG']['damping']
            RG_ang  = self.__generatorsParams['RG']['amplitude']
            maxDist = FLOAT_TYPE(np.max(dists))
            TG_ampInterval = TG_amp-TG_amp*TG_dam
            for idx in range(len(self.engine.groups)):
                g = self.engine.groups[idx]
                damping = ((maxDist-dists[idx])/maxDist)*TG_ampInterval
                coll = [TranslationTowardsCenterGenerator(center={"fixed":self.__center}, amplitude=TG_amp-damping, angle=TG_ang, direction=not self.__expand)]
                if len(g) > 1:
                    coll.append(RotationGenerator(amplitude=RG_ang))
                mg = MoveGeneratorCollector(collection=coll, randomize=True)
                g.set_move_generator( mg )
                                
    @property
    def expand(self):
        """ expand flag."""
        return self.__expand
    
    @property
    def center(self):
        """ center (X,Y,Z) coordinates."""
        return self.__center
    
    @property
    def adjustMoveGenerators(self):
        """ adjustMoveGenerators flag."""
        return self.__adjustMoveGenerators    
    
    @property
    def generatorsParams(self):
        """ Automatic generators parameters."""
        return self.__generatorsParams 
        
    def set_generators_parameters(self, generatorsParams):
        """
        Set move generators parameters.
        
        #. generatorsParams (None, dict): The automatically created moves generators parameters.
           If None is given, default parameters are used. If a dictionary is given, only two keys are allowed.
           'TG' key is for TranslationTowardsCenterGenerator parameters and 'RG' key is
           for RotationGenerator parameters. TranslationTowardsCenterGenerator amplitude parameter
           is not the same for all groups but intelligently allowing certain groups to move more than
           others according to damping parameter.
           
           **Parameters are the following:**\n
           * TG_amp = generatorsParams['TG']['amplitude']: Used for TranslationTowardsCenterGenerator amplitude parameters.
           * TG_ang = generatorsParams['TG']['angle']: Used as TranslationTowardsCenterGenerator angle parameters.
           * TG_dam = generatorsParams['TG']['damping']: Also used for TranslationTowardsCenterGenerator amplitude parameters.
           * RG_ang = generatorsParams['RG']['amplitude']: Used as RotationGenerator angle parameters.
           
           **Parameters are used as the following:**\n
           * TG = TranslationTowardsCenterGenerator(center={"fixed":center}, amplitude=AMPLITUDE, angle=TG_ang)\n
             Where TG_amp < AMPLITUDE < TG_amp.TG_dam
           * RG = RotationGenerator(amplitude=RG_ang)         
           * MoveGeneratorCollector(collection=[TG,RG], randomize=True)
           
           **NB: The parameters are not checked for errors until engine runtime.** 
        """
        if generatorsParams is None:
            generatorsParams = {}
        assert isinstance(generatorsParams, dict), LOGGER.error("generatorsParams must be a python dictionary")
        newGenParams = {"TG":{"amplitude":0.1, "damping":0.1, "angle":90},
                        "RG":{"amplitude":10}}
        # update  TranslationTowardsCenterGenerator values
        for gkey in newGenParams.keys():
            if not generatorsParams.has_key(gkey):
                continue
            assert isinstance(generatorsParams[gkey], dict), LOGGER.error("generatorsParams value must be a python dictionary")
            for key in newGenParams[gkey].keys():
                newGenParams[gkey][key] = generatorsParams[gkey].get(key, newGenParams[gkey][key])
        # check generatorsParams damping parameters
        assert is_number(generatorsParams["TG"]["damping"]), LOGGER.error("generatorsParams['TG']['damping'] must be a number")
        generatorsParams["TG"]["damping"] = FLOAT_TYPE(generatorsParams["TG"]["damping"])
        assert generatorsParams["TG"]["damping"]>=0, LOGGER.error("generatorsParams['TG']['damping'] must be bigger than 0")
        assert generatorsParams["TG"]["damping"]<=1, LOGGER.error("generatorsParams['TG']['damping'] must be smaller than 1")
        # set generatorsParams
        self.__generatorsParams = newGenParams   
        
    def set_center(self, center):
        """
        Set the center.
        
        :Parameters:
            #. center (None, list, set, tuple, numpy.ndarray): The center of expansion.
               If None, the center is automatically set to the origin (0,0,0).
        """
        if center is None:
            center = np.array((0,0,0), dtype=FLOAT_TYPE)
        else:
            assert isinstance(center, (list, set, tuple, np.ndarray)), LOGGER.error("center must a instance among list, set, tuple or numpy.ndarray")
            if isinstance(center, np.ndarray):
                assert len(center.shape)==1,LOGGER.error("center numpy.ndarray must have one dimension")
            center = list(center)
            assert len(center) == 3, LOGGER.error("center must have exactly three items")
            assert is_number(center[0]), LOGGER.error("center items must be numbers")
            assert is_number(center[1]), LOGGER.error("center items must be numbers")
            assert is_number(center[2]), LOGGER.error("center items must be numbers")
            center = np.array(([float(c) for c in center]), dtype=FLOAT_TYPE)
        # set center
        self.__center = center

    def set_expand(self, expand): 
        """
        Set expand.
        
        :Parameters:
            #. expand (bool): Whether to set the order from the the further to the closest 
               or from the closest to the further if it is set to False.   
        """  
        assert isinstance(expand, bool), LOGGER.error("expand must be boolean")
        self.__expand = expand
    
    def set_adjust_move_generators(self, adjustMoveGenerators):
        """
        Set expand.
        
        :Parameters:
            #. adjustMoveGenerators (bool): If set to True, all groups move generator instances will
               be changed automatically at engine runtime to a MoveGeneratorCollector of 
               TranslationTowardsCenterGenerator and a randomRotation (for only more than 2 atoms groups). 
               Generators parameters can be given by generatorsParams. It is advisable to 
               set this flag to True in order to take advantage of an automatic and intelligent directional moves.  
        """  
        assert isinstance(adjustMoveGenerators, bool), LOGGER.error("adjustMoveGenerators must be boolean")
        self.__adjustMoveGenerators = adjustMoveGenerators
        
    