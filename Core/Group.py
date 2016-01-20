"""
Group contains parent classes for all groups.
A Group is a set of atoms indexes used to gather atoms and apply actions  such as moves upon them. 
Therefore it has become possible to fully customize and separate atoms to groups and perform reverse monte carlo actions
on groups rather than on single atoms.

.. inheritance-diagram:: fullrmc.Core.Group
    :parts: 1
"""

# standard libraries imports
import inspect

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.MoveGenerator import MoveGenerator
from fullrmc.Generators.Translations import TranslationGenerator

class Group(object):
    """
    A Group is a set of atoms indexes container. 
    
    :Parameters:
        #. indexes (np.ndarray, list, set, tuple): list of atoms indexes.
        #. moveGenerator (None, MoveGenerator): Move generator instance.
           If None is given TranslationGenerator is considered by default.
        #. refine (bool): The refinement flag used by the Engine.
    
    .. code-block:: python
        
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Core.Group import Group
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        
        # re-define groups as atoms.
        groups = [Group([idx]) for idx in ENGINE.pdb.indexes]
        ENGINE.set_groups( groups )
        
        # Re-define groups generators as needed ... By default TranslationGenerator is used.
        
    """
    def __init__(self, indexes, moveGenerator=None, refine=False):
        self.__moveGenerator = TranslationGenerator(group=self)
        self.set_indexes(indexes)
        self.set_move_generator(moveGenerator)
        # set refine
        self.set_refine(refine)
        # initialize engine
        self.__engine = None
    
    def __len__(self):
        return len(self.__indexes)
        
    def _set_engine(self, engine):
        """ This is a fullrmc API method to set the group engine. Use with caution and better not to."""
        self.__engine = engine
    
    def _get_engine(self):
        """ This is a fullrmc API method to get the group engine. Normally a group doesn't need to know the engine."""
        return self.__engine
        
    @property
    def indexes(self):
        """ Get the indexes array."""
        return self.__indexes
    
    @property
    def moveGenerator(self):
        """ Get the move generator instance."""
        return self.__moveGenerator
        
    @property
    def refine(self):
        """ Get refine flag."""
        return self.__refine
    
    def set_refine(self, refine):
        """
        Sets the selector refine flag.
        
        :Parameters:
            #. refine (bool): The selector refinement flag.
        """
        assert isinstance(refine, bool), LOGGER.error("refine must be a boolean")
        self.__refine = refine
        
    def set_indexes(self, indexes):
        """
        Sets the group indexes. Indexes redundancy is not checked and indexes order is preserved. 
        
        :Parameters:
            #. indexes (list,set,tuple,np.ndarray): The group atoms indexes.
        """
        assert isinstance(indexes, (list,set,tuple,np.ndarray)), LOGGER.error("indexes must be either a list or a numpy.array")
        if isinstance(indexes, np.ndarray):
            assert len(indexes.shape) == 1, LOGGER.error("each group must be a numpy.ndarray of dimension 1")
        # create group of indexes
        group = []
        for idx in list(indexes):
            assert is_integer(idx), LOGGER.error("group indexes must be integers")
            assert idx>=0, LOGGER.error("group index must equal or bigger than 0")
            group.append(int(idx))
        # create indexes
        self.__indexes = np.array(group, dtype=INT_TYPE)
    
    def set_move_generator(self, generator):
        """
        Set group move generator.
        
        :Parameters:
            #. generator (None, MoveGenerator): Move generator instance.
               If None is given TranslationGenerator is considered by default.
        """
        if generator is None:
            generator = TranslationGenerator(group=self)
        else:
            assert isinstance(generator, MoveGenerator), LOGGER.error("generator must be a MoveGenerator instance")
            try:
                generator.set_group(self)
            except Exception as e:
                raise Exception( LOGGER.error(e) )
        # set moveGenerator
        self.__moveGenerator = generator
    
   


    
    
            