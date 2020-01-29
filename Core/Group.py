"""
Group contains parent classes for all groups.
A Group is a set of atoms indexes used to gather atoms and apply actions
such as moves upon them. Therefore it has become possible to fully
customize and separate atoms to groups and perform stochastic actions
on groups rather than on single atoms.

.. inheritance-diagram:: fullrmc.Core.Group
    :parts: 1
"""
# standard libraries imports
from __future__ import print_function
import inspect, re

# external libraries imports
import numpy as np

# fullrmc imports
from ..Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from ..Globals import str, long, unicode, bytes, basestring, range, xrange, maxint
from ..Core.Collection import is_number, is_integer, get_path
from ..Core.MoveGenerator import MoveGenerator, RemoveGenerator
from ..Generators.Translations import TranslationGenerator
from ..Generators.Removes import AtomsRemoveGenerator

class Group(object):
    """
    A Group is a set of atoms indexes container.

    :Parameters:
        #. indexes (np.ndarray, list, set, tuple): list of atoms indexes.
        #. moveGenerator (None, MoveGenerator): Move generator instance.
           If None is given AtomsRemoveGenerator is considered by default.
        #. refine (bool): The refinement flag used by the Engine.
        #. name (str): The group user defined name.

    .. code-block:: python

        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Core.Group import Group

        # create engine
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb('system.pdb')

        # Add constraints ...

        # re-define groups as atoms.
        groups = [Group([idx]) for idx in ENGINE.pdb.indexes]
        ENGINE.set_groups( groups )

        # Re-define groups generators as needed ... By default AtomsRemoveGenerator is used.

    """
    def __init__(self, indexes, moveGenerator=None, refine=False, name=''):
        self.set_indexes(indexes)
        self.set_move_generator(moveGenerator)
        # set refine
        self.set_refine(refine)
        # set name
        self.set_name(name)
        # initialize engine
        self.__engine = None

    def _codify__(self, name='group', addDependencies=True, codifyGenerator=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        dependencies  = ['from fullrmc.Core import Group']
        code          = []
        moveGenerator = None
        if codifyGenerator:
            genDep, genCode = self.__moveGenerator._codify__(name='groupGenerator', group=name, addDependencies=addDependencies)
            dependencies.extend(genDep)
            code.append(genCode)
            moveGenerator = 'groupGenerator'
        if addDependencies:
            code = dependencies + code
        code.append("{name} = Group.Group\
(indexes={indexes}, moveGenerator={moveGenerator}, refine={refine}, name='{n}')".format(name=name,
        moveGenerator=moveGenerator, indexes=list(self.__indexes),
        refine=self.__refine, n=self.__name))
        # return
        return dependencies, '\n'.join(code)


    def __len__(self):
        return len(self.__indexes)

    def _set_engine(self, engine):
        """ This is a fullrmc API method to set the group engine.
        Use with caution and better not to."""
        self.__engine = engine

    def _get_engine(self):
        """ This is a fullrmc API method to get the group engine.
        Normally a group doesn't need to know the engine."""
        return self.__engine

    @property
    def indexes(self):
        """ Atoms index array."""
        return self.__indexes

    @property
    def moveGenerator(self):
        """ Group's move generator instance."""
        return self.__moveGenerator

    @property
    def refine(self):
        """ Refine flag."""
        return self.__refine

    @property
    def name(self):
        """ groud user defined name."""
        return self.__name

    def set_refine(self, refine):
        """
        Set the selector refine flag.

        :Parameters:
            #. refine (bool): The selector refinement flag.
        """
        assert isinstance(refine, bool), LOGGER.error("refine must be a boolean")
        self.__refine = refine

    def set_name(self, name):
        """
        Set the group's name.

        :Parameters:
            #. name (str): The group user defined name.
        """
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        self.__name = name

    def set_indexes(self, indexes):
        """
        Set group atoms index. Indexes redundancy is not checked
        and indexes order is preserved.

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
        # check length
        assert len(group) or isinstance(self, EmptyGroup), LOGGER.error("Group is found to be empty. Use EmptyGroup instead.")
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
            assert not isinstance(generator, RemoveGenerator), LOGGER.error("generator must not be a RemoveGenerator instance")
            try:
                generator.set_group(self)
            except Exception as e:
                raise Exception( LOGGER.error(e) )
        # set moveGenerator
        self.__moveGenerator = generator


class EmptyGroup(Group):
    """
    Empty group is a special group that takes no atoms indexes. It's mainly
    used to remove atoms from system upon fitting.

    .. code-block:: python

        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Core.Group import EmptyGroup

        # create engine
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb('system.pdb')

        # Add constraints ...

        # re-define groups and set a single empty group
        ENGINE.set_groups( EmptyGroup() )

        # Re-define groups generators as needed ... By default RemoveGenerator is used.

    """
    def __init__(self, *args, **kwargs):
        super(EmptyGroup, self).__init__(indexes=None, *args, **kwargs)

    def _codify__(self, name='group', group=None, addDependencies=True, codifyGenerator=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        dependencies  = ['from fullrmc.Core import Group']
        code          = []
        moveGenerator = None
        if codifyGenerator:
            genDep, genCode = self.__moveGenerator._codify__(name='groupGenerator',group=name, addDependencies=addDependencies)
            dependencies.extend(genDep)
            code.append(genCode)
            moveGenerator = 'groupGenerator'
        if addDependencies:
            code = dependencies + code
        code.append("{name} = Group.EmptyGroup(moveGenerator={moveGenerator}, \
refine={refine})".format(name=name, moveGenerator=moveGenerator, refine=self.refine))
        # return
        return dependencies, '\n'.join(code)

    @property
    def moveGenerator(self):
        """ Group's move generator instance."""
        return self.__moveGenerator

    @property
    def indexes(self):
        """ Always returns None for EmptyGroup"""
        return self.__indexes

    def set_move_generator(self, generator):
        """
        Set group move generator.

        :Parameters:
            #. generator (None, MoveGenerator): Move generator instance.
               If None is given TranslationGenerator is considered by default.
        """
        if generator is None:
            generator = AtomsRemoveGenerator(group=self)
        else:
            assert isinstance(generator, RemoveGenerator), LOGGER.error("EmptyGroup generator must be a RemoveGenerator instance")
        self.__moveGenerator = generator

    def set_indexes(self, indexes):
        """
        Sets the group indexes. For an EmptyGroup, this method will disregard
        given indexes argument and will always set indexes property to None.

        :Parameters:
            #. indexes (object): The group atoms indexes. This argument will always be
               disregarded in this particular case.
        """
        self.__indexes = None
