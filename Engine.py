"""
Engine is fullrmc's main module. It contains 'Engine' the main class
of fullrmc which is the stochastic artist. The engine class
takes only Protein Data Bank formatted files
`'.pdb' <http://deposit.rcsb.org/adit/docs/pdb_atom_format.html>`_ as
atomic/molecular input structure. It handles and fits simultaneously many
experimental data while controlling the evolution of the system using
user-defined molecular and atomistic constraints such as bond-length,
bond-angles, dihedral angles, inter-molecular-distances, etc.
"""

# standard libraries imports
from __future__ import print_function
import os
import time
import sys
import uuid
import tempfile
import multiprocessing
import copy
import inspect


# external libraries imports
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from pdbparser.pdbparser import pdbparser
from pdbparser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries
from pyrep import Repository

# fullrmc library imports
from .__pkginfo__ import __version__
from .Globals import INT_TYPE, FLOAT_TYPE, LOGGER, WATER_NUMBER_DENSITY
from .Globals import str, long, unicode, bytes, basestring, range, xrange, maxint
from .Core.boundary_conditions_collection import transform_coordinates
from .Core.Collection import Broadcaster, is_number, is_integer, get_elapsed_time, generate_random_float
from .Core.Collection import _AtomsCollector, _Container, get_caller_frames
from .Core.Constraint import Constraint, SingularConstraint, RigidConstraint, ExperimentalConstraint
from .Core.Group import Group, EmptyGroup
from .Core.MoveGenerator import SwapGenerator, RemoveGenerator
from .Core.GroupSelector import GroupSelector
from .Selectors.RandomSelectors import RandomSelector


class Engine(object):
    """
    fulrmc's engine, is used to launch a stochastic modelling which is
    different than traditional Reverse Monte Carlo (RMC).
    It has the capability to use and fit simultaneously multiple sets of
    experimental data. One can also define constraints such as distances,
    bonds length, angles and many others.

    :Parameters:
        #. path (None, string): Engine repository (directory) path to save the
           engine. If None is given path will be set when saving the engine
           using Engine.save method. If a non-empty directory is found at the
           given path an error will be raised unless freshStart flag attribute
           is set to True.
        #. logFile (None, string): Logging file basename. A logging file full
           name will be the given logFile appended '.log' extension
           automatically. If None is given, logFile is left unchanged.
        #. freshStart (boolean): Whether to remove any existing fullrmc engine
           at the given path if found. If set to False, an error will be raise
           if a fullrmc engine or a non-empty directory is found at the given
           path.
        #. timeout (number): The maximum delay or time allowed to successfully
           set the lock upon reading or writing the engine repository

    .. code-block:: python

        # import engine
        from fullrmc.Engine import Engine

        # create engine
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb(pdbFileName)

        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        # Re-define moves generators if needed ...

        # save engine
        ENGINE.save()

        # run engine for 10000 steps and save only at the end
        ENGINE.run(numberOfSteps=10000, saveFrequency=10000)

    """
    def __init__(self, path=None, logFile=None, freshStart=False, timeout=10):
        # set repository and frame data
        ENGINE_DATA   = ('_Engine__frames', '_Engine__usedFrame', )
        ## MUST ADD ORIGINAL DATA TO STORE ALL ORIGINAL PDB ATTRIBUTES
        ## THIS IS NEEDED TO SET GROUPS AND ETC ESPECIALLY AFTER REMOVING
        ## ATOMS FROM SYSTEM.
        self.__frameOriginalData = {}
        FRAME_DATA      = ('_Engine__pdb', '_Engine__tolerance', ### MIGHT NEED TO MOVE _Engine__pdb TO ENGINE_DATA
                           '_Engine__boundaryConditions', '_Engine__isPBC', '_Engine__isIBC',
                           '_Engine__basisVectors', '_Engine__reciprocalBasisVectors',
                           '_Engine__numberDensity', '_Engine__volume',
                           '_Engine__realCoordinates','_Engine__boxCoordinates',
                           '_Engine__groups', '_Engine__groupSelector', '_Engine__state',
                           '_Engine__generated', '_Engine__tried', '_Engine__accepted',
                           '_Engine__removed', '_Engine__tolerated', '_Engine__totalStandardError',
                           '_Engine__lastSelectedGroupIndex', '_Engine__numberOfMolecules',
                           '_Engine__moleculesIndex', '_Engine__moleculesName',
                           '_Engine__allElements', '_Engine__elements',
                           '_Engine__elementsIndex', '_Engine__numberOfAtomsPerElement',
                           '_Engine__allNames', '_Engine__names',
                           '_Engine__namesIndex', '_Engine__numberOfAtomsPerName',
                           '_atomsCollector',)
                           #'_atomsCollector', '_Container')
        MULTIFRAME_DATA = ('_Engine__constraints',)#'_Engine__broadcaster') # multiframe data will be save appart with ENGINE_DATA and pulled for traditional frames
        RUNTIME_DATA    = ('_Engine__realCoordinates','_Engine__boxCoordinates',
                           '_Engine__state', '_Engine__generated', '_Engine__tried',
                           '_Engine__accepted','_Engine__tolerated', '_Engine__removed',
                           '_Engine__totalStandardError', '_Engine__lastSelectedGroupIndex',
                           '_atomsCollector',  # RUNTIME_DATA must have all atomsCollector data keys and affected attributes upon amputating atoms
                           '_Engine__moleculesIndex', '_Engine__moleculesName',
                           '_Engine__elementsIndex', '_Engine__allElements',
                           '_Engine__namesIndex', '_Engine__allNames',
                           '_Engine__numberOfAtomsPerName',
                           '_Engine__numberOfAtomsPerElement',
                           '_Engine__names','_Engine__elements',
                           '_Engine__numberOfMolecules','_Engine__numberDensity',)

        # might need to add groups to FRAME_DATA
        object.__setattr__(self, 'ENGINE_DATA',     tuple( ENGINE_DATA)     )
        object.__setattr__(self, 'FRAME_DATA',      tuple( FRAME_DATA)      )
        object.__setattr__(self, 'MULTIFRAME_DATA', tuple( MULTIFRAME_DATA) )
        object.__setattr__(self, 'RUNTIME_DATA',    tuple( RUNTIME_DATA)    )

        # initialize engine' info
        self.__frames    = {'0':None}
        self.__usedFrame = '0'
        self.__id        = str(uuid.uuid1())
        self.__version   = __version__

        # set timeout
        self.set_timeout(timeout)

        # check whether an engine exists at this path
        if self.is_engine(path):
            if freshStart:
                #Repository().remove_repository(path, relatedFiles=True, relatedFolders=True)
                rep = Repository(timeout=self.__timeout)
                rep.DEBUG_PRINT_FAILED_TRIALS = False
                rep.remove_repository(path, removeEmptyDirs=True)
            else:
                m  = "An Engine is found at '%s'. "%path
                m += "If you wish to override it set freshStart argument to True. "
                m += "If you wish to load it set path to None and use Engine.load method instead. "
                raise Exception( LOGGER.error(m) )

        # initialize path and repository
        if path is not None:
            result, message = self.__check_path_to_create_repository(path)
            assert result, LOGGER.error(message)
        self.__path       = path
        self.__repository = None

        # initialize atoms collector
        dataKeys = ('realCoordinates', 'boxCoordinates',
                    'moleculesIndex',  'moleculesName',
                    'elementsIndex',   'allElements',
                    'namesIndex',      'allNames')
        self._atomsCollector = _AtomsCollector(self, dataKeys=dataKeys)

        ## initialize objects container
        #self._container = _Container()

        # initialize engine attributes
        self.__broadcaster   = Broadcaster()
        self.__constraints   = []
        self.__state         = time.time()
        self.__groups        = []
        self.__groupSelector = None
        self.__tolerance     = 0.

        # set mustSave flag, it indicates  whether saving whole engine is needed before running
        self.__mustSave       = False
        self.__saveGroupsFlag = True

        # set pdb
        self.set_pdb(pdb=None)

        # create runtime variables and arguments
        self._runtime_ncores = INT_TYPE(1)

        # set LOGGER file path
        if logFile is not None:
            self.set_log_file(logFile)

    def __repr__(self):
        repr = "fullrmc %s (Version %s)"%(self.__class__.__name__, self.__version)
        if self.__repository is None:
            return repr
        repoStats = self.__repository.get_stats()[:2]
        ndirs    = repoStats[0]
        nfiles   = repoStats[1]
        nframes  = sum([1 for f in self.__frames if self.__frames[f] is None])
        nmframes = sum([1 for f in self.__frames if self.__frames[f] is not None])
        repr = "%s @%s [%i directories] [%i files] (%i frames) (%i multiframes)"%(repr,self.__repository.path, ndirs, nfiles, nframes, nmframes)
        return repr

    def __str__(self):
        return self.__repr__()

    def __setattr__(self, name, value):
        if name in ('ENGINE_DATA', 'FRAME_DATA', 'RUNTIME_DATA', 'MULTIFRAME_DATA'):
            raise Exception(LOGGER.error("Setting '%s' is not allowed."%name))
        else:
            object.__setattr__(self, name, value)

    def __getstate__(self):
        state = {}
        for k in self.__dict__:
            if k in self.ENGINE_DATA:
                continue
            elif k in self.FRAME_DATA:
                continue
            elif k in self.MULTIFRAME_DATA:
                continue
            state[k] = self.__dict__[k]
        # no need to pickle repository. This might cause locker problems. It
        # will be instanciated upon loading
        state['_Engine.__repository'] = None
        # return state
        return state

    #def __setstate__(self, d):
    #    self.__dict__ = d

    def __check_get_frame_name(self, name):
        assert isinstance(name, basestring), "Frame name must be a string"
        name = str(name)
        assert name.replace('_','').replace('-','').replace(' ','').isalnum(), LOGGER.error("Frame name must be strictly alphanumeric with the exception of '-' and '_'")
        return name

    def __check_frames(self, frames, raiseExisting=True):
        if not isinstance(frames, (list,set,tuple)):
            frames = [frames]
        else:
            frames = list(frames)
        assert len(frames), LOGGER.error("frames must be a non-empty list.")
        for idx, frm in enumerate(frames):
            if isinstance(frm, basestring):
                frameName = self.__check_get_frame_name(frm)
            elif isinstance(frm, dict):
                #assert self.__repository is not None, LOGGER.error("Creating multiframe is not allowed before initializing repository. Save engine then proceed.")
                assert 'name' in frm, "multiframe dictionary must contain 'name'"
                frameName = self.__check_get_frame_name(frm['name'])
                assert 'frames_name' in frm, "multiframe dictionary must contain 'frames_name'"
                multiFramesName = frm['frames_name']
                if not isinstance(multiFramesName, (list,set,tuple)):
                    assert isinstance(multiFramesName, int), LOGGER.error("multiframe dictionary 'frames_name' value must be a list of names of an integer indicating number of frames")
                    assert multiFramesName>=1, LOGGER.error("multiframe dictionary integer 'frames_name' value must be >=1")
                    multiFramesName = [str(i) for i in range(multiFramesName)]
                assert len(multiFramesName)>=1, LOGGER.error("multiframe dictionary 'frames_name' list number of items must be >=1")
                multiFramesName = [self.__check_get_frame_name(str(i)) for i in multiFramesName]
                assert len(multiFramesName) == len(set(multiFramesName)),  LOGGER.error("Multiframe dictionary 'frames_name' list redundancy is not allowed")
                #if 'type' not in frm:
                #    frm['type'] = 'statistical'
                #assert isinstance(frm['type'], basestring), LOGGER.error("Multiframe dictionary 'type' value must be a string")
                #assert frm['type'] in ('statistical',), LOGGER.error("known multiframe types are 'statistical'")
                frm = copy.deepcopy(frm)
                frm['frames_name'] = tuple(multiFramesName)
                frm['name']        = str(frameName)
                #frm['type']        = str(frm['type'])
            else:
                assert isinstance(frm, int), LOGGER.error('Each frame must be either interger a string or a dict')
                frameName = self.__check_get_frame_name(str(frm))
            if raiseExisting:
                assert frameName not in self.__frames, LOGGER.error("frame name '%s' exists already."%frameName)
            frames[idx] = frm
        # check for redundancy
        assert len(frames) == len(set([i if isinstance(i, basestring) else i['name'] for i in frames])), LOGGER.error("Redundancy is not allowed in frame names.")
        # all is good
        return frames

    def __check_path_to_create_repository(self, path):
        # check for string
        if not isinstance(path, basestring):
             return False, "Repository path must be a string. '%s' is given"%path
        # test if directory is empty
        if os.path.isdir(path):
            if len(os.listdir(path)):
                return False, "Repository path directory at '%s' is not empty"%path
        # all is good unless directory is not writable.
        return True, ""

    def __set_runtime_ncores(self, ncores):
        if ncores is None:
            ncores = INT_TYPE(1)
        else:
            assert is_integer(ncores), LOGGER.error("ncores must be an integer")
            ncores = INT_TYPE(ncores)
            assert ncores>0, LOGGER.error("ncores must be > 0")
            if ncores > multiprocessing.cpu_count():
                LOGGER.warn("ncores '%s' is reset to %s which is the number of available cores on your machine"%(ncores, multiprocessing.cpu_count()))
                ncores = INT_TYPE(multiprocessing.cpu_count())
        self._runtime_ncores = ncores

    def _reinit_engine(self):
        """ Initialize all engine arguments and flags. """
        # engine state
        self.__state = time.time()
        # engine last moved group index
        self.__lastSelectedGroupIndex = None
        # engine generated steps number
        self.__generated = 0
        # engine tried steps number
        self.__tried = 0
        # engine accepted steps number
        self.__accepted = 0
        # engine tolerated steps number
        self.__tolerated = 0
        # engine removed atoms
        self.__removed = [0.,0.,0.]
        # current model totalStandardError from constraints
        self.__totalStandardError = None
        # set groups as atoms
        self.set_groups(None)
        # set group selector as random
        self.set_group_selector(None)
        # update constraints in repository
        if self.__repository is not None:
            #from pprint import pprint;pprint(self.__getstate__())
            self.__repository.update_file(value=self, relativePath='engine')
            self.__repository.update_file(value=self.__state, relativePath=os.path.join(self.__usedFrame, '_Engine__state'))
            self.__repository.update_file(value=self.__lastSelectedGroupIndex, relativePath=os.path.join(self.__usedFrame, '_Engine__lastSelectedGroupIndex'))
            self.__repository.update_file(value=self.__generated, relativePath=os.path.join(self.__usedFrame, '_Engine__generated'))
            self.__repository.update_file(value=self.__removed, relativePath=os.path.join(self.__usedFrame, '_Engine__removed'))
            self.__repository.update_file(value=self.__tried, relativePath=os.path.join(self.__usedFrame, '_Engine__tried'))
            self.__repository.update_file(value=self.__accepted, relativePath=os.path.join(self.__usedFrame, '_Engine__accepted'))
            self.__repository.update_file(value=self.__tolerated, relativePath=os.path.join(self.__usedFrame, '_Engine__tolerated'))
            self.__repository.update_file(value=self.__totalStandardError, relativePath=os.path.join(self.__usedFrame, '_Engine__totalStandardError'))

    def _dump_to_repository(self, value, relativePath, repository=None):
        if repository is None:
            repository = self.__repository
        if repository is None:
            return
        isRepoFile, fileOnDisk, infoOnDisk, classOnDisk = repository.is_repository_file(relativePath)
        if isRepoFile and fileOnDisk and infoOnDisk and classOnDisk:
            repository.update(value=value, relativePath=relativePath)
        else:
            repository.dump(value=value, relativePath=relativePath, replace=True)

    def _set_path(self, path):
        self.__path = path

    def _set_repository(self, repo):
        self.__repository = repo

    def _get_repository(self):
        return self.__repository

    def _get_broadcaster(self):
        return self.__broadcaster

    def _on_collector_reset(self):
        pass

    def _on_collector_collect_atom(self, realIndex):
        assert not self._atomsCollector.is_collected(realIndex), LOGGER.error("Trying to collect atom index %i which is already collected."%realIndex)
        relativeIndex = self._atomsCollector.get_relative_index(realIndex)
        # create dataDict and remove
        dataDict = {}
        dataDict['realCoordinates'] = self.__realCoordinates[relativeIndex,:]
        dataDict['boxCoordinates']  = self.__boxCoordinates[relativeIndex, :]
        dataDict['moleculesIndex']  = self.__moleculesIndex[relativeIndex]
        dataDict['moleculesName']   = self.__moleculesName[relativeIndex]
        dataDict['elementsIndex']   = self.__elementsIndex[relativeIndex]
        dataDict['allElements']     = self.__allElements[relativeIndex]
        dataDict['namesIndex']      = self.__namesIndex[relativeIndex]
        dataDict['allNames']        = self.__allNames[relativeIndex]
        assert self.__numberOfAtomsPerElement[dataDict['allElements']]-1>0, LOGGER.error("Collecting last atom of any element type is not allowed. It's better to restart your simulation without any '%s' rather than removing them all!"%dataDict['allElements'])
        # collect atom
        self._atomsCollector.collect(index=realIndex, dataDict=dataDict)
        # collect all constraints BEFORE removing data from engine.
        for c in self.__constraints:
            c._on_collector_collect_atom(realIndex=realIndex)
        # remove data from engine AFTER collecting constraints data.
        self.__realCoordinates = np.delete(self.__realCoordinates, relativeIndex, axis=0)
        self.__boxCoordinates  = np.delete(self.__boxCoordinates,  relativeIndex, axis=0)
        self.__moleculesIndex  = np.delete(self.__moleculesIndex,relativeIndex, axis=0)
        self.__moleculesName.pop(relativeIndex)
        self.__elementsIndex   = np.delete(self.__elementsIndex, relativeIndex, axis=0)
        self.__allElements.pop(relativeIndex)
        self.__namesIndex      = np.delete(self.__namesIndex,    relativeIndex, axis=0)
        self.__allNames.pop(relativeIndex)
        # adjust other attributes
        self.__numberOfAtomsPerName[dataDict['allNames']]       -= 1
        self.__numberOfAtomsPerElement[dataDict['allElements']] -= 1
        #self.__elements = sorted(set(self.__allElements)) # no element should disappear
        self.__names             = sorted(set(self.__names))
        self.__numberOfMolecules = len(set(self.__moleculesIndex))
        # update number density in periodic boundary conditions only
        if self.__isPBC:
            self.__numberDensity = FLOAT_TYPE(self.numberOfAtoms) / FLOAT_TYPE(self.__volume)

    def _on_collector_release_atom(self, realIndex):
        # get relative index
        relativeIndex = self._atomsCollector.get_relative_index(realIndex)
        # get dataDict
        dataDict = self._atomsCollector.release(realIndex)
        # release all constraints
        for c in self.__constraints:
            c._on_collector_release_atom(realIndex=realIndex)
        # re-insert data
        self.__realCoordinates = np.insert(self.__realCoordinates,  relativeIndex, dataDict["realCoordinates"], axis=0)
        self.__boxCoordinates  = np.insert(self.__boxCoordinates,   relativeIndex, dataDict["boxCoordinates"],  axis=0)
        self.__moleculesIndex  = np.insert(self.__moleculesIndex, relativeIndex, dataDict["moleculesIndex"],axis=0)
        self.__moleculesName.insert(relativeIndex, dataDict["moleculesName"])
        self.__elementsIndex   = np.insert(self.__elementsIndex,  relativeIndex, dataDict["elementsIndex"], axis=0)
        self.__allElements.insert(relativeIndex, dataDict["allElements"])
        self.__namesIndex      = np.insert(self.__namesIndex,     relativeIndex, dataDict["namesIndex"],    axis=0)
        self.__allNames.insert(relativeIndex, dataDict["allNames"])
        # adjust other attributes
        self.__numberOfAtomsPerName[dataDict['allNames']]       += 1
        self.__numberOfAtomsPerElement[dataDict['allElements']] += 1
        self.__elements = list(set(self.__allElements))
        self.__names    = sorted(set(self.__names))
        self.__numberOfMolecules = len(set(self.__moleculesIndex))
        # update number density in periodic boundary conditions only
        if self.__isPBC:
            self.__numberDensity = FLOAT_TYPE(self.numberOfAtoms) / FLOAT_TYPE(self.__volume)

    @property
    def path(self):
        """ Engine's repository path if set or save."""
        return self.__path

    @property
    def info(self):
        """ Engine's information (version, id) tuple."""
        return (self.__version, self.__id)

    @property
    def frames(self):
        """ Engine's frames list copy."""
        return copy.deepcopy(self.__frames)

    @property
    def usedFrame(self):
        """ Engine's frame in use."""
        return copy.deepcopy(self.__usedFrame)

    @property
    def lastSelectedGroupIndex(self):
        """ The last moved group instance index in groups list. """
        return self.__lastSelectedGroupIndex

    @property
    def lastSelectedGroup(self):
        """ The last moved group instance. """
        if self.__lastSelectedGroupIndex is None:
            return None
        return self.__groups[self.__lastSelectedGroupIndex]

    @property
    def lastSelectedAtomsIndex(self):
        """ The last moved atoms index. """
        if self.__lastSelectedGroupIndex is None:
            return None
        return self.lastSelectedGroup.indexes

    @property
    def state(self):
        """ Engine's state. """
        return self.__state

    @property
    def generated(self):
        """ Number of generated moves. """
        return self.__generated

    @property
    def removed(self):
        """ Removed atoms tuple (tried, accepted, ratio)"""
        return tuple(self.__removed)

    @property
    def tried(self):
        """ Number of tried moves. """
        return self.__tried

    @property
    def accepted(self):
        """ Number of accepted moves. """
        return self.__accepted

    @property
    def tolerated(self):
        """ Number of tolerated steps in spite of increasing
        totalStandardError"""
        return self.__tolerated

    @property
    def tolerance(self):
        """ Tolerance in percent. """
        return self.__tolerance*100.

    @property
    def groups(self):
        """ Engine's defined groups list. """
        return self.__groups

    @property
    def pdb(self):
        """ Engine's pdbparser instance. """
        return self.__pdb

    @property
    def boundaryConditions(self):
        """ Engine's boundaryConditions instance. """
        return self.__boundaryConditions

    @property
    def isPBC(self):
        """ Whether boundaryConditions are periodic. """
        return self.__isPBC

    @property
    def isIBC(self):
        """ Whether boundaryConditions are infinte. """
        return self.__isIBC

    @property
    def basisVectors(self):
        """ The boundary conditions basis vectors in case of
        PeriodicBoundaries, None in case of InfiniteBoundaries. """
        return self.__basisVectors

    @property
    def reciprocalBasisVectors(self):
        """ The boundary conditions reciprocal basis vectors in case of
        PeriodicBoundaries, None in case of InfiniteBoundaries. """
        return self.__reciprocalBasisVectors

    @property
    def volume(self):
        """ The boundary conditions basis volume in case of PeriodicBoundaries,
        None in case of InfiniteBoundaries. """
        return self.__volume

    @property
    def realCoordinates(self):
        """ The real coordinates of the current configuration. """
        return self.__realCoordinates

    @property
    def boxCoordinates(self):
        """ The box coordinates of the current configuration in case of
        PeriodicBoundaries. Similar to realCoordinates in case of
        InfiniteBoundaries."""
        return self.__boxCoordinates

    @property
    def numberOfMolecules(self):
        """ Number of molecules."""
        return self.__numberOfMolecules

    @property
    def moleculesIndex(self):
        """ Atoms molecule index list. """
        return self.__moleculesIndex

    @property
    def moleculesName(self):
        """ Atoms molecule name list. """
        return self.__moleculesName

    @property
    def elementsIndex(self):
        """ Atoms element index list indexing elements sorted set. """
        return self.__elementsIndex

    @property
    def elements(self):
        """ Sorted set of all existing atom elements. """
        return self.__elements

    @property
    def allElements(self):
        """ Atoms element list. """
        return self.__allElements

    @property
    def namesIndex(self):
        """ Atoms name index list indexing names sorted set"""
        return self.__namesIndex

    @property
    def names(self):
        """ Sorted set of all existing atom names. """
        return self.__names

    @property
    def allNames(self):
        """ Atoms name list. """
        return self.__allNames

    @property
    def numberOfNames(self):
        """ Length of atoms name set. """
        return len(self.__names)

    @property
    def numberOfAtoms(self):
        """ Number of atoms in the pdb."""
        return self.__realCoordinates.shape[0]

    @property
    def numberOfAtomsPerName(self):
        """ Number of atoms per name dictionary. """
        return self.__numberOfAtomsPerName

    @property
    def numberOfElements(self):
        """ Number of different elements in the pdb. """
        return len(self.__elements)

    @property
    def numberOfAtomsPerElement(self):
        """ Number of atoms per element dictionary. """
        return self.__numberOfAtomsPerElement

    @property
    def numberDensity(self):
        """
        System's number density computed as :math:`\\rho_{0}=\\frac{N}{V}`
        where N is the total number of atoms and V the volume of the system.
        """
        return self.__numberDensity

    @property
    def constraints(self):
        """ Copy list of all constraints instances. """
        return [c for c in self.__constraints]

    @property
    def groupSelector(self):
        """ Engine's group selector instance. """
        return self.__groupSelector

    @property
    def totalStandardError(self):
        """ Engine's last recorded totalStandardError of the current
        configuration. """
        return self.__totalStandardError

    def timeout(self):
        """Timeout to successfully acquire the lock upon reading or writing to
        the repository"""
        return self.__timeout

    def set_timeout(self, timeout):
        """Set repository access timeout

        :Parameters:
           #. timeout (number): The maximum delay or time allowed to successfully
              set the lock upon reading or writing the engine repository
        """
        assert isinstance(timeout, (float,int)),"timeout must be a number"
        assert timeout>1, "timeout is not allowed below 1 second"
        self.__timeout = timeout


    def get_original_data(self, name):
        """
        Get original data as initialized and parsed from pdb.

        :Parameters:
            #. name (string): Data name.

        :Returns:
            #. value (object): Data value
        """
        dname = "_original__"+name
        if self.__repository is None:
            assert dname in self.__frameOriginalData, LOGGER.error("data '%s' doesn't exist, available data are %s"%(name,list(self.__frameOriginalData)))
            value = self.__frameOriginalData[dname]
            assert value is not None, LOGGER.error("data '%s' value seems to be deleted"%name)
        else:
            #info, m = self.__repository.get_file_info(relativePath=os.path.join(self.__usedFrame,dname))
            #assert info is not None, LOGGER.error("unable to pull data '%s' (%s)"%(name, m) )
            #value = self.__repository.pull(relativePath=self.__usedFrame, name=dname)
            isRepoFile,fileOnDisk, infoOnDisk, classOnDisk = self.__repository.is_repository_file(os.path.join(self.__usedFrame,dname))
            assert isRepoFile, LOGGER.error("Original data '%s' is not a repository file"%(dname, ) )
            assert fileOnDisk, LOGGER.error("Original data '%s' is a repository file but not found on disk"%(dname, ) )
            value = self.__repository.pull(relativePath=os.path.join(self.__usedFrame,dname))
        return value

    def is_engine(self, path, repo=False, mes=False, safeMode=True):
        """
        Get whether a fullrmc engine is stored in the given path.

        :Parameters:
            #. path (string): The path to fetch.
            #. repo (boolean): Whether to return repository if an engine is
               found. Otherwise None is returned.
            #. mes (boolean): Whether to return explanatory message.

        :Returns:
            #. result (boolean): The fetch result, True if engine is found
               False otherwise.
            #. repo (pyrep.Repository): The repository instance.
               This is returned only if 'repo' argument is set to True.
            #. message (string): The explanatory message.
               This is returned only if 'mes' argument is set to True.
            #. safeMode (boolean): whether to acquire the lock upon loading.
               this is not necessary unless another process is writing the
               to the repository at the same time
        """
        assert isinstance(repo, bool), LOGGER.error("repo must be boolean")
        assert isinstance(mes, bool), LOGGER.error("mes must be boolean")
        rep = Repository(timeout=self.__timeout)
        rep.DEBUG_PRINT_FAILED_TRIALS = False
        # check if this is a repository
        if path is None:
            result  = False
            rep     = None
            message = "No Path given"
        elif not isinstance(path, basestring):
            result  = False
            rep     = None
            message = "Given path '%s' is not valid"%path
        elif not rep.is_repository(path):
            result  = False
            rep     = None
            message = "No repository found at '%s'"%path
        else:
            # check if this repository is a fullrmc's engine
            info = {'repository type':'fullrmc engine', 'fullrmc version':__version__, 'engine id':self.__id}
            rep  = rep.load_repository(path, safeMode=safeMode)
            if not isinstance(rep.info, dict):
                result  = False
                rep     = None
                message = "Existing repository at '%s' is not a known fullrmc engine. Info must be a dictionary while '%s' is given"%(path, type(rep.info))
            elif len(rep.info) < 3:
                result  = False
                rep     = None
                message = "Existing repository at '%s' is not a known fullrmc engine. Info dictionary length must be >3"%path
            elif rep.info.get('repository type', None) != 'fullrmc engine':
                result  = False
                rep     = None
                message = "Existing repository at '%s' is not a known fullrmc engine. Info dictionary 'repository_type' key value must be 'fullrmc engine'"%path
            elif 'fullrmc version' not in rep.info:
                result  = False
                rep     = None
                message = "Existing repository at '%s' is not a known fullrmc engine. Info dictionary 'fullrmc version' key is not found"%path
            elif 'engine id' not in rep.info:
                result  = False
                rep     = None
                message = "Existing repository at '%s' is not a known fullrmc engine. Info dictionary 'engine id' key was not found"%path
            else:
                result  = True
                # check repository version
                message = ""
                if info['fullrmc version'] != rep.info['fullrmc version']:
                    message += "Engine's version is found to be %s but current installation version is %s. "%(rep.info['fullrmc version'], info['fullrmc version'])
        # return
        if repo and mes:
            return result, rep, message
        elif mes:
            return result, message
        elif repo:
            return result, rep
        else:
            return result

    def __runtime_save(self, frame):
        LOGGER.saved("@%s Runtime saving ... DON'T INTERRUPT"%frame)
        # dump engine's used frame FRAME_DATA
        for dname in self.RUNTIME_DATA:
            value = self.__dict__[dname]
            name = dname
            self.__repository.update_file(value=value, relativePath=os.path.join(frame, dname))
        # dump constraints' used frame FRAME_DATA
        for c in self.__constraints:
            cp = os.path.join(frame, 'constraints', c.constraintName)
            for dname in c.RUNTIME_DATA:
                value = c.__dict__[dname]
                self.__repository.update_file(value=value, relativePath=os.path.join(cp, dname))
        # engine saved
        LOGGER.saved("@%s Runtime save is successful"%(frame,) )

    def save(self, path=None, copyFrames=True):
        """
        Save engine to disk.

        :Parameters:
            #. path (None, string): Repository path to save the engine.
               If path is None, engine's path will be used.
               If path and engine's path are both None, and error will
               be raised.
            #. copyFrames (boolean): If path is None, this argument is
               discarded. This argument sets whether to copy all frames data
               to the new repository path. If path is not None and this
               argument is False, Only used frame data will be copied and other
               frames will be discarded in new engine.

        N.B. If path is given, it will automatically update engine's path to
        point towards given path.
        """
        assert isinstance(copyFrames, bool), LOGGER.error("copyFrames must be boolean")
        LOGGER.saved("Saving Engine and frame %s data... DON'T INTERRUPT"%self.__usedFrame)
        # create info dict
        info = {'repository type':'fullrmc engine', 'fullrmc version':__version__, 'engine id':self.__id}
        # path is given
        if path is not None:
            result, message = self.__check_path_to_create_repository(path)
            assert result, LOGGER.error(message)
            REP = Repository(timeout=self.__timeout)
            REP.DEBUG_PRINT_FAILED_TRIALS = False
            REP.create_repository(path, info=info)
            self.__path = path
        # first time saving this engine
        elif self.__repository is None:
            assert self.__path is not None, LOGGER.error("Given path and engine's path are both None, must give a valid path for saving.")
            REP = Repository(timeout=self.__timeout)
            REP.DEBUG_PRINT_FAILED_TRIALS = False
            REP.create_repository(self.__path, info=info)
        # engine loaded or saved before
        else:
            REP = self.__repository
        # create repository frames
        if self.__repository is None:
            for frameName in self.__frames:
                REP.add_directory( frameName )
        # dump engine
        #REP.dump(value=self, relativePath='engine', replace=True)
        self._dump_to_repository(value=self, relativePath='engine', repository=REP)
        # dump used frame ENGINE_DATA
        for dname in self.ENGINE_DATA:
            value = self.__dict__[dname]
            #REP.dump(value=value, relativePath=dname, replace=True)
            self._dump_to_repository(value=value, relativePath=dname, repository=REP)
        # dump engine's used frame FRAME_DATA
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(self.__usedFrame)
        assert not isMultiframe, LOGGER.report("Used frame '%s' is a multiframe. This should have never had happened. Please report issue"%self.__usedFrame)
        #if isNormalFrame or isSubframe:
        for dname in self.FRAME_DATA:
            value = self.__dict__[dname]
            #REP.dump(value=value, relativePath=os.path.join(self.__usedFrame, dname), replace=True)
            self._dump_to_repository(value=value, relativePath=os.path.join(self.__usedFrame, dname), repository=REP)
        for dname in self.MULTIFRAME_DATA:
            value = self.__dict__[dname]
            rpath = dname if isNormalFrame else os.path.join(self.__usedFrame, dname)
            #REP.dump(value=value, relativePath=rpath, replace=True)
            self._dump_to_repository(value=value, relativePath=rpath, repository=REP)
        # dump original frame data
        #if isNormalFrame or isSubframe:
        for name in self.__frameOriginalData:
            value = self.__frameOriginalData[name]
            if value is not None:
                if isinstance(self.__usedFrame, basestring):
                    #REP.dump(value=value, relativePath=os.path.join(self.__usedFrame, name), replace=True)
                    self._dump_to_repository(value=value, relativePath=os.path.join(self.__usedFrame, name), repository=REP)
                else:
                    for i in self.__usedFrame['frames_name']:
                        #REP.dump(value=value, relativePath=os.path.join(self.__usedFrame, i,name), replace=True)
                        self._dump_to_repository(value=value, relativePath=os.path.join(self.__usedFrame, i,name), repository=REP)
                self.__frameOriginalData[name] = None
        # dump constraints' used frame FRAME_DATA
        #if isNormalFrame or isSubframe:
        for c in self.__constraints:
            REP.add_directory( os.path.join(self.__usedFrame, 'constraints', c.constraintName) )
            for dname in c.FRAME_DATA:
                value = c.__dict__[dname]
                #REP.dump(value=value, relativePath=os.path.join(self.__usedFrame, 'constraints', c.constraintName, dname), replace=True)
                self._dump_to_repository(value=value, relativePath=os.path.join(self.__usedFrame, 'constraints', c.constraintName, dname), repository=REP)
        # copy rest of frames
        if (self.__repository is not None) and (path is not None and copyFrames):
            raise Exception("MUST IMPLEMENT copy_directory_from_repository in REPOSITORY")
            # dump rest of frames
            for frameName in self.__frames:
                if frameName == self.__usedFrame:
                    continue
                REP.copy_directory_from_repository(fromRepository=self.__repository, fromRelativePath=frameName, relativePath=frameName, repository=REP)
                #for rp, _ in self.__repository.walk_files_path(relativePath=frameName, recursive=True):
                #    value = self.__repository.pull(relativePath=os.path.join(frameName,rp))
                #    REP.dump(value=value, relativePath=os.path.join(frameName, rp), replace=True)
                #    self._dump_to_repository(value=value, relativePath=os.path.join(frameName, rp), repository=REP)
        # set repository
        self.__repository = REP
        # set mustSave flag
        self.__mustSave = False
        # engine saved
        LOGGER.saved("Engine and frame %s data saved successfuly to '%s'"%(self.__usedFrame, self.__path) )


    def load(self, path, safeMode=True):
        """
        Load and return engine instance. None of the current engine attribute
        will be updated. must be used as the following:


        .. code-block:: python

            # import engine
            from fullrmc.Engine import Engine

            # create engine
            ENGINE = Engine().load(path)


        :Parameters:
            #. path (string): Directory path to save the engine.
            #. safeMode (boolean): whether to acquire the lock upon loading.
               this is not necessary unless another process is writing the
               to the repository at the same time

        :Returns:
            #. engine (Engine): Engine instance.
        """
        #with open(os.path.join(path, '.pyreplock')) as fd:
        #    print('start',time.time(),fd.readlines(),os.getpid())
        # check whether an engine exists at this path
        isEngine, REP, message = self.is_engine(path=path, repo=True, mes=True, safeMode=safeMode)
        #with open(os.path.join(path, '.pyreplock')) as fd:
        #    print('is engine',message,fd.readlines(),time.time(),os.getpid())
        if not isEngine:
            raise Exception(LOGGER.error(message))
        if len(message):
            LOGGER.warn(message)
        # load engine
        #if self.__repository is not None:
        #    self.__repository.locker.stop()
        #REP.locker.start()
        engine = REP.pull(relativePath='engine')
        engine._set_repository(REP)
        engine._set_path(path)
        # pull engine's ENGINE_DATA
        for name in engine.ENGINE_DATA:
            value = REP.pull(relativePath=name)
            object.__setattr__(engine, name, value)
        # convert all frames to new version
        if isinstance(engine._Engine__frames, (list,set,tuple)):
            engine._Engine__frames = dict([(f,None) for f in engine._Engine__frames])
        # pull engine's FRAME_DATA
        isNormalFrame, isMultiframe, isSubframe = engine.get_frame_category(engine.usedFrame)
        assert not isMultiframe, LOGGER.report("Used frame '%s' is a multiframe. This should have never had happened. Please report issue"%engine.usedFrame)
        # add frame data
        for dname in engine.FRAME_DATA:
            value = REP.pull(relativePath=os.path.join(engine.usedFrame,dname))
            object.__setattr__(engine, dname, value)
        for dname in engine.MULTIFRAME_DATA:
            rpath = dname if isNormalFrame else os.path.join(engine.usedFrame,dname)
            value = REP.pull(relativePath=rpath)
            object.__setattr__(engine, dname, value)
        # set constraints engine
        [object.__setattr__(c, '_Constraint__engine', engine) for c in engine._Engine__constraints]
        # remove old constraints from broadcaster and add new ones
        # pull constraints data
        for c in engine.constraints:
            for dname in c.FRAME_DATA:
                value = REP.pull(relativePath=os.path.join(engine.usedFrame, 'constraints', c.constraintName, dname))
                object.__setattr__(c, dname, value)
        [getattr(engine, '_Engine__broadcaster').remove_listener(l) for l in getattr(engine, '_Engine__broadcaster').listeners if isinstance(l, Constraint)]
        [getattr(engine, '_Engine__broadcaster').add_listener(c)    for c in getattr(engine, '_Engine__constraints')]
        # set engine group selector
        engine.groupSelector.set_engine(engine)
        # set engine must save to false
        object.__setattr__(engine, '_Engine__mustSave', False)
        # return engine instance
        return engine

    def set_log_file(self, logFile):
        """
        Set the log file basename.

        :Parameters:
            #. logFile (None, string): Logging file basename. A logging file
               full name will be the given logFile appended '.log'
               extension automatically.
        """
        assert isinstance(logFile, basestring), LOGGER.error("logFile must be a string, '%s' is given"%logFile)
        LOGGER.set_log_file_basename(logFile)

    def __create_frame_data(self, frame):
        # THIS METHOD IS CALLED in set_used_frame. IT WILL BEEXECUTED AFTER ENGINE HAS BEEN SAVED
        def check_create_or_raise(this, relativePath, isMultiFrame, originalData=False):
            if isMultiFrame:
                frameData = [fd for fd in this.FRAME_DATA] + [fd for fd in this.MULTIFRAME_DATA]
            else:
                frameData = [fd for fd in this.FRAME_DATA] # this is a traditional frame
            # find missing data
            missingFrameData    = []
            missingOriginalData = []
            if not self.__repository.is_repository_directory(relativePath):
                self.__repository.add_directory( relativePath )
                missingFrameData = [item for item in frameData]
                if originalData:
                    missingOriginalData = list(self.__frameOriginalData)
            else:
                # check for abnormalities in missing frame data files
                for name in frameData:
                    response = self.__repository.is_repository_file(os.path.join(relativePath,name))
                    if not response[0]:
                        missingFrameData.append(name)
                if len(missingFrameData):
                    assert self.__frames[self.__usedFrame.split(os.sep)[0]] is None, LOGGER.error("Creating frame '%s' data for the first time when used frame is a multiframe is not allowed."%(relativePath,))
                    assert len(missingFrameData) == len(frameData), LOGGER.error("Data files %s are missing from frame '%s'. Consider deleting and rebuilding frame."%(missingFrameData,relativePath,))
                # check original missing data
                if originalData:
                    for name in self.__frameOriginalData:
                        response = self.__repository.is_repository_file(os.path.join(relativePath,name))
                        if not response[0]:
                            missingOriginalData.append(name)
                    if len(missingOriginalData):
                        assert self.__frames[self.__usedFrame.split(os.sep)[0]] is None, LOGGER.error("Creating frame '%s' original data for the first time when used frame is a multiframe is not allowed."%(relativePath,))
                        assert len(missingFrameData), LOGGER.error("Frame '%s' original data is missing while other data are not"%(relativePath,))
                        assert len(missingOriginalData) == len(self.__frameOriginalData), LOGGER.error("Data files %s are missing from frame '%s'. Consider deleting and rebuilding frame."%(missingFrameData,frame,))
            # if nothing is missing then frame is already built
            if not len(missingFrameData) and not len(missingOriginalData):
                return False
            isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(self.__usedFrame)
            _normalFrames = sorted([f for f in self.__frames if self.__frames[f] is None])
            assert isNormalFrame, LOGGER.error("It's not allowed to create frame data when used frame is a multiframe or a multiframe subframe. Use set_used_frame to a normal frame (%s) then attend to create '%s' data"%(_normalFrames, relativePath,))
            # check all or None missing
            if len(missingFrameData):
                LOGGER.frame("Using frame '%s' data to create '%s' frame '%s' data."%(self.__usedFrame, this.__class__.__name__, relativePath))
                # create frame data
                for name in frameData:
                    value = this.__dict__[name]
                    #self.__repository.dump(value=value, relativePath=os.path.join(relativePath, name), replace=True)
                    self._dump_to_repository(value=value, relativePath=os.path.join(relativePath, name))
            if len(missingOriginalData):
                LOGGER.frame("Using frame '%s' data to create frame '%s' original data."%(self.__usedFrame, relativePath))
                for name in self.__frameOriginalData:
                    value = self.__repository.pull(relativePath=os.path.join(self.__usedFrame,name))
                    #self.__repository.dump(value=value, relativePath=os.path.join(relativePath, name),replace=True)
                    self._dump_to_repository(value=value, relativePath=os.path.join(relativePath, name))
            # return
            return True
        # get frame paths from frameName
        firstLevel = frame.split(os.sep)[0]
        # get frame paths list
        if self.__frames[firstLevel] is None:
            isMultiFrame = False
            paths        = [frame]
        else:
            isMultiFrame = True
            paths        = [os.path.join(firstLevel,fn) for fn in self.__frames[firstLevel]['frames_name']]
        # check and create first frame data in path
        built = check_create_or_raise(this=self, relativePath=paths[0], isMultiFrame=isMultiFrame, originalData=True)
        # create constraints frame data
        for c in self.__constraints:
            # isMultiFrame for constraint shall be always False because MULTIFRAME_DATA is not defined
            _ = check_create_or_raise(this=c, relativePath=os.path.join(paths[0],'constraints',c.constraintName), isMultiFrame=False, originalData=False)
        # duplicate multiframe data
        if len(paths) > 1 and built:
            for relativePath in paths[1:]:
                LOGGER.frame("Creating '%s' by duplicating '%s'"%(relativePath, paths[0]))
                success, error = self.__repository.copy_directory(relativePath=paths[0], newRelativePath=relativePath, raiseError=False)
                assert success, LOGGER.error(error)

    def is_frame(self, frame):
        """
        Check whether a given frame exists.

        :Parameters:
            #. frame (string): Frame name.

        :Returns:
            #. result (boolean): True if frame exists False otherwise.
        """
        assert isinstance(frame, basestring), LOGGER.error("frame must be a string, '%s' is given instead"%frame)
        return frame in self.__frames
        #return frame in self.__frames

    def add_frames(self, frames):
        """
        Add a one or many (multi)frame to engine.

        :Parameters:
            #. frames (string, dict, list): It can be a string to add a single
               frame, a dictionary to add a single multiframe or a list of
               strings and/or dictionaries to add multiple (multi)frames.
        """
        _frames = []
        for frm in self.__check_frames(frames, raiseExisting=False):
            fname = frm if isinstance(frm, basestring) else frm['name']
            if fname in self.__frames:
                LOGGER.frame("frame '%s' exists already. Adding ommitted"%(fname,))
                continue
            else:
                _frames.append(frm)
        # create frames directories
        if self.__repository is not None:
            for frm in _frames:
                fname = frm if isinstance(frm, basestring) else frm['name']
                self.__repository.add_directory( fname )
        # append frames to list
        for frm in _frames:
            if isinstance(frm, basestring):
                self.__frames[frm] = None
            else:
                self.__frames[frm['name']] = frm
        # save frames
        if self.__repository is not None:
            self._dump_to_repository(value=self.__frames, relativePath='_Engine__frames')

    def add_frame(self, frame):
        """
        Add a single (multi)frame to engine.

        :Parameters:
            #. frame (string): Frame name.
        """
        self.add_frames([frame])

    def reinit_frame(self, frame):
        """
        Reset frame data to initial pdb coordinates.

        :Parameters:
            #. frame (string): The frame name to set.
        """
        if self.__repository is None and frame != self.__usedFrame:
            raise Exception(LOGGER.error("It's not allowed to re-initialize frame other than usedFrame prior to building engine's repository. Save engine using Engine.save method first."))
        # get frame type
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
        if isNormalFrame or isSubframe:
            allFrames = [frame]
        else:
            LOGGER.usage("Re-init multiframe '%s' all %i subframes"%(frame,len(self.__frames[frame]['frames_name']),) )
            allFrames = [os.path.join(frame, frm) for frm in self.__frames[frame]['frames_name']]
        # get old used frame
        oldUsedFrame = self.__usedFrame
        for frm in allFrames:
            LOGGER.info("Re-init frame '%s'"%(frm,) )
            # temporarily set used frame
            if frm != self.__usedFrame:
                self.set_used_frame(frm)
            # reset pdb
            self.set_pdb(self.__pdb)
            #  re-set old used frame
        if self.__usedFrame != oldUsedFrame:
            self.set_used_frame(oldUsedFrame)

    def __validate_frame_name(self, frame):
        assert isinstance(frame, basestring), LOGGER.error("Frame must be a string, '%s' is given instead"%frame)
        splitFrame = frame.split(os.sep)
        assert len(splitFrame) <=2, LOGGER.error("Frame must be 1 level deep or at most 2 levels deep if multiframe")
        if splitFrame[0] not in self.__frames:
            if len(splitFrame) ==1:
                raise Exception(LOGGER.error("Unkown frame name '%s'"%frame))
            else:
                raise Exception(LOGGER.error("Unkown multiframe '%s'"%splitFrame[0]))
        if self.__frames[splitFrame[0]] is not None and len(splitFrame)==2:
            assert splitFrame[1] in self.__frames[splitFrame[0]]['frames_name'], LOGGER.error("Unkown subframe '%s' of registered multiframe '%s'"%(splitFrame[1],splitFrame[0]))
        #return tuple(frame)

    def get_frame_category(self, frame):
        """Get whether a given frame name is a normal frame or a multiframe or
        a multiframe subframe. If frame does not exist an error will be raised.

        :Parameters:
            #. frame (string): Frame name or repository relative path

        :Returns:
            #. isNormalFrame (boolean): Whether it's a normal single frame
            #. isMultiframe (boolean): Whether it's a multiframe
            #. isSubframe (boolean): Whether it's a multiframe subframe path
        """
        self.__validate_frame_name(frame)
        isNormalFrame = self.__frames.get(frame, -1) is None
        isMultiframe  = isinstance(self.__frames.get(frame, None), dict)
        isSubframe    = not (isMultiframe or isNormalFrame)
        # return
        return isNormalFrame, isMultiframe, isSubframe

    def set_used_frame(self, frame, updateRepo=True):
        """
        Switch engine frame.

        :Parameters:
            #. frame (string): The frame to switch to and use from now on.
            #. updateRepo (boolean): whether to update repository usedFrame
               value
        """
        if frame == self.__usedFrame:
            return
        assert isinstance(updateRepo, bool), "updateRepo must be boolean"
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
        if isMultiframe:
            _frame = os.path.join(frame, self.__frames[frame]['frames_name'][0])
            LOGGER.usage("Given frame '%s' is a multiframe, using multiple frames at the same time is not allowed, used frame is adjusted to first subframe '%s'"%(frame,_frame))
            frame        = _frame
            isMultiframe = False
            isSubframe   = True
        firstLevel = frame.split(os.sep)[0]
        if self.__repository is None:
            raise Exception(LOGGER.error("It's not allowed to set used frame prior to building engine's repository. Save engine using Engine.save method first."))
        # create frame data in repository if not yet created
        self.__create_frame_data(frame=frame)
        # pull engine's FRAME_DATA
        for dname in self.FRAME_DATA:
            value = self.__repository.pull(relativePath=os.path.join(frame,dname))
            object.__setattr__(self, dname, value)
        for dname in self.MULTIFRAME_DATA:
            rpath = dname if isNormalFrame else os.path.join(frame, dname)
            value = self.__repository.pull(rpath)
            object.__setattr__(self, dname, value)
        # set constraints engine
        [object.__setattr__(c, '_Constraint__engine', self) for c in self.__constraints]
        # remove old constraints from broadcaster and add new ones
        [self.__broadcaster.remove_listener(l) for l in self.__broadcaster.listeners if isinstance(l, Constraint)]
        [self.__broadcaster.add_listener(c) for c in self.__constraints]
        # pull constraints' used frame FRAME_DATA
        for c in self.__constraints:
            for dname in c.FRAME_DATA:
                value = self.__repository.pull(relativePath=os.path.join(frame, 'constraints', c.constraintName, dname))
                object.__setattr__(c, dname, value)
        # set group selector engine
        self.__groupSelector.set_engine(self)
        # save used frame
        self.__usedFrame = frame
        # update repository
        if updateRepo:
            self.__repository.update_file(value=self.__usedFrame, relativePath='_Engine__usedFrame')

    def use_frame(self, *args, **kwargs):
        """alias to set_used_frame."""
        self.set_used_frame(*args, **kwargs)

    def delete_frame(self, frame):
        """
        Delete frame data from Engine as well as from system.

        :Parameters:
            #. frame (string): The frame to delete.
        """
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
        assert frame != self.__usedFrame, LOGGER.error("It's not safe to delete the used frame '%s'"%frame)
        if isNormalFrame:
            _f = [f for f in self.__frames if self.__frames[f] is None]
            assert len(_f)>=1, LOGGER.error("No traditional frames found. This shouldn't have happened. Report issue ...")
            assert len(_f)>=2, LOGGER.error("It's not allowed to delete the last traditional frame in engine '%s'"%(_f[0],))
        if isSubframe:
            _name = frame.split(os.sep)[0]
            if len(self.__frames[_name]['frames_name']) == 1:
                LOGGER.usage("Deleting last subframe '%s' of multiframe '%s' has resulted in deleting the multiframe"%(frame, _name))
                frame         = _name
                isNormalFrame = False
                isSubframe    = False
                isMultiframe  = True
        # remove frame directory
        if self.__repository is not None:
            self.__repository.remove_directory(relativePath=frame, clean=True)
        # reset frames
        if isNormalFrame or isMultiframe:
            self.__frames.pop(frame)
        else:
            _multiframe, _subframe = frame.split(os.sep)
            self.__frames[_multiframe]['frames_name'] = [frm for frm in self.__frames[_multiframe]['frames_name'] if frm !=_subframe]
        # save frames
        if self.__repository is not None:
            self.__repository.update_file(value=self.__frames, relativePath='_Engine__frames')

    def rename_frame(self, frame, newName):
        """
        Rename (multi)frame.

        :Parameters:
            #. frame (string): The frame to rename.
            #. newName (string): The new name.
        """
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
        newName = self.__check_get_frame_name(newName)
        if isNormalFrame or isMultiframe:
           assert newName not in self.__frames, LOGGER.error("Give frame new name '%s' already exists"%newName)
           newFrame = newName
        else:
           _multiframe, _= frame.split(os.sep)
           assert newName not in self.__frames[_multiframe]['frames_name'], LOGGER.error("Give frame new name '%s' already exists in multiframe '%s'"%(newName,_multiframe))
           newFrame = os.path.join(_multiframe, newName)
        # rename frame in repository
        if self.__repository is not None:
            try:
                self.__repository.rename_directory(relativePath=frame, newName=newName)
            except Exception as err:
                raise Exception(LOGGER.error("Unable to rename frame (%s)"%(str(err),)) )
        # reset frames
        if isNormalFrame or isMultiframe:
            self.__frames[newFrame] = self.__frames.pop(frame)
        else:
            _multiframe, _subframe = frame.split(os.sep)
            self.__frames[_multiframe]['frames_name'] = [frm if frm!=_subframe else newName for frm in self.__frames[_multiframe]['frames_name']]
        # check used frame
        if self.__usedFrame == frame:
            self.__usedFrame = newFrame
        elif self.__usedFrame.split(os.sep)[0] == frame:
            self.__usedFrame = os.path.join(newName, self.__usedFrame.split(os.sep)[1])
        # save frames
        if self.__repository is not None:
            self.__repository.update_file(value=self.__frames, relativePath='_Engine__frames')
            self.__repository.update_file(value=self.__usedFrame, relativePath='_Engine__usedFrame')

    def export_pdb(self, path, frame=None):
        """
        Export a pdb file of the last refined and saved configuration state.

        :Parameters:
            #. path (string): the pdb file path.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used.
        """
        if frame is None:
            frame = self.__usedFrame
        else:
            isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
            assert not isMultiframe, LOGGER.error("Given frame is a multiframe, only normal frame or subframe is allowed")
        # get data
        if frame == self.__usedFrame:
            _ac  = self._atomsCollector
            _pdb = self.__pdb
            _rc  = self.__realCoordinates
            _bc  = self.__boundaryConditions
        else:
            assert self.__repository is not None, LOGGER.error("Repository is not built yet. Save engine before attempting to export pdb")
            assert self.__repository.is_repository_directory(frame), LOGGER.error("Frame is not found in repository. Try using set_used_frame('%s') prior to exporting pdb"%(frame,))
            try:
                _ac  = self.__repository.pull(relativePath=os.path.join(frame,'_atomsCollector'))
                _pdb = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__pdb'))
                _rc  = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__realCoordinates'))
                _bc  = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__boundaryConditions'))
            except Exception as err:
                assert False, LOGGER.error("Unable to pull data to export pdb (%s)"%(str(err),))
        # export pdb
        # MUST TRANSFORM TO PDB COORDINATES SYSTEM FIRST
        if len(_ac.indexes):
            indexes = sorted(set(_pdb.indexes)-set(_ac.indexes))
            pdb = _pdb.get_copy(indexes=indexes)
        else:
            pdb = _pdb
        pdb.export_pdb(path, coordinates=_rc, boundaryConditions=_bc )


    def get_pdb(self, frame=None):
        """
        Get a pdb instance of the last refined and save configuration state.

        :Parameters:
            #. frame (None, string): Target frame name. If None, engine used
               frame is used.

        :Returns:
            #. pdb (pdbparser): The pdb instance.
        """
        if frame is None:
            frame = self.__usedFrame
        else:
            isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
            assert not isMultiframe, LOGGER.error("Given frame is a multiframe, only normal frame or subframe is allowed")
        # get data
        if frame == self.__usedFrame:
            _ac  = self._atomsCollector
            _pdb = self.__pdb
            _rc  = self.__realCoordinates
            _bc  = self.__boundaryConditions
        else:
            assert self.__repository is not None, LOGGER.error("Repository is not built yet. Save engine before attempting to export pdb")
            assert self.__repository.is_repository_directory(frame), LOGGER.error("Frame is not found in repository. Try using set_used_frame('%s') prior to exporting pdb"%(frame,))
            try:
                _ac  = self.__repository.pull(relativePath=os.path.join(frame,'_atomsCollector'))
                _pdb = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__pdb'))
                _rc  = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__realCoordinates'))
                _bc  = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__boundaryConditions'))
            except Exception as err:
                assert False, LOGGER.error("Unable to pull data to export pdb (%s)"%(str(err),))
        # create and return pdb
        indexes = None
        if len(_ac.indexes):
            indexes = sorted(set(_pdb.indexes)-set(_ac.indexes))
        pdb = _pdb.get_copy(indexes=indexes)
        pdb.set_coordinates(_rc)
        pdb.set_boundary_conditions(_bc)
        return pdb

    def set_tolerance(self, tolerance):
        """
        Set engine's runtime tolerance value.

        :Parameters:
            #. tolerance (number): The runtime tolerance parameters.
               It's the percentage [0,100] of allowed unsatisfactory 'tried'
               moves.
        """
        assert is_number(tolerance), LOGGER.error("tolerance must be a number")
        tolerance = FLOAT_TYPE(tolerance)
        assert tolerance>=0, LOGGER.error("tolerance must be positive")
        assert tolerance<=100, LOGGER.error("tolerance must be smaller than 100")
        self.__tolerance = FLOAT_TYPE(tolerance/100.)
        # dump to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__tolerance, relativePath=os.path.join(self.__usedFrame,'_Engine__tolerance'))

    def set_group_selector(self, selector, frame=None):
        """
        Set engine's group selector instance.

        :Parameters:
            #. selector (None, GroupSelector): The GroupSelector instance.
               If None is given, RandomSelector is set automatically.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used.
        """
        if selector is None:
            selector = RandomSelector(self)
        else:
            assert isinstance(selector, GroupSelector), LOGGER.error("Given selector must a GroupSelector instance")
        # get frames
        usedIncluded, frame, allFrames = get_caller_frames(engine=self,
                                                           frame=frame,
                                                           subframeToAll=False,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        # change old selector engine instance to None
        if usedIncluded:
            if self.__groupSelector is not None:
                self.__groupSelector.set_engine(None)
            # remove from broadcaster listeners list
            #self.__broadcaster.remove_listener(self.__groupSelector)
            # set new selector
            selector.set_engine(self)
            self.__groupSelector = selector
        # add to broadcaster listeners list
        #self.__broadcaster.add_listener(self.__groupSelector)
        # save selector to repository
        if frame != self.__usedFrame:
            assert self.__repository is not None, LOGGER.error("Repository is not created. Unable to set tolerance for other than engine used frame")
        if self.__repository is not None:
            for frm in allFrames:
                self.__repository.update_file(value=selector, relativePath=os.path.join(frm,'_Engine__groupSelector'))
        #if self.__repository is not None:
        #    #self.__repository.dump(value=self.__groupSelector, relativePath=self.__usedFrame, name='_Engine__groupSelector', replace=True)
        #    self.__repository.update_file(value=self.__groupSelector, relativePath=os.path.join(self.__usedFrame,'_Engine__groupSelector'))

    def clear_groups(self):
        """ Clear all engine's defined groups.
        """
        self.__groups = []
        # save groups to repository
        if self.__repository is not None:
            #self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)
            #self.__repository.update_file(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
            self._dump_to_repository(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))

    def add_group(self, g, broadcast=True):
        """
        Add a group to engine's groups list.

        :Parameters:
            #. g (Group, integer, list, set, tuple numpy.ndarray): Group
               instance, integer, list, tuple, set or numpy.ndarray of atoms
               index.
            #. broadcast (boolean): Whether to broadcast "update groups".
               This is to be used interally only. Keep default value unless
               you know what you are doing.
        """
        if isinstance(g, EmptyGroup):
            gr = g
        elif isinstance(g, Group):
            assert np.max(g.indexes)<self.__pdb.numberOfAtoms, LOGGER.error("group index must be smaller than number of atoms in system")
            gr = g
        elif is_integer(g):
            g = INT_TYPE(g)
            assert g<self.__pdb.numberOfAtoms, LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = Group(indexes= [g] )
        elif isinstance(g, (list, set, tuple)):
            sortedGroup = sorted(set(g))
            assert len(sortedGroup) == len(g), LOGGER.error("redundant indexes found in group")
            assert is_integer(sortedGroup[-1]), LOGGER.error("group indexes must be integers")
            assert sortedGroup[-1]<self.__pdb.numberOfAtoms, LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = Group(indexes= sortedGroup )
        else:
            assert isinstance(g, np.ndarray), LOGGER.error("each group in groups can be either a list, set, tuple, numpy.ndarray or fullrmc Group instance")
            # check group dimension
            assert len(g.shape) == 1, LOGGER.error("each group must be a numpy.ndarray of dimension 1")
            assert len(g), LOGGER.error("group found to have no indexes")
            # check type
            assert "int" in g.dtype.name, LOGGER.error("each group in groups must be of integer type")
            # sort and check limits
            sortedGroup = sorted(set(g))
            assert len(sortedGroup) == len(g), LOGGER.error("redundant indexes found in group")
            assert sortedGroup[-1]<self.__pdb.numberOfAtoms, LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = Group(indexes= g )
        # append group
        self.__groups.append( gr )
        # set group engine
        gr._set_engine(self)
        # broadcast to constraints
        if broadcast:
            self.__broadcaster.broadcast("update groups")
         # save groups
        if self.__repository is not None and self.__saveGroupsFlag:
            #self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)
            #self.__repository.update_file(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'), replace=True)
            self._dump_to_repository(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))

    def set_groups(self, groups):
        """
        Set engine's groups.

        :Parameters:
            #. groups (None, Group, list): A single Group instance or a list,
               tuple, set of any of Group instance, integer, list, set, tuple
               or numpy.ndarray of atoms index that will be set one by one
               by set_group method. If None is given, single atom groups of
               all atoms will be all automatically created which is the same
               as using set_groups_as_atoms method.
        """
        self.__saveGroupsFlag = False
        lastGroups            = self.__groups
        self.__groups         = []
        try:
            if groups is None:
                self.__groups = [Group(indexes=[idx]) for idx in self.__pdb.xindexes]
            elif isinstance(groups, Group):
                self.add_group(groups, broadcast=False)
            else:
                assert isinstance(groups, (list,tuple,set)), LOGGER.error("groups must be a None, Group, list, set or tuple")
                for g in groups:
                    self.add_group(g, broadcast=False)
        except Exception as e:
            self.__saveGroupsFlag = True
            self.__groups         = lastGroups
            raise Exception(LOGGER.error(e))
            return
        # save groups to repository
        if self.__repository is not None:
            #self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)
            #self.__repository.update_file(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
            self._dump_to_repository(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")

    def add_groups(self, groups):
        """
        Add groups to engine.

        :Parameters:
            #. groups (Group, list): Group instance or list of groups,
               where every group must be a Group instance or a
               numpy.ndarray of atoms index of type numpy.int32.
        """
        self.__saveGroupsFlag = False
        lastGroups            = [g for g in self.__groups]
        try:
            if isinstance(groups, Group):
                self.add_group(groups, broadcast=False)
            else:
                assert isinstance(groups, (list,tuple,set)), LOGGER.error("groups must be a list of numpy.ndarray")
                for g in groups:
                    self.add_group(g, broadcast=False)
        except Exception as e:
            self.__saveGroupsFlag = True
            self.__groups = lastGroups
            raise Exception(LOGGER.error(e))
            return
        # save groups to repository
        if self.__repository is not None:
            #self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)
            #self.__repository.update_file(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
            self._dump_to_repository(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")

    def set_groups_as_atoms(self):
        """ Automatically set engine's groups as single atom group for
        all atoms."""
        self.set_groups(None)

    def set_groups_as_molecules(self):
        """ Automatically set engine's groups indexes according to
        molecules indexes. """
        molecules = list(set(self.__moleculesIndex))
        moleculesIndex = {}
        for idx in range(len(self.__moleculesIndex)):
            mol = self.__moleculesIndex[idx]
            if not mol in moleculesIndex:
                moleculesIndex[mol] = []
            moleculesIndex[mol].append(idx)
        # create groups
        keys = sorted(moleculesIndex)
        # reset groups
        self.__groups = []
        # add groups
        for k in keys:
            self.add_group(np.array(moleculesIndex[k], dtype=INT_TYPE), broadcast=False)
        # save groups to repository
        if self.__repository is not None:
            #self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)
            #self.__repository.update_file(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
            self._dump_to_repository(value=self.__groups, relativePath=os.path.join(self.__usedFrame,'_Engine__groups'))
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")

    def set_pdb(self, pdb, boundaryConditions=None, names=None, elements=None, moleculesIndex=None, moleculesName=None):
        """
        Set used frame pdb configuration. Engine and constraints data will be
        automatically reset but not constraints definitions. If pdb was already
        set and this is a resetting to a different atomic configuration, with
        different elements or atomic order, or different size and number of
        atoms, constraints definitions must be reset manually. In general,
        their is no point in changing the atomic configuration of a completely
        different atomic nature. It is advisable to create a new engine from
        scratch or redefining all constraints definitions.

        :Parameters:
            #. pdb (pdbparser, string): the configuration pdb as a pdbparser
               instance or a path string to a pdb file.
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries,
               numpy.ndarray, number): The configuration's boundary conditions.
               If None, boundaryConditions will be parsed from pdb if existing
               otherwise, InfiniteBoundaries with no periodic boundaries will
               be set. If numpy.ndarray is given, it must be pass-able
               to a PeriodicBoundaries instance. Normally any real
               numpy.ndarray of shape (1,), (3,1), (9,1), (3,3) is allowed.
               If number is given, it's like a numpy.ndarray of shape (1,),
               it is assumed as a cubic box of box length equal to number.
            #. names (None, list): Atoms names list. If None is given, names
               will be calculated automatically by parsing pdb instance.
            #. elements (None, list): Atoms elements list. If None is given,
               elements will be calculated automatically by parsing pdb
               instance.
            #. moleculesIndex (None, list, numpy.ndarray): Molecules index
               list. Must have the length of number of atoms. If None is given,
               moleculesIndex will be calculated automatically by parsing pdb
               instance.
            #. moleculesName (None, list): Molecules name list. Must have the
               length of the number of atoms. If None is given, it is
               automatically generated as the pdb residues name.
        """
        if pdb is None:
            pdb = pdbparser()
            bc  = PeriodicBoundaries()
            bc.set_vectors(1)
            pdb.set_boundary_conditions(bc)
        if not isinstance(pdb, pdbparser):
            try:
                pdb = pdbparser(pdb)
            except:
                raise Exception( LOGGER.error("pdb must be None, pdbparser instance or a string path to a protein database (pdb) file.") )
        # set pdb
        self.__pdb = pdb
        # get coordinates
        self.__realCoordinates = np.array(self.__pdb.coordinates, dtype=FLOAT_TYPE)
        # save data to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__pdb, relativePath=os.path.join(self.__usedFrame,'_Engine__pdb'))
            self.__repository.update_file(value=self.__realCoordinates, relativePath=os.path.join(self.__usedFrame,'_Engine__realCoordinates'))
        # reset AtomsCollector
        self._atomsCollector.reset()
        # set boundary conditions
        if boundaryConditions is None:
            boundaryConditions = pdb.boundaryConditions
        self.set_boundary_conditions(boundaryConditions)
        # get elementsIndex
        self.set_elements_index(elements)
        # get namesIndex
        self.set_names_index(names)
        # get moleculesIndex
        self.set_molecules_index(moleculesIndex=moleculesIndex, moleculesName=moleculesName)
        # broadcast to constraints
        self.__broadcaster.broadcast("update pdb")
        # reset engine flags
        self.reset_engine()

    def set_boundary_conditions(self, boundaryConditions):
        """
        Sets the configuration's boundary conditions. Any type of periodic or
        infinite boundary conditions is allowed and not restricted to cubic.
        Engine and constraints data will be automatically reset. Number density
        will be automatically calculated upon setting boundary conditions. In
        the case where inifinite boundaries are set, which is needed to
        simulate isolated atomic systems such as nano-particles, the volume
        is theoretically infinite and therefore number density must be 0.
        But experimentally the measured samples are diluted in solvants and the
        actual number density must be the experimental one. To avoid numerical
        instabilities, number density will be automatically set to water's one
        equal to 0.0333679 and volume will be adjusted to the ratio of number
        of atoms devided to the given water number density. If this number is
        not accurate, user can always set the appropriate number density using
        the stochastic engine 'set_number_density' method


        :Parameters:
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries,
               numpy.ndarray, number): The configuration's boundary conditions.
               If None, InfiniteBoundaries with no periodic boundaries will
               be set. If numpy.ndarray is given, it must be pass-able
               to a PeriodicBoundaries instance. Normally any real
               numpy.ndarray of shape (1,), (3,1), (9,1), (3,3) is allowed.
               If number is given, it's like a numpy.ndarray of shape (1,),
               it is assumed as a cubic box of box length equal to number.
        """
        if boundaryConditions is None:
            boundaryConditions = InfiniteBoundaries()
        if is_number(boundaryConditions) or isinstance(boundaryConditions, (list, tuple, np.ndarray)):
            try:
                bc = PeriodicBoundaries()
                bc.set_vectors(boundaryConditions)
                self.__boundaryConditions = bc
                self.__basisVectors = np.array(bc.get_vectors(), dtype=FLOAT_TYPE)
                self.__reciprocalBasisVectors = np.array(bc.get_reciprocal_vectors(), dtype=FLOAT_TYPE)
                self.__volume = FLOAT_TYPE(bc.get_box_volume())
            except:
                raise Exception( LOGGER.error("boundaryConditions must be an InfiniteBoundaries or PeriodicBoundaries instance or a valid vectors numpy.array or a positive number") )
        elif isinstance(boundaryConditions, InfiniteBoundaries) and not isinstance(boundaryConditions, PeriodicBoundaries):
            self.__boundaryConditions = boundaryConditions
            self.__basisVectors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=FLOAT_TYPE)
            self.__reciprocalBasisVectors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=FLOAT_TYPE)
            self.__volume = FLOAT_TYPE( 1./WATER_NUMBER_DENSITY * self.numberOfAtoms )
        elif isinstance(boundaryConditions, PeriodicBoundaries):
            self.__boundaryConditions = boundaryConditions
            self.__basisVectors = np.array(boundaryConditions.get_vectors(), dtype=FLOAT_TYPE)
            self.__reciprocalBasisVectors = np.array(boundaryConditions.get_reciprocal_vectors(), dtype=FLOAT_TYPE)
            self.__volume = FLOAT_TYPE(boundaryConditions.get_box_volume())
        else:
            raise Exception( LOGGER.error("Unkown boundary conditions. boundaryConditions must be an InfiniteBoundaries or PeriodicBoundaries instance or a valid vectors numpy.array or a positive number") )
        # get box coordinates
        if isinstance(self.__boundaryConditions, PeriodicBoundaries):
            self.__boxCoordinates = transform_coordinates(transMatrix=self.__reciprocalBasisVectors , coords=self.__realCoordinates)
            self.__isPBC = True
            self.__isIBC = False
        else:
            #self.__boxCoordinates = None
            self.__boxCoordinates = self.__realCoordinates
            self.__isPBC = False
            self.__isIBC = True
        # set number density
        self.__numberDensity = FLOAT_TYPE(self.numberOfAtoms) / FLOAT_TYPE(self.__volume)
        # save data to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__boundaryConditions, relativePath=os.path.join(self.__usedFrame,'_Engine__boundaryConditions'))
            self.__repository.update_file(value=self.__basisVectors, relativePath=os.path.join(self.__usedFrame,'_Engine__basisVectors'))
            self.__repository.update_file(value=self.__reciprocalBasisVectors, relativePath=os.path.join(self.__usedFrame,'_Engine__reciprocalBasisVectors'))
            self.__repository.update_file(value=self.__numberDensity, relativePath=os.path.join(self.__usedFrame,'_Engine__numberDensity'))
            self.__repository.update_file(value=self.__volume, relativePath=os.path.join(self.__usedFrame,'_Engine__volume'))
            self.__repository.update_file(value=self.__isPBC, relativePath=os.path.join(self.__usedFrame,'_Engine__isPBC'))
            self.__repository.update_file(value=self.__isIBC, relativePath=os.path.join(self.__usedFrame,'_Engine__isIBC'))
            self.__repository.update_file(value=self.__boxCoordinates, relativePath=os.path.join(self.__usedFrame,'_Engine__boxCoordinates'))
            self.__repository.update_file(value=self.numberOfAtoms, relativePath=os.path.join(self.__usedFrame,'_original__numberOfAtoms'))
            self.__repository.update_file(value=self.__volume, relativePath=os.path.join(self.__usedFrame,'_original__volume'))
            self.__repository.update_file(value=self.__numberDensity, relativePath=os.path.join(self.__usedFrame,'_original__numberDensity'))
            self.__frameOriginalData['_original__numberOfAtoms'] = None
            self.__frameOriginalData['_original__volume']        = None
            self.__frameOriginalData['_original__numberDensity'] = None
        else:
            self.__frameOriginalData['_original__numberOfAtoms'] = self.numberOfAtoms
            self.__frameOriginalData['_original__volume']        = self.__volume
            self.__frameOriginalData['_original__numberDensity'] = self.__numberDensity

        # broadcast to constraints
        self.__broadcaster.broadcast("update boundary conditions")

        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IS
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET.
#        # set mustSave flag
#        self.__mustSave = True

    def set_number_density(self, numberDensity):
        """
        Sets system's number density. This is used to correct system's
        volume. It can only be used with InfiniteBoundaries.

        :Parameters:
            #. numberDensity (number): Number density value that should be
               bigger than zero.
        """
        if isinstance(self.__boundaryConditions, PeriodicBoundaries):
        #if isinstance(self.__boundaryConditions, InfiniteBoundaries) and not isinstance(self.__boundaryConditions, PeriodicBoundaries): # COMMENTED 2017-07-16
            LOGGER.warn("Setting number density is not allowed when boundary conditions are periodic.")
            return
        if self.__isPBC:
            LOGGER.warn("Setting number density is not allowed when boundary conditions are periodic.")
            return
        assert is_number(numberDensity), LOGGER.error("numberDensity must be a number.")
        numberDensity = FLOAT_TYPE(numberDensity)
        assert numberDensity>0, LOGGER.error("numberDensity must be bigger than 0.")
        if numberDensity>1:
            LOGGER.warn("numberDensity value is %.6f value isn't it too big?"%numberDensity)
        self.__numberDensity = numberDensity
        self.__volume = FLOAT_TYPE( 1./numberDensity * self.numberOfAtoms )
        # save data to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__numberDensity, relativePath=os.path.join(self.__usedFrame,'_Engine__numberDensity'))
            self.__repository.update_file(value=self.__volume, relativePath=os.path.join(self.__usedFrame,'_Engine__volume'))
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET.
#        # set mustSave flag
#        self.__mustSave = True

    def set_molecules_index(self, moleculesIndex=None, moleculesName=None):
        """
        Set moleculesIndex list, assigning each atom to a molecule.

        :Parameters:
            #. moleculesIndex (None, list, numpy.ndarray): Molecules index
               list. Must have the length of the number of atoms. If None is
               given, moleculesIndex will be calculated automatically by
               parsing pdb instance.
            #. moleculesName (None, list): Molecules name list. Must have the
               length of the number of atoms. If None is given, it will be
               automatically generated as the pdb residues name.
        """
        if not self.__pdb.numberOfAtoms:
            moleculesIndex = []
        elif moleculesIndex is None:
            moleculesIndex = []
            residues   = self.__pdb.residues
            sequences  = self.__pdb.sequences
            segments   = self.__pdb.segments
            currentRes = residues[0]
            currentSeq = sequences[0]
            currentSeg = segments[0]
            molIndex   = 0
            for idx in range(len(residues)):
                res = residues[idx]
                seq = sequences[idx]
                seg = segments[idx]
                if not(res==currentRes and seq==currentSeq and seg==currentSeg):
                    molIndex += 1
                    currentRes = res
                    currentSeq = seq
                    currentSeg = seg
                moleculesIndex.append(molIndex)
        else:
            assert isinstance(moleculesIndex, (list,set,tuple, np.ndarray)), LOGGER.error("moleculesIndex must be a list of indexes")
            assert len(moleculesIndex)==self.__pdb.numberOfAtoms, LOGGER.error("moleculesIndex must have the same length as pdb")
            if isinstance(moleculesIndex, np.ndarray):
                assert len(moleculesIndex.shape)==1, LOGGER.error("moleculesIndex numpy.ndarray must have a dimension of 1")
                assert moleculesIndex.dtype.type is INT_TYPE, LOGGER.error("moleculesIndex must be of type numpy.int32")
            else:
                for molIdx in moleculesIndex:
                    assert is_integer(molIdx), LOGGER.error("molecule's index must be an integer")
                    molIdx = INT_TYPE(molIdx)
                    assert int(molIdx)>=0, LOGGER.error("molecule's index must positive")
        # check molecules name
        if moleculesName is not None:
            assert isinstance(moleculesName, (list, set, tuple)), LOGGER.error("moleculesName must be a list")
            moleculesName = list(moleculesName)
            assert len(moleculesName)==self.__pdb.numberOfAtoms, LOGGER.error("moleculesName must have the same length as pdb")
        else:
            moleculesName = self.__pdb.residues
        if len(moleculesName):
            molName  = moleculesName[0]
            molIndex = moleculesIndex[0]
            for idx in range(len(moleculesIndex)):
                newMolIndex = moleculesIndex[idx]
                newMolName  = moleculesName[idx]
                if newMolIndex == molIndex:
                    assert newMolName == molName, LOGGER.error("Same molecule atoms can't have different molecule name")
                else:
                    molName  = newMolName
                    molIndex = newMolIndex
        # set moleculesIndex
        self.__numberOfMolecules = len(set(moleculesIndex))
        self.__moleculesIndex    = np.array(moleculesIndex, dtype=INT_TYPE)
        self.__moleculesName     = list(moleculesName)
        # save data to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__numberOfMolecules, relativePath=os.path.join(self.__usedFrame,'_Engine__numberOfMolecules'))
            self.__repository.update_file(value=self.__moleculesIndex, relativePath=os.path.join(self.__usedFrame,'_Engine__moleculesIndex'))
            self.__repository.update_file(value=self.__moleculesName, relativePath=os.path.join(self.__usedFrame,'_Engine__moleculesName'))
            # save original data
            self.__repository.update_file(value=self.__numberOfMolecules, relativePath=os.path.join(self.__usedFrame,'_original__numberOfMolecules'))
            self.__repository.update_file(value=self.__moleculesIndex, relativePath=os.path.join(self.__usedFrame,'_original__moleculesIndex'))
            self.__repository.update_file(value=self.__moleculesName, relativePath=os.path.join(self.__usedFrame,'_original__moleculesName'))
            self.__frameOriginalData['_original__numberOfMolecules'] = None
            self.__frameOriginalData['_original__moleculesIndex']    = None
            self.__frameOriginalData['_original__moleculesName']     = None
        else:
            self.__frameOriginalData['_original__numberOfMolecules'] = copy.deepcopy( self.__numberOfMolecules )
            self.__frameOriginalData['_original__moleculesIndex']    = copy.deepcopy( self.__moleculesIndex  )
            self.__frameOriginalData['_original__moleculesName']     = copy.deepcopy( self.__moleculesName    )
        # broadcast to constraints
        self.__broadcaster.broadcast("update molecules indexes")

        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IS
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET.
#        # set mustSave flag
#        self.__mustSave = True

    def set_elements_index(self, elements=None):
        """
        Set elements and elementsIndex lists, assigning a type element
        to each atom.

        :Parameters:
            #. elements (None, list): Elements list. Must have the
               length of the number of atoms. If None is given,
               elements will be calculated automatically  by parsing pdb
               instance.
        """
        if elements is None:
            elements = self.__pdb.elements
        else:
            assert isinstance(elements, (list,set,tuple)), LOGGER.error("elements must be a list of indexes")
            assert len(elements)==self.__pdb.numberOfAtoms, LOGGER.error("elements have the same length as pdb")
        # set all atoms elements
        self.__allElements = elements
        # get elements
        self.__elements = sorted(set(self.__allElements))
        # get elementsIndex
        lut = dict(zip(self.__elements,range(len(self.__elements))))
        self.__elementsIndex = np.array([lut[el] for el in self.__allElements], dtype=INT_TYPE)
        # number of atoms per element
        self.__numberOfAtomsPerElement = {}
        for el in self.__allElements:
            if not el in self.__numberOfAtomsPerElement:
                self.__numberOfAtomsPerElement[el] = 0
            self.__numberOfAtomsPerElement[el] += 1
        # save data to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__allElements, relativePath=os.path.join(self.__usedFrame,'_Engine__allElements'))
            self.__repository.update_file(value=self.__elements, relativePath=os.path.join(self.__usedFrame,'_Engine__elements'))
            self.__repository.update_file(value=self.__elementsIndex, relativePath=os.path.join(self.__usedFrame,'_Engine__elementsIndex'))
            self.__repository.update_file(value=self.__numberOfAtomsPerElement, relativePath=os.path.join(self.__usedFrame,'_Engine__numberOfAtomsPerElement'))
            # save original data
            self.__repository.update_file(value=self.__allElements, relativePath=os.path.join(self.__usedFrame,'_original__allElements'))
            self.__repository.update_file(value=self.__elements, relativePath=os.path.join(self.__usedFrame,'_original__elements'))
            self.__repository.update_file(value=self.__elementsIndex, relativePath=os.path.join(self.__usedFrame,'_original__elementsIndex'))
            self.__repository.update_file(value=self.__numberOfAtomsPerElement, relativePath=os.path.join(self.__usedFrame,'_original__numberOfAtomsPerElement'))
            self.__frameOriginalData['_original__allElements']             = None
            self.__frameOriginalData['_original__elements']                = None
            self.__frameOriginalData['_original__elementsIndex']           = None
            self.__frameOriginalData['_original__numberOfAtomsPerElement'] = None
        else:
            self.__frameOriginalData['_original__allElements']             = copy.deepcopy( self.__allElements )
            self.__frameOriginalData['_original__elements']                = copy.deepcopy( self.__elements )
            self.__frameOriginalData['_original__elementsIndex']           = copy.deepcopy( self.__elementsIndex )
            self.__frameOriginalData['_original__numberOfAtomsPerElement'] = copy.deepcopy( self.__numberOfAtomsPerElement )
        # broadcast to constraints
        self.__broadcaster.broadcast("update elements indexes")

        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IS
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET.
#        # set mustSave flag
#        self.__mustSave = True

    def set_names_index(self, names=None):
        """
        Set names and namesIndex list, assigning a name to each atom.

        :Parameters:
            #. names (None, list): The names list. If None is given, names
            will be generated automatically by parsing pdbparser instance.
        """
        if names is None:
            names = self.__pdb.names
        else:
            assert isinstance(names, (list,set,tuple)), LOGGER.error("names must be a list of indexes")
            assert len(names)==self.self.__pdb.numberOfAtoms, LOGGER.error("names have the same length as pdb")
        # set all atoms names
        self.__allNames = names
        # get atom names
        self.__names = sorted(set(self.__allNames))
        # get namesIndex
        lut = dict(zip(self.__names,range(len(self.__names))))
        self.__namesIndex = np.array([lut[n] for n in self.__allNames], dtype=INT_TYPE)
        # number of atoms per name
        self.__numberOfAtomsPerName = {}
        for n in self.__allNames:
            if not n in self.__numberOfAtomsPerName:
                self.__numberOfAtomsPerName[n] = 0
            self.__numberOfAtomsPerName[n] += 1
        # save data to repository
        if self.__repository is not None:
            self.__repository.update_file(value=self.__allNames, relativePath=os.path.join(self.__usedFrame,'_Engine__allNames'))
            self.__repository.update_file(value=self.__names, relativePath=os.path.join(self.__usedFrame,'_Engine__names'))
            self.__repository.update_file(value=self.__namesIndex, relativePath=os.path.join(self.__usedFrame,'_Engine__namesIndex'))
            self.__repository.update_file(value=self.__numberOfAtomsPerName, relativePath=os.path.join(self.__usedFrame,'_Engine__numberOfAtomsPerName'))
            # save original data
            self.__repository.update_file(value=self.__allNames, relativePath=os.path.join(self.__usedFrame,'_original__allNames'))
            self.__repository.update_file(value=self.__names, relativePath=os.path.join(self.__usedFrame,'_original__names'))
            self.__repository.update_file(value=self.__namesIndex, relativePath=os.path.join(self.__usedFrame,'_original__namesIndex'))
            self.__repository.update_file(value=self.__numberOfAtomsPerName, relativePath=os.path.join(self.__usedFrame,'_original__numberOfAtomsPerName'))
            self.__frameOriginalData['_original__allNames']             = None
            self.__frameOriginalData['_original__names']                = None
            self.__frameOriginalData['_original__namesIndex']           = None
            self.__frameOriginalData['_original__numberOfAtomsPerName'] = None
        else:
            self.__frameOriginalData['_original__allNames']             = copy.deepcopy( self.__allNames )
            self.__frameOriginalData['_original__names']                = copy.deepcopy( self.__names )
            self.__frameOriginalData['_original__namesIndex']           = copy.deepcopy( self.__namesIndex )
            self.__frameOriginalData['_original__numberOfAtomsPerName'] = copy.deepcopy( self.__numberOfAtomsPerName )
        # broadcast to constraints
        self.__broadcaster.broadcast("update names indexes")

        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IF
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET.
#        # set mustSave flag
#        self.__mustSave = True

    def visualize(self, frame=None, commands=None, foldIntoBox=False, boxToCenter=False,
                        boxWidth=2, boxStyle="solid", boxColor="yellow",
                        bgColor="black", displayParams=None,
                        representationParams="Lines", otherParams=None):
        """
        Visualize the last configuration using pdbparser visualize_vmd method.

        :Parameters:
            #. frame (None, string): The frame to visualize. If None, used frame
               will be visualized. If given, frame must be created in repostory.
            #. commands (None, list, tuple): List of commands to pass
               upon calling vmd.
            #. foldIntoBox (boolean): Whether to fold all atoms into
               PeriodicBoundaries box before visualization. If boundary
               conditions are InfiniteBoundaries then nothing will be done.
            #. boxToCenter (boolean): Translate box center to atom coordinates
               center.
            #. boxWidth (number): Visualize the simulation box by giving the
               lines width. If 0 or boundary conditions are InfiniteBoundaries
               then nothing is visualized.
            #. boxStyle (str): The box line style, it can be either solid or
               dashed.  If boundary conditions are InfiniteBoundaries then
               nothing will be done.
            #. boxColor (str): Choose the simulation box color. If boundary
               conditions are InfiniteBoundaries then  nothing will be done.
               available colors are:\n
               blue, red, gray, orange, yellow, tan, silver, green,
               white, pink, cyan, purple, lime, mauve, ochre, iceblue,
               black, yellow2, yellow3, green2, green3, cyan2, cyan3, blue2,
               blue3, violet, violet2, magenta, magenta2, red2, red3,
               orange2, orange3.
            #. bgColor (str): Set visualization background color.
            #. displayParams(None, dict): Set display parameters.
               If None is given, default parameters will be applied.
               If dictionary is given, the following keys can be used.\n
               * 'depth cueing' (default True): Set the depth cueing flag.
               * 'cue density' (default 0.1): Set the depth density.
               * 'cue mode' (default 'Exp'): Set the depth mode among 'linear',
                 'Exp' and 'Exp2'.
            #. representationParams(str): Set representation method among
               the following:\n
               Lines, Bonds, DynamicBonds, HBonds, Points, VDW, CPK, Licorice,
               Beads, Dotted, Solvent.\n
               Add parameters accordingly if needed like the following.\n
               * Points representation accept only size parameter
                 e.g. 'Points 5'
               * CPK representation can accept respectively 4 parameters as the
                 following 'Sphere Scale', 'Bond Radius', 'Sphere Resolution',
                 'Bond Resolution' e.g. 'CPK 1.0 0.2 50 50'
               * VDW representation can accept respectively 2 parameters as
                 the following 'Sphere Scale', 'Sphere Resolution'
                 e.g. 'VDW 0.7 100'
            #. otherParams(None, list, set, tuple): Any other parameters
               in a form of a list of strings.\n
               e.g. ['display resize 700 700', 'rotate x to 45',
               'scale to 0.02', 'axes location off']
        """
        # check frame
        if frame is None:
            frame = self.usedFrame
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
        assert not isMultiframe, LOGGER.error("Given frame '%s' is a multiframe. Visualize is only possible with a normal frame or a subframe."%(frame,))
        # check boxWidth argument
        assert is_integer(boxWidth), LOGGER.error("boxWidth must be an integer")
        boxWidth = int(boxWidth)
        assert boxWidth>=0, LOGGER.error("boxWidth must be a positive")
        # check boxColor argument
        colors = ['blue', 'red', 'gray', 'orange', 'yellow', 'tan', 'silver', 'green',
                  'white', 'pink', 'cyan', 'purple', 'lime', 'mauve', 'ochre', 'iceblue',
                  'black', 'yellow2', 'yellow3', 'green2', 'green3', 'cyan2', 'cyan3', 'blue2',
                  'blue3', 'violet', 'violet2', 'magenta', 'magenta2', 'red2', 'red3',
                  'orange2','orange3']
        assert boxColor in colors, LOGGER.error("boxColor is not a recognized color name among %s"%str(colors))
        assert boxStyle in ('solid', 'dashed'), LOGGER.error("boxStyle must be either 'solid' or 'dashed'")
        # check representation argument
        reps = ["Lines","Bonds","DynamicBonds","HBonds","Points","VDW","CPK",
                "Licorice","Beads","Dotted","Solvent"]
        assert len(representationParams), "representation parameters must at least contain a method of representation"
        reprParams = representationParams.split()
        reprMethod = reprParams[0]
        reprParams.pop(0)
        reprParams = " ".join(reprParams)
        assert reprMethod in reps, LOGGER.error("representation method is not a recognized among allowed ones %s"%str(reps))
        # get needed data for visualization
        if frame == self.usedFrame:
            _pdb = self.__pdb
            _bc  = self.__boundaryConditions
            _rc  = self.__realCoordinates
            _ac  = self._atomsCollector
        else:
            assert self.__repository is not None, LOGGER.error("Repository is not built yet. Save engine before attempting to visualize")
            assert self.__repository.is_repository_directory(frame), LOGGER.error("Frame is not found in repository. Try using set_used_frame('%s') prior to visualize"%(frame,))
            try:
                _pdb = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__pdb'))
                _bc  = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__boundaryConditions'))
                _rc  = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__realCoordinates'))
                _ac  = self.__repository.pull(relativePath=os.path.join(frame,'_atomsCollector'))
            except Exception as err:
                assert False, LOGGER.error("Unable to pull data to export pdb (%s)"%(str(err),))
        # create .tcl file
        (vmdfd, tclFile) = tempfile.mkstemp()
        # write tclFile
        tclFile += ".tcl"
        try:
            fd = open(tclFile, "w")
            # visualize box
            if boxWidth>0 and isinstance(_bc, PeriodicBoundaries):
                if foldIntoBox and boxToCenter:
                    foldIntoBox = False
                    LOGGER.fixed("foldIntoBox and boxToCenter cannot both be set to True. foldIntoBox is reset to False.")
                try:
                    X,Y,Z = _bc.get_vectors()
                    lines = []
                    ### Z=0 plane
                    # 000 -> 100
                    lines.append( [0.,0.,0.,  X[0],X[1],X[2]] )
                    # 000 -> 010
                    lines.append( [0.,0.,0.,  Y[0],Y[1],Y[2]] )
                    # 100 -> 110
                    lines.append( [X[0],X[1],X[2],  X[0]+Y[0],X[1]+Y[1],X[2]+Y[2]] )
                    # 110 -> 010
                    lines.append( [X[0]+Y[0],X[1]+Y[1],X[2]+Y[2],  Y[0],Y[1],Y[2]] )
                    ### Z=1 plane
                    # 001 -> 101
                    lines.append( [Z[0],Z[1],Z[2],  Z[0]+X[0],Z[1]+X[1],Z[2]+X[2]] )
                    # 001 -> 011
                    lines.append( [Z[0],Z[1],Z[2],  Z[0]+Y[0],Z[1]+Y[1],Z[2]+Y[2]] )
                    # 101 -> 111
                    lines.append( [Z[0]+X[0],Z[1]+X[1],Z[2]+X[2],  X[0]+Y[0]+Z[0],X[1]+Y[1]+Z[1],X[2]+Y[2]+Z[2]] )
                    # 111 -> 011
                    lines.append( [X[0]+Y[0]+Z[0],X[1]+Y[1]+Z[1],X[2]+Y[2]+Z[2],  Z[0]+Y[0],Z[1]+Y[1],Z[2]+Y[2]] )
                    ### Z=1 verticals
                    # 000 -> 001
                    lines.append( [0.,0.,0.,  Z[0],Z[1],Z[2]] )
                    # 100 -> 101
                    lines.append( [X[0],X[1],X[2],  X[0]+Z[0],X[1]+Z[1],X[2]+Z[2]] )
                    # 010 -> 011
                    lines.append( [Y[0],Y[1],Y[2],  Y[0]+Z[0],Y[1]+Z[1],Y[2]+Z[2]] )
                    # 110 -> 111
                    lines.append( [X[0]+Y[0],X[1]+Y[1],X[2]+Y[2],  X[0]+Y[0]+Z[0],X[1]+Y[1]+Z[1],X[2]+Y[2]+Z[2]] )
                    # translate box
                    if boxToCenter:
                        bc = (X+Y+Z)/2.
                        cc = np.sum(_rc, axis=0)/_rc.shape[0]
                        tv = cc-bc
                        for idx, line in enumerate(lines):
                            lines[idx] = [item+tv[i%3] for i,item in enumerate(line)]
                    # write box
                    fd.write("draw color %s\n"%(boxColor,))
                    for l in lines:
                        fd.write( "draw line {%s %s %s} {%s %s %s} width %s style %s\n"%(l[0],l[1],l[2],l[3],l[4],l[5],boxWidth,boxStyle,) )
                except:
                    LOGGER.warn("Unable to write simulation box .tcl script for visualization.")
            # representation
            fd.write("mol delrep 0 top\n")
            fd.write("mol representation %s %s\n"%(reprMethod, reprParams) )
            fd.write("mol delrep 0 top\n")
            fd.write('mol addrep top\n')
            # display parameters
            if displayParams is None:
                displayParams = {}
            depthCueing = displayParams.get("depth cueing", True)
            cueDensity  = displayParams.get("cue density",  0.1)
            cueMode     = displayParams.get("cue mode",  'Exp')
            assert bgColor in colors, LOGGER.error("display background color is not a recognized color name among %s"%str(colors))
            assert depthCueing in [True, False], LOGGER.error("depth cueing must be boolean")
            assert is_number(cueDensity), LOGGER.error("cue density must be a number")
            assert cueMode in ['linear','Exp','Exp2'], LOGGER.error("cue mode must be either 'linear','Exp' or 'Exp2'")
            fd.write('color Display Background %s \n'%bgColor)
            if depthCueing is True:
                depthCueing = 'on'
            else:
                depthCueing = 'off'
            fd.write('display depthcue %s \n'%depthCueing)
            fd.write('display cuedensity  %s \n'%cueDensity)
            fd.write('display cuemode  %s \n'%cueMode)
            # other parameters
            if otherParams is None:
                otherParams = []
            for op in otherParams:
                fd.write('%s \n'%op)
            fd.close()
            coords = _rc
            # MUST TRANSFORM TO PDB COORDINATES SYSTEM FIRST
            if foldIntoBox and isinstance(_bc, PeriodicBoundaries):
                coords = _bc.fold_real_array(_rc)
            # copy pdb if atoms where amputated
            if len(_ac.indexes):
                indexes = sorted(set(_pdb.indexes)-set(_ac.indexes))
                pdb = _pdb.get_copy(indexes=indexes)
            else:
                pdb = _pdb
            pdb.visualize(commands=commands, coordinates=coords, startupScript=tclFile)
        except Exception as err:
            print("unable to visualize (%s)"%err)
            # remove tclFile
            os.remove(tclFile)
        else:
            # remove tclFile
            os.remove(tclFile)


    def add_constraints(self, constraints, toAllSubframes=False):
        """
        Add constraints to the engine. If used frame is a normal frame, all
        other normal frames will have the constraints added. If used frame
        is a subframe, all other subframes of the same multiframe will have
        the experimental constraints added to. But given non-experimental
        constraints will be only added to the used frame.

        :Parameters:
            #. constraints (Constraint, list, set, tuple): A constraint instance or
               list of constraints instances
            #. toAllSubframes (boolean): Whether to also add non-experimental
               constraints to all other multiframe subframes in case engine
               used frame is a multiframe subframe.
        """
        # check arguments
        assert isinstance(toAllSubframes, bool), LOGGER.usage("toAllSubframes must be boolean")
        if isinstance(constraints,(list,set,tuple)):
            constraints = list(constraints)
        else:
            constraints = [constraints]
        ## check all constraints and real constraint instance
        for c in constraints:
            assert isinstance(c, Constraint), LOGGER.error("Constraints must be a Constraint instance or a list of Constraint instances. None of the constraints have been added to the engine.")
            assert not c.is_in_engine(self), LOGGER.error("Constraint '%s' of unique id '%s' already exist in engine. None of the constraints have been added to the engine."%(c.__class__.__name__,c.constraintId))

        ## get used frame category
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(self.__usedFrame)
        ## copy engine just for testing purposes
        _engine = copy.deepcopy(self)
        _engine._Engine__constraints = self.constraints
        ## check singular constraints
        if isNormalFrame:
            _engine._Engine__constraints.extend(constraints)
            [object.__setattr__(_c, '_Constraint__engine',_engine) for _c in constraints]
            for _c in _engine._Engine__constraints:
                if isinstance(_c, SingularConstraint):
                    assert _c.is_singular(_engine), LOGGER.error("Only one instance of constraint '%s' is allowed in the same engine. None of the constraints have been added to the engine."%_c.__class__.__name__)
            LOGGER.info("Given constraints will be added to all normal frames in engine")

        else:
            assert self.__repository is not None, LOGGER.error("Adding constraints to multiframe is not allowed before building repository. Use engine save method first.")
            _multi  = self.__usedFrame.split(os.sep)[0]
            _frames = [os.path.join(_multi, frm) for frm in self.__frames[_multi]['frames_name']]
            for frm in _frames:
                _constraints = self.__repository.pull(relativePath=os.path.join(frm,'_Engine__constraints'))
                [object.__setattr__(_c, '_Constraint__engine',_engine) for _c in _constraints]
                _engine._Engine__constraints = _constraints
                for c in constraints:
                    assert not c.is_in_engine(_engine), LOGGER.error("Constraint '%s' of unique id '%s' already exist in frame '%s' engine. None of the constraints have been added to the engine."%(c.__class__.__name__,c.constraintId,frm))
                _engine._Engine__constraints.extend(constraints)
                [object.__setattr__(_c, '_Constraint__engine',_engine) for _c in constraints]
                # check constraints being singular
                if frm == self.usedFrame:
                    _notExpConsts = [_c for _c in getattr(_engine, '_Engine__constraints') if not isinstance(_c, ExperimentalConstraint)]
                    for _c in _engine._Engine__constraints:
                        if isinstance(_c, SingularConstraint):
                            assert _c.is_singular(_engine), LOGGER.error("Only one instance of constraint '%s' is allowed in the same engine for frame '%s'. None of the constraints have been added to the engine."%(_c.__class__.__name__,frm))
                else:
                    _expConsts = [_c for _c in getattr(_engine, '_Engine__constraints') if isinstance(_c, ExperimentalConstraint)]
                    for _c in _expConsts:
                        if isinstance(_c, SingularConstraint):
                            assert _c.is_singular(_engine), LOGGER.error("Only one instance of constraint '%s' is allowed in the same engine for frame '%s'. None of the constraints have been added to the engine."%(_c.__class__.__name__,frm))
            if toAllSubframes:
                LOGGER.info("All given constraints will be added all of %i subframes of multiframe '%s'"%(len(self.__frames[_multi]['frames_name']),_multi))
            else:
                LOGGER.info("All given constraints will be added to the used subframe '%s'. But experimental constraints only will be added to all other '%i' subframes of multiframe '%s'"%(self.__usedFrame,len(self.__frames[_multi]['frames_name'])-1,_multi))
        ## add constraints to current used frame
        for c in constraints:
            c._Constraint__engine = None
            # add engine to constraint
            c._set_engine(self)
            # add to broadcaster listeners list
            self.__broadcaster.add_listener(c)
            # add constraint to engine
            self.__constraints.append(c)
            # broadcast 'engine changed' to constraint this will create some of constraints files if repository is not None
            c.listen("engine set")
        if self.__repository is not None:
            # add constraints to used frame
            for c in constraints:
                cp = os.path.join(self.__usedFrame, 'constraints', c.constraintName)
                self.__repository.add_directory( cp )
                for dname in c.FRAME_DATA:
                    value = c.__dict__[dname]
                    #self.__repository.dump(value=value, relativePath=os.path.join(cp,dname), replace=True) # set replace to True because of early listen method call
                    self._dump_to_repository(value=value, relativePath=os.path.join(cp,dname))
            # Add constraint to all traditional frames
            if isNormalFrame:
                self.__repository.update_file(value=self.__constraints, relativePath='_Engine__constraints')
                for frame in self.__frames:
                    if self.__frames[frame] is not None:
                        continue
                    if frame == self.__usedFrame:
                        continue
                    # copy constraints directory
                    for c in constraints:
                        tpath = os.path.join(self.__usedFrame, 'constraints', c.constraintName)
                        cpath = os.path.join(frame, 'constraints', c.constraintName)
                        success, error = self.__repository.copy_directory(relativePath=tpath, newRelativePath=cpath, raiseError=False)
                        assert success, LOGGER.error(error)
            else:
                # update _Engine__constraints file
                self.__repository.update_file(value=self.__constraints, relativePath=os.path.join(self.__usedFrame,'_Engine__constraints'))
                if toAllSubframes:
                    toAddConstraints = constraints
                    toAddConstsName  = [_c.constraintName for _c in toAddConstraints]
                else:
                    toAddConstraints = [_c for _c in constraints if isinstance(_c, ExperimentalConstraint)]
                    toAddConstsName  = [_c.constraintName for _c in toAddConstraints]
                if len(toAddConstraints):
                    for subframe in self.__frames[_multi]['frames_name']:
                        frame = os.path.join(_multi,subframe)
                        if frame == self.__usedFrame:
                            continue
                        _constraints = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__constraints'))
                        _engine._Engine__constraints = _constraints
                        [object.__setattr__(_c, '_Constraint__engine',_engine) for _c in _constraints]
                        [object.__setattr__(_c, '_Constraint__engine',None) for _c in toAddConstraints]
                        [_c._set_engine(_engine) for _c in toAddConstraints]
                        _engine._Engine__constraints.extend(toAddConstraints)
                        # copy constraints directory
                        for c, cn in zip(toAddConstraints,toAddConstsName):
                            tpath = os.path.join(self.__usedFrame, 'constraints', cn)
                            cpath = os.path.join(frame, 'constraints', c.constraintName)
                            success, error = self.__repository.copy_directory(relativePath=tpath, newRelativePath=cpath, raiseError=False)
                            assert success, LOGGER.error(error)
                        # update _Engine__constraints file
                        self.__repository.update_file(value=_engine._Engine__constraints, relativePath=os.path.join(frame,'_Engine__constraints'))


    def remove_constraints(self, constraints, toAllSubframes=False):
        """
        Remove constraints from engine list of constraints.

        :Parameters:
            #. constraints (Constraint, list, set, tuple): A constraint
               instance or list of constraints instances.
            #. toAllSubframes (boolean): Whether to also remove non-experimental
               constraints from all other multiframe subframes in case engine
               used frame is a multiframe subframe.
        """
        assert isinstance(toAllSubframes, bool), LOGGER.usage("toAllSubframes must be boolean")
        if not len(self.__constraints):
            LOGGER.warn("No constraint found in engine")
            return
        if isinstance(constraints,(list,set,tuple)):
            constraints = list(constraints)
        else:
            constraints = [constraints]
        for c in constraints:
            assert isinstance(c, Constraint), LOGGER.error("Constraints must be a Constraint instance or a list of Constraint instances. None of the constraints have been removed engine.")
        removingIds = [c.constraintId for c in constraints]
        ## get used frame category
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(self.__usedFrame)
        if isSubframe:
            assert self.__repository is not None, LOGGER.error("Removing constraints to multiframe is not allowed before building repository. Use engine save method first.")
        ## loop engine constraints
        engineConstraints  = []
        removedConstraints = []
        for c in self.__constraints:
            if c.constraintId in removingIds:
                c._Constraint__engine = None
                self.__broadcaster.remove_listener(c)
                removedConstraints.append(c)
            else:
                engineConstraints.append(c)
        if not len(removedConstraints):
            LOGGER.warn("None of the given constraints were found in frame '%s' engine list of constraints"%(self.__usedFrame,))
        else:
            self.__constraints = engineConstraints
            if len(removedConstraints)<len(constraints):
                _removedIds = [c.constraintId for c in removedConstraints]
                for c in constraints:
                    if c.constraintId not in _removedIds:
                        LOGGER.warn("Constraint '%s' of id '%s' is not found in frame '%s' engine list of constraints"%(c.__class__.__name__, c.constraintId, self.__usedFrame,))
        # save changes to repository
        if self.__repository is not None:
            if len(removedConstraints):
                self.__repository.update_file(value=self.__constraints, relativePath=os.path.join(self.__usedFrame, '_Engine__constraints'))
                for c in removedConstraints:
                    cp = os.path.join(self.__usedFrame, 'constraints', c.constraintName)
                    self.__repository.remove_directory(relativePath=cp, clean=True)
            if isNormalFrame:
                for c in removedConstraints:
                    for fn in self.__frames:
                        if self.__frames[fn] is not None:
                            continue
                        if fn == self.__usedFrame:
                            continue
                        cp = os.path.join(fn, 'constraints', c.constraintName)
                        self.__repository.remove_directory(relativePath=cp, clean=True)
            else:
                if not toAllSubframes:
                    constraints = [_c for _c in constraints if isinstance(_c, ExperimentalConstraint)]
                    removingIds = [c.constraintId for c in constraints]
                if len(constraints):
                    _multi = self.__usedFrame.split(os.sep)[0]
                    for subframe in self.__frames[_multi]['frames_name']:
                        frame = os.path.join(_multi,subframe)
                        if frame == self.__usedFrame:
                            continue
                        _constraints       = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__constraints'))
                        engineConstraints  = []
                        removedConstraints = []
                        for c in _constraints:
                            if c.constraintId in removingIds:
                                removedConstraints.append(c)
                            else:
                                engineConstraints.append(c)
                        if not len(removedConstraints):
                            LOGGER.warn("None of the given constraints were found in frame '%s' engine list of constraints"%(frame,))
                        else:
                            if len(removedConstraints)<len(_constraints):
                                _removedIds = [c.constraintId for c in removedConstraints]
                                for c in _constraints:
                                    if c.constraintId not in _removedIds:
                                        LOGGER.warn("Constraint '%s' of id '%s' is not found in frame '%s' engine list of constraints"%(c.__class__.__name__, c.constraintId,frame,))
                            # update repostory
                            self.__repository.update_file(value=engineConstraints, relativePath=os.path.join(frame, '_Engine__constraints'))
                            for c in removedConstraints:
                                cp = os.path.join(frame, 'constraints', c.constraintName)
                                self.__repository.remove_directory(relativePath=cp, clean=True)


    def reset_constraints(self):
        """ Reset constraints flags. """
        for c in self.__constraints:
            c.reset_constraint(reinitialize=True)
        # update constraints in repository used frame only
        #if self.__repository is not None:
        #    #self.__repository.dump(value=self, relativePath='.', name='engine', replace=True)
        #    self.__repository.dump(value=self, relativePath='engine', replace=True)
        #    for c in self.__constraints:
        #        cp = os.path.join(self.__usedFrame, 'constraints', c.constraintName)
        #        for dname in c.FRAME_DATA:
        #            value = c.__dict__[dname]
        #            #name  = dname.split('__')[1]
        #            name = dname
        #            #self.__repository.dump(value=value, relativePath=cp, name=name, replace=True)
        #            self.__repository.dump(value=value, relativePath=os.path.join(cp,name), replace=True)
        if self.__repository is not None:
            #self.__repository.dump(value=self, relativePath='.', name='engine', replace=True)
            #self.__repository.update_file(value=self, relativePath='engine') ### NOT NEEDED ANYMORE SINCE __constraints is a MULTIFRAME_DATA
            for c in self.__constraints:
                if isinstance(self.__usedFrame, basestring):
                    _framep = [self.__usedFrame]
                else:
                    _framep = [os.path.join(self.__usedFrame['name'], i) for i in self.__usedFrame['frames_name']]
                for dname in c.FRAME_DATA:
                    value = c.__dict__[dname]
                    for _fp in _framep:
                        rp = os.path.join(_fp, 'constraints', c.constraintName, dname)
                        self.__repository.update_file(value=value, relativePath=rp)
#        # set mustSave flag
#        self.__mustSave = True

    def reset_engine(self):
        """ Re-initialize engine and resets constraints flags and data. """
        self._reinit_engine()
        # reset constraints flags
        self.reset_constraints()
#        # set mustSave flag
#        self.__mustSave = True

    def compute_total_standard_error(self, constraints, current="standardError"):
        """
        Compute the total standard error as the sum of all constraints'
        standard error.

        .. math::
            \\chi^{2} = \\sum \\limits_{i}^{N} (\\frac{stdErr_{i}}{variance_{i}})^{2}

        Where:\n
        :math:`variance_{i}` is the variance value of the constraint i. \n
        :math:`stdErr_{i}` the standard error of the constraint i defined
        as :math:`\\sum \\limits_{j}^{points} (target_{i,j}-computed_{i,j})^{2} = (Y_{i,j}-F(X_{i,j}))^{2}` \n

        :Parameters:
            #. constraints (list): All constraints used to calculate total
               totalStandardError.
            #. current (str): which standard error to use. can be anything like
               standardError, afterMoveStandardError or
               amputatedStandardError, etc.

        :Returns:
            #. totalStandardError (list): The computed total total standard
               error.
        """
        TSE = []
        for c in constraints:
            SD = getattr(c, current)
            assert SD is not None, LOGGER.error("constraint %s %s is not computed yet. Try to initialize constraint"%(c.__class__.__name__,current))
            TSE.append(SD/c.varianceSquared)
        return np.sum(TSE)

    def update_total_standard_error(self):
        """
        Compute and set engine's total totalStandardError of used constraints.
        """
        # get and initialize used constraints
        _usedConstraints, _constraints, _rigidConstraints = self.initialize_used_constraints()
        # compute totalStandardError
        self.__totalStandardError = self.compute_total_standard_error(_constraints, current="standardError")

    def get_used_constraints(self, sortConstraints=False):
        """
        Parses all engine's constraints and returns different lists of
        the active (used) ones.

        :Parameters:
            #. sortConstraints (boolean): Whether to sort used constraints
               according to their computation cost property. This is can
               minimize computations and enhance performance by computing
               less costly constraints first.

        :Returns:
            #. usedConstraints (list): All types of active constraints
               instances that are used at engine's runtime.
            #. constraints (list): All active constraints instance among
               usedConstraints list that will contribute to engine's
               totalStandardError.
            #. RigidConstraint (list): All active RigidConstraint constraints
               instance among usedConstraints list that won't contribute
               engine's totalStandardError.
        """
        assert isinstance(sortConstraints, bool), LOGGER.error("sortConstraints must be boolean")
        # sort constraints
        if sortConstraints:
            indexes = np.argsort( [c.computationCost for c in self.__constraints] )
            allEngineConstraints = [self.__constraints[idx] for idx in indexes]
        else:
            allEngineConstraints = self.__constraints
        # get used constraints
        usedConstraints = []
        for c in allEngineConstraints:
            if c.used:
                usedConstraints.append(c)
        # get rigidConstraints list
        rigidConstraints = []
        constraints = []
        for c in usedConstraints:
            if isinstance(c, RigidConstraint):
                rigidConstraints.append(c)
            else:
                constraints.append(c)
        # return constraints
        return usedConstraints, constraints, rigidConstraints

    def initialize_used_constraints(self, force=False, sortConstraints=False):
        """
        Calls get_used_constraints method, re-initializes constraints when
        needed and return them all.

        :Parameters:
            #. force (boolean): Whether to force initializing constraints
               regardless of their state.
            #. sortConstraints (boolean): Whether to sort used constraints
               according to their computation cost property. This is can
               minimize computations and enhance performance by computing less
               costly constraints first.

        :Returns:
            #. usedConstraints (list): All types of active constraints
               instances that are used at engine's runtime.
            #. constraints (list): All active constraints instance among
               usedConstraints list that will contribute to engine's
               totalStandardError.
            #. RigidConstraint (list): All active RigidConstraint constraints
               instance among usedConstraints list that won't contribute
               engine's totalStandardError.
        """
        assert isinstance(force, bool), LOGGER.error("force must be boolean")
        # get used constraints
        usedConstraints, constraints, rigidConstraints = self.get_used_constraints(sortConstraints=sortConstraints)
        # initialize out-of-dates constraints
        for c in usedConstraints:
            if c.state != self.__state or force:
                LOGGER.info("@%s Initializing constraint data '%s'"%(self.__usedFrame, c.__class__.__name__))
                c.compute_data()
                c.set_state(self.__state)
                if c.originalData is None:
                    c._set_original_data(c.data)
        # return constraints
        return usedConstraints, constraints, rigidConstraints

    def __runtime_get_number_of_steps(self, numberOfSteps):
        # check numberOfSteps
        assert is_integer(numberOfSteps), LOGGER.error("numberOfSteps must be an integer")
        assert numberOfSteps<=maxint, LOGGER.error("number of steps must be smaller than maximum integer number allowed by the system '%i'"%maxint)
        assert numberOfSteps>=0, LOGGER.error("number of steps must be positive")
        # return
        return int(numberOfSteps)

    def __runtime_get_save_engine(self, saveFrequency, frame):
        # check saveFrequency
        assert is_integer(saveFrequency), LOGGER.error("saveFrequency must be an integer")
        if saveFrequency is not None:
            assert is_integer(saveFrequency), LOGGER.error("saveFrequency must be an integer")
            assert saveFrequency>=0, LOGGER.error("saveFrequency must be positive")
            saveFrequency = int(saveFrequency)
        if saveFrequency == 0:
            saveFrequency = None
        # check frame
        if frame is None:
            frame = self.__usedFrame
        frame = str(frame)
        # set used frame
        self.set_used_frame(frame)
        # return
        return saveFrequency, frame

    def __runtime_get_save_xyz(self, xyzFrequency, xyzPath):
        # check saveFrequency
        if xyzFrequency is not None:
            assert is_integer(xyzFrequency), LOGGER.error("xyzFrequency must be an integer")
            assert xyzFrequency>=0, LOGGER.error("xyzFrequency must be positive")
            xyzFrequency = int(xyzFrequency)
        if xyzFrequency == 0:
            xyzFrequency = None
        # check xyzPath
        assert isinstance(xyzPath, basestring), LOGGER.error("xyzPath must be a string")
        xyzPath = str(xyzPath)
        # return
        return xyzFrequency, xyzPath


##########################################################################################
##########################################################################################
#### NEW RUN METHOD IS PARTITIONED AND SPLIT INTO DIFFERENT __on_runtime_step METHODS ####

    def __on_runtime_step_select_group(self, _coordsBeforeMove, _movedRealCoordinates, _moveTried):
        # get group
        self.__lastSelectedGroupIndex = self.__groupSelector.select_index()
        self._RT_selectedGroup = self.__groups[self.__lastSelectedGroupIndex]
        # get move generator
        self._RT_moveGenerator = self._RT_selectedGroup.moveGenerator
        # remove generator
        if isinstance(self._RT_moveGenerator, RemoveGenerator):
            _movedRealCoordinates = None
            _movedBoxCoordinates  = None
            self._RT_groupAtomsIndexes    = self._RT_moveGenerator.pick_from_list(self)
            notCollectedAtomsIndexes      = np.array(self._atomsCollector.are_not_collected(self._RT_groupAtomsIndexes), dtype=bool)
            self._RT_groupAtomsIndexes    = self._RT_groupAtomsIndexes[ notCollectedAtomsIndexes ]
            self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
            _coordsBeforeMove             = None
        # move generator
        else:
            # get atoms indexes
            self._RT_groupAtomsIndexes = self._RT_selectedGroup.indexes
            notCollectedAtomsIndexes   = np.array(self._atomsCollector.are_not_collected(self._RT_groupAtomsIndexes), dtype=bool)
            self._RT_groupAtomsIndexes = self._RT_groupAtomsIndexes[ notCollectedAtomsIndexes ]
            # check if all group atoms are collected
            if not len(self._RT_groupAtomsIndexes):
                self._RT_groupRelativeIndexes = self._RT_groupAtomsIndexes
                _coordsBeforeMove             = np.array([], dtype=self.__realCoordinates.dtype).reshape((0,3))
            # get group atoms coordinates before applying move
            elif isinstance(self._RT_moveGenerator, SwapGenerator):
                if len(self._RT_groupAtomsIndexes) == self._RT_moveGenerator.swapLength:
                    self._RT_groupAtomsIndexes    = self._RT_moveGenerator.get_ready_for_move(engine=self,  groupAtomsIndexes=self._RT_groupAtomsIndexes)
                    notCollectedAtomsIndexes      = np.array(self._atomsCollector.are_not_collected(self._RT_groupAtomsIndexes), dtype=bool)
                    self._RT_groupAtomsIndexes    = self._RT_groupAtomsIndexes[ notCollectedAtomsIndexes ]
                    self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                    _coordsBeforeMove = np.array(self.__realCoordinates[self._RT_groupRelativeIndexes], dtype=self.__realCoordinates.dtype)
                else:
                    self._RT_groupAtomsIndexes    = np.array([], dtype=self._RT_selectedGroup.indexes.dtype)
                    self._RT_groupRelativeIndexes = self._RT_groupAtomsIndexes
                    _coordsBeforeMove             = np.array([], dtype=self.__realCoordinates.dtype).reshape((0,3))
            elif _coordsBeforeMove is None or not self.__groupSelector.isRecurring:
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                _coordsBeforeMove = np.array(self.__realCoordinates[self._RT_groupRelativeIndexes], dtype=self.__realCoordinates.dtype)
            elif self.__groupSelector.explore:
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                if _moveTried:
                    _coordsBeforeMove = _movedRealCoordinates
            elif not self.__groupSelector.refine:
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                _coordsBeforeMove = np.array(self.__realCoordinates[self._RT_groupRelativeIndexes], dtype=self.__realCoordinates.dtype)
            ## WHEN AT THIS POINT PROPERTIES ARE GroupSelector (RecursiveGroupSelector) explore (False) refine (True).
            #else:
            #    m  = "Unknown fitting mode, unable to get coordinates before applying move. "
            #    m += "GroupSelector (%s) explore (%s) refine (%s). "%(self.__groupSelector.__class__.__name__, self.__groupSelector.explore, self.__groupSelector.refine)
            #    m += "Group (%s) index (%s) len (%s). "%(self._RT_selectedGroup.__class__.__name__, self.__lastSelectedGroupIndex, len(self._RT_groupAtomsIndexes))
            #    m += "MoveGenerator (%s) index (%s). "%(self._RT_moveGenerator.__class__.__name__, self.__lastSelectedGroupIndex,)
            #    raise Exception(LOGGER.critical(m))
            # compute moved coordinates
            if len(_coordsBeforeMove):
                _movedRealCoordinates = self._RT_moveGenerator.move(_coordsBeforeMove)
                _movedBoxCoordinates  = transform_coordinates(transMatrix=self.__reciprocalBasisVectors , coords=_movedRealCoordinates)
            else:
                _movedRealCoordinates = _coordsBeforeMove
                _movedBoxCoordinates  = _coordsBeforeMove
        # return
        return _coordsBeforeMove, _movedRealCoordinates, _movedBoxCoordinates


    def __on_runtime_step_try_remove(self, _constraints, _usedConstraints, _rigidConstraints):
        ###################################### reject remove atoms #####################################
        self.__tried      += 1
        self.__removed[0] += 1.
        self.__removed[2]  = self.__removed[1]/self.__removed[0]
        _rejectRemove = False
        for c in _constraints:
            c.compute_as_if_amputated(realIndex=self._RT_groupAtomsIndexes, relativeIndex=self._RT_groupRelativeIndexes)
        ################################ compute new totalStandardError ################################
        oldStandardError      = self.compute_total_standard_error(_constraints, current="standardError")
        newTotalStandardError = self.compute_total_standard_error(_constraints, current="amputationStandardError")
        if newTotalStandardError > self.__totalStandardError:
            if generate_random_float() > self.__tolerance:
                _rejectRemove = True
            else:
                self.__tolerated += 1
                self.__totalStandardError  = newTotalStandardError
        else:
            self.__totalStandardError = newTotalStandardError
        ################################# reject tried remove #################################
        if _rejectRemove:
            # set selector move rejected
            self.__groupSelector.move_rejected(self.__lastSelectedGroupIndex)
            for c in _constraints:
                c.reject_amputation(realIndex=self._RT_groupAtomsIndexes, relativeIndex=self._RT_groupRelativeIndexes)
            # log tried move rejected
            LOGGER.rejected("@%s Tried remove %i is rejected"%(self.__usedFrame, self.__generated))
        ################################# accept tried remove #################################
        else:
            self.__accepted += 1
            self.__removed[1] += 1.
            self.__removed[2]  = self.__removed[1]/self.__removed[0]
            # set selector move accepted
            self.__groupSelector.move_accepted(self.__lastSelectedGroupIndex)
            # constraints reject move
            for c in _usedConstraints:
                c.accept_amputation(realIndex=self._RT_groupAtomsIndexes, relativeIndex=self._RT_groupRelativeIndexes)
            # collect atoms
            self._on_collector_collect_atom(realIndex = self._RT_groupAtomsIndexes[0])
            # log new successful move
            triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
            acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
            LOGGER.accepted("@%s Gen:%i - Tr:%i(%.3f%%) - Acc:%i(%.3f%%) - Rem:%i(%.3f%%) - Err:%.6f" %(self.__usedFrame, self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__removed[1], 100.*self.__removed[2], self.__totalStandardError))
        # return _moveTried and _rejectRemove flags
        return True, _rejectRemove


    def __on_runtime_step_try_move(self, _constraints, _usedConstraints, _rigidConstraints, _movedRealCoordinates, _movedBoxCoordinates):
        ########################### compute rigidConstraints ############################
        rejectMove      = False
        for c in _rigidConstraints:
            # compute before move
            c.compute_before_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes = self._RT_groupRelativeIndexes)
            # compute after move
            c.compute_after_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes = self._RT_groupRelativeIndexes, movedBoxCoordinates=_movedBoxCoordinates)
            # get rejectMove
            rejectMove = c.should_step_get_rejected(c.afterMoveStandardError)
            #print(c.__class__.__name__, c.standardError, c.afterMoveStandardError, rejectMove)
            if rejectMove:
                break
        _moveTried = not rejectMove
        ############################## reject move before trying ##############################
        if rejectMove:
            # rigidConstraints reject move
            for c in _rigidConstraints:
                c.reject_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
            # log generated move rejected before getting tried
            LOGGER.nottried("@%s Generated move %i is not tried"%(self.__usedFrame,self.__generated))
        ###################################### try move #######################################
        else:
            self.__tried += 1
            for c in _constraints:
                # compute before move
                c.compute_before_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
                # compute after move
                c.compute_after_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes = self._RT_groupRelativeIndexes, movedBoxCoordinates=_movedBoxCoordinates)
        ################################ compute new totalStandardError ################################
            newTotalStandardError = self.compute_total_standard_error(_constraints, current="afterMoveStandardError")
            #if len(_constraints) and (newTotalStandardError >= self.__totalStandardError):
            if newTotalStandardError > self.__totalStandardError:
                if generate_random_float() > self.__tolerance:
                    rejectMove = True
                else:
                    self.__tolerated += 1
                    self.__totalStandardError  = newTotalStandardError
            else:
                self.__totalStandardError = newTotalStandardError
        ################################## reject tried move ##################################
        if rejectMove:
            # set selector move rejected
            self.__groupSelector.move_rejected(self.__lastSelectedGroupIndex)
            if _moveTried:
                # constraints reject move
                for c in _constraints:
                    c.reject_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
                # log tried move rejected
                LOGGER.rejected("@%s Tried move %i is rejected"%(self.__usedFrame,self.__generated))
        ##################################### accept move #####################################
        else:
            self.__accepted  += 1
            # set selector move accepted
            self.__groupSelector.move_accepted(self.__lastSelectedGroupIndex)
            # constraints reject move
            for c in _usedConstraints:
                c.accept_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
            # set new coordinates
            self.__realCoordinates[self._RT_groupRelativeIndexes] = _movedRealCoordinates
            self.__boxCoordinates[self._RT_groupRelativeIndexes]  = _movedBoxCoordinates
            # log new successful move
            triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
            acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
            LOGGER.accepted("@%s Gen:%i - Tr:%i(%.3f%%) - Acc:%i(%.3f%%) - Rem:%i(%.3f%%) - Err:%.6f" %(self.__usedFrame,self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__removed[1], 100.*self.__removed[2],self.__totalStandardError))
        # return _moveTried and rejectMove flags
        return _moveTried, rejectMove

    def __on_runtime_step_save_engine(self, _saveFrequency, step, _frame, _usedConstraints, _lastSavedTotalStandardError):
        ##################################### save engine #####################################
        if _saveFrequency is not None:
            if not(step+1)%_saveFrequency:
                if _lastSavedTotalStandardError==self.__totalStandardError:
                    LOGGER.saved("@%s Save engine omitted because no improvement made since last save."%(self.__usedFrame,))
                else:
                    # update state
                    self.__state  = time.time()
                    for c in _usedConstraints:
                       #c.increment_tried()
                       c.set_state(self.__state)
                    # save engine
                    _lastSavedTotalStandardError = self.__totalStandardError
                    #self.save(_frame)
                    self.__runtime_save(_frame)
        return _lastSavedTotalStandardError


    def __on_runtime_step_save_xyz(self, _xyzFrequency, _xyzfd, step):
        ############################### dump coords to xyz file ###############################
        ## special care must be taken because once atoms are collected xyz files needs to adapt
        if _xyzFrequency is not None:
            if not(step+1)%_xyzFrequency:
                _xyzfd.write("%s\n"%self.numberOfAtoms)
                triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
                acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
                _xyzfd.write("Gen:%i - Tr:%i(%.3f%%) - Acc:%i(%.3f%%) - Rem:%i(%.3f%%) - Err:%.6f\n" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__removed[1], 100.*self.__removed[2],self.__totalStandardError))
                frame = [self.__allNames[idx]+ " " + "%10.5f"%self.__realCoordinates[idx][0] + " %10.5f"%self.__realCoordinates[idx][1] + " %10.5f"%self.__realCoordinates[idx][2] + "\n" for idx in self.__pdb.xindexes]
                _xyzfd.write("".join(frame))


    def run(self, numberOfSteps=100000,     sortConstraints=True,
                  saveFrequency=1000,       frame=None,
                  xyzFrequency=None,        xyzPath="trajectory.xyz",
                  restartPdb='restart.pdb', ncores=None):
        """
        Run stochastic fitting engine.

        :Parameters:
            #. numberOfSteps (integer): The number of steps to run.
            #. sortConstraints (boolean): Whether to sort used constraints
               according to their computation cost property. This is can
               minimize computations and enhance performance by computing less
               costly constraints first.
            #. saveFrequency (integer): Save engine every saveFrequency steps.
               Save will be omitted if no moves are accepted.
            #. xyzFrequency (None, integer): Save coordinates to .xyz file
               every xyzFrequency steps regardless if totalStandardError
               has decreased or not. If None is given, no .xyz file will be
               generated.
            #. xyzPath (string): Save coordinates to .xyz file.
            #. restartPdb (None, string): Export a pdb file of the last
               configuration at the end of the run. If None is given, no pdb
               file will be exported. If string is given, it should be the
               full path of the pdb file.
            #. ncores (None, integer): set the number of cores to use.
               If None is given, ncores will be set automatically to 1.
               This argument is only effective if fullrmc is compiled with
               openmp.
        """
        # make sure it's a normal frame
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(self.__usedFrame)
        assert isNormalFrame or isSubframe, LOGGER.error("Calling run is only allowed when used frame is a traditional frame or a subframe")
        # get arguments
        _numberOfSteps          = self.__runtime_get_number_of_steps(numberOfSteps)
        _saveFrequency, _frame  = self.__runtime_get_save_engine(saveFrequency, frame)
        _xyzFrequency, _xyzPath = self.__runtime_get_save_xyz(xyzFrequency, xyzPath)
        assert _frame == self.__usedFrame, LOGGER.error("Must save engine before changing frame.")
        if _saveFrequency<=_numberOfSteps:
            assert self.__repository is not None, LOGGER.error("Engine might be saving during this run but repository is not defined. Use Engine.save method before calling run method.")
        # set runtime ncores
        self.__set_runtime_ncores(ncores)
        # create xyz file
        _xyzfd = None
        if _xyzFrequency is not None:
            _xyzfd = open(_xyzPath, 'a')
        # set restartPdb
        if restartPdb is None:
            restartPdb = False
        else:
            assert isinstance(restartPdb, basestring), LOGGER.error("restartPdb must be None or a string")
            restartPdb = str(restartPdb)
            if not restartPdb.endswith('.pdb'):
                LOGGER.warn(".pdb appended to restartPdb '%s'"%restartPdb)
                restartPdb += ".pdb"
        # get and initialize used constraints
        _usedConstraints, _constraints, _rigidConstraints = self.initialize_used_constraints(sortConstraints=sortConstraints)
        if not len(_usedConstraints):
            LOGGER.warn("@%s No constraints are used. Configuration will be randomized"%self.__usedFrame)
        # runtime initialize group selector
        self.__groupSelector._runtime_initialize()
        # runtime initialize constraints
        [c._runtime_initialize() for c in _usedConstraints]
        # compute totalStandardError
        self.__totalStandardError = self.compute_total_standard_error(_constraints, current="standardError")
        # initialize useful arguments
        _engineStartTime             = time.time()
        _lastSavedTotalStandardError = self.__totalStandardError
        _coordsBeforeMove            = None
        _moveTried                   = False
        _rejectMove                  = False
        _rejectRemove                = False
        _movedRealCoordinates        = None
        _movedBoxCoordinates         = None
        # save whole engine if must be done
        if self.__mustSave: # Currently it is always False. will check and fix it later
            self.save()
        #   #####################################################################################   #
        #   #################################### RUN ENGINE #####################################   #
        LOGGER.info("Engine @%s started %i steps, total standard error is: %.6f"%( self.__usedFrame, _numberOfSteps, self.__totalStandardError) )
        for step in xrange(_numberOfSteps):
            ## constraint runtime_on_step
            [c._runtime_on_step() for c in _usedConstraints]
            ## increment generated
            self.__generated += 1
            ## get selected indexes and coordinates
            _coordsBeforeMove,     \
            _movedRealCoordinates, \
            _movedBoxCoordinates =  \
            self.__on_runtime_step_select_group(_coordsBeforeMove     = _coordsBeforeMove,
                                                _movedRealCoordinates = _movedRealCoordinates,
                                                _moveTried            = _moveTried)
            if not len(self._RT_groupAtomsIndexes):
                LOGGER.nottried("@%s Generated move %i can't be tried because all atoms are collected."%(self.__usedFrame,self.__generated))
            else:
                # try move atom
                if _movedRealCoordinates is None:
                    _moveTried, _rejectRemove = self.__on_runtime_step_try_remove(_constraints = _constraints,
                                                      _rigidConstraints    = _rigidConstraints,
                                                      _usedConstraints     = _usedConstraints)
                # try remove atom
                else:
                    _moveTried, _rejectMove = self.__on_runtime_step_try_move(_constraints = _constraints,
                                                    _rigidConstraints     = _rigidConstraints,
                                                    _usedConstraints      = _usedConstraints,
                                                    _movedRealCoordinates = _movedRealCoordinates,
                                                    _movedBoxCoordinates  = _movedBoxCoordinates)
            ## save engine
            ## MUST CHECK __totalStandardError IF IT'S SAVED !!! SEEMS NOT TO BE SAVED
            _lastSavedTotalStandardError = \
            self.__on_runtime_step_save_engine(_saveFrequency               = _saveFrequency,
                                               step                         = step,
                                               _frame                       = _frame,
                                               _usedConstraints             = _usedConstraints,
                                               _lastSavedTotalStandardError = _lastSavedTotalStandardError)
            ## save xyz trajecctory
            ## special care must be taken because once atoms are collected xyz files needs to adapt
            self.__on_runtime_step_save_xyz(_xyzFrequency=_xyzFrequency, _xyzfd=_xyzfd, step=step)
        # close .xyz file
        if _xyzFrequency is not None:
            _xyzfd.close()
        # export restart pdb
        if restartPdb:
            self.export_pdb( restartPdb )

        #   #####################################################################################   #
        #   ################################# FINISH ENGINE RUN #################################   #
        LOGGER.info("Engine @%s finishes executing all '%i' steps in %s"%(self.__usedFrame,_numberOfSteps, get_elapsed_time(_engineStartTime, format="%d(days) %d:%d:%d")))



    def plot_constraints(self, frame=None, constraintsID=None, plotKwargs=None):
        """
        Plot constraints
        """
        # check plotKwargs
        if plotKwargs is not None:
            assert isinstance(plotKwargs, dict), "plotKwargs must be None or a dict"
        else:
            plotKwargs = {}
        # check frame
        if frame is None:
            frame = self.__usedFrame
        isNormalFrame, isMultiframe, isSubframe = self.get_frame_category(frame)
        # get contsraints
        if constraintsID is not None:
            assert isinstance(constraintsID, (list,set,tuple)), LOGGER.error("constraintsID must be None or a list")
            assert len(constraintsID)>=1, LOGGER.error("constraintsID list length must be >=1")
            assert all([isinstance(cid, basestring) for cid in constraintsID]), LOGGER.error("constraintsID lits items must be string")
        if frame == self.__usedFrame:
            _constraints = self.__constraints
        elif isNormalFrame:
            _constraints = self.__repository.pull(relativePath='_Engine__constraints')
        elif isSubframe:
            _constraints = self.__repository.pull(relativePath=os.path.join(frame,'_Engine__constraints'))
        else:
            _sf = self.__frames[frame]['frames_name'][0]
            _constraints = self.__repository.pull(relativePath=os.path.join(_sf,'_Engine__constraints'))
        # filter constraints
        if constraintsID is not None:
            _constraints = [c for c in _constraints if c.constraintName in constraintsID]
        assert len(_constraints), LOGGER.error("No constraints are left after keeping only given constraintsID")

        # plot normal frame
        if isNormalFrame or isSubframe:
            _oldFrame = self.__usedFrame
            self.set_used_frame(frame)
            for c in _constraints[:-1]:
                _kw = plotKwargs.get(c.constraintName,{})
                _kw['show'] = False
                c.plot(**_kw)
            self.set_used_frame(_oldFrame)
            c = _constraints[-1]
            _kw = plotKwargs.get(c.constraintName,{})
            _kw['show'] = True
            c.plot(**_kw)
        print(_constraints)





#
