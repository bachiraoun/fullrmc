"""
Engine is fullrmc main module. It contains 'Engine' the main class 
of fullrmc which is the Reverse Monte Carlo artist. The engine class 
takes only Protein Data Bank formatted files 
`'.pdb' <http://deposit.rcsb.org/adit/docs/pdb_atom_format.html>`_ as 
atomic/molecular input structure. It handles and fits simultaneously many 
experimental data while controlling the evolution of the system using 
user-defined molecular or atomistic constraints such as bond-length, 
bond-angles, inter-molecular-distances, dihedral angles, etc. 
"""

# standard libraries imports
import os
import time
import sys
import uuid
import tempfile
import multiprocessing
import copy

# external libraries imports
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries
from pyrep import Repository

# fullrmc library imports
from __pkginfo__ import __version__
from Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from Core.boundary_conditions_collection import transform_coordinates
from Core.Collection import Broadcaster, is_number, is_integer, get_elapsed_time, generate_random_float
from Core.Collection import AtomsCollector
from Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from Core.Group import Group, EmptyGroup
from Core.MoveGenerator import SwapGenerator, RemoveGenerator
from Core.GroupSelector import GroupSelector
from Selectors.RandomSelectors import RandomSelector


class Engine(object):
    """ 
    fulrmc's Reverse Monte Carlo (RMC) engine, used to launched a RMC simulation. 
    It has the capability to use and fit simultaneously multiple sets of 
    experimental data. One can also define constraints such as distances, 
    bonds length, angles and many others.   
    
    :Parameters:
        #. path (None, string): Engine repository path to save the engine. If None is 
           given path will be set when saving the engine using Engine.save method. if a 
           non-empty directory is found at the given path an error will be raised.
        #. frames (None, list): List of frames name. Frames are used to store fitting
           data. Multiple frames can be used to create a fitting story or to fit 
           multiple structures simultaneously. Also multiple frames can be used to 
           launch multiple simulations at the same time and merge structures at some 
           predefined merging frequency.
           If None is given, a single frame '0' is initialized automatically.
        #. logFile (None, string): Logging file basename. A logging file full name will
           be the given logFile appended '.log' extension automatically.
           If None is given, logFile is left unchanged.
        #. freshStart (boolean): Whether to remove any existing fullrmc engine at the 
           given path if found. If set to False, an error will be raise if a fullrmc 
           engine or a non-empty directory is found at the given path.
    
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
        ENGINE.run(numberOfSteps=10000, saveFrequency=10000, savePath="system.rmc")
    
    """
    def __init__(self, path=None, frames=None, logFile=None, freshStart=False):
        # set repository and frame data
        ENGINE_DATA   = ('_Engine__frames', '_Engine__usedFrame', )
        ## MUST ADD ORIGINAL DATA TO STORE ALL ORIGINAL PDB ATTRIBUTES
        ## THIS IS NEEDED TO SET GROUPS AND ETC ESPECIALLY AFTER REMOVING
        ## ATOMS FROM SYSTEM.
        self.__frameOriginalData = {}
        FRAME_DATA    = ('_Engine__pdb', '_Engine__tolerance', ### MIGHT NEED TO MOVE _Engine__pdb TO ENGINE_DATA
                         '_Engine__boundaryConditions', '_Engine__isPBC', '_Engine__isIBC',
                         '_Engine__basisVectors', '_Engine__reciprocalBasisVectors',
                         '_Engine__numberDensity', '_Engine__volume',
                         '_Engine__realCoordinates','_Engine__boxCoordinates',
                         '_Engine__groups', '_Engine__groupSelector', '_Engine__state', 
                         '_Engine__generated', '_Engine__tried', '_Engine__accepted',
                         '_Engine__removed', '_Engine__tolerated', '_Engine__totalStandardError',
                         '_Engine__lastSelectedGroupIndex', '_Engine__numberOfMolecules',
                         '_Engine__moleculesIndexes', '_Engine__moleculesNames',
                         '_Engine__allElements', '_Engine__elements',
                         '_Engine__elementsIndexes', '_Engine__numberOfAtomsPerElement',
                         '_Engine__allNames', '_Engine__names',
                         '_Engine__namesIndexes', '_Engine__numberOfAtomsPerName',
                         '_atomsCollector',) 
        RUNTIME_DATA  = ('_Engine__realCoordinates','_Engine__boxCoordinates',
                         '_Engine__state', '_Engine__generated', '_Engine__tried', 
                         '_Engine__accepted','_Engine__tolerated', '_Engine__removed', 
                         '_Engine__totalStandardError', '_Engine__lastSelectedGroupIndex',
                         '_atomsCollector',  # RUNTIME_DATA must have all atomsCollector data keys and affected attributes upon amputating atoms
                         '_Engine__moleculesIndexes', '_Engine__moleculesNames',
                         '_Engine__elementsIndexes', '_Engine__allElements',
                         '_Engine__namesIndexes', '_Engine__allNames', 
                         '_Engine__numberOfAtomsPerName',
                         '_Engine__numberOfAtomsPerElement',
                         '_Engine__names','_Engine__elements',
                         '_Engine__numberOfMolecules','_Engine__numberDensity',)   
                                                                   
        # might need to add groups to FRAME_DATA
        object.__setattr__(self, 'ENGINE_DATA', tuple( ENGINE_DATA)  )
        object.__setattr__(self, 'FRAME_DATA',  tuple( FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA',tuple( RUNTIME_DATA) )
        
        # initialize engine' info
        if frames is None:
            self.__frames = ('0')
        else:        
            self.__frames = []
            self.__frames = tuple( self.__get_normalized_frames_name(frames) )
        self.__usedFrame  = self.__frames[0]
        self.__id         = str(uuid.uuid1())
        self.__version    = __version__
        
        # check whether an engine exists at this path
        if self.is_engine(path):
            if freshStart: 
                Repository().remove_repository(path, relatedFiles=True, relatedFolders=True)                
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
        dataKeys = ('realCoordinates',  'boxCoordinates',
                    'moleculesIndexes', 'moleculesNames',
                    'elementsIndexes',  'allElements',
                    'namesIndexes',     'allNames')
        self._atomsCollector = AtomsCollector(self, dataKeys=dataKeys)

        # initialize engine attributes
        self.__broadcaster   = Broadcaster()
        self.__constraints   = []
        self.__state         = time.time()
        self.__groups        = []
        self.__groupSelector = None
        self.__tolerance     = 0.
        
        # set mustSave flag, it indicates  whether saving whole engine is needed before running
        self.__mustSave = False
        self.__saveGroupsFlag = True
        
        # set pdb
        self.set_pdb(pdb=None)
        
        # create runtime variables and arguments
        self._runtime_ncores = INT_TYPE(1)
        
        # set LOGGER file path
        if logFile is not None:
            self.set_log_file(logFile)
        
    def __setattr__(self, name, value):
        if name in ('ENGINE_DATA', 'FRAME_DATA', 'RUNTIME_DATA'):
            raise LOGGER.error("Setting '%s' is not allowed."%name)
        else:
            object.__setattr__(self, name, value)
            
    def __getstate__(self):
        state = {}
        for k, v in self.__dict__.items():
            if k in self.ENGINE_DATA:
                continue
            if k in self.FRAME_DATA:
                continue
            state[k] = v
        return state

    def __get_normalized_frames_name(self, frames, raiseExisting=True):
        if not isinstance(frames, (list,set,tuple)):
            frames = [frames]
        else:
            frames = list(frames)
        assert len(frames), LOGGER.error("frames must be a non-empty list.")
        for idx, f in enumerate(frames):
            if isinstance(f, basestring):
                f = str(f)
                assert str(f).replace('_','').replace('-','').replace(' ','').isalnum(), LOGGER.error("String frame must be strictly alphanumeric allowing only '-' and '_'")
            else:
                assert isinstance(f, int), LOGGER.error('Each frame must be either interger or string')
            f = str(f)
            if raiseExisting:
                assert f not in self.__frames, LOGGER.error("frame '%s' exists already."%f)
            frames[idx] = f
        # check for redundancy
        assert len(frames) == len(set(frames)), "Redundancy is not allowed in frame names."
        # all is good
        return frames
     
    def __check_path_to_create_repository(self, path):
        # check for string
        if not isinstance(path, basestring):
             return False, "path must be a string. '%s' is given"%path
        # test if directory is empty
        if os.path.exists(path):
            # test directory
            if not os.path.isdir(path):
                return False, "path must be a directory. '%s' is given"%path
            if len(os.listdir(path)):
                return False, "path directory at '%s' is not empty"%path
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
            self.__repository.dump(value=self, relativePath='.', name='engine', replace=True) 
            self.__repository.dump(value=self.__state, relativePath=self.__usedFrame, name='_Engine__state', replace=True)    
            self.__repository.dump(value=self.__lastSelectedGroupIndex, relativePath=self.__usedFrame, name='_Engine__lastSelectedGroupIndex', replace=True)    
            self.__repository.dump(value=self.__generated, relativePath=self.__usedFrame, name='_Engine__generated', replace=True)    
            self.__repository.dump(value=self.__removed, relativePath=self.__usedFrame, name='_Engine__removed', replace=True)    
            self.__repository.dump(value=self.__tried, relativePath=self.__usedFrame, name='_Engine__tried', replace=True)    
            self.__repository.dump(value=self.__accepted, relativePath=self.__usedFrame, name='_Engine__accepted', replace=True)    
            self.__repository.dump(value=self.__tolerated, relativePath=self.__usedFrame, name='_Engine__tolerated', replace=True)    
            self.__repository.dump(value=self.__totalStandardError, relativePath=self.__usedFrame, name='_Engine__totalStandardError', replace=True)                
            
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
        dataDict['realCoordinates']  = self.__realCoordinates[relativeIndex,:]
        dataDict['boxCoordinates']   = self.__boxCoordinates[relativeIndex, :]
        dataDict['moleculesIndexes'] = self.__moleculesIndexes[relativeIndex]
        dataDict['moleculesNames']   = self.__moleculesNames[relativeIndex]
        dataDict['elementsIndexes']  = self.__elementsIndexes[relativeIndex]
        dataDict['allElements']      = self.__allElements[relativeIndex]
        dataDict['namesIndexes']     = self.__namesIndexes[relativeIndex]
        dataDict['allNames']         = self.__allNames[relativeIndex]   
        assert self.__numberOfAtomsPerElement[dataDict['allElements']]-1>0, LOGGER.error("Collecting last atom of any element type is not allowed. It's better to restart your simulation without any '%s' rather than removing them all!"%dataDict['allElements'])
        # collect atom
        self._atomsCollector.collect(index=realIndex, dataDict=dataDict)
        # collect all constraints BEFORE removing data from engine. 
        for c in self.__constraints:
            c._on_collector_collect_atom(realIndex=realIndex)
        # remove data from engine AFTER collecting constraints data.
        self.__realCoordinates   = np.delete(self.__realCoordinates, relativeIndex, axis=0)
        self.__boxCoordinates    = np.delete(self.__boxCoordinates,  relativeIndex, axis=0)
        self.__moleculesIndexes  = np.delete(self.__moleculesIndexes,relativeIndex, axis=0)
        self.__moleculesNames.pop(relativeIndex)
        self.__elementsIndexes   = np.delete(self.__elementsIndexes, relativeIndex, axis=0)
        self.__allElements.pop(relativeIndex)
        self.__namesIndexes      = np.delete(self.__namesIndexes,    relativeIndex, axis=0)
        self.__allNames.pop(relativeIndex)
        # adjust other attributes
        self.__numberOfAtomsPerName[dataDict['allNames']]       -= 1
        self.__numberOfAtomsPerElement[dataDict['allElements']] -= 1
        #self.__elements = sorted(set(self.__allElements)) # no element should disappear
        self.__names = sorted(set(self.__names))
        self.__numberOfMolecules = len(set(self.__moleculesIndexes))
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
        self.__realCoordinates  = np.insert(self.__realCoordinates,  relativeIndex, dataDict["realCoordinates"], axis=0)
        self.__boxCoordinates   = np.insert(self.__boxCoordinates,   relativeIndex, dataDict["boxCoordinates"],  axis=0)
        self.__moleculesIndexes = np.insert(self.__moleculesIndexes, relativeIndex, dataDict["moleculesIndexes"],axis=0)
        self.__moleculesNames.insert(relativeIndex, dataDict["moleculesNames"])
        self.__elementsIndexes  = np.insert(self.__elementsIndexes,  relativeIndex, dataDict["elementsIndexes"], axis=0)
        self.__allElements.insert(relativeIndex, dataDict["allElements"])
        self.__namesIndexes     = np.insert(self.__namesIndexes,     relativeIndex, dataDict["namesIndexes"],    axis=0)
        self.__allNames.insert(relativeIndex, dataDict["allNames"])
        # adjust other attributes
        self.__numberOfAtomsPerName[dataDict['allNames']]       += 1
        self.__numberOfAtomsPerElement[dataDict['allElements']] += 1
        self.__elements = list(set(self.__allElements))
        self.__names = sorted(set(self.__names))
        self.__numberOfMolecules = len(set(self.__moleculesIndexes))
        self.__numberDensity = FLOAT_TYPE(self.numberOfAtoms) / FLOAT_TYPE(self.__volume)
        
    @property
    def info(self):
        """ Engine's information (version, id) tuple."""
        return (self.__version, self.__id)
    
    @property
    def frames(self):
        """ Engine's frames list copy."""
        return [f for f in self.__frames]
    
    @property
    def usedFrame(self):
        """ Engine's frame in use."""
        return self.__usedFrame
       
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
    def lastSelectedAtomsIndexes(self):
        """ The last moved atoms indexes. """
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
        """ removed atoms tuple (tried, accepted, ratio)"""
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
        """ Number of tolerated steps in spite of increasing total totalStandardError"""
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
        """ Engine's pdbParser instance. """
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
        """ The boundary conditions basis vectors in case of PeriodicBoundaries, None in 
        case of InfiniteBoundaries. """
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
        PeriodicBoundaries. Similar to realCoordinates in case of InfiniteBoundaries."""
        return self.__boxCoordinates
        
    @property
    def numberOfMolecules(self):
        """ Number of molecules."""
        return self.__numberOfMolecules
        
    @property
    def moleculesIndexes(self):
        """ All atoms molecules indexes. """
        return self.__moleculesIndexes
    
    @property
    def moleculesNames(self):
        """ Al atoms molecules names. """    
        return self.__moleculesNames 
        
    @property
    def elementsIndexes(self):
        """ All atoms element index in elements list. """
        return self.__elementsIndexes
    
    @property
    def elements(self):
        """ Sorted set of all existing atom elements. """
        return self.__elements
    
    @property
    def allElements(self):
        """ All atoms elements. """
        return self.__allElements
        
    @property
    def namesIndexes(self):
        """ All atoms name index in names list"""
        return self.__namesIndexes
        
    @property
    def names(self):
        """ Srted set of all existing atom names. """
        return self.__names
    
    @property
    def allNames(self):
        """ All atoms name list. """
        return self.__allNames
        
    @property
    def numberOfNames(self):
        """ Number of defined atom names set. """
        return len(self.__names)
    
    @property
    def numberOfAtoms(self):
        """ Number of atoms in pdb structure."""
        return self.__realCoordinates.shape[0]
        
    @property
    def numberOfAtomsPerName(self):
        """ Number of atoms per name dictionary. """
        return self.__numberOfAtomsPerName
        
    @property
    def numberOfElements(self):
        """ Number of different elements in the configuration. """
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
        """ List copy of all constraints instances. """
        return [c for c in self.__constraints]
    
    @property
    def groupSelector(self):
        """ Engine's group selector instance. """
        return self.__groupSelector
        
    @property
    def totalStandardError(self):
        """ Engine's last recorded totalStandardError of the current configuration. """
        return self.__totalStandardError
    
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
            assert self.__frameOriginalData.has_key(dname), LOGGER.error("data '%s' doesn't exist"%name)
            value = self.__frameOriginalData[dname]
            assert value is not None, LOGGER.error("data '%s' value seems to be deleted"%name)
        else:
            info, m = self.__repository.get_file_info(relativePath=self.__usedFrame, name=dname)
            assert info is not None, LOGGER.error("unable to pull data '%s' (%s)"%(name, m) )
            value = self.__repository.pull(relativePath=self.__usedFrame, name=dname)
        return value
        
    def is_engine(self, path, repo=False, mes=False):
        """
        Get whether a fullrmc engine is stored in the given path.
        
        :Parameters:
            #. path (string): The path to fetch.
            #. repo (boolean): Whether to return repository if an engine is found. 
               Otherwise None is returned.
            #. mes (boolean): Whether to return explanatory message.
        
        :Returns:
            #. result (boolean): The fetch result, True if engine is found False 
               otherwise.
            #. repo (pyrep.Repository): The repository instance. 
               This is returned only if 'repo' argument is set to True.
            #. message (string): The explanatory message.
               This is returned only if 'mes' argument is set to True.
        """
        assert isinstance(repo, bool), "repo must be boolean"
        assert isinstance(mes, bool), "mes must be boolean"
        rep = Repository()
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
            rep = rep.load_repository(path)
            if not isinstance(rep.info, dict):
                result  = False
                rep     = None 
                message = "Existing repository at '%s' is not a known fullrmc engine"%path
            elif len(rep.info) < 3:
                result  = False
                rep     = None 
                message = "Existing repository at '%s' is not a known fullrmc engine"%path
            elif rep.info.get('repository type', None) != 'fullrmc engine':
                result  = False
                rep     = None 
                message = "Existing repository at '%s' is not a known fullrmc engine"%path
            elif rep.info.get('fullrmc version', None) is None:
                result  = False
                rep     = None 
                message = "Existing repository at '%s' is not a known fullrmc engine"%path
            elif rep.info.get('engine id', None) is None:
                result  = False
                rep     = None 
                message = "Existing repository at '%s' is not a known fullrmc engine"%path
            else:
                result  = True 
                message = "Existing repository at '%s' is not a known fullrmc engine"%path
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
        LOGGER.saved("Runtime saving frame %s... DON'T INTERRUPT"%frame)
        # dump engine's used frame FRAME_DATA
        for dname in self.RUNTIME_DATA:
            value = self.__dict__[dname]
            #name  = dname.split('_Engine__')[1]
            name = dname
            #tic = time.time()
            self.__repository.dump(value=value, relativePath=frame, name=name, replace=True) 
            #print "engine: %s - %s"%(name, time.time()-tic)
        # dump constraints' used frame FRAME_DATA
        for c in self.__constraints:
            cp = os.path.join(frame, 'constraints', c.constraintId)
            #self.__repository.add_directory( cp )
            for dname in c.RUNTIME_DATA:
                value = c.__dict__[dname]
                #name  = dname.split('__')[1]
                name = dname
                #tic = time.time()
                self.__repository.dump(value=value, relativePath=cp, name=name, replace=True)
                #print "%s: %s - %s"%(c.__class__.__name__, name, time.time()-tic)    
        # engine saved
        LOGGER.saved("Runtime frame %s is successfuly saved"%(frame,) )
               
    def save(self, path=None, copyFrames=True):
        """
        Save engine to disk. 
        
        :Parameters:
            #. path (None, string): Repository path to save the engine. 
               If path is None, engine's path is used to update already saved engine. 
               If path and engine's path are both None, and error will be raised.
            #. copyFrames (boolean): If path is None, this argument is discarded. 
               This argument sets whether to copy all frames data to the new repository 
               path. If path is not None and this argument is False, Only used frame data 
               will be copied and other frames will be discarded in new engine.
               
        N.B. If path is given, it will automatically updates engine's path to point 
        towards given path.
        """
        LOGGER.saved("Saving Engine and frame %s data... DON'T INTERRUPT"%self.__usedFrame)
        # create info dict
        info = {'repository type':'fullrmc engine', 'fullrmc version':__version__, 'engine id':self.__id}
        # path is given
        if path is not None:
            result, message = self.__check_path_to_create_repository(path)
            assert result, LOGGER.error(message)
            REP = Repository()
            REP.create_repository(path, info=info)
            self.__path = path
        # first time saving this engine
        elif self.__repository is None:
            assert self.__path is not None, LOGGER.error("Given path and engine's path are both None, must give a valid path for saving.")
            REP = Repository()
            REP.create_repository(self.__path, info=info)
        # engine loaded or saved before
        else:
            REP = self.__repository
        # create repository frames
        if (self.__repository is None) or (path is not None and copyFrames):
            for frame in self.__frames:
                REP.add_directory( os.path.join(frame, 'constraints') )
        elif path is not None:
            self.__frames = (self.__usedFrame, )
            REP.add_directory( os.path.join(self.__usedFrame, 'constraints') )
        # dump engine
        REP.dump(value=self, relativePath='.', name='engine', replace=True)
        # dump used frame ENGINE_DATA
        for dname in self.ENGINE_DATA:
            value = self.__dict__[dname]
            #name  = dname.split('_Engine__')[1]
            name = dname
            REP.dump(value=value, relativePath='.', name=name, replace=True)
        # dump engine's used frame FRAME_DATA
        for dname in self.FRAME_DATA:
            value = self.__dict__[dname]
            #name  = dname.split('_Engine__')[1]
            name = dname
            REP.dump(value=value, relativePath=self.__usedFrame, name=name, replace=True)    
        # dump original frame data
        for name, value in self.__frameOriginalData.items():
            if value is not None:
                REP.dump(value=value, relativePath=self.__usedFrame, name=name, replace=True)   
                self.__frameOriginalData[name] = None
        # dump constraints' used frame FRAME_DATA
        for c in self.__constraints:
            cp = os.path.join(self.__usedFrame, 'constraints', c.constraintId)
            REP.add_directory( cp )
            for dname in c.FRAME_DATA:
                value = c.__dict__[dname]
                #name  = dname.split('__')[1]
                name = dname
                REP.dump(value=value, relativePath=cp, name=name, replace=True)    
        # copy rest of frames
        if (self.__repository is not None) and (path is not None and copyFrames):
            # dump rest of frames
            for frame in self.__frames:
                if frame == self.__usedFrame:
                    continue
                for rp, _ in self.__repository.walk_files_info(relativePath=frame):
                    value = self.__repository.pull(relativePath=frame, name=rp)
                    REP.dump(value=value, relativePath=frame, name=rp, replace=True)    
        # set repository
        self.__repository = REP
        # set mustSave flag
        self.__mustSave = False
        # engine saved
        LOGGER.saved("Engine and frame %s data saved successfuly to '%s'"%(self.__usedFrame, self.__path) )
    
    def load(self, path):
        """
        Load and return engine instance. None of the current engine attribute will be 
        updated. must be used as the following
        
        
        .. code-block:: python
        
            # import engine
            from fullrmc.Engine import Engine
        
            # create engine 
            ENGINE = Engine().load(path)
        
        
        :Parameters:
            #. path (string): the file path to save the engine
        
        :Returns:
            #. engine (Engine): the engine's instance.
        """
        # check whether an engine exists at this path
        isEngine, REP, message = self.is_engine(path=path, repo=True, mes=True)
        if not isEngine:
            raise LOGGER.error(message)
        if len(message):
            LOGGER.warn(message)
        # load engine
        engine = REP.pull(relativePath='.', name='engine')
        engine._set_repository(REP)
        engine._set_path(path)
        # pull engine's ENGINE_DATA
        for dname in engine.ENGINE_DATA:
            #name  = dname.split('_Engine__')[1]
            name = dname
            value = REP.pull(relativePath='.', name=name)
            object.__setattr__(engine, dname, value)   
        # pull engine's FRAME_DATA
        for dname in engine.FRAME_DATA:
            #name  = dname.split('_Engine__')[1]
            name = dname
            value = REP.pull(relativePath=engine.usedFrame, name=name)
            object.__setattr__(engine, dname, value)             
        # pull constraints' used frame FRAME_DATA
        for c in engine.constraints:
            cp = os.path.join(engine.usedFrame, 'constraints', c.constraintId)
            for dname in c.FRAME_DATA:
                #name  = dname.split('__')[1]
                name = dname
                value = REP.pull(relativePath=cp, name=name)
                object.__setattr__(c, dname, value) 
        # set engine must save to false
        object.__setattr__(engine, '_Engine__mustSave', False)
        # set engine group selector
        engine.groupSelector.set_engine(engine)
        # return engine instance
        return engine
    
    def set_log_file(self, logFile):
        """
        Set the log file basename.
    
        :Parameters:
           #. logFile (None, string): Logging file basename. A logging file full name will
              be the given logFile appended '.log' extension automatically.
        """
        assert isinstance(logFile, basestring), LOGGER.error("logFile must be a string, '%s' is given"%logFile)
        LOGGER.set_log_file_basename(logFile)
        
    def __create_frame_data(self, frame):
        def check_set_or_raise(this, relativePath):
            # find missing data
            missing = []
            for name in this.FRAME_DATA:
                info, _ = self.__repository.get_file_info(relativePath=relativePath, name=name)
                if info is None:
                    missing.append(name)
            # check all or None missing
            if len(missing) == 0:
                return
            else:
                assert len(missing) == len(this.FRAME_DATA), LOGGER.error("Data files %s are missing from frame '%s'. Consider deleting and rebuilding frame."%(missing,frame,))     
                LOGGER.warn("Using frame '%s' data to create '%s' frame '%s' data."%(self.__usedFrame, this.__class__.__name__, frame))     
                # create frame data
                for name in this.FRAME_DATA:
                    value = this.__dict__[name]
                    self.__repository.dump(value=value, relativePath=relativePath, name=name, replace=True)  
        # check engine frame data
        check_set_or_raise(this=self, relativePath=frame)
        # create original data
        missing = []
        for name in self.__frameOriginalData.keys():
            info, _ = self.__repository.get_file_info(relativePath=frame, name=name)
            if info is None:
                missing.append(name)
        if len(missing):
            assert len(missing) == len(self.__frameOriginalData.keys()), LOGGER.error("Data files %s are missing from frame '%s'. Consider deleting and rebuilding frame."%(missing,frame,))     
            LOGGER.warn("Using frame '%s' data to create frame '%s' original data."%(self.__usedFrame, frame))     
            for name in self.__frameOriginalData.keys():
                value = self.__repository.pull(relativePath=self.__usedFrame, name=name) 
                self.__repository.dump(value=value, relativePath=frame, name=name, replace=True)  
        # create constraints frame data
        for c in self.__constraints:
            cp = os.path.join(frame, 'constraints', c.constraintId)
            check_set_or_raise(this=c, relativePath=cp)
        
    def add_frames(self, frames):
        """
        Add a single or multiple frames to engine.
        
        :Parameters:
            #. frames (string, list): Frames name. It can be a string to add a single
               frame or a list of strings to add multiple frames.
        """
        frames = self.__get_normalized_frames_name(frames, raiseExisting=False)
        # create frames directories
        if self.__repository is not None:
            for frame in frames:
                if frame in self.__frames:
                    LOGGER.warn("frame '%s' exists already. Adding ommitted"%frame)
                    continue
                self.__repository.add_directory( os.path.join(frame, 'constraints') )
        # append frames to list
        self.__frames = list(self.__frames)
        self.__frames.extend(frames)
        self.__frames = tuple( self.__frames )
        # save frames
        if self.__repository is not None:
            self.__repository.dump(value=self.__frames, relativePath='.', name='_Engine__frames', replace=True)

    def reinit_frame(self, frame):
        """
        Reset frame data to initial pdb coordinates.
        
        :Parameters:
            #. frame (string): The frame name to set.
        """
        assert frame in self.__frames, LOGGER.error("Unkown given frame '%s'"%frame)
        if self.__repository is None and frame != self.__usedFrame:
            raise LOGGER.error("It's not allowed to re-initialize frame other than usedFrame prior to building engine's repository. Save engine using Engine.save method first.")
        oldUsedFrame = self.__usedFrame
        # temporarily set used frame
        if frame != self.__usedFrame:
            self.set_used_frame(frame)
        # reset pdb
        self.set_pdb(self.__pdb)
        #  re-set old used frame
        if self.__usedFrame != oldUsedFrame:
            self.set_used_frame(oldUsedFrame)
        
    def set_used_frame(self, frame):
        """
        Set engine frame in use.
        
        :Parameters:
            #. frame (string): The frame name to set.
        """
        if frame == self.__usedFrame:
            return
        assert frame in self.__frames, LOGGER.error("Unkown given frame '%s'"%frame)
        if self.__repository is None:
            raise LOGGER.error("It's not allowed to set used frame prior to building engine's repository. Save engine using Engine.save method first.")
        else:
            # create frame data if missing or raise if partially missing
            self.__create_frame_data(frame=frame)
            self.__usedFrame = frame
            # pull engine's FRAME_DATA
            for dname in self.FRAME_DATA:
                #name  = dname.split('_Engine__')[1]
                name = dname
                value = self.__repository.pull(relativePath=self.__usedFrame, name=name)
                # set data 
                object.__setattr__(self, dname, value)
            # pull constraints' used frame FRAME_DATA
            for c in self.__constraints:
                cp = os.path.join(self.usedFrame, 'constraints', c.constraintId)
                for dname in c.FRAME_DATA:
                    name = dname
                    value = self.__repository.pull(relativePath=cp, name=name)
                    # set data 
                    object.__setattr__(c, dname, value)
        # set engine to specific frame data
        self.__groupSelector.set_engine(self)
        # save used frames to disk
        self.__repository.dump(value=self.__usedFrame, relativePath='.', name='_Engine__usedFrame', replace=True)
        
    def delete_frame(self, frame):
        """
        Delete frame data from Engine as well as from system.
        
        :Parameters:
            #. frame (string): The frame name to delete.
        """
        assert frame != self.__usedFrame, LOGGER.error("Can't delete used frame '%s'"%frame)
        # remove frame directory        
        if self.__repository is not None:
            self.__repository.remove_directory(relativePath=frame, removeFromSystem=True)
        # reset frames
        self.__frames = tuple([f for f in self.__frames if f != frame])
        # save frames
        if self.__repository is not None:
            self.__repository.dump(value=self.__frames, relativePath='.', name='_Engine__frames', replace=True)
        
    def rename_frame(self, frame, newName):
        """
        Rename frame.
        
        :Parameters:
            #. frame (string): The frame name to rename.
            #. newName (string): The frame new name.
        """
        assert frame in self.__frames, LOGGER.error("Unkown given frame '%s'"%frame)
        newName = self.__get_normalized_frames_name(newName)[0]
        # rename frame in repository
        if self.__repository is not None:
            try:
                self.__repository.rename_directory(relativePath=frame, newName=newName, replace=False)
            except Exception as e:
                raise LOGGER.error("Unable to rename frame (%s)"%e)
        # reset frames
        self.__frames = tuple([f if f != frame else newName for f in self.__frames])
        # check used frame
        if self.__usedFrame == frame:
            self.__usedFrame = newName
        # save frames
        if self.__repository is not None:
            self.__repository.dump(value=self.__frames, relativePath='.', name='_Engine__frames', replace=True)
            self.__repository.dump(value=self.__usedFrame, relativePath='.', name='_Engine__usedFrame', replace=True)
           
    def export_pdb(self, path):
        """
        Export a pdb file of the last refined and save configuration state.
        
        :Parameters:
            #. path (string): the pdb file path.
        """
        # MUST TRANSFORM TO PDB COORDINATES SYSTEM FIRST
        if len(self._atomsCollector.indexes):
            indexes = sorted(set(self.__pdb.indexes)-set(self._atomsCollector.indexes))
            pdb = self.__pdb.get_copy(indexes=indexes)
        else:
            pdb = self.__pdb
        pdb.export_pdb(path, coordinates=self.__realCoordinates, boundaryConditions=self.__boundaryConditions )
    
    def get_pdb(self):
        """
        get a pdb instance of the last refined and save configuration state.
        
        :Returns:
            #. pdb (pdbParser): the pdb instance.
        """
        indexes = None
        if len(self._atomsCollector.indexes):
            indexes = sorted(set(self.__pdb.indexes)-set(self._atomsCollector.indexes))
        pdb = self.__pdb.get_copy(indexes=indexes)
        pdb.set_coordinates(self.__realCoordinates)
        pdb.set_boundary_conditions(self.__boundaryConditions)
        return pdb
                
    def set_tolerance(self, tolerance):
        """   
        Sets the runtime engine tolerance value.
        
        :Parameters:
            #. tolerance (number): The runtime tolerance parameters. 
               It's the percentage of allowed unsatisfactory 'tried' moves. 
        """
        assert is_number(tolerance), LOGGER.error("tolerance must be a number")
        tolerance = FLOAT_TYPE(tolerance)
        assert tolerance>=0, LOGGER.error("tolerance must be positive")
        assert tolerance<=100, LOGGER.error("tolerance must be smaller than 100")
        self.__tolerance = FLOAT_TYPE(tolerance/100.)
        # save tolerance to disk
        if self.__repository is not None:
            self.__repository.dump(value=self.__tolerance, relativePath='.', name='_Engine__tolerance', replace=True)
        
    def set_group_selector(self, selector):
        """
        Sets the engine group selector instance.
        
        :Parameters:
            #. groups (None, GroupSelector): the GroupSelector instance. 
               If None, RandomSelector is set automatically.
        """
        if selector is None:
            selector = RandomSelector(self)
        else:
            assert isinstance(selector, GroupSelector), LOGGER.error("selector must a GroupSelector instance")
        # change old selector engine instance to None
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
        if self.__repository is not None:
            self.__repository.dump(value=self.__groupSelector, relativePath=self.__usedFrame, name='_Engine__groupSelector', replace=True)    
    
    def clear_groups(self):
        """ Clear all engine's defined groups.
        """
        self.__groups = []
        # save groups to repository
        if self.__repository is not None:
            self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)    
        
    def add_group(self, g, broadcast=True):
        """
        Add a group to engine's groups list.
        
        :Parameters:
            #. g (Group, integer, list, set, tuple numpy.ndarray): Group instance, integer, 
               list, tuple, set or numpy.ndarray of atoms indexes of atoms indexes.
            #. broadcast (boolean): Whether to broadcast "update groups". Keep True 
               unless you know what you are doing.
        """
        if isinstance(g, Group):
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
            self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)    
      
    def set_groups(self, groups):
        """
        Sets the engine groups of indexes.
        
        :Parameters:
            #. groups (None, Group, list): A single Group instance or a list, tuple, 
               set of any of Group instance, integer, list, set, tuple or numpy.ndarray 
               of atom indexes that will be set one by one by set_group method.
               If None, single atom groups of all atoms will be all automatically created 
               which is the same as using set_groups_as_atoms method.
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
            raise LOGGER.error(e)
            return
        # save groups to repository       
        if self.__repository is not None:
            self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)    
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
    
    def add_groups(self, groups):
        """
        Add groups to engine.
        
        :Parameters:
            #. groups (Group, list): Group instance or list of groups, where every 
               group must be a Group instance or a numpy.ndarray of atoms indexes of 
               type numpy.int32.
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
            raise LOGGER.error(e)
            return
        # save groups to repository       
        if self.__repository is not None:
            self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)    
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
        
    def set_groups_as_atoms(self):
        """ Automatically set engine's groups as single atom groups of all atoms. """
        self.set_groups(None)
        
    def set_groups_as_molecules(self):
        """ Automatically set engine's groups indexes according to molecules indexes. """
        molecules = list(set(self.__moleculesIndexes))
        moleculesIndexes = {}
        for idx in range(len(self.__moleculesIndexes)):
            mol = self.__moleculesIndexes[idx]
            if not moleculesIndexes.has_key(mol):
                moleculesIndexes[mol] = []
            moleculesIndexes[mol].append(idx)
        # create groups
        keys = sorted(moleculesIndexes.keys())
        # reset groups
        self.__groups = []
        # add groups
        for k in keys:
            self.add_group(np.array(moleculesIndexes[k], dtype=INT_TYPE), broadcast=False) 
        # save groups to repository       
        if self.__repository is not None:
            self.__repository.dump(value=self.__groups, relativePath=self.__usedFrame, name='_Engine__groups', replace=True)    
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
            
    def set_pdb(self, pdb, boundaryConditions=None, names=None, elements=None, moleculesIndexes=None, moleculesNames=None):
        """
        Set used frame pdb configuration. Engine and constraints data will be 
        automatically reset but not constraints definitions. If pdb was already set and 
        this is a resetting of a different atomic configuration, with different elements 
        or atomic order, or different size and number of atoms, constraints definitions 
        must be reset manually. In general, their is no point in changing the atomic 
        configuration of a completely different atomic nature. It is advisable to create 
        a new engine from scratch or redefining all constraints definitions.
        
        :Parameters:
            #. pdb (pdbParser, string): the configuration pdb as a pdbParser instance or 
               a path string to a pdb file.
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries, 
               numpy.ndarray, number): The configuration's boundary conditions.
               If None, boundaryConditions are set to InfiniteBoundaries with no periodic 
               boundaries. If numpy.ndarray is given, it must be pass-able to a 
               PeriodicBoundaries. Normally any real numpy.ndarray of shape (1,), (3,1), 
               (9,1), (3,3) is allowed. If number is given, it's like a numpy.ndarray of 
               shape (1,), it is assumed as a cubic box of box length equal to number.
            #. names (None, list): All pdb atoms names list.
               If None names will be calculated automatically by parsing pdb instance.
            #. elements (None, list): All pdb atoms elements list.
               If None elements will be calculated automatically by parsing pdb instance.
            #. moleculesIndexes (None, list, numpy.ndarray): The molecules indexes list.
               If None moleculesIndexes will be calculated automatically by parsing pdb 
               instance.
            #. moleculesNames (None, list): The molecules names list. Must have the 
               length of the number of atoms. If None, it is automatically generated as 
               the pdb residues name.
        """
        if pdb is None:
            pdb = pdbParser()
            bc  = PeriodicBoundaries()
            bc.set_vectors(1)
            pdb.set_boundary_conditions(bc)
        if not isinstance(pdb, pdbParser):
            try:
                pdb = pdbParser(pdb)
            except:
                raise Exception( LOGGER.error("pdb must be None, pdbParser instance or a string path to a protein database (pdb) file.") )
        # set pdb
        self.__pdb = pdb 
        # get coordinates
        self.__realCoordinates = np.array(self.__pdb.coordinates, dtype=FLOAT_TYPE)
        # save data to repository       
        if self.__repository is not None:
            self.__repository.dump(value=self.__pdb, relativePath=self.__usedFrame, name='_Engine__pdb', replace=True)    
            self.__repository.dump(value=self.__realCoordinates, relativePath=self.__usedFrame, name='_Engine__realCoordinates', replace=True)    
        # set boundary conditions
        if boundaryConditions is None:
            boundaryConditions = pdb.boundaryConditions
        self.set_boundary_conditions(boundaryConditions)
        # get elementsIndexes
        self.set_elements_indexes(elements)
        # get namesIndexes
        self.set_names_indexes(names)
        # get moleculesIndexes
        self.set_molecules_indexes(moleculesIndexes=moleculesIndexes, moleculesNames=moleculesNames)
        # broadcast to constraints
        self.__broadcaster.broadcast("update pdb")
        # reset engine flags
        self.reset_engine()
        
    def set_boundary_conditions(self, boundaryConditions):
        """
        Sets the configuration boundary conditions. Any type of periodic boundary 
        conditions are allowed and not restricted to cubic. Engine and constraintsData 
        will be automatically reset.
        
        :Parameters:
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries, 
               numpy.ndarray, number): The configuration's boundary conditions.
               If None, boundaryConditions are set to InfiniteBoundaries with no periodic 
               boundaries. If numpy.ndarray is given, it must be pass-able to a
               PeriodicBoundaries. Normally any real numpy.ndarray of shape (1,), (3,1), 
               (9,1), (3,3) is allowed. If number is given, it's like a numpy.ndarray of 
               shape (1,), it is assumed as a cubic box of box length equal to number.
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
            self.__volume = FLOAT_TYPE( 1./.0333679 * self.numberOfAtoms ) 
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
            self.__repository.dump(value=self.__boundaryConditions, relativePath=self.__usedFrame, name='_Engine__boundaryConditions', replace=True)    
            self.__repository.dump(value=self.__basisVectors, relativePath=self.__usedFrame, name='_Engine__basisVectors', replace=True)    
            self.__repository.dump(value=self.__reciprocalBasisVectors, relativePath=self.__usedFrame, name='_Engine__reciprocalBasisVectors', replace=True)    
            self.__repository.dump(value=self.__numberDensity, relativePath=self.__usedFrame, name='_Engine__numberDensity', replace=True)    
            self.__repository.dump(value=self.__volume, relativePath=self.__usedFrame, name='_Engine__volume', replace=True)       
            self.__repository.dump(value=self.__isPBC, relativePath=self.__usedFrame, name='_Engine__isPBC', replace=True)    
            self.__repository.dump(value=self.__isIBC, relativePath=self.__usedFrame, name='_Engine__isIBC', replace=True)    
            self.__repository.dump(value=self.__boxCoordinates, relativePath=self.__usedFrame, name='_Engine__boxCoordinates', replace=True)    
            self.__repository.dump(value=self.numberOfAtoms, relativePath=self.__usedFrame, name='_original__numberOfAtoms', replace=True)    
            self.__repository.dump(value=self.__volume, relativePath=self.__usedFrame, name='_original__volume', replace=True)    
            self.__repository.dump(value=self.__numberDensity, relativePath=self.__usedFrame, name='_original__numberDensity', replace=True)    
            self.__frameOriginalData['_original__numberOfAtoms'] = None
            self.__frameOriginalData['_original__volume'] = None
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
        volume. It can be used only with InfiniteBoundaries. 
        
        :Parameters:
            #. numberDensity (number): The number density value. 
        """
        if isinstance(self.__boundaryConditions, InfiniteBoundaries) and not isinstance(self.__boundaryConditions, PeriodicBoundaries):
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
            self.__repository.dump(value=self.__numberDensity, relativePath=self.__usedFrame, name='_Engine__numberDensity', replace=True)    
            self.__repository.dump(value=self.__volume, relativePath=self.__usedFrame, name='_Engine__volume', replace=True)    
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET. 
#        # set mustSave flag
#        self.__mustSave = True      
        
    def set_molecules_indexes(self, moleculesIndexes=None, moleculesNames=None):
        """
        Sets moleculesIndexes list, assigning each atom to a molecule.
        
        :Parameters:
            #. moleculesIndexes (None, list, numpy.ndarray): The molecules indexes list.
               If None moleculesIndexes will be calculated automatically by parsing pdb 
               instance.
            #. moleculesNames (None, list): The molecules names list. Must have the 
               length of the number of atoms. If None, it is automatically generated as 
               the pdb residues name.
        """
        if not self.__pdb.numberOfAtoms:
            moleculesIndexes = []
        elif moleculesIndexes is None:
            moleculesIndexes = []
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
                moleculesIndexes.append(molIndex)
        else:
            assert isinstance(moleculesIndexes, (list,set,tuple, np.ndarray)), LOGGER.error("moleculesIndexes must be a list of indexes")
            assert len(moleculesIndexes)==self.__pdb.numberOfAtoms, LOGGER.error("moleculesIndexes must have the same length as pdb")
            if isinstance(moleculesIndexes, np.ndarray):
                assert len(moleculesIndexes.shape)==1, LOGGER.error("moleculesIndexes numpy.ndarray must have a dimension of 1")
                assert moleculesIndexes.dtype.type is INT_TYPE, LOGGER.error("moleculesIndexes must be of type numpy.int32")
            else:
                for molIdx in moleculesIndexes:
                    assert is_integer(molIdx), LOGGER.error("molecule's index must be an integer")
                    molIdx = INT_TYPE(molIdx)
                    assert int(molIdx)>=0, LOGGER.error("molecule's index must positive")
        # check molecules names
        if moleculesNames is not None:
            assert isinstance(moleculesNames, (list, set, tuple)), LOGGER.error("moleculesNames must be a list")
            moleculesNames = list(moleculesNames)
            assert len(moleculesNames)==self.__pdb.numberOfAtoms, LOGGER.error("moleculesNames must have the same length as pdb")
        else:
            moleculesNames = self.__pdb.residues
        if len(moleculesNames):
            molName  = moleculesNames[0]
            molIndex = moleculesIndexes[0]
            for idx in range(len(moleculesIndexes)):
                newMolIndex = moleculesIndexes[idx]
                newMolName  = moleculesNames[idx]
                if newMolIndex == molIndex:
                    assert newMolName == molName, LOGGER.error("Same molecule atoms can't have different molecule name")
                else:
                    molName  = newMolName
                    molIndex = newMolIndex
        # set moleculesIndexes
        self.__numberOfMolecules = len(set(moleculesIndexes))
        self.__moleculesIndexes  = np.array(moleculesIndexes, dtype=INT_TYPE)
        self.__moleculesNames    = list(moleculesNames)
        # save data to repository
        if self.__repository is not None:
            self.__repository.dump(value=self.__numberOfMolecules, relativePath=self.__usedFrame, name='_Engine__numberOfMolecules', replace=True)    
            self.__repository.dump(value=self.__moleculesIndexes, relativePath=self.__usedFrame, name='_Engine__moleculesIndexes', replace=True)    
            self.__repository.dump(value=self.__moleculesNames, relativePath=self.__usedFrame, name='_Engine__moleculesNames', replace=True)    
            # save original data
            self.__repository.dump(value=self.__numberOfMolecules, relativePath=self.__usedFrame, name='_original__numberOfMolecules', replace=True)    
            self.__repository.dump(value=self.__moleculesIndexes, relativePath=self.__usedFrame, name='_original__moleculesIndexes', replace=True)    
            self.__repository.dump(value=self.__moleculesNames, relativePath=self.__usedFrame, name='_original__moleculesNames', replace=True)    
            self.__frameOriginalData['_original__numberOfMolecules'] = None
            self.__frameOriginalData['_original__moleculesIndexes']  = None
            self.__frameOriginalData['_original__moleculesNames']    = None
        else:
            self.__frameOriginalData['_original__numberOfMolecules'] = copy.deepcopy( self.__numberOfMolecules )
            self.__frameOriginalData['_original__moleculesIndexes']  = copy.deepcopy( self.__moleculesIndexes  )
            self.__frameOriginalData['_original__moleculesNames']    = copy.deepcopy( self.__moleculesNames    )
        # broadcast to constraints
        self.__broadcaster.broadcast("update molecules indexes")
        
        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IS
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET. 
#        # set mustSave flag
#        self.__mustSave = True

    def set_elements_indexes(self, elements=None):
        """
        Sets elements list, assigning a type element to each atom.
        
        :Parameters:
            #. elements (None, list): All pdb atoms elements list.
               If None elements will be calculated automatically by parsing pdb instance.
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
        # get elementsIndexes
        lut = dict(zip(self.__elements,range(len(self.__elements))))
        self.__elementsIndexes = np.array([lut[el] for el in self.__allElements], dtype=INT_TYPE)
        # number of atoms per element
        self.__numberOfAtomsPerElement = {}
        for el in self.__allElements:
            if not self.__numberOfAtomsPerElement.has_key(el):
                self.__numberOfAtomsPerElement[el] = 0
            self.__numberOfAtomsPerElement[el] += 1
        # save data to repository
        if self.__repository is not None:
            self.__repository.dump(value=self.__allElements, relativePath=self.__usedFrame, name='_Engine__allElements', replace=True)    
            self.__repository.dump(value=self.__elements, relativePath=self.__usedFrame, name='_Engine__elements', replace=True)    
            self.__repository.dump(value=self.__elementsIndexes, relativePath=self.__usedFrame, name='_Engine__elementsIndexes', replace=True)    
            self.__repository.dump(value=self.__numberOfAtomsPerElement, relativePath=self.__usedFrame, name='_Engine__numberOfAtomsPerElement', replace=True)    
            # save original data
            self.__repository.dump(value=self.__allElements, relativePath=self.__usedFrame, name='_original__allElements', replace=True)    
            self.__repository.dump(value=self.__elements, relativePath=self.__usedFrame, name='_original__elements', replace=True)    
            self.__repository.dump(value=self.__elementsIndexes, relativePath=self.__usedFrame, name='_original__elementsIndexes', replace=True)    
            self.__repository.dump(value=self.__numberOfAtomsPerElement, relativePath=self.__usedFrame, name='_original__numberOfAtomsPerElement', replace=True)    
            self.__frameOriginalData['_original__allElements']             = None
            self.__frameOriginalData['_original__elements']                = None
            self.__frameOriginalData['_original__elementsIndexes']         = None
            self.__frameOriginalData['_original__numberOfAtomsPerElement'] = None
        else:
            self.__frameOriginalData['_original__allElements']             = copy.deepcopy( self.__allElements )
            self.__frameOriginalData['_original__elements']                = copy.deepcopy( self.__elements )
            self.__frameOriginalData['_original__elementsIndexes']         = copy.deepcopy( self.__elementsIndexes )
            self.__frameOriginalData['_original__numberOfAtomsPerElement'] = copy.deepcopy( self.__numberOfAtomsPerElement )
        # broadcast to constraints
        self.__broadcaster.broadcast("update elements indexes")
        
        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IS
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET. 
#        # set mustSave flag
#        self.__mustSave = True
        
    def set_names_indexes(self, names=None):
        """
        Sets names list, assigning a name to each atom.
        
        :Parameters:
            #. names (None, list): The names indexes list.
               If None names will be generated automatically by parsing pdbParser instance.
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
        # get namesIndexes
        lut = dict(zip(self.__names,range(len(self.__names))))
        self.__namesIndexes = np.array([lut[n] for n in self.__allNames], dtype=INT_TYPE)
        # number of atoms per name
        self.__numberOfAtomsPerName = {}
        for n in self.__allNames:
            if not self.__numberOfAtomsPerName.has_key(n):
                self.__numberOfAtomsPerName[n] = 0
            self.__numberOfAtomsPerName[n] += 1
        # save data to repository
        if self.__repository is not None:
            self.__repository.dump(value=self.__allNames, relativePath=self.__usedFrame, name='_Engine__allNames', replace=True)    
            self.__repository.dump(value=self.__names, relativePath=self.__usedFrame, name='_Engine__names', replace=True)    
            self.__repository.dump(value=self.__namesIndexes, relativePath=self.__usedFrame, name='_Engine__namesIndexes', replace=True)    
            self.__repository.dump(value=self.__numberOfAtomsPerName, relativePath=self.__usedFrame, name='_Engine__numberOfAtomsPerName', replace=True)    
            # save original data
            self.__repository.dump(value=self.__allNames, relativePath=self.__usedFrame, name='_original__allNames', replace=True)    
            self.__repository.dump(value=self.__names, relativePath=self.__usedFrame, name='_original__names', replace=True)    
            self.__repository.dump(value=self.__namesIndexes, relativePath=self.__usedFrame, name='_original__namesIndexes', replace=True)    
            self.__repository.dump(value=self.__numberOfAtomsPerName, relativePath=self.__usedFrame, name='_original__numberOfAtomsPerName', replace=True) 
            self.__frameOriginalData['_original__allNames']             = None
            self.__frameOriginalData['_original__names']                = None
            self.__frameOriginalData['_original__namesIndexes']         = None
            self.__frameOriginalData['_original__numberOfAtomsPerName'] = None
        else:
            self.__frameOriginalData['_original__allNames']             = copy.deepcopy( self.__allNames )
            self.__frameOriginalData['_original__names']                = copy.deepcopy( self.__names )
            self.__frameOriginalData['_original__namesIndexes']         = copy.deepcopy( self.__namesIndexes )
            self.__frameOriginalData['_original__numberOfAtomsPerName'] = copy.deepcopy( self.__numberOfAtomsPerName )
        # broadcast to constraints
        self.__broadcaster.broadcast("update names indexes")
        
        # MUST DO SOMETHING ABOUT IT HERE, BECAUSE THIS CAN BE A BIG PROBLEM IS
        # SETTING A NEW PDB AT THE MIDDLE OF A FIT, ALL FRAMES MUST BE RESET. 
#        # set mustSave flag
#        self.__mustSave = True
    
    def visualize(self, commands=None, foldIntoBox=False, boxToCenter=False,
                        boxWidth=2, boxStyle="solid", boxColor="yellow", 
                        bgColor="black", displayParams=None, 
                        representationParams="Lines", otherParams=None):
        """
        Visualize the last configuration using pdbParser visualize_vmd method.
        
        :Parameters:
            #. commands (None, list, tuple): List of commands to pass upon calling vmd.
               commands can be a .dcd file to load a trajectory for instance.
            #. foldIntoBox (boolean): Whether to fold all atoms into simulation box 
               before visualization.
            #. boxToCenter (boolean): Translate box center to atom coordinates center.
            #. boxWidth (number): Visualize the simulation box by giving the lines width.
               If 0 then the simulation box is not visualized.
            #. boxStyle (str): The box line style, it can be either solid or dashed.
            #. boxColor (str): Choose the simulation box color among the following:\n
               blue, red, gray, orange, yellow, tan, silver, green,
               white, pink, cyan, purple, lime, mauve, ochre, iceblue, 
               black, yellow2, yellow3, green2, green3, cyan2, cyan3, blue2,
               blue3, violet, violet2, magenta, magenta2, red2, red3, 
               orange2, orange3.
            #. bgColor (str): Set the background color.
            #. displayParams(None, dict): Set the display parameters. If None, default 
               parameters will be applied. If dictionary the following keys can be used.\n
               * 'depth cueing' (default True): Set the depth cueing flag.
               * 'cue density' (default 0.1): Set the depth density.
               * 'cue mode' (default 'Exp'): Set the depth mode among 'linear', 
                 'Exp' and 'Exp2'.
            #. representationParams(str): Set representation method among the following:\n
               Lines, Bonds, DynamicBonds, HBonds, Points, VDW, CPK, Licorice, Beads, 
               Dotted, Solvent. And add parameters accordingly if needed. e.g.\n
               * Points representation accept only size parameter e.g. 'Points 5'
               * CPK representation can accept respectively 4 parameters as the 
                 following 'Sphere Scale', 'Bond Radius', 'Sphere Resolution', 
                 'Bond Resolution' e.g. 'CPK 1.0 0.2 50 50'
               * VDW representation can accept respectively 2 parameters as the following 
                 'Sphere Scale', 'Sphere Resolution' e.g. 'VDW 0.7 100'
            #. otherParams(None, list, set, tuple): Any other parameters in a form of a 
               list of strings.\n
               e.g. ['display resize 700 700', 'rotate x to 45', 'scale to 0.02', 
               'axes location off']
        """
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
        # create .tcl file
        (vmdfd, tclFile) = tempfile.mkstemp()
        # write tclFile
        tclFile += ".tcl"
        fd = open(tclFile, "w")
        # visualize box
        if boxWidth>0 and isinstance(self.__boundaryConditions, PeriodicBoundaries):
            if foldIntoBox and boxToCenter:
                foldIntoBox = False
                LOGGER.fixed("foldIntoBox and boxToCenter cannot both be set to True. foldIntoBox is reset to False.")
            try:
                X,Y,Z = self.__boundaryConditions.get_vectors()
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
                    cc = np.sum(self.__realCoordinates, axis=0)/self.__realCoordinates.shape[0]
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
        coords = self.__realCoordinates
        # MUST TRANSFORM TO PDB COORDINATES SYSTEM FIRST
        if foldIntoBox and isinstance(self.__boundaryConditions, PeriodicBoundaries):
            coords = self.__boundaryConditions.fold_real_array(self.__realCoordinates)
        # copy pdb if atoms where amputated
        if len(self._atomsCollector.indexes):
            indexes = sorted(set(self.__pdb.indexes)-set(self._atomsCollector.indexes))
            pdb = self.__pdb.get_copy(indexes=indexes)
        else:
            pdb = self.__pdb
        pdb.visualize(commands=commands, coordinates=coords, startupScript=tclFile)
        # remove .tcl file
        os.remove(tclFile)                    

    def add_constraints(self, constraints):
        """
        Add constraints to the engine.
        
        :Parameters:
            #. constraints (Constraint, list, set, tuple): A constraint instance or 
               list of constraints instances
        """
        if isinstance(constraints,(list,set,tuple)):
            constraints = list(constraints)
        else:
            constraints = [constraints]
        # check all constraints
        for c in constraints:
            assert isinstance(c, Constraint), LOGGER.error("constraints must be a Constraint instance or a list of Constraint instances. None of the constraints have been added to the engine.")
            # check for singularity
            if isinstance(c, SingularConstraint):
                assert c.is_singular(self), LOGGER.error("Only one instance of constraint '%s' is allowed in the same engine. None of the constraints have been added to the engine."%self.__class__.__name__)
        # set constraints        
        for c in constraints:
            # check whether same instance added twice
            if c in self.__constraints:
                LOGGER.warn("constraint '%s' already exist in list of constraints"%c)
                continue
            # add engine to constraint
            c._set_engine(self)
            # add to broadcaster listeners list
            self.__broadcaster.add_listener(c)
            # add constraint to engine
            self.__constraints.append(c)
            # broadcast 'engine changed' to constraint
            c.listen("engine set")        
        # add constraints to repository       
        if self.__repository is not None:
            self.__repository.dump(value=self, relativePath='.', name='engine', replace=True)    
            for frame in self.__frames:
                for c in constraints:
                    # Add constraint to all frames
                    cp = os.path.join(frame, 'constraints', c.constraintId)
                    self.__repository.add_directory( cp )
                    for dname in c.FRAME_DATA:
                        value = c.__dict__[dname]
                        #name  = dname.split('__')[1]
                        name = dname
                        self.__repository.dump(value=value, relativePath=cp, name=name, replace=True)    

    def remove_constraints(self, constraints):
        """
        Remove constraints from engine list of constraints.
        
        :Parameters:
            #. constraints (Constraint, list, set, tuple): A constraint instance or 
               list of constraints instances
        """
        if isinstance(constraints,(list,set,tuple)):
            constraints = list(constraints)
        else:
            constraints = [constraints]
        for c in constraints:
            if c in self.__constraints:
                self.__constraints.remove(c)
                #c.set_engine(None) # COMMENTED 2016-DEC-18
                # add to broadcaster listeners list
                self.__broadcaster.remove_listener(c) 
        # remove constraints from all frames in repository       
        if self.__repository is not None:
            self.__repository.dump(value=self, relativePath='.', name='engine', replace=True)   
            for frame in self.__frames: 
                for c in constraints:
                    # Add constraint to all frames
                    cp = os.path.join(frame, 'constraints', c.constraintId)
                    self.__repository.remove_directory(relativePath=cp, removeFromSystem=True)

    def reset_constraints(self):
        """ Reset constraints flags. """
        for c in self.__constraints:
            c.reset_constraint(reinitialize=True)
        # update constraints in repository used frame only
        if self.__repository is not None:
            self.__repository.dump(value=self, relativePath='.', name='engine', replace=True)    
            for c in self.__constraints:
                # Add constraint to all frames
                cp = os.path.join(self.__usedFrame, 'constraints', c.constraintId)
                for dname in c.FRAME_DATA:
                    value = c.__dict__[dname]
                    #name  = dname.split('__')[1]
                    name = dname
                    self.__repository.dump(value=value, relativePath=cp, name=name, replace=True)    
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
        Computes the total standard error as the sum of all constraints' standard error.
        
        .. math::
            \\chi^{2} = \\sum \\limits_{i}^{N} (\\frac{stdErr_{i}}{variance_{i}})^{2}
          
        Where:\n    
        :math:`variance_{i}` is the variance value of the constraint i. \n
        :math:`stdErr_{i}` the standard error of the constraint i defined as :math:`\\sum \\limits_{j}^{points} (target_{i,j}-computed_{i,j})^{2} = (Y_{i,j}-F(X_{i,j}))^{2}` \n
             
        :Parameters:
            #. constraints (list): All constraints used to calculate total 
               totalStandardError.
            #. current (str): which standard error to use. can be anything like
               standardError, afterMoveStandardError or amputatedStandardError, etc.
        
        :Returns:
            #. totalStandardError (list): The computed total total standard error.
        """
        TSE = []
        for c in constraints:
            SD = getattr(c, current)
            assert SD is not None, LOGGER.error("constraint %s %s is not computed yet. Try to initialize constraint"%(c.__class__.__name__,current))
            TSE.append(SD/c.varianceSquared)
        return np.sum(TSE)
    
    def set_total_standard_error(self):
        """
        Computes and sets the total totalStandardError of active constraints.
        """
        # get and initialize used constraints
        _usedConstraints, _constraints, _rigidConstraints = self.initialize_used_constraints()
        # compute totalStandardError
        self.__totalStandardError = self.compute_total_standard_error(_constraints, current="standardError")
        
    def get_used_constraints(self, sortConstraints=False):
        """
        Parses all engine constraints and returns different lists of the active ones.
        
        :parameters:
            #. sortConstraints (boolean): Whether to sort used constraints according 
               to their computation cost property. This is can minimize computations
               and enhance performance by computing less costly constraints first.
               
        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used 
               in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints 
               list that will contribute to the engine total totalStandardError
            #. RigidConstraint (list): All active RigidConstraint constraints instances 
               among usedConstraints list that won't contribute to the engine total 
               totalStandardError
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
        Calls get_used_constraints method, re-initializes constraints when needed and 
        return them all.
        
        :parameters:
            #. force (boolean): Whether to force initializing constraints regardless 
               of their state.
            #. sortConstraints (boolean): Whether to sort used constraints according 
               to their computation cost property. This is can minimize computations
               and enhance performance by computing less costly constraints first.
               
        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used 
               in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints 
               list that will contribute to the engine total totalStandardError
            #. RigidConstraint (list): All active RigidConstraint constraints instances 
               among usedConstraints list that won't contribute to the engine total 
               totalStandardError.
        """    
        assert isinstance(force, bool), LOGGER.error("force must be boolean")
        # get used constraints
        usedConstraints, constraints, rigidConstraints = self.get_used_constraints(sortConstraints=sortConstraints)
        # initialize out-of-dates constraints
        for c in usedConstraints:
            if c.state != self.__state or force:
                LOGGER.info("Initializing constraint data '%s'"%c.__class__.__name__)
                c.compute_data()
                c.set_state(self.__state)
                if c.originalData is None:
                    c._set_original_data(c.data)
        # return constraints
        return usedConstraints, constraints, rigidConstraints
        
    def __runtime_get_number_of_steps(self, numberOfSteps):
        # check numberOfSteps
        assert is_integer(numberOfSteps), LOGGER.error("numberOfSteps must be an integer")
        assert numberOfSteps<=sys.maxint, LOGGER.error("number of steps must be smaller than maximum integer number allowed by the system '%i'"%sys.maxint)
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

    def __on_runtime_step_select_group(self, _coordsBeforeMove, movedRealCoordinates, _moveTried):
        # get group
        self.__lastSelectedGroupIndex = self.__groupSelector.select_index()
        self._RT_selectedGroup = self.__groups[self.__lastSelectedGroupIndex]
        # get move generator
        self._RT_moveGenerator = self._RT_selectedGroup.moveGenerator
        # remover generator
        if isinstance(self._RT_moveGenerator, RemoveGenerator):
            movedRealCoordinates = None 
            movedBoxCoordinates  = None
            self._RT_groupAtomsIndexes    = self._RT_moveGenerator.pick_from_list(self)
            notCollectedAtomsIndexes      = np.array(self._atomsCollector.are_not_collected(self._RT_groupAtomsIndexes), dtype=bool)
            self._RT_groupAtomsIndexes    = self._RT_groupAtomsIndexes[ notCollectedAtomsIndexes ]
            self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
            _coordsBeforeMove    = None
        # move generator
        else:
            # get atoms indexes
            self._RT_groupAtomsIndexes = self._RT_selectedGroup.indexes
            notCollectedAtomsIndexes   = np.array(self._atomsCollector.are_not_collected(self._RT_groupAtomsIndexes), dtype=bool)
            self._RT_groupAtomsIndexes = self._RT_groupAtomsIndexes[ notCollectedAtomsIndexes ]
            # get group atoms coordinates before applying move 
            if isinstance(self._RT_moveGenerator, SwapGenerator):
                self._RT_groupAtomsIndexes    = self._RT_moveGenerator.get_ready_for_move(self._RT_groupAtomsIndexes)
                notCollectedAtomsIndexes      = np.array(self._atomsCollector.are_not_collected(self._RT_groupAtomsIndexes), dtype=bool)
                self._RT_groupAtomsIndexes    = self._RT_groupAtomsIndexes[ notCollectedAtomsIndexes ]
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                _coordsBeforeMove = np.array(self.__realCoordinates[self._RT_groupRelativeIndexes], dtype=self.__realCoordinates.dtype)
            elif _coordsBeforeMove is None or not self.__groupSelector.isRecurring:
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                _coordsBeforeMove = np.array(self.__realCoordinates[self._RT_groupRelativeIndexes], dtype=self.__realCoordinates.dtype)
            elif self.__groupSelector.explore:
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                if _moveTried:
                    _coordsBeforeMove = movedRealCoordinates
            elif not self.__groupSelector.refine:
                self._RT_groupRelativeIndexes = np.array([self._atomsCollector.get_relative_index(idx) for idx in self._RT_groupAtomsIndexes], dtype=INT_TYPE)
                _coordsBeforeMove = np.array(self.__realCoordinates[self._RT_groupRelativeIndexes], dtype=self.__realCoordinates.dtype)
            #else:
            #    raise Exception(LOGGER.critical("Unknown recurrence mode, unable to get coordinates before applying move."))
            # compute moved coordinates
            movedRealCoordinates = self._RT_moveGenerator.move(_coordsBeforeMove)
            movedBoxCoordinates  = transform_coordinates(transMatrix=self.__reciprocalBasisVectors , coords=movedRealCoordinates)
        # return
        return _coordsBeforeMove, movedRealCoordinates, movedBoxCoordinates
    
    
    def __on_runtime_step_try_remove(self, _constraints, _usedConstraints, _rigidConstraints):
        ###################################### reject remove atoms #####################################
        self.__tried += 1
        self.__removed[0] += 1.
        self.__removed[2]  = self.__removed[1]/self.__removed[0]
        rejectRemove = False
        for c in _constraints:
            c.compute_as_if_amputated(realIndex=self._RT_groupAtomsIndexes, relativeIndex=self._RT_groupRelativeIndexes)
        ################################ compute new totalStandardError ################################
        oldStandardError      = self.compute_total_standard_error(_constraints, current="standardError")
        newTotalStandardError = self.compute_total_standard_error(_constraints, current="amputationStandardError")
        if newTotalStandardError > self.__totalStandardError:
            if generate_random_float() > self.__tolerance:
                rejectRemove = True
            else:
                self.__tolerated += 1
                self.__totalStandardError  = newTotalStandardError
        else:
            self.__totalStandardError = newTotalStandardError
        ################################# reject tried remove #################################
        if rejectRemove:
            # set selector move rejected
            self.__groupSelector.move_rejected(self.__lastSelectedGroupIndex)
            for c in _constraints:
                c.reject_amputation(realIndex=self._RT_groupAtomsIndexes, relativeIndex=self._RT_groupRelativeIndexes)
            # log tried move rejected
            LOGGER.rejected("Tried remove %i is rejected"%self.__generated)
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
            LOGGER.accepted("Gen:%i - Tr:%i(%.3f%%) - Acc:%i(%.3f%%) - Rem:%i(%.3f%%) - Err:%.6f" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__removed[1], 100.*self.__removed[2], self.__totalStandardError))

    def __on_runtime_step_try_move(self, _constraints, _usedConstraints, _rigidConstraints, movedRealCoordinates, movedBoxCoordinates):
        ########################### compute rigidConstraints ############################
        rejectMove      = False
        for c in _rigidConstraints:
            # compute before move
            c.compute_before_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes = self._RT_groupRelativeIndexes)
            # compute after move
            c.compute_after_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes = self._RT_groupRelativeIndexes, movedBoxCoordinates=movedBoxCoordinates)
            # get rejectMove
            rejectMove = c.should_step_get_rejected(c.afterMoveStandardError)
            #print c.__class__.__name__, c.standardError, c.afterMoveStandardError, rejectMove
            if rejectMove:
                break
        _moveTried = not rejectMove
        ############################## reject move before trying ##############################
        if rejectMove:
            # rigidConstraints reject move
            for c in _rigidConstraints:
                c.reject_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
            # log generated move rejected before getting tried
            LOGGER.nottried("Generated move %i is not tried"%self.__generated)
        ###################################### try move #######################################
        else:
            self.__tried += 1
            for c in _constraints:
                # compute before move
                c.compute_before_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
                # compute after move
                c.compute_after_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes = self._RT_groupRelativeIndexes, movedBoxCoordinates=movedBoxCoordinates)
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
                LOGGER.rejected("Tried move %i is rejected"%self.__generated)
        ##################################### accept move #####################################
        else:
            self.__accepted  += 1
            # set selector move accepted
            self.__groupSelector.move_accepted(self.__lastSelectedGroupIndex)
            # constraints reject move
            for c in _usedConstraints:
                c.accept_move(realIndexes=self._RT_groupAtomsIndexes, relativeIndexes=self._RT_groupRelativeIndexes)
            # set new coordinates
            self.__realCoordinates[self._RT_groupRelativeIndexes] = movedRealCoordinates
            self.__boxCoordinates[self._RT_groupRelativeIndexes]  = movedBoxCoordinates
            # log new successful move
            triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
            acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
            LOGGER.accepted("Gen:%i - Tr:%i(%.3f%%) - Acc:%i(%.3f%%) - Rem:%i(%.3f%%) - Err:%.6f" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__removed[1], 100.*self.__removed[2],self.__totalStandardError))
        
        
    def __on_runtime_step_save_engine(self, _saveFrequency, step, _frame, _usedConstraints, _lastSavedTotalStandardError):
        ##################################### save engine #####################################
        if _saveFrequency is not None:
            if not(step+1)%_saveFrequency:
                if _lastSavedTotalStandardError==self.__totalStandardError:
                    LOGGER.saved("Save engine omitted because no improvement made since last save.")
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
        Run the Reverse Monte Carlo engine by performing random moves on engine groups.
        
        :Parameters:
            #. numberOfSteps (integer): The number of steps to run.
            #. sortConstraints (boolean): Whether to sort used constraints according 
               to their computation cost property. This is can minimize computations
               and enhance performance by computing less costly constraints first.
            #. saveFrequency (integer): Save engine every saveFrequency steps.
               Save will be omitted if totalStandardError has not decreased. 
            #. xyzFrequency (None, integer): Save coordinates to .xyz file every 
               xyzFrequency steps regardless totalStandardError has decreased or not.
               If None, no .xyz file will be generated.
            #. xyzPath (string): Save coordinates to .xyz file.
            #. restartPdb (None, string): Export a pdb file of the last configuration at 
               the end of the run. If None is given, no pdb file will be exported. If 
               string is given, it should be the path and the pdb file name.
            #. ncores (None, integer): set the number of cores to use. If None, is  
               given, ncores will be set automatically to 1. This argument is only
               effective if fullrmc is compiled with openmp.
        """
        # get arguments
        _numberOfSteps          = self.__runtime_get_number_of_steps(numberOfSteps)
        _saveFrequency, _frame  = self.__runtime_get_save_engine(saveFrequency, frame)
        _xyzFrequency, _xyzPath = self.__runtime_get_save_xyz(xyzFrequency, xyzPath)
        assert _frame == self.__usedFrame, LOGGER.error("Must save engine before changing frame.")
        if _saveFrequency<=_numberOfSteps:
            assert self.__repository is not None, LOGGER.error("engine might be saving during this run but repository is not defined. Use Engine.save method before calling run method.")
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
            LOGGER.warn("No constraints are used. Configuration will be randomized")
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
        movedRealCoordinates         = None
        # save whole engine if must be done
        if self.__mustSave: # Currently it is always False. will check and fix it later
            self.save()
        #   #####################################################################################   #
        #   #################################### RUN ENGINE #####################################   #
        LOGGER.info("Engine started %i steps, total standard error is: %.6f"%(_numberOfSteps, self.__totalStandardError) )
        for step in xrange(_numberOfSteps):
            ## constraint runtime_on_step
            [c._runtime_on_step() for c in _usedConstraints]
            ## increment generated
            self.__generated += 1
            ## get selected indexes and coordinates
            _coordsBeforeMove,     \
            movedRealCoordinates,  \
            movedBoxCoordinates =  \
            self.__on_runtime_step_select_group(_coordsBeforeMove    = _coordsBeforeMove, 
                                                movedRealCoordinates = movedRealCoordinates,
                                                _moveTried           = _moveTried)
            if not len(self._RT_groupAtomsIndexes):
                LOGGER.nottried("Generated remove %i reached maximum allowed and therefore it's not tried"%self.__generated)
            else:
                # try move atom
                if movedRealCoordinates is None:
                    self.__on_runtime_step_try_remove(_constraints         = _constraints,
                                                      _rigidConstraints    = _rigidConstraints, 
                                                      _usedConstraints     = _usedConstraints, )
                # try remove atom
                else:
                    self.__on_runtime_step_try_move(_constraints         = _constraints,
                                                    _rigidConstraints    = _rigidConstraints, 
                                                    _usedConstraints     = _usedConstraints, 
                                                    movedRealCoordinates = movedRealCoordinates, 
                                                    movedBoxCoordinates  = movedBoxCoordinates)
            ## save engine
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
        LOGGER.info("Engine finishes executing all '%i' steps in %s" % (_numberOfSteps, get_elapsed_time(_engineStartTime, format="%d(days) %d:%d:%d")))  
        

            
        