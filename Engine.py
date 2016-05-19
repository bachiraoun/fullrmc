"""
Engine is the fullrmc main module. It contains 'Engine' the main class 
of fullrmc which is the Reverse Monte Carlo artist. The engine class 
takes only Protein Data Bank formatted files 
`'.pdb' <http://deposit.rcsb.org/adit/docs/pdb_atom_format.html>`_ as 
atomic/molecular input structure. It handles and fits simultaneously many 
experimental data while controlling the evolution of the system using 
user-defined molecular or atomistic constraints such as bond-length, 
bond-angles, inter-molecular-distances, etc. 
"""

# standard libraries imports
import os
import time
import sys
import atexit
import tempfile

# external libraries imports
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries

# fullrmc library imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.boundary_conditions_collection import transform_coordinates
from fullrmc.Core.Collection import Broadcaster, is_number, is_integer, get_elapsed_time, generate_random_float
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.Group import Group
from fullrmc.Core.MoveGenerator import SwapGenerator
from fullrmc.Core.GroupSelector import GroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector


class Engine(object):
    """ 
    The Reverse Monte Carlo (RMC) engine, used to launched an RMC simulation. 
    It has the capability to use and fit simultaneously multiple sets of 
    experimental data. One can also define other constraints such as distances, 
    bonds length, angles and many others.   
    
    :Parameters:
            #. pdb (pdbParser, string): The configuration pdb as a pdbParser instance or a path string to a pdb file.
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries, numpy.ndarray, number): The configuration's boundary conditions.
               If None, boundaryConditions are set to InfiniteBoundaries with no periodic boundaries.
               If numpy.ndarray is given, it must be pass-able to a PeriodicBoundaries. Normally any real numpy.ndarray of shape (1,), (3,1), (9,1), (3,3) is allowed.
               If number is given, it's like a numpy.ndarray of shape (1,), it is assumed as a cubic box of box length equal to number.
            #. names (None, list): All pdb atoms names list. 
               List length must be equal to the number of atoms in the pdbParser instance or pdb file.
               If None names will be assigned automatically by parsing pdbParser instance.
            #. elements (None, list): All pdb atoms elements list.
               List length must be equal to the number of atoms in the pdbParser instance or pdb file.
               If None elements will be assigned automatically by parsing pdbParser instance.
            #. moleculesIndexes (None, list, numpy.ndarray): The molecules indexes list.
               List length must be equal to the number of atoms in the pdbParser instance or pdb file.
               If None moleculesIndexes will be assigned automatically by parsing pdbParser instance.
            #. moleculesNames (None, list): The molecules names list. 
               List length must be equal to the number of atoms in the pdbParser instance or pdb file.
               If None, it is automatically generated as the pdb residues name.
            #. groups (None, list): list of groups, where every group must be a numpy.ndarray of atoms indexes of type numpy.int32.
               If None, single atom groups of all atoms will be all automatically created.
            #. groupSelector (None, GroupSelector): The GroupSelector instance. 
               If None, RandomGroupSelector is set automatically.
            #. constraints (None, list): The list of constraints instances.
            #. tolerance (number): The runtime tolerance parameters. 
               It's the percentage of allowed unsatisfactory 'tried' moves. 
      
    
    .. code-block:: python
        
        # import engine
        from fullrmc.Engine import Engine
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # save engine
        ENGINE.save("system.rmc")
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        # Re-define moves generators if needed ...
        
        # run engine for 10000 steps and save only at the end
        ENGINE.run(numberOfSteps=10000, saveFrequency=10000, savePath="system.rmc")
    
    """
    
    def __init__(self, pdb, boundaryConditions=None,
                       names=None, elements=None, 
                       moleculesIndexes=None, moleculesNames=None,
                       groups=None, groupSelector=None, 
                       constraints=None, tolerance=0.):
        # initialize
        self.__broadcaster   = Broadcaster()
        self.__constraints   = []
        self.__state         = None
        self.__groups        = []
        self.__groupSelector = None
        # initialize engine flags and arguments
        #self._initialize_engine()
        # set pdb
        self.set_pdb(pdb=pdb, boundaryConditions=boundaryConditions, elements=elements, moleculesIndexes=moleculesIndexes, moleculesNames=moleculesNames)
        # set set groups
        if groups is not None:
            self.set_groups(groups)
        # set group selector
        if groupSelector is not None:
            self.set_group_selector(groupSelector)
        # set constraints
        if constraints is not None:
            self.add_constraints(constraints)
        # set tolerance
        self.set_tolerance(tolerance)
        # set LOGGER file path to saveEngine path
        #logFile = os.path.join( os.path.dirname(os.path.realpath(_savePath)), LOGGER.logFileBasename)
        #LOGGER.set_log_file_basename(logFile)
        
    def _initialize_engine(self):
        """ Initialize all engine arguments and flags. """
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
        # current model totalStandardError from constraints
        self.__totalStandardError = None
        # set groups as atoms
        self.set_groups(None)
        # set group selector as random
        self.set_group_selector(None)
        
    def _set_generated(self, generated):
        """ Set generated flag. """
        assert is_integer(generated), LOGGER.error("generated must be an integer")
        generated = int(generated)
        assert generated>=0, LOGGER.error("generated must be positive")
        assert generated>=self.__tried, LOGGER.error("generated must be bigger than tried")
        self.__generated = generated
    
    def _set_tried(self, tried):
        """ Set tried flag. """
        assert is_integer(tried), LOGGER.error("tried must be an integer")
        tried = int(tried)
        assert tried>=0, LOGGER.error("tried must be positive")
        assert tried<=self.__generated, LOGGER.error("tried must be smaller than generated")
        self.__tried = tried
    
    def _set_accepted(self, accepted):
        """ Set accepted flag. """
        assert is_integer(accepted), LOGGER.error("accepted must be an integer")
        accepted = int(accepted)
        assert accepted>=0, LOGGER.error("accepted must be positive")
        assert accepted<=self.__tried, LOGGER.error("accepted must be smaller than tried")
        self.__accepted = accepted
        
    def _set_tolerance(self, tolerated):
        """ Set tolerance flag. """
        assert is_integer(tolerated), LOGGER.error("tolerated must be an integer")
        tolerated = int(tolerated)
        assert tolerated>=0, LOGGER.error("tolerated must be positive")
        assert tolerated<=self.__generated, LOGGER.error("tolerated must be smaller than generated")
        assert tolerated<=self.__tried, LOGGER.error("tolerated must be smaller than tried")
        self.__tolerated = tolerated
        
    @property
    def lastSelectedGroupIndex(self):
        """ Get the last moved group instance index in groups list. """
        return self.__lastSelectedGroupIndex
    
    @property
    def lastSelectedGroup(self):
        """ Get the last moved group instance. """
        if self.__lastSelectedGroupIndex is None:
            return None
        return self.__groups[self.__lastSelectedGroupIndex]
    
    @property
    def lastSelectedAtomsIndexes(self):
        """ Get the last moved atoms indexes. """
        if self.__lastSelectedGroupIndex is None:
            return None
        return self.lastSelectedGroup.indexes
        
    @property
    def state(self):
        """ Get engine's state. """
        return self.__state
    
    @property
    def generated(self):
        """ Get number of generated moves. """
        return self.__generated
            
    @property
    def tried(self):
        """ Get number of tried moves. """
        return self.__tried
    
    @property
    def accepted(self):
        """ Get number of accepted moves. """
        return self.__accepted
    
    @property
    def tolerated(self):
        """ Get the number of tolerated steps in spite of increasing total totalStandardError"""
        return self.__tolerated
        
    @property
    def tolerance(self):
        """ Get the tolerance in percent. """
        return self.__tolerance*100.
    
    @property
    def groups(self):
        """ Get engine's defined groups. """
        return self.__groups
    
    @property
    def pdb(self):
        """ Get pdbParser instance. """
        return self.__pdb
    
    @property
    def boundaryConditions(self):
        """ Get boundaryConditions instance. """
        return self.__boundaryConditions
    
    @property
    def basisVectors(self):
        """ Get the basis vectors in case of PeriodicBoundaries or None in case of InfiniteBoundaries. """
        return self.__basisVectors
    
    @property
    def reciprocalBasisVectors(self):
        """ Get the basis vectors in case of PeriodicBoundaries or None in case of InfiniteBoundaries. """
        return self.__reciprocalBasisVectors
    
    @property
    def volume(self):
        """ Get basis volume or None in case of InfiniteBoundaries. """
        return self.__volume
        
    @property
    def realCoordinates(self):
        """ Get the real coordinates of the current configuration. """
        return self.__realCoordinates
        
    @property
    def boxCoordinates(self):
        """ Get the box coordinates of the current configuration or None in case of InfiniteBoundaries.. """
        return self.__boxCoordinates
        
    @property
    def numberOfMolecules(self):
        """ Get the defined number of molecules."""
        return self.__numberOfMolecules
        
    @property
    def moleculesIndexes(self):
        """ Get all atoms molecules indexes. """
        return self.__moleculesIndexes
    
    @property
    def moleculesNames(self):
        """ Get all atoms molecules names. """    
        return self.__moleculesNames 
        
    @property
    def elementsIndexes(self):
        """ Get all atoms element index in elements list. """
        return self.__elementsIndexes
    
    @property
    def elements(self):
        """ Get sorted set of all existing atom elements. """
        return self.__elements
    
    @property
    def allElements(self):
        """ Get all atoms elements. """
        return self.__allElements
        
    @property
    def namesIndexes(self):
        """ Get all atoms name index in names list"""
        return self.__namesIndexes
        
    @property
    def names(self):
        """ Get sorted set of all existing atom names. """
        return self.__names
    
    @property
    def allNames(self):
        """ Get all atoms name list. """
        return self.__allNames
        
    @property
    def numberOfNames(self):
        """ Get the number of defined atom names set. """
        return len(self.__names)
    
    @property
    def numberOfAtoms(self):
        """ Get the number of atoms in pdb."""
        return len(self.__pdb)
        
    @property
    def numberOfAtomsPerName(self):
        """ Get the number of atoms per name dictionary. """
        return self.__numberOfAtomsPerName
        
    @property
    def numberOfElements(self):
        """ Get the number of different elements in the configuration. """
        return len(self.__elements)
     
    @property
    def numberOfAtomsPerElement(self):
        """ Get the number of atoms per element dictionary. """
        return self.__numberOfAtomsPerElement
    
    @property
    def numberDensity(self):
        """ 
        get system's number density computed as :math:`\\rho_{0}=\\frac{N}{V}`
        where N is the total number of atoms and V the volume of the system.
        """
        return self.__numberDensity
        
    @property
    def constraints(self):
        """ Get a list copy of all constraints instances. """
        return [c for c in self.__constraints]
    
    @property
    def groupSelector(self):
        """ Get the group selector instance. """
        return self.__groupSelector
        
    @property
    def totalStandardError(self):
        """ Get the last recorded totalStandardError of the current configuration. """
        return self.__totalStandardError
    
    def save(self, path):
        """
        Save engine to disk.
        
        :Parameters:
            #. path (string): the file path to save the engine
        """
        LOGGER.saved("Saving Engine... DON'T INTERRUPT")
        dirPath  = os.path.os.path.dirname(path)
        fileName = os.path.basename(path)
        fileName = fileName.split(".")[0]
        assert len(fileName), LOGGER.error("Path filename must be a non zero string")
        fileName += ".rmc"
        path = os.path.join(dirPath,fileName)
        # open file
        try:
            fd = open(path, 'wb')
        except Exception as e:
            raise Exception( LOGGER.error("Unable to open file '%s' to save engine. (%s)"%(path,e)) )
        # save engine
        try:
            pickle.dump( self, fd, protocol=pickle.HIGHEST_PROTOCOL )
        except Exception as e:
            fd.close()
            raise Exception( LOGGER.error("Unable to save engine instance. (%s)"%e) )
        finally:
            fd.close()
        LOGGER.saved("Engine saved '%s'"%path)
    
    def load(self, path):
        """
        Load and return engine instance. None of the current engine attribute will be updated.
        
        :Parameters:
            #. path (string): the file path to save the engine
        
        :Returns:
            #. engine (Engine): the engine instance.
        """
        # open file for reading
        try:
            fd = open(path, 'rb')
        except Exception as e:
            raise Exception(LOGGER.error("Can't open '%s' file for reading. (%s)"%(path,e) ))
        # unpickle file
        try:
            engine = pickle.load( fd )
        except Exception as e:
            fd.close()
            raise Exception( LOGGER.error("Unable to open fullrmc engine file '%s'. (%s)"%(path,e)) )
        finally:
            fd.close()
        assert isinstance(engine, Engine), LOGGER.error("%s is not a fullrmc Engine file"%path)
        # return engineInstance
        return engine
              
    def export_pdb(self, path):
        """
        Export a pdb file of the last refined and save configuration state.
        
        :Parameters:
            #. path (string): the pdb file path.
        """
        # MUST TRANSFORM TO PDB COORDINATES SYSTEM FIRST
        self.pdb.export_pdb(path, coordinates=self.__realCoordinates, boundaryConditions=self.__boundaryConditions )
    
    def get_pdb(self):
        """
        get a pdb instance of the last refined and save configuration state.
        
        :Returns:
            #. pdb (pdbParser): the pdb instance.
        """
        pdb = self.pdb.get_copy()
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
            self.__broadcaster.remove_listener(self.__groupSelector)
        # set new selector
        selector.set_engine(self)
        self.__groupSelector = selector
        # add to broadcaster listeners list
        self.__broadcaster.add_listener(self.__groupSelector)
    
    def clear_groups(self):
        """ Clear all engine defined groups
        """
        self.__groups = []
        
    def add_group(self, g, broadcast=True):
        """
        Add a group to engine groups list.
        
        :Parameters:
            #. g (Group, integer, list, set, tuple numpy.ndattay): Group instance, integer, 
               list, tuple, set or numpy.ndarray of atoms indexes of atoms indexes.
            #. broadcast (boolean): Whether to broadcast "update groups". Keep True unless you know what you are doing.
        """
        if isinstance(g, Group):
            assert np.max(g.indexes)<len(self.__pdb), LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = g
        elif is_integer(g):
            g = INT_TYPE(g)
            assert g<len(self.__pdb), LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = Group(indexes= [g] )
        elif isinstance(g, (list, set, tuple)):
            sortedGroup = sorted(set(g))
            assert len(sortedGroup) == len(g), LOGGER.error("redundant indexes found in group")
            assert is_integer(sortedGroup[-1]), LOGGER.error("group indexes must be integers")
            assert sortedGroup[-1]<len(self.__pdb), LOGGER.error("group index must be smaller than number of atoms in pdb")
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
            assert sortedGroup[-1]<len(self.__pdb), LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = Group(indexes= g )
        # append group
        self.__groups.append( gr )
        # set group engine
        gr._set_engine(self)
        # broadcast to constraints
        if broadcast:
            self.__broadcaster.broadcast("update groups")
      
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
        self.__groups = []
        if groups is None:
            self.__groups = [Group(indexes=[idx]) for idx in self.__pdb.indexes]
        elif isinstance(groups, Group):
            self.add_group(groups, broadcast=False)
        else:
            assert isinstance(groups, (list,tuple,set)), LOGGER.error("groups must be a None, Group, list, set or tuple")
            for g in groups:
                self.add_group(g, broadcast=False)
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
    
    def add_groups(self, groups):
        """
        Add groups to engine.
        
        :Parameters:
            #. groups (Group, list): Group instance or list of groups, where every group must be a Group instance or a numpy.ndarray of atoms indexes of type numpy.int32.
        """
        if isinstance(groups, Group):
            self.add_group(groups, broadcast=False)
        else:
            assert isinstance(groups, (list,tuple,set)), LOGGER.error("groups must be a list of numpy.ndarray")
            for g in groups:
                self.add_group(g, broadcast=False)
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
        
    def set_groups_as_atoms(self):
        """ Automatically set engine groups as single atom groups of all atoms. """
        self.set_groups(None)
        
    def set_groups_as_molecules(self):
        """ Automatically set engine groups indexes according to molecules indexes. """
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
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
            
    def set_pdb(self, pdb, boundaryConditions=None, names=None, elements=None, moleculesIndexes=None, moleculesNames=None):
        """
        Sets the configuration pdb. Engine and constraintsData will be automatically reset
        
        :Parameters:
            #. pdb (pdbParser, string): the configuration pdb as a pdbParser instance or a path string to a pdb file.
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries, numpy.ndarray, number): The configuration's boundary conditions.
               If None, boundaryConditions are set to InfiniteBoundaries with no periodic boundaries.
               If numpy.ndarray is given, it must be pass-able to a PeriodicBoundaries. Normally any real numpy.ndarray of shape (1,), (3,1), (9,1), (3,3) is allowed.
               If number is given, it's like a numpy.ndarray of shape (1,), it is assumed as a cubic box of box length equal to number.
            #. names (None, list): All pdb atoms names list.
               If None names will be calculated automatically by parsing pdb instance.
            #. elements (None, list): All pdb atoms elements list.
               If None elements will be calculated automatically by parsing pdb instance.
            #. moleculesIndexes (None, list, numpy.ndarray): The molecules indexes list.
               If None moleculesIndexes will be calculated automatically by parsing pdb instance.
            #. moleculesNames (None, list): The molecules names list. Must have the length of the number of atoms.
               If None, it is automatically generated as the pdb residues name.
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
                raise Exception( LOGGER.error("pdb must be a pdbParser instance or a string path to a protein database (pdb) file.") )
        # set pdb
        self.__pdb = pdb        
        
        #### CONVERT PDB COORDINATES TO CARTESIAN CAN BE ADDED HERE ####
        ##### transform coords into cartesian orthonormal system
        ####self.__pdbBasis    = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=FLOAT_TYPE)
        ####self.__pdbRecBasis = get_reciprocal_basis(self.__pdbBasis)[0]
        ####RC = np.array(self.__pdb.coordinates, dtype=FLOAT_TYPE)  
        ####cartesianCoords = transform_coordinates(self.__pdbRecBasis, RC).astype(FLOAT_TYPE)
        
        # get coordinates
        self.__realCoordinates = np.array(self.__pdb.coordinates, dtype=FLOAT_TYPE) 
        # reset configuration state
        self.__state = time.time()
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
        Sets the configuration boundary conditions. Any type of periodic boundary conditions are allowed and not restricted to cubic.
        Engine and constraintsData will be automatically reset.
        
        :Parameters:
            #. boundaryConditions (None, InfiniteBoundaries, PeriodicBoundaries, numpy.ndarray, number): The configuration's boundary conditions.
               If None, boundaryConditions are set to InfiniteBoundaries with no periodic boundaries.
               If numpy.ndarray is given, it must be pass-able to a PeriodicBoundaries. Normally any real numpy.ndarray of shape (1,), (3,1), (9,1), (3,3) is allowed.
               If number is given, it's like a numpy.ndarray of shape (1,), it is assumed as a cubic box of box length equal to number.
        """
        if boundaryConditions is None:
            boundaryConditions = InfiniteBoundaries()
        if is_number(boundaryConditions) or isinstance(boundaryConditions, (list, tuple, np.ndarray)):
            self.__canSetNumberDensity = False
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
            self.__canSetNumberDensity = True
            coordsCenter = np.sum(self.__realCoordinates, axis=0)/self.__realCoordinates.shape[0]
            coordinates  = self.__realCoordinates-coordsCenter
            distances    = np.sqrt( np.sum(coordinates**2, axis=1) )
            maxDistance  = 2.*np.max(distances) 
            cubixBoxLength = np.ceil( 3.*maxDistance )
            bc = PeriodicBoundaries()
            bc.set_vectors(cubixBoxLength)
            self.__boundaryConditions = bc
            self.__basisVectors = np.array(bc.get_vectors(), dtype=FLOAT_TYPE)
            self.__reciprocalBasisVectors = np.array(bc.get_reciprocal_vectors(), dtype=FLOAT_TYPE)
            #self.__volume = FLOAT_TYPE(bc.get_box_volume()) 
            self.__volume = FLOAT_TYPE( 1./.0333679 * self.numberOfAtoms )      
            LOGGER.warn("Not periodic but InfiniteBoundaries boundary conditions is not directly implemented yet. \
Therefore cubic PeriodicBoundaries is automatically created with size '%s Angstroms' equal to 3 times the system size. \
System's number density is set to '.0333679' which is the same as water. The system's volume is then computed \
equal to %.6f. If this is incorrect, use set_number_density method to set the system's correct number density \
and therefore the volume."%(cubixBoxLength,self.__volume,))
        elif isinstance(boundaryConditions, PeriodicBoundaries):
            self.__canSetNumberDensity = False
            self.__boundaryConditions = boundaryConditions
            self.__basisVectors = np.array(boundaryConditions.get_vectors(), dtype=FLOAT_TYPE)
            self.__reciprocalBasisVectors = np.array(boundaryConditions.get_reciprocal_vectors(), dtype=FLOAT_TYPE)
            self.__volume = FLOAT_TYPE(boundaryConditions.get_box_volume())
        else:
            raise Exception( LOGGER.error("Unkown boundary conditions. boundaryConditions must be an InfiniteBoundaries or PeriodicBoundaries instance or a valid vectors numpy.array or a positive number") )
        # get box coordinates
        if isinstance(self.__boundaryConditions, PeriodicBoundaries):
            self.__boxCoordinates = transform_coordinates(transMatrix=self.__reciprocalBasisVectors , coords=self.__realCoordinates)
        else:
            self.__boxCoordinates = None
        # check box coordinates for none periodic boundary conditions
        if self.__boxCoordinates is None:
            raise Exception( LOGGER.error("Not periodic boundary conditions is not implemented yet") )
        # set number density
        self.__numberDensity = FLOAT_TYPE(self.numberOfAtoms) / FLOAT_TYPE(self.__volume)
        # broadcast to constraints
        self.__broadcaster.broadcast("update boundary conditions")
    
    def set_number_density(self, numberDensity):
        """   
        Sets system's number density. This is used to correct system's
        volume. It can be used only with InfiniteBoundaries. 
        
        :Parameters:
            #. numberDensity (number): The number density value. 
        """
        if isinstance(self.__boundaryConditions, InfiniteBoundaries) and not isinstance(self.__boundaryConditions, PeriodicBoundaries):
            LOGGER.warn("Setting number density is not when boundary conditions are periodic.") 
            return
        if not self.__canSetNumberDensity:
            LOGGER.warn("Setting number density is not when boundary conditions are periodic.") 
            return
        assert is_number(numberDensity), LOGGER.error("numberDensity must be a number.")
        numberDensity = FLOAT_TYPE(numberDensity)
        assert numberDensity>0, LOGGER.error("numberDensity must be bigger than 0.")
        if numberDensity>1: 
            LOGGER.warn("numberDensity value is %.6f value isn't it too big?"%numberDensity)
        self.__numberDensity = numberDensity
        self.__volume = FLOAT_TYPE( 1./numberDensity * self.numberOfAtoms )      
        
    def set_molecules_indexes(self, moleculesIndexes=None, moleculesNames=None):
        """
        Sets moleculesIndexes list, assigning each atom to a molecule.
        
        :Parameters:
            #. moleculesIndexes (None, list, numpy.ndarray): The molecules indexes list.
               If None moleculesIndexes will be calculated automatically by parsing pdb instance.
            #. moleculesNames (None, list): The molecules names list. Must have the length of the number of atoms.
               If None, it is automatically generated as the pdb residues name.
        """
        if not len(self.__pdb):
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
            assert len(moleculesIndexes)==len(self.__pdb), LOGGER.error("moleculesIndexes must have the same length as pdb")
            if isinstance(moleculesIndexes, np.ndarray):
                assert len(moleculesIndexes.shape)==1, LOGGER.error("moleculesIndexes numpy.ndarray must have a dimension of 1")
                assert moleculesIndexes.dtype.type is INT_TYPE, LOGGER.error("moleculesIndexes must be of type numpy.int32")
            else:
                for idx in moleculesIndexes:
                    try:
                        idx = float(idx)
                    except:
                        raise Exception(LOGGER.error("moleculesIndexes must be a list of numbers"))
                    assert is_integer(idx), LOGGER.error("moleculesIndexes must be a list of integers")
        # check molecules names
        if moleculesNames is not None:
            assert isinstance(moleculesNames, (list, set, tuple)), LOGGER.error("moleculesNames must be a list")
            moleculesNames = list(moleculesNames)
            assert len(moleculesNames)==len(self.__pdb), LOGGER.error("moleculesNames must have the same length as pdb")
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
        # broadcast to constraints
        self.__broadcaster.broadcast("update molecules indexes")

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
            assert len(elements)==len(self.__pdb), LOGGER.error("elements have the same length as pdb")
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
        # broadcast to constraints
        self.__broadcaster.broadcast("update elements indexes")
        
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
            assert len(names)==len(self.__pdb), LOGGER.error("names have the same length as pdb")
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
        # broadcast to constraints
        self.__broadcaster.broadcast("update names indexes")
    
    def visualize(self, commands=None, foldIntoBox=False, boxToCenter=False,
                        boxWidth=2, boxStyle="solid", boxColor="yellow", 
                        bckColor="black", displayParams=None, 
                        representationParams="Lines", otherParams=None):
        """
        Visualize the last configuration using pdbParser visualize_vmd method.
        
        :Parameters:
            #. commands (None, list, tuple): List of commands to pass upon calling vmd.
               commands can be a .dcd file to load a trajectory for instance.
            #. foldIntoBox (boolean): Whether to fold all atoms into simulation box before visualization.
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
            #. bckColor (str): Set the background color.
            #. displayParams(None, dict): Set the display parameters. If None, default parameters will be applied.
               If dictionary the following keys can be used.\n
               * 'depth cueing' (default True): Set the depth cueing flag.
               * 'cue density' (default 0.1): Set the depth density.
               * 'cue mode' (default 'Exp'): Set the depth mode among 'linear', 'Exp' and 'Exp2'.
            #. representationParams(str): Set representation method among the following:\n
               Lines, Bonds, DynamicBonds, HBonds, Points, VDW, CPK, Licorice, Beads, Dotted, Solvent.
               And add parameters accordingly if needed. e.g.\n
               * Points representation accept only size parameter e.g. 'Points 5'
               * CPK representation can accept respectively 4 parameters as the following 'Sphere Scale',
                 'Bond Radius', 'Sphere Resolution', 'Bond Resolution' e.g. 'CPK 1.0 0.2 50 50'
               * VDW representation can accept respectively 2 parameters as the following 'Sphere Scale',
                 'Sphere Resolution' e.g. 'VDW 0.7 100'
            #. otherParams(None, list, set, tuple): Any other parameters in a form of a list of strings.\n
               e.g. ['display resize 700 700', 'rotate x to 45', 'scale to 0.02', 'axes location off']
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
        assert bckColor in colors, LOGGER.error("display background color is not a recognized color name among %s"%str(colors))
        assert depthCueing in [True, False], LOGGER.error("depth cueing must be boolean")
        assert is_number(cueDensity), LOGGER.error("cue density must be a number") 
        assert cueMode in ['linear','Exp','Exp2'], LOGGER.error("cue mode must be either 'linear','Exp' or 'Exp2'")
        fd.write('color Display Background %s \n'%bckColor)
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
        self.__pdb.visualize(commands=commands, coordinates=coords, startupScript=tclFile)
        # remove .tcl file
        os.remove(tclFile)
        
    def add_constraints(self, constraints):
        """
        Add constraints to the engine.
        
        :Parameters:
            #. constraints (Constraint, list, set, tuple): A constraint instance or list of constraints instances
        """
        if isinstance(constraints,(list,set,tuple)):
            constraints = list(constraints)
        else:
            constraints = [constraints]
        for c in constraints:
            assert isinstance(c, Constraint), LOGGER.error("constraints must be a Constraint instance or a list of Constraint instances")
            # check whether same instance added twice
            if c in self.__constraints:
                LOGGER.warn("constraint '%s' already exist in list of constraints"%c)
                continue
            # add engine to constraint
            c.set_engine(self)
            # add to broadcaster listeners list
            self.__broadcaster.add_listener(c)
            # check for singularity
            if isinstance(c, SingularConstraint):
                c.assert_singular()
            # add constraint to engine
            self.__constraints.append(c)
            # broadcast 'engine changed' to constraint
            c.listen("engine changed")
    
    def remove_constraints(self, constraints):
        """
        Remove constraints from engine list of constraints.
        
        :Parameters:
            #. constraints (Constraint, list, set, tuple): A constraint instance or list of constraints instances
        """
        if isinstance(constraints,(list,set,tuple)):
            constraints = list(constraints)
        else:
            constraints = [constraints]
        for c in constraints:
            if c in self.__constraints:
                self.__constraints.remove(c)
                c.set_engine(None)
                # add to broadcaster listeners list
                self.__broadcaster.remove_listener(c) 
       
    def reset_constraints(self):
        """ Reset constraints flags. """
        for c in self.__constraints:
            c.reset_constraint()
    
    def reset_engine(self):
        """ Re-initialize engine and resets constraints flags and data. """
        self._initialize_engine()
        # reset constraints flags
        self.reset_constraints()
    
    def compute_total_standard_error(self, constraints, current=True):
        """
        Computes the total standard error as the sum of all constraints' standard error.
        
        .. math::
            \\chi^{2} = \\sum \\limits_{i}^{N} (\\frac{stdErr_{i}}{variance_{i}})^{2}
          
        Where:\n    
        :math:`variance_{i}` is the variance value of the constraint i. \n
        :math:`stdErr_{i}` the standard error of the constraint i defined as :math:`\\sum \\limits_{j}^{points} (target_{i,j}-computed_{i,j})^{2} = (Y_{i,j}-F(X_{i,j}))^{2}` \n
             
        :Parameters:
            #. constraints (list): All constraints used to calculate total totalStandardError.
            #. current (bool): If True it uses constraints standardError argument, 
               False it uses constraint's afterMoveStandardError argument.
        
        :Returns:
            #. totalStandardError (list): The computed total total standard error.
        """
        if current:
            attr = "standardError"
        else:
            attr = "afterMoveStandardError"
        chis = []
        for c in constraints:
            SD = getattr(c, attr)
            assert SD is not None, LOGGER.error("constraint %s %s is not computed yet. Try to initialize constraint"%(c,attr))
            chis.append(SD/c.varianceSquared)
        return np.sum(chis)
    
    def set_total_standard_error(self):
        """
        Computes and sets the total totalStandardError of active constraints.
        """
        # get and initialize used constraints
        _usedConstraints, _constraints, _rigidConstraints = self.initialize_used_constraints()
        # compute totalStandardError
        self.__totalStandardError = self.compute_total_standard_error(_constraints, current=True)
        
    def get_used_constraints(self):
        """
        Parses all engine constraints and returns different lists of the active ones.
        
        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints list that will contribute to the engine total totalStandardError
            #. RigidConstraint (list): All active RigidConstraint constraints instances among usedConstraints list that won't contribute to the engine total totalStandardError
        """
        usedConstraints = []
        for c in self.__constraints:
            if c.used:
                usedConstraints.append(c)
        # get EnhanceOnlyConstraints list
        rigidConstraints = []
        constraints = []
        for c in usedConstraints:
            if isinstance(c, RigidConstraint):
                rigidConstraints.append(c)
            else:
                constraints.append(c)
        # return constraints
        return usedConstraints, constraints, rigidConstraints
        
    def initialize_used_constraints(self, force=False):
        """
        Calls get_used_constraints method, re-initializes constraints when needed and return them all.
        
        :parameters:
            #. force (bool): Whether to force initializing constraints regardless of their state.

        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints list that will contribute to the engine total totalStandardError
            #. RigidConstraint (list): All active RigidConstraint constraints instances among usedConstraints list that won't contribute to the engine total totalStandardError
        """
        assert isinstance(force, bool), LOGGER.error("force must be boolean")
        # get used constraints
        usedConstraints, constraints, rigidConstraints = self.get_used_constraints()
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
        
    def __runtime_get_save_engine(self, saveFrequency, savePath): 
        # check saveFrequency
        assert is_integer(saveFrequency), LOGGER.error("saveFrequency must be an integer")
        if saveFrequency is not None:
            assert is_integer(saveFrequency), LOGGER.error("saveFrequency must be an integer")
            assert saveFrequency>=0, LOGGER.error("saveFrequency must be positive")
            saveFrequency = int(saveFrequency)
        if saveFrequency == 0:
            saveFrequency = None
        # check savePath
        assert isinstance(savePath, basestring), LOGGER.error("savePath must be a string")
        savePath = str(savePath)
        # return
        return saveFrequency, savePath
    
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
        
    def run(self, numberOfSteps=100000, saveFrequency=1000, savePath="restart", 
                  xyzFrequency=None, xyzPath="trajectory.xyz"):
        """
        Run the Reverse Monte Carlo engine by performing random moves on engine groups.
        
        :Parameters:
            #. numberOfSteps (integer): The number of steps to run.
            #. saveFrequency (integer): Save engine every saveFrequency steps.
               Save will be omitted if totalStandardError has not decreased. 
            #. savePath (string): Save engine file path.
            #. xyzFrequency (None, integer): Save coordinates to .xyz file every xyzFrequency steps 
               regardless totalStandardError has decreased or not.
               If None, no .xyz file will be generated.
            #. xyzPath (string): Save coordinates to .xyz file.
        """
        # get arguments
        _numberOfSteps            = self.__runtime_get_number_of_steps(numberOfSteps)
        _saveFrequency, _savePath = self.__runtime_get_save_engine(saveFrequency, savePath)
        _xyzFrequency, _xyzPath   = self.__runtime_get_save_xyz(xyzFrequency, xyzPath)
        # create xyz file
        if _xyzFrequency is not None:
            _xyzfd = open(_xyzPath, 'a')
        # get and initialize used constraints
        _usedConstraints, _constraints, _rigidConstraints = self.initialize_used_constraints()
        if not len(_usedConstraints):
            LOGGER.warn("No constraints are used. Configuration will be randomized")
        # runtime initialize group selector
        self.__groupSelector._runtime_initialize()
        # runtime initialize constraints
        [c._runtime_initialize() for c in _usedConstraints]
        # compute totalStandardError
        self.__totalStandardError = self.compute_total_standard_error(_constraints, current=True)
        # initialize useful arguments
        _engineStartTime    = time.time()
        _lastSavedTotalStandardError = self.__totalStandardError
        _coordsBeforeMove   = None
        _moveTried          = False
        
        #   #####################################################################################   #
        #   #################################### RUN ENGINE #####################################   #
        LOGGER.info("Engine started %i steps, total standard error is: %.6f"%(_numberOfSteps, self.__totalStandardError) )
        for step in xrange(_numberOfSteps):
            # constraint runtime_on_step
            [c._runtime_on_step() for c in _usedConstraints]
            # increment generated
            self.__generated += 1
            # get group
            self.__lastSelectedGroupIndex = self.__groupSelector.select_index()
            group = self.__groups[self.__lastSelectedGroupIndex]
            # get atoms indexes
            groupAtomsIndexes = group.indexes
            # get move generator
            groupMoveGenerator = group.moveGenerator
            # get group atoms coordinates before applying move 
            if isinstance(groupMoveGenerator, SwapGenerator):
                groupAtomsIndexes = groupMoveGenerator.get_ready_for_move(groupAtomsIndexes)
                _coordsBeforeMove = np.array(self.__realCoordinates[groupAtomsIndexes], dtype=self.__realCoordinates.dtype)
            elif _coordsBeforeMove is None or not self.__groupSelector.isRecurring:
                _coordsBeforeMove = np.array(self.__realCoordinates[groupAtomsIndexes], dtype=self.__realCoordinates.dtype)
            elif self.__groupSelector.explore:
                if _moveTried:
                    _coordsBeforeMove = movedRealCoordinates
            elif not self.__groupSelector.refine:
                _coordsBeforeMove = np.array(self.__realCoordinates[groupAtomsIndexes], dtype=self.__realCoordinates.dtype)
            #else:
            #    raise Exception(LOGGER.critical("Unknown recurrence mode, unable to get coordinates before applying move."))
            # compute moved coordinates
            movedRealCoordinates = groupMoveGenerator.move(_coordsBeforeMove)
            movedBoxCoordinates  = transform_coordinates(transMatrix=self.__reciprocalBasisVectors , coords=movedRealCoordinates)

            ########################### compute rigidConstraints ############################
            rejectMove = False
            for c in _rigidConstraints:
                # compute before move
                c.compute_before_move(indexes = groupAtomsIndexes)
                # compute after move
                c.compute_after_move(indexes = groupAtomsIndexes, movedBoxCoordinates=movedBoxCoordinates)
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
                    c.reject_move(indexes=groupAtomsIndexes)
                # log generated move rejected before getting tried
                LOGGER.untried("Generated move %i is not tried"%self.__tried)
            ###################################### try move #######################################
            else:
                self.__tried += 1
                for c in _constraints:
                    # compute before move
                    c.compute_before_move(indexes = groupAtomsIndexes)
                    # compute after move
                    c.compute_after_move(indexes = groupAtomsIndexes, movedBoxCoordinates=movedBoxCoordinates)
            ################################ compute new totalStandardError ################################
                newTotalStandardError = self.compute_total_standard_error(_constraints, current=False)
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
                        c.reject_move(indexes=groupAtomsIndexes)
                    # log tried move rejected
                    LOGGER.rejected("Tried move %i is rejected"%self.__generated)
            ##################################### accept move #####################################
            else:
                self.__accepted  += 1
                # set selector move accepted
                self.__groupSelector.move_accepted(self.__lastSelectedGroupIndex)
                # constraints reject move
                for c in _usedConstraints:
                    c.accept_move(indexes=groupAtomsIndexes)
                # set new coordinates
                self.__realCoordinates[groupAtomsIndexes] = movedRealCoordinates
                self.__boxCoordinates[groupAtomsIndexes]  = movedBoxCoordinates
                # log new successful move
                triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
                acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
                LOGGER.accepted("Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - totStdErr:%.6f" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__totalStandardError))
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
                        self.save(_savePath)
            ############################### dump coords to xyz file ###############################
            if _xyzFrequency is not None:
                if not(step+1)%_xyzFrequency:
                    _xyzfd.write("%s\n"%self.__pdb.numberOfAtoms)
                    triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
                    acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
                    _xyzfd.write("Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - totStdErr:%.6f\n" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__totalStandardError))
                    frame = [self.__allNames[idx]+ " " + "%10.5f"%self.__realCoordinates[idx][0] + " %10.5f"%self.__realCoordinates[idx][1] + " %10.5f"%self.__realCoordinates[idx][2] + "\n" for idx in self.__pdb.xindexes]
                    _xyzfd.write("".join(frame)) 
                    
        #   #####################################################################################   #
        #   ################################# FINISH ENGINE RUN #################################   #        
        LOGGER.info("Engine finishes executing all '%i' steps in %s" % (_numberOfSteps, get_elapsed_time(_engineStartTime, format="%d(days) %d:%d:%d")))
        # close .xyz file
        if _xyzFrequency is not None:
            _xyzfd.close()
        
        