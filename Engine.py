"""
Engine is the fullrmc main module. It contains 'Engine' the main class of fullrmc which is the 
Reverse Monte Carlo artist. The engine class takes only Protein Data Bank formatted files '.pdb' 
(http://deposit.rcsb.org/adit/docs/pdb_atom_format.html) as atomic/molecular input structure. 
It handles and fits simultaneously many experimental data while controlling the evolution of the 
system using user-defined molecular or atomistic constraints such as bond-length, bond-angles, 
inter-molecular-distances, etc. 
"""

# standard libraries imports
import os
import time
import sys
import warnings
import atexit
from collections import OrderedDict

# external libraries imports
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries

# fullrmc library imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, generate_random_float
from fullrmc.Core.transform_coordinates import transform_coordinates
from fullrmc.Core.Collection import Broadcaster, is_number, is_integer, get_elapsed_time
from fullrmc.Core.Constraint import Constraint, SingularConstraint, EnhanceOnlyConstraint
from fullrmc.Core.Group import Group
from fullrmc.Core.GroupSelector import GroupSelector
from fullrmc.Selectors.RandomSelectors import RandomSelector


class Engine(object):
    """ 
    The Reverse Monte Carlo (RMC) engine, used to launched an RMC simulation. 
    It has the capability to use and fit simultaneously multiple sets of experimental data. 
    One can also define other constraints such as distances, bonds length, angles and many others.   
    
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
    """
    
    def __init__(self, pdb, boundaryConditions=None,
                       names=None, elements=None, 
                       moleculesIndexes=None, moleculesNames=None,
                       groups=None, groupSelector=None, 
                       constraints=None, tolerance=0.):
        # initialize
        self.__broadcaster = Broadcaster()
        self.__constraints = []
        self.__state = None
        # initialize engine flags and arguments
        self.__initialize_engine__()
        # set pdb
        self.set_pdb(pdb=pdb, boundaryConditions=boundaryConditions, elements=elements, moleculesIndexes=moleculesIndexes, moleculesNames=moleculesNames)
        # set set groups
        self.set_groups(groups)
        # set set groups
        self.__groupSelector = None
        self.set_group_selector(groupSelector)
        # set tolerance
        self.set_tolerance(tolerance)
        # set constraints
        if constraints is not None:
            self.add_constraints(constraints)

    def __initialize_engine__(self):
        """ Initialize all engine arguments and flags. """
        # engine last moved group index
        self.__lastMovedGroupIndex = None
        # engine generated steps number
        self.__generated = 0
        # engine tried steps number
        self.__tried = 0
        # engine accepted steps number
        self.__accepted = 0
        # engine tolerated steps number
        self.__tolerated = 0
        # current model total chiSquare from constraints
        self.__chiSquare = None
        # grouping atoms into list of indexes arrays. All atoms of same group evolve together upon engine run time
        self.__groups = []
        
    @property
    def lastMovedGroupIndex(self):
        """ Get the last moved group instance index in groups list. """
        return self.__lastMovedGroupIndex
    
    @property
    def lastMovedGroup(self):
        """ Get the last moved group instance. """
        if self.__lastMovedGroupIndex is None:
            return None
        return self.__groups[self.__lastMovedGroupIndex]
    
    @property
    def lastMovedAtomsIndexes(self):
        """ Get the last moved atoms indexes. """
        if self.__lastMovedGroupIndex is None:
            return None
        return self.lastMovedGroup.indexes
        
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
        """ Get the number of tolerated steps in spite of increasing total chiSquare"""
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
        """ Get the defined elements indexes list. """
        return self.__elementsIndexes
    
    @property
    def elements(self):
        """ Get the defined elements set. """
        return self.__elements
    
    @property
    def allElements(self):
        """ Get all atoms elements. """
        return self.__allElements
        
    @property
    def namesIndexes(self):
        """ """
        return self.__namesIndexes
        
    @property
    def names(self):
        """ Get the defined atom names set. """
        return self.__names
    
    @property
    def allNames(self):
        """ Get all atoms names. """
        return self.__allNames
        
    @property
    def numberOfNames(self):
        """ Get the number of defined atom names set. """
        return len(self.__names)
    
    @property
    def numberOfAtomsPerName(self):
        """ Get the number of atoms per name dictionary. """
        return self.__numberOfAtomsPerName
        
    @property
    def numberOfElements(self):
        """ Get the number of defined elements in the configuration. """
        return len(self.__elements)
     
    @property
    def numberOfAtomsPerElement(self):
        """ Get the number of atoms per element dictionary. """
        return self.__numberOfAtomsPerElement
    
    @property
    def constraints(self):
        """ Get a list copy of all constraints instances. """
        return [c for c in self.__constraints]
    
    @property
    def groupSelector(self):
        """ Get the group selector instance. """
        return self.__groupSelector
        
    @property
    def chiSquare(self):
        """ Get the last recorded chiSquare of the current configuration. """
        return self.__chiSquare
    
    def save(self, path):
        """
        Save engine to disk.
        
        :Parameters:
            #. path (string): the file path to save the engine
        """
        log.LocalLogger("fullrmc").logger.info("Saving Engine... DON'T INTERRUPT")
        dirPath  = os.path.os.path.dirname(path)
        fileName = os.path.basename(path)
        fileName = fileName.split(".")[0]
        assert len(fileName), log.LocalLogger("fullrmc").logger.error("Path filename must be a non zero string")
        fileName += ".rmc"
        path = os.path.join(dirPath,fileName)
        # open file
        try:
            fd = open(path, 'wb')
        except Exception as e:
            raise Exception( log.LocalLogger("fullrmc").logger.error("Unable to open file '%s' to save engine. (%s)"%(path,e)) )
        # save engine
        try:
            pickle.dump( self, fd, protocol=pickle.HIGHEST_PROTOCOL )
        except Exception as e:
            fd.close()
            raise Exception( log.LocalLogger("fullrmc").logger.error("Unable to save engine instance. (%s)"%e) )
        finally:
            fd.close()
        log.LocalLogger("fullrmc").logger.info("Engine saved '%s'"%path)
    
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
            raise Exception(log.LocalLogger("fullrmc").logger.error("Can't open '%s' file for reading. (%s)"%(path,e) ))
        # unpickle file
        try:
            engine = pickle.load( fd )
        except Exception as e:
            fd.close()
            raise Exception( log.LocalLogger("fullrmc").logger.error("Unable to open fullrmc engine file '%s'. (%s)"%(path,e)) )
        finally:
            fd.close()
        assert isinstance(engine, Engine), log.LocalLogger("fullrmc").logger.error("%s is not a fullrmc Engine file"%path)
        # return engineInstance
        return engine
              
    def export_pdb(self, path):
        """
        Export a pdb file for the last configuration state
        
        :Parameters:
            #. path (string): the pdb file path.
        """
        self.pdb.export_pdb(path, coordinates=self.__realCoordinates, boundaryConditions=self.__boundaryConditions )

    def set_tolerance(self, tolerance):
        """   
        Sets the runtime engine tolerance value.
        
        :Parameters:
            #. tolerance (number): The runtime tolerance parameters. 
               It's the percentage of allowed unsatisfactory 'tried' moves. 
        """
        assert is_number(tolerance), log.LocalLogger("fullrmc").logger.error("tolerance must be a number")
        tolerance = FLOAT_TYPE(tolerance)
        assert tolerance>=0, log.LocalLogger("fullrmc").logger.error("tolerance must be positive")
        assert tolerance<=100, log.LocalLogger("fullrmc").logger.error("tolerance must be smaller than 100")
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
            assert isinstance(selector, GroupSelector), log.LocalLogger("fullrmc").logger.error("selector must a GroupSelector instance")
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
        
    def set_groups(self, groups):
        """
        Sets the engine groups of indexes.
        
        :Parameters:
            #. groups (None, list): list of groups, where every group must be a numpy.ndarray of atoms indexes of type numpy.int32.
               If None, single atom groups of all atoms will be all automatically created.
        """
        if groups is None:
            self.__groups = [Group(indexes=[idx]) for idx in self.__pdb.indexes]
        else:
            assert isinstance(groups, (list,tuple,set)), log.LocalLogger("fullrmc").logger.error("groups must be a list of numpy.ndarray")
            self.__groups = []
            for g in groups:
                # check group type
                if isinstance(g, Group):
                    assert np.max(g.indexes)<len(self.__pdb), log.LocalLogger("fullrmc").logger.error("group index must be smaller than number of atoms in pdb")
                    gr = g
                else:
                    assert isinstance(g, (np.ndarray)), log.LocalLogger("fullrmc").logger.error("each group in groups must be a numpy.ndarray or fullrmc Group instance")
                    # check group dimension
                    assert len(g.shape) == 1, log.LocalLogger("fullrmc").logger.error("each group must be a numpy.ndarray of dimension 1")
                    if len(g)==0:
                        continue
                    # check type
                    assert g.dtype.type is INT_TYPE, log.LocalLogger("fullrmc").logger.error("each group in groups must be of type numpy.int32")
                    # sort and check limits
                    g = sorted(set(g))
                    assert g[0]>=0, log.LocalLogger("fullrmc").logger.error("group index must equal or bigger than 0")
                    assert g[-1]<len(self.__pdb), log.LocalLogger("fullrmc").logger.error("group index must be smaller than number of atoms in pdb")
                    gr = Group(indexes=g)
                # append group
                self.__groups.append( gr )
        # set group engine
        [gr._set_engine(self) for gr in self.__groups]
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
        
    def reset_groups_as_molecules(self):
        """ Reset engine groups indexes according to molecules indexes. """
        molecules = list(set(self.__moleculesIndexes))
        moleculesIndexes = {}
        for idx in range(len(self.__moleculesIndexes)):
            mol = self.__moleculesIndexes[idx]
            if not moleculesIndexes.has_key(mol):
                moleculesIndexes[mol] = []
            moleculesIndexes[mol].append(idx)
        # create groups
        keys = sorted(moleculesIndexes.keys())
        self.__groups = [Group(indexes=moleculesIndexes[k]) for k in keys] 
        # set group engine
        [gr._set_engine(self) for gr in self.__groups]
        # broadcast to constraints
        self.__broadcaster.broadcast("update groups")
            
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
        if is_number(boundaryConditions) or isinstance(boundaryConditions, np.ndarray):
            try:   
                bc = PeriodicBoundaries()
                bc.set_vectors(boundaryConditions)
                self.__boundaryConditions = bc
                self.__basisVectors = np.array(bc.get_vectors(), dtype=FLOAT_TYPE)
                self.__reciprocalBasisVectors = np.array(bc.get_reciprocal_vectors(), dtype=FLOAT_TYPE)
                self.__volume = FLOAT_TYPE(bc.get_box_volume())
            except: 
                raise Exception( log.LocalLogger("fullrmc").logger.error("boundaryConditions must be an InfiniteBoundaries or PeriodicBoundaries instance or a valid vectors numpy array or a positive number") )
        elif not isinstance(boundaryConditions, PeriodicBoundaries):
            self.__boundaryConditions = boundaryConditions
            self.__basisVectors = None
            self.__reciprocalBasisVectors = None
            self.__volume = None
        elif isinstance(boundaryConditions, PeriodicBoundaries):
            self.__boundaryConditions = boundaryConditions
            self.__basisVectors = np.array(boundaryConditions.get_vectors(), dtype=FLOAT_TYPE)
            self.__reciprocalBasisVectors = np.array(boundaryConditions.get_reciprocal_vectors(), dtype=FLOAT_TYPE)
            self.__volume = FLOAT_TYPE(boundaryConditions.get_box_volume())
        # get box coordinates
        if isinstance(self.__boundaryConditions, PeriodicBoundaries):
            self.__boxCoordinates = np.array( self.__boundaryConditions.real_to_box_array(self.__realCoordinates), dtype=FLOAT_TYPE)
        else:
            self.__boxCoordinates = None
            raise Exception( log.LocalLogger("fullrmc").logger.error("Not periodic boundary conditions is not implemented yet") )
        # broadcast to constraints
        self.__broadcaster.broadcast("update boundary conditions")
        
    def visualize(self):
        """ Visualize the last configuration using pdbParser visualize method. """
        self.__pdb.visualize(coordinates=self.__realCoordinates)
        
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
                raise Exception( log.LocalLogger("fullrmc").logger.error("pdb must be a pdbParser instance or a string path to a protein database (pdb) file.") )
        # set pdb
        self.__pdb = pdb        
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
            assert isinstance(moleculesIndexes, (list,set,tuple, np.ndarray)), log.LocalLogger("fullrmc").logger.error("moleculesIndexes must be a list of indexes")
            assert len(moleculesIndexes)==len(self.__pdb), log.LocalLogger("fullrmc").logger.error("moleculesIndexes must have the same length as pdb")
            if isinstance(moleculesIndexes, np.ndarray):
                assert len(moleculesIndexes.shape)==1, log.LocalLogger("fullrmc").logger.error("moleculesIndexes numpy.ndarray must have a dimension of 1")
                assert moleculesIndexes.dtype.type is INT_TYPE, log.LocalLogger("fullrmc").logger.error("moleculesIndexes must be of type numpy.int32")
            else:
                for idx in moleculesIndexes:
                    try:
                        idx = float(idx)
                    except:
                        raise Exception(log.LocalLogger("fullrmc").logger.error("moleculesIndexes must be a list of numbers"))
                    assert is_integer(idx), log.LocalLogger("fullrmc").logger.error("moleculesIndexes must be a list of integers")
        # check molecules names
        if moleculesNames is not None:
            assert isinstance(moleculesNames, (list, set, tuple)), log.LocalLogger("fullrmc").logger.error("moleculesNames must be a list")
            moleculesNames = list(moleculesNames)
            assert len(moleculesNames)==len(self.__pdb), log.LocalLogger("fullrmc").logger.error("moleculesNames must have the same length as pdb")
        else:
            moleculesNames = self.__pdb.residues
        if len(moleculesNames):
            molName  = moleculesNames[0]
            molIndex = moleculesIndexes[0]
            for idx in range(len(moleculesIndexes)):
                newMolIndex = moleculesIndexes[idx]
                newMolName  = moleculesNames[idx]
                if newMolIndex == molIndex:
                    assert newMolName == molName, log.LocalLogger("fullrmc").logger.error("Same molecule atoms can't have different molecule name")
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
            assert isinstance(elements, (list,set,tuple)), log.LocalLogger("fullrmc").logger.error("elements must be a list of indexes")
            assert len(elements)==len(self.__pdb), log.LocalLogger("fullrmc").logger.error("elements have the same length as pdb")
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
            assert isinstance(names, (list,set,tuple)), log.LocalLogger("fullrmc").logger.error("names must be a list of indexes")
            assert len(names)==len(self.__pdb), log.LocalLogger("fullrmc").logger.error("names have the same length as pdb")
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
            assert isinstance(c, Constraint), log.LocalLogger("fullrmc").logger.error("constraints must be a Constraint instance or a list of Constraint instances")
            # check whether same instance added twice
            if c in self.__constraints:
                log.LocalLogger("fullrmc").logger.warn("constraint '%s' already exist in list of constraints"%c)
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
        self.__initialize_engine__()
        # reset constraints flags
        self.reset_constraints()
    
    def compute_chi_square(self, constraints, current=True):
        """
        Computes the total chiSquare of the given constraints.
        
        :Parameters:
            #. constraints (list): All constraints used to calculate total chiSquare.
            #. current (bool): If True it uses constraints chiSquare argument, 
               False it uses constraint's afterMoveChiSquare argument.
        
        :Returns:
            #. totalChiSquare (list): The computed total chiSquare.
        """
        if current:
            attr = "chiSquare"
        else:
            attr = "afterMoveChiSquare"
        chis = []
        for c in constraints:
            chi = getattr(c, attr)
            assert chi is not None, log.LocalLogger("fullrmc").logger.error("chiSquare for constraint %s is not computed yet. Try to initialize constraint"%c)
            chis.append(c.contribution*chi)
        return np.sum(chis)
    
    def __runtime_get_arguments__(self, numberOfSteps, saveFrequency, savePath):
        # check numberOfSteps
        assert is_integer(numberOfSteps), log.LocalLogger("fullrmc").logger.error("numberOfSteps must be an integer")
        assert numberOfSteps<=sys.maxint, log.LocalLogger("fullrmc").logger.error("number of steps must be smaller than maximum integer number allowed by the system '%i'"%sys.maxint)
        assert numberOfSteps>=0, log.LocalLogger("fullrmc").logger.error("number of steps must be positive")
        # check saveFrequency
        assert is_integer(saveFrequency), log.LocalLogger("fullrmc").logger.error("saveFrequency must be an integer")
        assert saveFrequency>0, log.LocalLogger("fullrmc").logger.error("saveFrequency must be bigger than 0")
        # check saveFrequency
        assert isinstance(savePath, basestring), log.LocalLogger("fullrmc").logger.error("savePath must be a string")
        savePath = str(savePath)
        # return
        return int(numberOfSteps), int(saveFrequency), savePath
    
    def get_used_constraints(self):
        """
        Parses all engine constraints and returns different lists of the active ones.
        
        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints list that will contribute to the engine total chiSquare
            #. EnhanceOnlyConstraint (list): All active EnhanceOnlyConstraint constraints instances among usedConstraints list that won't contribute to the engine total chiSquare
        """
        usedConstraints = []
        for c in self.__constraints:
            if c.used:
                usedConstraints.append(c)
        # get EnhanceOnlyConstraints list
        enhanceOnlyConstraints = []
        constraints = []
        for c in usedConstraints:
            if isinstance(c, EnhanceOnlyConstraint):
                enhanceOnlyConstraints.append(c)
            else:
                constraints.append(c)
        # return constraints
        return usedConstraints, constraints, enhanceOnlyConstraints
        
    def initialize_used_constraints(self):
        """
        Calls get_used_constraints method, re-initializes constraints when needed and return them all.
        
        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints list that will contribute to the engine total chiSquare
            #. EnhanceOnlyConstraint (list): All active EnhanceOnlyConstraint constraints instances among usedConstraints list that won't contribute to the engine total chiSquare
        """
        # get used constraints
        usedConstraints, constraints, enhanceOnlyConstraints = self.get_used_constraints()
        # initialize out-of-dates constraints
        for c in usedConstraints:
            if c.state != self.__state:
                log.LocalLogger("fullrmc").logger.info("Initializing constraint data '%s'"%c.__class__.__name__)
                c.compute_data()
                c.set_state(self.__state)
        # return constraints
        return usedConstraints, constraints, enhanceOnlyConstraints
        
    
    def run(self, numberOfSteps=sys.maxint, saveFrequency=1000, savePath="restart"):
        """
        Run the Reverse Monte Carlo engine by performing random moves on engine groups.
        
        :Parameters:
            #. numberOfSteps (integer): The maximum number of steps to run.
               By default maximum integer number allowed is given. 
               0 Will result in only initializing constraints.
            #. saveFrequency (integer): Save engine every saveFrequency steps.
            #. savePath (string): Save engine file path.
        """
        # get arguments
        _numberOfSteps, _saveFrequency, _savePath = self.__runtime_get_arguments__(numberOfSteps, saveFrequency, savePath)
        # get and initialize used constraints
        _usedConstraints, _constraints, _enhanceOnlyConstraints = self.initialize_used_constraints()
        if not len(_usedConstraints):
            log.LocalLogger("fullrmc").logger.warn("No constraints are used. Configuration will be randomize")
        # compute chiSquare
        self.__chiSquare = self.compute_chi_square(_constraints, current=True)
        # initialize useful arguments
        _engineStartTime    = time.time()
        _lastSavedChiSquare = self.__chiSquare
        _beforeMoveCoords   = None
        
        #   #####################################################################################   #
        #   #################################### RUN ENGINE #####################################   #
        log.LocalLogger("fullrmc").logger.info("Engine started chiSquare is: %.6f"%self.__chiSquare)
        for step in xrange(_numberOfSteps):
            # increment generated
            self.__generated += 1
            # get group
            self.__lastMovedGroupIndex = self.__groupSelector.select_index()
            group = self.__groups[self.__lastMovedGroupIndex]
            # get atoms indexes
            groupAtomsIndexes = group.indexes
            # get move generator
            groupMoveGenerator = group.moveGenerator
            # get before move coordinates
            if _beforeMoveCoords is None or not self.__groupSelector.isRefining:
                _beforeMoveCoords = np.array(self.__realCoordinates[groupAtomsIndexes], dtype=self.__realCoordinates.dtype)
            # compute moved coordinates
            movedRealCoordinates = groupMoveGenerator.move(_beforeMoveCoords)
            movedBoxCoordinates  = transform_coordinates(transMatrix=self.__reciprocalBasisVectors , coords=movedRealCoordinates)
            ########################### compute enhanceOnlyConstraints ############################
            rejectMove = False
            for c in _enhanceOnlyConstraints:
                # compute before move
                c.compute_before_move(indexes = groupAtomsIndexes)
                # compute after move
                c.compute_after_move(indexes = groupAtomsIndexes, movedBoxCoordinates=movedBoxCoordinates)
                # calculate rejectMove
                rejectMove = c.should_step_get_rejected(c.afterMoveChiSquare)
                if rejectMove:
                    break
            ##################################### reject move #####################################
            if rejectMove:
                for c in _enhanceOnlyConstraints:
                    c.reject_move(indexes=groupAtomsIndexes)
            ###################################### try move #######################################
            else:
                self.__tried += 1
                for c in _constraints:
                    # compute before move
                    c.compute_before_move(indexes = groupAtomsIndexes)
                    # compute after move
                    c.compute_after_move(indexes = groupAtomsIndexes, movedBoxCoordinates=movedBoxCoordinates)
            ################################ compute new chiSquare ################################
                newChiSquare = self.compute_chi_square(_constraints, current=False)
                if len(_constraints) and (newChiSquare >= self.__chiSquare):
                    if generate_random_float() > self.__tolerance:
                        rejectMove = True
                    else:
                        self.__tolerated += 1
                        self.__chiSquare  = newChiSquare
                else:
                    self.__chiSquare = newChiSquare
            ################################## reject tried move ##################################
            if rejectMove:
                for c in _constraints:
                    c.reject_move(indexes=groupAtomsIndexes)
            ##################################### accept move #####################################
            else:
                self.__accepted  += 1
                for c in _usedConstraints:
                    c.accept_move(indexes=groupAtomsIndexes)
                # set new coordinates
                self.__realCoordinates[groupAtomsIndexes] = movedRealCoordinates
                self.__boxCoordinates[groupAtomsIndexes]  = movedBoxCoordinates
                # log new successful move
                triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
                acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
                log.LocalLogger("fullrmc").logger.info("Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - ChiSquare:%.6f" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__chiSquare))
            ##################################### save engine #####################################
            if not(step+1)%_saveFrequency:
                if _lastSavedChiSquare==self.__chiSquare:
                    log.LocalLogger("fullrmc").logger.info("Save engine omitted because no improvement made since last save.")
                else:
                    # update state
                    self.__state  = time.time()
                    for c in _usedConstraints:
                       c.increment_tried()
                       c.set_state(self.__state)
                    # save engine
                    _lastSavedChiSquare = self.__chiSquare
                    self.save(_savePath)
        
        #   #####################################################################################   #
        #   ################################# FINISH ENGINE RUN #################################   #        
        log.LocalLogger("fullrmc").logger.info("Engine finishes executing all '%i' steps in %s" % (_numberOfSteps, get_elapsed_time(_engineStartTime, format="%d(days) %d:%d:%d")))

        

        