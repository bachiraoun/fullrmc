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
from fullrmc.Core.transform_coordinates import transform_coordinates
from fullrmc.Core.Collection import Broadcaster, is_number, is_integer, get_elapsed_time, generate_random_float
from fullrmc.Core.Constraint import Constraint, SingularConstraint, RigidConstraint
from fullrmc.Core.Group import Group
from fullrmc.Core.MoveGenerator import  SwapGenerator
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
        
        # set LOGGER file path to saveEngine path
        #logFile = os.path.join( os.path.dirname(os.path.realpath(_savePath)), LOGGER.logFileBasename)
        #LOGGER.set_log_file_basename(logFile)
        
    def __initialize_engine__(self):
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
        # current model total chiSquare from constraints
        self.__chiSquare = None
        # grouping atoms into list of indexes arrays. All atoms of same group evolve together upon engine run time
        self.__groups = []
    
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
    def numberOfAtoms(self):
        """ Get the number of atoms in pdb."""
        return len(self.__pdb)
        
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
        LOGGER.log("save engine","Saving Engine... DON'T INTERRUPT")
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
        LOGGER.log("save engine","Engine saved '%s'"%path)
    
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
        """
        Clear all engine defined groups
        """
        self.__groups = []
        
    def add_group(self, g, broadcast=True):
        """
        Add group to engine groups list.
        
        :Parameters:
            #. g (Group, numpy.ndattay): Group instance or a numpy.ndarray of atoms indexes of type fullrmc INT_TYPE.
            #. broadcast (boolean): Whether to broadcast "update groups". Keep True unless you know what you are doing.
        """
        if isinstance(g, Group):
            assert np.max(g.indexes)<len(self.__pdb), LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = g
        else:
            assert isinstance(g, np.ndarray), LOGGER.error("each group in groups must be a numpy.ndarray or fullrmc Group instance")
            # check group dimension
            assert len(g.shape) == 1, LOGGER.error("each group must be a numpy.ndarray of dimension 1")
            assert len(g), LOGGER.error("group found to have no indexes")
            # check type
            assert g.dtype.type is INT_TYPE, LOGGER.error("each group in groups must be of type numpy.int32")
            # sort and check limits
            sortedGroup = sorted(set(g))
            assert len(sortedGroup) == len(g), LOGGER.error("redundant indexes found in group")
            assert sortedGroup[0]>=0, LOGGER.error("group index must equal or bigger than 0")
            assert sortedGroup[-1]<len(self.__pdb), LOGGER.error("group index must be smaller than number of atoms in pdb")
            gr = Group(indexes=g)
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
            #. groups (None, list): list of groups, where every group must be a Group instance or 
               a numpy.ndarray of atoms indexes of type  fullrmc INT_TYPE.\n
               If None, single atom groups of all atoms will be all automatically created 
               which is the same as using set_groups_as_atoms method.
        """
        self.__groups = []
        if groups is None:
            self.__groups = [Group(indexes=[idx]) for idx in self.__pdb.indexes]
        elif isinstance(groups, Group):
            self.add_group(groups, broadcast=False)
        else:
            assert isinstance(groups, (list,tuple,set)), LOGGER.error("groups must be a list of numpy.ndarray")
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
                raise Exception( LOGGER.error("boundaryConditions must be an InfiniteBoundaries or PeriodicBoundaries instance or a valid vectors numpy array or a positive number") )
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
            raise Exception( LOGGER.error("Not periodic boundary conditions is not implemented yet") )
        # broadcast to constraints
        self.__broadcaster.broadcast("update boundary conditions")
        
    def visualize(self, boxWidth=2, boxColor="yellow", representation="Lines", params=""):
        """ Visualize the last configuration using pdbParser visualize method.
        
        :Parameters:
            #. boxWidth (number): Visualize the simulation box by giving the lines width.
               If 0 then the simulation box is not visualized.
            #. boxWidth (str): Choose the simulation box color among the following:\n
               blue, red, gray, orange, yellow, tan, silver, green,
               white, pink, cyan, purple, lime, mauve, ochre, iceblue, 
               black, yellow2, yellow3, green2, green3, cyan2, cyan3, blue2,
               blue3, violet, violet2, magenta, magenta2, red2, red3, 
               orange2, orange3.
            #. representation(str): Choose representation method among the following:\n
               Lines, Bonds, DynamicBonds, HBonds, Points, 
               VDW, CPK, Licorice, Beads, Dotted, Solvent.
            #. params(str): Set the representation parameters:\n
               Points representation accept only size parameter e.g. 5\n
               CPK representation accept respectively 4 parameters as the following 'Sphere Scale',
               'Sphere Resolution', 'Bond Radius', 'Bond Resolution' e.g. 0.5 20 0.1 20
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
        # check representation argument
        reps = ["Lines","Bonds","DynamicBonds","HBonds","Points","VDW","CPK",
                "Licorice","Beads","Dotted","Solvent"]
        assert representation in reps, LOGGER.error("representation is not a recognized among allowed ones %s"%str(reps))
        # create .tcl file
        (vmdfd, tclFile) = tempfile.mkstemp()
        # write tclFile
        tclFile += ".tcl"
        fd = open(tclFile, "w")
        # visualize box
        if boxWidth>0 and isinstance(self.__boundaryConditions, PeriodicBoundaries):
            try:
                a = self.__boundaryConditions.get_a()
                b = self.__boundaryConditions.get_b()
                c = self.__boundaryConditions.get_c()
                alpha = self.__boundaryConditions.get_alpha()*180./np.pi
                beta = self.__boundaryConditions.get_beta()*180./np.pi
                gamma = self.__boundaryConditions.get_gamma()*180./np.pi
                fd.write("set cell [pbc set {%.3f %.3f %.3f %.3f %.3f %.3f} -all]\n"%(a,b,c,alpha,beta,gamma))
                fd.write("pbc box -center origin -color %s -width %.2f\n"%(boxColor,boxWidth))
            except:
                LOGGER.warn("Unable to write simulation box .tcl script for visualization.") 
        # representation
        fd.write("mol delrep 0 top\n")
        fd.write("mol representation %s %s\n"%(representation, params) )
        fd.write("mol delrep 0 top\n")
        fd.write('mol addrep top\n')
        #fd.write("sel default style VDW\n")
        fd.close()
        self.__pdb.visualize(coordinates=self.__realCoordinates, startupScript=tclFile)
        # remove .tcl file
        os.remove(tclFile)
        
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
        self.__initialize_engine__()
        # reset constraints flags
        self.reset_constraints()
    
    def compute_chi_square(self, constraints, current=True):
        """
        Computes the total chiSquare of the given the squared deviations of the constraints.
        
        .. math::
            \\chi^{2} = \\sum \\limits_{i}^{N} (\\frac{SD_{i}}{variance_{i}})^{2}
          
        Where:\n    
        :math:`variance_{i}` is the variance value of the constraint i. \n
        :math:`SD_{i}` the squared deviations of the constraint i defined as :math:`\\sum \\limits_{j}^{points} (target_{i,j}-computed_{i,j})^{2} = (Y_{i,j}-F(X_{i,j}))^{2}` \n
             
        :Parameters:
            #. constraints (list): All constraints used to calculate total chiSquare.
            #. current (bool): If True it uses constraints squaredDeviations argument, 
               False it uses constraint's afterMoveSquaredDeviations argument.
        
        :Returns:
            #. totalChiSquare (list): The computed total chiSquare.
        """
        if current:
            attr = "squaredDeviations"
        else:
            attr = "afterMoveSquaredDeviations"
        chis = []
        for c in constraints:
            SD = getattr(c, attr)
            assert SD is not None, LOGGER.error("constraint %s %s is not computed yet. Try to initialize constraint"%(c,attr))
            chis.append(SD/c.varianceSquared)
        return np.sum(chis)
    
    def set_chi_square(self):
        """
        Computes and sets the total chiSquare of active constraints.
        """
        # get and initialize used constraints
        _usedConstraints, _constraints, _rigidConstraints = self.initialize_used_constraints()
        # compute chiSquare
        self.__chiSquare = self.compute_chi_square(_constraints, current=True)
        
    def get_used_constraints(self):
        """
        Parses all engine constraints and returns different lists of the active ones.
        
        :Returns:
            #. usedConstraints (list): All types of active constraints that will be used in engine runtime.
            #. constraints (list): All active constraints instances among usedConstraints list that will contribute to the engine total chiSquare
            #. RigidConstraint (list): All active RigidConstraint constraints instances among usedConstraints list that won't contribute to the engine total chiSquare
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
            #. constraints (list): All active constraints instances among usedConstraints list that will contribute to the engine total chiSquare
            #. RigidConstraint (list): All active RigidConstraint constraints instances among usedConstraints list that won't contribute to the engine total chiSquare
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
               Save will be omitted if chiSquare has not decreased. 
            #. savePath (string): Save engine file path.
            #. xyzFrequency (None, integer): Save coordinates to .xyz file every xyzFrequency steps 
               regardless chiSquare has decreased or not.
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
            LOGGER.warn("No constraints are used. Configuration will be randomize")
        # compute chiSquare
        self.__chiSquare = self.compute_chi_square(_constraints, current=True)
        # initialize useful arguments
        _engineStartTime    = time.time()
        _lastSavedChiSquare = self.__chiSquare
        _coordsBeforeMove   = None
        _moveTried          = False
        # initialize group selector
        self.__groupSelector._runtime_initialize()
        
        #   #####################################################################################   #
        #   #################################### RUN ENGINE #####################################   #
        LOGGER.info("Engine started %i steps, chiSquare is: %.6f"%(_numberOfSteps, self.__chiSquare) )
        for step in xrange(_numberOfSteps):
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
                rejectMove = c.should_step_get_rejected(c.afterMoveSquaredDeviations)
                #print c.__class__.__name__, c.squaredDeviations, c.afterMoveSquaredDeviations, rejectMove
                if rejectMove:
                    break
            _moveTried = not rejectMove
            ############################## reject move before trying ##############################
            if rejectMove:
                # rigidConstraints reject move
                for c in _rigidConstraints:
                    c.reject_move(indexes=groupAtomsIndexes)
                # log generated move rejected before getting tried
                LOGGER.log("move not tried","Generated move %i is not tried"%self.__tried)
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
                #if len(_constraints) and (newChiSquare >= self.__chiSquare):
                if newChiSquare > self.__chiSquare:
                    if generate_random_float() > self.__tolerance:
                        rejectMove = True
                    else:
                        self.__tolerated += 1
                        self.__chiSquare  = newChiSquare
                else:
                    self.__chiSquare = newChiSquare
            ################################## reject tried move ##################################
            if rejectMove:
                # set selector move rejected
                self.__groupSelector.move_rejected(self.__lastSelectedGroupIndex)
                if _moveTried:
                    # constraints reject move
                    for c in _constraints:
                        c.reject_move(indexes=groupAtomsIndexes)
                    # log tried move rejected
                    LOGGER.log("move rejected","Tried move %i is rejected"%self.__generated)
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
                LOGGER.log("move accepted","Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - ChiSquare:%.6f" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__chiSquare))
            ##################################### save engine #####################################
            if _saveFrequency is not None:
                if not(step+1)%_saveFrequency:
                    if _lastSavedChiSquare==self.__chiSquare:
                        LOGGER.info("Save engine omitted because no improvement made since last save.")
                    else:
                        # update state
                        self.__state  = time.time()
                        for c in _usedConstraints:
                           #c.increment_tried()
                           c.set_state(self.__state)
                        # save engine
                        _lastSavedChiSquare = self.__chiSquare
                        self.save(_savePath)
            ############################### dump coords to xyz file ###############################
            if _xyzFrequency is not None:
                if not(step+1)%_xyzFrequency:
                    _xyzfd.write("%s\n"%self.__pdb.numberOfAtoms)
                    triedRatio    = 100.*(float(self.__tried)/float(self.__generated))
                    acceptedRatio = 100.*(float(self.__accepted)/float(self.__generated))
                    _xyzfd.write("Generated:%i - Tried:%i(%.3f%%) - Accepted:%i(%.3f%%) - ChiSquare:%.6f\n" %(self.__generated , self.__tried, triedRatio, self.__accepted, acceptedRatio, self.__chiSquare))
                    frame = [self.__allNames[idx]+ " " + "%10.5f"%self.__realCoordinates[idx][0] + " %10.5f"%self.__realCoordinates[idx][1] + " %10.5f"%self.__realCoordinates[idx][2] + "\n" for idx in self.__pdb.xindexes]
                    _xyzfd.write("".join(frame)) 
                    
        #   #####################################################################################   #
        #   ################################# FINISH ENGINE RUN #################################   #        
        LOGGER.info("Engine finishes executing all '%i' steps in %s" % (_numberOfSteps, get_elapsed_time(_engineStartTime, format="%d(days) %d:%d:%d")))
        # close .xyz file
        if _xyzFrequency is not None:
            _xyzfd.close()
        
        