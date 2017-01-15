"""        
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, LOGGER
from fullrmc.Core.Collection import is_integer, is_number, generate_random_integer
from fullrmc.Core.MoveGenerator import RemoveGenerator


class AtomsRemoveGenerator(RemoveGenerator):
    """  
    This generator allows removing single atoms at a time from the system. Atoms are
    randomly picked from atomsList and returned upon calling pick_from_list method.
    
    :Parameters:
        #. group (None, Group): The group instance which is this case must be 
           fullrmc EmptyGroup.
        #. maximumCollected (None, Integer): The maximum number allowed of atoms to be 
           removed and collected from the engine. This property is general to the 
           system and checks engine's collected atoms not the number of removed atoms 
           via this generator. If None is given, the remover will not check for the
           number of already removed atoms before attempting a remove.
        #. atomsList (None, list,set,tuple,np.ndarray): The list of atom indexes to chose
           and remove from.  If None, then all atoms in system will be used.     
    """
    def __init__(self, group=None, maximumCollected=None, atomsList=None):
        super(AtomsRemoveGenerator, self).__init__(group=group, 
                                                   maximumCollected=maximumCollected,
                                                   atomsList=atomsList)
    
    def pick_from_list(self, engine):
        """
        Randomly picks an atom's index from atomsList. This method checks and 
        verifies maximumCollected prior to picking and index. 
        
        :Parameters:
            #. engine (Engine): The engine calling the method.
            
        :Returns:
            #. index (None, np.ndarray): Atom index wrapped in a numpy.ndarray. 
               If None is returned than picking is not allowed.
        """
        if self.atomsList is None:
            # use len(engine.pdb) in order to select real index
            index = generate_random_integer(0, len(engine.pdb)-1)
            index = np.array([index], dtype=INT_TYPE)
        else:
            index = np.random.choice(self.atomsList,1)
        # check maximumCollected
        if self.maximumCollected is not None:   
            if len(engine._atomsCollector)>=self.maximumCollected:
                index = np.array([], dtype=INT_TYPE)
        # check if already collected 
        if len(index):
            if engine._atomsCollector.is_collected(index[0]):
                index = np.array([], dtype=INT_TYPE)
        # return
        return index

    
 

        