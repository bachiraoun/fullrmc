"""
Agitations contains all MoveGenerator classes that agitate and shake structures such as distances, angles, etc.

.. inheritance-diagram:: fullrmc.Generators.Agitations
    :parts: 1 

+------------------------------------------------------+------------------------------------------------------+
|.. figure:: distanceAgitation.png                     |                                                      |
|   :width: 375px                                      |                                                      |
|   :height: 300px                                     |                                                      |
|   :align: left                                       |                                                      |
|                                                      |                                                      |
|   Random H-H bond length agitations generated on     |                                                      |
|   dihydrogen molecules. At room temperature, H2      |                                                      |
|   molecule bond length fluctuates around 0.74        |                                                      |
|   Angstroms. Red hydrogen atoms represent the shrank |                                                      |
|   H-H bond length molecule while blue hydrogen atoms |                                                      |
|   represent the expanded H-H bond length molecules.  |                                                      |
|   (:class:`DistanceAgitationGenerator`)              |                                                      |
+------------------------------------------------------+------------------------------------------------------+
 
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, generate_random_float
from fullrmc.Core.Collection import is_number, is_integer
from fullrmc.Core.MoveGenerator import MoveGenerator, PathGenerator


class DistanceAgitationGenerator(MoveGenerator):
    """
    Generates random agitation moves upon a distance separating two atoms by translating
    both atoms away from each other or closer to each other along the direction line between them.
    This is mainly used to shake two atoms bond distance by increasing and decreasing 
    the bond length.
     
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number):  The maximum translation amplitude in Angstroms applied on every atom.
        #. symmetric (bool): Whether to apply the same amplitude of translation on both atoms or not.
        #. shrink (None, bool): Whether to always shrink the distance or expand it.
           If True, moves will always bring atoms closer to each other.
           If False, moves will always bring atoms away from each other.
           If None, no orientation is forced, therefore atoms can randomly get closer to each other or away from each other.         
    """
    def __init__(self, group=None, amplitude=0.2, symmetric=True, shrink=None):
        super(DistanceAgitationGenerator, self).__init__(group=group)
        # set amplitude
        self.set_amplitude(amplitude)
        # set symmetric
        self.set_symmetric(symmetric)
        # set shrink
        self.set_shrink(shrink)
        
    @property
    def amplitude(self):
        """Gets the maximum agitation amplitude."""
        return self.__amplitude
    
    @property
    def shrink(self):
        """Gets shrink flag value."""
        return self.__shrink
    
    @property
    def symmetric(self):
        """Gets symmetric flag value."""
        return self.__symmetric
    
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)!=2:
            return False, "two atoms are needed in a group to perform bond distance movements."
        else:
            return True, "" 
            
    def set_amplitude(self, amplitude):
        """
        Sets maximum translation vector allowed amplitude.
        
        :Parameters:
            #. amplitude (number): the maximum allowed translation vector amplitude.
        """
        assert is_number(amplitude), log.LocalLogger("fullrmc").logger.error("Translation amplitude must be a number")
        amplitude = float(amplitude)
        assert amplitude>0, log.LocalLogger("fullrmc").logger.error("Translation amplitude must be bigger than 0")
        self.__amplitude = FLOAT_TYPE(amplitude)
    
    def set_symmetric(self, symmetric):
        """
        Sets symmetric flag value.
        
        :Parameters:
            #. symmetric (bool): Whether to apply the same amplitude of translation on both atoms or not.         
        """
        assert isinstance(symmetric, bool), log.LocalLogger("fullrmc").logger.error("symmetric must be boolean")
        self.__symmetric = symmetric
    
    def set_shrink(self, shrink):
        """
        Sets shrink flag value.
        
        :Parameters:
            #. shrink (None, bool): Whether to always shrink the distance or expand it.
               If True, moves will always bring atoms closer to each other.
               If False, moves will always bring atoms away from each other.
               If None, no orientation is forced, therefore distance can increase or decrease randomly at every step.           
        """
        assert shrink in (None, True, False), log.LocalLogger("fullrmc").logger.error("shrink can only be None, True or False")
        self.__shrink = shrink
 
    def transform_coordinates(self, coordinates, argument=None):
        """
        Translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # get normalized direction vector
        vector  = FLOAT_TYPE( coordinates[0,:]-coordinates[1,:] )
        vector /= FLOAT_TYPE( np.linalg.norm(vector) )
        # create amplitudes
        if self.__symmetric:
            amp0 = amp1 =  FLOAT_TYPE(generate_random_float()*self.__amplitude)
        else:
            amp0 =  FLOAT_TYPE(generate_random_float()*self.__amplitude)
            amp1 =  FLOAT_TYPE(generate_random_float()*self.__amplitude)
        # create shrink flag
        if self.__shrink is None:
            shrink = (1-2*generate_random_float())>0
        else:
            shrink = self.__shrink    
        # create directions
        if shrink:
            dir0 = FLOAT_TYPE(-1)
            dir1 = FLOAT_TYPE( 1)            
        else:
            dir0 = FLOAT_TYPE( 1)
            dir1 = FLOAT_TYPE(-1) 
        # create translation vectors
        translationVectors = np.empty((2,3), dtype=FLOAT_TYPE)
        translationVectors[0,:] = dir0*amp0*vector
        translationVectors[1,:] = dir1*amp1*vector
        # translate and return
        return coordinates+translationVectors
 

       