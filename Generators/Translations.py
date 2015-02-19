"""
Translations contains all translation like MoveGenerator classes.

.. inheritance-diagram:: fullrmc.Generators.Translations
    :parts: 2 
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, generate_random_float
from fullrmc.Core.Collection import is_number, is_integer, get_path, get_principal_axis
from fullrmc.Core.MoveGenerator import MoveGenerator, PathGenerator


class TranslationGenerator(MoveGenerator):
    """
    Generates random translations moves upon groups of atoms.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number):  The maximum translation amplitude in Angstroms.
    """
 
    def __init__(self, group=None, amplitude=0.2):
        super(TranslationGenerator, self).__init__(group=group)
        # set amplitude
        self.set_amplitude(amplitude)
        
    @property
    def amplitude(self):
        return self.__amplitude
        
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
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, ""
        
    def transform_coordinates(self, coordinates, argument=None):
        """
        Translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # generate random vector
        #vector = np.array(np.random.random(3)-np.random.random(3), dtype=FLOAT_TYPE)
        vector = np.array(1-2*np.random.random(3), dtype=FLOAT_TYPE)
        # normalize vector
        vector /= FLOAT_TYPE( np.linalg.norm(vector) )
        # amplify vector
        vector *= FLOAT_TYPE(np.random.random(1)[0]*self.__amplitude)
        # translate and return
        return coordinates+vector
        
        
class TranslationAlongAxisGenerator(TranslationGenerator):    
    """ 
    Generates random translation moves upon groups of atoms along a pre-defined axis.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint RMC engine.
        #. amplitude (number): The maximum allowed translation amplitude in Angstroms.
        #. axis (list,set,tuple,numpy.ndarray): The translation axis vector.
    """
    def __init__(self, group=None, amplitude=0.5, axis=0):
        super(TranslationAlongAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get translation axis."""
        return self.__axis
        
    def set_axis(self, axis):
        """
        Sets the axis along which the translation will be performed.
        
        :Parameters:
            #. axis (list,set,tuple,numpy.ndarray): The translation axis vector.
        """
        assert isinstance(axis, list,set,tuple,np.ndarray), log.LocalLogger("fullrmc").logger.error("axis must be a list")
        axis = list(axis)
        assert len(axis)==3, log.LocalLogger("fullrmc").logger.error("axis list must have 3 items")
        for pos in axis:
            assert is_number(pos), log.LocalLogger("fullrmc").logger.error( "axis items must be numbers")
        axis = [FLOAT_TYPE(pos) for pos in axis]
        axis =  np.array(axis, dtype=FLOAT_TYPE)
        self.__axis = axis/FLOAT_TYPE( np.linalg.norm(axis) )
    
    def transform_coordinates(self, coordinates, argument=None):
        """
        translates coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # get translation amplitude
        amplitude = (generate_random_float()-generate_random_float())*self.amplitude
        # amplify vector
        vector = self.__axis*FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector

        
class TranslationAlongSymmetryAxisGenerator(TranslationGenerator):    
    """ 
    Generates random translation moves upon groups of atoms along one of their symmetry axis.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number): the maximum translation angle in Angstroms.
        #. axis (integer): Must be 0,1 or 2 for respectively the mains, secondary or tertiary symmetry axis.
    """
    
    def __init__(self, group=None, amplitude=2, axis=0):
        super(TranslationAlongSymmetryAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get translation axis index."""
        return self.__axis
        
    def set_axis(self, axis):
        """
        Sets the symmetry axis index to translate along.
        
        :Parameters:
            #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis
        """
        assert is_integer(axis), log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be an integer")
        axis = INT_TYPE(axis)
        assert axis>=0, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be positive.")
        assert axis<=2, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be smaller or equal to 2")
        # convert to radian and store amplitude
        self.__axis = axis
    
    def transform_coordinates(self, coordinates, argument=None):
        """
        translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # get translation amplitude
        amplitude = (generate_random_float()-generate_random_float())*self.amplitude
        # get vector of translation
        _,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        vector = [X,Y,Z][self.__axis]
        # amplify vector
        vector *= FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector
    

class TranslationAlongSymmetryAxisPath(PathGenerator):    
    """ 
    Generates translation moves upon groups of atoms along one of their symmetry axis.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis 
        #. path (List): list of distances.
        #. randomize (boolean): Whether to pull moves randomly from path or pull moves in order at every step.
    """
    
    def __init__(self, group=None,  axis=0, path=None, randomize=False):
        # initialize PathGenerator
        PathGenerator.__init__(self, group=group, path=path, randomize=randomize)
        # set axis
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get translation axis index."""
        return self.__axis
        
    def set_axis(self, axis):
        """
        Sets the symmetry axis index to translate along.
        
        :Parameters:
            #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis
        """
        assert is_integer(axis), log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be an integer.")
        axis = INT_TYPE(axis)
        assert axis>=0, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be positive.")
        assert axis<=2, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be smaller or equal to 2.")
        # convert to radian and store amplitude
        self.__axis = axis
        
    def check_path(self, path):
        """
        Checks the generator's path.
        
        :Parameters:
            #. path (None, list): The list of moves.
        """
        if not isinstance(path, (list, set, tuple, np.ndarray)):
            return False, "path must be a list"
        path = list(path)
        if not len(path):
            return False, "path can't be empty"
        for distance in path:
            if not is_number(distance):
                return False, "path items must be numbers"
        return True, ""
        
    def normalize_path(self, path):
        """
        Transforms all path distances to floating numbers.
        
        :Parameters:
            #. path (list): The list of moves.
        
        :Returns:
            #. path (list): The list of moves.
        """
        return [FLOAT_TYPE(distance) for distance in path]
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, ""
        
    def transform_coordinates(self, coordinates, argument):
        """
        Rotate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (float): The move distance.
        """
        # get translation amplitude
        amplitude = FLOAT_TYPE(argument)
        # get vector of translation
        _,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        vector = [X,Y,Z][self.__axis]
        # amplify vector
        vector *= FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector
    

class TranslationCenterDirectionGenerator(TranslationGenerator):    
    """ 
    Generates random translation moves of every atom of the group along its direction vector to the geometric center of the group.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. center (None, numpy.array): The center value. 
           If None, then center is calculated as the center of geometry of group atoms.
           If numpy.array of indexes than center is calculated as the center of geometry of the given atoms indexes.
           If numpy.array of three float numbers, it is then considered the center.
        #. amplitude (number): The maximum translation amplitude in Angstroms.
        #. randomize (boolean): Whether randomize the amplitude and direction (towards or away from center) of translation of every atom in the group.
    """
    def __init__(self, group=None, center=None, amplitude=0.1, randomize=False):
        # initialize PathGenerator
        TranslationGenerator.__init__(self, group=group, amplitude=amplitude)
        # set axis
        self.set_randomize(randomize)
        # set axis
        self.set_center(center)
    
    @property
    def randomize(self):
        """ Get randomize value."""
        return self.__randomize
    
    @property
    def center(self):
        """ Get the center value."""
        return self.__center
        
    def set_randomize(self, randomize):
        """
        Sets randomize flag value.
        
        :Parameters:
            #. randomize (boolean): Whether randomize the amplitude and direction (towards or away from center) of translation of every atom in the group.
        """
        assert isinstance(randomize, bool), log.LocalLogger("fullrmc").logger.error("randomize must be boolean")
        self.__randomize = randomize
    
    def set_center(self, center):
        """
        Sets center value.
        
        :Parameters:
           #. center (None, numpy.array): The center value. 
           If None, then center is calculated as the center of geometry of group atoms.
           If numpy.array of indexes than center is calculated as the center of geometry of the given atoms indexes.
           If numpy.array of three float numbers, it is then considered the center.
        """
        #not implemented yet ! IT NEEDS ACCESS TO THE ENGINE IN SOME WAY
        pass
            
        
    def __amplify_vector__(self, vector):
        # amplify vectorsToCenter
        if self.__randomize:
            length = vector.shape[0]
            #amplitude = self.amplitude * np.array(np.random.random(length)-np.random.random(length), dtype=FLOAT_TYPE)
            amplitude = self.amplitude * np.array(1-2*np.random.random(length), dtype=FLOAT_TYPE)
            vector[:,0] *= amplitude
            vector[:,1] *= amplitude
            vector[:,2] *= amplitude
        else:
            amplitude = self.amplitude * FLOAT_TYPE(generate_random_float()-generate_random_float())
            vector *= amplitude
        return vector
            
    def transform_coordinates(self, coordinates, argument=None):
        """
        Translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # compute center
        center = np.mean(coordinates, 0)
        # compute vectorsToCenter
        vectorsToCenter = coordinates-center
        # normalize vectorsToCenter
        norm = np.linalg.norm(vectorsToCenter,axis=1)
        vectorsToCenter[:,0] /= norm
        vectorsToCenter[:,1] /= norm
        vectorsToCenter[:,2] /= norm
        # amplify vectorsToCenter
        vectorsToCenter = self.__amplify_vector__(vectorsToCenter)
        # translate and return
        return coordinates+vectorsToCenter
 

class ShrinkGenerator(TranslationCenterDirectionGenerator):    
    """ 
    Generates random translation of atoms in a group towards the group center.
    """
    def __amplify_vector__(self, vector):
        # amplify vectorsToCenter
        if self.randomize:
            length = vector.shape[0]
            amplitude = -self.amplitude * np.array(np.random.random(length), dtype=FLOAT_TYPE)
            vector[:,0] *= amplitude
            vector[:,1] *= amplitude
            vector[:,2] *= amplitude
        else:
            amplitude = -self.amplitude * FLOAT_TYPE(generate_random_float())
            vector *= amplitude
        return vector
        
class ExpandGenerator(TranslationCenterDirectionGenerator):    
    """ 
    Generates random translation of atoms in a group away from group center.
    """
    def __amplify_vector__(self, vector):
        if self.randomize:
            length = vector.shape[0]
            amplitude = self.amplitude * np.array(np.random.random(length), dtype=FLOAT_TYPE)
            vector[:,0] *= amplitude
            vector[:,1] *= amplitude
            vector[:,2] *= amplitude
        else:
            amplitude = self.amplitude * FLOAT_TYPE(generate_random_float())
            vector *= amplitude
        return vector        
        

       