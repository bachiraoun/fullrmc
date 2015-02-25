"""
Translations contains all translation like MoveGenerator classes.

.. inheritance-diagram:: fullrmc.Generators.Translations
    :parts: 1 
    
+-----------------------------------------------------+-----------------------------------------------------+
|.. figure:: 500randomTranslations.png                |.. figure:: 10TranslationsAlongAxis.png              |
|   :width: 375px                                     |   :width: 375px                                     |
|   :height: 300px                                    |   :height: 300px                                    |
|   :align: left                                      |   :align: left                                      |
|                                                     |                                                     |
|   Random translation vectors generated from         |   Random translation vectors generated from         |
|   atom at origin. (:class:`TranslationGenerator`)   |   atom at origin along a pre-defined axis.          |
|                                                     |   (:class:`TranslationAlongAxisGenerator`)          |
|                                                     |                                                     |
|                                                     |                                                     |
|                                                     |                                                     |
|                                                     |                                                     |
+-----------------------------------------------------+-----------------------------------------------------+
|.. figure:: translationAlongSymmetryAxis.png         | .. figure:: translationTowardsAxis.png              |
|   :width: 375px                                     |    :width: 375px                                    |
|   :height: 300px                                    |    :height: 300px                                   |
|   :align: left                                      |    :align: left                                     |
|                                                     |                                                     |
|   Random translation vector generated along a       |    Random translation vectors generated towards an  |
|   predefined axis or one of the symmetry axes of the|    axis within some maximum angle.                  |
|   hexane molecule and applied on all the molecule's |    Legend is formatted as axis (angle) (direction)  |
|   atoms at the same time.                           |    (:class:`TranslationTowardsAxisGenerator`        |
|   (:class:`TranslationAlongAxisGenerator`           |    :class:`TranslationTowardsSymmetryAxisGenerator`)|
|   :class:`TranslationAlongSymmetryAxisGenerator`)   |                                                     |
|                                                     |                                                     |
+-----------------------------------------------------+-----------------------------------------------------+
|.. figure:: translationTowardsCenter.png             |                                                     |
|   :width: 375px                                     |                                                     |
|   :height: 300px                                    |                                                     |
|   :align: left                                      |                                                     |
|                                                     |                                                     |
|   Random translation vectors generated towards a    |                                                     |
|   pre-defined center or towards the geometric center|                                                     |
|   of a group of atoms. Here 20 vectors are generated|                                                     |
|   within a maximum separation angle of 30 deg.      |                                                     |
|   (:class:`TranslationTowardsCenterGenerator`)      |                                                     |
|                                                     |                                                     |
|                                                     |                                                     |
+-----------------------------------------------------+-----------------------------------------------------+
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, generate_random_float
from fullrmc.Core.Collection import is_number, is_integer, get_path, get_principal_axis, generate_vectors_in_solid_angle
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
        #. axis (list,set,tuple,numpy.ndarray): The pre-defined translation axis vector.  
    """
    def __init__(self, group=None, amplitude=0.2, axis=(1,0,0)):
        super(TranslationAlongAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set axis
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
        assert isinstance(axis, (list,set,tuple,np.ndarray)), log.LocalLogger("fullrmc").logger.error("axis must be a list")
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


class TranslationTowardsAxisGenerator(TranslationAlongAxisGenerator):    
    """ 
    Generates random translation moves upon groups of atoms towards a pre-defined axis
    within a tolerance angle between translation vectors and the pre-defined axis.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint RMC engine.
        #. amplitude (number): The maximum allowed translation amplitude in Angstroms.
        #. axis (list,set,tuple,numpy.ndarray): The pre-defined translation axis vector.
        #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
           If None generated axis can be in the same direction of axis or in the opposite.
           If True all generated vectors are in the same direction of axis.
           If False all generated vectors are in the opposite direction of axis.
    """
    def __init__(self, group=None, amplitude=0.2, axis=(1,0,0), angle=30, direction=True):
        super(TranslationTowardsAxisGenerator, self).__init__(group=group, amplitude=amplitude, axis=axis)
        # set angle
        self.set_angle(angle)
        # set angle
        self.set_direction(direction)
    
    @property
    def angle(self):
        """ Get tolerance maximum angle in rad."""
        return self.__angle
    
    @property
    def direction(self):
        """ Get generated translation vectors direction."""
        return self.__direction
        
    def set_angle(self, angle):
        """
        Sets the tolerance maximum angle.
        
        :Parameters:
            #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        """
        assert is_number(angle), log.LocalLogger("fullrmc").logger.error("angle must be numbers")
        assert angle>0, log.LocalLogger("fullrmc").logger.error("angle must be positive")
        assert angle<=360, log.LocalLogger("fullrmc").logger.error("angle must be smaller than 360")
        self.__angle = FLOAT_TYPE(angle)*PI/FLOAT_TYPE(180.)
        
    def set_direction(self, direction):
        """
        Sets the generated translation vectors direction.
        
        :Parameters:
            #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
               If None generated axis can be in the same direction of axis or in the opposite.
               If True all generated vectors are in the same direction of axis.
               If False all generated vectors are in the opposite direction of axis.
        """
        assert direction in (None, True, False), log.LocalLogger("fullrmc").logger.error("direction can only be None, True or False")
        self.__direction = direction
    
    def transform_coordinates(self, coordinates, argument=None):
        """
        translates coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # generate vector
        vector = generate_vectors_in_solid_angle(direction=self.axis,
                                                 maxAngle=self.__angle,
                                                 numberOfVectors=1)[0]  
        # get translation amplitude
        if self.__direction is None:
            amplitude = (generate_random_float()-generate_random_float())*self.amplitude
        elif self.__direction:
            amplitude = generate_random_float()*self.amplitude
        else:
            amplitude = -generate_random_float()*self.amplitude
        # amplify vector
        vector = vector*FLOAT_TYPE(amplitude)
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
    
    def __init__(self, group=None, amplitude=0.2, axis=0):
        super(TranslationAlongSymmetryAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get translation axis index."""
        return self.__axis
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)<=1:
            return False, "At least two atoms needed in a group to perform translation along symmetry axis."
        else:
            return True, "" 
            
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
        amplitude = (1-2*generate_random_float())*self.amplitude
        # get vector of translation
        _,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        vector = [X,Y,Z][self.__axis]
        # amplify vector
        vector *= FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector
    

class TranslationTowardsSymmetryAxisGenerator(TranslationAlongSymmetryAxisGenerator):    
    """ 
    Generates random translation moves upon groups of atoms towards one of its symmetry
    axis within a tolerance angle between translation vectors and the axis.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint RMC engine.
        #. amplitude (number): The maximum allowed translation amplitude in Angstroms.
        #. axis (integer): Must be 0,1 or 2 for respectively the mains, secondary or tertiary symmetry axis.
        #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
           If None generated axis can be in the same direction of axis or in the opposite.
           If True all generated vectors are in the same direction of axis.
           If False all generated vectors are in the opposite direction of axis.
    """
    
    def __init__(self, group=None, amplitude=0.2, axis=0, angle=30, direction=True):
        super(TranslationTowardsSymmetryAxisGenerator, self).__init__(group=group, amplitude=amplitude, axis=axis)

    @property
    def angle(self):
        """ Get tolerance maximum angle in rad."""
        return self.__angle
    
    @property
    def direction(self):
        """ Get generated translation vectors direction."""
        return self.__direction
        
    def set_angle(self, angle):
        """
        Sets the tolerance maximum angle.
        
        :Parameters:
            #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        """
        assert is_number(angle), log.LocalLogger("fullrmc").logger.error("angle must be numbers")
        assert angle>0, log.LocalLogger("fullrmc").logger.error("angle must be positive")
        assert angle<=360, log.LocalLogger("fullrmc").logger.error("angle must be smaller than 360")
        self.__angle = FLOAT_TYPE(angle)*PI/FLOAT_TYPE(180.)
        
    def set_direction(self, direction):
        """
        Sets the generated translation vectors direction.
        
        :Parameters:
            #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
               If None generated axis can be in the same direction of axis or in the opposite.
               If True all generated vectors are in the same direction of axis.
               If False all generated vectors are in the opposite direction of axis.
        """
        assert direction in (None, True, False), log.LocalLogger("fullrmc").logger.error("direction can only be None, True or False")
        self.__direction = direction
    
    def transform_coordinates(self, coordinates, argument=None):
        """
        translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # get axis
        _,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        axis = [X,Y,Z][self.axis]
        # generate vector
        vector = generate_vectors_in_solid_angle(direction=axis,
                                                 maxAngle=self.__angle,
                                                 numberOfVectors=1)[0]  
        # get translation amplitude
        if self.__direction is None:
            amplitude = (generate_random_float()-generate_random_float())*self.amplitude
        elif self.__direction:
            amplitude = generate_random_float()*self.amplitude
        else:
            amplitude = -generate_random_float()*self.amplitude
        # amplify vector
        vector = vector*FLOAT_TYPE(amplitude)
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
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)<=1:
            return False, "At least two atoms needed in a group to perform translation along symmetry axis."
        else:
            return True, "" 
            
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
    

class TranslationTowardsCenterGenerator(TranslationGenerator):    
    """ 
    Generates random translation moves of every atom of the group along its direction vector to the geometric center of the group.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. center (dict): The center value dictionary. Must have a single key and this can only be 'fixed' or 'indexes'.
           If key is fixed, value must be a list or a numpy.array of a point coordinates such as [X,Y,Z]
           If key is indexes, value must be a list or a numpy array of indexes.
        #. amplitude (number): The maximum translation amplitude in Angstroms.
        #. angle (None, number): The maximum tolerance angle in degrees between a generated translation vector and the computed direction. 
           If None is given, all generated translation vectors will be along the direction to center.        
        #. direction (None, True, False): Whether to generate translation vectors pointing towards the center or not.
           If None generated axis can be randomly generated towards the center or away from the center.
           If True all generated vectors point towards the center.
           If False all generated vectors point away from the center.        
    """
    def __init__(self, group=None, center={"fixed":(0,0,0)}, amplitude=0.1, angle=30, direction=True):
        # initialize PathGenerator
        TranslationGenerator.__init__(self, group=group, amplitude=amplitude)
        # set direction
        self.set_direction(direction)
        # set point
        self.set_center(center)
        # set angle
        self.set_angle(angle)
    
    @property
    def direction(self):
        """ Get direction value."""
        return self.__direction
        
    @property
    def center(self):
        """ Get the center value."""
        return self.__center
    
    @property
    def angle(self):
        """ Get the angle value in rad."""
        return self.__angle    
    
    def set_direction(self, direction):
        """
        Sets the generated translation vectors direction.
        
        :Parameters:
            #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
               If None generated axis can be in the same direction of axis or in the opposite.
               If True all generated vectors are in the same direction of axis.
               If False all generated vectors are in the opposite direction of axis.
        """
        assert direction in (None, True, False), log.LocalLogger("fullrmc").logger.error("direction can only be None, True or False")
        self.__direction = direction
        
    def set_angle(self, angle):
        """
        Sets the tolerance maximum angle.
        
        :Parameters:
            #. angle (None, number): The maximum tolerance angle in degrees between a generated translation vector and the computed direction. 
               If None is given, all generated translation vectors will be along the direction to center.        
        """
        if angle is not None:
            assert is_number(angle), log.LocalLogger("fullrmc").logger.error("angle must be numbers")
            assert angle>0, log.LocalLogger("fullrmc").logger.error("angle must be positive")
            assert angle<=360, log.LocalLogger("fullrmc").logger.error("angle must be smaller than 360")
            angle = FLOAT_TYPE(angle)*PI/FLOAT_TYPE(180.)
        self.__angle = angle
         
    def set_center(self, center):
        """
        Sets center value.
        
        :Parameters:
           #. center (dict): The center value dictionary. Must have a single key and this can only be 'fixed' or 'indexes'.
              If key is fixed, value must be a list or a numpy.array of a point coordinates such as [X,Y,Z]
              If key is indexes, value must be a list or a numpy array of indexes.
        """
        assert isinstance(center, dict), log.LocalLogger("fullrmc").logger.error("center must be a dictionary")
        assert len(center) == 1, log.LocalLogger("fullrmc").logger.error("center must have a single key")       
        key = center.keys()[0]
        val = center[key]
        assert isinstance(val, (list,set,tuple,np.ndarray)), log.LocalLogger("fullrmc").logger.error("center must be a list")
        if isinstance(center, np.ndarray):
            assert len(center.shape) == 1, log.LocalLogger("fullrmc").logger.error("center value must have a single dimension")
        assert len(val)>0, log.LocalLogger("fullrmc").logger.error("center value must be a non-zero list.")
        for v in val:
            assert is_number(v), log.LocalLogger("fullrmc").logger.error("center value item must be numbers") 
        if key == "fixed":
            self.__mustCompute = False
            assert len(val) == 3, log.LocalLogger("fullrmc").logger.error("fixed center must have exactly 3 elements corresponding to X,Y and Z coordinates of the center point.")
            val = np.array([FLOAT_TYPE(v) for v in val], dtype=FLOAT_TYPE)
        elif key == "indexes":
            self.__mustCompute = True
            for v in val:
                assert is_integer(v), log.LocalLogger("fullrmc").logger.error("indexes center items be integers")
            val =  np.array([INT_TYPE(v) for v in val], dtype=INT_TYPE)
            for v in val:
                assert v>=0, log.LocalLogger("fullrmc").logger.error("indexes center items be positive integers")            
        else:
            self.__mustCompute = None
            raise Exception(log.LocalLogger("fullrmc").logger.error("center key must be either 'fixed' or 'indexes'"))        
        # set center
        self.__center = {key:val}
        
    def __get_amplitude__(self):
        # get translation amplitude
        if self.__direction is None:
            amplitude = (generate_random_float()-generate_random_float())*self.amplitude
        elif self.__direction:
            amplitude = generate_random_float()*self.amplitude
        else:
            amplitude = -generate_random_float()*self.amplitude
        return amplitude
    
    def __get_translation_vector__(self, direction):
        # normalize direction
        norm = np.sqrt(np.add.reduce(direction**2))
        direction[0] /= norm
        direction[1] /= norm
        direction[2] /= norm
        # get vector
        if self.__angle is None:
            vector = direction
        else:
            vector = generate_vectors_in_solid_angle(direction=direction,
                                                     maxAngle=self.__angle,
                                                     numberOfVectors=1)[0]  
        # return
        return vector
        
    def __get_center__(self):
        if self.__mustCompute:
            center  = self.group._get_engine().realCoordinates[self.__center["indexes"]]
            center = np.mean(center, axis=0).astype(FLOAT_TYPE)
        else:
            center  = self.__center["fixed"]
        return center
        
    def transform_coordinates(self, coordinates, argument=None):
        """
        Translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # get center
        center = self.__get_center__()
        # compute coordinates center
        coordsCenter = np.mean(coordinates, axis=0)
        direction    = center-coordsCenter
        # translation vector
        vector = self.__get_translation_vector__(direction)
        # amplify vector
        vector *= self.__get_amplitude__()
        # translate and return
        return coordinates+vector
 


       