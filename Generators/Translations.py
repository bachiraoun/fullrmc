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

.. raw:: html                                             
                                                          
     <iframe width="560" height="315"                     
     src="https://www.youtube.com/embed/YRTrsDrVSvI?rel=0"
     frameborder="0" allowfullscreen>                     
     </iframe>
     
     <p></p>    
                                                          
     <iframe width="560" height="315"                     
     src="https://www.youtube.com/embed/Ik0RSQT4DzQ?rel=0"
     frameborder="0" allowfullscreen>                     
     </iframe>                                            
                                                         
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, get_principal_axis, generate_vectors_in_solid_angle, generate_random_float
from fullrmc.Core.MoveGenerator import MoveGenerator, PathGenerator


class TranslationGenerator(MoveGenerator):
    """
    Generates random translations moves upon groups of atoms.
     
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number, tuple): The translation amplitude in Angstroms.
           If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
           If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to random translations.
        # Maximum translation amplitude is set to 0.3A to all defined groups
        for g in ENGINE.groups:
            g.set_move_generator( TranslationGenerator(amplitude=0.3) )
        
            
    """
    def __init__(self, group=None, amplitude=0.2):
        super(TranslationGenerator, self).__init__(group=group)
        # set amplitude
        self.set_amplitude(amplitude)
        
    @property
    def amplitude(self):
        """The translation amplitude limits."""
        return self.__amplitude
        
    def set_amplitude(self, amplitude):
        """
        Sets maximum translation vector allowed amplitude.
        
        :Parameters:
            #. amplitude (number, tuple): The translation amplitude in Angstroms.
               If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
               If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
        """
        if isinstance(amplitude, (list, tuple, set)):
            assert len(amplitude) == 2, LOGGER.error("Translation amplitude tuple must have exactly two items")
            assert is_number(amplitude[0]), LOGGER.error("Translation amplitude first item must be a number")
            assert is_number(amplitude[1]), LOGGER.error("Translation amplitude second item must be a number")
            min = FLOAT_TYPE(amplitude[0])
            max = FLOAT_TYPE(amplitude[1])
            assert min>=0, LOGGER.error("Translation amplitude first item must be bigger than 0")
            assert max>min, LOGGER.error("Translation amplitude first item must be bigger than the second item")
            amplitude = (min,max)
        else:
            assert is_number(amplitude), LOGGER.error("Translation amplitude must be a number")
            amplitude = float(amplitude)
            assert amplitude>0, LOGGER.error("Translation amplitude must be bigger than 0")
            amplitude = (FLOAT_TYPE(0),FLOAT_TYPE(amplitude))
        self.__amplitude = amplitude
        
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
        # generate random vector and ensure it is not zero
        vector = np.array(1-2*np.random.random(3), dtype=FLOAT_TYPE)
        norm   = np.linalg.norm(vector) 
        if norm == 0:
            while norm == 0:
                vector = np.array(1-2*np.random.random(3), dtype=FLOAT_TYPE)
                norm   = np.linalg.norm(vector)  
        # normalize vector
        vector /= FLOAT_TYPE( norm )
        # compute baseVector
        baseVector = FLOAT_TYPE(vector*self.__amplitude[0])
        # amplify vector
        maxAmp  = FLOAT_TYPE(self.__amplitude[1]-self.__amplitude[0])
        vector *= FLOAT_TYPE(generate_random_float()*maxAmp)
        vector += baseVector
        # translate and return
        return coordinates+vector
        
        
class TranslationAlongAxisGenerator(TranslationGenerator):    
    """ 
    Generates random translation moves upon groups of atoms along a pre-defined axis.

    :Parameters:
        #. group (None, fullrmc.Engine): The constraint RMC engine.
        #. amplitude (number, tuple): The translation amplitude in Angstroms.
           If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
           If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
        #. axis (list,set,tuple,numpy.ndarray): The pre-defined translation axis vector. 
        #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
           If None generated axis can be in the same direction of axis or in the opposite.
           If True all generated vectors are in the same direction of axis.
           If False all generated vectors are in the opposite direction of axis.        
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationAlongAxisGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to translations along pre-defined axis (1,1,1).
        # Maximum translation amplitude is set to 0.3A to all defined groups
        for g in ENGINE.groups:
            g.set_move_generator( TranslationAlongAxisGenerator(amplitude=0.3, axis=(1,1,1)) )
    
    """
    def __init__(self, group=None, amplitude=0.2, axis=(1,0,0), direction=None):
        super(TranslationAlongAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set axis
        self.set_axis(axis)
        # set direction
        self.set_direction(direction)
    
    @property
    def axis(self):
        """ Get translation axis."""
        return self.__axis
    
    @property
    def direction(self):
        """ Get generated translation vectors direction."""
        return self.__direction
        
    def set_axis(self, axis):
        """
        Sets the axis along which the translation will be performed.
        
        :Parameters:
            #. axis (list,set,tuple,numpy.ndarray): The translation axis vector.
        """
        assert isinstance(axis, (list,set,tuple,np.ndarray)), LOGGER.error("axis must be a list")
        axis = list(axis)
        assert len(axis)==3, LOGGER.error("axis list must have 3 items")
        for pos in axis:
            assert is_number(pos), LOGGER.error( "axis items must be numbers")
        axis = [FLOAT_TYPE(pos) for pos in axis]
        axis =  np.array(axis, dtype=FLOAT_TYPE)
        self.__axis = axis/FLOAT_TYPE( np.linalg.norm(axis) )
    
    def set_direction(self, direction):
        """
        Sets the generated translation vectors direction.
        
        :Parameters:
            #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
               If None generated axis can be in the same direction of axis or in the opposite.
               If True all generated vectors are in the same direction of axis.
               If False all generated vectors are in the opposite direction of axis.
        """
        assert direction in (None, True, False), LOGGER.error("direction can only be None, True or False")
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
        # get translation amplitude
        maxAmp = self.amplitude[1]-self.amplitude[0]
        if self.__direction is None:
            amplitude = (1-2*generate_random_float())*maxAmp
        elif self.__direction:
            amplitude = generate_random_float()*maxAmp
        else:
            amplitude = -generate_random_float()*maxAmp
        # compute baseVector
        baseVector = FLOAT_TYPE( np.sign(amplitude)*self.__axis*self.amplitude[0] )
        # compute translation vector
        vector = baseVector + self.__axis*FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector


class TranslationTowardsAxisGenerator(TranslationAlongAxisGenerator):    
    """ 
    Generates random translation moves upon groups of atoms towards a pre-defined axis
    within a tolerance angle between translation vectors and the pre-defined axis.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint RMC engine.
        #. amplitude (number, tuple): The translation amplitude in Angstroms.
           If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
           If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
        #. axis (list,set,tuple,numpy.ndarray): The pre-defined translation axis vector.
        #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
           If None generated axis can be in the same direction of axis or in the opposite.
           If True all generated vectors are in the same direction of axis.
           If False all generated vectors are in the opposite direction of axis.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationTowardsAxisGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to translations towards a pre-defined axis (1,1,1) within 10 degrees.
        # Maximum translation amplitude is set to 0.3A to all defined groups 
        for g in ENGINE.groups:
            g.set_move_generator( TranslationTowardsAxisGenerator(amplitude=0.3, axis=(1,1,1), angle=10) )
           
    """
    def __init__(self, group=None, amplitude=0.2, axis=(1,0,0), angle=30, direction=True):
        super(TranslationTowardsAxisGenerator, self).__init__(group=group, amplitude=amplitude, axis=axis, direction=direction)
        # set angle
        self.set_angle(angle)
    
    @property
    def angle(self):
        """ Get tolerance maximum angle in rad."""
        return self.__angle
        
    def set_angle(self, angle):
        """
        Sets the tolerance maximum angle.
        
        :Parameters:
            #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        """
        assert is_number(angle), LOGGER.error("angle must be numbers")
        assert angle>0, LOGGER.error("angle must be positive")
        assert angle<=360, LOGGER.error("angle must be smaller than 360")
        self.__angle = FLOAT_TYPE(angle)*PI/FLOAT_TYPE(180.)

    def transform_coordinates(self, coordinates, argument=None):
        """
        translates coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # generate translation axis
        translationAxis = generate_vectors_in_solid_angle(direction=self.axis,
                                                          maxAngle=self.__angle,
                                                          numberOfVectors=1)[0]  
        # get translation amplitude
        maxAmp = self.amplitude[1]-self.amplitude[0]
        if self.direction is None:
            amplitude = (1-2*generate_random_float())*maxAmp
        elif self.direction:
            amplitude = generate_random_float()*maxAmp
        else:
            amplitude = -generate_random_float()*maxAmp
        # compute baseVector
        baseVector = FLOAT_TYPE(np.sign(amplitude)*translationAxis*self.amplitude[0])
        # compute translation vector
        vector = baseVector + translationAxis*FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector

        
class TranslationAlongSymmetryAxisGenerator(TranslationGenerator):    
    """ 
    Generates random translation moves upon groups of atoms along one of their symmetry axis.
    Only groups containing more than 1 atoms allow computing symmetry axis.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number, tuple): The translation amplitude in Angstroms.
           If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
           If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
        #. axis (integer): Must be 0,1 or 2 for respectively the mains, secondary or tertiary symmetry axis.
        #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
           If None generated axis can be in the same direction of axis or in the opposite.
           If True all generated vectors are in the same direction of axis.
           If False all generated vectors are in the opposite direction of axis.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationAlongSymmetryAxisGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to translations along the second symmetry axis of every group.
        # Maximum translation amplitude is set to 0.3A to all defined groups.
        for g in ENGINE.groups:
            if len(g)>1:
                g.set_move_generator( TranslationAlongSymmetryAxisGenerator(amplitude=0.3, axis=(1,1,1), axis=1) )
           
    """
    
    def __init__(self, group=None, amplitude=0.2, axis=0, direction=None):
        super(TranslationAlongSymmetryAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
        # set direction
        self.set_direction(direction)
    
    @property
    def axis(self):
        """ Get translation axis index."""
        return self.__axis
    
    @property
    def direction(self):
        """ Get generated translation vectors direction."""
        return self.__direction
        
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
        assert is_integer(axis), LOGGER.error("rotation symmetry axis must be an integer")
        axis = INT_TYPE(axis)
        assert axis>=0, LOGGER.error("rotation symmetry axis must be positive.")
        assert axis<=2, LOGGER.error("rotation symmetry axis must be smaller or equal to 2")
        # convert to radian and store amplitude
        self.__axis = axis
    
    def set_direction(self, direction):
        """
        Sets the generated translation vectors direction.
        
        :Parameters:
            #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
               If None generated axis can be in the same direction of axis or in the opposite.
               If True all generated vectors are in the same direction of axis.
               If False all generated vectors are in the opposite direction of axis.
        """
        assert direction in (None, True, False), LOGGER.error("direction can only be None, True or False")
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
        # get translation amplitude
        maxAmp = self.amplitude[1]-self.amplitude[0]
        if self.direction is None:
            amplitude = (1-2*generate_random_float())*maxAmp
        elif self.direction:
            amplitude = generate_random_float()*maxAmp
        else:
            amplitude = -generate_random_float()*maxAmp
        # get axis of translation
        _,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        translationAxis = [X,Y,Z][self.__axis]
        # compute baseVector
        baseVector = FLOAT_TYPE(np.sign(amplitude)*translationAxis*self.amplitude[0])
        # compute translation vector
        vector = baseVector + translationAxis*FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector
    

class TranslationTowardsSymmetryAxisGenerator(TranslationAlongSymmetryAxisGenerator):    
    """ 
    Generates random translation moves upon groups of atoms towards one of its symmetry
    axis within a tolerance angle between translation vectors and the axis.
    Only groups of more than 1 atom are accepted.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint RMC engine.
        #. amplitude (number, tuple): The translation amplitude in Angstroms.
           If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
           If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
        #. axis (integer): Must be 0,1 or 2 for respectively the mains, secondary or tertiary symmetry axis.
        #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        #. direction (None, True, False): Whether to generate translation vector in the same direction of axis or not.
           If None generated axis can be in the same direction of axis or in the opposite.
           If True all generated vectors are in the same direction of axis.
           If False all generated vectors are in the opposite direction of axis.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationTowardsSymmetryAxisGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to translations towards the first symmetry axis of every group within 15 degrees.
        # Maximum translation amplitude is set to 0.3A to all defined groups.
        for g in ENGINE.groups:
            if len(g)>1:
                g.set_move_generator( TranslationTowardsSymmetryAxisGenerator(amplitude=0.3, axis=0, angle=15) )
               
    """
    
    def __init__(self, group=None, amplitude=0.2, axis=0, angle=30, direction=True):
        super(TranslationTowardsSymmetryAxisGenerator, self).__init__(group=group, amplitude=amplitude, axis=axis, direction=direction)
        # set angle
        self.set_angle(angle)
        
    @property
    def angle(self):
        """ Get tolerance maximum angle in rad."""
        return self.__angle
 
    def set_angle(self, angle):
        """
        Sets the tolerance maximum angle.
        
        :Parameters:
            #. angle (number): The maximum tolerance angle in degrees between a generated translation vector and the pre-defined axis.        
        """
        assert is_number(angle), LOGGER.error("angle must be numbers")
        assert angle>0, LOGGER.error("angle must be positive")
        assert angle<=360, LOGGER.error("angle must be smaller than 360")
        self.__angle = FLOAT_TYPE(angle)*PI/FLOAT_TYPE(180.)

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
        # generate translation axis
        translationAxis = generate_vectors_in_solid_angle(direction=axis,
                                                          maxAngle=self.__angle,
                                                          numberOfVectors=1)[0]  
        # get translation amplitude
        maxAmp = self.amplitude[1]-self.amplitude[0]
        if self.direction is None:
            amplitude = (1-2*generate_random_float())*maxAmp
        elif self.direction:
            amplitude = generate_random_float()*maxAmp
        else:
            amplitude = -generate_random_float()*maxAmp
        # compute baseVector
        baseVector = FLOAT_TYPE(np.sign(amplitude)*translationAxis*self.amplitude[0])
        # compute translation vector
        vector = baseVector + translationAxis*FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector
        
        
class TranslationAlongSymmetryAxisPath(PathGenerator):    
    """ 
    Generates translation moves upon groups of atoms along one of their symmetry axis.
    Only groups of more than 1 atom are accepted.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis 
        #. path (List): list of distances.
        #. randomize (boolean): Whether to pull moves randomly from path or pull moves in order at every step.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationAlongSymmetryAxisPath
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to translations of predefined amplitudes along the first symmetry axis of every group.
        amps = [-0.1, 0.075, -0.05, -0.25, 0.01, 0.02, 0.03, 0.1, 0.3]
        for g in ENGINE.groups:
            if len(g)>1:
                g.set_move_generator( TranslationAlongSymmetryAxisPath(axis=0, path=amps) )
               
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
        assert is_integer(axis), LOGGER.error("rotation symmetry axis must be an integer.")
        axis = INT_TYPE(axis)
        assert axis>=0, LOGGER.error("rotation symmetry axis must be positive.")
        assert axis<=2, LOGGER.error("rotation symmetry axis must be smaller or equal to 2.")
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
        #. amplitude (number, tuple): The translation amplitude in Angstroms.
           If number is given, it is the maximum translation amplitude in Angstroms and must be bigger than 0.
           If tuple is given, it is the limits of translation boundaries as [min,max] where min>=0 and max>min.
        #. angle (None, number): The maximum tolerance angle in degrees between a generated translation vector and the computed direction. 
           If None is given, all generated translation vectors will be along the direction to center.        
        #. direction (None, True, False): Whether to generate translation vectors pointing towards the center or not.
           If None generated axis can be randomly generated towards the center or away from the center.
           If True all generated vectors point towards the center.
           If False all generated vectors point away from the center.        
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Translations import TranslationTowardsCenterGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to translations towards the origin defined as (0,0,0) within 20 degrees.
        # Maximum translation amplitude is set to 0.2A to all defined groups.
        for g in ENGINE.groups:
            g.set_move_generator( TranslationTowardsCenterGenerator(amplitude=0.2, center={"fixed":(0,0,0)}, axis=0, angle=25) )
    
    """
    def __init__(self, group=None, center={"fixed":(0,0,0)}, amplitude=0.1, angle=30, direction=True):
        # initialize TranslationGenerator
        TranslationGenerator.__init__(self, group=group, amplitude=amplitude)
        # set direction
        self.set_direction(direction)
        # set center
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
        assert direction in (None, True, False), LOGGER.error("direction can only be None, True or False")
        self.__direction = direction
        
    def set_angle(self, angle):
        """
        Sets the tolerance maximum angle.
        
        :Parameters:
            #. angle (None, number): The maximum tolerance angle in degrees between a generated translation vector and the computed direction. 
               If None is given, all generated translation vectors will be along the direction to center.        
        """
        if angle is not None:
            assert is_number(angle), LOGGER.error("angle must be numbers")
            assert angle>=0, LOGGER.error("angle must be positive")
            assert angle<=360, LOGGER.error("angle must be smaller than 360")
            if FLOAT_TYPE(angle) == FLOAT_TYPE(0.0):
                angle = None
            else:
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
        assert isinstance(center, dict), LOGGER.error("center must be a dictionary")
        assert len(center) == 1, LOGGER.error("center must have a single key")       
        key = center.keys()[0]
        val = center[key]
        assert isinstance(val, (list,set,tuple,np.ndarray)), LOGGER.error("center value must be a list")
        if isinstance(val, np.ndarray):
            assert len(val.shape) == 1, LOGGER.error("center value must have a single dimension")
        assert len(val)>0, LOGGER.error("center value must be a non-zero list.")
        for v in val:
            assert is_number(v), LOGGER.error("center value item must be numbers") 
        if key == "fixed":
            self.__mustCompute = False
            assert len(val) == 3, LOGGER.error("fixed center must have exactly 3 elements corresponding to X,Y and Z coordinates of the center point.")
            val = np.array([FLOAT_TYPE(v) for v in val], dtype=FLOAT_TYPE)
        elif key == "indexes":
            self.__mustCompute = True
            for v in val:
                assert is_integer(v), LOGGER.error("indexes center items be integers")
            val =  np.array([INT_TYPE(v) for v in val], dtype=INT_TYPE)
            for v in val:
                assert v>=0, LOGGER.error("indexes center items be positive integers")            
        else:
            self.__mustCompute = None
            raise Exception(LOGGER.error("center key must be either 'fixed' or 'indexes'"))        
        # set center
        self.__center = {key:val}
        
    def __get_amplitude(self):
        # get translation amplitude
        maxAmp = self.amplitude[1]-self.amplitude[0]
        if self.__direction is None:
            amplitude = (1-2*generate_random_float())*maxAmp
        elif self.__direction:
            amplitude = generate_random_float()*maxAmp
        else:
            amplitude = -generate_random_float()*maxAmp
        return amplitude
    
    def __get_translation_axis(self, direction):
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
        
    def __get_center(self):
        if self.__mustCompute:
            center = self.group._get_engine().realCoordinates[self.__center["indexes"]]
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
        center = self.__get_center()
        # compute coordinates center
        coordsCenter = np.mean(coordinates, axis=0)
        direction    = center-coordsCenter
        # translation vector
        translationAxis = self.__get_translation_axis(direction)
        # get amplitude
        amplitude = self.__get_amplitude()
        # compute baseVector
        baseVector = FLOAT_TYPE(np.sign(amplitude)*translationAxis*self.amplitude[0])
        # compute translation vector
        vector = baseVector + translationAxis*FLOAT_TYPE(amplitude)
        # translate and return
        return coordinates+vector
        
        
 


       