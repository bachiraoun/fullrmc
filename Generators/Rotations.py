"""
Rotations contains all rotation like MoveGenerator classes.

.. inheritance-diagram:: fullrmc.Generators.Rotations
    :parts: 1 
    
    
+-----------------------------------------------------+-----------------------------------------------------+
|.. figure:: randomRotation.png                       |.. figure:: randomRotationAboutAxis.png              |
|   :width: 375px                                     |   :width: 375px                                     |
|   :height: 300px                                    |   :height: 300px                                    |
|   :align: left                                      |   :align: left                                      |
|                                                     |                                                     |
|   Random rotation axis and angle generated and      |   Random rotation generated about a pre-defined axis|
|   applied on a Tetrahydrofuran molecule. Solid      |   or one of the symmetry axes of the Tetrahydrofuran|
|   colours are of the origin molecule position while |   molecule. Solid colours are of the origin molecule|
|   fading ones are of the rotated molecule.          |   position while fading ones are of the rotated     |
|   (:class:`RotationGenerator`)                      |   molecule.                                         |
|                                                     |   (:class:`RotationAboutAxisGenerator`              |
|                                                     |   :class:`RotationAboutSymmetryAxisGenerator`)      |
+-----------------------------------------------------+-----------------------------------------------------+
|.. figure:: orientationGenerator.png                 |                                                     |
|   :width: 375px                                     |                                                     |
|   :height: 300px                                    |                                                     |
|   :align: left                                      |                                                     |
|                                                     |                                                     |
|   Random orientation of hexane molecule to [1,1,1]  |                                                     |
|   axis with maximumOffsetAngle of 10 degrees is     |                                                     |
|   generated. First principal axis of hexane molecule|                                                     |
|   is considered as groupAxis. Solid colors are of   |                                                     |
|   original molecule while fading ones are of the    |                                                     |
|   oriented one. (:class:`OrientationGenerator`)     |                                                     |
|                                                     |                                                     |
+-----------------------------------------------------+-----------------------------------------------------+

.. raw:: html

        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/-clLvYiaC8w?rel=0" 
        frameborder="0" allowfullscreen>
        </iframe>
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path, generate_random_float, get_principal_axis, get_rotation_matrix, orient, generate_vectors_in_solid_angle
from fullrmc.Core.MoveGenerator import MoveGenerator, PathGenerator


class RotationGenerator(MoveGenerator):
    """ 
    Generates random rotational moves upon groups of atoms.
    Only groups of more than 1 atom are accepted.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number): the maximum rotation angle allowed in degrees.
           It must be strictly bigger than 0 and strictly smaller than 360.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Rotations import RotationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to random rotations.
        # Maximum rotation amplitude is set to 5 degrees to all defined groups
        for g in ENGINE.groups:
            if len(g) >1:
                g.set_move_generator( RotationGenerator(amplitude=5) )
           
    """
    def __init__(self, group=None, amplitude=2):
        super(RotationGenerator, self).__init__(group=group)
        # set amplitude
        self.set_amplitude(amplitude)
        
    @property
    def amplitude(self):
        """ Get the maximum allowed angle of rotation in rad."""
        return self.__amplitude
        
    def set_amplitude(self, amplitude):
        """
        Sets maximum rotation angle in degrees and transforms it to rad.
        
        :Parameters:
            #. amplitude (number): the maximum allowed rotation angle in degrees.
               It must be strictly bigger than 0 and strictly smaller than 360.
        """
        assert is_number(amplitude), LOGGER.error("rotation amplitude must be a number")
        amplitude = float(amplitude)
        assert amplitude>0, LOGGER.error("rotation amplitude must be bigger than 0 deg.")
        assert amplitude<360, LOGGER.error("rotation amplitude must be smaller than 360 deg.")
        # convert to radian and store amplitude
        self.__amplitude = FLOAT_TYPE(PI*amplitude/180.)
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)<=1:
            return False, "At least two atoms needed in a group to perform rotation."
        else:
            return True, "" 
        
    def transform_coordinates(self, coordinates, argument=None):
        """
        Rotate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the rotation
            #. argument (object): Any python object. Not used in this generator.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the rotation.
        """
        # get rotation axis
        n = 0
        while n<PRECISION:
            rotationAxis = 1-2*np.random.random(3)
            n = np.linalg.norm(rotationAxis)
        rotationAxis /= n
        # get rotation angle
        rotationAngle = (1-2*generate_random_float())*self.amplitude
        # get rotation matrix
        rotationMatrix = get_rotation_matrix(rotationAxis, rotationAngle)
        # get atoms group center
        center = np.sum(coordinates, 0)/coordinates.shape[0]
        # translate to origin
        rotatedCoordinates = coordinates-center
        # rotate
        for idx in range(rotatedCoordinates.shape[0]):
            rotatedCoordinates[idx,:] = np.dot( rotationMatrix, rotatedCoordinates[idx,:])
        # translate back to center and return rotated coordinates
        return np.array(rotatedCoordinates+center, dtype=FLOAT_TYPE)
        
  
class RotationAboutAxisGenerator(RotationGenerator):    
    """ 
    Generates random rotational moves upon groups of atoms about a pre-defined axis.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number): The maximum allowed rotation angle in degrees.
           It must be strictly bigger than 0 and strictly smaller than 360.
        #. axis (list,set,tuple,numpy.ndarray): The rotational axis vector.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Rotations import RotationAboutAxisGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to random rotations about (1,1,1) a pre-defined axis.
        # Maximum rotation amplitude is set to 5 degrees to all defined groups
        for g in ENGINE.groups:
            if len(g) >1:
                g.set_move_generator( RotationAboutAxisGenerator(amplitude=5, axis=(1,1,1)) )
                
    """
    
    def __init__(self, group=None, amplitude=2, axis=(1,0,0)):
        super(RotationAboutAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get rotation axis."""
        return self.__axis
    
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        return True, "" 
            
    def set_axis(self, axis):
        """
        Sets the axis along which the rotation will be performed.
        
        :Parameters:
            #. axis (list,set,tuple,numpy.ndarray): The rotation axis vector.
        """
        assert isinstance(axis, (list,set,tuple,np.ndarray)), LOGGER.error("axis must be a list")
        axis = list(axis)
        assert len(axis)==3, LOGGER.error("axis list must have 3 items")
        for pos in axis:
            assert is_number(pos), LOGGER.error("axis items must be numbers")
        axis = [FLOAT_TYPE(pos) for pos in axis]
        axis =  np.array(axis, dtype=FLOAT_TYPE)
        self.__axis = axis/FLOAT_TYPE( np.linalg.norm(axis) )
    
    def transform_coordinates(self, coordinates, argument=None):
        """
        rotate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the rotation.
            #. argument (object): Any python object. Not used in this generator.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the rotation.
        """
        # get rotation angle
        rotationAngle  = (1-2*generate_random_float())*self.amplitude
        # get rotation matrix
        rotationMatrix = get_rotation_matrix(self.__axis, rotationAngle)
        # get atoms group center and rotation axis
        center,_,_,_,_,_,_ = get_principal_axis(coordinates)        
        # translate to origin
        rotatedCoordinates = coordinates-center
        # rotate
        for idx in range(rotatedCoordinates.shape[0]):
            rotatedCoordinates[idx,:] = np.dot( rotationMatrix, rotatedCoordinates[idx,:])
        # translate back to center and return rotated coordinates
        return np.array(rotatedCoordinates+center, dtype=FLOAT_TYPE) 

        
class RotationAboutSymmetryAxisGenerator(RotationGenerator):    
    """ 
    Generates random rotational moves upon groups of atoms about one of their symmetry axis.
    Only groups of more than 1 atom are accepted.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint fullrmc engine.
        #. amplitude (number): The maximum rotation angle in degrees.
           It must be strictly bigger than 0 and strictly smaller than 360.
        #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis 
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Rotations import RotationAboutSymmetryAxisGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to random rotations about the second symmetry axis of each group.
        # Maximum rotation amplitude is set to 5 degrees to all defined groups
        for g in ENGINE.groups:
            if len(g) >1:
                g.set_move_generator( RotationAboutSymmetryAxisGenerator(amplitude=5, axis=1) )
                
    """
    def __init__(self, group=None, amplitude=2, axis=0):
        super(RotationAboutSymmetryAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get rotation axis index."""
        return self.__axis
        
    def set_axis(self, axis):
        """
        Sets the symmetry axis index to rotate about.
        
        :Parameters:
            #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis
        """
        assert is_integer(axis), LOGGER.error("rotation symmetry axis must be an integer")
        axis = INT_TYPE(axis)
        assert axis>=0, LOGGER.error("rotation symmetry axis must be positive.")
        assert axis<=2, LOGGER.error("rotation symmetry axis must be smaller or equal to 2")
        # convert to radian and store amplitude
        self.__axis = axis
    
    def transform_coordinates(self, coordinates, argument=None):
        """
        Rotate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the rotation.
            #. argument (object): Any python object. Not used in this generator.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the rotation.
        """
        # get rotation angle
        rotationAngle = (1-2*generate_random_float())*self.amplitude
        # get atoms group center and rotation axis
        center,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        rotationAxis = [X,Y,Z][self.__axis]
        # get rotation matrix
        rotationMatrix = get_rotation_matrix(rotationAxis, rotationAngle)
        # translate to origin
        rotatedCoordinates = coordinates-center
        # rotate
        for idx in range(rotatedCoordinates.shape[0]):
            rotatedCoordinates[idx,:] = np.dot( rotationMatrix, rotatedCoordinates[idx,:])
        # translate back to center and return rotated coordinates
        return np.array(rotatedCoordinates+center, dtype=FLOAT_TYPE) 
      
        
class RotationAboutSymmetryAxisPath(PathGenerator):    
    """ 
    Generates rotational moves upon groups of atoms about one of their symmetry axis.
    Only groups of more than 1 atom are accepted.
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint fullrmc engine.
        #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis.
        #. path (List): list of angles.
        #. randomize (boolean): Whether to pull moves randomly from path or pull moves in order at every step.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Rotations import RotationAboutSymmetryAxisPath
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to pre-defined rotations about the second symmetry axis of each group.
        angles = [-0.1, -0.5, -0.05, 0.5, 0.01, 2, 3, 1, -3]
        for g in ENGINE.groups:
            if len(g) >1:
                g.set_move_generator( RotationAboutSymmetryAxisPath(axis=1, path=angles) )
                
    """
    def __init__(self, group=None,  axis=0, path=None, randomize=False):
        # initialize PathGenerator
        PathGenerator.__init__(self, group=group, path=path, randomize=randomize)
        # set axis
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get rotation axis index."""
        return self.__axis
        
    def set_axis(self, axis):
        """
        Sets the symmetry axis index to rotate about.
        
        :Parameters:
            #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis
        """
        assert is_integer(axis), LOGGER.error("rotation symmetry axis must be an integer")
        axis = INT_TYPE(axis)
        assert axis>=0, LOGGER.error("rotation symmetry axis must be positive.")
        assert axis<=2,LOGGER.error("rotation symmetry axis must be smaller or equal to 2")
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
        for angle in path:
            if not is_number(angle):
                return False, "path items must be numbers"
        return True, ""
        
    def normalize_path(self, path):
        """
        Transforms all path angles to radian.
        
        :Parameters:
            #. path (list): The list of moves.
        
        :Returns:
            #. path (list): The list of moves.
        """
        path = [FLOAT_TYPE(angle)*PI/FLOAT_TYPE(180.) for angle in path]
        return list(path)
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)<=1:
            return False, "At least two atoms needed in a group to perform rotation."
        else:
            return True, "" 
            
    def transform_coordinates(self, coordinates, argument):
        """
        Rotate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the rotation.
            #. argument (object): The rotation angle.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the rotation.
        """
        # get atoms group center and rotation axis
        center,_,_,_,X,Y,Z =get_principal_axis(coordinates)
        rotationAxis = [X,Y,Z][self.__axis]
        # get rotation matrix
        rotationMatrix = get_rotation_matrix(rotationAxis, argument)
        # translate to origin
        rotatedCoordinates = coordinates-center
        # rotate
        for idx in range(rotatedCoordinates.shape[0]):
            rotatedCoordinates[idx,:] = np.dot( rotationMatrix, rotatedCoordinates[idx,:])
        # translate back to center and return rotated coordinates
        return np.array(rotatedCoordinates+center, dtype=FLOAT_TYPE) 



class OrientationGenerator(MoveGenerator):
    """ 
    Generates rotational moves upon groups of atoms to align and orient along an axis.
    Orientation rotations are computed randomly allowing offset angle between grouAxis and orientationAxis 
    Only groups of more than 1 atom are accepted.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. maximumOffsetAngle (number): The maximum offset angle in degrees between groupAxis and orientationAxis.
        #. groupAxis (dict): The group axis. Only one key is allowed.
           If key is 'fixed', value must be a list, tuple or a numpy.array of a vector such as [X,Y,Z]
           If key is 'symmetry', in this case the group axis is computed as one of the three 
           symmetry axis of the group atoms. the value must be even 0, 1 or 2 for respectively 
           the first, second and tertiary symmetry axis.
        #. orientationAxis (dict): The axis to align the group with.
           If key is 'fixed', value must be a list, tuple or a numpy.array of a vector such as [X,Y,Z]
           if Key is 'symmetry', in this case the the value must be a list of two items, the first one is a list
           of atoms indexes to compute symmetry axis and the second item must be even 0, 1 or 2 for respectively 
           the first, second and tertiary symmetry axis. 
        #. flip (None, bool): Whether to allow flipping axis orientation or not.
           If True, orientationAxis will be flipped forcing anti-parallel orientation.
           If False, orientationAxis will not be flipped forcing parallel orientation.
           If None, no flipping is forced, flipping can be set randomly to True or False during run time execution.     
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Rotations import OrientationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to orientations of each group third symmetry axis 
        # towards the (-1,0,2) the predefined axis within maximum 5 degrees.
        for g in ENGINE.groups:
            if len(g) >1:
                g.set_move_generator( OrientationGenerator(maximumOffsetAngle=5, 
                                                           groupAxis={"symmetry":2},
                                                           orientationAxis={"fixed":(-1,0,2)}) )
    """
    def __init__(self, group=None, maximumOffsetAngle=10, groupAxis={"symmetry":0}, orientationAxis={"fixed":(1,0,0)}, flip=None):
        super(OrientationGenerator, self).__init__(group=group)
        # set maximumOffsetAngle
        self.set_maximum_offset_angle(maximumOffsetAngle)
        # set group axis
        self.set_group_axis(groupAxis)
        # set orientation axis
        self.set_orientation_axis(orientationAxis)
        # set flip
        self.set_flip(flip)
        
    @property
    def maximumOffsetAngle(self):
        """ The maximum offset angle in degrees between groupAxis and orientationAxis in rad."""
        return self.__maximumOffsetAngle
      
    @property
    def orientationAxis(self):
        """ The orientation axis value or definition."""
        return self.__orientationAxis 
    
    @property
    def groupAxis(self):
        """ The group axis value or definition."""
        return self.__groupAxis 
    
    @property
    def flip(self):
        """ The flip value."""
        return self.__flip
        
    def set_maximum_offset_angle(self, maximumOffsetAngle):
        """
        Sets the maximum offset angle allowed.
        
        :Parameters:
            #. maximumOffsetAngle (number): The maximum offset angle in degrees between groupAxis and orientationAxis in degrees.
        """
        assert is_number(maximumOffsetAngle), LOGGER.error("maximumOffsetAngle must be a number")
        maximumOffsetAngle = float(maximumOffsetAngle)
        assert maximumOffsetAngle>0, LOGGER.error("maximumOffsetAngle must be bigger than 0 deg.")
        assert maximumOffsetAngle<180, LOGGER.error("maximumOffsetAngle must be smaller than 180 deg.")
        # convert to radian and store amplitude
        self.__maximumOffsetAngle = FLOAT_TYPE(PI*maximumOffsetAngle/180.)
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)<=1:
            return False, "At least two atoms needed in a group to perform rotation."
        else:
            return True, "" 
    
    def set_flip(self, flip):
        """
        Sets flip flag value.
        
        :Parameters:
            #. flip (None, bool): Whether to allow flipping axis orientation or not.
               If True, orientationAxis will be flipped forcing anti-parallel orientation.
               If False, orientationAxis will not be flipped forcing parallel orientation.
               If None, no flipping is forced, flipping can be set randomly to True or False during run time execution. 
        """
        assert flip in (None, True, False), LOGGER.error("flip can only be None, True or False")
        self.__flip = flip
        
    def set_group_axis(self, groupAxis):
        """
        Sets group axis value.
        
        :Parameters:
           #. groupAxis (dict): The group axis. Only one key is allowed.
              If key is fixed, value must be a list, tuple or a numpy.array of a vector such as [X,Y,Z]
              If key is symmetry, in this case the group axis is computed as one of the three 
              symmetry axis of the group atoms. the value must be even 0, 1 or 2 for respectively 
              the first, second and tertiary symmetry axis.
        """
        assert isinstance(groupAxis, dict), LOGGER.error("groupAxis must be a dictionary")
        assert len(groupAxis) == 1, LOGGER.error("groupAxis must have a single key")       
        key = groupAxis.keys()[0]
        val = groupAxis[key]
        if key == "fixed":
            self.__mustComputeGroupAxis = False
            assert isinstance(val, (list,set,tuple,np.ndarray)), LOGGER.error("groupAxis value must be a list")
            if isinstance(val, np.ndarray):
                assert len(val.shape) == 1, LOGGER.error("groupAxis value must have a single dimension")
            val = list(val)
            assert len(val)==3, LOGGER.error("groupAxis fixed value must be a vector")
            for v in val:
                assert is_number(v), LOGGER.error("groupAxis value item must be numbers") 
            val  = np.array([FLOAT_TYPE(v) for v in val], dtype=FLOAT_TYPE)  
            norm = FLOAT_TYPE(np.sqrt(np.sum(val**2)))    
            val /= norm              
        elif key == "symmetry":
            self.__mustComputeGroupAxis = True
            assert is_integer(val), LOGGER.error("groupAxis symmetry value must be an integer") 
            val = INT_TYPE(val)
            assert val>=0 and val<3, LOGGER.error("groupAxis symmetry value must be positive smaller than 3") 
        else:
            self.__mustComputeGroupAxis = None
            raise Exception(LOGGER.error("groupAxis key must be either 'fixed' or 'symmetry'"))        
        # set groupAxis
        self.__groupAxis = {key:val}

    def set_orientation_axis(self, orientationAxis):
        """
        Sets orientation axis value.
        
        :Parameters:
           #. orientationAxis (dict): The axis to align the group axis with.
              If key is fixed, value must be a list, tuple or a numpy.array of a vector such as [X,Y,Z]
              if Key is symmetry, in this case the the value must be a list of two items, the first one is a list
              of atoms indexes to compute symmetry axis and the second item must be even 0, 1 or 2 for respectively 
              the first, second and tertiary symmetry axis. 
        """
        assert isinstance(orientationAxis, dict), LOGGER.error("orientationAxis must be a dictionary")
        assert len(orientationAxis) == 1, LOGGER.error("orientationAxis must have a single key")       
        key = orientationAxis.keys()[0]
        val = orientationAxis[key]
        if key == "fixed":
            self.__mustComputeOrientationAxis = False
            assert isinstance(val, (list,set,tuple,np.ndarray)), LOGGER.error("orientationAxis value must be a list")
            if isinstance(val, np.ndarray):
                assert len(val.shape) == 1, LOGGER.error("orientationAxis value must have a single dimension")
            val = list(val)
            assert len(val)==3, LOGGER.error("orientationAxis fixed value must be a vector")
            for v in val:
                assert is_number(v), LOGGER.error("orientationAxis value item must be numbers") 
            val  = np.array([FLOAT_TYPE(v) for v in val], dtype=FLOAT_TYPE) 
            norm = FLOAT_TYPE(np.sqrt(np.sum(val**2)))   
            val /= norm            
        elif key == "symmetry":
            self.__mustComputeOrientationAxis = True
            assert isintance(val, (list, tuple)), LOGGER.error("orientationAxis symmetry value must be a list") 
            assert len(val) == 2, LOGGER.error("orientationAxis symmetry value must be a list of two items")
            val0 = []
            for v in val[0]:
                assert is_integer(v), LOGGER.error("orientationAxis symmetry value list items must be integers") 
                v0 = INT_TYPE(v)
                assert v0>=0, LOGGER.error("orientationAxis symmetry value list items must be positive") 
                val0.append(v0)
            assert len(set(val0))==len(val[0]), LOGGER.error("orientationAxis symmetry value list redundant items indexes found") 
            val0 = np.array(val0, dtype=INT_TYPE)
            val1 = val[1]
            assert is_integer(val1), LOGGER.error("orientationAxis symmetry value second item must be an integer") 
            val1 = INT_TYPE(val1)
            assert val1>=0 and val1<3, LOGGER.error("orientationAxis symmetry value second item must be positive smaller than 3") 
            val = (val0,val1)
        else:
            self.__mustComputeOrientationAxis = None
            raise Exception(LOGGER.error("orientationAxis key must be either 'fixed' or 'symmetry'"))        
        # set orientationAxis
        self.__orientationAxis = {key:val}

    def __get_orientation_axis__(self):
        if self.__mustComputeOrientationAxis:
            coordinates   = self.group._get_engine().realCoordinates[self.__orientationAxis["symmetry"][0]]
            _,_,_,_,X,Y,Z = get_principal_axis(coordinates)
            axis = [X,Y,Z][self.__orientationAxis["symmetry"][1]]
        else:
            axis  = self.__orientationAxis["fixed"]
        return axis
        
    def __get_group_axis__(self, coordinates):
        if self.__mustComputeGroupAxis:
            _,_,_,_,X,Y,Z = get_principal_axis(coordinates)
            axis = [X,Y,Z][self.__groupAxis["symmetry"]]
        else:
            axis = self.__groupAxis["fixed"]
        return axis

    def transform_coordinates(self, coordinates, argument=None):
        """
        Rotate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the rotation
            #. argument (object): Any python object. Not used in this generator.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the rotation.
        """
         # create flip flag
        if self.__flip is None:
            flip = FLOAT_TYPE( np.sign(1-2*generate_random_float()) )
        elif self.__flip:
            flip = FLOAT_TYPE(-1)
        else:
            flip = FLOAT_TYPE(1)
        # get group axis
        groupAxis = self.__get_group_axis__(coordinates)
        # get align axis within offset angle
        orientationAxis = flip*self.__get_orientation_axis__()
        orientationAxis = generate_vectors_in_solid_angle(direction=orientationAxis,
                                                          maxAngle=self.__maximumOffsetAngle,
                                                          numberOfVectors=1)[0]  
        # get coordinates center
        center = np.array(np.sum(coordinates, 0)/coordinates.shape[0] , dtype=FLOAT_TYPE)
        # translate to origin
        rotatedCoordinates = coordinates-center
        # align coordinates
        rotatedCoordinates = orient(rotatedCoordinates, groupAxis, orientationAxis)
        # translate back to center and return rotated coordinates
        return np.array(rotatedCoordinates+center, dtype=FLOAT_TYPE)
        
        
        