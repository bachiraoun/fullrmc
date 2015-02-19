"""
Rotations contains all rotation like MoveGenerator classes.

.. inheritance-diagram:: fullrmc.Generators.Rotations
    :parts: 2 
"""

# standard libraries imports

# external libraries imports
import numpy as np
from pdbParser.Utilities.Geometry import get_rotation_matrix

# fullrmc imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, generate_random_float
from fullrmc.Core.Collection import is_number, is_integer, get_path, get_principal_axis
from fullrmc.Core.MoveGenerator import MoveGenerator, PathGenerator


class RotationGenerator(MoveGenerator):
    """ 
    Generates random rotational moves upon groups of atoms.
    
    :Parameters:
        #. group (None, Group): The group instance.
        #. amplitude (number): the maximum rotation angle allowed in degrees.
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
        """
        assert is_number(amplitude), log.LocalLogger("fullrmc").logger.error("rotation amplitude must be a number")
        amplitude = float(amplitude)
        assert amplitude>0, log.LocalLogger("fullrmc").logger.error("rotation amplitude must be bigger than 0 deg.")
        assert amplitude<360, log.LocalLogger("fullrmc").logger.error("rotation amplitude must be smaller than 360 deg.")
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
            #rotationAxis = np.random.random(3)-np.random.random(3)
            rotationAxis = 1-2*np.random.random(3)
            n = np.linalg.norm(rotationAxis)
        rotationAxis /= n
        # get rotation angle
        rotationAngle  = generate_random_float()*self.__amplitude
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
        #. axis (list,set,tuple,numpy.ndarray): The rotational axis vector.
    """
    
    def __init__(self, group=None, amplitude=0.5, axis=0):
        super(RotationAboutAxisGenerator, self).__init__(group=group, amplitude=amplitude)
        # set amplitude
        self.set_axis(axis)
    
    @property
    def axis(self):
        """ Get rotation axis."""
        return self.__axis
        
    def set_axis(self, axis):
        """
        Sets the axis along which the rotation will be performed.
        
        :Parameters:
            #. axis (list,set,tuple,numpy.ndarray): The rotation axis vector.
        """
        assert isinstance(axis, list,set,tuple,np.ndarray), log.LocalLogger("fullrmc").logger.error("axis must be a list")
        axis = list(axis)
        assert len(axis)==3, log.LocalLogger("fullrmc").logger.error("axis list must have 3 items")
        for pos in axis:
            assert is_number(pos), log.LocalLogger("fullrmc").logger.error("axis items must be numbers")
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
        rotationAngle  = generate_random_float()*self.amplitude
        # get rotation matrix
        rotationMatrix = get_rotation_matrix(self.__axis, rotationAngle)
        # get atoms group center and rotation axis
        center,_,_,_,_,_,_ =get_principal_axis(coordinates)        
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
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint fullrmc engine.
        #. amplitude (number): The maximum rotation angle in degrees.
        #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis 
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
        assert is_integer(axis), log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be an integer")
        axis = INT_TYPE(axis)
        assert axis>=0, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be positive.")
        assert axis<=2, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be smaller or equal to 2")
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
        rotationAngle  = generate_random_float()*self.amplitude
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
    
    :Parameters:
        #. group (None, fullrmc.Engine): The constraint fullrmc engine.
        #. axis (integer): Must be 0,1 or 2 for respectively the main, secondary or tertiary symmetry axis.
        #. path (List): list of angles.
        #. randomize (boolean): Whether to pull moves randomly from path or pull moves in order at every step.
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
        assert is_integer(axis), log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be an integer")
        axis = INT_TYPE(axis)
        assert axis>=0, log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be positive.")
        assert axis<=2,log.LocalLogger("fullrmc").logger.error("rotation symmetry axis must be smaller or equal to 2")
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




        