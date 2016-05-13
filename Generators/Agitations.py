"""
Agitations contains all MoveGenerator classes that agitate and shake structures such as distances, angles, etc.

.. inheritance-diagram:: fullrmc.Generators.Agitations
    :parts: 1 

+------------------------------------------------------+------------------------------------------------------+
|.. figure:: distanceAgitation.png                     |.. figure:: angleAgitation.png                        |
|   :width: 375px                                      |   :width: 375px                                      |
|   :height: 300px                                     |   :height: 300px                                     |
|   :align: left                                       |   :align: left                                       |
|                                                      |                                                      |
|   Random H-H bond length agitations generated on     |   Random H-O-H angle agitation generated on water    |
|   dihydrogen molecules. At room temperature, H2      |   molecules. At room temperature, water molecule     |
|   molecule bond length fluctuates around 0.74        |   angle formed between the two vectors formed between|
|   Angstroms. Red hydrogen atoms represent the shrank |   consecutively the Oxygen atom and the two hydrogen |
|   H-H bond length molecule while blue hydrogen atoms |   atoms is about 105 deg. Shrank H-O-H angles are    |
|   represent the expanded H-H bond length molecules.  |   represented by the red hydrogen while expanded     |
|                                                      |   angles are represented in blue.                    |
|   (:class:`DistanceAgitationGenerator`)              |   (:class:`AngleAgitationGenerator`)                 |
+------------------------------------------------------+------------------------------------------------------+
 
 .. raw:: html

        <iframe width="560" height="315" 
        src="https://www.youtube.com/embed/qTJux9kZCOo?rel=0" 
        frameborder="0" allowfullscreen>
        </iframe>
        
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_rotation_matrix, generate_random_float
from fullrmc.Core.MoveGenerator import MoveGenerator, PathGenerator


class DistanceAgitationGenerator(MoveGenerator):
    """
    Generates random agitation moves upon a distance separating two atoms by translating
    both atoms away from each other or closer to each other along the direction line between them.
    This is mainly used to shake two atoms bond distance by increasing and decreasing 
    the bond length.
    Only groups of length 2 are accepted.
     
    :Parameters:
        #. group (None, Group): The group instance. It must contain exactly 2 indexes.
        #. amplitude (number):  The maximum translation amplitude in Angstroms applied on every atom.
        #. symmetric (bool): Whether to apply the same amplitude of translation on both atoms or not.
        #. shrink (None, bool): Whether to always shrink the distance or expand it.
           If True, moves will always bring atoms closer to each other.
           If False, moves will always bring atoms away from each other.
           If None, no orientation is forced, therefore atoms can randomly get closer to each other or away from each other.     
        #. agitate (tuple): It's a tuple of two boolean values, at least one of them must be True.
           Whether to agitate the first atom, the second or both. This is useful to set an atom fixed while only 
           the other succumb the agitation to adjust the distance. For instance in a C-H group it can be useful and 
           logical to adjust the bond length by moving only the hydrogen atom along the bond direction.
                   
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Agitations import DistanceAgitationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to random agitations of distance seperating two atoms.
        # Maximum agitation amplitude is set to to 0.5A
        for g in ENGINE.groups:
            if len(g)==2:
                g.set_move_generator( DistanceAgitationGenerator(amplitude=0.5) )
                
    """
    def __init__(self, group=None, amplitude=0.2, symmetric=True, shrink=None, agitate=(True,True)):
        super(DistanceAgitationGenerator, self).__init__(group=group)
        # set amplitude
        self.set_amplitude(amplitude)
        # set symmetric
        self.set_symmetric(symmetric)
        # set shrink
        self.set_shrink(shrink)
        # set agitated
        self.set_agitate(agitate)
        
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
    
    @property
    def agitate(self):
        """Gets agitate tuple flags value."""
        return self.__agitate
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)!=2:
            return False, "two atoms are needed in a group to perform distance agitation movements."
        else:
            return True, "" 
            
    def set_amplitude(self, amplitude):
        """
        Sets maximum translation vector allowed amplitude.
        
        :Parameters:
            #. amplitude (number): the maximum allowed translation vector amplitude.
        """
        assert is_number(amplitude), LOGGER.error("Translation amplitude must be a number")
        amplitude = float(amplitude)
        assert amplitude>0, LOGGER.error("Translation amplitude must be bigger than 0")
        self.__amplitude = FLOAT_TYPE(amplitude)
    
    def set_symmetric(self, symmetric):
        """
        Sets symmetric flag value.
        
        :Parameters:
            #. symmetric (bool): Whether to apply the same amplitude of translation on both atoms or not.         
        """
        assert isinstance(symmetric, bool), LOGGER.error("symmetric must be boolean")
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
        assert shrink in (None, True, False), LOGGER.error("shrink can only be None, True or False")
        self.__shrink = shrink
    
    def set_agitate(self, agitate):
        """
        Sets agitate tuple value.
        
        :Parameters:
            #. agitate (tuple): It's a tuple of two boolean values, at least one of them must be True.
               Whether to agitate the first atom, the second or both. This is useful to set an atom fixed while only 
               the other succumb the agitation to adjust the distance. For instance in a C-H group it can be useful and 
               logical to adjust the bond length by moving only the hydrogen atom along the bond direction.
        """
        assert isinstance(agitate, (list,tuple)), LOGGER.error("agitate must be a list or a tuple")
        assert len(agitate)==2, LOGGER.error("agitate must have 2 items")
        assert [isinstance(a,bool) for a in agitate]==[True,True], LOGGER.error("agitate items must be boolean")
        assert agitate[0] or agitate[1], LOGGER.error("agitate both items can't be False")
        self.__agitate = (agitate[0], agitate[1])     

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
        translationVectors      = np.empty((2,3), dtype=FLOAT_TYPE)
        translationVectors[0,:] = self.__agitate[0]*dir0*amp0*vector
        translationVectors[1,:] = self.__agitate[1]*dir1*amp1*vector
        # translate and return
        return coordinates+translationVectors
 

class AngleAgitationGenerator(MoveGenerator):
    """
    Generates random agitation moves upon an angle defined between two vectors left-central and right-central 
    where (central, left, right) are three atoms. Move will be performed on left and/or right atom while 
    central atom will always remain fixed. Distances between left/right and central atoms will remain
    unchanged. This is mainly used to shake bonded atoms angles by increasing and decreasing 
    the bond length.
    Only groups of length 3 are accepted.
    
    :Parameters:
        #. group (None, Group): The group instance. It must contain exactly three indexes in respective
           order (central, left, right) atoms indexes.
        #. amplitude (number):  The maximum agitation angle amplitude in degrees of left and right atoms separately.
        #. symmetric (bool): Whether to apply the same amplitude of rotation on both left and right atoms or not.
        #. shrink (None, bool): Whether to always shrink the angle or expand it.
           If True, moves will always reduce angle.
           If False, moves will always increase angle.
           If None, no orientation is forced, therefore angle can randomly get wider or tighter.     
        #. agitate (tuple): It's a tuple of two boolean values for respectively (left, right) atoms, 
           at least one of them must be True. Whether to agitate the left atom, the right or both. 
           This is useful to set an atom fixed while only the other succumb the agitation to adjust the angle.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Generators.Agitations import AngleAgitationGenerator
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # Add constraints ...
        # Re-define groups if needed ...
        # Re-define groups selector if needed ...
        
        # set moves generators to random agitations of the angle formed between
        # one central atom and other two. Maximum agitation amplitude is set to to 10.
        for g in ENGINE.groups:
            if len(g)==3:
                g.set_move_generator( AngleAgitationGenerator(amplitude=10) )
                
    """
    def __init__(self, group=None, amplitude=2, symmetric=True, shrink=None, agitate=(True,True)):
        super(AngleAgitationGenerator, self).__init__(group=group)
        # set amplitude
        self.set_amplitude(amplitude)
        # set symmetric
        self.set_symmetric(symmetric)
        # set shrink
        self.set_shrink(shrink)
        # set agitated
        self.set_agitate(agitate)
        
    @property
    def amplitude(self):
        """Gets the maximum agitation angle amplitude in rad."""
        return self.__amplitude
    
    @property
    def shrink(self):
        """Gets shrink flag value."""
        return self.__shrink
    
    @property
    def symmetric(self):
        """Gets symmetric flag value."""
        return self.__symmetric
    
    @property
    def agitate(self):
        """Gets agitate tuple flags value."""
        return self.__agitate
        
    def check_group(self, group):
        """
        Checks the generator's group.
        
        :Parameters:
            #. group (Group): the Group instance.
        """
        if len(group.indexes)!=3:
            return False, "Exactly three atoms are needed in a group to perform angle agitation movements."
        else:
            return True, "" 
            
    def set_amplitude(self, amplitude):
        """
        Sets maximum allowed agitation rotation angle amplitude in degrees of left and right atoms separately and transforms it to rad.
        
        :Parameters:
            #. amplitude (number):  The maximum agitation angle amplitude in degrees of left and right atoms separately.
        """
        assert is_number(amplitude), LOGGER.error("Agitation angle amplitude must be a number")
        amplitude = float(amplitude)
        assert amplitude>0, LOGGER.error("Agitation angle amplitude must be bigger than 0")
        assert amplitude<=90, LOGGER.error("Agitation angle amplitude must be smaller than 90")
        self.__amplitude = FLOAT_TYPE(amplitude*PI/180.)
    
    def set_symmetric(self, symmetric):
        """
        Sets symmetric flag value.
        
        :Parameters:
            #. symmetric (bool): Whether to apply the same amplitude of translation on both atoms or not.         
        """
        assert isinstance(symmetric, bool), LOGGER.error("symmetric must be boolean")
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
        assert shrink in (None, True, False), LOGGER.error("shrink can only be None, True or False")
        self.__shrink = shrink
    
    def set_agitate(self, agitate):
        """
        Sets agitate tuple value.
        
        :Parameters:
            #. agitate (tuple): It's a tuple of two boolean values, at least one of them must be True.
               Whether to agitate the first atom, the second or both. This is useful to set an atom fixed while only 
               the other succumb the agitation to adjust the distance. For instance in a C-H group it can be useful and 
               logical to adjust the bond length by moving only the hydrogen atom along the bond direction.
        """
        assert isinstance(agitate, (list,tuple)), LOGGER.error("agitate must be a list or a tuple")
        assert len(agitate)==2, LOGGER.error("agitate must have 2 items")
        assert [isinstance(a,bool) for a in agitate]==[True,True], LOGGER.error("agitate items must be boolean")
        assert agitate[0] or agitate[1], LOGGER.error("agitate both items can't be False")
        self.__agitate = (agitate[0], agitate[1])     

    def transform_coordinates(self, coordinates, argument=None):
        """
        Translate coordinates.
        
        :Parameters:
            #. coordinates (np.ndarray): The coordinates on which to apply the translation.
            
        :Returns:
            #. coordinates (np.ndarray): The new coordinates after applying the translation.
            #. argument (object): Any python object. Not used in this generator.
        """
        # get atoms group center
        center = np.sum(coordinates, 0)/coordinates.shape[0]
        # translate to origin
        rotatedCoordinates = coordinates-center
        # get normalized direction vectors
        leftVector   = FLOAT_TYPE( rotatedCoordinates[1,:]-rotatedCoordinates[0,:] )
        leftVector  /= FLOAT_TYPE( np.linalg.norm(leftVector) )
        rightVector  = FLOAT_TYPE( rotatedCoordinates[2,:]-rotatedCoordinates[0,:] )
        rightVector /= FLOAT_TYPE( np.linalg.norm(rightVector) )
        # get rotation axis
        rotationAxis = np.cross(leftVector, rightVector)
        if rotationAxis[0]==rotationAxis[1]==rotationAxis[2]==0.:
            rotationAxis = np.array(1-2*np.random.random(3), dtype=FLOAT_TYPE)
            rotationAxis /= FLOAT_TYPE( np.linalg.norm(rotationAxis) )
        # create shrink flag
        if self.__shrink is None:
            shrink = (1-2*generate_random_float())>0
        else:
            shrink = self.__shrink    
        # get rotation angles
        if self.__symmetric:
            angleLeft  = angleRight = FLOAT_TYPE(generate_random_float()*self.__amplitude)
        else:
            angleLeft  = FLOAT_TYPE(generate_random_float()*self.__amplitude)
            angleRight = FLOAT_TYPE(generate_random_float()*self.__amplitude)
        # create directions
        if shrink:
            angleLeft  *= FLOAT_TYPE(-1)
            angleRight *= FLOAT_TYPE( 1)            
        else:
            angleLeft  *= FLOAT_TYPE( 1)
            angleRight *= FLOAT_TYPE(-1) 
        # rotate
        if self.__agitate[0]:
            rotationMatrix = get_rotation_matrix(rotationAxis, angleLeft)
            rotatedCoordinates[1,:] = np.dot( rotationMatrix, rotatedCoordinates[1,:])
        if self.__agitate[1]:
            rotationMatrix = get_rotation_matrix(rotationAxis, angleRight)
            rotatedCoordinates[2,:] = np.dot( rotationMatrix, rotatedCoordinates[2,:])
        # translate back from center and return
        return np.array(rotatedCoordinates+center, dtype=FLOAT_TYPE)
        
        
        
        