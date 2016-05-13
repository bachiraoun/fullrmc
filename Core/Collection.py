"""
It contains a collection of methods and classes that are useful for the package.
"""

# standard libraries imports
import os
import sys
import time
from random import random  as generate_random_float   # generates a random float number between 0 and 1
from random import randint as generate_random_integer # generates a random integer number between given lower and upper limits

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, LOGGER


def SingletonDecorator(cls):
    """ A class singleton decorator. """
    instance = cls()
    instance.__call__ = lambda: instance
    return instance
    
def is_number(number):
    """
    check if number is convertible to float.
    
    :Parameters:
        #. number (str, number): input number
                   
    :Returns:
        #. result (bool): True if convertible, False otherwise
    """
    if isinstance(number, (int, long, float, complex)):
        return True
    try:
        float(number)
    except:
        return False
    else:
        return True
        
def is_integer(number, precision=10e-10):
    """
    check if number is convertible to integer.
    
    :Parameters:
        #. number (str, number): input number
        #. precision (number): To avoid floating errors, a precision should be given.
                   
    :Returns:
        #. result (bool): True if convertible, False otherwise
    """
    if isinstance(number, (int, long)):
        return True
    try:
        number = float(number)
    except:
        return False
    else:
        if np.abs(number-int(number)) < precision:
            return True
        else:
            return False
    
def get_elapsed_time(start, format="%d days, %d hours, %d minutes, %d seconds"):
    """
    Gets formated time elapsed.
        
    :Parameters:
        #. start (time.time): A time instance.
        #. format (string): The format string. must contain exactly four '%d'.
    """
    # get all time info
    days    = divmod(time.time()-start,86400)
    hours   = divmod(days[1],3600)
    minutes = divmod(hours[1],60)
    seconds = minutes[1]
    return format % (days[0],hours[0],minutes[0],seconds)

   
def get_path(key=None):
    """
    get all information needed about the script, the current, and the python executable path.
    
    :Parameters:
        #. key (None, string): the path to return. If not None, it can take any of the following:
        
            #. cwd:                 current working directory
            #. script:              the script's total path
            #. exe:                 python executable path
            #. script_name:         the script name
            #. relative_script_dir: the script's relative directory path
            #. script_dir:          the script's absolute directory path
            #. fullrmc:             fullrmc package path
                   
    :Returns:
        #. path (dictionary, value): If key is not None it returns the value of paths dictionary key.
           Otherwise all the dictionary is returned.
    """
    import fullrmc
    # check key type
    if key is not None:
        assert isinstance(key, basestring), LOGGER.error("key must be a string of None")
        key=str(key).lower().strip()
    # create paths
    paths = {}
    paths["cwd"]                 = os.getcwd()
    paths["script"]              = sys.argv[0]
    paths["exe"]                 = os.path.dirname(sys.executable)
    pathname, scriptName         = os.path.split(sys.argv[0])
    paths["script_name"]         = scriptName
    paths["relative_script_dir"] = pathname
    paths["script_dir"]          = os.path.abspath(pathname)
    paths["fullrmc"]             = os.path.split(fullrmc.__file__)[0]
    # return paths
    if key is None:
        return paths
    else:
        assert paths.has_key(key), LOGGER.error("key is not defined")
        return paths[key]

        
def rebin(data, bin=0.05, check=False):
    """
    Re-bin 2D data of shape (N,2).
    
    :Parameters:
        #. data (numpy.ndarray): the (N,2) shape data.
        #. bin (number): the new bin size.
        #. check (boolean): whether to check arguments before rebining.
        
    :Returns:
        #. X (numpy.ndarray): the first column re-binned.
        #. Y (numpy.ndarray): the second column re-binned.
    """
    if check:
        assert isinstance(data, np.ndarray), Logger.error("data must be numpy.ndarray instance")
        assert len(data.shape)==2, Logger.error("data must be of 2 dimensions")
        assert data.shape[1] ==2, Logger.error("data must have 2 columns")
        assert is_number(bin), LOGGER.error("bin must be a number")
        bin = float(bin)
        assert bin>0, LOGGER.error("bin must be a positive")
    # rebin
    x = data[:,0].astype(float)
    y = data[:,1].astype(float)
    rx = []
    ry = []
    x0 = int(x[0]/bin)*bin-bin/2.
    xn = int(x[-1]/bin)*bin+bin/2.
    bins = np.arange(x0,xn, bin)
    if bins[-1] != xn:
        bins = np.append(bins, xn)
    # get weights histogram
    W,E = np.histogram(x, bins=bins)
    W[np.where(W==0)[0]] = 1
    # get data histogram
    S,E = np.histogram(x, bins=bins, weights=y)
    # return
    return (E[1:]+E[:-1])/2., S/W
    
def smooth(data, winLen=11, window='hanning', check=False):
    """
    Smooth 1D data using window function and length.
    
    :Parameters:
        #. data (numpy.ndarray): the 1D numpy data.
        #. winLen (integer): the smoothing window length.
        #. window (str): The smoothing window type. Can be anything among
           'flat', 'hanning', 'hamming', 'bartlett' and 'blackman'.
        #. check (boolean): whether to check arguments before smoothing data.
        
    :Returns:
        #. smoothed (numpy.ndarray): the smoothed 1D data array.
    """
    if check:
        assert isinstance(data, np.ndarray), Logger.error("data must be numpy.ndarray instance")
        assert len(data.shape)==1, Logger.error("data must be of 1 dimensions")
        assert is_integer(winLen), LOGGER.error("winLen must be an integer")
        winLen = int(bin)
        assert winLen>=3, LOGGER.error("winLen must be bigger than 3")
        assert data.size < winLen, LOGGER.error("data needs to be bigger than window size.")
        assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'], LOGGER.error("window must be any of ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')")
    # compute smoothed data
    s=np.r_[data[winLen-1:0:-1],data,data[-1:-winLen:-1]]
    if window == 'flat': #moving average
        w=np.ones(winLen,'d')
    else:
        w=eval('np.'+window+'(winLen)')
    S=np.convolve(w/w.sum(),s, mode='valid')
    # get data and return
    f = winLen/2
    t = f-winLen+1
    return S[f:t]
 
def get_random_perpendicular_vector(vector):
    """
    Get random normalized perpendicular vector to a given vector.
    
    :Parameters:
        #. vector (numpy.ndarray, list, set, tuple): the vector to compute a random perpendicular vector to it
        
    :Returns:
        #. perpVector (numpy.ndarray): the perpendicular vector  of type fullrmc.Globals.FLOAT_TYPE 
    """
    vectorNorm = np.linalg.norm(vector) 
    assert vectorNorm, LOGGER.error("vector returned 0 norm")
    # easy cases
    if np.abs(vector[0])<PRECISION:
        return np.array([1,0,0], dtype=FLOAT_TYPE)
    elif np.abs(vector[1])<PRECISION:
        return np.array([0,1,0], dtype=FLOAT_TYPE)
    elif np.abs(vector[2])<PRECISION:
        return np.array([0,0,1], dtype=FLOAT_TYPE)
    # generate random vector
    randVect = 1-2*np.random.random(3)
    randvect = np.array([vector[idx]*randVect[idx] for idx in range(3)])
    # get perpendicular vector
    perpVector = np.cross(randvect,vector)
    # normalize, coerce and return
    return np.array(perpVector/np.linalg.norm(perpVector), dtype=FLOAT_TYPE)
    
        
def get_principal_axis(coordinates, weights=None):
    """
    Calculates the principal axis of a set of atoms coordinates
        
    :Parameters:
        #. coordinates (np.ndarray): The atoms coordinates. 
        #. weights (numpy.ndarray, None): the list of weights for the COM calculation. 
                                          Must be a numpy.ndarray of numbers of the same length as indexes.
                                          None is accepted for equivalent weighting.
        
    :Returns:
        #. center (numpy.ndarray): the geometric center of the records.
        #. eval1 (fullrmc.Globals.FLOAT_TYPE): the biggest eigen value.
        #. eval2 (fullrmc.Globals.FLOAT_TYPE): the second biggest eigen value.
        #. eval3 (fullrmc.Globals.FLOAT_TYPE): the smallest eigen value.
        #. axis1 (numpy.ndarray): the principal axis corresponding to the biggest eigen value.
        #. axis2 (numpy.ndarray): the principal axis corresponding to the second biggest eigen value.
        #. axis3 (numpy.ndarray): the principal axis corresponding to the smallest eigen value.
    """
    # multiply by weights
    if weights is not None:
        coordinates[:,0] *= weights
        coordinates[:,1] *= weights
        coordinates[:,2] *= weights
        norm = np.sum(weights)
    else:
        norm = coordinates.shape[0]   
    # compute center
    center = np.array(np.sum(coordinates, 0)/norm, dtype=FLOAT_TYPE)
    # coordinates in center
    centerCoords = coordinates - center
    # compute principal axis matrix
    inertia = np.dot(centerCoords.transpose(), centerCoords)
    # compute eigen values and eigen vectors
    # warning eigen values are not necessary ordered!
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    e_values, e_vectors = np.linalg.eig(inertia)
    e_values  = list(e_values)
    e_vectors = list(e_vectors.transpose())        
    # get eval1 and axis1
    eval1 = max(e_values)
    vect1 = np.array(e_vectors.pop(e_values.index(eval1)), dtype=FLOAT_TYPE)
    e_values.remove(eval1)    
    # get eval1 and axis1
    eval2 = max(e_values)
    vect2 = np.array(e_vectors.pop(e_values.index(eval2)), dtype=FLOAT_TYPE)
    e_values.remove(eval2)
    # get eval3 and axis3
    eval3 = e_values[0]
    vect3 = np.array(e_vectors[0], dtype=FLOAT_TYPE)
    return center, FLOAT_TYPE(eval1), FLOAT_TYPE(eval2), FLOAT_TYPE(eval3), vect1, vect2, vect3       
        

def get_rotation_matrix(rotationVector, angle):
    """
        Calculates the rotation (3X3) matrix about an axis by a rotation angle.\n
        
        :Parameters:
            #. rotationVector (list, tuple, numpy.ndarray): the rotation vector coordinates.
            #. angle (float): the rotation angle in rad.
           
       :Returns:
            #. rotationMatrix (numpy.ndarray): the (3X3) rotation matrix
    """ 
    angle = float(angle)
    axis = rotationVector/np.sqrt(np.dot(rotationVector , rotationVector))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ] , dtype = FLOAT_TYPE)

def rotate(xyzArray , rotationMatrix):
    """
    Rotates a numpy.array using a rotation matrix.
    The array itself will be rotated and not a copy of it.
    
    :Parameters:
        #. indexes (numpy.ndarray): the xyz (N,3) array to rotate.
        #. rotationMatrix (numpy.ndarray): the (3X3) rotation matrix.
    """
    arrayType = xyzArray.dtype
    for idx in range(xyzArray.shape[0]):
        xyzArray[idx,:] = np.dot( rotationMatrix, xyzArray[idx,:]).astype(arrayType)
    return xyzArray

def get_orientation_matrix(arrayAxis, alignToAxis):
    """
    Get the rotation matrix that aligns arrayAxis to alignToAxis
    
    :Parameters:
        #. arrayAxis (list, tuple, numpy.ndarray): xyzArray axis.
        #. alignToAxis (list, tuple, numpy.ndarray): The axis to align to.
    """
    # normalize alignToAxis
    alignToAxisNorm = np.linalg.norm(alignToAxis)
    assert alignToAxisNorm>0, LOGGER.error("alignToAxis returned 0 norm")
    alignToAxis = np.array(alignToAxis, dtype=FLOAT_TYPE)/alignToAxisNorm
    # normalize arrayAxis
    arrayAxisNorm = np.linalg.norm(arrayAxis)
    assert arrayAxisNorm>0, LOGGER.error("arrayAxis returned 0 norm")
    arrayAxis = np.array(arrayAxis, dtype=FLOAT_TYPE)/arrayAxisNorm
    # calculate rotationAngle
    dotProduct = np.dot(arrayAxis, alignToAxis)
    if np.abs(dotProduct-1) <= PRECISION :
        rotationAngle = 0
    elif np.abs(dotProduct+1) <= PRECISION :
        rotationAngle = PI
    else:
        rotationAngle = np.arccos( dotProduct )
    if np.isnan(rotationAngle) or np.abs(rotationAngle) <= PRECISION :
        return np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]).astype(FLOAT_TYPE)
    # calculate rotation axis.
    if np.abs(rotationAngle-PI) <= PRECISION:
        rotationAxis = get_random_perpendicular_vector(arrayAxis)
    else:
        rotationAxis = np.cross(alignToAxis, arrayAxis)
    #rotationAxis /= np.linalg.norm(rotationAxis)
    # calculate rotation matrix
    return get_rotation_matrix(rotationAxis, rotationAngle)
  
  
def orient(xyzArray, arrayAxis, alignToAxis):
    """
    Rotates xyzArray using the rotation matrix that rotates and aligns arrayAxis to alignToAXis.
    
    :Parameters:
        #. xyzArray (numpy.ndarray): the xyz (N,3) array to rotate.
        #. arrayAxis (list, tuple, numpy.ndarray): xyzArray axis.
        #. alignToAxis (list, tuple, numpy.ndarray): The axis to align to.
    """
    # normalize alignToAxis
    alignToAxisNorm = np.linalg.norm(alignToAxis)
    assert alignToAxisNorm>0, LOGGER.error("alignToAxis returned 0 norm")
    alignToAxis = np.array(alignToAxis, dtype=FLOAT_TYPE)/alignToAxisNorm
    # normalize arrayAxis
    arrayAxisNorm = np.linalg.norm(arrayAxis)
    assert arrayAxisNorm>0, LOGGER.error("arrayAxis returned 0 norm")
    arrayAxis = np.array(arrayAxis, dtype=FLOAT_TYPE)/arrayAxisNorm
    # calculate rotationAngle
    dotProduct = np.dot(arrayAxis, alignToAxis)
    if np.abs(dotProduct-1) <= PRECISION :
        rotationAngle = 0
    elif np.abs(dotProduct+1) <= PRECISION :
        rotationAngle = PI
    else:
        rotationAngle = np.arccos( dotProduct )
    if np.isnan(rotationAngle) or np.abs(rotationAngle) <= PRECISION :
        return xyzArray
    # calculate rotation axis.
    if np.abs(rotationAngle-PI) <= PRECISION:
        rotationAxis = get_random_perpendicular_vector(arrayAxis)
    else:
        rotationAxis = np.cross(alignToAxis, arrayAxis)
    #rotationAxis /= np.linalg.norm(rotationAxis)
    # calculate rotation matrix
    rotationMatrix = get_rotation_matrix(rotationAxis, rotationAngle)
    # rotate and return
    return rotate(xyzArray , rotationMatrix)
  


def get_superposition_transformation(refArray, array, check=False):
    """
        Calculates the rotation matrix and the translations that minimizes the root mean 
        square deviation between and array of vectors and a reference array.\n   
        
        :Parameters:
            #. refArray (numpy.ndarray): the NX3 reference array to superpose to.
            #. array (numpy.ndarray): the NX3 array to calculate the transformation of.
            #. check (boolean): whether to check arguments before generating points.
        
        :Returns:
            #. rotationMatrix (numpy.ndarray): the 3X3 rotation tensor.
            #. refArrayCOM (numpy.ndarray): the 1X3 vector center of mass of refArray.
            #. arrayCOM (numpy.ndarray): the 1X3 vector center of mass of array.
            #. rms (number)
    """ 
    if check:
        # check array
        assert isinstance(array, np.ndarray), Logger.error("array must be numpy.ndarray instance")
        assert len(array.shape)<=2, Logger.error("array must be a vector or a matrix")
        if len(array.shape)==2:
            assert array.shape[1]==3, Logger.error("array number of columns must be 3")
        else:
            assert array.shape[1]==3, Logger.error("vector array number of columns must be 3")
        # check refArray
        assert isinstance(refArray, np.ndarray), Logger.error("refArray must be numpy.ndarray instance")
        assert len(refArray.shape)<=2, Logger.error("refArray must be a vector or a matrix")
        if len(refArray.shape)==2:
            assert refArray.shape[1]==3, Logger.error("refArray number of columns must be 3")
        else:
            assert refArray.shape[1]==3, Logger.error("vector refArray number of columns must be 3")
        # check weights
        assert array.shape == refArray.shape, Logger.error("refArray and array must have the same number of vectors")     
    # calculate center of mass of array
    arrayCOM = np.sum(array, axis=0)/array.shape[0]
    # calculate cross matrix and reference config center of mass
    r_ref = array-arrayCOM
    refArrayCOM = np.sum(refArray, axis=1)
    cross = np.dot(refArray.transpose(),r_ref)
    possq = np.add.reduce(refArray**2,1)+np.add.reduce(r_ref**2,1)
    possq = np.sum(possq)
    # calculate kross
    kross = np.zeros((4, 4), dtype=FLOAT_TYPE)
    kross[0, 0] = -cross[0, 0]-cross[1, 1]-cross[2, 2]
    kross[0, 1] =  cross[1, 2]-cross[2, 1]
    kross[0, 2] =  cross[2, 0]-cross[0, 2]
    kross[0, 3] =  cross[0, 1]-cross[1, 0]
    kross[1, 1] = -cross[0, 0]+cross[1, 1]+cross[2, 2]
    kross[1, 2] = -cross[0, 1]-cross[1, 0]
    kross[1, 3] = -cross[0, 2]-cross[2, 0]
    kross[2, 2] =  cross[0, 0]-cross[1, 1]+cross[2, 2]
    kross[2, 3] = -cross[1, 2]-cross[2, 1]
    kross[3, 3] =  cross[0, 0]+cross[1, 1]-cross[2, 2]
    for i in range(1, 4):
        for j in range(i):
            kross[i, j] = kross[j, i]
    kross = 2.*kross
    offset = possq - np.add.reduce(refArrayCOM**2)
    for i in range(4):
        kross[i, i] = kross[i, i] + offset
    # get eigen values
    e, v = np.linalg.eig(kross)
    i = np.argmin(e)
    v = np.array(v[:,i], dtype=FLOAT_TYPE)
    if v[0] < 0: v = -v
    if e[i] <= 0.:
        rms = FLOAT_TYPE(0.)
    else:
        rms = np.sqrt(e[i])
    # calculate the rotation matrix
    rot = np.zeros((3,3,4,4), dtype=FLOAT_TYPE)
    rot[0,0, 0,0] = FLOAT_TYPE( 1.0)
    rot[0,0, 1,1] = FLOAT_TYPE( 1.0)
    rot[0,0, 2,2] = FLOAT_TYPE(-1.0)
    rot[0,0, 3,3] = FLOAT_TYPE(-1.0)
    rot[1,1, 0,0] = FLOAT_TYPE( 1.0)
    rot[1,1, 1,1] = FLOAT_TYPE(-1.0)
    rot[1,1, 2,2] = FLOAT_TYPE( 1.0)
    rot[1,1, 3,3] = FLOAT_TYPE(-1.0)
    rot[2,2, 0,0] = FLOAT_TYPE( 1.0)
    rot[2,2, 1,1] = FLOAT_TYPE(-1.0)
    rot[2,2, 2,2] = FLOAT_TYPE(-1.0)
    rot[2,2, 3,3] = FLOAT_TYPE( 1.0)
    rot[0,1, 1,2] = FLOAT_TYPE( 2.0)
    rot[0,1, 0,3] = FLOAT_TYPE(-2.0)
    rot[0,2, 0,2] = FLOAT_TYPE( 2.0)
    rot[0,2, 1,3] = FLOAT_TYPE( 2.0)
    rot[1,0, 0,3] = FLOAT_TYPE( 2.0)
    rot[1,0, 1,2] = FLOAT_TYPE( 2.0)
    rot[1,2, 0,1] = FLOAT_TYPE(-2.0)
    rot[1,2, 2,3] = FLOAT_TYPE( 2.0)
    rot[2,0, 0,2] = FLOAT_TYPE(-2.0)
    rot[2,0, 1,3] = FLOAT_TYPE( 2.0)
    rot[2,1, 0,1] = FLOAT_TYPE( 2.0)
    rot[2,1, 2,3] = FLOAT_TYPE( 2.0)
    rotationMatrix = np.dot(np.dot(rot, v), v)
    return rotationMatrix, refArrayCOM, arrayCOM, rms


def superpose_array(refArray, array, check=False):
    """
        Calculates the rotation matrix and the translations that minimizes the root mean 
        square deviation between and array of vectors and a reference array.\n   
        
        :Parameters:
            #. refArray (numpy.ndarray): the NX3 reference array to superpose to.
            #. array (numpy.ndarray): the NX3 array to calculate the transformation of.
            #. check (boolean): whether to check arguments before generating points.
        
        :Returns:
            #. superposedArray (numpy.ndarray): the NX3 array to superposed array.
    """ 
    rotationMatrix, _,_,_ = get_superposition_transformation(refArray=refArray, array=array, check=check)     
    return np.dot( rotationMatrix, np.transpose(array).\
                   reshape(1,3,-1)).transpose().reshape(-1,3)
    
    
def generate_points_on_sphere(thetaFrom, thetaTo,
                              phiFrom, phiTo,
                              npoints=1,
                              check=False):
    """
    Generate random points on a sphere of radius 1. Points are generated using spherical coordinates
    arguments as in figure below. Theta [0,Pi] is the angle between the generated point and Z axis.
    Phi [0,2Pi] is the angle between the generated point and x axis.
        
    .. image:: sphericalCoordsSystem.png   
       :align: left 
       :height: 200px
       :width: 200px
        
    :Parameters:
        #. thetaFrom (number): The minimum theta value.
        #. thetaTo (number): The maximum theta value.
        #. phiFrom (number): The minimum phi value.
        #. phiTo (number): The maximum phi value.
        #. npoints (integer): The number of points to generate
        #. check (boolean): whether to check arguments before generating points.
                   
    :Returns:
        #. x (numpy.ndarray): The (npoints,1) numpy array of all generated points x coordinates.
        #. y (numpy.ndarray): The (npoints,1) numpy array of all generated points y coordinates.
        #. z (numpy.ndarray): The (npoints,1) numpy array of all generated points z coordinates.
    """
    if check:
        assert isinteger(npoints)   , LOGGER.error("npoints must be an integer")
        assert is_number(thetaFrom) , LOGGER.error("thetaFrom must be a number") 
        assert is_number(thetaTo)   , LOGGER.error("thetaTo must be a number")  
        assert is_number(phiFrom)   , LOGGER.error("phiFrom must be a number")
        assert is_number(phiTo)     , LOGGER.error("phiTo must be a number")
        npoints   = INT_TYPE(npoints)
        thetaFrom = FLOAT_TYPE(thetaFrom)
        thetaTo   = FLOAT_TYPE(thetaTo)
        phiFrom   = FLOAT_TYPE(phiFrom)
        phiTo     = FLOAT_TYPE(phiTo)
        assert npoints>=1        , LOGGER.error("npoints must be bigger than 0")
        assert thetaFrom>=0      , LOGGER.error("thetaFrom must be positive")
        assert thetaTo<=PI       , LOGGER.error("thetaTo must be smaller than %.6f"%PI)
        assert thetaFrom<thetaTo , LOGGER.error("thetaFrom must be smaller than thetaTo")
        assert phiFrom>=0        , LOGGER.error("phiFrom must be positive")
        assert phiTo<=2*PI       , LOGGER.error("phiTo mast be smaller than %.6f"%2*PI)
        assert phiFrom<phiTo     , LOGGER.error("phiFrom must be smaller than phiTo")
    else:
        # cast variables
        npoints   = INT_TYPE(npoints)
        thetaFrom = FLOAT_TYPE(thetaFrom)
        thetaTo   = FLOAT_TYPE(thetaTo)
        phiFrom   = FLOAT_TYPE(phiFrom)
        phiTo     = FLOAT_TYPE(phiTo)
    # calculate differences
    deltaTheta = thetaTo-thetaFrom
    deltaPhi = phiTo-phiFrom 
    # theta
    theta  = thetaFrom+np.random.random(npoints).astype(FLOAT_TYPE)*deltaTheta
    # phi
    phi  = phiFrom+np.random.random(npoints).astype(FLOAT_TYPE)*deltaPhi
    # compute sin and cos
    sinTheta = np.sin(theta) 
    sinPhi   = np.sin(phi)
    cosTheta = np.cos(theta)
    cosPhi   = np.cos(phi)
    # compute x,y,z
    x = sinTheta * cosPhi
    y = sinTheta * sinPhi
    z = cosTheta
    # return
    return x,y,z
           

def find_extrema(x, max = True, min = True, strict = False, withend = False):
    """
    Get a vector extrema indexes and values.
    
    :Parameters:
        #. max (boolean): Whether to index the maxima
        #. min (boolean): Whether to index the minima
    	#. strict (boolean): Whether not to index changes to zero gradient
    	#. withend (boolean): Whether to always include x[0] and x[-1]
    
    :Returns:
        #. indexes (numpy.ndarray): Extrema indexes 
        #. values (numpy.ndarray): Extrema values
    """
    # This is the gradient
    dx = np.empty(len(x))
    dx[1:] = np.diff(x)
    dx[0] = dx[1]
    # Clean up the gradient in order to pick out any change of sign
    dx = np.sign(dx)
    # define the threshold for whether to pick out changes to zero gradient
    threshold = 0
    if strict:
    	threshold = 1
    # Second order diff to pick out the spikes
    d2x = np.diff(dx)
    if max and min:
    	d2x = abs(d2x)
    elif max:
    	d2x = -d2x
    # Take care of the two ends
    if withend:
    	d2x[0] = 2
    	d2x[-1] = 2
    # Sift out the list of extremas
    ind = np.nonzero(d2x > threshold)[0]
    return ind, x[ind]
    

def convert_Gr_to_gr(Gr, minIndex):
    """
    Converts G(r) to g(r) by computing the 
    following :math:`g(r)=1+(\\frac{G(r)}{4 \\pi \\rho_{0} r})`
    
    :Parameters:
       #. Gr (numpy.ndarray): The G(r) numpy array of shape (number of points, 2)
       #. minIndex (int, tuple): The minima indexes to compute the number density rho0.
          It can be a single peak or a list of peaks to compute the mean slope instead.
    
    :Returns:
       #. minimas (numpy.ndarray): The minimas array found using minIndex and used to compute the slope and therefore :math:`\\rho_{0}`.
       #. slope (float): The computed slope from the minimas.
       #. rho0 (float): The number density of the material.
       #. g(r) (numpy.ndarray): the computed g(r).
       
    
    **To visualize convertion**
       
    .. code-block:: python
        
        # peak indexes can be different, adjust according to your data
        minPeaksIndex = [1,3,4]
        minimas, slope, rho0, gr =  convert_Gr_to_gr(Gr, minIndex=minPeaksIndex)
        print 'slope: %s --> rho0: %s'%(slope,rho0)
        import matplotlib.pyplot as plt
        line = np.transpose( [[0, Gr[-1,0]], [0, slope*Gr[-1,0]]] )
        plt.plot(Gr[:,0],Gr[:,1], label='G(r)')
        plt.plot(minimas[:,0], minimas[:,1], 'o', label='minimas')
        plt.plot(line[:,0], line[:,1], label='density')
        plt.plot(gr[:,0],gr[:,1], label='g(r)')
        plt.legend()
        plt.show()
    
    
    """
    # check G(r)
    assert isinstance(Gr, np.ndarray), "Gr must be a numpy.ndarray"
    assert len(Gr.shape)==2, "Gr be of shape length 2"
    assert Gr.shape[1] == 2, "Gr must be of shape (n,2)"
    # check minIndex
    if not isinstance(minIndex, (list, set, tuple)):
        minIndex = [minIndex]
    else:
        minIndex = list(minIndex)
    for idx, mi in enumerate(minIndex):
        assert is_integer(mi), "minIndex must be integers"
        mi = int(mi)
        assert mi>=0, "minIndex must be >0"
        minIndex[idx] = mi
    # get minsIndexes
    minsIndexes = find_extrema(x=Gr[:,1], max=False)[0]
    # get minimas points
    minIndex = [minsIndexes[mi] for mi in minIndex]
    minimas  = np.transpose([Gr[minIndex,0], Gr[minIndex,1]])
    # compute slope
    x = float( np.mean(minimas[:,0]) )
    y = float( np.mean(minimas[:,1]) )
    slope = y/x
    # compute rho
    rho0 = -slope/4./np.pi
    # compute g(r)
    gr = 1+Gr[:,1]/(-slope*Gr[:,0])
    gr = np.transpose( [Gr[:,0], gr] )
    # return
    return minimas, slope, rho0, gr
    #minimas, slope, rho0, gr =  convert_Gr_to_gr(Gr, minIndex=peakIndex)
    #print 'slope: %s --> rho0: %s'%(slope,rho0)
    #import matplotlib.pyplot as plt
    #line = np.transpose( [[0, Gr[-1,0]], [0, slope*Gr[-1,0]]] )
    #plt.plot(Gr[:,0],Gr[:,1], label='G(r)')
    #plt.plot(minimas[:,0], minimas[:,1], 'o', label='minimas')
    #plt.plot(line[:,0], line[:,1], label='density')
    #plt.plot(gr[:,0],gr[:,1], label='g(r)')
    #plt.legend()
    #plt.show()

    
def generate_vectors_in_solid_angle(direction,
                                    maxAngle,
                                    numberOfVectors=1,
                                    check=False):
    """
    Generate random vectors that satisfy angle condition with a direction vector.
    Angle between any generated vector and direction must be smaller than given maxAngle.
    
    +----------------------------------------+----------------------------------------+----------------------------------------+ 
    |.. figure:: 100randomVectors30deg.png   |.. figure:: 200randomVectors45deg.png   |.. figure:: 500randomVectors100deg.png  | 
    |   :width: 400px                        |   :width: 400px                        |   :width: 400px                        | 
    |   :height: 300px                       |   :height: 300px                       |   :height: 300px                       |
    |   :align: left                         |   :align: left                         |   :align: left                         | 
    |                                        |                                        |                                        |
    |   a) 100 vectors generated around      |   b) 200 vectors generated around      |   b) 500 vectors generated around      |
    |   OX axis within a maximum angle       |   [1,-1,1] axis within a maximum angle |   [2,5,1] axis within a maximum angle  |
    |   separation of 30 degrees.            |   separation of 45 degrees.            |   separation of 100 degrees.           |
    +----------------------------------------+----------------------------------------+----------------------------------------+

    :Parameters:
        #. direction (number): The direction around which to create the vectors.
        #. maxAngle (number): The maximum angle allowed.
        #. numberOfVectors (integer): The number of vectors to generate.
        #. check (boolean): whether to check arguments before generating vectors.
                   
    :Returns:
        #. vectors (numpy.ndarray): The (numberOfVectors,3) numpy array of generated vectors.
    """
    if check:
        assert isinstance(direction, (list,set,tuple,np.array)), LOGGER.error("direction must be a vector like list or array of length 3")
        direction = list(direction)
        assert len(direction)==3, LOGGER.error("direction vector must be of length 3")
        dir = []
        for d in direction:
            assert is_number(d), LOGGER.error("direction items must be numbers")
            dir.append(FLOAT_TYPE(d))
        direction = np.array(dir)
        assert direction[0]>PRECISION or direction[1]>PRECISION or direction[2]>PRECISION, LOGGER.error("direction must be a non zero vector")
        assert is_number(maxAngle), LOGGER.error("maxAngle must be a number")
        maxAngle = FLOAT_TYPE(maxAngle)
        assert maxAngle>0, LOGGER.error("maxAngle must be bigger that zero")
        assert maxAngle<=PI, LOGGER.error("maxAngle must be smaller than %.6f"%PI)
        assert is_integer(numberOfVectors), LOGGER.error("numberOfVectors must be integer")
        numberOfVectors = INT_TYPE(numberOfVectors)
        assert numberOfVectors>0, LOGGER.error("numberOfVectors must be bigger than 0")
    # create axis
    axis = np.array([1,0,0]) 
    if np.abs(direction[1])<=PRECISION and np.abs(direction[2])<=PRECISION:
        axis *= np.sign(direction[0])
    # create vectors
    vectors = np.zeros((numberOfVectors,3))
    vectors[:,1] = 1-2*np.random.random(numberOfVectors)
    vectors[:,2] = 1-2*np.random.random(numberOfVectors)#np.sign(np.random.random(numberOfVectors)-0.5)*np.sqrt(1-vectors[:,1]**2)
    norm = np.sqrt(np.add.reduce(vectors**2, axis=1))
    vectors[:,1] /= norm
    vectors[:,2] /= norm
    # create rotation angles
    rotationAngles = PI/2.-np.random.random(numberOfVectors)*maxAngle
    # create rotation axis
    rotationAxes = np.cross(axis, vectors)
    # rotate vectors to axis
    for idx in range(numberOfVectors):
        rotationMatrix = get_rotation_matrix(rotationAxes[idx,:], rotationAngles[idx])
        vectors[idx,:] = np.dot( rotationMatrix, vectors[idx,:])
    # normalize direction
    direction /= np.linalg.norm(direction)
    # get rotation matrix of axis to direction
    rotationAngle  = np.arccos( np.dot(direction, axis) )
    if rotationAngle > PRECISION:       
        rotationAxis   = np.cross(direction, axis)
        rotationMatrix = get_rotation_matrix(rotationAxis, rotationAngle)
        # rotate vectors to direction
        for idx in range(numberOfVectors):
            vectors[idx,:] = np.dot( rotationMatrix, vectors[idx,:])
    return vectors.astype(FLOAT_TYPE)


def gaussian(x, center=0, FWHM=1, normalize=True, check=True):
    """
    Compute the normal distribution or gaussian distribution of a given vector.
    The probability density of the gaussian distribution is:
    :math:`f(x,\\mu,\\sigma) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{\\frac{-(x-\\mu)^{2}}{2\\sigma^2}}`
     
    Where:\n
    * :math:`\\mu` is the center of the gaussian, it is the mean or expectation of the distribution it is called the distribution's median or mode. 
    * :math:`\\sigma` is its standard deviation.
    * :math:`FWHM=2\\sqrt{2 ln 2} \\sigma` is the Full Width at Half Maximum of the gaussian. 
    
    :Parameters:
        #. x (numpy.ndarray): The vector to compute the gaussian
        #. center (number): The center of the gaussian.
        #. FWHM (number): The Full Width at Half Maximum of the gaussian.
        #. normalize(boolean): Whether to normalize the generated gaussian by :math:`\\frac{1}{\\sigma\\sqrt{2\\pi}}` so the integral is equal to 1. 
        #. check (boolean): whether to check arguments before generating vectors.
    """
    if check:
        assert is_number(center), LOGGER.error("center must be a number")
        center = FLOAT_TYPE(center)
        assert is_number(FWHM), LOGGER.error("FWHM must be a number")
        FWHM = FLOAT_TYPE(FWHM)
        assert FWHM>0, LOGGER.error("FWHM must be bigger than 0")
        assert isinstance(normalize, bool), LOGGER.error("normalize must be boolean")
    sigma       = FWHM/(2.*np.sqrt(2*np.log(2)))
    expKernel   = ((x-center)**2) / (-2*sigma**2)
    exp         = np.exp(expKernel)
    scaleFactor = 1.
    if normalize:
        scaleFactor /= sigma*np.sqrt(2*np.pi)
    return (scaleFactor * exp).astype(FLOAT_TYPE)
    

def step_function(x, center=0, FWHM=0.1, height=1, check=True):
    """
    Compute a step function as the cumulative summation of a gaussian distribution of a given vector.
    
    :Parameters:
        #. x (numpy.ndarray): The vector to compute the gaussian
        #. center (number): The center of the step function which is the the center of the gaussian.
        #. FWHM (number): The Full Width at Half Maximum of the gaussian.
        #. height (number): The height of the step function.
        #. check (boolean): whether to check arguments before generating vectors.
    """
    if check:
        assert is_number(height), LOGGER.error("height must be a number")
        height = FLOAT_TYPE(height)
    g  = gaussian(x, center=center, FWHM=FWHM, normalize=False, check=check)
    sf = np.cumsum(g)
    sf /= sf[-1]
    return (sf*height).astype(FLOAT_TYPE)    

    
class Broadcaster(object):
    """ 
    A broadcaster broadcasts a message to all listener throughout execution.
    """
    def __init__(self):
        self.__listeners = []
        
    def add_listener(self, listener):
        """
        Add listener to the list of listeners
        
        :Parameters:
            #. listener (object): Any python object having a listen method.
        """
        assert callable(getattr(listener, "listen", None)), LOGGER.error("listener has no 'listen' method.")
        if listener not in self.__listeners:
            self.__listeners.append(listener)
        else:
            LOGGER.warn("listener already exist in broadcaster list")
            
    def remove_listener(self, listener):
        """
        Remove listener to the list of listeners
        
        :Parameters:
            #. listener (object): The listener object to remove.
        """
        if listener in self.__listeners:
            self.__listeners.remove(listener)
            
    def broadcast(self, message, arguments=None):
        """
        Broadcast a message to all the listeners
        
        :Parameters:
            #. message (object): Any type of message object to pass to the listeners.
            #. arguments (object): Any type of argument to pass to the listeners.
        """
        for l in self.__listeners:
            l.listen(message, arguments)
            
        
class RandomFloatGenerator(object):
    """
    Generate random float number between a lower and an upper limit.
    
    :Parameters:
        #. lowerLimit (number): The lower limit allowed.
        #. upperLimit (number): The upper limit allowed.
    """
    def __init__(self, lowerLimit, upperLimit):
         self.__lowerLimit = None
         self.__upperLimit = None
         self.set_lower_limit(lowerLimit)
         self.set_upper_limit(upperLimit)
         
    @property
    def lowerLimit(self):
        """The lower limit of the number generation."""
        return self.__lowerLimit
        
    @property
    def upperLimit(self):
        """The upper limit of the number generation."""
        return self.__upperLimit
        
    @property
    def rang(self):
        """The range defined as upperLimit-lowerLimit."""
        return self.__rang
        
    def set_lower_limit(self, lowerLimit):   
        """
        Set lower limit.
        
        :Parameters:
            #. lowerLimit (number): The lower limit allowed.
        """
        assert is_number(lowerLimit), LOGGER.error("lowerLimit must be numbers")
        self.__lowerLimit = FLOAT_TYPE(lowerLimit) 
        if self.__upperLimit is not None:
            assert self.__lowerLimit<self.__upperLimit, LOGGER.error("lower limit must be smaller than the upper one")
            self.__rang = FLOAT_TYPE(self.__upperLimit-self.__lowerLimit)
    
    def set_upper_limit(self, upperLimit):
        """
        Set upper limit.
        
        :Parameters:
            #. upperLimit (number): The upper limit allowed.
        """
        assert is_number(upperLimit), LOGGER.error("upperLimit must be numbers")
        self.__upperLimit = FLOAT_TYPE(upperLimit)
        if self.__lowerLimit is not None:
            assert self.__lowerLimit<self.__upperLimit, LOGGER.error("lower limit must be smaller than the upper one")
            self.__rang = FLOAT_TYPE(self.__upperLimit-self.__lowerLimit)
        
    def generate(self):
        """Generate a random float number between lowerLimit and upperLimit."""
        return FLOAT_TYPE(self.__lowerLimit+generate_random_float()*self.__rang)

        
class BiasedRandomFloatGenerator(RandomFloatGenerator):
    """ 
    Generate biased random float number between a lower and an upper limit.
    To bias the generator at a certain number, a bias gaussian is added to the 
    weights scheme at the position of this particular number.
    
    .. image:: biasedFloatGenerator.png   
       :align: center 
       
    :Parameters:
        #. lowerLimit (number): The lower limit allowed.
        #. upperLimit (number): The upper limit allowed.
        #. weights (None, list, numpy.ndarray): The weights scheme. The length defines the number of bins and the edges.
           The length of weights array defines the resolution of the biased numbers generation.
           If None is given, ones array of length 10000 is automatically generated.
        #. biasRange(None, number): The bias gaussian range. 
           It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
           If None, it will be automatically set to (upperLimit-lowerLimit)/5
        #. biasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
           It must be smaller than half of biasRange.
           If None, it will be automatically set to biasRange/10
        #. biasHeight(number): The bias gaussian maximum intensity.
        #. unbiasRange(None, number): The bias gaussian range. 
           It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
           If None, it will be automatically set to biasRange.
        #. unbiasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
           It must be smaller than half of biasRange.
           If None, it will be automatically set to biasFWHM.
        #. unbiasHeight(number): The unbias gaussian maximum intensity.
           If None, it will be automatically set to biasHeight.
        #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
           It must be a positive number.
    """
    def __init__(self, lowerLimit, upperLimit, 
                       weights=None, 
                       biasRange=None, biasFWHM=None, biasHeight=1,
                       unbiasRange=None, unbiasFWHM=None, unbiasHeight=None, unbiasThreshold=1):
         # initialize random generator              
         super(BiasedRandomFloatGenerator, self).__init__(lowerLimit=lowerLimit, upperLimit=upperLimit)
         # set scheme 
         self.set_weights(weights)
         # set bias function
         self.set_bias(biasRange=biasRange, biasFWHM=biasFWHM, biasHeight=biasHeight)
         # set unbias function
         self.set_unbias(unbiasRange=unbiasRange, unbiasFWHM=unbiasFWHM, unbiasHeight=unbiasHeight, unbiasThreshold=unbiasThreshold)
    
    @property
    def originalWeights(self):
        """The original weights as initialized."""
        return self.__originalWeights
    
    @property
    def weights(self):
        """The current value weights vector."""
        weights = self.__scheme[1:]-self.__scheme[:-1]
        weights = list(weights)
        weights.insert(0,self.__scheme[0])
        return weights
        
    @property
    def scheme(self):
        """The numbers generation scheme."""
        return self.__scheme 
    
    @property
    def bins(self):
        """The number of bins that is equal to the length of weights vector."""
        return self.__bins
    
    @property
    def binWidth(self):
        """The bin width defining the resolution of the biased random number generation."""
        return self.__binWidth 
          
    @property
    def bias(self):
        """The bias step-function."""
        return self.__bias
    
    @property
    def biasGuassian(self):
        """The bias gaussian function."""
        return self.__biasGuassian
        
    @property
    def biasRange(self):
        """The bias gaussian extent range."""
        return self.__biasRange    
        
    @property
    def biasBins(self):
        """The bias gaussian number of bins."""
        return self.__biasBins
    
    @property
    def biasFWHM(self):
        """The bias gaussian Full Width at Half Maximum."""
        return self.__biasFWHM 

    @property
    def biasFWHMBins(self):
        """The bias gaussian Full Width at Half Maximum number of bins."""
        return self.__biasFWHMBins

    @property
    def unbias(self):
        """The unbias step-function."""
        return self.__unbias
    
    @property
    def unbiasGuassian(self):
        """The unbias gaussian function."""
        return self.__unbiasGuassian
        
    @property
    def unbiasRange(self):
        """The unbias gaussian extent range."""
        return self.__unbiasRange    
        
    @property
    def unbiasBins(self):
        """The unbias gaussian number of bins."""
        return self.__unbiasBins
    
    @property
    def unbiasFWHM(self):
        """The unbias gaussian Full Width at Half Maximum."""
        return self.__unbiasFWHM 

    @property
    def unbiasFWHMBins(self):
        """The unbias gaussian Full Width at Half Maximum number of bins."""
        return self.__unbiasFWHMBins

    def set_weights(self, weights=None):
        """
        Set generator's weights.
        
        :Parameters:
            #. weights (None, list, numpy.ndarray): The weights scheme. The length defines the number of bins and the edges.
               The length of weights array defines the resolution of the biased numbers generation.
               If None is given, ones array of length 10000 is automatically generated.
        """
        # set original weights
        if weights is None:
           self.__bins = 10000
           self.__originalWeights = np.ones(self.__bins)
        else:
            assert isinstance(weights, (list, set, tuple, np.ndarray)), LOGGER.error("weights must be a list of numbers")
            if isinstance(weights,  np.ndarray):
                assert len(weights.shape)==1, LOGGER.error("weights must be uni-dimensional")
            wgts = []
            assert len(weights)>=100, LOGGER.error("weights minimum length allowed is 100")
            for w in weights:
                assert is_number(w), LOGGER.error("weights items must be numbers")
                w = FLOAT_TYPE(w)
                assert w>=0, LOGGER.error("weights items must be positive")
                wgts.append(w)
            self.__originalWeights = np.array(wgts, dtype=FLOAT_TYPE)
            self.__bins = len(self.__originalWeights)
        # set bin width
        self.__binWidth     = FLOAT_TYPE(self.rang/self.__bins)
        self.__halfBinWidth = FLOAT_TYPE(self.__binWidth/2.)
        # set scheme    
        self.__scheme = np.cumsum( self.__originalWeights )
    
    def set_bias(self, biasRange, biasFWHM, biasHeight):
        """
        Set generator's bias gaussian function
        
        :Parameters:
            #. biasRange(None, number): The bias gaussian range. 
               It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
               If None, it will be automatically set to (upperLimit-lowerLimit)/5
            #. biasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
               It must be smaller than half of biasRange.
               If None, it will be automatically set to biasRange/10
            #. biasHeight(number): The bias gaussian maximum intensity.
        """
        # check biasRange
        if biasRange is None:
            biasRange = FLOAT_TYPE(self.rang/5.)
        else:
            assert is_number(biasRange), LOGGER.error("biasRange must be numbers")
            biasRange = FLOAT_TYPE(biasRange)
            assert biasRange>0, LOGGER.error("biasRange must be positive")
            assert biasRange<=self.rang/2., LOGGER.error("biasRange must be smaller than 50%% of limits range which is %s in this case"%str(self.rang))
        self.__biasRange = FLOAT_TYPE(biasRange)
        self.__biasBins  = INT_TYPE(self.bins*self.__biasRange/self.rang)
        # check biasFWHM
        if biasFWHM is None:
            biasFWHM = FLOAT_TYPE(self.__biasRange/10.)
        else:
            assert is_number(biasFWHM), LOGGER.error("biasFWHM must be numbers")
            biasFWHM = FLOAT_TYPE(biasFWHM)
            assert biasFWHM>=0, LOGGER.error("biasFWHM must be positive")
            assert biasFWHM<=self.__biasRange/2., LOGGER.error("biasFWHM must be smaller than 50%% of bias range which is %s in this case"%str(self.__biasRange))
        self.__biasFWHM     = FLOAT_TYPE(biasFWHM) 
        self.__biasFWHMBins = INT_TYPE(self.bins*self.__biasFWHM/self.rang)
        # check height
        assert is_number(biasHeight), LOGGER.error("biasHeight must be a number")
        self.__biasHeight = FLOAT_TYPE(biasHeight)
        assert self.__biasHeight>=0, LOGGER.error("biasHeight must be positive")
        # create bias step function
        b = self.__biasRange/self.__biasBins
        x = [-self.__biasRange/2.+idx*b for idx in range(self.__biasBins) ]
        self.__biasGuassian = gaussian(x, center=0, FWHM=self.__biasFWHM, normalize=False)
        self.__biasGuassian -= self.__biasGuassian[0]
        self.__biasGuassian /= np.max(self.__biasGuassian)
        self.__biasGuassian *= self.__biasHeight
        self.__bias = np.cumsum(self.__biasGuassian)
    
    def set_unbias(self, unbiasRange, unbiasFWHM, unbiasHeight, unbiasThreshold):
        """
        Set generator's unbias gaussian function
        
        :Parameters:
            #. unbiasRange(None, number): The bias gaussian range. 
               It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
               If None, it will be automatically set to biasRange.
            #. unbiasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
               It must be smaller than half of biasRange.
               If None, it will be automatically set to biasFWHM.
            #. unbiasHeight(number): The unbias gaussian maximum intensity.
               If None, it will be automatically set to biasHeight.
            #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
               It must be a positive number.
        """
        # check biasRange
        if unbiasRange is None:
            unbiasRange = self.__biasRange
        else:
            assert is_number(unbiasRange), LOGGER.error("unbiasRange must be numbers")
            unbiasRange = FLOAT_TYPE(unbiasRange)
            assert unbiasRange>0, LOGGER.error("unbiasRange must be positive")
            assert unbiasRange<=self.rang/2., LOGGER.error("unbiasRange must be smaller than 50%% of limits range which is %s in this case"%str(self.rang))
        self.__unbiasRange = FLOAT_TYPE(unbiasRange)
        self.__unbiasBins  = INT_TYPE(self.bins*self.__unbiasRange/self.rang)
        # check biasFWHM
        if unbiasFWHM is None:
            unbiasFWHM = self.__biasFWHM
        else:
            assert is_number(unbiasFWHM), LOGGER.error("unbiasFWHM must be numbers")
            unbiasFWHM = FLOAT_TYPE(unbiasFWHM)
            assert unbiasFWHM>=0, LOGGER.error("unbiasFWHM must be positive")
            assert unbiasFWHM<=self.__unbiasRange/2., LOGGER.error("unbiasFWHM must be smaller than 50%% of bias range which is %s in this case"%str(self.__biasRange))
        self.__unbiasFWHM     = FLOAT_TYPE(unbiasFWHM) 
        self.__unbiasFWHMBins = INT_TYPE(self.bins*self.__unbiasFWHM/self.rang)
        # check height
        if unbiasHeight is None:
            unbiasHeight = self.__biasHeight
        assert is_number(unbiasHeight), LOGGER.error("unbiasHeight must be a number")
        self.__unbiasHeight = FLOAT_TYPE(unbiasHeight)
        assert self.__unbiasHeight>=0, LOGGER.error("unbiasHeight must be bigger than 0")
        # check unbiasThreshold
        assert is_number(unbiasThreshold), LOGGER.error("unbiasThreshold must be a number")
        self.__unbiasThreshold = FLOAT_TYPE(unbiasThreshold)
        assert self.__unbiasThreshold>=0, LOGGER.error("unbiasThreshold must be positive")
        # create bias step function
        b = self.__unbiasRange/self.__unbiasBins
        x = [-self.__unbiasRange/2.+idx*b for idx in range(self.__unbiasBins) ]
        self.__unbiasGuassian = gaussian(x, center=0, FWHM=self.__unbiasFWHM, normalize=False)
        self.__unbiasGuassian -= self.__unbiasGuassian[0]
        self.__unbiasGuassian /= np.max(self.__unbiasGuassian)
        self.__unbiasGuassian *= -self.__unbiasHeight
        self.__unbias = np.cumsum(self.__unbiasGuassian)
         
    def bias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Bias the generator's scheme using the defined bias gaussian function at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to bias
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__biasHeight>0: return
        if check:
            assert is_integer(index), LOGGER.error("index must be an integer")
            index = INT_TYPE(index)
            assert index>=0, LOGGER.error("index must be bigger than 0")
            assert index<=self.__bins, LOGGER.error("index must be smaller than number of bins")
            if scaleFactor is not None:
                assert is_number(scaleFactor), LOGGER.error("scaleFactor must be a number")
                scaleFactor = FLOAT_TYPE(scaleFactor)
                assert scaleFactor>=0, LOGGER.error("scaleFactor must be bigger than 0")
        # get start indexes
        startIdx = index-int(self.__biasBins/2)
        if startIdx < 0:
            biasStartIdx = -startIdx
            startIdx = 0
            bias = np.cumsum(self.__biasGuassian[biasStartIdx:]).astype(FLOAT_TYPE)
        else:
            biasStartIdx = 0
            bias = self.__bias
        # scale bias
        if scaleFactor is None:
            scaledBias = bias
        else:
            scaledBias = bias*scaleFactor         
        # get end indexes
        endIdx = startIdx+self.__biasBins-biasStartIdx
        biasEndIdx = len(scaledBias)
        if endIdx > self.__bins-1:
            biasEndIdx -= endIdx-self.__bins
            endIdx = self.__bins
        # bias scheme
        self.__scheme[startIdx:endIdx] += scaledBias[0:biasEndIdx]
        self.__scheme[endIdx:] += scaledBias[biasEndIdx-1]
        
    def bias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Bias the generator's scheme using the defined bias gaussian function at the given number.
        
        :Parameters:
            #. position(number): The number to bias.
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_number(position), LOGGER.error("position must be a number")
            position = FLOAT_TYPE(position)
            assert position>=self.lowerLimit, LOGGER.error("position must be bigger than lowerLimit")
            assert position<=self.upperLimit, LOGGER.error("position must be smaller than upperLimit")
        index = INT_TYPE(self.__bins*(position-self.lowerLimit)/self.rang) 
        # bias scheme by index
        self.bias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)
    
    def unbias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme using the defined bias gaussian function at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to unbias
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__unbiasHeight>0: return
        if check:
            assert is_integer(index), LOGGER.error("index must be an integer")
            index = INT_TYPE(index)
            assert index>=0, LOGGER.error("index must be bigger than 0")
            assert index<=self.__bins, LOGGER.error("index must be smaller than number of bins")
            if scaleFactor is not None:
                assert is_number(scaleFactor), LOGGER.error("scaleFactor must be a number")
                scaleFactor = FLOAT_TYPE(scaleFactor)
                assert scaleFactor>=0, LOGGER.error("scaleFactor must be bigger than 0")
        # get start indexes
        startIdx = index-int(self.__unbiasBins/2)
        if startIdx < 0:
            biasStartIdx = -startIdx
            startIdx = 0
            unbias = self.__unbiasGuassian[biasStartIdx:]
        else:
            biasStartIdx = 0
            unbias = self.__unbiasGuassian
        # get end indexes
        endIdx = startIdx+self.__unbiasBins-biasStartIdx
        biasEndIdx = len(unbias)
        if endIdx > self.__bins-1:
            biasEndIdx -= endIdx-self.__bins
            endIdx = self.__bins
        # scale unbias
        if scaleFactor is None:
            scaledUnbias = unbias 
        else:
            scaledUnbias = unbias*scaleFactor
        # unbias weights
        weights = np.array(self.weights)
        weights[startIdx:endIdx] += scaledUnbias[0:biasEndIdx]
        # correct for negatives
        weights[np.where(weights<self.__unbiasThreshold)] = self.__unbiasThreshold
        # set unbiased scheme
        self.__scheme = np.cumsum(weights)
                                    
    def unbias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme using the defined bias gaussian function at the given number.
        
        :Parameters:
            #. position(number): The number to unbias.
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_number(position), LOGGER.error("position must be a number")
            position = FLOAT_TYPE(position)
            assert position>=self.lowerLimit, LOGGER.error("position must be bigger than lowerLimit")
            assert position<=self.upperLimit, LOGGER.error("position must be smaller than upperLimit")
        index = INT_TYPE(self.__bins*(position-self.lowerLimit)/self.rang) 
        # bias scheme by index
        self.unbias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)

    def generate(self):
        """Generate a random float number between the biased range lowerLimit and upperLimit."""
        # get position
        position = self.lowerLimit + self.__binWidth*np.searchsorted(self.__scheme, generate_random_float()*self.__scheme[-1]) + self.__halfBinWidth
        # find limits
        minLim = max(self.lowerLimit, position-self.__halfBinWidth)
        maxLim = min(self.upperLimit, position+self.__halfBinWidth)
        # generate number
        return minLim+generate_random_float()*(maxLim-minLim) + self.__halfBinWidth    
        

class RandomIntegerGenerator(object):
    """
    Generate random integer number between a lower and an upper limit.
    
    :Parameters:
        #. lowerLimit (number): The lower limit allowed.
        #. upperLimit (number): The upper limit allowed.
    """
    def __init__(self, lowerLimit, upperLimit):
         self.__lowerLimit = None
         self.__upperLimit = None
         self.set_lower_limit(lowerLimit)
         self.set_upper_limit(upperLimit)
    
    @property
    def lowerLimit(self):
        """The lower limit of the number generation."""
        return self.__lowerLimit
        
    @property
    def upperLimit(self):
        """The upper limit of the number generation."""
        return self.__upperLimit
        
    @property
    def rang(self):
        """The range defined as upperLimit-lowerLimit"""
        return self.__rang
            
    def set_lower_limit(self, lowerLimit):    
        """
        Set lower limit.
        
        :Parameters:
            #. lowerLimit (number): The lower limit allowed.
        """
        assert is_integer(lowerLimit), LOGGER.error("lowerLimit must be numbers")
        self.__lowerLimit = INT_TYPE(lowerLimit) 
        if self.__upperLimit is not None:
            assert self.__lowerLimit<self.__upperLimit, LOGGER.error("lower limit must be smaller than the upper one")
            self.__rang = self.__upperLimit-self.__lowerLimit+1
    
    def set_upper_limit(self, upperLimit):
        """
        Set upper limit.
        
        :Parameters:
            #. upperLimit (number): The upper limit allowed.
        """
        assert is_integer(upperLimit), LOGGER.error("upperLimit must be numbers")
        self.__upperLimit = INT_TYPE(upperLimit) 
        if self.__lowerLimit is not None:
            assert self.__lowerLimit<self.__upperLimit, LOGGER.error("lower limit must be smaller than the upper one")
            self.__rang = self.__upperLimit-self.__lowerLimit+1
        
    def generate(self):
        """Generate a random integer number between lowerLimit and upperLimit."""
        return generate_random_integer(self.__lowerLimit, self.__upperLimit)

        
class BiasedRandomIntegerGenerator(RandomIntegerGenerator):
    """ 
    Generate biased random integer number between a lower and an upper limit.
    To bias the generator at a certain number, a bias height is added to the 
    weights scheme at the position of this particular number.
    
    .. image:: biasedIntegerGenerator.png   
       :align: center 
       
    :Parameters:
        #. lowerLimit (integer): The lower limit allowed.
        #. upperLimit (integer): The upper limit allowed.
        #. weights (None, list, numpy.ndarray): The weights scheme. The length must be equal to the range between lowerLimit and upperLimit.
           If None is given, ones array of length upperLimit-lowerLimit+1 is automatically generated.
        #. biasHeight(number): The weight bias intensity.
        #. unbiasHeight(None, number): The weight unbias intensity.
           If None, it will be automatically set to biasHeight.
        #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
           It must be a positive number.
    """
    def __init__(self, lowerLimit, upperLimit, 
                       weights=None, 
                       biasHeight=1, unbiasHeight=None, unbiasThreshold=1):
        # initialize random generator              
        super(BiasedRandomIntegerGenerator, self).__init__(lowerLimit=lowerLimit, upperLimit=upperLimit)
        # set weights
        self.set_weights(weights=weights)
        # set bias height
        self.set_bias_height(biasHeight=biasHeight)
        # set bias height
        self.set_unbias_height(unbiasHeight=unbiasHeight)
        # set bias height
        self.set_unbias_threshold(unbiasThreshold=unbiasThreshold)
    
    @property
    def originalWeights(self):
        """The original weights as initialized."""
        return self.__originalWeights
    
    @property
    def weights(self):
        """The current value weights vector."""
        weights = self.__scheme[1:]-self.__scheme[:-1]
        weights = list(weights)
        weights.insert(0,self.__scheme[0])
        return weights
        
    @property
    def scheme(self):
        """The numbers generation scheme."""
        return self.__scheme 
    
    @property
    def bins(self):
        """The number of bins that is equal to the length of weights vector."""
        return self.__bins
        
    def set_weights(self, weights):
        """
        Set the generator integer numbers weights.
        
        #. weights (None, list, numpy.ndarray): The weights scheme. The length must be equal to the range between lowerLimit and upperLimit.
           If None is given, ones array of length upperLimit-lowerLimit+1 is automatically generated.
        """
        if weights is None:
            self.__originalWeights = np.ones(self.upperLimit-self.lowerLimit+1)
        else:
            assert isinstance(weights, (list, set, tuple, np.ndarray)), LOGGER.error("weights must be a list of numbers")
            if isinstance(weights,  np.ndarray):
                assert len(weights.shape)==1, LOGGER.error("weights must be uni-dimensional")
            wgts = []
            assert len(weights)==self.upperLimit-self.lowerLimit+1, LOGGER.error("weights length must be exactly equal to 'upperLimit-lowerLimit+1' which is %i"%self.upperLimit-self.lowerLimit+1)
            for w in weights:
                assert is_number(w), LOGGER.error("weights items must be numbers")
                w = FLOAT_TYPE(w)
                assert w>=0, LOGGER.error("weights items must be positive")
                wgts.append(w)
            self.__originalWeights = np.array(wgts, dtype=FLOAT_TYPE)
        # set bins
        self.__bins = len( self.__originalWeights )
        # set scheme    
        self.__scheme = np.cumsum( self.__originalWeights )
        
    def set_bias_height(self, biasHeight):
        """
        Set weight bias intensity.
        
        :Parameters:
            #. biasHeight(number): The weight bias intensity.
        """
        assert is_number(biasHeight), LOGGER.error("biasHeight must be a number")
        self.__biasHeight = FLOAT_TYPE(biasHeight)
        assert self.__biasHeight>0, LOGGER.error("biasHeight must be bigger than 0")
        
    def set_unbias_height(self, unbiasHeight):
        """
        Set weight unbias intensity.
        
        :Parameters:
            #. unbiasHeight(None, number): The weight unbias intensity.
               If None, it will be automatically set to biasHeight.
        """
        if unbiasHeight is None:
            unbiasHeight = self.__biasHeight
        assert is_number(unbiasHeight), LOGGER.error("unbiasHeight must be a number")
        self.__unbiasHeight = FLOAT_TYPE(unbiasHeight)
        assert self.__unbiasHeight>=0, LOGGER.error("unbiasHeight must be positive")
        
    def set_unbias_threshold(self, unbiasThreshold):
        """
        Set weight unbias threshold.
        
        :Parameters:
            #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
               It must be a positive number.
        """
        assert is_number(unbiasThreshold), LOGGER.error("unbiasThreshold must be a number")
        self.__unbiasThreshold = FLOAT_TYPE(unbiasThreshold)
        assert self.__unbiasThreshold>=0, LOGGER.error("unbiasThreshold must be positive")

    def bias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Bias the generator's scheme at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to bias
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__biasHeight>0: return
        if check:
            assert is_integer(index), LOGGER.error("index must be an integer")
            index = INT_TYPE(index)
            assert index>=0, LOGGER.error("index must be bigger than 0")
            assert index<=self.__bins, LOGGER.error("index must be smaller than number of bins")
            if scaleFactor is not None:
                assert is_number(scaleFactor), LOGGER.error("scaleFactor must be a number")
                scaleFactor = FLOAT_TYPE(scaleFactor)
                assert scaleFactor>=0, LOGGER.error("scaleFactor must be bigger than 0")
        # scale bias
        if scaleFactor is None:
            scaledBias = self.__biasHeight
        else:
            scaledBias = self.__biasHeight*scaleFactor         
        # bias scheme
        self.__scheme[index:] += scaledBias
          
    def bias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Bias the generator's scheme at the given number.
        
        :Parameters:
            #. position(number): The number to bias.
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_integer(position), LOGGER.error("position must be an integer")
            position = INT_TYPE(position)
            assert position>=self.lowerLimit, LOGGER.error("position must be bigger than lowerLimit")
            assert position<=self.upperLimit, LOGGER.error("position must be smaller than upperLimit")
        index = position-self.lowerLimit
        # bias scheme by index
        self.bias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)

    def unbias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to unbias
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__unbiasHeight>0: return
        if check:
            assert is_integer(index), LOGGER.error("index must be an integer")
            index = INT_TYPE(index)
            assert index>=0, LOGGER.error("index must be bigger than 0")
            assert index<=self.__bins, LOGGER.error("index must be smaller than number of bins")
            if scaleFactor is not None:
                assert is_number(scaleFactor), LOGGER.error("scaleFactor must be a number")
                scaleFactor = FLOAT_TYPE(scaleFactor)
                assert scaleFactor>=0, LOGGER.error("scaleFactor must be bigger than 0")
        # scale unbias
        if scaleFactor is None:
            scaledUnbias = self.__unbiasHeight 
        else:
            scaledUnbias = self.__unbiasHeight*scaleFactor
        # check threshold
        if index == 0:
            scaledUnbias = max(scaledUnbias, self.__scheme[index]-self.__unbiasThreshold)   
        elif self.__scheme[index]-scaledUnbias < self.__scheme[index-1]+self.__unbiasThreshold:
            scaledUnbias = self.__scheme[index]-self.__scheme[index-1]-self.__unbiasThreshold
        # unbias scheme
        self.__scheme[index:] -= scaledUnbias
                                   
    def unbias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme using the defined bias gaussian function at the given number.
        
        :Parameters:
            #. position(number): The number to unbias.
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_integer(position), LOGGER.error("position must be an integer")
            position = INT_TYPE(position)
            assert position>=self.lowerLimit, LOGGER.error("position must be bigger than lowerLimit")
            assert position<=self.upperLimit, LOGGER.error("position must be smaller than upperLimit")
        index = position-self.lowerLimit
        # unbias scheme by index
        self.unbias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)
        
    def generate(self):
        """Generate a random intger number between the biased range lowerLimit and upperLimit."""
        index = INT_TYPE( np.searchsorted(self.__scheme, generate_random_float()*self.__scheme[-1]) )
        return self.lowerLimit + index
       

