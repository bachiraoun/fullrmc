"""
Constraint contains parent classes for all constraints.
A Constraint is used to set certain rules to evolve the configuration.
Therefore it has become possible to fully customize and set any possibly imaginable rule.

.. inheritance-diagram:: fullrmc.Core.Constraint
    :parts: 1
"""

# standard libraries imports
import inspect
from random import random as randfloat

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path

   
class Constraint(object):
    """
    A constraint is used to direct the evolution of the configuration towards the desired and most meaningful one.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint fullrmc engine.
    """
    def __init__(self, engine):
        self.set_used(True)
        self.set_engine(engine)
        # initialize data
        self.__initialize_constraint()
    
    def __initialize_constraint(self):
        # initialize flags
        self.__state           = None
        self.__tried           = 0
        self.__accepted        = 0
        self.__varianceSquared = 1
        # initialize data
        self.__originalData              = None
        self.__data                      = None
        self.__activeAtomsDataBeforeMove = None
        self.__activeAtomsDataAfterMove  = None
        # initialize standard error
        self.__standardError           = None
        self.__afterMoveStandardError  = None
    
    @property
    def engine(self):
        """ Get the engine fullrmc instance."""
        return self.__engine 
        
    @property
    def state(self):
        """ Get constraint's state."""
        return self.__state
        
    @property
    def tried(self):
        """ Get constraint's number of tried moves."""
        return self.__tried
    
    @property
    def accepted(self):
        """ Get constraint's number of accepted moves."""
        return self.__accepted
    
    @property
    def used(self):
        """ Get whether this constraint is used in the engine run time or set inactive."""
        return self.__used
    
    @property
    def varianceSquared(self):
        """ Get constraint's varianceSquared used in the engine run time to calculate the total chi square."""
        return self.__varianceSquared
    
    @property
    def standardError(self):
        """ Get constraint's current standard error."""
        return self.__standardError
    
    @property
    def originalData(self):
        """ Get constraint's original calculated data upon initialization."""
        return self.__originalData
        
    @property
    def data(self):
        """ Get constraint's current calculated data."""
        return self.__data
    
    @property
    def activeAtomsDataBeforeMove(self):
        """ Get constraint's current calculated data before last move."""
        return self.__activeAtomsDataBeforeMove
    
    @property
    def activeAtomsDataAfterMove(self):
        """ Get constraint's current calculated data after last move."""
        return self.__activeAtomsDataAfterMove
    
    @property
    def afterMoveStandardError(self):
        """ Get constraint's current calculated StandardError after last move."""
        return self.__afterMoveStandardError
        
    def _set_original_data(self, data):
        """ Used only by the engine to set constraint's data as initialized for the first time."""
        self.__originalData = data
        
    def listen(self, message, argument=None):
        """   
        Listen's to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. arguments (object): Any type of argument to pass to the listeners.
        """
        pass
        
    def _runtime_initialize(self):
        """   
        This is called once everytime engine.run method is executed.
        It is meant to be used as a final setup call for all constraints.
        """
        pass
    
    def _runtime_on_step(self):
        """   
        This is called at everytime engine.run method main loop step.
        """
        pass
        
    def set_variance_squared(self, value):
        """
        Sets constraint's variance squared that is used in the computation of the total engine chi square.
        
        :Parameters:
            #. value (number): Any positive non zero number.
        """
        assert is_number(value), LOGGER.error("accepted value must be convertible to a number")
        value = float(value)
        assert value>0 , LOGGER.error("Variance must be positive non zero number.")
        self.__varianceSquared = value
        
    def set_used(self, value):
        """
        Sets used flag.
        
        :Parameters:
            #. value (boolean): True to use this constraint in engine run time.
        """
        assert isinstance(value, bool), LOGGER.error("value must be boolean")
        self.__used = value
    
    def set_state(self, value):
        """
        Sets constraint's state. 
        When constraint's state and engine's state don't match, constraint's data must be recalculated.
        
        :Parameters:
            #. value (object): constraint state value
        """
        self.__state = value
    
    def set_tried(self, value):
        """
        Sets constraint's engine tried moves.
        
        :Parameters:
            #. value (integer): constraint tried moves value
        """
        try:
            value = float(value)
        except:
            raise Exception(LOGGER.error("tried value must be convertible to a number"))
        assert is_integer(value), LOGGER.error("tried value must be integer")
        assert value>=0, LOGGER.error("tried value must be positive")
        self.__tried = int(value)
    
    def increment_tried(self):
        """ Increment engine tried moves. """
        self.__tried += 1
    
    def set_accepted(self, value):
        """
        Sets constraint's engine accepted moves.
        
        :Parameters:
            #. value (integer): constraint accepted moves value
        """
        try:
            value = float(value)
        except:
            raise Exception(LOGGER.error("accepted value must be convertible to a number"))
        assert is_integer(value), LOGGER.error("accepted value must be integer")
        assert value>=0, LOGGER.error("accepted value must be positive")
        assert value<=self.__tried, LOGGER.error("accepted value can't be bigger than number of tried moves")
        self.__accepted = int(value)
    
    def increment_accepted(self):
        """ Increment engine accepted moves. """
        self.__accepted += 1
        
    def set_standard_error(self, value):
        """
        Sets constraint's standardError value.
        
        :Parameters:
            #. value (number): standardError value
        """
        self.__standardError = value
        
    def set_data(self, value):
        """
        Sets constraint's data value
        
        :Parameters:
            #. value (number): standardError value.
        """
        self.__data = value
    
    def set_active_atoms_data_before_move(self, value):
        """
        Sets constraint's before move happens active atoms data value.
        
        :Parameters:
            #. value (number): data value
        """
        self.__activeAtomsDataBeforeMove = value
    
    def set_active_atoms_data_after_move(self, value):
        """
        Sets constraint's after move happens active atoms data value.
        
        :Parameters:
            #. value (number): data value
        """
        self.__activeAtomsDataAfterMove = value
    
    def set_after_move_standard_error(self, value):
        """
        Sets constraint's standardError value after move happens.
        
        :Parameters:
            #. value (number): standardError value.
        """
        self.__afterMoveStandardError = value
        
    def reset_constraint(self, reinitialize=True, flags=False, data=False):
        """ 
        Resets constraint.
        
        :Parameters:
            #. reinitialize (boolean): If set to True, it will override the rest of the flags 
               and will completely reinitialize the constraint.
            #. flags (boolean): Reset the state, tried and accepted flags of the constraint.
            #. data (boolean): Reset the constraints computed data.
        """
        # reinitialize constraint
        if reinitialize:
            flags = False
            data  = False
            self.__initialize_constraint()
        # initialize flags
        if flags:
            self.__state        = None
            self.__tried        = 0
            self.__accepted     = 0
        # initialize data
        if data:
            self.__originalData              = None
            self.__data                      = None
            self.__activeAtomsDataBeforeMove = None
            self.__activeAtomsDataAfterMove  = None
        # initialize standard error
        self.__standardError           = None
        self.__afterMoveStandardError  = None
   
    def set_engine(self, engine):
        """
        Sets the constraints fullrmc engine instance.
        'engine changed' message will be broadcasted automatically to the constraint's listener listen method.
        
        :Parameters:
            #. engine (None, fullrmc.Engine): The constraint fullrmc engine.
        """
        if engine is not None:
            from fullrmc.Engine import Engine
            assert isinstance(engine, Engine),LOGGER.error("engine must be None or fullrmc Engine instance")
        self.__engine = engine
        # reset flags
        self.reset_constraint(reinitialize=False, flags=True, data=False)
        #self.set_state(None)
        #self.set_tried(0)
        #self.set_accepted(0)
    
    def compute_and_set_standard_error(self):
        """ Computes and sets the constraint's standardError by calling compute_standard_error and passing the constraint's data."""
        self.set_standard_error(self.compute_standard_error(data = self.data))
        
    def get_constraint_value(self):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
    
    def get_constraint_original_value(self):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def compute_standard_error(self):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def compute_data(self, indexes):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def compute_before_move(self, indexes):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def compute_after_move(self, indexes, movedBoxCoordinates):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def accept_move(self, indexes):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
        
    def reject_move(self, indexes):
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
    
    def plot(self, *args, **kwargs):
        LOGGER.warn("%s plot method is not implemented"%(self.__class__.__name__))
    
    
class ExperimentalConstraint(Constraint):
    """
    An ExperimentalConstraint is any constraint related to experimental data.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        #. dataWeights (None, numpy.ndarray): A weights array of the same number of points of experimentalData used in the constraint's standard error computation.
           Therefore particular fitting emphasis can be put on different data points that might be considered as more or less
           important in order to get a reasonable and plausible modal.\n
           If None is given, all data points are considered of the same importance in the computation of the constraint's standard error.\n
           If numpy.ndarray is given, all weights must be positive and all zeros weighted data points won't contribute to the 
           total constraint's standard error. At least a single weight point is required to be non-zeros and the weights 
           array will be automatically scaled upon setting such as the the sum of all the weights is equal to the number of data points.       
        #. scaleFactor (number): A scaling constant multiplying the computed data to normalize to the experimental ones.
        #. adjustScaleFactor (list, tuple): Used to adjust fit or guess the best scale factor during EMC runtime. 
           It must be a list of exactly three entries.\n
           1. The frequency in number of accepted moves of finding the best scale factor. 
              If 0 frequency is given, it means that the scale factor is fixed.
           2. The minimum allowed scale factor value.
           3. The maximum allowed scale factor value.
    
    **NB**: If adjustScaleFactor first item (frequency) is 0, the scale factor will remain 
    untouched and the limits minimum and maximum won't be checked.
           
    """
    def __init__(self, engine, experimentalData, dataWeights=None, scaleFactor=1.0, adjustScaleFactor=(0, 0.8, 1.2) ):
        # initialize constraint
        super(ExperimentalConstraint, self).__init__(engine=engine)
        # set the constraint's experimental data
        self.__dataWeights      = None
        self.__experimentalData = None
        self.set_experimental_data(experimentalData)
        # set scale factor
        self.set_scale_factor(scaleFactor)
        # set adjust scale factor
        self.set_adjust_scale_factor(adjustScaleFactor)
        # set data weights
        self.set_data_weights(dataWeights)
    
    def _set_fitted_scale_factor_value(self, scaleFactor):
        """
        This method is a scaleFactor value without any validity checking.
        Meant to be used internally only.
        """
        if self.__scaleFactor != scaleFactor:
            LOGGER.info("Experimental constraint '%s' scale factor updated from %.6f to %.6f" %(self.__class__.__name__, self.__scaleFactor, scaleFactor))
            self.__scaleFactor = scaleFactor
        
    
    @property
    def experimentalData(self):
        """ Gets the experimental data of the constraint. """
        return self.__experimentalData
    
    @property
    def dataWeights(self):
        """ Get experimental data points weight"""
        return self.__dataWeights
        
    @property
    def scaleFactor(self):
        """ Get the scaleFactor. """
        return self.__scaleFactor
    
    @property
    def adjustScaleFactor(self):
        return (self.__adjustScaleFactorFrequency, self.__adjustScaleFactorMinimum, self.__adjustScaleFactorMaximum)

    @property
    def adjustScaleFactorFrequency(self):
        """ Get the scaleFactor adjustment frequency. """
        return self.__adjustScaleFactorFrequency
    
    @property
    def adjustScaleFactorMinimum(self):
        """ Get the scaleFactor adjustment minimum number allowed. """
        return self.__adjustScaleFactorMinimum
    
    @property
    def adjustScaleFactorMaximum(self):
        """ Get the scaleFactor adjustment maximum number allowed. """
        return self.__adjustScaleFactorMaximum
    
    def set_scale_factor(self, scaleFactor):
        """
        Sets the scale factor.
        
        :Parameters:
             #. scaleFactor (number): A normalization scale factor used to normalize the computed data to the experimental ones.
        """
        assert is_number(scaleFactor), LOGGER.error("scaleFactor must be a number")
        self.__scaleFactor = FLOAT_TYPE(scaleFactor)
        ## reset constraint
        self.reset_constraint()
    
    def set_adjust_scale_factor(self, adjustScaleFactor):
        """
        Sets adjust scale factor.
        
        :Parameters:
             #. adjustScaleFactor (list, tuple): Used to adjust fit or guess the best scale factor during EMC runtime. 
                It must be a list of exactly three entries.\n
                1. The frequency in number of accepted moves of finding the best scale factor. 
                   If 0 frequency is given, it means that the scale factor is fixed.
                2. The minimum allowed scale factor value.
                3. The maximum allowed scale factor value.
        """
        assert isinstance(adjustScaleFactor, (list, tuple)), LOGGER.error('adjustScaleFactor must be a list.')
        assert len(adjustScaleFactor) == 3, LOGGER.error('adjustScaleFactor must be a list of exactly three items.')
        freq  = adjustScaleFactor[0]
        minSF = adjustScaleFactor[1]
        maxSF = adjustScaleFactor[2]
        assert is_integer(freq), LOGGER.error("adjustScaleFactor first item (frequency) must be an integer.")
        freq = INT_TYPE(freq)
        assert freq>=0, LOGGER.error("adjustScaleFactor first (frequency) item must be bigger or equal to 0.")
        assert is_number(minSF), LOGGER.error("adjustScaleFactor second item (minimum) must be a number.")
        minSF = FLOAT_TYPE(minSF)
        assert is_number(maxSF), LOGGER.error("adjustScaleFactor third item (maximum) must be a number.")
        maxSF = FLOAT_TYPE(maxSF)
        assert minSF<=maxSF, LOGGER.error("adjustScaleFactor second item (minimum) must be smaller or equal to third second item (maximum).")
        # set values
        self.__adjustScaleFactorFrequency = freq
        self.__adjustScaleFactorMinimum   = minSF
        self.__adjustScaleFactorMaximum   = maxSF
        # reset constraint
        self.reset_constraint()
        
    def set_experimental_data(self, experimentalData):
        """
        Sets the constraint's experimental data.
        
        :Parameters:
            #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        """
        if isinstance(experimentalData, basestring):
            try:
                experimentalData = np.loadtxt(str(experimentalData), dtype=FLOAT_TYPE)
            except Exception as e:
                raise Exception(LOGGER.error("unable to load experimentalData path '%s' (%s)"%(experimentalData, e)))
        assert isinstance(experimentalData, np.ndarray), LOGGER.error("experimentalData must be a numpy.ndarray or string path to load data using numpy.loadtxt.")
        # check data format
        valid, message = self.check_experimental_data(experimentalData)
        # set experimental data
        if valid:
            self.__experimentalData = experimentalData
        else:
            raise Exception( LOGGER.error("%s"%message) )
    
    def set_data_weights(self, dataWeights):
        """
        Set experimental data points weight.
        
        :Parameters: 
        
        #. dataWeights (None, numpy.ndarray): A weights array of the same number of points of experimentalData used in the constraint's standard error computation.
           Therefore particular fitting emphasis can be put on different data points that might be considered as more or less
           important in order to get a reasonable and plausible modal.\n
           If None is given, all data points are considered of the same importance in the computation of the constraint's standard error.\n
           If numpy.ndarray is given, all weights must be positive and all zeros weighted data points won't contribute to the 
           total constraint's standard error. At least a single weight point is required to be non-zeros and the weights 
           array will be automatically scaled upon setting such as the the sum of all the weights is equal to the number of data points.       
        """
        if dataWeights is not None:
            assert isinstance(dataWeights, (list, tuple, np.ndarray)), LOGGER.error("dataWeights must be None or a numpy array of weights")
            try:
                dataWeights = np.array(dataWeights, dtype=FLOAT_TYPE)
            except Exception as e:
                raise Exception(LOGGER.error("unable to cast dataWeights as a numpy array (%s)"%(e)))
            assert len(dataWeights.shape) == 1, LOGGER.error("dataWeights must be a vector")
            assert len(dataWeights) == self.__experimentalData.shape[0], LOGGER.error("dataWeights must be a of the same length as experimental data")
            assert np.min(dataWeights) >=0, LOGGER.error("dataWeights negative values are not allowed")
            assert np.sum(dataWeights), LOGGER.error("dataWeights must be a non-zero array")
            dataWeights /= FLOAT_TYPE( np.sum(dataWeights) )
            dataWeights *= FLOAT_TYPE( len(dataWeights) )                      
        self.__dataWeights = dataWeights
        
    def check_experimental_data(self, experimentalData):
        """
        Checks the constraint's experimental data
        This method must be overloaded in all ExperimentalConstraint sub-classes.
        
        :Parameters:
            #. experimentalData (numpy.ndarray): the experimental data numpy.ndarray.
        """
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))
    
    def fit_scale_factor(self, experimentalData, modelData, dataWeights):
        """
        The best scale factor value is computed by minimizing :math:`E=sM`.\n
        
        Where:
            #. :math:`E` is the experimental data.
            #. :math:`s` is the scale factor.
            #. :math:`M` is the model constraint data.

        :Parameters:
            #. experimentalData (numpy.ndarray): the experimental data.
            #. modelData (numpy.ndarray): the constraint modal data.
            #. dataWeights (None, numpy.ndarray): the data points weights to compute the scale factor.
               If None, all data points will be considered as having the same weight.
            
        :Returns:
            #. scaleFactor (number): The new scale factor fit value.
        
        **NB**: This method won't update the internal scale factor value of the constraint.
        It always computes the best scale factor given some experimental and model data
        """
        if dataWeights is None:
            SF = FLOAT_TYPE( np.sum(modelData*experimentalData)/np.sum(modelData**2) )
        else:
            SF = FLOAT_TYPE( np.sum(dataWeights*modelData*experimentalData)/np.sum(modelData**2) )
        SF = max(SF, self.__adjustScaleFactorMinimum)
        SF = min(SF, self.__adjustScaleFactorMaximum)
        return SF
        
    def get_adjusted_scale_factor(self, experimentalData, modelData, dataWeights):
        """
        Checks if scale factor should be updated according to the given scale factor frequency
        and engine's accepted steps. If adjustment is due, a new scale factor will be computed  
        using fit_scale_factor method, otherwise the the constraint's scale factor will be returned.

        :Parameters:
            #. experimentalData (numpy.ndarray): the experimental data.
            #. modelData (numpy.ndarray): the constraint modal data.
            #. dataWeights (None, numpy.ndarray): the data points weights to compute the scale factor.
               If None, all data points will be considered as having the same weight.
        
        :Returns:
            #. scaleFactor (number): The constraint's scale factor or the new scale factor fit value.
            
        **NB**: This method WILL NOT UPDATE the internal scale factor value of the constraint. 
        """
        SF = self.__scaleFactor
        # check to update scaleFactor
        if self.__adjustScaleFactorFrequency:
            if not self.engine.accepted%self.adjustScaleFactorFrequency:
                SF = self.fit_scale_factor(experimentalData, modelData, dataWeights)                 
        return SF
    
    
    def compute_standard_error(self, experimentalData, modelData):
        """ 
        Compute the squared deviation between modal computed data and the experimental ones. 
        
        .. math::
            SD = \\sum \\limits_{i}^{N} W_{i}(Y(X_{i})-F(X_{i}))^{2}
         
        Where:\n
        :math:`N` is the total number of experimental data points. \n
        :math:`W_{i}` is the data point weight. It becomes equivalent to 1 when dataWeights is set to None. \n
        :math:`Y(X_{i})` is the experimental data point :math:`X_{i}`. \n
        :math:`F(X_{i})` is the computed from the model data  :math:`X_{i}`. \n

        :Parameters:
            #. experimentalData (numpy.ndarray): the experimental data.
            #. modelData (numpy.ndarray): The data to compare with the experimental one and compute the squared deviation.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
        """
        # compute difference
        diff = experimentalData-modelData
        # return squared deviation
        if self.__dataWeights is None:
            return np.add.reduce((diff)**2)
        else:
            return np.add.reduce(self.__dataWeights*(diff)**2)
        
        
class SingularConstraint(Constraint):
    """ A singular constraint is a constraint that doesn't allow multiple instances in the same engine."""
    
    @property
    def is_singular(self):
        """
        Get whether only one instance of this constraint type is present in the engine.
        True for only itself found, False for other instance of the same __class__.__name__
        """
        for c in self.engine.constraints:
            if c is self: 
                continue
            if c.__class__.__name__ == self.__class__.__name__:
                return False
        return True
            
    def assert_singular(self):
        """
        Checks whether only one instance of this constraint type is present in the engine.
        Raises Exception if multiple instances are present.
        """
        assert self.is_singular, LOGGER.error("Only one instance of constraint '%s' is allowed in the same engine"%self.__class__.__name__)
        
        
class RigidConstraint(Constraint):
    """
    A rigid constraint is a constraint that doesn't count into the total standardError of the Engine.
    But it's internal standardError must monotonously decrease or remain the same from one engine step to another.
    If standardError of an RigidConstraint increases the step will be rejected even before engine's new standardError get computed.
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint fullrmc engine.
        #. rejectProbability (Number): Rejecting probability of all steps where standardError increases. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
    """
    def __init__(self, engine, rejectProbability):
        # initialize constraint
        super(RigidConstraint, self).__init__(engine=engine)
        # set probability
        self.set_reject_probability(rejectProbability)
        
    @property
    def rejectProbability(self):
        """ Get rejection probability. """
        return self.__rejectProbability
        
    def set_reject_probability(self, rejectProbability):
        """
        Set the rejection probability.
        
        :Parameters:
            #. rejectProbability (Number): rejecting probability of all steps where standardError increases. 
               It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
               and 0 means accepting all steps regardless whether standardError increases or not.
        """
        assert is_number(rejectProbability), LOGGER.error("rejectProbability must be a number")
        rejectProbability = FLOAT_TYPE(rejectProbability)
        assert rejectProbability>=0 and rejectProbability<=1, LOGGER.error("rejectProbability must be between 0 and 1")
        self.__rejectProbability = rejectProbability
    
    def should_step_get_rejected(self, standardError):
        """
        Given a standardError, return whether to keep or reject new standardError according to the constraint rejectProbability.
        
        :Parameters:
            #. standardError (number): The standardError to compare with the Constraint standardError
        
        :Return:
            #. result (boolean): True to reject step, False to accept
        """
        if standardError<=self.standardError:
            return False
        return randfloat() < self.__rejectProbability
        
    def should_step_get_accepted(self, standardError):
        """
        Given a standardError, return whether to keep or reject new standardError according to the constraint rejectProbability.
        
        :Parameters:
            #. standardError (number): The standardError to compare with the Constraint standardError
        
        :Return:
            #. result (boolean): True to accept step, False to reject
        """
        return not self.should_step_get_reject(standardError)


class QuasiRigidConstraint(RigidConstraint):
    """
    A quasi-rigid constraint is a another rigid constraint but it becomes free above a certain threshold
    ratio of satisfied data. Every quasi-rigid constraint has its own definition of maximum standard 
    error. The ratio is computed as between current standard error and maximum standard error.
    
     .. math::
        
        ratio = 1-\\frac{current\ standard\ error}{maximum\ standard\ error} 
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint fullrmc engine.
        #. rejectProbability (Number): Rejecting probability of all steps where standardError increases
           only before threshold ratio is reached. 
           It must be between 0 and 1 where 1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standardError increases or not.
        #. thresholdRatio(Number): The threshold of satisfied data, above which the constraint become free.
           It must be between 0 and 1 where 1 means all data must be satisfied and therefore the constraint
           behave like a RigidConstraint and 0 means none of the data must be satisfied and therefore the
           constraint becomes always free and useless.
    """    
    def __init__(self, engine, rejectProbability, thresholdRatio):
        # initialize constraint
        super(QuasiRigidConstraint, self).__init__(engine=engine, rejectProbability=rejectProbability)
        # set probability
        self.set_threshold_ratio(thresholdRatio)
        # initialize maximum standard error
        self.__maximumStandardError = None
        
    def _set_maximum_standard_error(self, maximumStandardError):
        """ Sets the maximum standard error. Use carefully, it's not meant to be used externally.
        maximum squared deviation is what is used to compute the ratio and compare to thresholdRatio.
        """
        if (maximumStandardError is not None) and maximumStandardError:
            assert is_number(maximumStandardError), LOGGER.error("maximumStandardError must be a number.")
            maximumStandardError = FLOAT_TYPE(maximumStandardError)
            assert maximumStandardError>0, LOGGER.error("maximumStandardError must be a positive.")
        self.__maximumStandardError = maximumStandardError
        
    @property
    def thresholdRatio(self):
        """ Get threshold ratio. """
        return self.__thresholdRatio
    
    @property
    def currentRatio(self):
        return 1-(self.standardError/self.__maximumStandardError)        
        
    def set_threshold_ratio(self, thresholdRatio):
        """
        Set the rejection probability function.
        
        :Parameters:
            #. thresholdRatio(Number): The threshold of satisfied data, above which the constraint become free.
               It must be between 0 and 1 where 1 means all data must be satisfied and therefore the constraint
               behave like a RigidConstraint and 0 means none of the data must be satisfied and therefore the
               constraint becomes always free and useless.
        """
        assert is_number(thresholdRatio), LOGGER.error("thresholdRatio must be a number")
        thresholdRatio = FLOAT_TYPE(thresholdRatio)
        assert thresholdRatio>=0 and thresholdRatio<=1, LOGGER.error("thresholdRatio must be between 0 and 1")
        self.__thresholdRatio = thresholdRatio
        
    def should_step_get_rejected(self, standardError):
        """
        Given a standardError, return whether to keep or reject new standardError according to the constraint rejectProbability function.
        
        :Parameters:
            #. standardError (number): The standardError to compare with the Constraint standardError
        
        :Return:
            #. result (boolean): True to reject step, False to accept
        """
        previousRatio = 1-(self.standardError/self.__maximumStandardError)
        currentRatio  = 1-(standardError/self.__maximumStandardError)
        if currentRatio>=self.__thresholdRatio: # must be accepted
            return False 
        elif previousRatio>=self.__thresholdRatio: # it must be rejected
            return randfloat() < self.rejectProbability
        elif standardError<=self.standardError: # must be accepted
            return False
        else: # must be rejected
            return randfloat() < self.rejectProbability
            
            
    
            