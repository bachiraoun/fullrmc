"""
Constraint contains parent classes for all constraints.
A Constraint is used to set certain rules for the stochastic engine to
evolve the atomic system. Therefore it has become possible to fully
customize and set any possibly imaginable rule.

.. inheritance-diagram:: fullrmc.Core.Constraint
    :parts: 1
"""
# standard libraries imports
from __future__ import print_function
import os, inspect, copy, uuid, re, itertools, shutil
from random import random as randfloat

# external libraries imports
import numpy as np

# fullrmc imports
from ..Globals import INT_TYPE, FLOAT_TYPE, LOGGER
from ..Globals import str, long, unicode, bytes, basestring, range, xrange, maxint
from ..Core.Collection import ListenerBase, is_number, is_integer, get_path
from ..Core.Collection import _AtomsCollector, reset_if_collected_out_of_date
from ..Core.Collection import get_caller_frames

class Constraint(ListenerBase):
    """ A constraint is used to direct the evolution of the atomic
    configuration towards the desired and most meaningful one.
    """
    def __init__(self):
        # init ListenerBase
        super(Constraint, self).__init__()
        # set engine
        self.__engine = None
        # set used flag
        self.set_used(True)
        # initialize variance squared
        self.__varianceSquared = 1
        # initialize atoms collector with datakeys to None. so it must be set in all subclasses.
        self._atomsCollector = _AtomsCollector(self, dataKeys=None)
        # initialize data
        self.__initialize_constraint()
        # computation cost
        self.set_computation_cost(0)
        # set frame data
        FRAME_DATA      = ('_Constraint__state', '_Constraint__used',
                           '_Constraint__tried', '_Constraint__accepted',
                           '_Constraint__varianceSquared','_Constraint__standardError',
                           '_Constraint__originalData', '_Constraint__data',
                           '_Constraint__activeAtomsDataBeforeMove', '_Constraint__activeAtomsDataAfterMove',
                           '_Constraint__amputationData', '_Constraint__afterMoveStandardError',
                           '_Constraint__amputationStandardError', '_Constraint__computationCost',
                           '_atomsCollector')
        RUNTIME_DATA    = ('_Constraint__state','_Constraint__tried', '_Constraint__accepted',
                           '_Constraint__varianceSquared','_Constraint__standardError','_Constraint__data',
                           '_atomsCollector',)
        object.__setattr__(self, 'FRAME_DATA',      tuple(FRAME_DATA)      )
        object.__setattr__(self, 'RUNTIME_DATA',    tuple(RUNTIME_DATA)    )


    def __setattr__(self, name, value):
        if name in ('FRAME_DATA','RUNTIME_DATA',):
            raise LOGGER.error("Setting '%s' is not allowed."%name)
        else:
            object.__setattr__(self, name, value)

    def __getstate__(self):
        state = {}
        for k in self.__dict__:
            if k in self.FRAME_DATA:
                continue
            else:
                state[k] = self.__dict__[k]
        # remove repository from engine
        #if state.get('_Constraint__engine', None) is not None:
        #    state['_Constraint__engine']._Engine__repository = None
        #print(self.__class__.__name__, '__getstate__')
        return state

    #def __setstate(self, state):
    #    #print(self.__class__.__name__, '__setstate__')
    #    self.__dict__ = state

    def _set_engine(self, engine):
        assert self.__engine is None, LOGGER.error("Re-setting constraint engine is not allowed.")
        from fullrmc.Engine import Engine
        assert isinstance(engine, Engine),LOGGER.error("Engine must be a fullrmc Engine instance")
        self.__engine = engine
        # set constraint unique id
        names = [c.constraintName for c in engine.constraints]
        idx = 0
        while True:
            name = self.__class__.__name__ + "_%i"%idx
            if name not in names:
                self.__constraintName = name
                break
            else:
                idx += 1
        # reset flags
        # resetting is commented because changing engine is not allowed anymore
        # and there is no need to reset_constraint anymore since at this point
        # all flag are not altered.
        #self.reset_constraint(reinitialize=False, flags=True, data=False)

    def __initialize_constraint(self, frame=None):
        usedIncluded, frame, allFrames = get_caller_frames(engine=self.engine,
                                                           frame=frame,
                                                           subframeToAll=False,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        # initialize flags
        if usedIncluded:
        #if frame is None:
            self.__state     = None
            self.__tried     = 0
            self.__accepted  = 0
            # initialize data
            self.__originalData              = None
            self.__data                      = None
            self.__activeAtomsDataBeforeMove = None
            self.__activeAtomsDataAfterMove  = None
            self.__amputationData            = None
            # initialize standard error
            self.__standardError           = None
            self.__afterMoveStandardError  = None
            self.__amputationStandardError = None
            ## reset atoms collector # ADDED 2017-JAN-08
            self._atomsCollector.reset()
            self._on_collector_reset()
            if self.engine is not None and len(self._atomsCollector.dataKeys):
                for realIndex in self.engine._atomsCollector.indexes:
                    self._on_collector_collect_atom(realIndex=realIndex)
        # loop all frames
        usedFrame = self.usedFrame
        for frm in allFrames:
            ac   = self._atomsCollector
            if frm != usedFrame:
                ac   = self._get_repository().pull(os.path.join(frm, 'constraints', self.__constraintName,'_atomsCollector'))
                this = copy.deepcopy(self)
                this._atomsCollector = ac
                this._atomsCollector.reset()
                this._on_collector_reset()
                if this.engine is not None and len(this._atomsCollector.dataKeys):
                    for realIndex in this.engine._atomsCollector.indexes:
                        this._on_collector_collect_atom(realIndex=realIndex)
            # dump into repostory
            self._dump_to_repository({'_Constraint__state'                    : None,
                                      '_Constraint__tried'                    : 0,
                                      '_Constraint__accepted'                 : 0,
                                      '_Constraint__originalData'             : None,
                                      '_Constraint__data'                     : None,
                                      '_Constraint__activeAtomsDataBeforeMove': None,
                                      '_Constraint__activeAtomsDataAfterMove' : None,
                                      '_Constraint__amputationData'           : None,
                                      '_Constraint__standardError'            : None,
                                      '_Constraint__afterMoveStandardError'   : None,
                                      '_Constraint__amputationStandardError'  : None,
                                      '_atomsCollector'                       : ac}, frame=frm)


    @property
    def constraintId(self):
        """Constraint unique ID create at instanciation time."""
        return self.listenerId

    @property
    def constraintName(self):
        """ Constraints unique name in engine given when added to engine."""
        return self.__constraintName

    @property
    def engine(self):
        """ Stochastic fullrmc's engine instance."""
        return self.__engine

    @property
    def usedFrame(self):
        """Get used frame in engine. If None then engine is not defined yet"""
        usedFrame = None
        if self.__engine is not None:
            usedFrame = self.__engine.usedFrame
        return usedFrame


    @property
    def computationCost(self):
        """ Computation cost number."""
        return self.__computationCost

    @property
    def state(self):
        """ Constraint's state."""
        return self.__state

    @property
    def tried(self):
        """ Constraint's number of tried moves."""
        return self.__tried

    @property
    def accepted(self):
        """ Constraint's number of accepted moves."""
        return self.__accepted

    @property
    def used(self):
        """ Constraint's used flag. Defines whether constraint is used
        in the stochastic engine at runtime or set inactive."""
        return self.__used

    @property
    def varianceSquared(self):
        """ Constraint's varianceSquared used in the stochastic engine
        at runtime to calculate the total constraint's standard error."""
        return self.__varianceSquared

    @property
    def standardError(self):
        """ Constraint's standard error value."""
        return self.__standardError

    @property
    def originalData(self):
        """ Constraint's original data calculated upon initialization."""
        return self.__originalData

    @property
    def data(self):
        """ Constraint's current calculated data."""
        return self.__data

    @property
    def activeAtomsDataBeforeMove(self):
        """ Constraint's current calculated data before last move."""
        return self.__activeAtomsDataBeforeMove

    @property
    def activeAtomsDataAfterMove(self):
        """ Constraint's current calculated data after last move."""
        return self.__activeAtomsDataAfterMove

    @property
    def afterMoveStandardError(self):
        """ Constraint's current calculated StandardError after last move."""
        return self.__afterMoveStandardError

    @property
    def amputationData(self):
        """ Constraint's current calculated data after amputation."""
        return self.__amputationData

    @property
    def amputationStandardError(self):
        """ Constraint's current calculated StandardError after amputation."""
        return self.__amputationStandardError

    @property
    def multiframeWeight(self):
        """Get constraint weight towards total in a multiframe system. """
        return FLOAT_TYPE(1.)

    def _apply_multiframe_prior(self, total):
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def _apply_scale_factor(self, total, scaleFactor):
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def _get_repository(self):
        if self.engine is None:
            return None
        else:
            return self.engine._get_repository()

    def _dump_to_repository(self, dataDict, frame=None):
        rep = self._get_repository()
        if rep is None:
            return
        if frame is None:
            frame = self.engine.usedFrame
        cp = os.path.join(frame, 'constraints', self.__constraintName)
        for name in dataDict:
            relativePath = os.path.join(cp,name)
            isRepoFile,fileOnDisk, infoOnDisk, classOnDisk = rep.is_repository_file(relativePath)
            if isRepoFile and fileOnDisk and infoOnDisk and classOnDisk:
                rep.update(value=dataDict[name], relativePath=relativePath)
            else:
                rep.dump(value=dataDict[name], relativePath=relativePath, replace=True)

    def _set_original_data(self, data, frame=None):
        """ Used only by the stochastic engine to set constraint's data as
        initialized for the first time."""
        self.__originalData = data
        # dump to repository
        self._dump_to_repository({'_Constraint__originalData' :self.__originalData}, frame=frame)

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

    def is_in_engine(self, engine):
        """
        Get whether constraint is already in defined and added to engine.
        It can be the same exact instance or a repository pulled instance
        of the same constraintId

        :Parameters:
            #. engine (stochastic fullrmc engine): Engine instance.

        :Returns:
            #. result (boolean): Whether constraint exists in engine.
        """
        if self in engine.constraints:
            return True
        elif self.constraintId in [c.constraintId for c in engine.constraints]:
            return True
        return False

    def set_variance_squared(self, value, frame=None):
        """
        Set constraint's variance squared that is used in the computation
        of the total stochastic engine standard error.

        :Parameters:
            #. value (number): Any positive non zero number.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used. If multiframe is given, all subframes will be
               targeted. If subframe is given, all other multiframe subframes
               will be targeted.
        """
        assert is_number(value), LOGGER.error("Variance squared accepted value must be convertible to a number")
        value = FLOAT_TYPE(value)
        assert value>0 , LOGGER.error("Variance squared must be positive non zero number.")
        usedIncluded, frame, allFrames = get_caller_frames(engine=self.engine,
                                                           frame=frame,
                                                           subframeToAll=True,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        if usedIncluded:
            self.__varianceSquared = value
        for frm in allFrames:
            self._dump_to_repository({'_Constraint__varianceSquared' :value}, frame=frm)


    def set_computation_cost(self, value, frame=None):
        """
        Set constraint's computation cost value. This is used at stochastic
        engine runtime to minimize computations and enhance performance by
        computing less costly constraints first. At every step, constraints
        will be computed in order starting from the less to the most
        computationally costly. Therefore upon rejection of a step because
        of an unsatisfactory rigid constraint, the left un-computed
        constraints at this step are guaranteed to be the most time coslty
        ones.

        :Parameters:
            #. value (number): computation cost.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used. If multiframe is given, all subframes will be
               targeted. If subframe is given, all other multiframe subframes
               will be targeted.
        """
        assert is_number(value), LOGGER.error("computation cost value must be convertible to a number")
        value = FLOAT_TYPE(value)
        usedIncluded, frame, allFrames = get_caller_frames(engine=self.engine,
                                                           frame=frame,
                                                           subframeToAll=True,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        if usedIncluded:
            self.__computationCost = value
        # dump to repository
        for frm in allFrames:
            self._dump_to_repository({'_Constraint__computationCost' :self.__computationCost}, frame=frm)


    @reset_if_collected_out_of_date # ADDED 2017-JAN-12
    def set_used(self, value, frame=None):
        """
        Set used flag.

        :Parameters:
            #. value (boolean): True to use this constraint in stochastic
               engine runtime.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used. If multiframe is given, all subframes will be
               targeted. If subframe is given, all other multiframe subframes
               will be targeted.
        """
        assert isinstance(value, bool), LOGGER.error("value must be boolean")
        # get used frame
        usedIncluded, frame, allFrames = get_caller_frames(engine=self.engine,
                                                           frame=frame,
                                                           subframeToAll=True,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        if usedIncluded:
            self.__used = value
        for frm in allFrames:
            self._dump_to_repository({'_Constraint__used' :value}, frame=frm)

    def set_state(self, value):
        """
        Set constraint's state. When constraint's state and stochastic
        engine's state don't match, constraint's data must be re-calculated.

        :Parameters:
            #. value (object): Constraint state value.
        """
        self.__state = value

    def set_tried(self, value):
        """
        Set constraint's number of tried moves.

        :Parameters:
            #. value (integer): Constraint tried moves value.
        """
        try:
            value = float(value)
        except:
            raise Exception(LOGGER.error("tried value must be convertible to a number"))
        assert is_integer(value), LOGGER.error("tried value must be integer")
        assert value>=0, LOGGER.error("tried value must be positive")
        self.__tried = int(value)

    def increment_tried(self):
        """ Increment number of tried moves. """
        self.__tried += 1

    def set_accepted(self, value):
        """
        Set constraint's number of accepted moves.

        :Parameters:
            #. value (integer): Constraint's number of accepted moves.
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
        """ Increment constraint's number of accepted moves. """
        self.__accepted += 1

    def set_standard_error(self, value):
        """
        Set constraint's standardError value.

        :Parameters:
            #. value (number): standard error value.
        """
        self.__standardError = value

    def set_data(self, value):
        """
        Set constraint's data value.

        :Parameters:
            #. value (number): constraint's data.
        """
        self.__data = value

    def set_active_atoms_data_before_move(self, value):
        """
        Set constraint's before move happens active atoms data value.

        :Parameters:
            #. value (number): Data value.
        """
        self.__activeAtomsDataBeforeMove = value

    def set_active_atoms_data_after_move(self, value):
        """
        Set constraint's after move happens active atoms data value.

        :Parameters:
            #. value (number): data value.
        """
        self.__activeAtomsDataAfterMove = value

    def set_after_move_standard_error(self, value):
        """
        Set constraint's standard error value after move happens.

        :Parameters:
            #. value (number): standard error value.
        """
        self.__afterMoveStandardError = value

    def set_amputation_data(self, value):
        """
        Set constraint's after amputation data.

        :Parameters:
            #. value (number): data value.
        """
        self.__amputationData = value

    def set_amputation_standard_error(self, value):
        """
        Set constraint's standardError after amputation.

        :Parameters:
            #. value (number): standard error value.
        """
        self.__amputationStandardError = value

    def reset_constraint(self, reinitialize=True, flags=False, data=False, frame=None):
        """
        Reset constraint.

        :Parameters:
            #. reinitialize (boolean): If set to True, it will override
               the rest of the flags and will completely reinitialize the
               constraint.
            #. flags (boolean): Reset the state, tried and accepted flags
               of the constraint.
            #. data (boolean): Reset the constraints computed data.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used. If multiframe is given, all subframes will be
               targeted. If subframe is given, rest of multiframe subframes
               will not be targeted.
        """
        usedIncluded, frame, allFrames = get_caller_frames(engine=self.engine,
                                                           frame=frame,
                                                           subframeToAll=False,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        # reinitialize constraint
        if reinitialize:
            flags = False
            data  = False
            self.__initialize_constraint(frame=frame)
        # initialize flags
        if flags:
            #if frame is None:
            if usedIncluded:
                self.__state                  = None
                self.__tried                  = 0
                self.__accepted               = 0
                self.__standardError          = None
                self.__afterMoveStandardError = None
            # dunp to repository
            for frm in allFrames:
                self._dump_to_repository({'_Constraint__state'                  : None,
                                          '_Constraint__tried'                  : 0,
                                          '_Constraint__accepted'               : 0,
                                          '_Constraint__standardError'          : None,
                                          '_Constraint__afterMoveStandardError' : None}, frame=frm)
        # initialize data
        if data:
            #if frame is None:
            if usedIncluded:
                self.__originalData              = None
                self.__data                      = None
                self.__activeAtomsDataBeforeMove = None
                self.__activeAtomsDataAfterMove  = None
                self.__standardError             = None
                self.__afterMoveStandardError    = None
            # dunp to repository
            for frm in allFrames:
                self._dump_to_repository({'_Constraint__originalData'             : None,
                                          '_Constraint__data'                     : None,
                                          '_Constraint__activeAtomsDataBeforeMove': None,
                                          '_Constraint__activeAtomsDataAfterMove' : None,
                                          '_Constraint__standardError'            : None,
                                          '_Constraint__afterMoveStandardError'   : None}, frame=frm)


    def update_standard_error(self):
        """ Compute and set constraint's standard error by calling
        compute_standard_error method and passing constraint's data."""
        self.set_standard_error(self.compute_standard_error(data = self.data))

    def get_constraint_value(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def get_constraint_original_value(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_standard_error(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_data(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_before_move(self, realIndexes, relativeIndexes):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def accept_move(self, realIndexes, relativeIndexes):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def reject_move(self, realIndexes, relativeIndexes):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_as_if_amputated(self, realIndex, relativeIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_as_if_inserted(self, realIndex, relativeIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def accept_amputation(self, realIndex, relativeIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def reject_amputation(self, realIndex, relativeIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def accept_insertion(self, realIndex, relativeIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def reject_insertion(self, realIndex, relativeIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def _on_collector_collect_atom(self, realIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def _on_collector_release_atom(self, realIndex):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def _on_collector_reset(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def export(self, *args, **kwargs):
        """Method must be overloaded in children classes."""
        LOGGER.warn("%s export method is not implemented"%(self.__class__.__name__))

    def plot(self, *args, **kwargs):
        """Method must be overloaded in children classes."""
        LOGGER.warn("%s plot method is not implemented"%(self.__class__.__name__))


class ExperimentalConstraint(Constraint):
    """
    Experimental constraint is any constraint related to experimental data.

    :Parameters:
        #. engine (None, fullrmc.Engine): Constraint's stochastic engine.
        #. experimentalData (numpy.ndarray, string): Experimental data goiven
           as numpy.ndarray or string path to load data using numpy.loadtxt
           method.
        #. dataWeights (None, numpy.ndarray): Weights array of the same number
           of points of experimentalData used in the constraint's standard
           error computation. Therefore particular fitting emphasis can be
           put on different data points that might be considered as more or less
           important in order to get a reasonable and plausible modal.\n
           If None is given, all data points are considered of the same
           importance in the computation of the constraint's standard error.\n
           If numpy.ndarray is given, all weights must be positive and all
           zeros weighted data points won't contribute to the total
           constraint's standard error. At least a single weight point is
           required to be non-zeros and the weights array will be automatically
           scaled upon setting such as the the sum of all the weights
           is equal to the number of data points.
        #. scaleFactor (number): A normalization scale factor used to normalize
           the computed data to the experimental ones.
        #. adjustScaleFactor (list, tuple): Used to adjust fit or guess
           the best scale factor during stochastic engine runtime.
           It must be a list of exactly three entries.\n
           #. The frequency in number of generated moves of finding the best
              scale factor. If 0 frequency is given, it means that the scale
              factor is fixed.
           #. The minimum allowed scale factor value.
           #. The maximum allowed scale factor value.

    **NB**: If adjustScaleFactor first item (frequency) is 0, the scale factor
    will remain untouched and the limits minimum and maximum won't be checked.

    """
    def __init__(self, experimentalData, dataWeights=None, scaleFactor=1.0, adjustScaleFactor=(0, 0.8, 1.2) ):
        # initialize constraint
        super(ExperimentalConstraint, self).__init__()
        # set plotting default parameters.
        self._plotDefaultParameters = {}
        self._plotDefaultParameters['expParams']    = {'label':"experimental","color":'red','marker':'o','markersize':7.5, 'markevery':1, 'zorder':0}
        self._plotDefaultParameters['totParams']    = {'label':"total", 'color':'black','linewidth':3.0, 'zorder':1}
        self._plotDefaultParameters['noWParams']    = {'label':"total - no window", 'color':'black','linewidth':1.0, 'zorder':1}
        self._plotDefaultParameters['shaParams']    = {'label':"shape function", 'color':'black','linewidth':1.0, 'linestyle':'dashed', 'zorder':2}
        self._plotDefaultParameters['parParams']    = {'linewidth':1.0, 'markevery':5, 'markersize':5, 'zorder':3}
        self._plotDefaultParameters['legendParams'] = {'frameon':False, 'ncol':2, 'loc':'best'}
        self._plotDefaultParameters['xlabelParams'] = None
        self._plotDefaultParameters['ylabelParams'] = None
        # set the constraint's experimental data
        self.__dataWeights      = None
        self._usedDataWeights   = None # this is the same as dataWeights but restricted to given limits
        self.__experimentalData = None
        # init limits
        self.__limits           = None
        self.__limitsIndexStart = None
        self.__limitsIndexEnd   = None
        # set constraint data prior. Prior is got from any other structures
        # contributing to the total constraint. This is used in multiframe
        # modeling e.g. distribution of nanoparticules size
        # prior weight can be different per type of constraint as systems
        # respond differently to different probing techniques.
        self._set_multiframe_prior_and_weight(multiframePrior=None, multiframeWeight=None)
        self.set_experimental_data(experimentalData)
        # set scale factor
        self.set_scale_factor(scaleFactor)
        # set adjust scale factor
        self.set_adjust_scale_factor(adjustScaleFactor)
        # set data weights
        self.set_data_weights(dataWeights)
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_ExperimentalConstraint__dataWeights',
                           '_ExperimentalConstraint__experimentalData',
                           '_ExperimentalConstraint__scaleFactor',
                           '_ExperimentalConstraint__multiframePrior',
                           '_ExperimentalConstraint__multiframeWeight',
                           '_ExperimentalConstraint__limits',
                           '_ExperimentalConstraint__limitsIndexStart',
                           '_ExperimentalConstraint__limitsIndexEnd',
                           '_usedDataWeights',
                           '_fittedScaleFactor'] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( ['_ExperimentalConstraint__scaleFactor',
                              '_ExperimentalConstraint__multiframePrior',
                              '_ExperimentalConstraint__multiframeWeight',
                              '_fittedScaleFactor',] )
        object.__setattr__(self, 'FRAME_DATA',  tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA',tuple(RUNTIME_DATA) )


    def _set_multiframe_prior_and_weight(self, multiframePrior, multiframeWeight):
        """To be used internally. Normally at engine run level after returning all
        result from all frames.
        multiframePrior and multiframeWeight can be both None. If not None,
        multiframePrior must have the same length as model constraint total
        and multiframeWeight must a positive value >=0 and <=1
        """
        if multiframePrior is None:
            assert multiframeWeight is None, LOGGER.error('multiframeWeight must be None if multiframePrior is None')
        else:
            # check multiframePrior
            assert isinstance(multiframePrior, np.ndarray), LOGGER.error('multiframePrior must be None or a numpy ndarray')
            assert multiframePrior.dtype == FLOAT_TYPE, LOGGER.error('multiframePrior numpy ndarray data type must be %s'%FLOAT_TYPE)
            assert multiframePrior.shape == self.__experimentalData.shape, LOGGER.error("multiframePrior must have the same shape as experimental data")
            # check multiframeWeight
            assert multiframeWeight is not None, LOGGER.error('multiframeWeight must not be None if multiframePrior is given')
            try:
                multiframeWeight = FLOAT_TYPE(multiframeWeight)
            except Exception as err:
                raise Exception(LOGGER.error("multiframeWeight must be None or a float"))
            assert multiframeWeight>=0, LOGGER.error("multiframeWeight number must be >=0")
            assert multiframeWeight<=1, LOGGER.error("multiframeWeight number must be <=1")
        self.__multiframePrior  = multiframePrior
        self.__multiframeWeight = multiframeWeight
        self._dump_to_repository({'_ExperimentalConstraint__multiframePrior' : self.__multiframePrior,
                                  '_ExperimentalConstraint__multiframeWeight': self.__multiframeWeight})


    def _set_fitted_scale_factor_value(self, scaleFactor):
        """
        This method is a scaleFactor value without any validity checking.
        Meant to be used internally only.
        """
        if self.__scaleFactor != scaleFactor:
            LOGGER.info("@%s Experimental constraint '%s' scale factor updated from %.6f to %.6f" %(self.engine.usedFrame, self.__class__.__name__, self.__scaleFactor, scaleFactor))
            self.__scaleFactor = scaleFactor

    def __set_limits(self, limits):
        """ for internal use only by ExperimentalConstraint children.
        call as such self._ExperimentalConstraint__set_limits(limits)
        """
        if limits is None:
            self.__limits = (None, None)
        else:
            assert isinstance(limits, (list, tuple)), LOGGER.error("limits must be None or a list")
            limits = list(limits)
            assert len(limits) == 2, LOGGER.error("limits list must have exactly two elements")
            if limits[0] is not None:
                assert is_number(limits[0]), LOGGER.error("if not None, the first limits element must be a number")
                limits[0] = FLOAT_TYPE(limits[0])
                assert limits[0]>=0, LOGGER.error("if not None, the first limits element must be bigger or equal to 0")
            if limits[1] is not None:
                assert is_number(limits[1]), LOGGER.error("if not None, the second limits element must be a number")
                limits[1] = FLOAT_TYPE(limits[1])
                assert limits[1]>0, LOGGER.error("if not None, the second limits element must be a positive number")
            if  limits[0] is not None and limits[1] is not None:
                assert limits[0]<limits[1], LOGGER.error("if not None, the first limits element must be smaller than the second limits element")
            self.__limits = (limits[0], limits[1])
        # get minimumDistance and maximumDistance indexes
        if self.limits[0] is None:
            self.__limitsIndexStart = 0
        else:
            self.__limitsIndexStart = (np.abs(self.__experimentalData[:,0]-self.__limits[0])).argmin()
        if self.limits[1] is None:
            self.__limitsIndexEnd = self.__experimentalData.shape[0]-1
        else:
            self.__limitsIndexEnd = (np.abs(self.__experimentalData[:,0]-self.__limits[1])).argmin()
        # dump to repository
        self._dump_to_repository({'_ExperimentalConstraint__limits'           : self.__limits,
                                  '_ExperimentalConstraint__limitsIndexStart' : self.__limitsIndexStart,
                                  '_ExperimentalConstraint__limitsIndexEnd'   : self.__limitsIndexEnd
                                 })

    @property
    def experimentalData(self):
        """ Experimental data of the constraint. """
        return self.__experimentalData

    @property
    def dataWeights(self):
        """ Experimental data points weight"""
        return self.__dataWeights

    @property
    def multiframeWeight(self):
        """Get constraint weight towards total in a multiframe system. """
        return self.__multiframeWeight

    @property
    def multiframePrior(self):
        """Get constraint multiframe prior array. """
        return self.__multiframePrior

    @property
    def scaleFactor(self):
        """ Constraint's scaleFactor. """
        return self.__scaleFactor

    @property
    def adjustScaleFactor(self):
        """Adjust scale factor tuple."""
        return (self.__adjustScaleFactorFrequency, self.__adjustScaleFactorMinimum, self.__adjustScaleFactorMaximum)

    @property
    def adjustScaleFactorFrequency(self):
        """ Scale factor adjustment frequency. """
        return self.__adjustScaleFactorFrequency

    @property
    def adjustScaleFactorMinimum(self):
        """ Scale factor adjustment minimum number allowed. """
        return self.__adjustScaleFactorMinimum

    @property
    def adjustScaleFactorMaximum(self):
        """ Scale factor adjustment maximum number allowed. """
        return self.__adjustScaleFactorMaximum

    @property
    def limits(self):
        """ Used data X limits."""
        return self.__limits

    @property
    def limitsIndexStart(self):
        """ Used data start index as calculated from limits."""
        return self.__limitsIndexStart

    @property
    def limitsIndexEnd(self):
        """ Used data end index as calculated from limits."""
        return self.__limitsIndexEnd


    def _apply_multiframe_prior(self, total):
        """Given a constraint data total, apply multiframe total prior.

        :Parameters:
            #. total (numpy.ndarray): Constraint total.

        :Returns:
            #. transformed (numpy.ndarray): Prior applied to constraint total.
        """
        # THIS MUST BE CORRECTED TO ACCOMODATE DIFFERENT DATA LIMITS
        # REVISITING set_limits IMPLEMENTATION IS NEEDED TO AUTOMATICALLY
        # ADJUST _apply_multiframe_prior TO DATA LIMITS
        # limitsIndexStart AND limitsIndexEnd MUST BE MOVED TO
        # ExperimentalConstraint
        if self.__multiframeWeight is None:
            return total
        else:
            return self.__multiframePrior + self.__multiframeWeight*total

    def set_scale_factor(self, scaleFactor):
        """
        Set the scale factor. This method doesn't allow specifying frames. It
        will target used frame only.

        :Parameters:
             #. scaleFactor (number): A normalization scale factor used to
                normalize the computed data to the experimental ones.
        """
        assert is_number(scaleFactor), LOGGER.error("scaleFactor must be a number")
        self.__scaleFactor      = FLOAT_TYPE(scaleFactor)
        self._fittedScaleFactor = self.__scaleFactor
        # dump to repository
        self._dump_to_repository({'_ExperimentalConstraint__scaleFactor': self.__scaleFactor})
        ## reset constraint
        self.reset_constraint()

    def set_adjust_scale_factor(self, adjustScaleFactor):
        """
        Set adjust scale factor. This method doesn't allow specifying frames. It
        will target used frame only.

        :Parameters:
            #. adjustScaleFactor (list, tuple): Used to adjust fit or guess
               the best scale factor during stochastic engine runtime.
               It must be a list of exactly three entries.\n
               #. The frequency in number of generated moves of finding the best
                  scale factor. If 0 frequency is given, it means that the scale
                  factor is fixed.
               #. The minimum allowed scale factor value.
               #. The maximum allowed scale factor value.
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
        # dump to repository
        self._dump_to_repository({'_ExperimentalConstraint__adjustScaleFactorFrequency': self.__adjustScaleFactorFrequency,
                                  '_ExperimentalConstraint__adjustScaleFactorMinimum'  : self.__adjustScaleFactorMinimum,
                                  '_ExperimentalConstraint__adjustScaleFactorMaximum'  : self.__adjustScaleFactorMaximum})
        # reset constraint
        self.reset_constraint()

    def _set_adjust_scale_factor_frequency(self, freq):
        """This must never be used externally. It's added to serve RemoveGenerators
         and only used internally upon calling compute_as_if_amputated """
        self.__adjustScaleFactorFrequency = freq

    def set_experimental_data(self, experimentalData):
        """
        Set the constraint's experimental data. This method will raise an error
        if called after adding constraint to stochastic engine.

        :Parameters:
            #. experimentalData (numpy.ndarray, string): Experimental data as
               numpy.ndarray or string path to load data using numpy.loadtxt
               method.
        """
        assert self.engine is None, LOGGER.error("Experimental data must be set before engine is set") # ADDED 2018-11-21
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
        # dump to repository
        self._dump_to_repository({'_ExperimentalConstraint__experimentalData': self.__experimentalData})

    def set_data_weights(self, dataWeights, frame=None):
        """
        Set experimental data points weight. Data weights will be automatically
        normalized.

        :Parameters:
            #. dataWeights (None, numpy.ndarray): Weights array of the same
               number of points of experimentalData used in the constraint's
               standard error computation. Therefore particular fitting
               emphasis can be put on different data points that might be
               considered as more or less important in order to get a
               reasonable and plausible model.\n
               If None is given, all data points are considered of the same
               importance in the computation of the constraint's standard error.\n
               If numpy.ndarray is given, all weights must be positive and all
               zeros weighted data points won't contribute to the total
               constraint's standard error. At least a single weight point is
               required to be non-zeros and the weights array will be
               automatically scaled upon setting such as the the sum of all
               the weights is equal to the number of data points.
            #. frame (None, string): Target frame name. If None, engine used
               frame is used. If multiframe is given, all subframes will be
               targeted. If subframe is given, rest of multiframe subframes
               will not be targeted.
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
        # get all frames
        usedIncluded, frame, allFrames = get_caller_frames(engine=self.engine,
                                                           frame=frame,
                                                           subframeToAll=False,
                                                           caller="%s.%s"%(self.__class__.__name__,inspect.stack()[0][3]) )
        if usedIncluded:
            self.__dataWeights = dataWeights
        # dump to repository
        for frm in allFrames:
            self._dump_to_repository({'_ExperimentalConstraint__dataWeights': self.__dataWeights},
                                     frame=frm)
        # set used data weights
        self._set_used_data_weights()

    def _set_used_data_weights(self, limitsIndexStart=None, limitsIndexEnd=None):
        # set used dataWeights
        if self.__dataWeights is None:
            self._usedDataWeights = None
        else:
            if limitsIndexStart is None:
                limitsIndexStart = 0
            if limitsIndexEnd is None:
                limitsIndexEnd = self.__experimentalData.shape[0]
            self._usedDataWeights  = np.copy(self.dataWeights[limitsIndexStart:limitsIndexEnd+1])
            assert np.sum(self._usedDataWeights), LOGGER.error("used points dataWeights are all zero.")
            self._usedDataWeights /= FLOAT_TYPE( np.sum(self._usedDataWeights) )
            self._usedDataWeights *= FLOAT_TYPE( len(self._usedDataWeights) )
        # dump to repository
        if self.engine is not None:
            isNormalFrame, isMultiframe, isSubframe = self.engine.get_frame_category(frame=self.engine.usedFrame)
            if isSubframe:
                LOGGER.usage("Setting experimental data weight for multiframe '%s' subframe. This is not going to automatically propagate to all other subframes."%(self.engine.usedFrame,))
        self._dump_to_repository({'_usedDataWeights': self._usedDataWeights})

    def check_experimental_data(self, experimentalData):
        """
        Checks the constraint's experimental data
        This method must be overloaded in all experimental constraint
        sub-classes.

        :Parameters:
            #. experimentalData (numpy.ndarray): Experimental data numpy.ndarray.
        """
        raise Exception(LOGGER.error("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def fit_scale_factor(self, experimentalData, modelData, dataWeights):
        """
        The best scale factor value is computed by minimizing :math:`E=sM`.\n

        Where:
            #. :math:`E` is the experimental data.
            #. :math:`s` is the scale factor.
            #. :math:`M` is the model constraint data.

        This method doesn't allow specifying frames. It will target used frame
        only.

        :Parameters:
            #. experimentalData (numpy.ndarray): Experimental data.
            #. modelData (numpy.ndarray): Constraint modal data.
            #. dataWeights (None, numpy.ndarray): Data points weights to
               compute the scale factor. If None is given, all data points
               will be considered as having the same weight.

        :Returns:
            #. scaleFactor (number): The new scale factor fit value.

        **NB**: This method won't update the internal scale factor value
        of the constraint. It always computes the best scale factor given
        experimental and atomic model data.
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
        Checks if scale factor should be updated according to the given scale
        factor frequency and engine's accepted steps. If adjustment is due,
        a new scale factor will be computed using fit_scale_factor method,
        otherwise the the constraint's scale factor will be returned.

        :Parameters:
            #. experimentalData (numpy.ndarray): the experimental data.
            #. modelData (numpy.ndarray): the constraint modal data.
            #. dataWeights (None, numpy.ndarray): the data points weights to
               compute the scale factor. If None is given, all data points
               will be considered as having the same weight.

        :Returns:
            #. scaleFactor (number): Constraint's scale factor or the
            new scale factor fit value.

        **NB**: This method WILL NOT UPDATE the internal scale factor
        value of the constraint.
        """
        SF = self.__scaleFactor
        # check to update scaleFactor
        if self.__adjustScaleFactorFrequency:
            if not self.engine.accepted%self.__adjustScaleFactorFrequency:
                SF = self.fit_scale_factor(experimentalData, modelData, dataWeights)
        return SF

    def compute_standard_error(self, experimentalData, modelData):
        """
        Compute the squared deviation between modal computed data
        and the experimental ones.

        .. math::
            SD = \\sum \\limits_{i}^{N} W_{i}(Y(X_{i})-F(X_{i}))^{2}

        Where:\n
        :math:`N` is the total number of experimental data points. \n
        :math:`W_{i}` is the data point weight. It becomes equivalent to 1
        when dataWeights is set to None. \n
        :math:`Y(X_{i})` is the experimental data point :math:`X_{i}`. \n
        :math:`F(X_{i})` is the computed from the model data  :math:`X_{i}`. \n

        :Parameters:
            #. experimentalData (numpy.ndarray): Experimental data.
            #. modelData (numpy.ndarray): The data to compare with the
               experimental one and compute the squared deviation.

        :Returns:
            #. standardError (number): The calculated standard error of
               the constraint.
        """
        # compute difference
        diff = experimentalData-modelData
        # return squared deviation
        if self.__dataWeights is None:
            return np.add.reduce((diff)**2)
        else:
            return np.add.reduce(self.__dataWeights*(diff)**2)

    def _get_needed_data_for_constraint_data_dictionary(self, *args, **kwargs):
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def get_constraint_data_dictionary(self, frame):
        """
        Get a dictionary of all constraint meaningful data to plot or export.

        :Parameters:
            #. frame (string): frame to pull and build contraint data. It can
               be a traditional frame, a multiframe or a subframe

        :Returns:
            #. data (dictionary): Dictionary of all meaningful data including:

                * output: list of frames dictionary constraint data
                * experimental_x: numpy array of experimental x data.
                * experimental_y: numpy array of experimental y data.
                * model_x: numpy array of model x data.
                * frames_weight: list of all frames weight.
                * frames_name: list of all frames name.
                * number_of_atoms_removed: list of number of removed atoms from
                  each frame.
                * shape_array: list of system shape function (numpy array) of
                  all frames.
                * window_function: list of window function (numpy array) of
                  all frames.
                * scale_factor: list of all frames scale factor.
                * standard_error: list of all frames standard error.
                * weighted_output: dictionary of all frames weighted constraint
                  data using 'frames_weight'
                * weighted_number_of_atoms_removed: All frames averaged number
                  of removed atoms using 'frames_weight'
                * weighted_scale_factor: All frames averaged scale factor
                  using 'frames_weight'
                * weighted_standard_error: All frames weighted standard error
                  using 'frames_weight'

        """
        from collections import OrderedDict
        neededData = self._get_needed_data_for_constraint_data_dictionary()
        isNormalFrame, isMultiframe, isSubframe = self.engine.get_frame_category(frame=frame)
        # get frames constraint data
        if isNormalFrame or isSubframe:
            frames       = [frame]
            framesName   = [frame,]
        else:
            frames       = [os.path.join(frame,frm) for frm in self.engine.frames[frame]['frames_name']]
            framesName   = self.engine.frames[frame]['frames_name']
        # initiate
        framesData = OrderedDict()
        framesData['output']                           = []
        framesData['experimental_x']                   = []
        framesData['experimental_y']                   = []
        framesData['model_x']                          = []
        framesData['frames_weight']                    = []
        framesData['frames_name']                      = []
        framesData['number_of_atoms_removed']          = []
        framesData['window_function']                  = []
        framesData['shape_array']                      = []
        framesData['scale_factor']                     = []
        framesData['standard_error']                   = []
        framesData['weighted_output']                  = None
        framesData['weighted_number_of_atoms_removed'] = None
        framesData['weighted_scale_factor']            = None
        framesData['weighted_standard_error']          = None
        # loop frames and get data
        _constraint = None
        for frm in frames:
            if frm == self.engine.usedFrame:
                output   = self._get_constraint_value(self.data, applyMultiframePrior=False)
                stdErr   = self.compute_standard_error(modelData = output["total"])
                expDis   = self._experimentalX
                expData  = self._experimentalY
                modelX   = self._modelX
                weight   = self.multiframeWeight if self.multiframeWeight is not None else FLOAT_TYPE(1.0)
                nRemAt   = len(self.engine._atomsCollector)
                sArray   = None
                if hasattr(self, '_shapeArray'):
                    sArray = self._shapeArray
                wFunc    = self.windowFunction
                sFactor  = self.scaleFactor
            else:
                repo = self.engine._get_repository()
                if _constraint is None:
                    _constraint = copy.deepcopy(self)
                object.__setattr__(_constraint.engine, '_Engine__usedFrame', frm)
                for name in neededData:
                    repoName = neededData[name]
                    if isinstance(name, tuple):
                        if name[0]=='engine':
                            value = repo.pull(relativePath=os.path.join(frm,repoName))
                            object.__setattr__(_constraint.engine, name[1], value)
                        else:
                            raise Exception(LOGGER.error('Wrong neededData format. Report issue'))
                    elif repoName.startswith('_Engine__'):
                        value = repo.pull(relativePath=os.path.join(frm,repoName))
                        object.__setattr__(_constraint.engine, name, value)
                    else:
                        value = repo.pull(relativePath=os.path.join(frm,'constraints',_constraint.constraintName,repoName))
                        object.__setattr__(_constraint, name, value)
                # append frames data
                if _constraint.data is None:
                    LOGGER.warn("%s constraint data for frame are not computed."%(self.__class__.__name__))
                    return None
                # get attributes
                output = _constraint._get_constraint_value(_constraint.data, applyMultiframePrior=False)
                stdErr = _constraint.compute_standard_error(modelData = output["total"])
                if len(framesData['experimental_x'])>0:
                    assert np.allclose(framesData['experimental_x'], _constraint._experimentalX), LOGGER.error("Frames experimentalX must match")
                    assert np.allclose(framesData['experimental_y'], _constraint._experimentalY), LOGGER.error("Frames experimentalY must match")
                    assert np.allclose(framesData['model_x'], _constraint._modelX), LOGGER.error("Frames modelX must match")
                else:
                    expDis   = _constraint._experimentalX
                    expData  = _constraint._experimentalY
                    modelX   = _constraint._modelX
                weight = _constraint.multiframeWeight if _constraint.multiframeWeight is not None else FLOAT_TYPE(1.0)
                nRemAt   = len(_constraint.engine._atomsCollector)
                sArray   = None
                if hasattr(self, '_shapeArray'):
                    sArray = _constraint._shapeArray
                wFunc    = _constraint.windowFunction
                sFactor  = _constraint.scaleFactor
            # append frame data
            framesData['output'].append(output)
            framesData['frames_name'].append(frm)
            framesData['experimental_x'].append(expDis)
            framesData['experimental_y'].append(expData)
            framesData['model_x'].append(modelX)
            framesData['frames_weight'].append(weight)
            framesData['number_of_atoms_removed'].append(nRemAt)
            framesData['shape_array'].append(sArray)
            framesData['window_function'].append(wFunc)
            framesData['scale_factor'].append(sFactor)
            framesData['standard_error'].append(stdErr)
        ## compute frames weighted
        framesWeight = np.array(framesData['frames_weight'], dtype=FLOAT_TYPE)
        if np.sum(framesWeight)==0:
            if len(framesWeight)==1:
                framesWeight = np.array([1])
            else:
                raise Exception("Frames weight sum is found to be 0. PLEASE REPORT THIS BUG")
        weightedOutput = {}
        for key in framesData['output'][0]:
            if any([framesData['output'][i][key] is None for i, w in enumerate(framesWeight)]):
                weightedOutput[key] = None
            else:
                weightedOutput[key] = np.sum([w*framesData['output'][i][key] for i, w in enumerate(framesWeight)], axis=0)
                weightedOutput[key] = weightedOutput[key]/sum(framesWeight)
        framesData['weighted_output']                  = weightedOutput
        framesData['weighted_number_of_atoms_removed'] = np.average(framesData['number_of_atoms_removed'], weights=framesWeight)
        framesData['weighted_standard_error']          = self.compute_standard_error(modelData=weightedOutput['total'])
        framesData['weighted_scale_factor']            = np.average(framesData['scale_factor'], weights=framesWeight)
        # return
        return framesData


    def _plot(self, frame, output, experimentalX, experimentalY,
                    shellCenters, shapeArray, numberOfRemovedAtoms,
                    standardError, scaleFactor, multiframeWeight,
                    # plotting arguments
                    ax=None, intra=True, inter=True, totalNoWindow=False,
                    xlabelParams=None,
                    ylabelParams=None,
                    legendParams=None,
                    titleFormat = "",
                    expParams = {'label':"experimental","color":'red','marker':'o','markersize':7.5, 'markevery':1, 'zorder':0},
                    totParams = {'label':"total", 'color':'black','linewidth':3.0, 'zorder':1},
                    noWParams = {'label':"total - no window", 'color':'black','linewidth':1.0, 'zorder':1},
                    shaParams = {'label':"shape function", 'color':'black','linewidth':1.0, 'linestyle':'dashed'},
                    parParams = {'linewidth':1.0, 'markevery':5, 'markersize':5, 'zorder':-1},
                    show=True):
        # import matplotlib
        import matplotlib.pyplot as plt
        # get axes
        if ax is None:
            FIG  = plt.figure()
            AXES = plt.gca()
        else:
            AXES = ax
            FIG  = AXES.get_figure()
        # Create plotting styles
        COLORS  = ["b",'g','c','y','m']
        MARKERS = ["",'.','+','^','|']
        INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], list(reversed(COLORS)))]
        INTRA_STYLES = [r[0] + r[1]for r in itertools.product(MARKERS, INTRA_STYLES)]
        INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], COLORS)]
        INTER_STYLES = [r[0] + r[1]for r in itertools.product(MARKERS, INTER_STYLES)]
        # plot experimental
        AXES.plot(experimentalX,experimentalY, **expParams)
        AXES.plot(shellCenters, output["total"], **totParams )
        if totalNoWindow and output["total_no_window"] is not None:
            AXES.plot(shellCenters, output["total_no_window"], **noWParams )
        if shapeArray is not None:
            AXES.plot(shellCenters, shapeArray, **shaParams )
        # plot partials
        intraStyleIndex = 0
        interStyleIndex = 0
        for key in output:
            val = output[key]
            if key in ("total", "total_no_window"):
                continue
            elif "intra" in key and intra:
                AXES.plot(shellCenters, val, INTRA_STYLES[intraStyleIndex], label=key, **parParams )
                intraStyleIndex+=1
            elif "inter" in key and inter:
                AXES.plot(shellCenters, val, INTER_STYLES[interStyleIndex], label=key, **parParams )
                interStyleIndex+=1
        # plot legend
        if legendParams is not None:
            AXES.legend(**legendParams)
        # set title
        if len(titleFormat):
            name = ' '.join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__))
            FIG.canvas.set_window_title(name)
            # format title
            title = titleFormat.format(frame=frame,
                                       standardError=standardError,
                                       numberOfRemovedAtoms=numberOfRemovedAtoms,
                                       scaleFactor=scaleFactor,
                                       multiframeWeight=multiframeWeight)
            AXES.set_title(title)
        # set axis labels
        if xlabelParams is not None:
            #AXES.set_xlabel("$r(\AA)$", size=16)
            AXES.set_xlabel(**xlabelParams)
        if ylabelParams is not None:
            AXES.set_ylabel(**ylabelParams)
        # set background color
        FIG.patch.set_facecolor('white')
        #show
        if show:
            plt.show()
        return FIG, AXES

    def plot(self, frame=None, multiplot=False,
                   ax=None, intra=True, inter=True, shapeFunc=True,
                   xlabelParams=True,
                   ylabelParams=True,
                   legendParams=True,
                   expParams=None,
                   totParams=None,
                   noWParams=None,
                   shaParams=None,
                   parParams=None,
                   titleFormat = "@{frame} (${numberOfRemovedAtoms:.1f}$ $rem.$ $at.$) $Std.Err.={standardError:.3f}$\n$scale$ $factor$=${scaleFactor:.2f}$ - $multiframe$ $weight$=${multiframeWeight:.3f}$",
                   show=True):
        """
        Plot constraint data

        :Parameters:
            #. frame (None, string): The frame name to plot. If None, used frame
               will be plotted.
            #. multiplot (boolean): If multiframe is given, multiplot insures
               plotting all frames differently in a single multiplot figure.
               In this case, given ax is ommitted.
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If None is given a new plot figure will be created.
            #. intra (boolean): Whether to add intra-molecular pair
               distribution function features to the plot.
            #. inter (boolean): Whether to add inter-molecular pair
               distribution function features to the plot.
            #. shapeFunc (boolean): Whether to add shape function to the plot
               only when exists.
            #. xlabelParams (dict, boolean): matplotlib.axes.Axes.set_xlabel
               parameters. If True, default parameters given by constraint's
               property _plotDefaultParameters is set. If False x axes label is
               omitted
            #. ylabelParams (dict, boolean): matplotlib.axes.Axes.set_ylabel
               parameters. If True,  default parameters given by constraint's
               property _plotDefaultParameters is set. If False y axes label is
               omitted
            #. legendParams (dict, boolean):matplotlib.axes.Axes.legend
               parameters. If True,  default parameters given by constraint's
               property _plotDefaultParameters is set. If False is omitted
            #. titleFormat (string): title format. If empty string is given no
               title will be added to figure axes
            #. expParams (None, dict): experimental data matplotlib.pyplot.plot
               parameters. If None,  default parameters given by constraint's
               property _plotDefaultParameters is set. If dict, it will be the
               update dict to _plotDefaultParameters
            #. totParams (None, dict): total data matplotlib.pyplot.plot
               parameters. If None,  default parameters given by constraint's
               property _plotDefaultParameters is set. If dict, it will be the
               update dict to _plotDefaultParameters
            #. noWParams (None, dict): constraint no-window total data matplotlib.pyplot.plot
               parameters. If None,  default parameters given by constraint's
               property _plotDefaultParameters is set. If dict, it will be the
               update dict to _plotDefaultParameters
            #. shaParams (None, dict): constraint shape array data matplotlib.pyplot.plot
               parameters. If None,  default parameters given by constraint's
               property _plotDefaultParameters is set. If dict, it will be the
               update dict to _plotDefaultParameters
            #. parParams (None, dict): constraint partial data matplotlib.pyplot.plot
               parameters. If None,  default parameters given by constraint's
               property _plotDefaultParameters is set. If dict, it will be the
               update dict to _plotDefaultParameters
            #. show (boolean): Whether to render and show figure before
               returning.

        :Returns:
            #. figure (matplotlib Figure): matplotlib used figure.
            #. axes (matplotlib Axes): matplotlib used axes.
        """
        assert isinstance(multiplot, bool), LOGGER.error("multiplot must be boolean")
        # check expParams
        if expParams is None:
            expParams = {}
        assert isinstance(expParams, dict), "expParams must be None or dict"
        default   = copy.deepcopy( self._plotDefaultParameters['expParams'] )
        default.update(expParams)
        expParams = default
        # check totParams
        if totParams is None:
            totParams = {}
        assert isinstance(totParams, dict), "totParams must be None or dict"
        default   = copy.deepcopy( self._plotDefaultParameters['totParams'] )
        default.update(totParams)
        totParams = default
        # check noWParams
        if noWParams is None:
            noWParams = {}
        assert isinstance(noWParams, dict), "noWParams must be None or dict"
        default   = copy.deepcopy( self._plotDefaultParameters['noWParams'] )
        default.update(noWParams)
        noWParams = default
        # check shaParams
        if shaParams is None:
            shaParams = {}
        assert isinstance(shaParams, dict), "shaParams must be None or dict"
        default   = copy.deepcopy( self._plotDefaultParameters['shaParams'] )
        default.update(shaParams)
        shaParams = default
        # check parParams
        if parParams is None:
            parParams = {}
        assert isinstance(parParams, dict), "parParams must be None or dict"
        default   = copy.deepcopy( self._plotDefaultParameters['parParams'] )
        default.update(parParams)
        parParams = default
        # check xlabel
        if xlabelParams is False:
            xlabelParams = None
        elif xlabelParams is True:
            xlabelParams = self._plotDefaultParameters['xlabelParams']
        elif  isinstance(xlabelParams, dict):
            default   = copy.deepcopy( self._plotDefaultParameters['xlabelParams'] )
            default.update(xlabelParams)
            xlabelParams = default
        else:
            assert xlabelParams is None, "xlabelParams must be None, boolean or dict"
        # check ylabel
        if ylabelParams is False:
            ylabelParams = None
        elif ylabelParams is True:
            ylabelParams = self._plotDefaultParameters['ylabelParams']
        elif  isinstance(ylabelParams, dict):
            default = copy.deepcopy( self._plotDefaultParameters['ylabelParams'] )
            default.update(ylabelParams)
            ylabelParams = default
        else:
            assert ylabelParams is None, "ylabelParams must be None, boolean or dict"
        # check ylabel
        if legendParams is False:
            legendParams = None
        elif legendParams is True:
            legendParams = self._plotDefaultParameters['legendParams']
        elif  isinstance(legendParams, dict):
            default = copy.deepcopy( self._plotDefaultParameters['legendParams'] )
            default.update(legendParams)
            legendParams = default
        else:
            assert legendParams is None, "legendParams must be None, boolean or dict"
        # get frame
        if frame is None:
            frame = self.engine.usedFrame
        isNormalFrame, isMultiframe, isSubframe = self.engine.get_frame_category(frame=frame)
        framesData = self.get_constraint_data_dictionary(frame=frame)
        if framesData is None:
            return
        # get number of frames
        numberOfFrames = len(framesData['frames_weight'])
        multiplot      = multiplot and numberOfFrames>1
        if multiplot:
            import matplotlib.pyplot as plt
            nrows     = int(np.sqrt(float(numberOfFrames)))
            ncols     = int(np.ceil(float(numberOfFrames)/nrows))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            axes      = axes.flatten()
            fig.subplots_adjust(hspace=0.4)
            for idx, output in enumerate(framesData['output']):
                _show = show and idx==len(framesData['output'])-1
                FIG, AXES = self._plot(frame                 = framesData['frames_name'][idx],
                                       output                = output,
                                       experimentalX         = framesData['experimental_x'][idx],
                                       experimentalY         = framesData['experimental_y'][idx],
                                       shellCenters          = framesData['model_x'][idx],
                                       shapeArray            = framesData['shape_array'][idx] if shapeFunc else None,
                                       numberOfRemovedAtoms  = framesData['number_of_atoms_removed'][idx],
                                       standardError         = framesData['standard_error'][idx],
                                       scaleFactor           = framesData['scale_factor'][idx],
                                       multiframeWeight      = framesData['frames_weight'][idx],
                                       ax=axes[idx], intra=intra, inter=inter,
                                       xlabelParams = xlabelParams,
                                       ylabelParams = ylabelParams,
                                       legendParams = legendParams,
                                       expParams    = expParams,
                                       totParams    = totParams,
                                       noWParams    = noWParams,
                                       shaParams    = shaParams,
                                       parParams    = parParams,
                                       titleFormat  = titleFormat,
                                       show=_show)
        else:
            FIG, AXES = self._plot(frame                 = frame,
                                   output                = framesData['weighted_output'],
                                   experimentalX         = framesData['experimental_x'][0],
                                   experimentalY         = framesData['experimental_y'][0],
                                   shellCenters          = framesData['model_x'][0],
                                   shapeArray            = framesData['shape_array'][0] if shapeFunc else None,
                                   numberOfRemovedAtoms  = framesData['weighted_number_of_atoms_removed'],
                                   standardError         = framesData['weighted_standard_error'],
                                   scaleFactor           = framesData['weighted_scale_factor'],
                                   multiframeWeight      = np.sum(framesData['frames_weight']),
                                   ax=ax, intra=intra, inter=inter,
                                   xlabelParams = xlabelParams,
                                   ylabelParams = ylabelParams,
                                   legendParams = legendParams,
                                   expParams    = expParams,
                                   totParams    = totParams,
                                   noWParams    = noWParams,
                                   shaParams    = shaParams,
                                   parParams    = parParams,
                                   titleFormat  = titleFormat,
                                   show=show)
        # return
        return FIG, AXES


    def plot_multiframe_weights(self, frame, ax=None, titleFormat = "@{frame} [subframes probability distribution]", show=True):
        """ plot multiframe subframes weight distribution histogram

        :Parameters:
           #. frame (string): multiframe name
           #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
              If None is given a new plot figure will be created.
           #. titleFormat (string): title format. If empty string is given no
              title will be added to figure axes
           #. show (boolean): Whether to render and show figure before
               returning.

        :Returns:
            #. figure (matplotlib Figure): matplotlib used figure.
            #. axes (matplotlib Axes): matplotlib used axes.

        """
        # import matplotlib
        import matplotlib.pyplot as plt
        # check frame
        if frame is None:
            frame = self.engine.usedFrame
        isNormalFrame, isMultiframe, isSubframe = self.engine.get_frame_category(frame=frame)
        assert isMultiframe, "Given frame '%s' is not a multiframe"%frame
        # get axes
        if ax is None:
            FIG  = plt.figure()
            AXES = plt.gca()
        else:
            AXES = ax
            FIG  = AXES.get_figure()
        #
        data   = self.get_constraint_data_dictionary(frame=frame)
        totals = [i['total'] for i in data['output']]
        frames = data['frames_name']
        stderr = data['standard_error']
        ratios = data['frames_weight']
        # plot bars
        bars = plt.bar(range(len(frames)), ratios, align='center', alpha=0.5)
        #plt.xticks(range(len(frames)), frames, rotation=90)
        plt.xticks(range(len(frames)), self.engine.frames[frame]['frames_name'])
        # plot weights text
        pos  = plt.gca().get_xticks()
        ylim = plt.gca().get_ylim()
        diff = 0.05*abs(ylim[1] - ylim[0])
        for i in range(len(ratios)):
            y = ratios[i]+diff
            if y < 0.35*ylim[1]:
                y = 0.35*ylim[1]
            if y >  0.65*ylim[1]:
                y = 0.65*ylim[1]
            plt.text(x=pos[i] , y=max(0.35*ylim[1], ratios[i]+diff), s='%.4f (%.2f)'%(ratios[i],stderr[i]), size = 10, ha='center', va='top', rotation=90)
        # set labels
        plt.ylabel('Coefficient')
        plt.xlabel('Sub-frames')
        # set title
        if len(titleFormat):
            name = ' '.join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__))
            FIG.canvas.set_window_title(name)
            # format title
            title = titleFormat.format(frame=frame)
            AXES.set_title(title)
        # show
        if show:
            plt.show()
        # return
        return FIG, AXES

    def _export(self, fileName, framesData, format='%12.5f', delimiter='\t', comments='# '):
        # create data, metadata and header
        metadata = []
        header   = []
        data     = []
        if 'frames_name' in framesData:
            metadata = ["Frames name:                    %s"%framesData['frames_name'],
                        "frames number of removed atoms: %s"%dict(list(zip(framesData['frames_name'],framesData['number_of_atoms_removed']))),
                        "frames multiframe weights:      %s"%dict(list(zip(framesData['frames_name'],framesData['frames_weight']))),
                        "Frames scale factor:            %s"%dict(list(zip(framesData['frames_name'],framesData['scale_factor']))),
                        "Frames standard error:          %s"%dict(list(zip(framesData['frames_name'],framesData['standard_error']))),
                        ]
            for idx, frm in enumerate(framesData['frames_name']):
                # append experimental distances
                header.append('%s:experimental_x'%frm)
                data.append(framesData['experimental_x'][idx])
                # append experimental data
                header.append('%s:experimental_y'%frm)
                data.append(framesData['experimental_y'][idx])
                # append experimental data
                header.append('%s:model_x'%frm)
                data.append(framesData['model_x'][idx] )
                # loop all outputs
                output = framesData['output'][idx]
                for dn in output:
                    header.append('%s:%s'%(frm,dn))
                    data.append(output[dn])
        # append weighted data
        if len([k for k in framesData if k.startswith('weighted_')]):
            metadata.append("Weighted number of removed atoms: %s"%framesData['weighted_number_of_atoms_removed'])
            metadata.append("Weighted frames scale factor:     %s"%framesData['weighted_scale_factor'])
            metadata.append("Weighted frames standard error:   %s"%framesData['weighted_standard_error'])
            output = framesData['weighted_output']
            for dn in output:
                header.append('weighted_%s'%(dn))
                data.append(output[dn])
        # finalize metadata
        metadata.append(" ".join(header))
        # create array and export
        data = np.transpose(data).astype(float)
        # save
        np.savetxt(fname     = fileName,
                   X         = data,
                   fmt       = format,
                   delimiter = delimiter,
                   header    = "\n".join(metadata),
                   comments  = comments)


    def export(self, fileName, frame=None, splitFrames=False, format='%12.5f', delimiter=' ', comments='# '):
        """
        Export constraint data to text file or to an archive of files.

        :Parameters:
            #. fileName (path): full file name and path.
            #. frame (None, string): frame name to export data from. If multiframe
               is given, multiple files will be created with subframe name
               appended to the end.
            #. splitFrames (boolean): whether to split frames into multiple
               files and create and archive (.zip) of frames data. If given
               frame is a normal frame or a subframe this option is not used.
            #. format (string): string format to export the data.
               format is as follows (%[flag]width[.precision]specifier)
            #. delimiter (string): String or character separating columns.
            #. comments (string): String that will be prepended to the header.
        """
        if frame is None:
            frame = self.engine.usedFrame
        framesData = self.get_constraint_data_dictionary(frame=frame)
        if framesData is None:
            return
        # get number of frames
        if len(framesData['frames_name'])==1 or not splitFrames:
            self._export(fileName = fileName,
                         framesData = framesData,
                         format=format, delimiter=delimiter,
                         comments=comments )
        else:
            dirname, filename   = os.path.split(fileName)
            if not len(dirname):
                dirname = os.getcwd()
            filename, extension = os.path.splitext(filename)
            zipFilePath = os.path.join(dirname, filename)
            if os.path.isfile( zipFilePath+'.zip' ):
                LOGGER.warn("File '%s' is removed and replaced with new one"%(zipFilePath,))
            _dirname = os.path.join(dirname, '.'+str(uuid.uuid1()))
            os.makedirs(_dirname)
            try:
                for idx, frameName in enumerate(framesData['frames_name']):
                    _framesData = dict([(key,[framesData[key][idx]]) for key in framesData if not key.startswith('weighted_')])
                    self._export(framesData=_framesData,
                                 fileName=os.path.join(_dirname,'%s.txt'%frameName.replace(os.sep, '_')),
                                 format=format, delimiter=delimiter,
                                 comments=comments )
                # exported weighted
                _framesData = dict([(key,framesData[key]) for key in framesData if key.startswith('weighted_')])
                self._export(framesData=_framesData,
                             fileName=os.path.join(_dirname,'%s_weighted.txt'%frame.replace(os.sep, '_')),
                             format=format, delimiter=delimiter,
                             comments=comments )
                # create zip file
                shutil.make_archive(zipFilePath, 'zip', _dirname)
            except Exception as err:
                print(Exception(err))
            finally:
                # remove directory
                shutil.rmtree(_dirname)



class SingularConstraint(Constraint):
    """ A singular constraint is a constraint that doesn't allow multiple
    instances in the same engine."""

    def is_singular(self, engine):
        """
        Get whether only one instance of this constraint type is present
        in the stochastic engine. True for only itself found, False for
        other instance of the same __class__.__name__ or constraintId.

        :Parameters:
            #. engine (stochastic fullrmc engine): Engine instance.

        :Returns:
            #. result (boolean): Whether constraint is singular in engine.
        """
        for c in engine.constraints:
            if c is self or c.constraintId==self.constraintId:
                continue
            if c.__class__.__name__ == self.__class__.__name__:
                return False
        return True

    def assert_singular(self, engine):
        """
        Checks whether only one instance of this constraint type is
        present in the stochastic engine. Raises Exception if multiple
        instances are present.
        """
        assert self.is_singular(engine), LOGGER.error("Only one instance of constraint '%s' is allowed in the same engine"%self.__class__.__name__)


class RigidConstraint(Constraint):
    """
    A rigid constraint is a constraint that doesn't count into the total
    standard error of the stochastic Engine. But it's internal standard error
    must monotonously decrease or remain the same from one engine step to
    another. If standard error of an rigid constraint increases the
    step will be rejected even before engine's new standardError get computed.

    :Parameters:
        #. rejectProbability (Number): Rejecting probability of all steps
           where standard error increases. It must be between 0 and 1 where
           1 means rejecting all steps where standardError increases
           and 0 means accepting all steps regardless whether standard error
           increases or not.
    """
    def __init__(self, rejectProbability):
        # initialize constraint
        super(RigidConstraint, self).__init__()
        # set probability
        self.set_reject_probability(rejectProbability)
        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_RigidConstraint__rejectProbability'] )
        object.__setattr__(self, 'FRAME_DATA',  tuple(FRAME_DATA) )

    @property
    def rejectProbability(self):
        """ Rejection probability. """
        return self.__rejectProbability

    def set_reject_probability(self, rejectProbability):
        """
        Set the rejection probability. This method doesn't allow specifying
        frames. It will target used frame only.

        :Parameters:
            #. rejectProbability (Number): rejecting probability of all steps
               where standard error increases. It must be between 0 and 1
               where 1 means rejecting all steps where standardError increases
               and 0 means accepting all steps regardless whether standard
               error increases or not.
        """
        assert is_number(rejectProbability), LOGGER.error("rejectProbability must be a number")
        rejectProbability = FLOAT_TYPE(rejectProbability)
        assert rejectProbability>=0 and rejectProbability<=1, LOGGER.error("rejectProbability must be between 0 and 1")
        self.__rejectProbability = rejectProbability
        # dump to repository
        self._dump_to_repository({'_RigidConstraint__rejectProbability': self.__rejectProbability})

    def should_step_get_rejected(self, standardError):
        """
        Given a standard error, return whether to keep or reject new
        standard error according to the constraint reject probability.

        :Parameters:
            #. standardError (number): The standard error to compare with
            the Constraint standard error

        :Return:
            #. result (boolean): True to reject step, False to accept
        """
        if self.standardError is None:
            raise Exception(LOGGER.error("must compute data first"))
        if standardError<=self.standardError:
            return False
        return randfloat() < self.__rejectProbability

    def should_step_get_accepted(self, standardError):
        """
        Given a standard error, return whether to keep or reject new standard
        error according to the constraint reject probability.

        :Parameters:
            #. standardError (number): The standard error to compare with
               the Constraint standard error

        :Return:
            #. result (boolean): True to accept step, False to reject
        """
        return not self.should_step_get_reject(standardError)


#class QuasiRigidConstraint(RigidConstraint):
#    """
#    A quasi-rigid constraint is a another rigid constraint but it becomes free
#    above a certain threshold ratio of satisfied data. Every quasi-rigid
#    constraint has its own definition of maximum standard error. The ratio is
#    computed as between current standard error and maximum standard error.
#
#     .. math::
#
#        ratio = 1-\\frac{current\ standard\ error}{maximum\ standard\ error}
#
#    :Parameters:
#        #. rejectProbability (Number): Rejecting probability of all steps
#           where standardError increases only before threshold ratio is reached.
#           It must be between 0 and 1 where 1 means rejecting all steps where
#           standardError increases and 0 means accepting all steps regardless
#           whether standardError increases or not.
#        #. thresholdRatio(Number): The threshold of satisfied data, above
#           which the constraint become free. It must be between 0 and 1 where
#           1 means all data must be satisfied and therefore the constraint
#           behave like a RigidConstraint and 0 means none of the data must be
#           satisfied and therefore the constraint becomes always free and
#           useless.
#    """
#    def __init__(self, engine, rejectProbability, thresholdRatio):
#        # initialize constraint
#        super(QuasiRigidConstraint, self).__init__(rejectProbability=rejectProbability)
#        # set probability
#        self.set_threshold_ratio(thresholdRatio)
#        # initialize maximum standard error
#        #self.__maximumStandardError = None
#        self._set_maximum_standard_error(None)
#        # set frame data
#        FRAME_DATA = [d for d in self.FRAME_DATA]
#        FRAME_DATA.extend(['_QuasiRigidConstraint__thresholdRatio',
#                           '_QuasiRigidConstraint__maximumStandardError'] )
#        object.__setattr__(self, 'FRAME_DATA',  tuple(FRAME_DATA) )
#
#
#    def _set_maximum_standard_error(self, maximumStandardError):
#        """ Set the maximum standard error. Use carefully, it's not meant to
#        be used externally. maximum squared deviation is what is used to
#        compute the ratio and compare to threshold ratio.
#        """
#        if maximumStandardError is not None:
#            assert is_number(maximumStandardError), LOGGER.error("maximumStandardError must be a number.")
#            maximumStandardError = FLOAT_TYPE(maximumStandardError)
#            assert maximumStandardError>0, LOGGER.error("maximumStandardError must be a positive.")
#        self.__maximumStandardError = maximumStandardError
#        # dump to repository
#        self._dump_to_repository({'_QuasiRigidConstraint__maximumStandardError': self.__maximumStandardError})
#
#    @property
#    def thresholdRatio(self):
#        """ Threshold ratio. """
#        return self.__thresholdRatio
#
#    @property
#    def currentRatio(self):
#        return 1-(self.standardError/self.__maximumStandardError)
#
#    def set_threshold_ratio(self, thresholdRatio):
#        """
#        Set the rejection probability function.
#
#        :Parameters:
#            #. thresholdRatio(Number): The threshold of satisfied data, above
#               which the constraint become free. It must be between 0 and 1
#               where 1 means all data must be satisfied and therefore the
#               constraint behave like a RigidConstraint and 0 means none of
#               the data must be satisfied and therefore the constraint
#               becomes always free and useless.
#        """
#        assert is_number(thresholdRatio), LOGGER.error("thresholdRatio must be a number")
#        thresholdRatio = FLOAT_TYPE(thresholdRatio)
#        assert thresholdRatio>=0 and thresholdRatio<=1, LOGGER.error("thresholdRatio must be between 0 and 1")
#        self.__thresholdRatio = thresholdRatio
#        # dump to repository
#        self._dump_to_repository({'_QuasiRigidConstraint__thresholdRatio': self.__thresholdRatio})
#
#    def should_step_get_rejected(self, standardError):
#        """
#        Given a standard error, return whether to keep or reject new
#        standard error according to the constraint reject probability function.
#
#        :Parameters:
#            #. standardError (number): Standard error to compare with the
#            Constraint standard error.
#
#        :Return:
#            #. result (boolean): True to reject step, False to accept.
#        """
#        previousRatio = 1-(self.standardError/self.__maximumStandardError)
#        currentRatio  = 1-(standardError/self.__maximumStandardError)
#        if currentRatio>=self.__thresholdRatio: # must be accepted
#            return False
#        elif previousRatio>=self.__thresholdRatio: # it must be rejected
#            return randfloat() < self.rejectProbability
#        elif standardError<=self.standardError: # must be accepted
#            return False
#        else: # must be rejected
#            return randfloat() < self.rejectProbability
#
