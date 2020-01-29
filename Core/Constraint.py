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
from collections import OrderedDict
from datetime import datetime

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

    def _codify__(self, *args, **kwargs):
        raise Exception(LOGGER.impl("'%s' method must be overloaded"%inspect.stack()[0][3]))

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

    def _get_constraints_copy(self, frame):
        """Get constraint copy for given frame. This is meant to be used
        internally. If used wrong, engine values can be altered unvoluntarely.
        It's generally meant to be used for plot and export purposes.

        :Parameters:
            #. frame (string): can be a traditional frame a d subframe or
               a multiframe

        :Returns:
            #. constraintsLUT (dict): a dictionary where keys are the given frame and
               all subframes if a multiframe is given. Values are the
               constraint copy or the constraint itself for used frame
        """
        neededData = self._constraint_copy_needs_lut()
        isNormalFrame, isMultiframe, isSubframe = self.engine.get_frame_category(frame=frame)
        # get frames constraint data
        if isNormalFrame or isSubframe:
            frames       = [frame]
            framesName   = [frame,]
        else:
            frames       = [os.path.join(frame,frm) for frm in self.engine.frames[frame]['frames_name']]
            framesName   = self.engine.frames[frame]['frames_name']
        # build frame data
        constraintsLUT  = OrderedDict()
        repo            = self.engine._get_repository()
        for frm in frames:
            if not repo.is_repository_directory(os.path.join(frm, 'constraints',self.constraintName)):
                LOGGER.usage("@{frm} constraint '{cn}' not found in stochastic engine repository. This can happen for subframes or if frame has never been used, try calling stochastic engine 'set_used_frame(\"{frm}\")'".format(cn=self.constraintName,frm=frm))
                continue
            if frm == self.engine.usedFrame:
                _constraint = self
            else:
                _constraint = copy.deepcopy(self)
                object.__setattr__(_constraint.engine, '_Engine__usedFrame', frm)
                if not repo.is_repository_directory(frm):
                    LOGGER.usage("Frame '{frm}' not found in stochastic engine repository. This can happen if frame has never been used, try calling stochastic engine 'set_used_frame(\"{frm}\")'".format(frm=frm))
                    continue
                # load needed data to restore constraint defined in _constraint_copy_needs_lut
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
            constraintsLUT[frm] = _constraint
        # return
        return constraintsLUT

    def _get_constraints_data(self, frame):
        """Get constraint and data for given frame. This is meant to be used
        internally. If used wrong, engine values can be altered unvoluntarely.
        It's generally meant to be used for plot and export purposes.

        :Parameters:
            #. frame (string): can be a traditional frame a d subframe or
               a multiframe

        :Returns:
            #. dataLUT (dict): a dictionary where keys are the given frame and
               all subframes if a multiframe is given. Values are dictionaries
               of the constraint and data copy
        """
        dataLUT = self._get_constraints_copy(frame)
        for frm in dataLUT:
            _constraint = dataLUT[frm]
            _data       = _constraint.data
            if _data is None or _constraint.engine.state != _constraint.state:
                LOGGER.usage("Computing constraint '{name}' data @{frame} without updating nor altering constraint properties and stochastic engine repository files".format(name=self.constraintName, frame=frm))
                _data, _ = _constraint.compute_data(update=False)
            dataLUT[frm] = {'constraint':_constraint, 'data':_data}
        # return
        return dataLUT

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
        if self.engine is not None:
            print(self.engine)
            print(self.engine.usedFrame)
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

    def get_constraints_properties(self, frame):
        """
        Get a dictionary look up table of constraint's properties

        :Parameters:
            #. frame (string): frame to pull and build contraint data. It can
               be a traditional frame, a multiframe or a subframe

        :Returns:
            #. propertiesLUT (dictionary): properties value look up table. Keys
               are described herein. All keys start with 'frames-' and values
               are list of properties for every and each frame.

                * frames-name: list of all frames name
                * frames-weight: list of all frames weight
                * frames-number_of_removed_atoms: list of number of removed atoms from each frame
                * frames-constraint: list of constraint copy
                * frames-data: list of constraint data
                * frames-standard_error: list of all frames standard error
        """
        constraintsData = self._get_constraints_data(frame)
        # initialise properties LUT
        propertiesLUT = OrderedDict()
        propertiesLUT['frames-name']                      = []
        propertiesLUT['frames-weight']                    = []
        propertiesLUT['frames-number_of_removed_atoms']   = []
        propertiesLUT['frames-constraint']                = []
        propertiesLUT['frames-data']                      = []
        propertiesLUT['frames-standard_error']            = []
        # set properties LUT
        for frm in constraintsData:
            _constraint = constraintsData[frm]['constraint']
            _data       = constraintsData[frm]['data']
            _stdErr     = _constraint.compute_standard_error(_data)
            _weight     = _constraint.multiframeWeight if _constraint.multiframeWeight is not None else FLOAT_TYPE(1.0)
            _nRemAt     = len(_constraint.engine._atomsCollector)
            # append properties
            propertiesLUT['frames-name'].append(frm)
            propertiesLUT['frames-weight'].append(_weight)
            propertiesLUT['frames-number_of_removed_atoms'].append(_nRemAt)
            propertiesLUT['frames-constraint'].append(_constraint)
            propertiesLUT['frames-data'].append(_data)
            propertiesLUT['frames-standard_error'].append(_stdErr)
        # return
        return propertiesLUT

    def _constraint_copy_needs_lut(self, *args, **kwargs):
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def get_constraint_value(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def get_constraint_original_value(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_standard_error(self):
        """Method must be overloaded in children classes."""
        raise Exception(LOGGER.impl("%s '%s' method must be overloaded"%(self.__class__.__name__,inspect.stack()[0][3])))

    def compute_data(self, update=True):
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

    def export(self, fileName, frame=None, format='%s', delimiter='\t', comments='#'):
        """
        Export constraint data to text file or to an archive of files.

        :Parameters:
            #. fileName (path): full file name and path.
            #. frame (None, string): frame name to export data from. If multiframe
               is given, multiple files will be created with subframe name
               appended to the end.
            #. format (string): string format to export the data.
               format is as follows (%[flag]width[.precision]specifier)
            #. delimiter (string): String or character separating columns.
            #. comments (string): String that will be prepended to the header.
        """
        if frame is None:
            frame = self.engine.usedFrame
        # get constraint properties LUT
        propertiesLUT = self.get_constraints_properties(frame=frame)
        # get number of frames
        metadata = ["This file is auto-generated by fullrmc '%s' at %s"%(self.engine.version, datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')),
                    "Contains exported data of constraint '%s'"%self.__class__.__name__,
                    "For questions and concerns use fullrmc's forum http://bachiraoun.github.io/fullrmc/QAForum.html",
                    "",
                    "Engine name:                    %s"%os.path.basename(self.engine.path),
                    "Frames name:                    %s"%propertiesLUT['frames-name'],
                    "frames number of removed atoms: %s"%dict(list(zip(propertiesLUT['frames-name'],propertiesLUT['frames-number_of_removed_atoms']))),
                    "Frames standard error:          %s"%dict(list(zip(propertiesLUT['frames-name'],propertiesLUT['frames-standard_error']))),
                    ]
        headers = []
        data    = []
        for idx, frameName in enumerate(propertiesLUT['frames-name']):
            cons = propertiesLUT['frames-constraint'][idx]
            h, d = cons._get_export(frameIndex=idx, format=format,
                                    propertiesLUT = propertiesLUT)
            headers.append(h)
            data.append(d)
        # write constraint data
        lines = []
        lines.append( '\n'.join(["%s %s"%(comments,i) for i in metadata]) )
        for h,d,fn in zip(headers, data, propertiesLUT['frames-name']):
            lines.append('\n%s'%comments)
            h = delimiter.join(h)
            d = '\n'.join([delimiter.join(l) for l in d])
            lines.append('\n%s @%s'%(comments,fn))
            lines.append('\n%s %s'%(comments,h))
            lines.append('\n%s'%(d))
        lines = ''.join(lines)
        if fileName is not None:
            with open(fileName, 'w') as fd:
                fd.write(lines)
        # return
        return lines
        #with open(fileName, 'w') as fd:
        #    fd.write('\n'.join(["%s %s"%(comments,i) for i in metadata]))
        #    for h,d,fn in zip(headers, data, propertiesLUT['frames-name']):
        #        fd.write('\n%s'%comments)
        #        h = delimiter.join(h)
        #        #d = '\n'.join(['%s %s'%(' '*len(comments),delimiter.join(l)) for l in d])
        #        d = '\n'.join([delimiter.join(l) for l in d])
        #        fd.write('\n%s @%s'%(comments,fn))
        #        fd.write('\n%s %s'%(comments,h))
        #        fd.write('\n%s'%(d))


    def plot(self, frame=None, axes=None,
                   subAdParams  = {'left':None,'bottom':None,'right':None,'top':None,'wspace':None,'hspace':0.4},
                   dataParams   = {'label':'Y', 'linewidth':2},
                   xlabelParams = {'xlabel':'Core-Shell atoms', 'size':10},
                   ylabelParams = {'ylabel':'Coordination number', 'size':10},
                   xticksParams = {'fontsize': 8, 'rotation':90},
                   yticksParams = {'fontsize': 8, 'rotation':0},
                   legendParams = {'frameon':False, 'ncol':1, 'loc':'upper right', "fontsize":8},
                   titleParams  = {'label':"@{frame} (${numberOfRemovedAtoms:.1f}$ $rem.$ $at.$) $Std.Err.={standardError:.3f}$ - $used$ $({used})$", "fontsize":10},
                   gridParams   = None,
                   show=True, **paramsKwargs):
        """
        Plot constraint data. This can be overloaded in children classes.

        :Parameters:
            #. frame (None, string): The frame name to plot. If None, used frame
               will be plotted.
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If None is given a new plot figure will be created.
            #. subAdParams (None, dict): matplotlib.artist.Artist.subplots_adjust
               parameters subplots adjust parameters.
            #. dataParams (None, dict): constraint data plotting parameters
            #. xlabelParams (None, dict): matplotlib.axes.Axes.set_xlabel
               parameters.
            #. ylabelParams (None, dict): matplotlib.axes.Axes.set_ylabel
               parameters.
            #. legendParams (None, dict):matplotlib.axes.Axes.legend
               parameters.
            #. xticksParams (None, dict):matplotlib.axes.Axes.set_xticklabels
               parameters.
            #. yticksParams (None, dict):matplotlib.axes.Axes.set_yticklabels
               parameters.
            #. titleParams (None, dict): matplotlib.axes.Axes.set_title parameters
            #. gridParams (None, dict): matplotlib.axes.Axes.grid parameters
            #. show (boolean): Whether to render and show figure before
               returning.

        :Returns:
            #. figure (matplotlib Figure): matplotlib used figure.
            #. axes (matplotlib Axes): matplotlib used axes.
        """
        # get frame
        if frame is None:
            frame = self.engine.usedFrame
        import matplotlib.pyplot as plt
        # get properties look up table
        propertiesLUT  = self.get_constraints_properties(frame=frame)
        # get matplotlib axes
        numberOfFrames = len(propertiesLUT['frames-weight'])
        if numberOfFrames == 0:
            LOGGER.warn("@{frm} constraint '{cn}' not data found to plot. This can happen if constraint definition doesn't match the pdb atomic content".format(cn=self.constraintName,frm=frame))
            return
        nrows          = int(np.sqrt(float(numberOfFrames)))
        ncols          = int(np.ceil(float(numberOfFrames)/nrows))
        if numberOfFrames >1 or axes is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            fig.patch.set_facecolor('white')
            name = ' '.join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__))
            fig.canvas.set_window_title(name)
        else:
            fig = axes.get_figure()
        # flatten axes representation and remove unused ones
        if numberOfFrames>1:
            axes = axes.flatten()
            [a.remove() for a in axes[len(propertiesLUT['frames-name']):]]
        else:
            axes = [axes]
        # adjust subplots
        if subAdParams is not None:
            fig.subplots_adjust(**subAdParams)
        # plot all frames
        for idx in range(numberOfFrames):
            cons = propertiesLUT['frames-constraint'][idx]
            cons._plot(frameIndex    = idx,
                       propertiesLUT = propertiesLUT,
                       ax            = axes[idx],
                       dataParams    = dataParams,
                       xlabelParams  = xlabelParams,
                       ylabelParams  = ylabelParams,
                       xticksParams  = xticksParams,
                       yticksParams  = yticksParams,
                       legendParams  = legendParams,
                       titleParams   = titleParams,
                       gridParams    = gridParams,
                       **paramsKwargs)
        # show
        if show:
            plt.show()
        # return
        return fig, axes


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
            #. experimentalData (numpy.ndarray, string, list, tuple):
               Experimental data as numpy.ndarray or string path to load data
               using numpy.loadtxt method. If list or tuple are given, they
               will be automatically converted to a numpy array by calling
               numpy.array(experimentalData). Finally experimental data type
               will be converted to fullrmc.Globals.FLOAT_TYPE
        """
        assert self.engine is None, LOGGER.error("Experimental data must be set before engine is set") # ADDED 2018-11-21
        if isinstance(experimentalData, basestring):
            try:
                experimentalData = np.loadtxt(str(experimentalData), dtype=FLOAT_TYPE)
            except Exception as err:
                raise Exception(LOGGER.error("unable to load experimentalData path '%s' (%s)"%(experimentalData, err)))
        if isinstance(experimentalData, (list,tuple)):
            try:
                experimentalData = np.array(experimentalData)
            except Exception as err:
                raise Exception(LOGGER.error("unable to convert given experimentalData list to a numpy array (%s)"%(experimentalData, err)))
        else:
            assert isinstance(experimentalData, np.ndarray), LOGGER.error("experimentalData must be a numpy.ndarray or string path to load data using numpy.loadtxt.")
        # cast to FLOAT_TYPE
        try:
            experimentalData = experimentalData.astype(FLOAT_TYPE)
        except Exception as err:
            raise Exception(LOGGER.error("unable to cast experimentalData numpy array data type to fullrmc.Globals.FLOAT_TYPE '%s' (%s)"%(experimentalData, FLOAT_TYPE, err)))
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
            assert len(dataWeights) == self.__experimentalData.shape[0], LOGGER.error("dataWeights length '%i' must be equal to experimental data length '%i'"%(len(dataWeights),self.__experimentalData.shape[0]))
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


    def get_constraints_properties(self, frame):
        """
        Get a dictionary look up table of constraint's properties that
        are needed to plot or export

        :Parameters:
            #. frame (string): frame to pull and build contraint data. It can
               be a traditional frame, a multiframe or a subframe

        :Returns:
            #. propertiesLUT (dictionary): properties value look up table. Keys
               are described herein. Values of keys that start with 'frames-'
               are a list for all frames. Values of keys that start with
               'weighted-' are weighted values for all frames

                * frames-name: list of all frames name.
                * frames-weight: list of all frames weight.
                * frames-number_of_removed_atoms: list of number of removed atoms from
                  each frame.
                * frames-experimental_x: list of numpy array of experimental x data.
                * frames-experimental_y: list of numpy array of experimental y data.
                * frames-output: list of frames dictionary constraint output data
                * frames-model_x: list of numpy array of model x data.
                * frames-shape_array: list of system shape function (numpy array) of
                  all frames.
                * frames-window_function: list of window function (numpy array) of
                  all frames.
                * frames-scale_factor: list of all frames scale factor.
                * frames-standard_error: list of all frames standard error.
                * weighted-output: dictionary of all frames weighted constraint
                  data using 'frames-weight'
                * weighted-number_of_removed_atoms: All frames averaged number
                  of removed atoms using 'frames-weight'
                * weighted-scale_factor: All frames averaged scale factor
                  using 'frames-weight'
                * weighted-standard_error: All frames weighted standard error
                  using 'frames-weight'

        """
        # get properties LUT
        constraintsData = self._get_constraints_data(frame)
        # initialize properties look up table
        propertiesLUT = OrderedDict()
        propertiesLUT['frames-name']                      = []
        propertiesLUT['frames-weight']                    = []
        propertiesLUT['frames-number_of_removed_atoms']   = []
        propertiesLUT['frames-constraint']                = []
        propertiesLUT['frames-data']                      = []
        propertiesLUT['frames-standard_error']            = []
        propertiesLUT['frames-experimental_x']            = []
        propertiesLUT['frames-experimental_y']            = []
        propertiesLUT['frames-output']                    = []
        propertiesLUT['frames-model_x']                   = []
        propertiesLUT['frames-limits']                    = []
        propertiesLUT['frames-limit_index_start']         = []
        propertiesLUT['frames-limit_index_end']           = []
        propertiesLUT['frames-shape_array']               = []
        propertiesLUT['frames-window_function']           = []
        propertiesLUT['frames-scale_factor']              = []
        propertiesLUT['frames-standard_error']            = []
        propertiesLUT['weighted-output']                  = None
        propertiesLUT['weighted-number_of_removed_atoms'] = None
        propertiesLUT['weighted-scale_factor']            = None
        propertiesLUT['weighted-standard_error']          = None
        # set properties LUT
        for frm in constraintsData:
            _constraint = constraintsData[frm]['constraint']
            _data       = constraintsData[frm]['data']
            _weight     = _constraint.multiframeWeight if _constraint.multiframeWeight is not None else FLOAT_TYPE(1.0)
            _nRemAt     = len(_constraint.engine._atomsCollector)
            # compute properties
            _output   = _constraint._get_constraint_value(_data, applyMultiframePrior=False)
            _stdErr   = _constraint.compute_standard_error(modelData = _output["total"])
            _expDis   = _constraint._experimentalX
            _expData  = _constraint._experimentalY
            _modelX   = _constraint._modelX
            _weight   = _constraint.multiframeWeight if _constraint.multiframeWeight is not None else FLOAT_TYPE(1.0)
            _nRemAt   = len(_constraint.engine._atomsCollector)
            _sArray   = None
            _limits   = _constraint.limits
            _idxStart = _constraint.limitsIndexStart
            _idxEnd   = _constraint.limitsIndexEnd
            if hasattr(_constraint, '_shapeArray'):
                _sArray = _constraint._shapeArray
            _wFunc    = _constraint.windowFunction
            _sFactor  = _constraint.scaleFactor
            # append properties
            propertiesLUT['frames-name'].append(frm)
            propertiesLUT['frames-weight'].append(_weight)
            propertiesLUT['frames-number_of_removed_atoms'].append(_nRemAt)
            propertiesLUT['frames-constraint'].append(_constraint)
            propertiesLUT['frames-data'].append(_data)
            propertiesLUT['frames-standard_error'].append(_stdErr)
            propertiesLUT['frames-experimental_x'].append(_expDis)
            propertiesLUT['frames-experimental_y'].append(_expData)
            propertiesLUT['frames-output'].append(_output)
            propertiesLUT['frames-limits'].append(_limits)
            propertiesLUT['frames-limit_index_start'].append(_idxStart)
            propertiesLUT['frames-limit_index_end'].append(_idxEnd)
            propertiesLUT['frames-model_x'].append(_modelX)
            propertiesLUT['frames-shape_array'].append(_sArray)
            propertiesLUT['frames-window_function'].append(_wFunc)
            propertiesLUT['frames-scale_factor'].append(_sFactor)
            propertiesLUT['frames-standard_error'].append(_stdErr)
        ## warn about data limit difference
        _setStart  = set(propertiesLUT['frames-limit_index_start'])
        _setEnd    = set(propertiesLUT['frames-limit_index_end'])
        _setLimits = set(propertiesLUT['frames-limits'])
        _allMatch  = len(_setStart)==len(_setEnd)==len(_setLimits)==1
        if not _allMatch:
            LOGGER.warn("@{frame} constraint '{cn}' used data limits '{lim}' are not unique for all subrames. Mesoscopic structure weighting will be performed by data point".format(cn=self.constraintName, frame=frame, lim=list(_setLimits)))
        ## compute frames weighted
        framesWeight = np.array(propertiesLUT['frames-weight'], dtype=FLOAT_TYPE)
        if np.sum(framesWeight)==0:
            if len(framesWeight)==1:
                framesWeight = np.array([1])
            else:
                raise Exception("Frames weight sum is found to be 0. PLEASE REPORT")
        weightedOutput = {}
        for key in propertiesLUT['frames-output'][0]:
            if any([propertiesLUT['frames-output'][i][key] is None for i, w in enumerate(framesWeight)]):
                weightedOutput[key] = None
                LOGGER.warn("@{frame} constraint '{cn}' mesoscopic structure building is skipped for output key '{key}' because it's not computed for all subframes".format(cn=self.constraintName, frame=frame, key=key))
            elif _allMatch:
                weightedOutput[key] = np.sum([w*propertiesLUT['frames-output'][i][key] for i, w in enumerate(framesWeight)], axis=0)
                weightedOutput[key] = weightedOutput[key]/sum(framesWeight)
            else:
                _len =  max(_setEnd)
                _out = np.zeros(_len+1)
                _pws = np.zeros(_len+1)
                for i, w in enumerate(framesWeight):
                    if w == 0:
                        continue
                    _s = propertiesLUT['frames-limit_index_start'][i]
                    _e = propertiesLUT['frames-limit_index_end'][i]
                    _w = np.zeros(_len+1)
                    _w[_s:_e+1]    = w
                    _pws[_s:_e+1] += w
                    _out[_s:_e+1] += w*propertiesLUT['frames-output'][i][key]
                assert not len(np.where(_pws==0)[0]), LOGGER.error("@{frame} constraint '{cn}' output key '{key}' data range weights contain 0 values. PLEASE REPORT".format(cn=self.constraintName, frame=frame, key=key))
                weightedOutput[key] = _out/_pws
        propertiesLUT['weighted-output']                  = weightedOutput
        propertiesLUT['weighted-number_of_removed_atoms'] = np.average(propertiesLUT['frames-number_of_removed_atoms'], weights=framesWeight)
        propertiesLUT['weighted-standard_error']          = _constraint.compute_standard_error(modelData=weightedOutput['total'])
        propertiesLUT['weighted-scale_factor']            = np.average(propertiesLUT['frames-scale_factor'], weights=framesWeight)
        # return
        return propertiesLUT


    def _plot(self, frame, output, experimentalX, experimentalY,
                    shellCenters, shapeArray, numberOfRemovedAtoms,
                    standardError, scaleFactor, multiframeWeight,
                    # plotting arguments
                    ax, intra=True, inter=True, totalNoWindow=False,
                    xlabelParams=None,
                    ylabelParams=None,
                    legendParams=None,
                    titleParams = "",
                    expParams = {'label':"experimental","color":'red','marker':'o','markersize':7.5, 'markevery':1, 'zorder':0},
                    totParams = {'label':"total", 'color':'black','linewidth':3.0, 'zorder':1},
                    noWParams = {'label':"total - no window", 'color':'black','linewidth':1.0, 'zorder':1},
                    shaParams = {'label':"shape function", 'color':'black','linewidth':1.0, 'linestyle':'dashed'},
                    parParams = {'linewidth':1.0, 'markevery':5, 'markersize':5, 'zorder':-1},
                    gridParams= None,
                    show=True):
        # import matplotlib
        import matplotlib.pyplot as plt
        # Create plotting styles
        COLORS  = ["b",'g','c','y','m']
        MARKERS = ["",'.','+','^','|']
        INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], list(reversed(COLORS)))]
        INTRA_STYLES = [r[0] + r[1]for r in itertools.product(MARKERS, INTRA_STYLES)]
        INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], COLORS)]
        INTER_STYLES = [r[0] + r[1]for r in itertools.product(MARKERS, INTER_STYLES)]
        # plot experimental
        ax.plot(experimentalX,experimentalY, **expParams)
        ax.plot(shellCenters, output["total"], **totParams )
        if totalNoWindow and output["total_no_window"] is not None:
            ax.plot(shellCenters, output["total_no_window"], **noWParams )
        if shapeArray is not None:
            ax.plot(shellCenters, shapeArray, **shaParams )
        # plot partials
        intraStyleIndex = 0
        interStyleIndex = 0
        for key in output:
            val = output[key]
            if key in ("total", "total_no_window"):
                continue
            elif "intra" in key and intra:
                ax.plot(shellCenters, val, INTRA_STYLES[intraStyleIndex], label=key, **parParams )
                intraStyleIndex+=1
            elif "inter" in key and inter:
                ax.plot(shellCenters, val, INTER_STYLES[interStyleIndex], label=key, **parParams )
                interStyleIndex+=1
        # plot legend
        if legendParams is not None:
            ax.legend(**legendParams)
        # set title
        # set title
        if titleParams is not None:
            title = copy.deepcopy(titleParams)
            label = title.pop('label',"").format(frame=frame,
                                                 standardError=standardError,
                                                 numberOfRemovedAtoms=numberOfRemovedAtoms,
                                                 scaleFactor=scaleFactor,
                                                 used=self.used,
                                                 multiframeWeight=multiframeWeight)
            ax.set_title(label=label, **title)

        # set axis labels
        if xlabelParams is not None:
            ax.set_xlabel(**xlabelParams)
        if ylabelParams is not None:
            ax.set_ylabel(**ylabelParams)
        # grid parameters
        if gridParams is not None:
            gp = copy.deepcopy(gridParams)
            axis = gp.pop('axis', 'both')
            if axis is None:
                axis = 'both'
            ax.grid(axis=axis, **gp)


    def plot(self, frame=None, axes = None, asMesoscopic=False,
                   intra=True, inter=True, shapeFunc=True,
                   subAdParams  = {'left':None,'bottom':None,'right':None,'top':None,'wspace':None,'hspace':0.4},
                   totParams    = {'label':"total", 'color':'black','linewidth':3.0, 'zorder':1},
                   expParams    = {'label':"experimental","color":'red','marker':'o','markersize':7.5, 'markevery':1, 'zorder':0},
                   noWParams    = {'label':"total - no window", 'color':'black','linewidth':1.0, 'zorder':1},
                   shaParams    = {'label':"shape function", 'color':'black','linewidth':1.0, 'linestyle':'dashed', 'zorder':2},
                   parParams    = {'linewidth':1.0, 'markevery':5, 'markersize':5, 'zorder':3},
                   xlabelParams = {'xlabel':'X', 'size':10},
                   ylabelParams = {'ylabel':'Y', 'size':10},
                   xticksParams = {'fontsize': 8, 'rotation':0},
                   yticksParams = {'fontsize': 8, 'rotation':0},
                   legendParams = {'frameon':False, 'ncol':2, 'loc':'upper right', "fontsize":8},
                   titleParams  = {'label':"@{frame} (${numberOfRemovedAtoms:.1f}$ $rem.$ $at.$) $Std.Err.={standardError:.3f}$\n$scale$ $factor$=${scaleFactor:.2f}$ - $multiframe$ $weight$=${multiframeWeight:.3f}$ - $used$ $({used})$", "fontsize":10},
                   gridParams   = None,
                   show=True, **paramsKwargs):
        """
        Plot constraint data. This can be overloaded in children classes.

        :Parameters:
            #. frame (None, string): The frame name to plot. If None, used frame
               will be plotted.
            #. axes (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If None is given a new plot figure will be created.
            #. asMesoscopic (boolean): If given frame is a multiframe is,
               when true, asMesoscopic considers all frames as a statistical
               average of all frames in a mesoscopic system. All subframes
               will be then plotted as a single weighted mesoscopic structure.
               If asMesocopic is False and given frame is a multiframe
               given ax will be disregarded
            #. intra (boolean): Whether to add intra-molecular pair
               distribution function features to the plot.
            #. inter (boolean): Whether to add inter-molecular pair
               distribution function features to the plot.
            #. shapeFunc (boolean): Whether to add shape function to the plot
               only when exists.
            #. subAdParams (None, dict): matplotlib.artist.Artist.subplots_adjust
               parameters subplots adjust parameters.
            #. totParams (None, dict): constraint total plotting parameters
            #. expParams (None, dict): constraint experimental data parameters
            #. noWParams (None, dict): constraint total without window parameters
            #. shaParams (None, dict): constraint shape function parameters
            #. parParams (None, dict): constraint partials parameters
            #. xlabelParams (None, dict): matplotlib.axes.Axes.set_xlabel
               parameters.
            #. ylabelParams (None, dict): matplotlib.axes.Axes.set_ylabel
               parameters.
            #. legendParams (None, dict):matplotlib.axes.Axes.legend
               parameters.
            #. xticksParams (None, dict):matplotlib.axes.Axes.set_xticklabels
               parameters.
            #. yticksParams (None, dict):matplotlib.axes.Axes.set_yticklabels
               parameters.
            #. titleParams (None, dict): matplotlib.axes.Axes.set_title parameters
            #. gridParams (None, dict): matplotlib.axes.Axes.grid parameters
            #. show (boolean): Whether to render and show figure before
               returning.

        :Returns:
            #. figure (matplotlib Figure): matplotlib used figure.
            #. axes (matplotlib Axes): matplotlib used axes.
        """
        assert isinstance(asMesoscopic, bool), LOGGER.error('asMesoscopic must be boolean')
        # get frame
        if frame is None:
            frame = self.engine.usedFrame
        import matplotlib.pyplot as plt
        # get properties look up table
        propertiesLUT  = self.get_constraints_properties(frame=frame)
        # reset asMesoscopic
        asMesoscopic = asMesoscopic and len(propertiesLUT['frames-name'])>1
        # get matplotlib axes
        if not asMesoscopic:
            numberOfFrames = len(propertiesLUT['frames-weight'])
            nrows          = int(np.sqrt(float(numberOfFrames)))
            ncols          = int(np.ceil(float(numberOfFrames)/nrows))
        else:
            numberOfFrames = nrows = ncols = 1
        if numberOfFrames >1 or axes is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            fig.patch.set_facecolor('white')
            name = ' '.join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__))
            fig.canvas.set_window_title(name)
        else:
            fig = axes.get_figure()
        # flatten axes representation and remove unused ones
        if numberOfFrames>1:
            axes = axes.flatten()
            [a.remove() for a in axes[len(propertiesLUT['frames-name']):]]
        else:
            axes = [axes]
        # adjust subplots
        if subAdParams is not None:
            fig.subplots_adjust(**subAdParams)
        if not asMesoscopic:
            for idx in range(len(propertiesLUT['frames-name'])):
                cons = propertiesLUT['frames-constraint'][idx]
                cons._plot(frame                 = propertiesLUT['frames-name'][idx],
                           output                = propertiesLUT['frames-output'][idx],
                           experimentalX         = propertiesLUT['frames-experimental_x'][idx],
                           experimentalY         = propertiesLUT['frames-experimental_y'][idx],
                           shellCenters          = propertiesLUT['frames-model_x'][idx],
                           shapeArray            = propertiesLUT['frames-shape_array'][idx] if shapeFunc else None,
                           numberOfRemovedAtoms  = propertiesLUT['frames-number_of_removed_atoms'][idx],
                           standardError         = propertiesLUT['frames-standard_error'][idx],
                           scaleFactor           = propertiesLUT['frames-scale_factor'][idx],
                           multiframeWeight      = propertiesLUT['frames-weight'][idx],
                           ax=axes[idx], intra=intra, inter=inter,
                           xlabelParams = xlabelParams,
                           ylabelParams = ylabelParams,
                           legendParams = legendParams,
                           expParams    = expParams,
                           totParams    = totParams,
                           noWParams    = noWParams,
                           shaParams    = shaParams,
                           parParams    = parParams,
                           gridParams   = gridParams,
                           titleParams  = titleParams)
        else:
            propertiesLUT['frames-experimental_x']
            lenT = len(propertiesLUT['weighted-output']['total'])
            idx  = [i for i,d in enumerate(propertiesLUT['frames-experimental_x']) if len(d)==lenT][0]
            self._plot(frame                 = frame,
                       output                = propertiesLUT['weighted-output'],
                       experimentalX         = propertiesLUT['frames-experimental_x'][idx],
                       experimentalY         = propertiesLUT['frames-experimental_y'][idx],
                       shellCenters          = propertiesLUT['frames-model_x'][idx],
                       shapeArray            = None,
                       numberOfRemovedAtoms  = propertiesLUT['weighted-number_of_removed_atoms'],
                       standardError         = propertiesLUT['weighted-standard_error'],
                       scaleFactor           = propertiesLUT['weighted-scale_factor'],
                       multiframeWeight      = np.sum(propertiesLUT['frames-weight']),
                       ax=axes[0], intra=intra, inter=inter,
                       xlabelParams = xlabelParams,
                       ylabelParams = ylabelParams,
                       legendParams = legendParams,
                       expParams    = expParams,
                       totParams    = totParams,
                       noWParams    = noWParams,
                       shaParams    = shaParams,
                       parParams    = parParams,
                       gridParams   = gridParams,
                       titleParams  = titleParams)
        # show
        if show:
            plt.show()
        # return
        return fig, axes


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
        # get constraint properties LUT
        propertiesLUT   = self.get_constraints_properties(frame=frame)
        totals = [i['total'] for i in propertiesLUT['frames-output']]
        frames = propertiesLUT['frames-name']
        stderr = propertiesLUT['frames-standard_error']
        ratios = propertiesLUT['frames-weight']
        # plot bars
        bars = plt.bar(range(len(frames)), ratios, align='center', alpha=0.5)
        #plt.xticks(range(len(frames)), frames, rotation=90)
        plt.xticks(range(len(frames)), self.engine.frames[frame]['frames-name'])
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


    def _get_export(self, propertiesLUT, format='%12.5f'):
        # create data, metadata and header
        header   = []
        data     = []
        if 'frames-name' in propertiesLUT:
            for idx, frm in enumerate(propertiesLUT['frames-name']):
                # append experimental distances
                header.append('@%s:experimental_x'%frm)
                data.append(propertiesLUT['frames-experimental_x'][idx])
                # append experimental data
                header.append('@%s:experimental_y'%frm)
                data.append(propertiesLUT['frames-experimental_y'][idx])
                # append experimental data
                header.append('@%s:model_x'%frm)
                data.append(propertiesLUT['frames-model_x'][idx] )
                # loop all outputs
                output = propertiesLUT['frames-output'][idx]
                for dn in output:
                    header.append('%s:%s'%(frm,dn))
                    data.append(output[dn])
        # append weighted data
        if len([k for k in propertiesLUT if k.startswith('weighted-')]):
            for dn in output:
                header.append('mesoscopic-%s'%(dn))
                data.append(output[dn])
        # adjust data length
        maxLen = max( [len(d) for d in data] )
        for idx, d in enumerate(data):
            d = [format%i for i in d]
            lenDiff = maxLen - len(d)
            if lenDiff>0:
                d += ['']*lenDiff
            data[idx] = d
        # transpose data
        tdata = []
        for idx in range(maxLen):
            tdata.append( [d[idx] for d in data ] )
        # return
        return header, tdata


    def export(self, fileName, frame=None, format='%12.5f', delimiter='\t', comments='#'):
        """
        Export constraint data to text file or to an archive of files.

        :Parameters:
            #. fileName (path): full file name and path.
            #. frame (None, string): frame name to export data from. If multiframe
               is given, multiple files will be created with subframe name
               appended to the end.
            #. format (string): string format to export the data.
               format is as follows (%[flag]width[.precision]specifier)
            #. delimiter (string): String or character separating columns.
            #. comments (string): String that will be prepended to the header.
        """
        if frame is None:
            frame = self.engine.usedFrame
        # get constraint properties LUT
        propertiesLUT = self.get_constraints_properties(frame=frame)
        # get metadata
        metadata = ["This file is auto-generated by fullrmc '%s' at %s"%(self.engine.version, datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')),
                    "Contains exported data of constraint '%s'"%self.__class__.__name__,
                    "For questions and concerns use fullrmc's forum http://bachiraoun.github.io/fullrmc/QAForum.html",
                    "",
                    "Engine name:                    %s"%os.path.basename(self.engine.path),
                    "Frames name:                    %s"%propertiesLUT['frames-name'],
                    "Frames number of removed atoms: %s"%dict(list(zip(propertiesLUT['frames-name'],propertiesLUT['frames-number_of_removed_atoms']))),
                    "Frames multiframe weights:      %s"%dict(list(zip(propertiesLUT['frames-name'],propertiesLUT['frames-weight']))),
                    "Frames scale factor:            %s"%dict(list(zip(propertiesLUT['frames-name'],propertiesLUT['frames-scale_factor']))),
                    "Frames standard error:          %s"%dict(list(zip(propertiesLUT['frames-name'],propertiesLUT['frames-standard_error']))),
                    ]
        if len([k for k in propertiesLUT if k.startswith('weighted-')]):
            metadata.append("Mesoscopic number of removed atoms: %s"%propertiesLUT['weighted-number_of_removed_atoms'])
            metadata.append("Mesoscopic frames scale factor:     %s"%propertiesLUT['weighted-scale_factor'])
            metadata.append("Mesoscopic frames standard error:   %s"%propertiesLUT['weighted-standard_error'])
        # get header and data
        header, data = self._get_export(propertiesLUT = propertiesLUT,format=format)
        # write constraint data
        lines = []
        lines.append( '\n'.join(["%s %s"%(comments,i) for i in metadata]) )
        lines.append( '\n%s'%comments )
        h = delimiter.join(header)
        d = '\n'.join([delimiter.join(l) for l in data])
        lines.append( '\n%s %s'%(comments,h) )
        lines.append( '\n%s'%(d) )
        lines = ''.join(lines)
        if fileName is not None:
            with open(fileName, 'w') as fd:
                fd.write(lines)
        return lines
        #with open(fileName, 'w') as fd:
            #fd.write('\n'.join(["%s %s"%(comments,i) for i in metadata]))
            #fd.write('\n%s'%comments)
            #h = delimiter.join(header)
            #d = '\n'.join([delimiter.join(l) for l in data])
            #fd.write('\n%s %s'%(comments,h))
            #fd.write('\n%s'%(d))





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
