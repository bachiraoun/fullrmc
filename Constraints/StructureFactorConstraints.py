"""
StructureFactorConstraints contains classes for all constraints related experimental static structure factor functions.

.. inheritance-diagram:: fullrmc.Constraints.StructureFactorConstraints
    :parts: 1
"""
# standard libraries imports
from __future__ import print_function
import itertools, re

# external libraries imports
import numpy as np
from pdbparser.Utilities.Database import is_element_property, get_element_property
from pdbparser.Utilities.Collection import get_normalized_weighting

# fullrmc imports
from ..Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, LOGGER
from ..Globals import str, long, unicode, bytes, basestring, range, xrange, maxint
from ..Core.Collection import is_number, is_integer, get_path
from ..Core.Collection import reset_if_collected_out_of_date, get_real_elements_weight
from ..Core.Collection import get_caller_frames
from ..Core.Constraint import Constraint, ExperimentalConstraint
from ..Core.pairs_histograms import multiple_pairs_histograms_coords, full_pairs_histograms_coords


class StructureFactorConstraint(ExperimentalConstraint):
    """
    Controls the Structure Factor noted as S(Q) and also called
    total-scattering structure function or Static Structure Factor.
    S(Q) is a dimensionless quantity and normalized such as the average
    value :math:`<S(Q)>=1`.

    It is worth mentioning that S(Q) is nothing other than the normalized and
    corrected diffraction pattern if all experimental artefacts powder.

    The computation of S(Q) is done through an inverse Sine Fourier transform
    of the computed pair distribution function G(r).

    .. math::

        S(Q) = 1+ \\frac{1}{Q} \\int_{0}^{\\infty} G(r) sin(Qr) dr

    From an atomistic model and histogram point of view, G(r) is computed as
    the following:

    .. math::

        G(r) = 4 \\pi r (\\rho_{r} - \\rho_{0})
             = 4 \\pi \\rho_{0} r (g(r)-1)
             = \\frac{R(r)}{r} - 4 \\pi \\rho_{0}

    g(r) is calculated after binning all pair atomic distances into a
    weighted histograms as the following:

    .. math::
        g(r) = \\sum \\limits_{i,j}^{N} w_{i,j} \\frac{\\rho_{i,j}(r)}{\\rho_{0}}
             = \\sum \\limits_{i,j}^{N} w_{i,j} \\frac{n_{i,j}(r) / v(r)}{N_{i,j} / V}

    Where:\n
    :math:`Q` is the momentum transfer. \n
    :math:`r` is the distance between two atoms. \n
    :math:`\\rho_{i,j}(r)` is the pair density function of atoms i and j. \n
    :math:`\\rho_{0}` is the  average number density of the system. \n
    :math:`w_{i,j}` is the relative weighting of atom types i and j. \n
    :math:`R(r)` is the radial distribution function (rdf). \n
    :math:`N` is the total number of atoms. \n
    :math:`V` is the volume of the system. \n
    :math:`n_{i,j}(r)` is the number of atoms i neighbouring j at a distance r. \n
    :math:`v(r)` is the annulus volume at distance r and of thickness dr. \n
    :math:`N_{i,j}` is the total number of atoms i and j in the system. \n



    +----------------------------------------------------------------------+
    |.. figure:: reduced_structure_factor_constraint_plot_method.png       |
    |   :width: 530px                                                      |
    |   :height: 400px                                                     |
    |   :align: left                                                       |
    |                                                                      |
    |   Reduced structure factor of memory shape Nickel-Titanium alloy.    |
    +----------------------------------------------------------------------+


    :Parameters:
        #. experimentalData (numpy.ndarray, string): Experimental data as
           numpy.ndarray or string path to load data using numpy.loadtxt
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
        #. weighting (string): The elements weighting scheme. It must be any
           atomic attribute (atomicNumber, neutronCohb, neutronIncohb,
           neutronCohXs, neutronIncohXs, atomicWeight, covalentRadius) defined
           in pdbparser database. In case of xrays or neutrons experimental
           weights, one can simply set weighting to 'xrays' or 'neutrons'
           and the value will be automatically adjusted to respectively
           'atomicNumber' and 'neutronCohb'. If attribute values are
           missing in the pdbparser database, atomic weights must be
           given in atomsWeight dictionary argument.
        #. atomsWeight (None, dict): Atoms weight dictionary where keys are
           atoms element and values are custom weights. If None is given
           or partially given, missing elements weighting will be fully set
           given weighting scheme.
        #. rmin (None, number): The minimum distance value to compute G(r)
           histogram. If None is given, rmin is computed as
           :math:`2 \\pi / Q_{max}`.
        #. rmax (None, number): The maximum distance value to compute G(r)
           histogram. If None is given, rmax is computed as
           :math:`2 \\pi / dQ`.
        #. dr (None, number): The distance bin value to compute G(r)
           histogram. If None is given, bin is computed as
           :math:`2 \\pi / (Q_{max}-Q_{min})`.
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
        #. windowFunction (None, numpy.ndarray): The window function to
           convolute with the computed pair distribution function of the
           system prior to comparing it with the experimental data. In
           general, the experimental pair distribution function G(r) shows
           artificial wrinkles, among others the main reason is because
           G(r) is computed by applying a sine Fourier transform to the
           experimental structure factor S(q). Therefore window function is
           used to best imitate the numerical artefacts in the experimental
           data.
        #. limits (None, tuple, list): The distance limits to compute the
           histograms. If None is given, the limits will be automatically
           set the the min and max distance of the experimental data.
           Otherwise, a tuple of exactly two items where the first is the
           minimum distance or None and the second is the maximum distance
           or None.

    **NB**: If adjustScaleFactor first item (frequency) is 0, the scale factor
    will remain untouched and the limits minimum and maximum won't be checked.

    .. code-block:: python

        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.StructureFactorConstraints import StructureFactorConstraint

        # create engine
        ENGINE = Engine(path='my_engine.rmc')

        # set pdb file
        ENGINE.set_pdb('system.pdb')

        # create and add constraint
        SFC = StructureFactorConstraint(experimentalData="sq.dat", weighting="atomicNumber")
        ENGINE.add_constraints(SFC)

    """
    def __init__(self, experimentalData, dataWeights=None,
                       weighting="atomicNumber", atomsWeight=None,
                       rmin=None, rmax=None, dr=None,
                       scaleFactor=1.0, adjustScaleFactor=(0, 0.8, 1.2),
                       windowFunction=None, limits=None):
        # initialize variables
        self.__experimentalQValues = None
        self.__experimentalSF      = None
        self.__rmin                = None
        self.__rmax                = None
        self.__dr                  = None
        self.__minimumDistance     = None
        self.__maximumDistance     = None
        self.__bin                 = None
        self.__shellCenters        = None
        self.__histogramSize       = None
        self.__shellVolumes        = None
        self.__Gr2SqMatrix         = None
        # initialize constraint
        super(StructureFactorConstraint, self).__init__( experimentalData=experimentalData, dataWeights=dataWeights, scaleFactor=scaleFactor, adjustScaleFactor=adjustScaleFactor)
        # set atomsWeight
        self.set_atoms_weight(atomsWeight)
        # set elements weighting
        self.set_weighting(weighting)
        self.__set_weighting_scheme()
        # set window function
        self.set_window_function(windowFunction)
        # set r parameters
        self.set_rmin(rmin)
        self.set_rmax(rmax)
        self.set_dr(dr)

        # set frame data
        FRAME_DATA = [d for d in self.FRAME_DATA]
        FRAME_DATA.extend(['_StructureFactorConstraint__experimentalQValues',
                           '_StructureFactorConstraint__experimentalSF',
                           '_StructureFactorConstraint__elementsPairs',
                           '_StructureFactorConstraint__weightingScheme',
                           '_StructureFactorConstraint__atomsWeight',
                           '_StructureFactorConstraint__qmin',
                           '_StructureFactorConstraint__qmax',
                           '_StructureFactorConstraint__rmin',
                           '_StructureFactorConstraint__rmax',
                           '_StructureFactorConstraint__dr',
                           '_StructureFactorConstraint__minimumDistance',
                           '_StructureFactorConstraint__maximumDistance',
                           '_StructureFactorConstraint__bin',
                           '_StructureFactorConstraint__shellCenters',
                           '_StructureFactorConstraint__histogramSize',
                           '_StructureFactorConstraint__shellVolumes',
                           '_StructureFactorConstraint__Gr2SqMatrix',
                           '_StructureFactorConstraint__windowFunction',
                           '_elementsWeight',] )
        RUNTIME_DATA = [d for d in self.RUNTIME_DATA]
        RUNTIME_DATA.extend( [] )
        object.__setattr__(self, 'FRAME_DATA',   tuple(FRAME_DATA)   )
        object.__setattr__(self, 'RUNTIME_DATA', tuple(RUNTIME_DATA) )

    def _codify_update__(self, name='constraint', addDependencies=True):
        dependencies = []
        code         = []
        if addDependencies:
            code.extend(dependencies)
        dw = self.dataWeights
        if dw is not None:
            dw = list(dw)
        code.append("dw = {dw}".format(dw=dw))
        wf = self.windowFunction
        if isinstance(wf, np.ndarray):
            code.append("wf = np.array({wf})".format(wf=list(wf)))
        else:
            code.append("wf = {wf}".format(wf=wf))

        code.append("{name}.set_used({val})".format(name=name, val=self.used))
        code.append("{name}.set_scale_factor({val})".format(name=name, val=self.scaleFactor))
        code.append("{name}.set_adjust_scale_factor({val})".format(name=name, val=self.adjustScaleFactor))
        code.append("{name}.set_data_weights(dw)".format(name=name))
        code.append("{name}.set_atoms_weight({val})".format(name=name, val=self.atomsWeight))
        code.append("{name}.set_window_function(wf)".format(name=name))
        code.append("{name}.set_rmin({val})".format(name=name, val=self.rmin))
        code.append("{name}.set_rmax({val})".format(name=name, val=self.rmax))
        code.append("{name}.set_dr({val})".format(name=name, val=self.dr))
        code.append("{name}.set_limits({val})".format(name=name, val=self.limits))
        # return
        return dependencies, '\n'.join(code)


    def _codify__(self, engine, name='constraint', addDependencies=True):
        assert isinstance(name, basestring), LOGGER.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, LOGGER.error("given name '%s' can't be used as a variable name"%name)
        klass        = self.__class__.__name__
        dependencies = ['import numpy as np','from fullrmc.Constraints import StructureFactorConstraints']
        code         = []
        if addDependencies:
            code.extend(dependencies)
        x = list(self.experimentalData[:,0])
        y = list(self.experimentalData[:,1])
        code.append("x = {x}".format(x=x))
        code.append("y = {y}".format(y=y))
        code.append("d = np.transpose([x,y]).astype(np.float32)")
        dw = self.dataWeights
        if dw is not None:
            dw = list(dw)
        code.append("dw = {dw}".format(dw=dw))
        wf = self.windowFunction
        if isinstance(wf, np.ndarray):
            code.append("wf = np.array({wf})".format(wf=list(wf)))
        else:
            code.append("wf = {wf}".format(wf=wf))
        code.append("{name} = {klass}s.{klass}\
(experimentalData=d, dataWeights=dw, weighting='{weighting}', atomsWeight={atomsWeight}, \
rmin={rmin}, rmax={rmax}, dr={dr}, scaleFactor={scaleFactor}, adjustScaleFactor={adjustScaleFactor}, \
shapeFuncParams=sfp, windowFunction=wf, limits={limits})".format(name=name, klass=klass,
                weighting=self.weighting, atomsWeight=self.atomsWeight, rmin=self.rmin,
                rmax=self.rmax, dr=self.dr, scaleFactor=self.scaleFactor,
                adjustScaleFactor=self.adjustScaleFactor, limits=self.limits))
        code.append("{engine}.add_constraints([{name}])".format(engine=engine, name=name))
        # return
        return dependencies, '\n'.join(code)


    #def __getstate__(self):
    #    # make sure that __Gr2SqMatrix is not pickled but saved to the disk as None
    #    state = super(StructureFactorConstraint, self).__getstate__()
    #    state["_StructureFactorConstraint__Gr2SqMatrix"] = None
    #    return state
    #
    #def __setstate__(self, state):
    #    # make sure to regenerate G(r) to S(q) matrix at loading time
    #    self.__dict__.update( state )
    #    self.__set_Gr_2_Sq_matrix()
    #

    def __set_Gr_2_Sq_matrix(self):
        if self.__experimentalQValues is None or self.__shellCenters is None:
            self.__Gr2SqMatrix = None
        else:
            Qs = self.__experimentalQValues
            Rs = self.__shellCenters
            dr = self.__shellCenters[1]-self.__shellCenters[0]
            qr = Rs.reshape((-1,1))*(np.ones((len(Rs),1), dtype=FLOAT_TYPE)*Qs)
            sinqr = np.sin(qr)
            sinqr_q = sinqr/Qs
            self.__Gr2SqMatrix = dr*sinqr_q

    def __set_weighting_scheme(self):
        if self.engine is not None:
            self.__elementsPairs   = sorted(itertools.combinations_with_replacement(self.engine.elements,2))
            #elementsWeight        = dict([(el,float(get_element_property(el,self.__weighting))) for el in self.engine.elements])
            #self._elementsWeight  = dict([(el,self.__atomsWeight.get(el, float(get_element_property(el,self.__weighting)))) for el in self.engine.elements])
            self._elementsWeight   = get_real_elements_weight(elements=self.engine.elements, weightsDict=self.__atomsWeight, weighting=self.__weighting)
            self.__weightingScheme = get_normalized_weighting(numbers=self.engine.numberOfAtomsPerElement, weights=self._elementsWeight)
            for k in self.__weightingScheme:
                self.__weightingScheme[k] = FLOAT_TYPE(self.__weightingScheme[k])
        else:
            self.__elementsPairs   = None
            self.__weightingScheme = None
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__elementsPairs'  : self.__elementsPairs,
                                  '_StructureFactorConstraint__weightingScheme': self.__weightingScheme})

    def __set_histogram(self):
        if self.__minimumDistance is None or self.__maximumDistance is None or self.__bin is None:
            self.__shellCenters  = None
            self.__histogramSize = None
            self.__shellVolumes  = None
        else:
            # compute edges
            if self.engine is not None and self.rmax is None:
                minHalfBox = np.min( [np.linalg.norm(v)/2. for v in self.engine.basisVectors])
                self.__edges = np.arange(self.__minimumDistance,minHalfBox, self.__bin).astype(FLOAT_TYPE)
            else:
                self.__edges = np.arange(self.__minimumDistance, self.__maximumDistance+self.__bin, self.__bin).astype(FLOAT_TYPE)
            # adjust rmin and rmax
            self.__minimumDistance  = self.__edges[0]
            self.__maximumDistance  = self.__edges[-1]
            # compute shellCenters
            self.__shellCenters = (self.__edges[0:-1]+self.__edges[1:])/FLOAT_TYPE(2.)
            # set histogram size
            self.__histogramSize = INT_TYPE( len(self.__edges)-1 )
            # set shell centers and volumes
            self.__shellVolumes = FLOAT_TYPE(4.0/3.)*PI*((self.__edges[1:])**3 - self.__edges[0:-1]**3)
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__minimumDistance': self.__minimumDistance,
                                  '_StructureFactorConstraint__maximumDistance': self.__maximumDistance,
                                  '_StructureFactorConstraint__shellCenters'   : self.__shellCenters,
                                  '_StructureFactorConstraint__histogramSize'  : self.__histogramSize,
                                  '_StructureFactorConstraint__shellVolumes'   : self.__shellVolumes})
        # reset constraint
        self.reset_constraint()
        # reset sq matrix
        self.__set_Gr_2_Sq_matrix()

    def _on_collector_reset(self):
        pass

    @property
    def rmin(self):
        """ Histogram minimum distance. """
        return self.__rmin

    @property
    def rmax(self):
        """ Histogram maximum distance. """
        return self.__rmax

    @property
    def dr(self):
        """ Histogram bin size."""
        return self.__dr

    @property
    def bin(self):
        """ Computed histogram distance bin size."""
        return self.__bin

    @property
    def minimumDistance(self):
        """ Computed histogram minimum distance. """
        return self.__minimumDistance

    @property
    def maximumDistance(self):
        """ Computed histogram maximum distance. """
        return self.__maximumDistance

    @property
    def qmin(self):
        """ Experimental data reciprocal distances minimum. """
        return self.__qmin

    @property
    def qmax(self):
        """ Experimental data reciprocal distances maximum. """
        return self.__qmax

    @property
    def dq(self):
        """ Experimental data reciprocal distances bin size. """
        return self.__experimentalQValues[1]-self.__experimentalQValues[0]

    @property
    def experimentalQValues(self):
        """ Experimental data used q values. """
        return self.__experimentalQValues

    @property
    def histogramSize(self):
        """ Histogram size"""
        return self.__histogramSize

    @property
    def shellCenters(self):
        """ Shells center array"""
        return self.__shellCenters

    @property
    def shellVolumes(self):
        """ Shells volume array"""
        return self.__shellVolumes

    @property
    def experimentalSF(self):
        """ Experimental Structure Factor or S(q)"""
        return self.__experimentalSF

    @property
    def elementsPairs(self):
        """ Elements pairs """
        return self.__elementsPairs

    @property
    def atomsWeight(self):
        """Custom atoms weight"""
        return self.__atomsWeight

    @property
    def weighting(self):
        """ Elements weighting definition. """
        return self.__weighting

    @property
    def weightingScheme(self):
        """ Elements weighting scheme. """
        return self.__weightingScheme

    @property
    def windowFunction(self):
        """ Convolution window function. """
        return self.__windowFunction

    @property
    def Gr2SqMatrix(self):
        """ G(r) to S(q) transformation matrix."""
        return self.__Gr2SqMatrix

    @property
    def _experimentalX(self):
        """For internal use only to interface
        ExperimentalConstraint.get_constraints_properties"""
        return self.__experimentalQValues

    @property
    def _experimentalY(self):
        """For internal use only to interface
        ExperimentalConstraint.get_constraints_properties"""
        return self.__experimentalSF

    @property
    def _modelX(self):
        """For internal use only to interface
        ExperimentalConstraint.get_constraints_properties"""
        return self.__experimentalQValues

    def listen(self, message, argument=None):
        """
        Listens to any message sent from the Broadcaster.

        :Parameters:
            #. message (object): Any python object to send to constraint's
               listen method.
            #. argument (object): Any type of argument to pass to the
               listeners.
        """
        if message in ("engine set","update pdb","update molecules indexes","update elements indexes","update names indexes"):
            self.__set_weighting_scheme()
            # reset histogram
            if self.engine is not None:
                self.__set_histogram()
            self.reset_constraint() # ADDED 2017-JAN-08
        elif message in("update boundary conditions",):
            self.reset_constraint()

    def set_rmin(self, rmin):
        """
        Set rmin value.

        :parameters:
            #. rmin (None, number): The minimum distance value to compute G(r)
               histogram. If None is given, rmin is computed as
               :math:`2 \\pi / Q_{max}`.
        """
        if rmin is None:
            minimumDistance = FLOAT_TYPE( 2.*PI/self.__qmax )
        else:
            assert is_number(rmin), LOGGER.error("rmin must be None or a number")
            minimumDistance = FLOAT_TYPE(rmin)
        if self.__maximumDistance is not None:
            assert minimumDistance<self.__maximumDistance, LOGGER.error("rmin must be smaller than rmax %s"%self.__maximumDistance)
        self.__rmin = rmin
        self.__minimumDistance = minimumDistance
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__rmin': self.__rmin,
                                  '_StructureFactorConstraint__minimumDistance': self.__minimumDistance})
        # reset histogram
        self.__set_histogram()

    def set_rmax(self, rmax):
        """
        Set rmax value.

        :Parameters:
            #. rmax (None, number): The maximum distance value to compute G(r)
               histogram. If None is given, rmax is computed as
               :math:`2 \\pi / dQ`.
        """
        if rmax is None:
            dq = self.__experimentalQValues[1]-self.__experimentalQValues[0]
            maximumDistance = FLOAT_TYPE( 2.*PI/dq )
        else:
            assert is_number(rmax), LOGGER.error("rmax must be None or a number")
            maximumDistance = FLOAT_TYPE(rmax)
        if self.__minimumDistance is not None:
            assert maximumDistance>self.__minimumDistance, LOGGER.error("rmax must be bigger than rmin %s"%self.__minimumDistance)
        self.__rmax = rmax
        self.__maximumDistance = maximumDistance
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__rmax': self.__rmax,
                                  '_StructureFactorConstraint__maximumDistance': self.__maximumDistance})
        # reset histogram
        self.__set_histogram()

    def set_dr(self, dr):
        """
        Set dr value.

        :Parameters:
            #. dr (None, number): The distance bin value to compute G(r)
               histogram. If None is given, bin is computed as
               :math:`2 \\pi / (Q_{max}-Q_{min})`.
        """
        if dr is None:
            bin  = 2.*PI/self.__qmax
            rbin = round(bin,1)
            if rbin>bin:
                rbin -= 0.1
            bin = FLOAT_TYPE( rbin  )
        else:
            assert is_number(dr), LOGGER.error("dr must be None or a number")
            bin = FLOAT_TYPE(dr)
        self.__dr = dr
        self.__bin = bin
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__dr': self.__dr,
                                  '_StructureFactorConstraint__bin': self.__bin})
        # reset histogram
        self.__set_histogram()

    def set_weighting(self, weighting):
        """
        Set elements weighting. It must be a valid entry of pdbparser atom's
        database.

        :Parameters:
            #. weighting (string): The elements weighting scheme. It must be
               any atomic attribute (atomicNumber, neutronCohb, neutronIncohb,
               neutronCohXs, neutronIncohXs, atomicWeight, covalentRadius)
               defined in pdbparser database. In case of xrays or neutrons
               experimental weights, one can simply set weighting to 'xrays'
               or 'neutrons' and the value will be automatically adjusted to
               respectively 'atomicNumber' and 'neutronCohb'. If attribute
               values are  missing in the pdbparser database, atomic weights
               must be given in atomsWeight dictionary argument.
        """
        if weighting.lower() in ["xrays","x-rays","xray","x-ray"]:
            LOGGER.fixed("'%s' weighting is set to atomicNumber"%weighting)
            weighting = "atomicNumber"
        elif weighting.lower() in ["neutron","neutrons"]:
            LOGGER.fixed("'%s' weighting is set to neutronCohb"%weighting)
            weighting = "neutronCohb"
        assert is_element_property(weighting),LOGGER.error( "weighting is not a valid pdbparser atoms database entry")
        assert weighting != "atomicFormFactor", LOGGER.error("atomicFormFactor weighting is not allowed")
        self.__weighting = weighting
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__weighting': self.__weighting})

    def set_atoms_weight(self, atomsWeight):
        """
        Custom set atoms weight. This is the way to setting a atoms weights
        different than the given weighting scheme.

        :Parameters:
            #. atomsWeight (None, dict): Atoms weight dictionary where keys are
               atoms element and values are custom weights. If None is given
               or partially given, missing elements weighting will be fully set
               given weighting scheme.
        """
        if atomsWeight is None:
            AW = {}
        else:
            assert isinstance(atomsWeight, dict),LOGGER.error("atomsWeight must be None or a dictionary")
            AW = {}
            for k in atomsWeight:
                assert isinstance(k, basestring),LOGGER.error("atomsWeight keys must be strings")
                try:
                    val = float(atomsWeight[k])
                except:
                    raise LOGGER.error( "atomsWeight values must be numerical")
                AW[k]=val
        # set atomsWeight
        self.__atomsWeight = AW
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__atomsWeight': self.__atomsWeight})

    def set_window_function(self, windowFunction):
        """
        Set convolution window function.

        :Parameters:
             #. windowFunction (None, numpy.ndarray): The window function to
                convolute with the computed pair distribution function of the
                system prior to comparing it with the experimental data. In
                general, the experimental pair distribution function G(r) shows
                artificial wrinkles, among others the main reason is because
                G(r) is computed by applying a sine Fourier transform to the
                experimental structure factor S(q). Therefore window function is
                used to best imitate the numerical artefacts in the experimental
                data.
        """
        if windowFunction is not None:
            assert isinstance(windowFunction, np.ndarray), LOGGER.error("windowFunction must be a numpy.ndarray")
            assert windowFunction.dtype.type is FLOAT_TYPE, LOGGER.error("windowFunction type must be %s"%FLOAT_TYPE)
            assert len(windowFunction.shape) == 1, LOGGER.error("windowFunction must be of dimension 1")
            assert len(windowFunction) <= self.experimentalData.shape[0], LOGGER.error("windowFunction length must be smaller than experimental data")
            # normalize window function
            windowFunction /= np.sum(windowFunction)
        # check window size
        # set windowFunction
        self.__windowFunction = windowFunction
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__windowFunction': self.__windowFunction})

    def set_experimental_data(self, experimentalData):
        """
        Set constraint's experimental data.

        :Parameters:
            #. experimentalData (numpy.ndarray, string): The experimental
               data as numpy.ndarray or string path to load data using
               numpy.loadtxt function.
        """
        # get experimental data
        super(StructureFactorConstraint, self).set_experimental_data(experimentalData=experimentalData)
        # set limits
        self.set_limits(self.limits)

    def set_limits(self, limits):
        """
        Set the reciprocal distance limits (qmin, qmax).

        :Parameters:
            #. limits (None, tuple, list): Distance limits to bound
               experimental data and compute histograms.
               If None is given, the limits will be automatically set to
               min and max reciprocal distance recorded in experimental data.
               If given, a tuple of minimum reciprocal distance (qmin) or None
               and maximum reciprocal distance (qmax) or None should be given.
        """
        self._ExperimentalConstraint__set_limits(limits)
        # set qvalues
        self.__experimentalQValues = self.experimentalData[self.limitsIndexStart:self.limitsIndexEnd+1,0].astype(FLOAT_TYPE)
        self.__experimentalSF      = self.experimentalData[self.limitsIndexStart:self.limitsIndexEnd+1,1].astype(FLOAT_TYPE)
        # set qmin and qmax
        self.__qmin = self.__experimentalQValues[0]
        self.__qmax = self.__experimentalQValues[-1]
        assert self.__qmin>0, LOGGER.error("qmin must be bigger than 0. Experimental null q values are ambigous. Try setting limits.")
        # dump to repository
        self._dump_to_repository({'_StructureFactorConstraint__experimentalQValues': self.__experimentalQValues,
                                  '_StructureFactorConstraint__experimentalSF'     : self.__experimentalSF,
                                  '_StructureFactorConstraint__qmin'               : self.__qmin,
                                  '_StructureFactorConstraint__qmax'               : self.__qmax})
        # set used dataWeights
        self._set_used_data_weights(limitsIndexStart=self.limitsIndexStart, limitsIndexEnd=self.limitsIndexEnd)
        # reset constraint
        self.reset_constraint()
        # reset sq matrix
        self.__set_Gr_2_Sq_matrix()

    def update_standard_error(self):
        """ Compute and set constraint's standardError."""
        # set standardError
        totalSQ = self.get_constraint_value()["total_no_window"]
        self.set_standard_error(self.compute_standard_error(modelData = totalSQ))

    def check_experimental_data(self, experimentalData):
        """
        Check whether experimental data is correct.

        :Parameters:
            #. experimentalData (object): The experimental data to check.

        :Returns:
            #. result (boolean): Whether it is correct or not.
            #. message (str): Checking message that explains whats's wrong
               with the given data
        """
        if not isinstance(experimentalData, np.ndarray):
            return False, "experimentalData must be a numpy.ndarray"
        if experimentalData.dtype.type is not FLOAT_TYPE:
            return False, "experimentalData type must be %s"%FLOAT_TYPE
        if len(experimentalData.shape) !=2:
            return False, "experimentalData must be of dimension 2"
        if experimentalData.shape[1] !=2:
            return False, "experimentalData must have only 2 columns"
        # check distances order
        inOrder = (np.array(sorted(experimentalData[:,0]), dtype=FLOAT_TYPE)-experimentalData[:,0])<=PRECISION
        if not np.all(inOrder):
            return False, "experimentalData distances are not sorted in order"
        if experimentalData[0][0]<0:
            return False, "experimentalData distances min value is found negative"
        # data format is correct
        return True, ""

    def compute_standard_error(self, modelData):
        """
        Compute the standard error (StdErr) as the squared deviations
        between model computed data and the experimental ones.

        .. math::
            StdErr = \\sum \\limits_{i}^{N} W_{i}(Y(X_{i})-F(X_{i}))^{2}

        Where:\n
        :math:`N` is the total number of experimental data points. \n
        :math:`W_{i}` is the data point weight. It becomes equivalent to 1 when dataWeights is set to None. \n
        :math:`Y(X_{i})` is the experimental data point :math:`X_{i}`. \n
        :math:`F(X_{i})` is the computed from the model data  :math:`X_{i}`. \n

        :Parameters:
            #. modelData (numpy.ndarray): The data to compare with the
               experimental one and compute the squared deviation.

        :Returns:
            #. standardError (number): The calculated constraint's
               standardError.
        """
        # compute difference
        diff = self.__experimentalSF-modelData
        # return standard error
        if self._usedDataWeights is None:
            return np.add.reduce((diff)**2)
        else:
            return np.add.reduce(self._usedDataWeights*((diff)**2))

    def _get_Sq_from_Gr(self, Gr):
        return np.sum(Gr.reshape((-1,1))*self.__Gr2SqMatrix, axis=0)+1

    def _apply_scale_factor(self, Sq, scaleFactor):
        if scaleFactor != 1:
            Sq = scaleFactor*(Sq-1) + 1
        return Sq

    def __get_total_Sq(self, data, rho0):
        """This method is created just to speed up the computation of
        the total Sq upon fitting."""
        Gr = np.zeros(self.__histogramSize, dtype=FLOAT_TYPE)
        for pair in self.__elementsPairs:
            # get weighting scheme
            wij = self.__weightingScheme.get(pair[0]+"-"+pair[1], None)
            if wij is None:
                wij = self.__weightingScheme[pair[1]+"-"+pair[0]]
            # get number of atoms per element
            ni = self.engine.numberOfAtomsPerElement[pair[0]]
            nj = self.engine.numberOfAtomsPerElement[pair[1]]
            # get index of element
            idi = self.engine.elements.index(pair[0])
            idj = self.engine.elements.index(pair[1])
            # get Nij
            if idi == idj:
                Nij = ni*(ni-1)/2.0
                Dij = Nij/self.engine.volume
                nij = data["intra"][idi,idj,:]+data["inter"][idi,idj,:]
                Gr += wij*nij/Dij
            else:
                Nij = ni*nj
                Dij = Nij/self.engine.volume
                nij = data["intra"][idi,idj,:]+data["intra"][idj,idi,:] + data["inter"][idi,idj,:]+data["inter"][idj,idi,:]
                Gr += wij*nij/Dij
        # Devide by shells volume
        Gr /= self.shellVolumes
        # compute total G(r)
        #rho0 = (self.engine.numberOfAtoms/self.engine.volume).astype(FLOAT_TYPE)
        Gr   = (FLOAT_TYPE(4.)*PI*self.__shellCenters*rho0)*(Gr-1)
        # Compute S(q) from G(r)
        Sq = self._get_Sq_from_Gr(Gr)
        # Multiply by scale factor
        self._fittedScaleFactor = self.get_adjusted_scale_factor(self.__experimentalSF, Sq, self._usedDataWeights)
        # apply scale factor
        Sq = self._apply_scale_factor(Sq, self._fittedScaleFactor)
        # apply multiframe prior and weight
        Sq = self._apply_multiframe_prior(Sq)
        # convolve total with window function
        if self.__windowFunction is not None:
            Sq = np.convolve(Sq, self.__windowFunction, 'same')
        return Sq

    def get_adjusted_scale_factor(self, experimentalData, modelData, dataWeights):
        """Overload to reduce S(q) prior to fitting scale factor.
        S(q) -> 1 at high q and this will create a wrong scale factor.
        Overloading can be avoided but it's better to for performance reasons
        """
        SF = self.scaleFactor
        # check to update scaleFactor
        if self.adjustScaleFactorFrequency:
            if not self.engine.accepted%self.adjustScaleFactorFrequency:
                SF = self.fit_scale_factor(experimentalData-1, modelData-1, dataWeights)
        return SF

    def _get_constraint_value(self, data, applyMultiframePrior=True):
        # http://erice2011.docking.org/upload/Other/Billinge_PDF/03-ReadingMaterial/BillingePDF2011.pdf    page 6
        #import time
        #startTime = time.clock()
        output = {}
        for pair in self.__elementsPairs:
            output["sf_intra_%s-%s" % pair] = np.zeros(self.__histogramSize, dtype=FLOAT_TYPE)
            output["sf_inter_%s-%s" % pair] = np.zeros(self.__histogramSize, dtype=FLOAT_TYPE)
            output["sf_total_%s-%s" % pair] = np.zeros(self.__histogramSize, dtype=FLOAT_TYPE)
        gr = np.zeros(self.__histogramSize, dtype=FLOAT_TYPE)
        for pair in self.__elementsPairs:
            # get weighting scheme
            wij = self.__weightingScheme.get(pair[0]+"-"+pair[1], None)
            if wij is None:
                wij = self.__weightingScheme[pair[1]+"-"+pair[0]]
            # get number of atoms per element
            ni = self.engine.numberOfAtomsPerElement[pair[0]]
            nj = self.engine.numberOfAtomsPerElement[pair[1]]
            # get index of element
            idi = self.engine.elements.index(pair[0])
            idj = self.engine.elements.index(pair[1])
            # get Nij
            if idi == idj:
                Nij = ni*(ni-1)/2.0
                output["sf_intra_%s-%s" % pair] += data["intra"][idi,idj,:]
                output["sf_inter_%s-%s" % pair] += data["inter"][idi,idj,:]
            else:
                Nij = ni*nj
                output["sf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] + data["intra"][idj,idi,:]
                output["sf_inter_%s-%s" % pair] += data["inter"][idi,idj,:] + data["inter"][idj,idi,:]
            # compute g(r)
            nij = output["sf_intra_%s-%s" % pair] + output["sf_inter_%s-%s" % pair]
            dij = nij/self.__shellVolumes
            Dij = Nij/self.engine.volume
            gr += wij*dij/Dij
            # calculate intensityFactor
            intensityFactor = (self.engine.volume*wij)/(Nij*self.__shellVolumes)
            # divide by factor
            output["sf_intra_%s-%s" % pair] *= intensityFactor
            output["sf_inter_%s-%s" % pair] *= intensityFactor
            output["sf_total_%s-%s" % pair]  = output["sf_intra_%s-%s" % pair] + output["sf_inter_%s-%s" % pair]
            # Compute S(q) from G(r)
            output["sf_intra_%s-%s" % pair] = self._get_Sq_from_Gr(output["sf_intra_%s-%s" % pair])
            output["sf_inter_%s-%s" % pair] = self._get_Sq_from_Gr(output["sf_inter_%s-%s" % pair])
            output["sf_total_%s-%s" % pair] = self._get_Sq_from_Gr(output["sf_total_%s-%s" % pair])
        # compute total G(r)
        rho0 = (self.engine.numberOfAtoms/self.engine.volume).astype(FLOAT_TYPE)
        Gr = (FLOAT_TYPE(4.)*PI*self.__shellCenters*rho0) * (gr-1)
        # Compute S(q) from G(r)
        Sq = self._get_Sq_from_Gr(Gr)
        # multiply by scale factor
        output["total_no_window"] = self._apply_scale_factor(Sq, self._fittedScaleFactor)
        # apply multiframe prior and weight
        if applyMultiframePrior:
            output["total_no_window"] = self._apply_multiframe_prior(output["total_no_window"])
        # convolve total with window function
        if self.__windowFunction is not None:
            output["total"] = np.convolve(output["total_no_window"], self.__windowFunction, 'same').astype(FLOAT_TYPE)
        else:
            output["total"] = output["total_no_window"]
        return output

    def get_constraint_value(self, applyMultiframePrior=True):
        """
        Compute all partial Structure Factor (SQs).

        :Parameters:
            #. applyMultiframePrior (boolean): Whether to apply subframe weight
               and prior to the total. This will only have an effect when used
               frame is a subframe and in case subframe weight and prior is
               defined.

        :Returns:
            #. SQs (dictionary): The SQs dictionnary, where keys are the
               element wise intra and inter molecular SQs and values are
               the computed SQs.
        """
        if self.data is None:
            LOGGER.warn("data must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.data, applyMultiframePrior=applyMultiframePrior)

    def get_constraint_original_value(self):
        """
        Compute all partial Pair Distribution Functions (PDFs).

        :Returns:
            #. PDFs (dictionary): The PDFs dictionnary, where keys are the
               element wise intra and inter molecular PDFs and values are the
               computed PDFs.
        """
        if self.originalData is None:
            LOGGER.warn("originalData must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.originalData)

    @reset_if_collected_out_of_date
    def compute_data(self, update=True):
        """ Compute constraint's data.

        :Parameters:
            #. update (boolean): whether to update constraint data and
               standard error with new computation. If data is computed and
               updated by another thread or process while the stochastic
               engine is running, this might lead to a state alteration of
               the constraint which will lead to a no additional accepted
               moves in the run

        :Returns:
            #. data (dict): constraint data dictionary
            #. standardError (float): constraint standard error
        """
        intra,inter = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates,
                                                    basis            = self.engine.basisVectors,
                                                    isPBC            = self.engine.isPBC,
                                                    moleculeIndex    = self.engine.moleculesIndex,
                                                    elementIndex     = self.engine.elementsIndex,
                                                    numberOfElements = self.engine.numberOfElements,
                                                    minDistance      = self.__minimumDistance,
                                                    maxDistance      = self.__maximumDistance,
                                                    histSize         = self.__histogramSize,
                                                    bin              = self.__bin,
                                                    ncores           = self.engine._runtime_ncores  )
        # create data and compute standard error
        data     = {"intra":intra, "inter":inter}
        totalSQ  = self.__get_total_Sq(data, rho0=self.engine.numberDensity)
        stdError = self.compute_standard_error(modelData = totalSQ)
        # update
        if update:
            self.set_data(data)
            self.set_active_atoms_data_before_move(None)
            self.set_active_atoms_data_after_move(None)
            self.set_standard_error(stdError)
            # set original data
            if self.originalData is None:
                self._set_original_data(self.data)
        # return
        return data, stdError

    def compute_before_move(self, realIndexes, relativeIndexes):
        """
        Compute constraint before move is executed

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Group atoms relative index
               the move will be applied to.
        """
        intraM,interM = multiple_pairs_histograms_coords( indexes          = relativeIndexes,
                                                          boxCoords        = self.engine.boxCoordinates,
                                                          basis            = self.engine.basisVectors,
                                                          isPBC            = self.engine.isPBC,
                                                          moleculeIndex    = self.engine.moleculesIndex,
                                                          elementIndex     = self.engine.elementsIndex,
                                                          numberOfElements = self.engine.numberOfElements,
                                                          minDistance      = self.__minimumDistance,
                                                          maxDistance      = self.__maximumDistance,
                                                          histSize         = self.__histogramSize,
                                                          bin              = self.__bin,
                                                          allAtoms         = True,
                                                          ncores           = self.engine._runtime_ncores )
        intraF,interF = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates[relativeIndexes],
                                                      basis            = self.engine.basisVectors,
                                                      isPBC            = self.engine.isPBC,
                                                      moleculeIndex    = self.engine.moleculesIndex[relativeIndexes],
                                                      elementIndex     = self.engine.elementsIndex[relativeIndexes],
                                                      numberOfElements = self.engine.numberOfElements,
                                                      minDistance      = self.__minimumDistance,
                                                      maxDistance      = self.__maximumDistance,
                                                      histSize         = self.__histogramSize,
                                                      bin              = self.__bin,
                                                      ncores           = self.engine._runtime_ncores )
        self.set_active_atoms_data_before_move( {"intra":intraM-intraF, "inter":interM-interF} )
        self.set_active_atoms_data_after_move(None)

    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """
        Compute constraint after move is executed

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Group atoms relative index
               the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[relativeIndexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[relativeIndexes] = movedBoxCoordinates
        # calculate pair distribution function
        intraM,interM = multiple_pairs_histograms_coords( indexes          = relativeIndexes,
                                                          boxCoords        = self.engine.boxCoordinates,
                                                          basis            = self.engine.basisVectors,
                                                          isPBC            = self.engine.isPBC,
                                                          moleculeIndex    = self.engine.moleculesIndex,
                                                          elementIndex     = self.engine.elementsIndex,
                                                          numberOfElements = self.engine.numberOfElements,
                                                          minDistance      = self.__minimumDistance,
                                                          maxDistance      = self.__maximumDistance,
                                                          histSize         = self.__histogramSize,
                                                          bin              = self.__bin,
                                                          allAtoms         = True,
                                                          ncores           = self.engine._runtime_ncores )
        intraF,interF = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates[relativeIndexes],
                                                      basis            = self.engine.basisVectors,
                                                      isPBC            = self.engine.isPBC,
                                                      moleculeIndex    = self.engine.moleculesIndex[relativeIndexes],
                                                      elementIndex     = self.engine.elementsIndex[relativeIndexes],
                                                      numberOfElements = self.engine.numberOfElements,
                                                      minDistance      = self.__minimumDistance,
                                                      maxDistance      = self.__maximumDistance,
                                                      histSize         = self.__histogramSize,
                                                      bin              = self.__bin,
                                                      ncores           = self.engine._runtime_ncores  )
        # set active atoms data
        self.set_active_atoms_data_after_move( {"intra":intraM-intraF, "inter":interM-interF} )
        # reset coordinates
        self.engine.boxCoordinates[relativeIndexes] = boxData
        # compute standardError after move
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        totalSQ   = self.__get_total_Sq({"intra":dataIntra, "inter":dataInter}, rho0=self.engine.numberDensity)
        self.set_after_move_standard_error( self.compute_standard_error(modelData = totalSQ) )
        # increment tried
        self.increment_tried()

    def accept_move(self, realIndexes, relativeIndexes):
        """
        Accept move

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        # change permanently _data
        self.set_data( {"intra":dataIntra, "inter":dataInter} )
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_standard_error( self.afterMoveStandardError )
        self.set_after_move_standard_error( None )
        # set new scale factor
        self._set_fitted_scale_factor_value(self._fittedScaleFactor)
        # increment accepted
        self.increment_accepted()

    def reject_move(self, realIndexes, relativeIndexes):
        """
        Reject move

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Not used here.
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_after_move_standard_error( None )

    def compute_as_if_amputated(self, realIndex, relativeIndex):
        """
        Compute and return constraint's data and standard error as if
        given atom is amputated.

        :Parameters:
            #. realIndex (numpy.ndarray): Atom's index as a numpy array
               of a single element.
            #. relativeIndex (numpy.ndarray): Atom's relative index as a
               numpy array of a single element.
        """
        # compute data
        self.compute_before_move(realIndexes=realIndex, relativeIndexes=relativeIndex)
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]
        data      = {"intra":dataIntra, "inter":dataInter}
        # temporarily adjust self.__weightingScheme
        weightingScheme = self.__weightingScheme
        relativeIndex = relativeIndex[0]
        selectedElement = self.engine.allElements[relativeIndex]
        self.engine.numberOfAtomsPerElement[selectedElement] -= 1
        self.__weightingScheme = get_normalized_weighting(numbers=self.engine.numberOfAtomsPerElement, weights=self._elementsWeight )
        for k in self.__weightingScheme:
            self.__weightingScheme[k] = FLOAT_TYPE(self.__weightingScheme[k])
        ## END OF ADDED 08 FEB 2017
        # compute standard error
        if not self.engine._RT_moveGenerator.allowFittingScaleFactor:
            SF = self.adjustScaleFactorFrequency
            self._set_adjust_scale_factor_frequency(0)
        rho0          = ((self.engine.numberOfAtoms-1)/self.engine.volume).astype(FLOAT_TYPE)
        totalSQ       = self.__get_total_Sq(data, rho0=rho0)
        standardError = self.compute_standard_error(modelData = totalSQ)
        if not self.engine._RT_moveGenerator.allowFittingScaleFactor:
            self._set_adjust_scale_factor_frequency(SF)
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        # set amputation
        self.set_amputation_data( {'data':data, 'weightingScheme':self.__weightingScheme} )
        # compute standard error
        self.set_amputation_standard_error( standardError )
        # reset weightingScheme and number of atoms per element
        self.__weightingScheme = weightingScheme
        self.engine.numberOfAtomsPerElement[selectedElement] += 1

    def accept_amputation(self, realIndex, relativeIndex):
        """
        Accept amputated atom and sets constraints data and standard error accordingly.

        :Parameters:
            #. realIndex (numpy.ndarray): Not used here.
            #. relativeIndex (numpy.ndarray): Not used here.
        """
        #self.set_data( self.amputationData ) ## COMMENTED 08 FEB 2017
        self.set_data( self.amputationData['data'] )
        self.__weightingScheme = self.amputationData['weightingScheme']
        self.set_standard_error( self.amputationStandardError )
        self.set_amputation_data( None )
        self.set_amputation_standard_error( None )
        # set new scale factor
        self._set_fitted_scale_factor_value(self._fittedScaleFactor)

    def reject_amputation(self, realIndex, relativeIndex):
        """
        Reject amputated atom and set constraint's data and standard
        error accordingly.

        :Parameters:
            #. realIndex (numpy.ndarray): Not used here.
            #. relativeIndex (numpy.ndarray): Not used here.
        """
        self.set_amputation_data( None )
        self.set_amputation_standard_error( None )

    def _on_collector_collect_atom(self, realIndex):
        pass

    def _on_collector_release_atom(self, realIndex):
        pass

    def _constraint_copy_needs_lut(self):
        return {'_StructureFactorConstraint__elementsPairs'       :'_StructureFactorConstraint__elementsPairs',
                '_StructureFactorConstraint__histogramSize'       :'_StructureFactorConstraint__histogramSize',
                '_StructureFactorConstraint__weightingScheme'     :'_StructureFactorConstraint__weightingScheme',
                '_StructureFactorConstraint__shellVolumes'        :'_StructureFactorConstraint__shellVolumes',
                '_StructureFactorConstraint__shellCenters'        :'_StructureFactorConstraint__shellCenters',
                '_StructureFactorConstraint__windowFunction'      :'_StructureFactorConstraint__windowFunction',
                '_StructureFactorConstraint__experimentalQValues' :'_StructureFactorConstraint__experimentalQValues',
                '_StructureFactorConstraint__experimentalSF'      :'_StructureFactorConstraint__experimentalSF',
                '_StructureFactorConstraint__Gr2SqMatrix'         :'_StructureFactorConstraint__Gr2SqMatrix',
                '_StructureFactorConstraint__minimumDistance'     :'_StructureFactorConstraint__minimumDistance',
                '_StructureFactorConstraint__maximumDistance'     :'_StructureFactorConstraint__maximumDistance',
                '_StructureFactorConstraint__bin'                 :'_StructureFactorConstraint__bin',
                '_ExperimentalConstraint__scaleFactor'            :'_ExperimentalConstraint__scaleFactor',
                '_ExperimentalConstraint__adjustScaleFactor'      :'_ExperimentalConstraint__adjustScaleFactor',
                '_ExperimentalConstraint__dataWeights'            :'_ExperimentalConstraint__dataWeights',
                '_ExperimentalConstraint__multiframePrior'        :'_ExperimentalConstraint__multiframePrior',
                '_ExperimentalConstraint__multiframeWeight'       :'_ExperimentalConstraint__multiframeWeight',
                '_ExperimentalConstraint__limits'                 :'_ExperimentalConstraint__limits',
                '_ExperimentalConstraint__limitsIndexStart'       :'_ExperimentalConstraint__limitsIndexStart',
                '_ExperimentalConstraint__limitsIndexEnd'         :'_ExperimentalConstraint__limitsIndexEnd',
                '_Constraint__used'                               :'_Constraint__used',
                '_Constraint__data'                               :'_Constraint__data',
                '_Constraint__state'                              :'_Constraint__state',
                '_Constraint__standardError'                      :'_Constraint__standardError',
                '_fittedScaleFactor'                              :'_fittedScaleFactor',
                '_usedDataWeights'                                :'_usedDataWeights',
                '_Engine__state'                                  :'_Engine__state',
                '_Engine__boxCoordinates'                         :'_Engine__boxCoordinates',
                '_Engine__basisVectors'                           :'_Engine__basisVectors',
                '_Engine__isPBC'                                  :'_Engine__isPBC',
                '_Engine__moleculesIndex'                         :'_Engine__moleculesIndex',
                '_Engine__elementsIndex'                          :'_Engine__elementsIndex',
                '_Engine__numberOfAtomsPerElement'                :'_Engine__numberOfAtomsPerElement',
                '_Engine__elements'                               :'_Engine__elements',
                '_Engine__numberDensity'                          :'_Engine__numberDensity',
                '_Engine__volume'                                 :'_Engine__volume',
                '_Engine__realCoordinates'                        :'_Engine__realCoordinates',
                '_atomsCollector'                                 :'_atomsCollector',
                ('engine','_atomsCollector')                      :'_atomsCollector',
               }

    def plot(self, xlabelParams={'xlabel':'$Q(\\AA^{-1})$', 'size':10},
                   ylabelParams={'ylabel':'$S(Q)$', 'size':10},
                   **kwargs):
        """
        Alias to ExperimentalConstraint.plot with additional parameters

        :Additional/Adjusted Parameters:
            #. xlabelParams (None, dict): modified matplotlib.axes.Axes.set_xlabel
               parameters.
            #. ylabelParams (None, dict): modified matplotlib.axes.Axes.set_ylabel
               parameters.
        """
        return super(StructureFactorConstraint, self).plot(xlabelParams= xlabelParams,
                                                           ylabelParams= ylabelParams,
                                                           **kwargs)



class ReducedStructureFactorConstraint(StructureFactorConstraint):
    """
    The Reduced Structure Factor that we will also note S(Q)
    is exactly the same quantity as the Structure Factor but with
    the slight difference that it is normalized to 0 rather than 1
    and therefore :math:`<S(Q)>=0`.

    The computation of S(Q) is done through a Sine inverse Fourier transform
    of the computed pair distribution function noted as G(r).

    .. math::

        S(Q) = \\frac{1}{Q} \\int_{0}^{\\infty} G(r) sin(Qr) dr

    The only reason why the Reduced Structure Factor is implemented, is because
    many experimental data are treated in this form. And it is just convenient
    not to manipulate the experimental data every time.
    """
    def _get_Sq_from_Gr(self, Gr):
        return np.sum(Gr.reshape((-1,1))*self.Gr2SqMatrix, axis=0)

    def _apply_scale_factor(self, Sq, scaleFactor):
        if scaleFactor != 1:
            Sq = scaleFactor*Sq
        return Sq

    def get_adjusted_scale_factor(self, experimentalData, modelData, dataWeights):
        """ dummy overload that does exactly the same thing
        """
        SF = self.scaleFactor
        # check to update scaleFactor
        if self.adjustScaleFactorFrequency:
            if not self.engine.accepted%self.adjustScaleFactorFrequency:
                SF = self.fit_scale_factor(experimentalData, modelData, dataWeights)
        return SF

    def plot(self, xlabelParams={'xlabel':'$Q(\\AA^{-1})$', 'size':10},
                   ylabelParams={'ylabel':'$S(Q)-1$', 'size':10},
                   **kwargs):
        """
        Alias to ExperimentalConstraint.plot with additional parameters

        :Additional/Adjusted Parameters:
            #. xlabelParams (None, dict): modified matplotlib.axes.Axes.set_xlabel
               parameters.
            #. ylabelParams (None, dict): modified matplotlib.axes.Axes.set_ylabel
               parameters.
        """
        return super(StructureFactorConstraint, self).plot(xlabelParams= xlabelParams,
                                                           ylabelParams= ylabelParams,
                                                           **kwargs)
