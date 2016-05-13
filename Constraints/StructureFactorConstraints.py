"""
StructureFactorConstraints contains classes for all constraints related experimental static structure factor functions.

.. inheritance-diagram:: fullrmc.Constraints.StructureFactorConstraints
    :parts: 1
"""

# standard libraries imports
import itertools

# external libraries imports
import numpy as np
from pdbParser.Utilities.Database import is_element_property, get_element_property
from pdbParser.Utilities.Collection import get_normalized_weighting

# fullrmc imports
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION, LOGGER
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, ExperimentalConstraint
from fullrmc.Core.pair_distribution_histogram import single_pair_distribution_histograms, multiple_pair_distribution_histograms, full_pair_distribution_histograms


class StructureFactorConstraint(ExperimentalConstraint):
    """
    It controls the Structure Factor noted as S(Q) and also called
    total-scattering structure function or Static Structure Factor. 
    S(Q) is a dimensionless quantity and the normalization is such 
    that the average value, :math:`<S(Q)>=1`.
    
    It is worth mentioning that S(Q) is nothing other than the normalized and 
    corrected diffraction pattern from all experimental artefacts powder.
    
    The computation of S(Q) is done through an inverse Sine Fourier transform  
    of the computed pair distribution function noted as G(r).
    
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
        #. weighting (string): The elements weighting.
        #. rmin (None, number): The minimum distance value to compute G(r) histogram.
           If None is given, rmin is computed as :math:`2 \\pi / Q_{max}`.
        #. rmax (None, number): The maximum distance value to compute G(r) histogram.
           If None is given, rmax is computed as :math:`2 \\pi / dQ`.
        #. dr (None, number): The distance bin value to compute G(r) histogram.
           If None is given, bin is computed as :math:`2 \\pi / (Q_{max}-Q_{min})`.
        #. scaleFactor (number): A normalization scale factor used to normalize the computed data to the experimental ones.
        #. adjustScaleFactor (list, tuple): Used to adjust fit or guess the best scale factor during EMC runtime. 
           It must be a list of exactly three entries.\n
           1. The frequency in number of generated moves of finding the best scale factor. 
              If 0 frequency is given, it means that the scale factor is fixed.
           2. The minimum allowed scale factor value.
           3. The maximum allowed scale factor value.
        #. windowFunction (None, numpy.ndarray): The window function to convolute with the computed pair distribution function
           of the system prior to comparing it with the experimental data. In general, the experimental pair
           distribution function G(r) shows artificial wrinkles, among others the main reason is because G(r) is computed
           by applying a sine Fourier transform to the experimental structure factor S(q). Therefore window function is
           used to best imitate the numerical artefacts in the experimental data.
        #. limits (None, tuple, list): The distance limits to compute the histograms.
           If None, the limits will be automatically set the the min and max distance of the experimental data.
           If not None, a tuple of exactly two items where the first is the minimum distance or None 
           and the second is the maximum distance or None.
    
    **NB**: If adjustScaleFactor first item (frequency) is 0, the scale factor will remain 
    untouched and the limits minimum and maximum won't be checked.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.StructureFactorConstraints import StructureFactorConstraint
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # create and add constraint
        SFC = StructureFactorConstraint(engine=None, experimentalData="sq.dat", weighting="atomicNumber")
        ENGINE.add_constraints(SFC)
    
    """
    def __init__(self, engine, experimentalData, 
                       dataWeights=None, weighting="atomicNumber", 
                       rmin=None, rmax=None, dr=None, 
                       scaleFactor=1.0, adjustScaleFactor=(0, 0.8, 1.2), 
                       windowFunction=None, limits=None):
        # initialize variables
        self.__limits              = limits
        self.__experimentalQValues = None
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
        super(StructureFactorConstraint, self).__init__(engine=engine, experimentalData=experimentalData, dataWeights=dataWeights, scaleFactor=scaleFactor, adjustScaleFactor=adjustScaleFactor)
        # set elements weighting
        self.set_weighting(weighting)
        self.__set_weighting_scheme()
        # set window function
        self.set_window_function(windowFunction)
        # set r parameters
        self.set_rmin(rmin)
        self.set_rmax(rmax)
        self.set_dr(dr)

    def __getstate__(self):
        # make sure that __Gr2SqMatrix is not pickled but saved to the disk as None
        D = {}
        for k, v in self.__dict__.iteritems():
            if k == "_StructureFactorConstraint__Gr2SqMatrix":
                v = None
            D[k] = v
        return D
        
    def __setstate__(self, d):
        # make sure to regenerate G(r) to S(q) matrix at loading time
        self.__dict__ = d
        self.__set_Gr_2_Sq_matrix()      
        
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
        
    def __set_used_data_weights(self, minDistIdx=None, maxDistIdx=None):
        # set used dataWeights
        if self.dataWeights is None:
            self._usedDataWeights = None
        else:
            if minDistIdx is None:
                minDistIdx = 0
            if maxDistIdx is None:
                maxDistIdx = self.experimentalData.shape[0]
            self._usedDataWeights  = np.copy(self.dataWeights[minDistIdx:maxDistIdx+1])
            assert np.sum(self._usedDataWeights), LOGGER.error("used points dataWeights are all zero.")
            self._usedDataWeights /= FLOAT_TYPE( np.sum(self._usedDataWeights) )
            self._usedDataWeights *= FLOAT_TYPE( len(self._usedDataWeights) ) 
            
    def __set_weighting_scheme(self):
        if self.engine is not None:
            self.__elementsPairs   = sorted(itertools.combinations_with_replacement(self.engine.elements,2))
            elementsWeights        = dict([(el,float(get_element_property(el,self.__weighting))) for el in self.engine.elements])
            self.__weightingScheme = get_normalized_weighting(numbers=self.engine.numberOfAtomsPerElement, weights=elementsWeights)
            for k, v in self.__weightingScheme.items():
                self.__weightingScheme[k] = FLOAT_TYPE(v)
        else:
            self.__elementsPairs   = None
            self.__weightingScheme = None
            
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
        # reset constraint
        self.reset_constraint()
        # reset sq matrix
        self.__set_Gr_2_Sq_matrix()
        
    @property
    def rmin(self):
        """ Get the given histogram minimum distance. """
        return self.__rmin
    
    @property
    def rmax(self):
        """ Get the given histogram maximum distance. """
        return self.__rmax
    
    @property
    def dr(self):
        """ Get the given histogram bin size."""
        return self.__dr
    
    @property
    def bin(self):
        """ Get the computed histogram distance bin size."""
        return self.__bin
          
    @property
    def minimumDistance(self):
        """ Get the computed histogram minimum distance. """
        return self.__minimumDistance
          
    @property
    def maximumDistance(self):
        """ Get the computed histogram maximum distance. """
        return self.__maximumDistance
        
    @property
    def qmin(self):
        """ Get the experimental data reciprocal distances minimum. """
        return self.__qmin
          
    @property
    def qmax(self):
        """ Get the experimental data reciprocal distances maximum. """
        return self.__qmax
    
    @property
    def dq(self):
        """ Get the experimental data reciprocal distances bin size. """
        return self.__experimentalQValues[1]-self.__experimentalQValues[0]
        
    @property
    def experimentalQValues(self):
        """ Gets the experimental data used q values. """
        return self.__experimentalQValues
        
    @property
    def histogramSize(self):
        """ Get the histogram size"""
        return self.__histogramSize
        
    @property
    def shellCenters(self):
        """ Get the shells center array"""
        return self.__shellCenters
        
    @property
    def shellVolumes(self):
        """ Get the shells volume array"""
        return self.__shellVolumes
        
    @property
    def experimentalSF(self):
        """ Get the experimental Structure Factor or S(q)"""
        return self.__experimentalSF
        
    @property
    def elementsPairs(self):
        """ Get elements pairs """
        return self.__elementsPairs
        
    @property
    def weightingScheme(self):
        """ Get elements weighting scheme. """
        return self.__weightingScheme
    
    @property
    def windowFunction(self):
        """ Get the window function. """
        return self.__windowFunction
    
    @property
    def limits(self):
        """ The histogram computation limits."""
        return self.__limits
     
    @property
    def Gr2SqMatrix(self):
        """ Get G(r) to S(q) transformation matrix."""
        return self.__Gr2SqMatrix
                
    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed", "update molecules indexes"):
            self.__set_weighting_scheme()
            # reset histogram
            if self.engine is not None:
                self.__set_histogram()
        elif message in("update boundary conditions",):
            self.reset_constraint()
            
    def set_rmin(self, rmin):
        """
        Set rmin value.
        
        :parameters:
            #. rmin (None, number): The minimum distance value to compute G(r) histogram.
               If None is given, rmin is computed as :math:`2 \\pi / Q_{max}`.
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
        # reset histogram
        self.__set_histogram()

    def set_rmax(self, rmax):
        """
        Set rmax value.
        
        :Parameters:
           #. rmax (None, number): The maximum distance value to compute G(r) histogram.
              If None is given, rmax is computed as :math:`2 \\pi / dQ`.
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
        # reset histogram
        self.__set_histogram()
    
    def set_dr(self, dr):
        """
        Set dr value.
        
        :Parameters:
            #. dr (None, number): The distance bin value to compute G(r) histogram.
               If None is given, bin is computed as :math:`2 \\pi / (Q_{max}-Q_{min})`.
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
        # reset histogram
        self.__set_histogram()
    
    def set_weighting(self, weighting):
        """
        Sets elements weighting. It must a valid entry of pdbParser atoms database
        
        :Parameters:
            #. weighting (string): The elements weighting.
        """
        assert is_element_property(weighting),LOGGER.error( "weighting is not a valid pdbParser atoms database entry")
        assert weighting != "atomicFormFactor", LOGGER.error("atomicFormFactor weighting is not allowed")
        self.__weighting = weighting
     
    def set_window_function(self, windowFunction):
        """
        Sets the window function.
        
        :Parameters:
             #. windowFunction (None, numpy.ndarray): The window function to convolute with the computed pair distribution function
                of the system prior to comparing it with the experimental data. In general, the experimental pair
                distribution function G(r) shows artificial wrinkles, among others the main reason is because G(r) is computed
                by applying a sine Fourier transform to the experimental structure factor S(q). Therefore window function is
                used to best imitate the numerical artefacts in the experimental data.
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
    
    def set_experimental_data(self, experimentalData):
        """
        Sets the constraint's experimental data.
        
        :Parameters:
            #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        """
        # get experimental data
        super(StructureFactorConstraint, self).set_experimental_data(experimentalData=experimentalData)
        # set limits
        self.set_limits(self.__limits)
    
    def set_limits(self, limits):
        """
        Set the reciprocal distance limits (qmin, qmax).
        
        :Parameters:
            #. limits (None, tuple, list): The distance limits to compute the histograms and compute with the experimental data.
               If None, the limits will be automatically set the the min and max reciprocal distance recorded in the experimental data.
               If not None, a tuple of minimum reciprocal distance (qmin) or None and maximum reciprocal distance (qmax) or None should be given.    
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
        if self.__limits[0] is None:
            minDistIdx = 0
        else:
            minDistIdx = (np.abs(self.experimentalData[:,0]-self.__limits[0])).argmin()
        if self.__limits[1] is None:
            maxDistIdx = self.experimentalData.shape[0]-1
        else:
            maxDistIdx =(np.abs(self.experimentalData[:,0]-self.__limits[1])).argmin()
        # set qvalues
        self.__experimentalQValues = self.experimentalData[minDistIdx:maxDistIdx+1,0].astype(FLOAT_TYPE)
        self.__experimentalSF      = self.experimentalData[minDistIdx:maxDistIdx+1,1].astype(FLOAT_TYPE)    
        # set qmin and qmax
        self.__qmin = self.__experimentalQValues[0]
        self.__qmax = self.__experimentalQValues[-1]
        # set used dataWeights
        self.__set_used_data_weights(minDistIdx=minDistIdx, maxDistIdx=maxDistIdx)   
        # reset constraint
        self.reset_constraint()
        # reset sq matrix
        self.__set_Gr_2_Sq_matrix()
        
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
        super(StructureFactorConstraint, self).set_data_weights(dataWeights=dataWeights)
        self.__set_used_data_weights()
        
    def compute_and_set_standard_error(self):
        """ Computes and sets the constraint's standardError."""
        # set standardError
        totalSQ = self.get_constraint_value()["sf_total"]
        self.set_standard_error(self.compute_standard_error(modelData = totalSQ))
 
    def check_experimental_data(self, experimentalData):
        """
        Check whether experimental data is correct.
 
        :Parameters:
            #. experimentalData (object): The experimental data to check.
 
        :Returns:
            #. result (boolean): Whether it is correct or not.
            #. message (str): Checking message that explains whats's wrong with the given data
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
        Compute the standard error (StdErr) as squared deviations
        between model computed data and the experimental ones. 
        
        .. math::
            StdErr = \\sum \\limits_{i}^{N} W_{i}(Y(X_{i})-F(X_{i}))^{2}
         
        Where:\n
        :math:`N` is the total number of experimental data points. \n
        :math:`W_{i}` is the data point weight. It becomes equivalent to 1 when dataWeights is set to None. \n
        :math:`Y(X_{i})` is the experimental data point :math:`X_{i}`. \n
        :math:`F(X_{i})` is the computed from the model data  :math:`X_{i}`. \n

        :Parameters:
            #. modelData (numpy.ndarray): The data to compare with the experimental one and compute the standard error.
            
        :Returns:
            #. standardError (number): The calculated standardError of the constraint.
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
    
    def __get_total_Sq(self, data):
        """
        This method is created just to speed up the computation of the total Sq upon fitting.
        """
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
        rho0 = (self.engine.numberOfAtoms/self.engine.volume).astype(FLOAT_TYPE)
        Gr   = (FLOAT_TYPE(4.)*PI*self.__shellCenters*rho0)*( Gr-1)
        # Compute S(q) from G(r)
        Sq = self._get_Sq_from_Gr(Gr)
        # Multiply by scale factor
        self._fittedScaleFactor = self.get_adjusted_scale_factor(self.__experimentalSF, Sq, self._usedDataWeights)
        Sq *= self._fittedScaleFactor
        # convolve total with window function
        if self.__windowFunction is not None:
            Sq = np.convolve(Sq, self.__windowFunction, 'same')
        return Sq
    
    def _get_constraint_value(self, data):
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
        output["sf_total"] = self.scaleFactor * Sq
        # convolve total with window function
        if self.__windowFunction is not None:
            output["sf"] = np.convolve(output["sf_total"], self.__windowFunction, 'same').astype(FLOAT_TYPE)
        else:
            output["sf"] = output["sf_total"]
        return output
    
    def get_constraint_value(self):
        """
        Compute all partial Pair Distribution Functions (PDFs). 
        
        :Returns:
            #. PDFs (dictionary): The PDFs dictionnary, where keys are the element wise intra and inter molecular PDFs and values are the computed PDFs.
        """
        if self.data is None:
            LOGGER.warn("data must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.data)
    
    def get_constraint_original_value(self):
        """
        Compute all partial Pair Distribution Functions (PDFs). 
        
        :Returns:
            #. PDFs (dictionary): The PDFs dictionnary, where keys are the element wise intra and inter molecular PDFs and values are the computed PDFs.
        """
        if self.originalData is None:
            LOGGER.warn("originalData must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.originalData)
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        intra,inter = full_pair_distribution_histograms( boxCoords        = self.engine.boxCoordinates,
                                                         basis            = self.engine.basisVectors,
                                                         moleculeIndex    = self.engine.moleculesIndexes,
                                                         elementIndex     = self.engine.elementsIndexes,
                                                         numberOfElements = self.engine.numberOfElements,
                                                         minDistance      = self.__minimumDistance,
                                                         maxDistance      = self.__maximumDistance,
                                                         histSize         = self.__histogramSize,
                                                         bin              = self.__bin )
        # update data
        self.set_data({"intra":intra, "inter":inter})
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        totalPDF = self.__get_total_Sq(self.data)
        self.set_standard_error(self.compute_standard_error(modelData = totalPDF))
    
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        intraM,interM = multiple_pair_distribution_histograms( indexes          = indexes,
                                                               boxCoords        = self.engine.boxCoordinates,
                                                               basis            = self.engine.basisVectors,
                                                               moleculeIndex    = self.engine.moleculesIndexes,
                                                               elementIndex     = self.engine.elementsIndexes,
                                                               numberOfElements = self.engine.numberOfElements,
                                                               minDistance      = self.__minimumDistance,
                                                               maxDistance      = self.__maximumDistance,
                                                               histSize         = self.__histogramSize,
                                                               bin              = self.__bin,
                                                               allAtoms         = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords        = self.engine.boxCoordinates[indexes],
                                                           basis            = self.engine.basisVectors,
                                                           moleculeIndex    = self.engine.moleculesIndexes[indexes],
                                                           elementIndex     = self.engine.elementsIndexes[indexes],
                                                           numberOfElements = self.engine.numberOfElements,
                                                           minDistance      = self.__minimumDistance,
                                                           maxDistance      = self.__maximumDistance,
                                                           histSize         = self.__histogramSize,
                                                           bin              = self.__bin )
        self.set_active_atoms_data_before_move( {"intra":intraM-intraF, "inter":interM-interF} )
        self.set_active_atoms_data_after_move(None)
    
    def compute_after_move(self, indexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new coordinates.
        """
        # change coordinates temporarily
        boxData = np.array(self.engine.boxCoordinates[indexes], dtype=FLOAT_TYPE)
        self.engine.boxCoordinates[indexes] = movedBoxCoordinates
        # calculate pair distribution function
        intraM,interM = multiple_pair_distribution_histograms( indexes          = indexes,
                                                               boxCoords        = self.engine.boxCoordinates,
                                                               basis            = self.engine.basisVectors,
                                                               moleculeIndex    = self.engine.moleculesIndexes,
                                                               elementIndex     = self.engine.elementsIndexes,
                                                               numberOfElements = self.engine.numberOfElements,
                                                               minDistance      = self.__minimumDistance,
                                                               maxDistance      = self.__maximumDistance,
                                                               histSize         = self.__histogramSize,
                                                               bin              = self.__bin,
                                                               allAtoms         = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords        = self.engine.boxCoordinates[indexes],
                                                           basis            = self.engine.basisVectors,
                                                           moleculeIndex    = self.engine.moleculesIndexes[indexes],
                                                           elementIndex     = self.engine.elementsIndexes[indexes],
                                                           numberOfElements = self.engine.numberOfElements,
                                                           minDistance      = self.__minimumDistance,
                                                           maxDistance      = self.__maximumDistance,
                                                           histSize         = self.__histogramSize,
                                                           bin              = self.__bin )
        self.set_active_atoms_data_after_move( {"intra":intraM-intraF, "inter":interM-interF} )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute standardError after move
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        totalPDF = self.__get_total_Sq({"intra":dataIntra, "inter":dataInter})
        self.set_after_move_standard_error( self.compute_standard_error(modelData = totalPDF) )
    
    def accept_move(self, indexes):
        """ 
        Accept move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
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
    
    def reject_move(self, indexes):
        """ 
        Reject move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update standardError
        self.set_after_move_standard_error( None )
   
    def plot(self, ax=None, intra=True, inter=True, 
                   xlabel=True, xlabelSize=16,
                   ylabel=True, ylabelSize=16,
                   legend=True, legendCols=2, legendLoc='best',
                   title=True, titleStdErr=True, titleScaleFactor=True):
        """ 
        Plot structure factor constraint.
        
        :Parameters:
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If ax is given, the figure won't be rendered and drawn.
               If None is given a new plot figure will be created and the figue will be rendered and drawn.
            #. intra (boolean): Whether to add intra-molecular pair distribution function features to the plot.
            #. inter (boolean): Whether to add inter-molecular pair distribution function features to the plot.
            #. xlabel (boolean): Whether to create x label.
            #. xlabelSize (number): The x label font size.
            #. ylabel (boolean): Whether to create y label.
            #. ylabelSize (number): The y label font size.
            #. legend (boolean): Whether to create the legend or not
            #. legendCols (integer): Legend number of columns.
            #. legendLoc (string): The legend location. Anything among
               'right', 'center left', 'upper right', 'lower right', 'best', 'center', 
               'lower left', 'center right', 'upper left', 'upper center', 'lower center'
               is accepted.
            #. title (boolean): Whether to create the title or not
            #. titleStdErr (boolean): Whether to show constraint standard error value in title.
            #. titleScaleFactor (boolean): Whether to show contraint's scale factor value in title.
        
        :Returns:
            #. axes (matplotlib Axes): The matplotlib axes.
        """
        # get constraint value
        output = self.get_constraint_value()
        if not len(output):
            LOGGER.warn("%s constraint data are not computed."%(self.__class__.__name__))
            return
        # import matplotlib
        import matplotlib.pyplot as plt
        # get axes
        if ax is None:
            AXES = plt.gca()
        else:
            AXES = ax   
        # Create plotting styles
        COLORS  = ["b",'g','r','c','y','m']
        MARKERS = ["",'.','+','^','|']
        INTRA_STYLES = [r[0] + r[1]for r in itertools.product(['--'], list(reversed(COLORS)))]
        INTRA_STYLES = [r[0] + r[1]for r in itertools.product(MARKERS, INTRA_STYLES)]
        INTER_STYLES = [r[0] + r[1]for r in itertools.product(['-'], COLORS)]
        INTER_STYLES = [r[0] + r[1]for r in itertools.product(MARKERS, INTER_STYLES)]
        # plot experimental
        AXES.plot(self.__experimentalQValues,self.experimentalSF, 'ro', label="experimental", markersize=7.5, markevery=1 )
        AXES.plot(self.__experimentalQValues, output["sf"], 'k', linewidth=3.0,  markevery=25, label="total" )
        # plot without window function
        if self.windowFunction is not None:
            AXES.plot(self.__experimentalQValues, output["sf_total"], 'k', linewidth=1.0,  markevery=5, label="total - no window" )
        # plot partials
        intraStyleIndex = 0
        interStyleIndex = 0
        for key, val in output.items():
            if key in ("sf_total", "sf"):
                continue
            elif "intra" in key and intra:
                AXES.plot(self.__experimentalQValues, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key )
                intraStyleIndex+=1
            elif "inter" in key and inter:
                AXES.plot(self.__experimentalQValues, val, INTER_STYLES[interStyleIndex], markevery=5, label=key )
                interStyleIndex+=1
        # plot legend
        if legend:
            AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
        # set title
        if title:
            t = ''
            if titleStdErr and self.standardError is not None:
                t += "$std$ $error=%.6f$ "%(self.standardError)
            if titleScaleFactor:
                t += " - "*(len(t)>0) + "$scale$ $factor=%.6f$"%(self.scaleFactor)
            if len(t):
                AXES.set_title(t)
        # set axis labels
        if xlabel:
            AXES.set_xlabel("$Q(\AA^{-1})$", size=xlabelSize)
        if ylabel:
            AXES.set_ylabel("$S(Q)$"  , size=ylabelSize)
        # set background color
        plt.gcf().patch.set_facecolor('white')
        #show
        if ax is None:
            plt.show()
        return AXES
        
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
    many experimental data are treated in this form. And it is just convenient not
    to manipulate the experimental data every time.
    """
    def _get_Sq_from_Gr(self, Gr):
        return np.sum(Gr.reshape((-1,1))*self.Gr2SqMatrix, axis=0)
        
        