"""
PairDistributionConstraints contains classes for all constraints related experimental pair distribution functions.

.. inheritance-diagram:: fullrmc.Constraints.PairDistributionConstraints
    :parts: 1
"""

# standard libraries imports
import itertools

# external libraries imports
import numpy as np
from pdbParser.Utilities.Database import is_element_property, get_element_property
from pdbParser.Utilities.Collection import get_normalized_weighting

# fullrmc imports
from fullrmc import log
from fullrmc.Globals import INT_TYPE, FLOAT_TYPE, PI, PRECISION
from fullrmc.Core.Collection import is_number, is_integer, get_path
from fullrmc.Core.Constraint import Constraint, ExperimentalConstraint
from fullrmc.Core.pair_distribution_histogram import single_pair_distribution_histograms, multiple_pair_distribution_histograms, full_pair_distribution_histograms


class PairDistributionConstraint(ExperimentalConstraint):
    """
    It controls the total pair distribution function (pdf) of the system noted as G(r). 
    The pair distribution function is the directly calculated quantity from a powder diffraction
    experiments. It is obtained from the experimentally determined total-scattering structure 
    function S(Q), by a Sine Fourier transform. pdf tells the probability of finding 
    atomic pairs separated by the real space distance r.
    The mathematical definition of the G(r) is:

    .. math::
        
        G(r) = \\frac{1}{\\pi} \\int_{0}^{\\infty} Q [S(Q)-1]sin(Qr)dQ = 4 \\pi [\\rho(r) - \\rho_{0}] \n
        G(r) = \\frac{R(r)}{r} - 4 \\pi r \\rho_{0} \n
        R(r) = 4 \\pi r^2 \\rho(r) =\\frac{1}{N} \\sum \\limits_{i}^{N} \\sum \\limits_{j \\neq i}^{N} \\frac{b_i b_j}{\\langle b \\rangle ^2} \\delta ( r - r_{ij} )  
    
    Where:\n
    :math:`Q` is the momentum transfer. \n
    :math:`r` is the distance between two atoms. \n
    :math:`\\rho(r)` is the sperical average defined as :math:`\\int n(r^{'}).n(r-r^{'}) dr`. \n
    :math:`\\rho_{0}` is the  average number density of the samples. \n
    :math:`R(r)` is the radial distribution function (rdf). \n
    :math:`N` is the total number of atoms. \n
    :math:`b_i` is the scattering length for atom i. \n
    :math:`r_{ij}` is the distance between atoms i and j. \n
    :math:`\\langle b \\rangle` is the average scattering length over all atoms  \n
    :math:`\\sum \\limits_{j \\neq i}^{N} \\delta ( r - r_{ij} )` is the probability density 
    of some atoms being r distant from each other.

    **NB**: pair distribution function G(r) and the pair correlation function g(r) are directly 
    related as in the following :math:`G(r)=r(g(r)-1)`
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        #. weighting (string): The elements weighting.
        #. scaleFactor (number): A normalization scale factor used to normalize the computed data to the experimental ones.
        #. windowFunction (None, numpy.ndarray): The window function to convolute with the computed pair distribution function
           of the system prior to comparing it with the experimental data. In general, the experimental pair
           distribution function G(r) shows artificial wrinkles, among others the main reason is because G(r) is computed
           by applying a sine Fourier transform to the experimental structure factor S(q). Therefore window function is
           used to best imitate the numerical artefacts in the experimental data.
    """
    def __init__(self, engine, experimentalData, weighting="atomicNumber", scaleFactor=1.0, windowFunction=None):
        # initialize constraint
        super(PairDistributionConstraint, self).__init__(engine=engine, experimentalData=experimentalData)
        # set elements weighting
        self.set_weighting(weighting)
        # set window function
        self.set_window_function(windowFunction)
        # set window function
        self.set_scale_factor(scaleFactor)
        
    @property
    def bin(self):
        """ Gets the experimental data distances bin. """
        return self.__bin
          
    @property
    def minimumDistance(self):
        """ Gets the experimental data distances minimum. """
        return self.__minimumDistance
          
    @property
    def maximumDistance(self):
        """ Gets the experimental data distances maximum. """
        return self.__maximumDistance
          
    @property
    def histogramSize(self):
        """ Get the histogram size"""
        return self.__histogramSize
    
    @property
    def experimentalDistances(self):
        """ Get the experimental distances array"""
        return self.__experimentalDistances
        
    @property
    def shellsCenter(self):
        """ Get the shells center array"""
        return self.__shellsCenter
        
    @property
    def shellsVolumes(self):
        """ Get the shells volume array"""
        return self.__shellsVolumes
        
    @property
    def experimentalPDF(self):
        """ Get the experimental pdf"""
        return self.__experimentalPDF
        
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
    def scaleFactor(self):
        """ Get the scaleFactor. """
        return self.__scaleFactor
        
    def listen(self, message, argument=None):
        """   
        Listens to any message sent from the Broadcaster.
        
        :Parameters:
            #. message (object): Any python object to send to constraint's listen method.
            #. argument (object): Any type of argument to pass to the listeners.
        """
        if message in("engine changed", "update molecules indexes"):
            if self.engine is not None:
                self.__elementsPairs   = sorted(itertools.combinations_with_replacement(self.engine.elements,2))
                elementsWeights        = dict([(el,float(get_element_property(el,self.__weighting))) for el in self.engine.elements])
                self.__weightingScheme = get_normalized_weighting(numbers=self.engine.numberOfAtomsPerElement, weights=elementsWeights)
                for k, v in self.__weightingScheme.items():
                    self.__weightingScheme[k] = FLOAT_TYPE(v)
            else:
                self.__elementsPairs   = None
                self.__weightingScheme = None
        elif message in("update boundary conditions",):
            self.__initialize_constraint__()
            
    def set_weighting(self, weighting):
        """
        Sets elements weighting. It must a valid entry of pdbParser atoms database
        
        :Parameters:
            #. weighting (string): The elements weighting.
        """
        assert is_element_property(weighting),log.LocalLogger("fullrmc").logger.error( "weighting is not a valid pdbParser atoms database entry")
        assert weighting != "atomicFormFactor", log.LocalLogger("fullrmc").logger.error("atomicFormFactor weighting is not allowed")
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
            assert isinstance(windowFunction, np.ndarray), log.LocalLogger("fullrmc").logger.error("windowFunction must be a numpy.ndarray")
            assert windowFunction.dtype.type is FLOAT_TYPE, log.LocalLogger("fullrmc").logger.error("windowFunction type must be %s"%FLOAT_TYPE)
            assert len(windowFunction.shape) == 1, log.LocalLogger("fullrmc").logger.error("experimentalData must be of dimension 2")
            # normalize window function
            windowFunction /= np.sum(windowFunction)
        # set windowFunction
        self.__windowFunction = windowFunction
    
    def set_scale_factor(self, scaleFactor):
        """
        Sets the scale factor.
        
        :Parameters:
             #. scaleFactor (string): A normalization scale factor used to normalize the computed data to the experimental ones.
        """
        assert is_number(scaleFactor), log.LocalLogger("fullrmc").logger.error("scaleFactor must be a number")
        self.__scaleFactor = FLOAT_TYPE(scaleFactor)
    
    def set_experimental_data(self, experimentalData):
        """
        Sets the constraint's experimental data.
        
        :Parameters:
            #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        """
        # get experimental data
        super(PairDistributionConstraint, self).set_experimental_data(experimentalData=experimentalData)
        self.__bin                   = FLOAT_TYPE(self.experimentalData[1,0] - self.experimentalData[0,0])
        self.__minimumDistance       = FLOAT_TYPE(self.experimentalData[0,0] - self.__bin/2. )
        self.__maximumDistance       = FLOAT_TYPE(self.experimentalData[-1,0]+ self.__bin/2. )
        self.__histogramSize         = INT_TYPE((self.__maximumDistance-self.__minimumDistance+self.__bin)/self.__bin)-INT_TYPE(1)
        self.__edges                 = np.array([self.__minimumDistance+idx*self.__bin for idx in xrange(self.__histogramSize+INT_TYPE(1))], dtype=FLOAT_TYPE)
        self.__experimentalDistances = self.experimentalData[:,0]
        self.__shellsCenter          = (self.__edges[1:]+self.__edges[0:-1])/FLOAT_TYPE(2.)
        self.__shellsVolumes         = FLOAT_TYPE(4.0)*PI*self.__shellsCenter*self.__shellsCenter*self.__bin
        self.__experimentalPDF       = self.experimentalData[:,1]    
           
        # check for experimental distances input error  
        for diff in self.__shellsCenter-self.__experimentalDistances:
            assert abs(diff)<=PRECISION, "experimental data distances are not coherent"
            
    def check_experimental_data(self, experimentalData):
        if not isinstance(experimentalData, np.ndarray):
            return False, "experimentalData must be a numpy.ndarray"
        if experimentalData.dtype.type is not FLOAT_TYPE:
            return False, "experimentalData type must be %s"%FLOAT_TYPE
        if len(experimentalData.shape) !=2:
            return False, "experimentalData must be of dimension 2"
        if experimentalData.shape[1] !=2:
            return False, "experimentalData must have only 2 columns"
        # check distances order
        if np.sum( np.array(sorted(experimentalData[0]), dtype=FLOAT_TYPE)-experimentalData[0] )>PRECISION:
            return False, "experimentalData distances are not sorted in order"
        if experimentalData[0][0]<0:
            return False, "experimentalData distances min value is found negative"
        bin  = experimentalData[1,0] -experimentalData[0,0]
        bins = experimentalData[1:,0]-experimentalData[0:-1,0]
        for b in bins:
            if np.abs(b-bin)>PRECISION:
                return False, "experimentalData distances bins are found not coherent"
        # data format is correct
        return True, ""

    def compute_chi_square(self, data):
        """ 
        Compute the chi square between data and the experimental one. 
        
        :Parameters:
            #. data (numpy.array): The data to compare with the experimental one and compute the chi square.
            
        :Returns:
            #. chiSquare (number): The calculated chiSquare multiplied by the contribution factor of the constraint.
        """
        # compute difference
        diff = self.__experimentalPDF-data
        # return chi square
        return np.add.reduce((diff)**2)*self.contribution
        
    def _get_constraint_value(self, data):
        ###################### THIS SHOULD BE OPTIMIZED ######################
        #import time
        #startTime = time.clock()
        output = {}
        for pair in self.__elementsPairs:
            output["rdf_intra_%s-%s" % pair] = np.zeros(self.__histogramSize, dtype=np.float32)
            output["rdf_inter_%s-%s" % pair] = np.zeros(self.__histogramSize, dtype=np.float32)
            output["rdf_total_%s-%s" % pair] = np.zeros(self.__histogramSize, dtype=np.float32)
        output["pdf_total"] = np.zeros(self.__histogramSize, dtype=np.float32)
        for pair in self.__elementsPairs:
            # get weighting scheme
            w = self.__weightingScheme.get(pair[0]+"-"+pair[1], None)
            if w is None:
                w = self.__weightingScheme[pair[1]+"-"+pair[0]]
            # get number of atoms per element
            ni = self.engine.numberOfAtomsPerElement[pair[0]]
            nj = self.engine.numberOfAtomsPerElement[pair[1]]
            # get index of element
            idi = self.engine.elements.index(pair[0])
            idj = self.engine.elements.index(pair[1])
            # get nij
            if idi == idj:
                nij = ni*(ni-1)/2.0 
                output["rdf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] 
                output["rdf_inter_%s-%s" % pair] += data["inter"][idi,idj,:]                
            else:
                nij = ni*nj
                output["rdf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] + data["intra"][idj,idi,:]
                output["rdf_inter_%s-%s" % pair] += data["inter"][idi,idj,:] + data["inter"][idj,idi,:]
            # calculate intensityFactor
            intensityFactor = (self.engine.volume*w)/(nij*self.__shellsVolumes)
            # divide by factor
            output["rdf_intra_%s-%s" % pair] *= intensityFactor
            output["rdf_inter_%s-%s" % pair] *= intensityFactor
            output["rdf_total_%s-%s" % pair]  = output["rdf_intra_%s-%s" % pair] + output["rdf_inter_%s-%s" % pair]
            output["pdf_total_%s-%s" % pair]  = output["rdf_total_%s-%s" % pair]
            # normalize to g(r)
            output["pdf_total_%s-%s" % pair]  = (output["pdf_total_%s-%s" % pair]-w)*self.__shellsCenter
            output["pdf_total"]              += self.__scaleFactor*output["pdf_total_%s-%s" % pair] 
        # convolve total with window function
        if self.__windowFunction is not None:
            output["pdf"] = np.convolve(output["pdf_total"], self.__windowFunction, 'same')
        else:
            output["pdf"] = output["pdf_total"]
        #t = time.clock()-startTime
        #print "%.7f(s) -->  %.7f(Ms)"%(t, 1000000*t)
        return output
    
    def get_constraint_value(self):
        """
        Compute all partial Pair Distribution Functions (PDFs). 
        
        :Returns:
            #. PDFs (dictionary): The PDFs dictionnary, where keys are the element wise intra and inter molecular PDFs and values are the computed PDFs.
        """
        if self.data is None:
            log.LocalLogger("fullrmc").logger.warn("data must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.data)
    
    def get_constraint_original_value(self):
        """
        Compute all partial Pair Distribution Functions (PDFs). 
        
        :Returns:
            #. PDFs (dictionary): The PDFs dictionnary, where keys are the element wise intra and inter molecular PDFs and values are the computed PDFs.
        """
        if self.data is None:
            log.LocalLogger("fullrmc").logger.warn("data must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.originalData)
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        intra,inter = full_pair_distribution_histograms( boxCoords=self.engine.boxCoordinates,
                                                         basis=self.engine.basisVectors,
                                                         moleculeIndex = self.engine.moleculesIndexes,
                                                         elementIndex = self.engine.elementsIndexes,
                                                         numberOfElements = self.engine.numberOfElements,
                                                         minDistance=self.__minimumDistance,
                                                         maxDistance=self.__maximumDistance,
                                                         histSize=self.__histogramSize,
                                                         bin=self.__bin )
        # update data
        self.set_data({"intra":intra, "inter":inter})
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set chiSquare
        totalPDF = self.get_constraint_value()["pdf_total"]
        self.set_chi_square(self.compute_chi_square(data = totalPDF))
    
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        intraM,interM = multiple_pair_distribution_histograms( indexes = indexes,
                                                               boxCoords=self.engine.boxCoordinates,
                                                               basis=self.engine.basisVectors,
                                                               moleculeIndex = self.engine.moleculesIndexes,
                                                               elementIndex = self.engine.elementsIndexes,
                                                               numberOfElements = self.engine.numberOfElements,
                                                               minDistance=self.__minimumDistance,
                                                               maxDistance=self.__maximumDistance,
                                                               histSize=self.__histogramSize,
                                                               bin=self.__bin,
                                                               allAtoms = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords=self.engine.boxCoordinates[indexes],
                                                             basis=self.engine.basisVectors,
                                                             moleculeIndex = self.engine.moleculesIndexes[indexes],
                                                             elementIndex = self.engine.elementsIndexes[indexes],
                                                             numberOfElements = self.engine.numberOfElements,
                                                             minDistance=self.__minimumDistance,
                                                             maxDistance=self.__maximumDistance,
                                                             histSize=self.__histogramSize,
                                                             bin=self.__bin )
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
        intraM,interM = multiple_pair_distribution_histograms( indexes = indexes,
                                                               boxCoords=self.engine.boxCoordinates,
                                                               basis=self.engine.basisVectors,
                                                               moleculeIndex = self.engine.moleculesIndexes,
                                                               elementIndex = self.engine.elementsIndexes,
                                                               numberOfElements = self.engine.numberOfElements,
                                                               minDistance=self.__minimumDistance,
                                                               maxDistance=self.__maximumDistance,
                                                               histSize=self.__histogramSize,
                                                               bin=self.__bin,
                                                               allAtoms = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords=self.engine.boxCoordinates[indexes],
                                                           basis=self.engine.basisVectors,
                                                           moleculeIndex = self.engine.moleculesIndexes[indexes],
                                                           elementIndex = self.engine.elementsIndexes[indexes],
                                                           numberOfElements = self.engine.numberOfElements,
                                                           minDistance=self.__minimumDistance,
                                                           maxDistance=self.__maximumDistance,
                                                           histSize=self.__histogramSize,
                                                           bin=self.__bin )
        self.set_active_atoms_data_after_move( {"intra":intraM-intraF, "inter":interM-interF} )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute chiSquare after move
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        data = self.data
        # change temporarily data
        self.set_data( {"intra":dataIntra, "inter":dataInter} )
        totalPDF = self.get_constraint_value()["pdf_total"]
        self.set_after_move_chi_square( self.compute_chi_square(data = totalPDF) )
        # change back data
        self.set_data( data )
    
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
        # update chiSquare
        self.set_chi_square( self.afterMoveChiSquare )
        self.set_after_move_chi_square( None )
    
    def reject_move(self, indexes):
        """ 
        Reject move
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # update chiSquare
        self.set_after_move_chi_square( None )



#class StructureFactor(PairDistributionFunction):
#     pass


    
    
    


    
    
            