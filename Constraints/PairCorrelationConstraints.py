"""
PairCorrelationConstraint contains classes for all constraints related experimental pair correlation functions.

.. inheritance-diagram:: fullrmc.Constraints.PairCorrelationConstraints
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
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint

class PairCorrelationConstraint(PairDistributionConstraint):
    """
    It controls the total pair correlation function (pcf) of the system noted as g(r).
    The pair correlation function g(r) and pair distribution function G(r) are directly 
    related as in the following :math:`g(r)=1+(\\frac{G(r)}{r})`
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        #. weighting (string): The elements weighting.
        #. scaleFactor (string): A normalization scale factor used to normalize the computed data to the experimental ones.
        #. windowFunction (None, numpy.ndarray): The window function to convolute with the computed pair distribution function
           of the system prior to comparing it with the experimental data. In general, the experimental pair
           distribution function G(r) shows artificial wrinkles, among others the main reason is because G(r) is computed
           by applying a sine Fourier transform to the experimental structure factor S(q). Therefore window function is
           used to best imitate the numerical artefacts in the experimental data.
        #. limits (None, tuple, list): The distance limits to compute the histograms.
           If None, the limits will be automatically set the the min and max distance recorded in the experimental data.
           If not None, a tuple of exactly two items where the first is of minimum distance or None 
           and the second is the maximum distance or None.
    """
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
        bin  = experimentalData[1,0] -experimentalData[0,0]
        bins = experimentalData[1:,0]-experimentalData[0:-1,0]
        for b in bins:
            if np.abs(b-bin)>PRECISION:
                return False, "experimentalData distances bins are found not coherent"
        # data format is correct
        return True, ""
        
    def _get_constraint_value(self, data):
        ###################### THIS SHOULD BE OPTIMIZED ######################
        #import time
        #startTime = time.clock()
        output = {}
        for pair in self.elementsPairs:
            output["pcf_intra_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=np.float32)
            output["pcf_inter_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=np.float32)
            output["pcf_total_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=np.float32)
        output["pcf_total"] = np.zeros(self.histogramSize, dtype=np.float32)
        for pair in self.elementsPairs:
            # get weighting scheme
            w = self.weightingScheme.get(pair[0]+"-"+pair[1], None)
            if w is None:
                w = self.weightingScheme[pair[1]+"-"+pair[0]]
            # get number of atoms per element
            ni = self.engine.numberOfAtomsPerElement[pair[0]]
            nj = self.engine.numberOfAtomsPerElement[pair[1]]
            # get index of element
            idi = self.engine.elements.index(pair[0])
            idj = self.engine.elements.index(pair[1])
            # get nij
            if idi == idj:
                nij = ni*(ni-1)/2.0  
                output["pcf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] 
                output["pcf_inter_%s-%s" % pair] += data["inter"][idi,idj,:]                 
            else:
                nij = ni*nj
                output["pcf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] + data["intra"][idj,idi,:]
                output["pcf_inter_%s-%s" % pair] += data["inter"][idi,idj,:] + data["inter"][idj,idi,:]   
            # calculate intensityFactor
            intensityFactor = (self.engine.volume*w)/(nij*self.shellsVolumes)
            # divide by factor
            output["pcf_intra_%s-%s" % pair] *= intensityFactor
            output["pcf_inter_%s-%s" % pair] *= intensityFactor
            output["pcf_total_%s-%s" % pair]  = output["pcf_intra_%s-%s" % pair] + output["pcf_inter_%s-%s" % pair]
            output["pcf_total"] += self.scaleFactor*output["pcf_total_%s-%s" % pair]
        # convolve total with window function
        if self.windowFunction is not None:
            output["pcf"] = np.convolve(output["pcf_total"], self.windowFunction, 'same')
        else:
            output["pcf"] = output["pcf_total"]
        #t = time.clock()-startTime
        #print "%.7f(s) -->  %.7f(Ms)"%(t, 1000000*t)
        return output
       
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        intra,inter = full_pair_distribution_histograms( boxCoords=self.engine.boxCoordinates,
                                                         basis=self.engine.basisVectors,
                                                         moleculeIndex=self.engine.moleculesIndexes,
                                                         elementIndex=self.engine.elementsIndexes,
                                                         numberOfElements=self.engine.numberOfElements,
                                                         minDistance=self.minimumDistance,
                                                         maxDistance=self.maximumDistance,
                                                         histSize=self.histogramSize,
                                                         bin=self.bin )
        # update data
        self.set_data({"intra":intra, "inter":inter})
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set squaredDeviations
        totalPCF = self.get_constraint_value()["pcf_total"]
        self.set_squared_deviations(self.compute_squared_deviations(data = totalPCF))
    
    def compute_before_move(self, indexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        intraM,interM = multiple_pair_distribution_histograms( indexes=indexes,
                                                               boxCoords=self.engine.boxCoordinates,
                                                               basis=self.engine.basisVectors,
                                                               moleculeIndex=self.engine.moleculesIndexes,
                                                               elementIndex=self.engine.elementsIndexes,
                                                               numberOfElements=self.engine.numberOfElements,
                                                               minDistance=self.minimumDistance,
                                                               maxDistance=self.maximumDistance,
                                                               histSize=self.histogramSize,
                                                               bin=self.bin,
                                                               allAtoms = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords=self.engine.boxCoordinates[indexes],
                                                           basis=self.engine.basisVectors,
                                                           moleculeIndex=self.engine.moleculesIndexes[indexes],
                                                           elementIndex=self.engine.elementsIndexes[indexes],
                                                           numberOfElements=self.engine.numberOfElements,
                                                           minDistance=self.minimumDistance,
                                                           maxDistance=self.maximumDistance,
                                                           histSize=self.histogramSize,
                                                           bin=self.bin )
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
        intraM,interM = multiple_pair_distribution_histograms( indexes=indexes,
                                                               boxCoords=self.engine.boxCoordinates,
                                                               basis=self.engine.basisVectors,
                                                               moleculeIndex=self.engine.moleculesIndexes,
                                                               elementIndex=self.engine.elementsIndexes,
                                                               numberOfElements=self.engine.numberOfElements,
                                                               minDistance=self.minimumDistance,
                                                               maxDistance=self.maximumDistance,
                                                               histSize=self.histogramSize,
                                                               bin=self.bin,
                                                               allAtoms = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords=self.engine.boxCoordinates[indexes],
                                                           basis=self.engine.basisVectors,
                                                           moleculeIndex=self.engine.moleculesIndexes[indexes],
                                                           elementIndex=self.engine.elementsIndexes[indexes],
                                                           numberOfElements=self.engine.numberOfElements,
                                                           minDistance=self.minimumDistance,
                                                           maxDistance=self.maximumDistance,
                                                           histSize=self.histogramSize,
                                                           bin=self.bin )
        self.set_active_atoms_data_after_move( {"intra":intraM-intraF, "inter":interM-interF} )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute squaredDeviations after move
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        data = self.data
        # change temporarily data
        self.set_data( {"intra":dataIntra, "inter":dataInter} )
        totalPCF = self.get_constraint_value()["pcf_total"]
        # set after move squared deviations
        self.set_after_move_squared_deviations( self.compute_squared_deviations(data = totalPCF) )
        # change back data
        self.set_data( data )
        
        