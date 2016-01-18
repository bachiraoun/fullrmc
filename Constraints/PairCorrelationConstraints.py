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
    pcf tells the probability of finding atomic pairs separated by the real space distance r.
    Theoretically g(r) oscillates around 1. Also :math:`g(r) \\rightarrow 1` when :math:`r \\rightarrow \\infty` 
    and it takes the exact value of zero for :math:`r` shorter than the distance of the closest possible 
    approach of pairs of atoms.\n
    The pair correlation function g(r) and pair distribution function G(r) are directly 
    related as in the following :math:`g(r)=1+(\\frac{G(r)}{4 \\pi \\rho_{0} r})`.
    
    g(r) is calculated after binning all pair atomic distances into a weighted
    histograms of values :math:`n(r)` from which local number densities are computed as in the following.
    
    .. math::
        g(r) = \\sum \\limits_{i,j}^{N} w_{i,j} \\frac{\\rho_{i,j}(r)}{\\rho_{0}} 
             = \\sum \\limits_{i,j}^{N} w_{i,j} \\frac{n_{i,j}(r) / v(r)}{N_{i,j} / V} 
    
    Where:\n
    :math:`r` is the distance between two atoms. \n
    :math:`\\rho_{i,j}(r)` is the pair density function of atoms i and j. \n
    :math:`\\rho_{0}` is the  average number density of the system. \n
    :math:`w_{i,j}` is the relative weighting of atom types i and j. \n
    :math:`N` is the total number of atoms. \n
    :math:`V` is the volume of the system. \n
    :math:`n_{i,j}(r)` is the number of atoms i neighbouring j at a distance r. \n
    :math:`v(r)` is the annulus volume at distance r and of thickness dr. \n
    :math:`N_{i,j}` is the total number of atoms i and j in the system. \n
    
    :Parameters:
        #. engine (None, fullrmc.Engine): The constraint RMC engine.
        #. experimentalData (numpy.ndarray, string): The experimental data as numpy.ndarray or string path to load data using numpy.loadtxt.
        #. dataWeights (None, numpy.ndarray): A weights array of the same number of points of experimentalData used in the constraint's squared deviations computation.
           Therefore particular fitting emphasis can be put on different data points that might be considered as more or less
           important in order to get a reasonable and plausible modal.\n
           If None is given, all data points are considered of the same importance in the computation of the constraint's squared deviations.\n
           If numpy.ndarray is given, all weights must be positive and all zeros weighted data points won't contribute to the 
           total constraint's squared deviations. At least a single weight point is required to be non-zeros and the weights 
           array will be automatically scaled upon setting such as the the sum of all the weights is equal to the number of data points.       
        #. weighting (string): The elements weighting.
        #. scaleFactor (string): A normalization scale factor used to normalize the computed data to the experimental ones.
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
           If None, the limits will be automatically set the the min and max distance recorded in the experimental data.
           If not None, a tuple of exactly two items where the first is of minimum distance or None 
           and the second is the maximum distance or None.
    
    **NB**: If adjustScaleFactor first item (frequency) is 0, the scale factor will remain 
    untouched and the limits minimum and maximum won't be checked.
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.PairCorrelationConstraints import PairCorrelationConstraint
        
        # create engine 
        ENGINE = Engine(pdb='system.pdb')
        
        # create and add constraint
        PCC = PairCorrelationConstraint(engine=None, experimentalData="pcf.dat", weighting="atomicNumber")
        ENGINE.add_constraints(PCC)
        
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
    
    def __get_total_gr(self, data):
        """
        This method is created just to speed up the computation of the total gr upon fitting.
        """
        gr = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
        for pair in self.elementsPairs:
            # get weighting scheme
            wij = self.weightingScheme.get(pair[0]+"-"+pair[1], None)
            if wij is None:
                wij = self.weightingScheme[pair[1]+"-"+pair[0]]
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
                gr += wij*nij/Dij      
            else:
                Nij = ni*nj
                Dij = Nij/self.engine.volume
                nij = data["intra"][idi,idj,:]+data["intra"][idj,idi,:] + data["inter"][idi,idj,:]+data["inter"][idj,idi,:]   
                gr += wij*nij/Dij
        # Multiply by scale factor and deviding by shells volume
        gr /= self.shellsVolumes
        # Multiply by scale factor
        self._fittedScaleFactor = self.get_adjusted_scale_factor(self.experimentalPDF, gr, self._usedDataWeights)
        gr *= self._fittedScaleFactor
        # convolve total with window function
        if self.windowFunction is not None:
            gr = np.convolve(gr, self.windowFunction, 'same')
        return gr

    def _get_constraint_value(self, data):
        # http://erice2011.docking.org/upload/Other/Billinge_PDF/03-ReadingMaterial/BillingePDF2011.pdf     page 6
        output = {}
        for pair in self.elementsPairs:
            output["rdf_intra_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
            output["rdf_inter_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
            output["rdf_total_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
        output["pcf_total"] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
        for pair in self.elementsPairs:
            # get weighting scheme
            wij = self.weightingScheme.get(pair[0]+"-"+pair[1], None)
            if wij is None:
                wij = self.weightingScheme[pair[1]+"-"+pair[0]]
            # get number of atoms per element
            ni = self.engine.numberOfAtomsPerElement[pair[0]]
            nj = self.engine.numberOfAtomsPerElement[pair[1]]
            # get index of element
            idi = self.engine.elements.index(pair[0])
            idj = self.engine.elements.index(pair[1])
            # get Nij
            if idi == idj:
                Nij = ni*(ni-1)/2.0 
                output["rdf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] 
                output["rdf_inter_%s-%s" % pair] += data["inter"][idi,idj,:]                
            else:
                Nij = ni*nj
                output["rdf_intra_%s-%s" % pair] += data["intra"][idi,idj,:] + data["intra"][idj,idi,:]
                output["rdf_inter_%s-%s" % pair] += data["inter"][idi,idj,:] + data["inter"][idj,idi,:]
            # calculate intensityFactor
            intensityFactor = (self.engine.volume*wij)/(Nij*self.shellsVolumes)
            # multiply by intensityFactor
            output["rdf_intra_%s-%s" % pair] *= intensityFactor
            output["rdf_inter_%s-%s" % pair] *= intensityFactor
            output["rdf_total_%s-%s" % pair]  = output["rdf_intra_%s-%s" % pair] + output["rdf_inter_%s-%s" % pair]
            output["pcf_total"]              += output["rdf_total_%s-%s" % pair]
        # multiply total by scale factor
        output["pcf_total"] *= self.scaleFactor
        # convolve total with window function
        if self.windowFunction is not None:
            output["pcf"] = np.convolve(output["pcf_total"], self.windowFunction, 'same')
        else:
            output["pcf"] = output["pcf_total"]
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
        
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        intra,inter = full_pair_distribution_histograms( boxCoords        = self.engine.boxCoordinates,
                                                         basis            = self.engine.basisVectors,
                                                         moleculeIndex    = self.engine.moleculesIndexes,
                                                         elementIndex     = self.engine.elementsIndexes,
                                                         numberOfElements = self.engine.numberOfElements,
                                                         minDistance      = self.minimumDistance,
                                                         maxDistance      = self.maximumDistance,
                                                         histSize         = self.histogramSize,
                                                         bin              = self.bin )
        # update data
        self.set_data({"intra":intra, "inter":inter})
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set squaredDeviations
        totalPCF = self.__get_total_gr(self.data)
        self.set_squared_deviations(self.compute_squared_deviations(modelData = totalPCF))
    
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
                                                               minDistance      = self.minimumDistance,
                                                               maxDistance      = self.maximumDistance,
                                                               histSize         = self.histogramSize,
                                                               bin              = self.bin,
                                                               allAtoms         = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords        = self.engine.boxCoordinates[indexes],
                                                           basis            = self.engine.basisVectors,
                                                           moleculeIndex    = self.engine.moleculesIndexes[indexes],
                                                           elementIndex     = self.engine.elementsIndexes[indexes],
                                                           numberOfElements = self.engine.numberOfElements,
                                                           minDistance      = self.minimumDistance,
                                                           maxDistance      = self.maximumDistance,
                                                           histSize         = self.histogramSize,
                                                           bin              = self.bin )
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
                                                               minDistance      = self.minimumDistance,
                                                               maxDistance      = self.maximumDistance,
                                                               histSize         = self.histogramSize,
                                                               bin              = self.bin,
                                                               allAtoms         = True)
        intraF,interF = full_pair_distribution_histograms( boxCoords        = self.engine.boxCoordinates[indexes],
                                                           basis            = self.engine.basisVectors,
                                                           moleculeIndex    = self.engine.moleculesIndexes[indexes],
                                                           elementIndex     = self.engine.elementsIndexes[indexes],
                                                           numberOfElements = self.engine.numberOfElements,
                                                           minDistance      = self.minimumDistance,
                                                           maxDistance      = self.maximumDistance,
                                                           histSize         = self.histogramSize,
                                                           bin              = self.bin )
        self.set_active_atoms_data_after_move( {"intra":intraM-intraF, "inter":interM-interF} )
        # reset coordinates
        self.engine.boxCoordinates[indexes] = boxData
        # compute squaredDeviations after move
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        totalPCF = self.__get_total_gr({"intra":dataIntra, "inter":dataInter})
        # set after move squared deviations
        self.set_after_move_squared_deviations( self.compute_squared_deviations(modelData = totalPCF) )

    def plot(self, ax=None, intra=True, inter=True, 
                   legend=True, legendCols=2, legendLoc='best',
                   title=True, titleChiSquare=True, titleScaleFactor=True):
        """ 
        Plot pair correlation constraint.
        
        :Parameters:
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If ax is given, the figure won't be rendered and drawn.
               If None is given a new plot figure will be created and the figue will be rendered and drawn.
            #. intra (boolean): Whether to add intra-molecular pair distribution function features to the plot.
            #. inter (boolean): Whether to add inter-molecular pair distribution function features to the plot.
            #. legend (boolean): Whether to create the legend or not
            #. legendCols (integer): Legend number of columns.
            #. legendLoc (string): The legend location. Anything among
               'right', 'center left', 'upper right', 'lower right', 'best', 'center', 
               'lower left', 'center right', 'upper left', 'upper center', 'lower center'
               is accepted.
            #. title (boolean): Whether to create the title or not
            #. titleChiSquare (boolean): Whether to show contraint's chi square value in title.
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
        AXES.plot(self.experimentalDistances,self.experimentalPDF, 'ro', label="experimental", markersize=7.5, markevery=1 )
        AXES.plot(self.shellsCenter, output["pcf"], 'k', linewidth=3.0,  markevery=25, label="total" )
        # plot without window function
        if self.windowFunction is not None:
            AXES.plot(self.shellsCenter, output["pcf_total"], 'k', linewidth=1.0,  markevery=5, label="total - no window" )
        # plot partials
        intraStyleIndex = 0
        interStyleIndex = 0
        for key, val in output.items():
            if key in ("pcf_total", "pcf"):
                continue
            elif "intra" in key and intra:
                AXES.plot(self.shellsCenter, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key )
                intraStyleIndex+=1
            elif "inter" in key and inter:
                AXES.plot(self.shellsCenter, val, INTER_STYLES[interStyleIndex], markevery=5, label=key )
                interStyleIndex+=1
        # plot legend
        if legend:
            AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
        # set title
        if title:
            t = ''
            if titleChiSquare and self.squaredDeviations is not None:
                t += "$\chi^2=%.6f$ "%(self.squaredDeviations)
            if titleScaleFactor:
                t += " - "*(len(t)>0) + "$scale$ $factor=%.6f$"%(self.scaleFactor)
            if len(t):
                AXES.set_title(t)
        # set axis labels
        AXES.set_xlabel("$r(\AA)$", size=16)
        AXES.set_ylabel("$g(r)$"  , size=16)
        # set background color
        plt.gcf().patch.set_facecolor('white')
        #show
        #show
        if ax is None:
            plt.show()
        return AXES
        
        
        
        