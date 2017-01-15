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
from fullrmc.Core.Collection import is_number, is_integer, get_path, reset_if_collected_out_of_date
from fullrmc.Core.Constraint import Constraint, ExperimentalConstraint
from fullrmc.Core.pairs_histograms import multiple_pairs_histograms_coords, full_pairs_histograms_coords
from fullrmc.Constraints.Collection import ShapeFunction
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
    
    :Parameters: Refer to :class:`.PairDistributionConstraint` 
    
    .. code-block:: python
    
        # import fullrmc modules
        from fullrmc.Engine import Engine
        from fullrmc.Constraints.PairCorrelationConstraints import PairCorrelationConstraint
        
        # create engine 
        ENGINE = Engine(path='my_engine.rmc')
        
        # set pdb file
        ENGINE.set_pdb('system.pdb')
        
        # create and add constraint
        PCC = PairCorrelationConstraint(experimentalData="pcf.dat", weighting="atomicNumber")
        ENGINE.add_constraints(PCC)
        
    """
    def _update_shape_array(self):
        rmin = self._shapeFuncParams['rmin']
        rmax = self._shapeFuncParams['rmax']
        dr   = self._shapeFuncParams['dr'  ]
        qmin = self._shapeFuncParams['qmin']
        qmax = self._shapeFuncParams['qmax']
        dq   = self._shapeFuncParams['dq'  ]
        if rmax is None:
            if self.engine.isPBC:
                a = self.engine.boundaryConditions.get_a()
                b = self.engine.boundaryConditions.get_b()
                c = self.engine.boundaryConditions.get_c()
                rmax = FLOAT_TYPE( np.max([a,b,c]) + 10 )
            else:
                coordsCenter = np.sum(self.engine.realCoordinates, axis=0)/self.engine.realCoordinates.shape[0]
                coordinates  = self.engine.realCoordinates-coordsCenter
                distances    = np.sqrt( np.sum(coordinates**2, axis=1) )
                maxDistance  = 2.*np.max(distances) 
                rmax = FLOAT_TYPE( maxDistance + 10 )
                LOGGER.warn("Better set shape function rmax with infinite boundary conditions. Here value is automatically set to %s"%rmax)
        shapeFunc  = ShapeFunction(engine    = self.engine, 
                                   weighting = self.weighting,
                                   qmin=qmin, qmax=qmax, dq=dq,
                                   rmin=rmin, rmax=rmax, dr=dr)
        self._shapeArray = shapeFunc.get_gr_shape_function( self.shellCenters )
        del shapeFunc
        
    def _reset_standard_error(self):
        # recompute squared deviation
        if self.data is not None:
            totalPCF = self.__get_total_gr(self.data)
            self.set_standard_error(self.compute_standard_error(modelData = totalPCF))   
        
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
        gr /= self.shellVolumes
        # remove shape function
        if self._shapeArray is not None:
            gr -= self._shapeArray
        # Multiply by scale factor
        self._fittedScaleFactor = self.get_adjusted_scale_factor(self.experimentalPDF, gr, self._usedDataWeights)
        if self._fittedScaleFactor != 1:
            gr *= self._fittedScaleFactor
        # convolve total with window function
        if self.windowFunction is not None:
            gr = np.convolve(gr, self.windowFunction, 'same')
        return gr

    def _on_collector_reset(self):
        pass  
        
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
            intensityFactor = (self.engine.volume*wij)/(Nij*self.shellVolumes)
            # multiply by intensityFactor
            output["rdf_intra_%s-%s" % pair] *= intensityFactor
            output["rdf_inter_%s-%s" % pair] *= intensityFactor
            output["rdf_total_%s-%s" % pair]  = output["rdf_intra_%s-%s" % pair] + output["rdf_inter_%s-%s" % pair]
            output["pcf_total"]              += output["rdf_total_%s-%s" % pair]
        # remove shape function
        if self._shapeArray is not None:
            output["pcf_total"] -= self._shapeArray 
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
    
    @reset_if_collected_out_of_date
    def compute_data(self):
        """ Compute data and update engine constraintsData dictionary. """
        intra,inter = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates,
                                                    basis            = self.engine.basisVectors,
                                                    isPBC            = self.engine.isPBC, 
                                                    moleculeIndex    = self.engine.moleculesIndexes,
                                                    elementIndex     = self.engine.elementsIndexes,
                                                    numberOfElements = self.engine.numberOfElements,
                                                    minDistance      = self.minimumDistance,
                                                    maxDistance      = self.maximumDistance,
                                                    histSize         = self.histogramSize,
                                                    bin              = self.bin,
                                                    ncores           = self.engine._runtime_ncores)                  
        # update data
        self.set_data({"intra":intra, "inter":inter})
        self.set_active_atoms_data_before_move(None)
        self.set_active_atoms_data_after_move(None)
        # set standardError
        totalPCF = self.__get_total_gr(self.data)
        self.set_standard_error(self.compute_standard_error(modelData = totalPCF))
        # set original data
        if self.originalData is None:
            self._set_original_data(self.data)
    
    def compute_before_move(self, realIndexes, relativeIndexes):
        """ 
        Compute constraint before move is executed
        
        :Parameters:
            #. realIndexes (numpy.ndarray): Group atoms indexes the move will be applied to
        """
        intraM,interM = multiple_pairs_histograms_coords( indexes          = relativeIndexes,
                                                          boxCoords        = self.engine.boxCoordinates,
                                                          basis            = self.engine.basisVectors,
                                                          isPBC            = self.engine.isPBC,
                                                          moleculeIndex    = self.engine.moleculesIndexes,
                                                          elementIndex     = self.engine.elementsIndexes,
                                                          numberOfElements = self.engine.numberOfElements,
                                                          minDistance      = self.minimumDistance,
                                                          maxDistance      = self.maximumDistance,
                                                          histSize         = self.histogramSize,
                                                          bin              = self.bin,
                                                          allAtoms         = True,
                                                          ncores           = self.engine._runtime_ncores) 
        intraF,interF = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates[relativeIndexes],
                                                      basis            = self.engine.basisVectors,
                                                      isPBC            = self.engine.isPBC,
                                                      moleculeIndex    = self.engine.moleculesIndexes[relativeIndexes],
                                                      elementIndex     = self.engine.elementsIndexes[relativeIndexes],
                                                      numberOfElements = self.engine.numberOfElements,
                                                      minDistance      = self.minimumDistance,
                                                      maxDistance      = self.maximumDistance,
                                                      histSize         = self.histogramSize,
                                                      bin              = self.bin,
                                                      ncores           = self.engine._runtime_ncores )                                             
        # set active atoms data
        self.set_active_atoms_data_before_move( {"intra":intraM-intraF, "inter":interM-interF} )
        self.set_active_atoms_data_after_move(None)
    
    def compute_after_move(self, realIndexes, relativeIndexes, movedBoxCoordinates):
        """ 
        Compute constraint after move is executed
        
        :Parameters:
            #. indexes (numpy.ndarray): Group atoms indexes the move will be applied to.
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
                                                          moleculeIndex    = self.engine.moleculesIndexes,
                                                          elementIndex     = self.engine.elementsIndexes,
                                                          numberOfElements = self.engine.numberOfElements,
                                                          minDistance      = self.minimumDistance,
                                                          maxDistance      = self.maximumDistance,
                                                          histSize         = self.histogramSize,
                                                          bin              = self.bin,
                                                          allAtoms         = True,
                                                          ncores           = self.engine._runtime_ncores )
        intraF,interF = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates[relativeIndexes],
                                                      basis            = self.engine.basisVectors,
                                                      isPBC            = self.engine.isPBC,
                                                      moleculeIndex    = self.engine.moleculesIndexes[relativeIndexes],
                                                      elementIndex     = self.engine.elementsIndexes[relativeIndexes],
                                                      numberOfElements = self.engine.numberOfElements,
                                                      minDistance      = self.minimumDistance,
                                                      maxDistance      = self.maximumDistance,
                                                      histSize         = self.histogramSize,
                                                      bin              = self.bin,
                                                      ncores           = self.engine._runtime_ncores  )                                             
        # set active atoms data
        self.set_active_atoms_data_after_move( {"intra":intraM-intraF, "inter":interM-interF} )
        # reset coordinates
        self.engine.boxCoordinates[relativeIndexes] = boxData
        # compute standardError after move
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]+self.activeAtomsDataAfterMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]+self.activeAtomsDataAfterMove["inter"]
        totalPCF = self.__get_total_gr({"intra":dataIntra, "inter":dataInter})
        # set after move standard error
        self.set_after_move_standard_error( self.compute_standard_error(modelData = totalPCF) )

    def compute_as_if_amputated(self, realIndex, relativeIndex):
        """ 
        Compute and return constraint's data and standard error as if atom given its 
        its was amputated.
        
        :Parameters:
            #. index (numpy.ndarray): atom index as a numpy array of a single element.
        """
        # compute data 
        self.compute_before_move(realIndexes=realIndex, relativeIndexes=relativeIndex)
        dataIntra = self.data["intra"]-self.activeAtomsDataBeforeMove["intra"]
        dataInter = self.data["inter"]-self.activeAtomsDataBeforeMove["inter"]
        data      = {"intra":dataIntra, "inter":dataInter}
        # temporarily adjust self.__weightingScheme
        weightingScheme = self.weightingScheme
        #relativeIndex   = self.engine._atomsCollector.get_relative_index(relativeIndex[0])
        relativeIndex = relativeIndex[0]
        selectedElement = self.engine.allElements[relativeIndex]
        self.engine.numberOfAtomsPerElement[selectedElement] -= 1
        #elementsWeights        = dict([(el,float(get_element_property(el,self.__weighting))) for el in self.engine.elements])
        WS = get_normalized_weighting(numbers=self.engine.numberOfAtomsPerElement, weights=self._elementsWeights )
        for k, v in WS.items():
            WS[k] = FLOAT_TYPE(v)
        self._set_weighting_scheme(WS)
        # compute standard error  
        if not self.engine._RT_moveGenerator.allowFittingScaleFactor:          
            SF = self.adjustScaleFactorFrequency
            self._set_adjust_scale_factor_frequency(0)
        totalPCF      = self.__get_total_gr(data)
        standardError = self.compute_standard_error(modelData = totalPCF)
        if not self.engine._RT_moveGenerator.allowFittingScaleFactor: 
            self._set_adjust_scale_factor_frequency(SF)
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        # set data
        self.set_amputation_data( {'data':data, 'weightingScheme':self.weightingScheme} )
        self.set_amputation_standard_error( standardError )
        # reset weightingScheme
        self._set_weighting_scheme(weightingScheme)
        self.engine.numberOfAtomsPerElement[selectedElement] += 1
        
    def _on_collector_collect_atom(self, realIndex):
        pass
    
    def _on_collector_release_atom(self, realIndex):
        pass
        
    def plot(self, ax=None, intra=True, inter=True, shapeFunc=True, 
                   legend=True, legendCols=2, legendLoc='best',
                   title=True, usedFrame=True, titleStdErr=True, titleScaleFactor=True):
        """ 
        Plot pair correlation constraint.
        
        :Parameters:
            #. ax (None, matplotlib Axes): matplotlib Axes instance to plot in.
               If ax is given, the figure won't be rendered and drawn.
               If None is given a new plot figure will be created and the figue will be rendered and drawn.
            #. intra (boolean): Whether to add intra-molecular pair distribution function features to the plot.
            #. inter (boolean): Whether to add inter-molecular pair distribution function features to the plot.
            #. shapeFunc (boolean): Whether to add shape function to the plot only when exists.
            #. legend (boolean): Whether to create the legend or not
            #. legendCols (integer): Legend number of columns.
            #. legendLoc (string): The legend location. Anything among
               'right', 'center left', 'upper right', 'lower right', 'best', 'center', 
               'lower left', 'center right', 'upper left', 'upper center', 'lower center'
               is accepted.
            #. title (boolean): Whether to create the title or not
            #. usedFrame(boolean): Whether to show used frame name.
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
        AXES.plot(self.experimentalDistances,self.experimentalPDF, 'ro', label="experimental", markersize=7.5, markevery=1 )
        AXES.plot(self.shellCenters, output["pcf"], 'k', linewidth=3.0,  markevery=25, label="total" )
        # plot without window function
        if self.windowFunction is not None:
            AXES.plot(self.shellCenters, output["pcf_total"], 'k', linewidth=1.0,  markevery=5, label="total - no window" )
        if shapeFunc and self._shapeArray is not None:
            AXES.plot(self.shellCenters, self._shapeArray, '--k', linewidth=1.0,  markevery=5, label="shape function" )
        # plot partials
        intraStyleIndex = 0
        interStyleIndex = 0
        for key, val in output.items():
            if key in ("pcf_total", "pcf"):
                continue
            elif "intra" in key and intra:
                AXES.plot(self.shellCenters, val, INTRA_STYLES[intraStyleIndex], markevery=5, label=key )
                intraStyleIndex+=1
            elif "inter" in key and inter:
                AXES.plot(self.shellCenters, val, INTER_STYLES[interStyleIndex], markevery=5, label=key )
                interStyleIndex+=1
        # plot legend
        if legend:
            AXES.legend(frameon=False, ncol=legendCols, loc=legendLoc)
        # set title
        if title:
            if usedFrame:
                t = '$frame: %s$ : '%self.engine.usedFrame.replace('_','\_')
            else:
                t = ''
            if titleStdErr and self.standardError is not None:
                t += "$std$ $error=%.6f$ "%(self.standardError)
            if titleScaleFactor:
                t += " - "*(len(t)>0) + "$scale$ $factor=%.6f$"%(self.scaleFactor)
            if len(t):
                AXES.set_title(t)
        # set axis labels
        AXES.set_xlabel("$r(\AA)$", size=16)
        AXES.set_ylabel("$g(r)(\AA^{-2})$"  , size=16)
        # set background color
        plt.gcf().patch.set_facecolor('white')
        #show
        #show
        if ax is None:
            plt.show()
        return AXES
        
        
        
        