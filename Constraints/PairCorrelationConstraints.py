"""
PairCorrelationConstraints contains classes for all constraints related to
experimental pair correlation functions.

.. inheritance-diagram:: fullrmc.Constraints.PairCorrelationConstraints
    :parts: 1
"""
# standard libraries imports
from __future__ import print_function
import itertools, os, copy, re

# external libraries imports
import numpy as np
from pdbparser.Utilities.Database import is_element_property, get_element_property
from pdbparser.Utilities.Collection import get_normalized_weighting

# fullrmc imports
from ..Globals import FLOAT_TYPE, LOGGER, PI
from ..Globals import str, long, unicode, bytes, basestring, range, xrange, maxint
from ..Core.Collection import reset_if_collected_out_of_date, get_caller_frames
from ..Core.Constraint import Constraint, ExperimentalConstraint
from ..Core.pairs_histograms import multiple_pairs_histograms_coords, full_pairs_histograms_coords
from ..Constraints.Collection import ShapeFunction
from ..Constraints.PairDistributionConstraints import PairDistributionConstraint

class PairCorrelationConstraint(PairDistributionConstraint):
    """
    Controls the total pair correlation function (pcf) of the system noted
    as g(r). pcf indicates the probability of finding atomic pairs separated
    by the real space distance r. Theoretically g(r) oscillates about 1.
    Also :math:`g(r) \\rightarrow 1` when :math:`r \\rightarrow \\infty`
    and it takes the exact value of zero for :math:`r` shorter than the
    distance of the closest possible approach of pairs of atoms.\n
    Pair correlation function g(r) and pair distribution function G(r) are
    directly related as in the following:
    :math:`g(r)=1+(\\frac{G(r)}{4 \\pi \\rho_{0} r})`.

    g(r) is calculated after binning all pair atomic distances into a weighted
    histograms of values :math:`n(r)` from which local number densities
    are computed as in the following:

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


    +----------------------------------------------------------------------+
    |.. figure:: pair_correlation_constraint_plot_method.png               |
    |   :width: 530px                                                      |
    |   :height: 400px                                                     |
    |   :align: left                                                       |
    +----------------------------------------------------------------------+



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
                LOGGER.warn("@%s Better set shape function rmax with infinite boundary conditions. Here value is automatically set to %s"%(self.engine.usedFrame, rmax))
        shapeFunc  = ShapeFunction(engine    = self.engine,
                                   weighting = self.weighting,
                                   qmin=qmin, qmax=qmax, dq=dq,
                                   rmin=rmin, rmax=rmax, dr=dr)
        self._shapeArray = shapeFunc.get_gr_shape_function( self.shellCenters )
        del shapeFunc

    def _reset_standard_error(self):
        # recompute squared deviation
        if self.data is not None:
            totalPCF = self.__get_total_gr(self.data, rho0=self.engine.numberDensity)
            self.set_standard_error(self.compute_standard_error(modelData = totalPCF))

    def update_standard_error(self):
        """ Compute and set constraint's standardError."""
        # set standardError
        totalPDF = self.get_constraint_value()["total"]
        self.set_standard_error(self.compute_standard_error(modelData = totalPDF))


    def __get_total_gr(self, data, rho0):
        """This method is created just to speed up the computation of
        the total gr upon fitting.
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
        self._fittedScaleFactor = self.get_adjusted_scale_factor(experimentalData=self.experimentalPDF, modelData=gr, dataWeights=self._usedDataWeights, rho0=rho0)
        if self._fittedScaleFactor != 1:
            Gr = (4.*PI*self.shellCenters*rho0)*(gr-1)
            Gr *= self._fittedScaleFactor
            gr  = 1. +  Gr/(4.*PI*self.shellCenters*rho0)
        # apply multiframe prior and weight
        gr = self._apply_multiframe_prior(gr)
        # convolve total with window function
        if self.windowFunction is not None:
            gr = np.convolve(gr, self.windowFunction, 'same')
        return gr

    def get_adjusted_scale_factor(self, experimentalData, modelData, dataWeights, rho0):
        """Overload to bring back g(r) to G(r) prior to fitting scale factor.
        g(r) -> 1 at high r and this will create a wrong scale factor.
        Overloading can be avoided but it's better to for performance reasons
        """
        #r    = self.shellCenters
        SF = self.scaleFactor
        # check to update scaleFactor
        if self.adjustScaleFactorFrequency:
            if not self.engine.accepted%self.adjustScaleFactorFrequency:
                expGr = (4.*PI*self.shellCenters*rho0)*(experimentalData-1)
                Gr    = (4.*PI*self.shellCenters*rho0)*(modelData-1)
                SF = self.fit_scale_factor(expGr, Gr, dataWeights)
        return SF

    def _on_collector_reset(self):
        pass

    def _get_constraint_value(self, data, applyMultiframePrior=True):
        # http://erice2011.docking.org/upload/Other/Billinge_PDF/03-ReadingMaterial/BillingePDF2011.pdf     page 6
        output = {}
        for pair in self.elementsPairs:
            output["rdf_intra_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
            output["rdf_inter_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
            output["rdf_total_%s-%s" % pair] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
        output["total_no_window"] = np.zeros(self.histogramSize, dtype=FLOAT_TYPE)
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
            output["total_no_window"]        += output["rdf_total_%s-%s" % pair]
        # remove shape function
        if self._shapeArray is not None:
            output["total_no_window"] -= self._shapeArray
        # multiply by scale factor
        if self.scaleFactor != 1:
            Gr = (4.*PI*self.shellCenters*self.engine.numberDensity)*(output["total_no_window"]-1)
            Gr *= self.scaleFactor
            output["total_no_window"]  = 1. +  Gr/(4.*PI*self.shellCenters*self.engine.numberDensity)
        # apply multiframe prior and weight
        if applyMultiframePrior:
            output["total_no_window"] = self._apply_multiframe_prior(output["total_no_window"])
        # convolve total with window function
        if self.windowFunction is not None:
            output["total"] = np.convolve(output["total_no_window"], self.windowFunction, 'same')
        else:
            output["total"] = output["total_no_window"]
        return output

    def get_constraint_value(self, applyMultiframePrior=True):
        """
        Get constraint's data dictionary value.

        :Parameters:
            #. applyMultiframePrior (boolean): Whether to apply subframe weight
               and prior to the total. This will only have an effect when used
               frame is a subframe and in case subframe weight and prior is
               defined.

        :Returns:
            #. PDFs (dictionary): The PDFs dictionnary, where keys are the
               element wise intra and inter molecular PDFs and values are the
               computed PDFs.
        """
        if self.data is None:
            LOGGER.warn("data must be computed first using 'compute_data' method.")
            return {}
        return self._get_constraint_value(self.data, applyMultiframePrior=applyMultiframePrior)

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
                                                    minDistance      = self.minimumDistance,
                                                    maxDistance      = self.maximumDistance,
                                                    histSize         = self.histogramSize,
                                                    bin              = self.bin,
                                                    ncores           = self.engine._runtime_ncores)
        # create data and compute standard error
        data     = {"intra":intra, "inter":inter}
        totalPCF = self.__get_total_gr(data, rho0=self.engine.numberDensity)
        stdError = self.compute_standard_error(modelData = totalPCF)
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
        Compute constraint's data before move is executed.

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
                                                          minDistance      = self.minimumDistance,
                                                          maxDistance      = self.maximumDistance,
                                                          histSize         = self.histogramSize,
                                                          bin              = self.bin,
                                                          allAtoms         = True,
                                                          ncores           = self.engine._runtime_ncores)
        intraF,interF = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates[relativeIndexes],
                                                      basis            = self.engine.basisVectors,
                                                      isPBC            = self.engine.isPBC,
                                                      moleculeIndex    = self.engine.moleculesIndex[relativeIndexes],
                                                      elementIndex     = self.engine.elementsIndex[relativeIndexes],
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
        Compute constraint's data after move is executed.

        :Parameters:
            #. realIndexes (numpy.ndarray): Not used here.
            #. relativeIndexes (numpy.ndarray): Group atoms relative index
               the move will be applied to.
            #. movedBoxCoordinates (numpy.ndarray): The moved atoms new
               coordinates.
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
                                                          minDistance      = self.minimumDistance,
                                                          maxDistance      = self.maximumDistance,
                                                          histSize         = self.histogramSize,
                                                          bin              = self.bin,
                                                          allAtoms         = True,
                                                          ncores           = self.engine._runtime_ncores )
        intraF,interF = full_pairs_histograms_coords( boxCoords        = self.engine.boxCoordinates[relativeIndexes],
                                                      basis            = self.engine.basisVectors,
                                                      isPBC            = self.engine.isPBC,
                                                      moleculeIndex    = self.engine.moleculesIndex[relativeIndexes],
                                                      elementIndex     = self.engine.elementsIndex[relativeIndexes],
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
        totalPCF = self.__get_total_gr({"intra":dataIntra, "inter":dataInter}, rho0=self.engine.numberDensity)
        # set after move standard error
        self.set_after_move_standard_error( self.compute_standard_error(modelData = totalPCF) )
        # increment tried
        self.increment_tried()

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
        weightingScheme = self.weightingScheme
        relativeIndex   = relativeIndex[0]
        selectedElement = self.engine.allElements[relativeIndex]
        self.engine.numberOfAtomsPerElement[selectedElement] -= 1
        WS = get_normalized_weighting(numbers=self.engine.numberOfAtomsPerElement, weights=self._elementsWeight )
        for k in WS:
            WS[k] = FLOAT_TYPE(WS[k])
        self._set_weighting_scheme(WS)
        # compute standard error
        if not self.engine._RT_moveGenerator.allowFittingScaleFactor:
            SF = self.adjustScaleFactorFrequency
            self._set_adjust_scale_factor_frequency(0)
        rho0     = ((self.engine.numberOfAtoms-1)/self.engine.volume).astype(FLOAT_TYPE)
        totalPCF = self.__get_total_gr(data, rho0=rho0)
        standardError = self.compute_standard_error(modelData = totalPCF)
        if not self.engine._RT_moveGenerator.allowFittingScaleFactor:
            self._set_adjust_scale_factor_frequency(SF)
        # reset activeAtoms data
        self.set_active_atoms_data_before_move(None)
        # set data
        self.set_amputation_data( {'data':data, 'weightingScheme':self.weightingScheme} )
        # compute standard error
        self.set_amputation_standard_error( standardError )
        # reset weightingScheme and number of atoms per element
        self._set_weighting_scheme(weightingScheme)
        self.engine.numberOfAtomsPerElement[selectedElement] += 1

    def _on_collector_collect_atom(self, realIndex):
        pass

    def _on_collector_release_atom(self, realIndex):
        pass

    def plot(self, xlabelParams={'xlabel':'$r(\\AA)$', 'size':10},
                   ylabelParams={'ylabel':'$g(r)(\\AA^{-2})$', 'size':10},
                   **kwargs):
        """
        Alias to ExperimentalConstraint.plot with additional parameters

        :Additional/Adjusted Parameters:
            #. xlabelParams (None, dict): modified matplotlib.axes.Axes.set_xlabel
               parameters.
            #. ylabelParams (None, dict): modified matplotlib.axes.Axes.set_ylabel
               parameters.
        """
        return super(PairCorrelationConstraint, self).plot(xlabelParams= xlabelParams,
                                                           ylabelParams= ylabelParams,
                                                           **kwargs)





#
