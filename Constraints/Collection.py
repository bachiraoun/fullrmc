"""
Collection of methods and classes definition useful for constraints computation 
"""

# standard libraries imports

# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER, FLOAT_TYPE, PI
from fullrmc.Core.Collection import is_number

        
             
class ShapeFunction(object):
    """
    Shape function used to correct for particle shape. The shape 
    function is subtracted from the total G(r) of g(r). It must 
    be used when non-periodic boundary conditions are used to take 
    into account the atomic density drop and to correct for the 
    :math:`\\rho_{0}` approximation.
    
    :Parameters:
        #. engine (Engine): The fitting engine.
        #. weighting (string): The elements weighting.
        #. qmin (number): The minimum reciprocal distance q in 
           :math:`\\AA^{-1}` considered to compute the shape function.
        #. qmax (number): The maximum reciprocal distance q in 
           :math:`\\AA^{-1}` considered to compute the shape function.
        #. dq (number): The reciprocal distance bin size in 
           :math:`\\AA^{-1}` considered to compute the shape function.
        #. rmin (number): The minimum distance in :math:`\\AA` considered 
           upon building the histogram prior to computing the shape function.
        #. rmax (number): The maximum distance in :math:`\\AA` 
           considered upon building the histogram prior to computing the shape function.
        #. dr (number): The bin size in :math:`\\AA` considered upon 
           building the histogram prior to computing the shape function. 
        
    **N.B: tweak qmax as small as possible to reduce the wriggles ...**
    """
    def __init__(self, engine, weighting="atomicNumber",
                       qmin=0.001, qmax=1, dq=0.005,
                       rmin=0.00, rmax=100, dr=1):
        # get qmin
        assert is_number(qmin), LOGGER.error("qmin must be a number")
        qmin = FLOAT_TYPE(qmin)
        assert qmin>0, LOGGER.error("qmin '%s' must be bigger than 0"%qmin)
        # get qmin
        assert is_number(qmax), LOGGER.error("qmax must be a number")
        qmax = FLOAT_TYPE(qmax)
        assert qmax>qmin, LOGGER.error("qmax '%s' must be bigger than qmin '%s'"%(qmin,qmax))
        # get dq
        assert is_number(dq), LOGGER.error("dq must be a number")
        dq = FLOAT_TYPE(dq)
        assert dq>0, LOGGER.error("dq '%s' must be bigger than 0"%dq)
        # import StructureFactorConstraint
        from fullrmc.Constraints.StructureFactorConstraints import StructureFactorConstraint
        # create StructureFactorConstraint
        Q = np.arange(qmin, qmax, dq)
        D = np.transpose([Q, np.zeros(len(Q))]).astype(FLOAT_TYPE)
        self._SFC = StructureFactorConstraint(engine=engine, rmin=rmin, rmax=rmax, dr=dr, experimentalData=D, weighting="atomicNumber")
        # set parameters
        self._rmin = FLOAT_TYPE(rmin)
        self._rmax = FLOAT_TYPE(rmax)
        self._dr   = FLOAT_TYPE(dr)
        self._qmin = FLOAT_TYPE(qmin)
        self._qmax = FLOAT_TYPE(qmax)
        self._dq   = FLOAT_TYPE(dq)
        self._weighting = weighting
        
    def __get_Gr_from_Sq(self, qValues, rValues, Sq):
        Gr    = np.zeros(len(rValues), dtype=FLOAT_TYPE)
        sq_1  = Sq-1
        qsq_1 = qValues*sq_1
        dq = qValues[1]-qValues[0]
        for ridx, r in enumerate(rValues):
            sinqr_dq = dq * np.sin(qValues*r)
            Gr[ridx] = (2./PI) * np.sum( qsq_1 * sinqr_dq )
        return Gr  
    
    def get_Gr_shape_function(self, rValues, compute=True):
        """
        Get shape function of G(r) used in a PairDistributionConstraint.
        
        :Parameters:
            #. rValues (numpy.ndarray): The r values array.
            #. compute (boolean): whether to recompute shape 
               function reciprocal data.
        
        :Returns:
            #. shapeFunction (numpy.ndarray): The compute shape function.
        """
        # compute data
        if compute: self._SFC.compute_data()
        # get shape function
        return self.__get_Gr_from_Sq( qValues=self._SFC.experimentalQValues,
                                      rValues=rValues,
                                      Sq=self._SFC.get_constraint_value()['sf'])
    
    def get_gr_shape_function(self, rValues, compute=True):
        """
        Get shape function of g(r) used in a PairCorrelationConstraint.
        
        :Parameters:
            #. rValues (numpy.ndarray): The r values array.
            #. compute (boolean): whether to recompute shape 
               function reciprocal data.
        
        :Returns:
            #. shapeFunction (numpy.ndarray): The compute shape function.
        """
        sFunc = self.get_Gr_shape_function(rValues=rValues, compute=compute)
        rho0  = self.engine.numberDensity #(self._SFC.engine.numberOfAtoms/self._SFC.engine.volume).astype(FLOAT_TYPE)
        return sFunc / (FLOAT_TYPE(4.)*PI*rho0*rValues)
        
        
        
        
        