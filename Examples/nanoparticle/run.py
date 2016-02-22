#from pdbParser.pdbParser import pdbParser
#from pdbParser.Utilities.Geometry import get_satisfactory_records_indexes
#from pdbParser.Utilities.Modify import reset_records_serial_number, reset_sequence_number_per_residue
#pdb=pdbParser('system.pdb')
#indexes = get_satisfactory_records_indexes(indexes=pdb.indexes, pdb=pdb, expression='np.sqrt(x**2 + y**2 + z**2) <= 15')
#pdbCopy = pdb.get_copy(indexes=indexes)
#reset_records_serial_number(pdb=pdbCopy)
#pdbCopy.export_pdb('nanoparticle.pdb' )
#exit()

# standard libraries imports
import os

# external libraries imports
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.PairCorrelationConstraints import PairCorrelationConstraint

# shut down logging
LOGGER.set_log_file_basename("fullrmc")
LOGGER.set_minimum_level(30)

pdbPath = "nanoparticle.pdb" 
ENGINE = Engine(pdb=pdbPath, constraints=None)
expData = np.arange(0.01, 25, 0.01)
expData = np.transpose( [expData, np.zeros((len(expData)))]).astype(np.float32)
PDF = PairCorrelationConstraint(engine=None, experimentalData=expData, weighting="atomicNumber")
ENGINE.add_constraints([PDF]) 

PDF.compute_data()
output = PDF.get_constraint_value()
#print output.keys()

print len(PDF.shellsCenter)
plt.plot(PDF.shellsCenter, output["pcf"], 'k', linewidth=2.0)
#plt.plot(PDF.shellsCenter, np.gradient( output["pcf"]))
plt.plot(PDF.shellsCenter[1:], np.abs(output["pcf"][1:]-output["pcf"][0:-1]))
plt.show()










