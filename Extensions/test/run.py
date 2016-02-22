# standard libraries imports
import os
import time
import multiprocessing as mp 

# external libraries imports
import numpy as np
import matplotlib.pyplot as plt

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Core.atomic_distances import real_distances, parallel_real_distances

nRUNS=10

# engine variables
expDataPath = "thf_pdf.exp"
pdbPath = "thf.pdb"
engineSavePath = "thf_engine.rmc"
    

# initialize engine
ENGINE = Engine(pdb=pdbPath, constraints=None)
# create constraints

coords = np.concatenate([ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates,
                         ENGINE.boxCoordinates])
print coords.shape
outputSerial = np.zeros((coords.shape[0],), dtype=np.float32)
outputMulti1 = np.zeros((coords.shape[0],), dtype=np.float32)
outputMulti4 = np.zeros((coords.shape[0],), dtype=np.float32)

tic = time.time()
for _ in range(nRUNS):
    outputSerial=real_distances(atomIndex=0,
               boxCoords=coords,
               basis=ENGINE.basisVectors,
               output=outputSerial)
print "single - single core: %s"%(time.time()-tic)


tic = time.time()
for _ in range(nRUNS):
    outputMulti1=parallel_real_distances(atomIndex=0,
                   boxCoords=coords,
                   basis=ENGINE.basisVectors,
                   output=outputMulti1,
                   ncores=1)
print "mutli - single core: %s"%(time.time()-tic)

def run(startIndex, endIndex, boxCoords, basis, output):
    partial_real_distances(atomIndex=np.int32(0),
                           startIndex=startIndex,
                           endIndex=endIndex,
                           boxCoords=boxCoords,
                           basis=basis,
                           output=output)    

NCORES = 1
indexes = np.linspace(0, coords.shape[0], NCORES+1, dtype=np.int32)

tic = time.time()
for _ in range(nRUNS):
    outputMulti4=parallel_real_distances(atomIndex=0,
                   boxCoords=coords,
                   basis=ENGINE.basisVectors,
                   output=outputMulti4,
                   ncores=4)
print "mutli - 4 core: %s"%(time.time()-tic)

#results = [p.get() for p in processes]
print "mutli - %s core: %s"%(NCORES, time.time()-tic)    
                            
#tic = time.time()
#for _ in range(nRUNS):
#    outputMulti1=parallel_real_distances(atomIndex=0,
#                   boxCoords=coords,
#                   basis=ENGINE.basisVectors,
#                   output=outputMulti1,
#                   ncores=1)
#print "mutli - single core: %s"%(time.time()-tic)
#
#tic = time.time()
#for _ in range(nRUNS):
#    outputMulti4=parallel_real_distances(atomIndex=0,
#                   boxCoords=coords,
#                   basis=ENGINE.basisVectors,
#                   output=outputMulti4,
#                   ncores=4)
#print "mutli - 4 core: %s"%(time.time()-tic)

