import random, math
import numpy as np
import matplotlib.pyplot as plt
from pdbparser import pdbparser
from pdbparser.Utilities import Geometry, Information, Modify

## read pdb file
PDB = pdbparser("SiOx.pdb")

## translate to origin
COM = Geometry.get_center(PDB.indexes, PDB)
Geometry.translate(PDB.indexes, PDB, -COM)

## get radius
minX,maxX, minY,maxY, minZ,maxZ = Geometry.get_min_max(PDB.indexes, PDB)
radiusX = (maxX-minX)/2.
radiusY = (maxY-minY)/2.
radiusZ = (maxZ-minZ)/2.
radius  = max(radiusX,radiusY,radiusZ)

# get oxygen/silicon ratio vs depth
oIdxs   = Information.get_records_indexes_by_attribute_value(PDB.indexes, PDB, 'element_symbol', 'O')
siIdxs  = Information.get_records_indexes_by_attribute_value(PDB.indexes, PDB, 'element_symbol', 'Si')
oDist   = np.sqrt(np.sum(PDB.coordinates[oIdxs]**2,1))
siDist  = np.sqrt(np.sum(PDB.coordinates[siIdxs]**2,1))
depth   = math.ceil(max(oDist) - min(oDist))
# get histogram of oxygen/silicon distribution ratio
ho,  e = np.histogram(oDist,  bins=range(math.ceil(radius) - depth, math.ceil(radius)))
hsi, e = np.histogram(siDist, bins=range(math.ceil(radius) - depth, math.ceil(radius)))
ratios = (ho.astype(float)/(ho.astype(float)+hsi.astype(float)))[::-1]
# get ratios slope
depthX  = np.array(range(len(ratios)))
slope,i = np.linalg.lstsq(np.vstack([depthX, np.ones(len(depthX))]).T,ratios)[0]
# compute oxygen ratio
oxygenRatio = slope*depthX + i
## plot and visualize
plt.bar(depthX,ratios, label='distribution')
plt.plot(depthX, oxygenRatio, color='r', label='regression')
plt.legend()
plt.xticks(depthX)
plt.xlabel('depth (A)')
plt.ylabel('Oxygen ratio')
plt.show(block=False)


## get 10 nano-particle of incremental 1 Ang. smaller radius
structures = {0:PDB}
for r in range(1,10):
    # get indexes
    idxs = Geometry.get_satisfactory_records_indexes(PDB.indexes, PDB, 'np.sqrt(x**2 + y**2 + z**2) <=%s'%(radius-r))
    # get pdb copy
    pdb  = PDB.get_copy(indexes=idxs)
    # randomly alter to oxygen 5 angstrom deep
    structures[r] = pdb

## adjust ratio of oxygen of every and each nano-particle
structures[0].export_pdb('multiframe_structure_%i.pdb'%0)
for idx in range(1, len(structures)):
    pdb = structures[idx]
    rad = math.ceil( max(np.sqrt(np.sum(pdb.coordinates**2,1))) )
    for ratio, depthMin in zip(oxygenRatio, depthX):
        if ratio<=0:
            continue
        depthMax = depthMin+1
        indexes = Geometry.get_satisfactory_records_indexes(pdb.indexes, pdb, '(np.sqrt(x**2 + y**2 + z**2)>%s) & (np.sqrt(x**2 + y**2 + z**2)<=%s)'%(rad-depthMax,rad-depthMin))
        oxIdxs  = Information.get_records_indexes_by_attribute_value(indexes, pdb, 'element_symbol', 'O')
        siIdxs  = Information.get_records_indexes_by_attribute_value(indexes, pdb, 'element_symbol', 'Si')
        # find out number of Si or O to transform
        neededOx = int(math.ceil(ratio*len(indexes)))
        neededSi = len(indexes)-neededOx
        if neededOx>len(oxIdxs):
            diff     = neededOx-len(oxIdxs)
            proba    = float(diff)/float(len(siIdxs))
            modIdxs  = [i for i in siIdxs if random.random()<=proba]
            Modify.set_records_attribute_value(modIdxs, pdb, 'element_symbol', 'O')
            Modify.set_records_attribute_value(modIdxs, pdb, 'atom_name', 'O')
        elif neededSi>len(siIdxs):
            diff     = neededSi-len(siIdxs)
            proba    = float(diff)/float(len(oxIdxs))
            modIdxs  = [i for i in oxIdxs if random.random()<=proba]
            Modify.set_records_attribute_value(modIdxs, pdb, 'element_symbol', 'Si')
            Modify.set_records_attribute_value(modIdxs, pdb, 'atom_name', 'Si')
    pdb.export_pdb('multiframe_structure_%i.pdb'%idx)
