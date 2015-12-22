# standard libraries imports
import time
import os
import itertools

# external libraries imports
import numpy as np
from pdbParser.pdbParser import pdbParser, pdbTrajectory


# pdbFiles
pdbFiles  = [fn for fn in os.listdir("pdbFiles") if ".pdb" in fn]
generated = [int(fn.split("_")[0]) for fn in pdbFiles]
pdbFiles = [os.path.join("pdbFiles",pdbFiles[idx]) for idx in  np.argsort(generated)]

# create trajectory
traj = pdbTrajectory()
traj.set_structure(pdbParser("thf.pdb"))
for idx in range(len(pdbFiles)):
    print "loading frame %i out of %i"%(idx, len(pdbFiles))
    fname = pdbFiles[idx]
    pdb = pdbParser(fname)
    traj.append_configuration(pdb=pdb, vectors=pdb.boundaryConditions.get_vectors(), time=idx)

traj.visualize()


