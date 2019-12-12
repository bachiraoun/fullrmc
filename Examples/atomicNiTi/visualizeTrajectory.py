# standard libraries imports
from __future__ import print_function
import os

# external libraries imports

# fullrmc library imports
from pdbparser.pdbparser import pdbparser, pdbTrajectory


# visualize
traj = pdbTrajectory()
traj.set_structure('system.pdb')
time=0

for num in sorted( [int(item.split('.pdb')[0]) for item in os.listdir("pdbFiles")] ):
    pdb = os.path.join("pdbFiles", str(num)+".pdb")
    print(pdb)
    traj.append_configuration(pdb=pdbparser(pdb), vectors=traj.structure.boundaryConditions.get_vectors(), time=time)
    time=time+1

# visualize
traj.visualize()
