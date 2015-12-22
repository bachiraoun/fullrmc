# standard libraries imports
import os

# external libraries imports

# fullrmc library imports
from pdbParser.pdbParser import pdbParser, pdbTrajectory


# visualize    
traj = pdbTrajectory()
traj.set_structure('system.pdb')
time=0

for num in sorted( [int(item.split('.pdb')[0]) for item in os.listdir("pdbFiles")] ):
    pdb = os.path.join("pdbFiles", str(num)+".pdb")
    print pdb
    traj.append_configuration(pdb=pdbParser(pdb), vectors=traj.structure.boundaryConditions.get_vectors(), time=time)
    time=time+1

# visualize
traj.visualize()
    
    
    
