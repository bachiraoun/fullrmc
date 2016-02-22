import os
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.Construct import AmorphousSystem
from pdbParser.Utilities.Collection import get_path


# create thf amorphous box
pdb = pdbParser(os.path.join(get_path("pdbparser"),"Data/Tetrahydrofuran.pdb" ))
#pdb.visualize()
#exit()
pdb = AmorphousSystem(pdb, boxSize=[48,48,48], 
                           recursionLimit = 1000000,
                           insertionNumber=730,
                           density = 0.8,
                           priorities={"boxSize":True, "insertionNumber":True, "density":False}).construct().get_pdb()
pdb.export_pdb("thf_system.pdb")
#pdb = pdbParser("thf_at_density%s.pdb"%str(density).replace(".","p") )
#pdb.visualize()
#exit()
