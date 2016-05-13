import os
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.Collection import get_path
from pdbParser.Utilities.Construct import AmorphousSystem
from pdbParser.Utilities.Geometry import get_satisfactory_records_indexes, translate, get_geometric_center
from pdbParser.Utilities.Modify import delete_records_and_models_records, reset_records_serial_number, reset_sequence_number_per_residue
from pdbParser.Utilities.Database import __WATER__

# read thf molecule and translate to the center
thfNAGMA = pdbParser(os.path.join(get_path("pdbparser"),"Data/NAGMA.pdb" ) ) 
center = get_geometric_center(thfNAGMA.indexes, thfNAGMA)
translate(thfNAGMA.indexes, thfNAGMA, -center)

# create pdbWATER
pdbWATER = pdbParser()
pdbWATER.records = __WATER__
pdbWATER.set_name("water")

# create amorphous
pdbWATER = AmorphousSystem(pdbWATER, boxSize=[40,40,40], density = 0.75).construct().get_pdb()
center = get_geometric_center(pdbWATER.indexes, pdbWATER)
translate(pdbWATER.indexes, pdbWATER, -center)

# make hollow
hollowIndexes = get_satisfactory_records_indexes(pdbWATER.indexes, pdbWATER, "np.sqrt(x**2 + y**2 + z**2) <= 10")
delete_records_and_models_records(hollowIndexes, pdbWATER)

# concatenate
thfNAGMA.concatenate(pdbWATER, pdbWATER.boundaryConditions)

# reset numbering
reset_sequence_number_per_residue(thfNAGMA.indexes, thfNAGMA)
reset_records_serial_number(thfNAGMA)

# export and visualize
thfNAGMA.export_pdb("nagma_in_water.pdb")
thfNAGMA.visualize()
