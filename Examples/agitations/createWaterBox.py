from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.Construct import AmorphousSystem
from pdbParser.Utilities.Database import __WATER__
from pdbParser.Utilities.Geometry import translate, get_center

# create pdbWATER of a single molecule of water
pdbWATER = pdbParser()
pdbWATER.records = __WATER__
pdbWATER.set_name("water")

# create amorphous water box
pdbWATER = AmorphousSystem(pdbWATER, density = 1, boxSize=[10,10,10]).construct().get_pdb()

# translate water box to center
translate(pdbWATER.indexes, pdbWATER, -get_center(pdbWATER.indexes, pdbWATER))

# export water box
pdbWATER.export_pdb("waterBox.pdb")

