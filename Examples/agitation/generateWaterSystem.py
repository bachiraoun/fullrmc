"""
In this test two water amorphous box are constructed
one with a hollow sphere inside, the other only a shere of water is kept
"""
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.Construct import AmorphousSystem
from pdbParser.Utilities.Database import __WATER__

# create pdbWATER
pdbWATER = pdbParser()
pdbWATER.records = __WATER__
pdbWATER.set_name("water")

pdbWATER = AmorphousSystem(pdbWATER, density = 1, boxSize=[7,7,7]).construct().get_pdb()
pdbWATER.export_pdb("water.pdb")
pdbWATER.visualize()
