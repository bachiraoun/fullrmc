import os
import copy
from pdbParser.pdbParser import pdbParser
from pdbParser.Utilities.Construct import AmorphousSystem
from pdbParser.Utilities.Database import __ATOM__

# Carbon atoms
C = copy.deepcopy(__ATOM__)
C['atom_name'] = "C"
C['residue_name'] = "C"
C['element_symbol'] = "C"
pdbC = pdbParser()
pdbC.records = [C]
pdbC.set_name("C")
# Argon atoms
Ar = copy.deepcopy(__ATOM__)
Ar['atom_name'] = "Ar"
Ar['residue_name'] = "Ar"
Ar['element_symbol'] = "Ar"
pdbAr = pdbParser()
pdbAr.records = [Ar]
pdbAr.set_name("Ar")
# Oxygen atoms
O = copy.deepcopy(__ATOM__)
O['atom_name'] = "O"
O['residue_name'] = "O"
O['element_symbol'] = "O"
pdbO = pdbParser()
pdbO.records = [O]
pdbO.set_name("O")
# Hydrogen atoms
H = copy.deepcopy(__ATOM__)
H['atom_name'] = "H"
H['residue_name'] = "H"
H['element_symbol'] = "H"
pdbH = pdbParser()
pdbH.records = [H]
pdbH.set_name("H")

# create amorphous system
pdb = AmorphousSystem([pdbH,pdbO,pdbC,pdbAr], 
                      insertionNumber = [100,100,100,100],
                      priorities = {'density': False, 'insertionNumber': True, 'boxSize': True},
                      boxSize = [22,22,22]).construct().get_pdb()
pdb.export_pdb('system.pdb')


