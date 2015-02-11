"""
Reverse Monte Carlo (RMC) is probably best known for its applications in condensed matter physics and solid state chemistry.
fullrmc (https://github.com/bachiraoun/fullrmc) is a RMC modelling package to solve an inverse problem whereby an atomic/molecular 
model is adjusted until its atoms positions have the greatest consistency with a set of experimental data.\n
fullrmc is a python package with its core and calculation modules optimized and compiled in Cython. 
fullrmc is not a standard RMC package but it is rather unique in its approach to solving an atomic or molecular structure. 
fullrmc package sub-module Engine is the main module and contains the definition of 'Engine' which 
is the main and only class used to launch and RMC calculation. Engine reads only Protein Data Bank formatted 
atomic configuration files '.pdb' (http://deposit.rcsb.org/adit/docs/pdb_atom_format.html) and handles  
other definitions and attributes such as:

    #. Group: Engine doesn't understand atoms or molecules but group of atom indexes instead. 
       A group is a set of atom indexes, allowing no indexes redundancy 
       within the same group definition. A Group instance can contain any set of indexes and as many atom indexes as needed. 
       Grouping atoms is essential to make clusters of atoms (residues, molecules, etc) evolve and move together. A group of 
       a single atom index can be used to make a single atom move separately from the others. An engine 'groups' attribute 
       is a simple list of group instances containing all the desired and defined groups that one wants to move.
    #. Group selector: Engine requires a GroupSelector instance which is the artist that selects a group from the engine's
       groups list at every engine runtime step. Among other properties, depending on which group selector is used by the
       engine, a GroupSelector can allow weighting which means selecting groups more or less frequently than the others, 
       it can also allow selection recurrence and refinement of a single group, ordered and random selection is also possible.
    #. Move generator: Every group instance has its own MoveGenerator. Therefore every group of atoms when selected by 
       the engine's group selector at the engine's runtime can perform a customizable and different kind of moves. 
    #. Constraint: A constraint is a rule that controls certain aspect of the configuration upon moving atoms. 
       An engine 'constraints' attribute is a list of all defined and used constraint instances, it is 
       the judge that controls the evolution of the system by accepting or rejecting the move of a group. 
       If engine's constraints list is empty and contains no constraint definition, this will result
       in accepting all the generated moves.
"""
__version__ = 1.0

    
    
    