"""
Must know
=========
- fullrmc supported version is always the newest one. Older version are
  never supported as updating is very simple.
- fullrmc uses pdbParser library to parse input pdb files and to visualize
  system's atomic structure using `VMD <http://www.ks.uiuc.edu/Research/vmd/>`_.  
  VMD which stands for Visual Molecular Dynamics is a free software for 
  displaying and animating large molecular systems.
- pdbParser in general automatically detects VMD executable when installed.
  fullrmc will work fine without VMD, but make sure it's installed if you
  want to use the convenient visualize method of fullrmc's Engine.
- Unless :class:`.Engine` set_molecules_indexes method is used explicitly, 
  molecules are classified by parsing the structure 
  `'.pdb' <http://deposit.rcsb.org/adit/docs/pdb_atom_format.html>`_ file 
  **Residue name**, **Residue sequence number** and **Segment identifier** 
  attributes. Therefore a molecule is the collection of all atoms sharing 
  the same three later attributes value.


Installation
============
Here we describe two methods to installing fullrmc.
   #. The first method is by using `pip <https://pip.pypa.io/en/stable/>`_. 
      If you don't know what pip is and you are lazy enough not to read through 
      the documentation, pip supports installing from PyPI, version control, 
      local projects, and directly from distribution files. To install pip 
      download `get_pip.py <https://bootstrap.pypa.io/get-pip.py>`_ file 
      and run the following from the command line.
      
      .. code-block:: bash  
       
          python get_pip.py
          
   #. The second method is by fullrmc's cloning the source code and  
      repository from `github <https://github.com/bachiraoun/fullrmc>`_ 
      and then compiling the extension files in place. 


For your general understanding, fullrmc requires the following:
   #. Python (>= 2.7 and < 3),
   #. NumPy (lowest version tested is 1.7.1)
   #. cython (lowest version tested is 0.21.1)
   #. matplotlib (lowest version tested is 1.4)
   #. pyrep (lowest version tested is 1.0.2 - 1.0.3 is used starting 
      from fullrmc 3.0.0 - 1.0.4 is used starting rom fullrmc 3.1.0)
   #. pdbParser (lowest version tested is 0.1.2 - 0.1.3 is used starting 
      from fullrmc 0.3.0 - 0.1.4 is used starting from fullrmc 1.0.0
      - 0.1.5 is used starting from fullrmc 1.0.1)
   #. pysimplelog (lowest version tested is 0.2.1)

**Installation using pip:**\n
Installing fullrmc with pip is done in two steps.
Firstly install numpy and cython manually and ensure that 
they meet with fullrmc's version requirement. 
   
   .. code-block:: bash  
    
       pip install -U "numpy>=1.7.1"
       pip install -U "cython>=0.21.1"
       
When you already have a working up-do-date installation of numpy and cython,
you can proceed to installing fullrmc. The rest of the dependencies will be 
automatically installed and updated while fullrmc is being installed.
   
   .. code-block:: bash  
    
       pip install fullrmc

**Installation by cloning github repository:**\n
   * Ensure all fullrmc required packages are installed and up-do-date by executing the
     following python script:
    
    .. code-block:: python
        
        # check whether all packages are already installed
        from pkg_resources import parse_version as PV
        for name, ver in [('numpy'      ,'1.7.1') ,
                          ('cython'     ,'0.21.1'),
                          ('pyrep'      ,'1.0.4') ,
                          ('pdbParser'  ,'0.1.5') ,
                          ('pysimplelog','0.3.0') ,
                          ('matplotlib' ,'1.4'  )]:
            try:
                lib = __import__(name)
            except:
                print '%s must be installed for fullrmc to run properly'%(name)
            else:
                if PV(lib.__version__) < PV(ver):
                    print '%s installed version %s is below minimum suggested version %s.\
Updating %s is highly recommended.'%(name, lib.__version__, ver, name)
                else:
                    print '%s is properly installed and minimum version requirement is \
met.'%(name)
        
           
   * Locate python's site-packages by executing the following python script:
     
     .. code-block:: python
     
        import os
        os.path.join(os.path.dirname(os.__file__), 'site_packages')

   * Navigate to site_packages folder and clone git repository from command line:
   
    .. code-block:: bash
       
       cd .../site_packages
       git clone https://github.com/bachiraoun/fullrmc.git   

   * Change directory to .../site_packages/fullrmc/Extensions. Then compile fullrmc 
     extensions from command line as the following:
     
    .. code-block:: bash
       
       cd .../site_packages/fullrmc/Extensions
       python setup.py build_ext --inplace   
      
      
Versions
========
Latest stable version source code: https://github.com/bachiraoun/fullrmc \n
Latest pypi installable version: https://pypi.python.org/pypi/fullrmc/ \n


Version 0.1.0:
--------------
This version is not published.

Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~
   #. Selectors:
      :class:`.RandomSelector`
      :class:`.WeightedRandomSelector`
      :class:`.DefinedOrderSelector`
      :class:`.DirectionalOrderSelector`
   #. Constraints:
      :class:`.InterMolecularDistanceConstraint`
      :class:`.IntraMolecularDistanceConstraint`
      :class:`.BondConstraint`
      :class:`.BondsAngleConstraint`
      :class:`.ImproperAngleConstraint`
      :class:`.PairDistributionConstraint`
      :class:`.PairCorrelationConstraint`  
   #. Generators:
      :class:`.TranslationGenerator`
      :class:`.TranslationAlongAxisGenerator`
      :class:`.TranslationAlongSymmetryAxisGenerator`
      :class:`.TranslationTowardsCenterGenerator`
      :class:`.TranslationTowardsAxisGenerator`
      :class:`.TranslationTowardsSymmetryAxisGenerator`
      :class:`.TranslationAlongSymmetryAxisPath`
      :class:`.RotationGenerator`
      :class:`.RotationAboutAxisGenerator`
      :class:`.RotationAboutSymmetryAxisGenerator`
      :class:`.RotationAboutSymmetryAxisPath`
      :class:`.DistanceAgitationGenerator`
      :class:`.AngleAgitationGenerator`
    
    
    
    
`Version 0.2.0 <https://pypi.python.org/pypi/fullrmc/0.2.0>`_:
--------------------------------------------------------------
This is the first officially published version. 

New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. Selectors:
      :class:`.SmartRandomSelector`
   #. Constraints:
      :class:`.AtomicCoordinationNumberConstraint`
   #. Generators:
      :class:`.SwapPositionsGenerator`
      
Known bugs and issues:
~~~~~~~~~~~~~~~~~~~~~~
   #. Boundary conditions handling for all non cubic, tetragonal nor 
      orthorhombic systems is prone to distances calculation errors 
      caused by pdbParser (<=0.1.2). 
 
 
 
 
`Version 0.3.0 <https://pypi.python.org/pypi/fullrmc/0.3.0>`_:
--------------------------------------------------------------
Starting from this version, pdbParser (>=0.1.3) is used. 

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~
   #. Periodic boundary conditions handling fixed and separated from pdbParser.
   #. :class:`.InterMolecularDistanceConstraint` squared deviation computation 
      changed from square law to absolute value. Therefore small values are not 
      under-represented in the total squared deviation calculation.
   #. :class:`.InterMolecularDistanceConstraint` 'flexible' flag added.

   
   
   
`Version 0.3.1 <https://pypi.python.org/pypi/fullrmc/0.3.1>`_:
--------------------------------------------------------------

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~
   #. Erroneous wx imports from thfSimulation Example is removed.
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` 
      check_experimental_data merged and sorted distances bug fixed. 
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` 
      set_limits bug fixed.
   #. :class:`.Engine` add_group method allows list, set, tuples as well as integers.
   
   
   
   
`Version 0.3.2 <https://pypi.python.org/pypi/fullrmc/0.3.2>`_:
--------------------------------------------------------------

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` 
      binning precision fixed. Histogram compiled C code adjusted accordingly.
   #. :class:`.AtomicCoordinationNumberConstraint` set_coordination_number_definition 
      method error message made more clear. 
   #. :class:`.AtomicCoordinationNumberConstraint` compiled C++ code boundary conditions
      distances computation fixed for multiple box separation.
  
  
  
  
`Version 0.3.3 <https://pypi.python.org/pypi/fullrmc/0.3.3>`_:
--------------------------------------------------------------

New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. :class:`.SwapCentersGenerator`
  
  
  
 
`Version 0.4.0 <https://pypi.python.org/pypi/fullrmc/0.4.0>`_:
--------------------------------------------------------------

New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. :class:`.StructureFactorConstraint` added computing Static Structure Factor
   #. :class:`.ReducedStructureFactorConstraint` added computing Reduced Static 
      Structure Factor
   
Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` 
      plot Y axis label units added.
   #. swap example changed to atomicSimulation and S(q) added to the fitting 
      along with G(r)
   #. thfSimulation example change to molecularSimulation
  
  
  
 
`Version 1.0.0 <https://pypi.python.org/pypi/fullrmc/1.0.0>`_:
--------------------------------------------------------------
This is a main version change from 0.x.y to 1.x.y and that's because non-periodic 
boundary conditions are now implemented. Starting from this version shape function 
correction can be used. For non-periodic boundary conditions, very big box with periodic 
boundaries is automatically set and system shape function is estimated to correct for 
the fixed :math:`\\rho_{0}` global density approximation.

Some testing were done prior to publishing this version and it was found out that 
computing angles for :class:`.BondsAngleConstraint` can in some cases be wrong.
We recommend updating to this version if :class:`.BondsAngleConstraint` is needed
for fitting molecular systems.

Starting from this version, pdbParser (>=0.1.4) is used.

New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. fullrmc.Constraints.Collection package added along with :class:`.ShapeFunction`  
   #. Engine numberDensity added. Used to set correct volume and density when 
      InfiniteBoundaries are used  
   #. fullrmc.Core.Collection.smooth function added. It can be used to smooth
      out any 1D data. This is useful for noise experimental data
   
Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
   #. BondAngles calculation bug fixed
   #. Constraints all squared deviations attributes and methods changed to 
      standard error to conserve the generality
   #. Engine chiSquare attributes and methods changed to total standard error 
      to conserve the generality
   #. Logger levels added
   #. rebin core method check argument added
   #. :class:`.StructureFactorConstraint` and :class:`.ReducedStructureFactorConstraint` 
      experimentalQValues set as property
    
   
   
 
`Version 1.0.1 <https://pypi.python.org/pypi/fullrmc/1.0.1>`_:
--------------------------------------------------------------
Starting from this version, pdbParser (>=0.1.5) is used. There is no bugs
fixed in this version comparably to version 1.0.0. 
This version corrects for cell visualization compatibility between fullrmc 
and `VMD <http://www.ks.uiuc.edu/Research/vmd/>`_. 
VMD `cell plugin <http://www.ks.uiuc.edu/Research/vmd/plugins/pbctools/>`_
uses a, b, c, unitcell parameters (the lengths of vectors A, B and C the 
3D-vectors of the unitcell sides with the convention that A is parallel to 
the x-axis) and alpha, beta, gamma (the angles of the unit cell) for 
non-orthorhombic unitcells. This version uses cell plugin no more but
draw the boundary conditions vectors directly. Also visualization method 
'boxToCenter' flag is added to correct for box visualization shifts by 
translating boundary conditions vectors' centre to atom coordinates one.
   
   
   
 
`Version 1.1.0 <https://pypi.python.org/pypi/fullrmc/1.1.0>`_:
--------------------------------------------------------------
Several levels of abstractions are made allowing multiple core computations
using openmp. Nevertheless more testing is needed and therefore the current
version is published as single core still.
:class:`.AtomicCoordinationNumberConstraint` uses python objects which 
makes it very slow, it must be revisited. 
          
New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. :class:`.Constraint` computation cost property added. This is used at engine 
      runtime to minimize computations and enhance performance by computing less
      costly constraints first. 

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
    #. :class:`.BondsAngleConstraint` computing angle right vector normalization bug fixed.
    #. :class:`.ImproperAngleConstraint` computing improper vector bug fixed.
   
   
   
   
`Version 1.2.0 <https://pypi.python.org/pypi/fullrmc/1.2.0>`_:
--------------------------------------------------------------
Starting from this version, running fullrmc on multicore and multiprocessor computers 
is possible. Also, pysimplelog (>=0.2.1) is used. Isolated system in non-periodic 
boundary conditions are treated correctly as in an infinte box using inifinite boundary 
conditions instead of a very big box with periodic boundaries.


New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. :class:`.AtomicCoordinationNumberConstraint` completely reimplemented and optimized.
      Coordination number definition is re-formatted allowing any type of core's and 
      shell's atoms. This new implementation computes mean coordination number per shell 
      definition rather than per atom. Fractional coordination numbers are allowed and 
      this implementation benefits from a tremendous performance increase of 1000 times 
      compared to the old one.
   #. Engine.run method 'restartPdb' and 'ncore' arguments added. 
   
   
 
 
`Version 1.2.1 <https://pypi.python.org/pypi/fullrmc/1.2.1>`_:
--------------------------------------------------------------
No bugs fixed but some examples corrected. 




`Version 2.0.0 <https://pypi.python.org/pypi/fullrmc/2.0.0>`_:
--------------------------------------------------------------
This is a main version change from 1.x.y to 2.x.y. Starting from this version engine 
is no more a single file but a `pyrep <http://bachiraoun.github.io/pyrep/>`_ 
repository. Using a repository engine instead of a single file has many advantages. 
The first and more important advantage is the limitation of accessing and saving single 
big engine files when the simulated atomic system is big. The second advantage 
is the introduction of the concept of engine frames. Frames can be used to build a 
fitting history or even better to creating a statistical structural solution rather than 
a single structure one which is prone primarily to overfitting and other errors.
Also, repository engine's insure a sort of backwards compatibility and files can be 
retrieved when updating fullrmc's versions. Implementation wise, switching from single 
file to repository is the ground layer to taking fullrmc to the next level of 
super-computing.
Engine repository is not a binary black box file but a directory or a folder that one 
can browse. It's advisable not to manually delete or add files or folders to the 
repository. Interacting with fullrmc.Engine instance is what one should do to creating 
or updating engine's repository.
The transition to fullrmc 2.x.y. should be done seemlessly at the end of the user 
except for small modifications that were made and that are listed below.

Modifications:
~~~~~~~~~~~~~~  
    #. :class:`.Engine`: Several modification happened to fullrmc.Engine class definition.
        * Instanciating engine arguments have changed. We found that it's a bad practice 
          to set the pdb structure, the constraints, the groups and all other attributes
          at the engine instanciation. Therefore, now engine's instanciation takes more 
          appropriate arguments such as the engine's repository path, the frames, the
          log file path and whether to delete a found engine at the given path and start 
          fresh using freshStart flag.
        * Engine.is_engine method is added to check whether a given path is a 
          fullrmc.Engine repository.
        * Engine frames related methods are also added to manipulate and switch between 
          frames.
        * Engine.run method takes no more savePath argument, as the path has to be the 
          repository. In case one wants to save in a new repository, he should save the 
          engine first using Engine.save method and then run the engine using 
          Engine.run method.
    #. :class:`.Constraint`: We also found that constraints must not be allowed to change 
       engine and therefore Constraint.set_engine method is now deprecated. Also 
       Constraints are now instanciated without engine argument. Constraint's engine is 
       set automatically when a constraint is added to an engine.

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
    #. abs function changed in all compiled files as it was creating rounding issues on
       some operating systems.
    #. arcos function cliped for floating errors in angles compiled file.
    #. :class:`.InterMolecularDistanceConstraint`: Computing constraint data in function
       single_atomic_distances_dists of compiled atomic_distances fixed.
    #. :class:`.BondConstraint`: bondsMap fixed from causing definition conflicts if set
       multiple times.
    #. :class:`.BondsAngleConstraint`: anglesMap fixed from causing definition conflicts 
       if set multiple times. 
    #. :class:`.ImproperAngleConstraint`: anglesMap fixed from causing definition  
       conflicts if set multiple times. 




`Version 3.0.0 <https://pypi.python.org/pypi/fullrmc/3.0.0>`_:
--------------------------------------------------------------
This is a main version change from 2.x.y to 3.x.y where pyrep (>=1.0.3) is used.
Thusfar until versions 2.x.y, system's number of atoms is fixed throughout the whole 
simulation. Starting from this version, fullrmc allows dynamically removing atoms 
from the system upon fitting. This is a revolutionary functionality enabling the 
reproduction of defects in systems. 

New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. :class:`.DihedralAngleConstraint` implemented to control dihedral angles.
   #. :class:`.AtomsCollector` implemented allowing the ability of dynamically remove
      and add re-insert atoms to the system.
   #. :class:`.AtomsRemoveGenerator` implemented to generate atoms removing from system
      upon fitting.
   #. :class:`.EmptyGroup` added to host Removes generators.
      
Modifications:
~~~~~~~~~~~~~~  
    #. all cython modules are completely de-pythonized.
    #. all angle constraints respect the same implementation and interface design.




`Version 3.0.1 <https://pypi.python.org/pypi/fullrmc/3.0.1>`_:
--------------------------------------------------------------

New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. Plot method added to all of :class:`.BondConstraint`,
      :class:`.BondsAngleConstraint`, :class:`.DihedralAngleConstraint` and
      :class:`.ImproperAngleConstraint`.     
      
Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
    #. :class:`.DihedralAngleConstraint` FRAME_DATA fixed which was preventing  
       constraint from saving to pyrep repository.




`Version 3.0.2 <https://pypi.python.org/pypi/fullrmc/3.0.2>`_:
--------------------------------------------------------------
  
New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. Redundant objects global container mechanism added. This will sometimes reduce 
      some engine's attributes size, allowing fast saving and loading of data.
         
Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
    #. :class:`.MoveGeneratorCombinator` bug from latest update fixed.
    #. :class:`.SwapGenerator` is not allowed to be collected or combined with other 
       generators. Also swapList will update automatically upon removing and
       releasing atoms.
    #. :class:`.RemoveGenerator` is not allowed to be collected or combined with other 
       generators.
    #. MLSelection, refine and explore examples BiasedEngine fixed




`Version 3.1.0 <https://pypi.python.org/pypi/fullrmc/3.1.0>`_:
--------------------------------------------------------------
Starting from this version, it is possible custom set atoms weight for all of
:class:`.PairDistributionConstraint`, :class:`.PairCorrelationConstraint`, 
:class:`.StructureFactorConstraint` and :class:`.ReducedStructureFactorConstraint`.
Also starting from this version, pysimplelog (>=0.3.0) and pyrep (>=1.0.4) are used. 
  
New Modules and definitions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #. All constraints export data method added.
   #. All constraints plot method added.
   #. :class:`.IntraMolecularDistanceConstraint` added again after being removed in
      early versions.
         
Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
    #. :class:`.Engine` repository ACID flag is changed to False as it's not needed.
    #. :class:`.StructureFactorConstraint` and :class:`.ReducedStructureFactorConstraint`
       atoms removal bug fixed.
    #. Computing standard error for :class:`.IntraMolecularDistanceConstraint` and
       :class:`.InterMolecularDistanceConstraint` is optimized for big systems and 
       constraint's definition. Calculation time is thousands of times faster for
       extended definitions.

"""

__version__    = '3.1.0'
               
__author__     = "Bachir Aoun"
               
__email__      = "fullrmc@gmail.com"
               
__forum__      = "https://groups.google.com/forum/#!forum/fullrmc"
               
__onlinedoc__  = "http://bachiraoun.github.io/fullrmc/"

__repository__ = "https://github.com/bachiraoun/fullrmc"

__pypi__       = "https://pypi.python.org/pypi/fullrmc"





