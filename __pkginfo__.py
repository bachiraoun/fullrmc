"""
Must know
=========
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
fullrmc requires:
   #. Python (>= 2.6 and < 3),
   #. NumPy (lowest version tested is 1.7.1)
   #. cython (lowest version tested is 0.21.1)
   #. matplotlib (lowest version tested is 1.4)
   #. pdbParser (lowest version tested is 0.1.2 - 0.1.3 is used starting from fullrmc 0.3.0)
   #. pysimplelog (lowest version tested is 0.1.7)

**Installation using pip:**\n
numpy and cython must be installed and updated manually. 
   
   .. code-block:: bash  
    
       pip install -U "numpy>=1.7.1"
       pip install -U "cython>=0.21.1"
       
When you already have a working installation of numpy and cython.
   
   .. code-block:: bash  
    
       pip install fullrmc

**Installation by cloning github repository:**\n
   * Ensure all fullrmc required packages are installed and up to data by executing the
     following python script:
    
    .. code-block:: python
        
        # check whether all packages are already installed
        from pkg_resources import parse_version as PV
        for name, ver in [('numpy'      ,'1.7.1') ,
                          ('cython'     ,'0.21.1'),
                          ('pdbParser'  ,'0.1.3') ,
                          ('pysimplelog','0.1.7') ,
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
                    print '%s is installed properly and minimum version requirement is met.'%(name)
        
           
   * Locate python's site-packages by executing the following python script:
     
     .. code-block:: python
     
        import os
        os.path.join(os.path.dirname(os.__file__), 'site_packages')

   * Navigate to site_packages folder and clone git repository from command line:
   
    .. code-block:: bash
       
       cd .../site_packages
       git clone https://github.com/bachiraoun/fullrmc.git   

   * Change directory to .../site_packages/fullrmc/Extensions. Then compile fullrmc extensions
     from command line as the following:
     
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
   #. :class:`.ReducedStructureFactorConstraint` added computing Reduced Static Structure Factor

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~  
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` plot Y axis
      label units added.
   #. swap example changed to atomicSimulation and S(q) added to the fitting along with G(r)
   #. thfSimulation example change to molecularSimulation
   
"""

__version__ = '0.4.0'

__author__ = "Bachir Aoun"

__email__ = "fullrmc@gmail.com"


