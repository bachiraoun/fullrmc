"""
Must know
=========
#. fullrmc uses pdbParser library to parse input pdb files and to visualize
   system's atomic structure using `VMD <http://www.ks.uiuc.edu/Research/vmd/>`_.  
   VMD which stands for Visual Molecular Dynamics is a free software for 
   displaying and animating large molecular systems.  
#. pdbParser in general automatically detects VMD executable when installed.
   fullrmc will work fine without VMD, but make sure it's installed if you
   want to use the convenient visualize method of fullrmc's Engine.
   
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
        
Version 0.2.0:
--------------
This is the first officially published version. 
Code can be downloaded and installed from 
`pypi fullrmc 0.2.0 <https://pypi.python.org/pypi/fullrmc/0.2.0>`_.

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
   #. Boundary conditions handling for all non cubic, tetragonal nor orthorhombic systems
      is prone to distances calculation errors caused by pdbParser (<=0.1.2). 
 
Version 0.3.0:
--------------
Starting from this version, pdbParser (>=0.1.3) is used. 
Code can be downloaded and installed from 
`pypi fullrmc 0.3.0 <https://pypi.python.org/pypi/fullrmc/0.3.0>`_.

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~
   #. Periodic boundary conditions handling fixed and seperated from pdbParser.
   #. :class:`.InterMolecularDistanceConstraint` squared deviation computation changed 
      from square law to absolute value. Therefore small values are not under-represented
      in the total squared deviation calculation.
   #. :class:`.InterMolecularDistanceConstraint` 'flexible' flag added.

Version 0.3.1:
--------------
Code can be downloaded and installed from 
`pypi fullrmc 0.3.1 <https://pypi.python.org/pypi/fullrmc/0.3.1>`_.

Fixes and improvements:
~~~~~~~~~~~~~~~~~~~~~~~
   #. Erroneous wx imports from thfSimulation Example is removed.
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` 
      check_experimental_data merged and sorted distances bug fixed. 
   #. :class:`.PairDistributionConstraint` and :class:`.PairCorrelationConstraint` 
      set_limits bug fixed.
   #. :class:`.Engine` add_group method allows list, set, tuples as well as integers.
"""

__version__ = '0.3.1'

__author__ = "Bachir Aoun"

__email__ = "fullrmc@gmail.com"


