"""
Latest stable version source code: https://github.com/bachiraoun/fullrmc \n
Latest pypi installable version: https://pypi.python.org/pypi/fullrmc/ \n

Version 0.1.0:
==============
This version was not published.

Modules and definitions:
------------------------
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
==============
This is the first officially published version. 
Source code can be found, downloaded and installed from 
`here <https://pypi.python.org/pypi/fullrmc/0.2.0>`_.

New Modules and definitions:
----------------------------
   #. Selectors:
      :class:`.SmartRandomSelector`
   #. Constraints:
      :class:`.AtomicCoordinationNumberConstraint`
      :class:`.AtomicCoordinationAngleConstraint` 
   #. Generators:
      :class:`.SwapPositionsGenerator`
        
Known bugs and issues:
----------------------
   #. Boundary conditions handling for all non cubic, tetragonal nor orthorhombic systems
      is prone to distances calculation errors caused by pdbParser (<=0.1.2). Newer
      fullrmc versions will include its own boundary conditions implementation.
 
Version 0.3.0:
==============
This version uses pdbParser (>0.1.2). 
Source code can be found, downloaded and installed from 
`here <https://pypi.python.org/pypi/fullrmc/0.3.0>`_.

Fixes and improvements:
-----------------------
   #. Periodic boundary conditions handling fixed and seperated from pdbParser.
   #. :class:`.InterMolecularDistanceConstraint` squared deviation computation changed 
      from square law to absolute value. Therefore small values are not under-represented
      in the total squared deviation calculation.
   #. :class:`.InterMolecularDistanceConstraint` 'flexible' flag added.
"""
__version__ = '0.3.0'

__author__ = "Bachir Aoun"

__email__ = "fullrmc@gmail.com"


