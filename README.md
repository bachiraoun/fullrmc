**This git repository is only updated with new releases of fullrmc. A private repository is used for development.**

fullrmc
=======
It's a Reverse Monte Carlo (RMC) python/Cython/C package, especially designed to solve an inverse 
problem whereby an atomic/molecular model is adjusted until its atoms positions have the greatest 
consistency with a set of experimental data. RMC is probably best known for its applications in 
condensed matter physics and solid state chemistry. fullrmc is a fully object-oriented package 
where everything can be overloaded allowing easy development, implementation and maintenance of the code. 
It's core sub-package and modules are fully optimized written in cython/C. fullrmc is unique in its approach, 
among other functionalities:

1. Atomic and molecular systems are supported.
2. All types (not limited to cubic) of periodic boundary conditions systems are supported.
3. Atoms can be grouped into groups so the system can evolve atomically, clusterly, molecularly or any combination of those.
4. Every group can be assigned a different move generator (translation, rotation, a combination of moves generators, etc).
5. Selection of groups to perform moves can be done manually OR automatically, randomly OR NOT !!
6. Supports Artificial Intelligence and Reinforcement Machine Learning algorithms. 

Next on the list
================
* Generators machine learning algorithms.
* Elements transmutation.

News
====
* None periodic boundary conditions added
* ShapeFunction added to correct for fixed density approximation

Ask your questions
==================
https://groups.google.com/forum/#!forum/fullrmc

Installation
============
##### fullrmc requires:
* Python (>= 2.7 and < 3),
* NumPy (lowest version tested is 1.7.1)
* cython (lowest version tested is 0.21.1)
* matplotlib (lowest version tested is 1.4)
* pdbParser (lowest version tested is 0.1.2 - 0.1.3 is used starting from fullrmc 0.3.0 - 
  0.1.4 is used starting from fullrmc 1.0.0 - 0.1.5 is used starting from fullrmc 1.0.1)
* pysimplelog (lowest version tested is 0.1.7 -  0.1.4 is used starting from fullrmc 1.2.0 )

##### Installation using pip:
numpy and cython must be installed and updated manually. 

```bash
pip install -U "numpy>=1.7.1"
pip install -U "cython>=0.21.1"
pip install fullrmc
```

##### Installation by cloning github repository
Ensure all fullrmc required packages are installed and up to data by executing the 
following python script:
```python
# check whether all packages are already installed
from pkg_resources import parse_version as PV
for name, ver in [('numpy'      ,'1.7.1') ,
                  ('cython'     ,'0.21.1'),
                  ('pdbParser'  ,'0.1.5') ,
                  ('pysimplelog','0.2.1') ,
                  ('matplotlib' ,'1.4'  )]:
    try:
        lib = __import__(name)
    except:
        print '%s must be installed for fullrmc to run properly.'%(name)
    else:
        if PV(lib.__version__) < PV(ver):
            print '%s installed version %s is below minimum suggested version %s. Updating %s is highly recommended.'%(name, lib.__version__, ver, name)
        else:
            print '%s is installed properly and minimum version requirement is met.'%(name)
```
Locate python's site-packages by executing the following python script:
```python
import os
os.path.join(os.path.dirname(os.__file__), 'site_packages')
```
Navigate to site_packages folder and clone git repository from command line:
```bash
cd .../site_packages
git clone https://github.com/bachiraoun/fullrmc.git  
``` 
Change directory to .../site_packages/fullrmc/Extensions. Then compile fullrmc extensions from command line as the following:
```bash
cd .../site_packages/fullrmc/Extensions
python setup.py build_ext --inplace 
```

Online documentation
====================
http://bachiraoun.github.io/fullrmc/

Citing fullrmc
==============
If you use fullrmc in a scientific publication, 
we would appreciate citations to the following paper:


**Text entry:**

Bachir Aoun; Fullrmc, a Rigid Body Reverse Monte Carlo Modeling Package Enabled with Machine Learning and Artificial Intelligence; *J. Comput. Chem.* 2016, 37, 1102–1111. DOI: 10.1002/jcc.24304

**Bibtex entry:** 
```

    @article {JCC:JCC24304,
    author = {Aoun, Bachir},
    title = {Fullrmc, a rigid body reverse monte carlo modeling package enabled with machine learning and artificial intelligence},
    journal = {Journal of Computational Chemistry},
    volume = {37},
    number = {12},
    issn = {1096-987X},
    url = {http://dx.doi.org/10.1002/jcc.24304},
    doi = {10.1002/jcc.24304},
    pages = {1102--1111},
    keywords = {reverse monte carlo, rigid body, machine learning, pair distribution function, modeling},
    year = {2016},
    }
```

**EndNote entry:** 
```
    
    Provider: John Wiley & Sons, Ltd
    Content:text/plain; charset="UTF-8"
    
    TY  - JOUR
    AU  - Aoun, Bachir
    TI  - Fullrmc, a rigid body reverse monte carlo modeling package enabled with machine learning and artificial intelligence
    JO  - Journal of Computational Chemistry
    JA  - J. Comput. Chem.
    VL  - 37
    IS  - 12
    SN  - 1096-987X
    UR  - http://dx.doi.org/10.1002/jcc.24304
    DO  - 10.1002/jcc.24304
    SP  - 1102
    EP  - 1111
    KW  - reverse Monte Carlo
    KW  - rigid body
    KW  - machine learning
    KW  - pair distribution function
    KW  - modeling
    PY  - 2016
    ER  - 
```

    
Authors and developers
======================
* [Bachir Aoun](https://www.linkedin.com/in/bachiraoun) (Author, Developer) 


