"""
In order to work properly, this script must be put one layer or directory
outside of fullrmc package directory.
""" 
# standard distribution imports
import os, sys, subprocess
# setup and distutils imports
try:
    from setuptools import setup
except:
    from distutils.core import setup
import fnmatch
from distutils.util import convert_path
from distutils.core import Extension
# import cython
try:
    from Cython.Distutils import build_ext
except:
    raise Exception("must install cython first. Try pip install cython")
# import numpy
try:
    import numpy as np
except:
    raise Exception("must install numpy first. Try pip install numpy")

    
# set package path and name
PACKAGE_PATH    = '.'
PACKAGE_NAME    = 'fullrmc'
EXTENSIONS_PATH = os.path.join(PACKAGE_NAME, "Extensions")

# check python version
if sys.version_info[:2] < (2, 6) or sys.version_info[:2] >= (3,):
    raise RuntimeError("Python version 2.6, 2.7 required.")

# automatically create MANIFEST.in
commands = [# include MANIFEST.in
            '# include this file, to ensure we can recreate source distributions',
            'include MANIFEST.in'
            # exclude all .log files
            '\n# exclude all logs',
            'global-exclude *.log',
            # exclude all .log files
            '\n# exclude all c cpp and compiled files',
            'global-exclude *.c',
            'global-exclude *.cpp',
            'global-exclude *.so',
            # exclude specific files
            'global-exclude debye_scattering*',
            'global-exclude atomic_distances*',
            'global-exclude *.rmc',
            # exclude all other non necessary files 
            '\n# exclude all other non necessary files ',
            'global-exclude .project',
            'global-exclude .pydevproject',
            # exclude all of the subversion metadata
            '\n# exclude all of the subversion metadata',
            'global-exclude *.svn*',
            'global-exclude .svn/*',
            'global-exclude *.git*',
            'global-exclude .git/*',
            # include all Example files
            '\n# include all Example files',
            'global-include %s/Examples/*.py'%PACKAGE_NAME,
            'global-include %s/Examples/*/*.py'%PACKAGE_NAME,
            'global-include %s/Examples/*/*.exp'%PACKAGE_NAME,
            'global-include %s/Examples/*/*.pdb'%PACKAGE_NAME,
            # include all LICENCE files
            '\n# include all license files found',
            'global-include %s/*LICENSE.*'%PACKAGE_NAME,
            # include all Extension .pyx files
            '\n# include all Extension .pyx files',
            'global-include %s/Extensions/*.pyx'%PACKAGE_NAME,
            # include all README files
            '\n# include all readme files found',
            'global-include %s/*README.*'%PACKAGE_NAME,
            'global-include %s/*readme.*'%PACKAGE_NAME
            ]         
with open('MANIFEST.in','w') as fd:
    for l in commands:
        fd.write(l)
        fd.write('\n')

# declare classifiers
CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Education
Intended Audience :: Developers
Natural Language :: English
License :: OSI Approved :: GNU Affero General Public License v3
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Topic :: Software Development
Topic :: Software Development :: Build Tools
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Physics
Operating System :: OS Independent
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# create descriptions
LONG_DESCRIPTION = ["fullrmc is a Reverse Monte Carlo (RMC) modelling package.",
                    "RMC is probably best known for its applications in condensed matter physics and solid state chemistry.",
                    "RMC is used to solve an inverse problem whereby an atomic/molecular model is adjusted until its atoms position have the greatest consistency with a set of experimental data.",
                    "fullrmc is a python package with its core and calculation modules optimized and compiled in Cython.",
                    "fullrmc is not a standard RMC package but it is rather unique in its approach to solving an atomic or molecular structure."
                    "fullrmc's Engine sub-module is the main module that contains the definition of 'Engine' which is the main and only class used to launch an RMC calculation.",
                    "Engine reads only Protein Data Bank formatted atomic configuration files '.pdb' and handles other definitions and attributes."]               
DESCRIPTION      = [ LONG_DESCRIPTION[0] ]

# get package info
PACKAGE_INFO={}
execfile(convert_path( os.path.join(PACKAGE_PATH, PACKAGE_NAME,'__pkginfo__.py') ), PACKAGE_INFO)
 
##############################################################################################
##################################### USEFUL DEFINITIONS #####################################
                            
def is_package(path):
    return (os.path.isdir(path) and os.path.isfile(os.path.join(path, '__init__.py')))

def get_packages(path, base="", exclude=None):
    if exclude is None:
        exclude = []
    assert isinstance(exclude, (list, set, tuple)), "exclude must be a list"
    exclude = [os.path.abspath(e) for e in exclude]
    packages = {}
    for item in os.listdir(path):
        d = os.path.join(path, item)
        if sum([e in os.path.abspath(d) for e in exclude]):
            continue
        if is_package(d):
            if base:
                module_name = "%(base)s.%(item)s" % vars()
            else:
                module_name = item
            packages[module_name] = d
            packages.update(get_packages(d, module_name, exclude))   
    return packages

DATA_EXCLUDE = ('*.rmc','*.data','*.png','*.xyz','*.log','*.pyc', '*~', '.*', '*.so', '*.pyd')
EXCLUDE_DIRECTORIES = ('*svn','*git','dist', 'EGG-INFO', '*.egg-info',)
def find_package_data(where='.', package='', relativePath='',
                      exclude=DATA_EXCLUDE, excludeDirectories=EXCLUDE_DIRECTORIES, 
                      onlyInPackages=True, showIgnored=False):
    out = {}
    stack = [(convert_path(where), '', package, onlyInPackages)]
    while stack:
        where, prefix, package, onlyInPackages = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if os.path.isdir(fn):
                bad_name = False
                for pattern in excludeDirectories:
                    if (fnmatch.fnmatchcase(name, pattern)
                        or fn.lower() == pattern.lower()):
                        bad_name = True
                        if showIgnored:
                            print >> sys.stderr, ("Directory %s ignored by pattern %s" % (fn, pattern))
                        break
                if bad_name:
                    continue
                if (os.path.isfile(os.path.join(fn, '__init__.py')) and not prefix):
                    if not package:
                        new_package = name
                    else:
                        new_package = package + '.' + name
                    stack.append((fn, '', new_package, False))
                else:
                    stack.append((fn, prefix + name + '/', package, onlyInPackages))
            elif package or not onlyInPackages:
                # is a file
                bad_name = False
                for pattern in exclude:
                    if (fnmatch.fnmatchcase(name, pattern)
                        or fn.lower() == pattern.lower()):
                        bad_name = True
                        if showIgnored:
                            print >> sys.stderr, ("File %s ignored by pattern %s" % (fn, pattern))
                        break
                if bad_name:
                    continue
                if len(relativePath):
                    out.setdefault(package, []).append(relativePath+'/'+prefix+name)
                else:
                    out.setdefault(package, []).append(prefix+name)
    return out
   
################################## END OF USEFUL DEFINITIONS #################################
##############################################################################################

# get extensions
EXTENSIONS = [# get_reciprocal_basis
              Extension('fullrmc.Core.get_reciprocal_basis',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"get_reciprocal_basis.pyx")]),
              # transform_coordinates
              Extension('fullrmc.Core.transform_coordinates',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"transform_coordinates.pyx")]),
              # pair_distribution_histogram
              Extension('fullrmc.Core.pair_distribution_histogram',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"pair_distribution_histogram.pyx")]),
              # distances
              Extension('fullrmc.Core.distances',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"distances.pyx")]),
              # bonds
              Extension('fullrmc.Core.bonds',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"bonds.pyx")]),
              # angles
              Extension('fullrmc.Core.angles',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"angles.pyx")]),
              # improper_angles
              Extension('fullrmc.Core.improper_angles',
              include_dirs=[np.get_include()],
              sources = [os.path.join(EXTENSIONS_PATH,"improper_angles.pyx")]),
              # atomic_coordination_number
              Extension('fullrmc.Core.atomic_coordination_number',
              include_dirs=[np.get_include()],
              language="c++",
              sources = [os.path.join(EXTENSIONS_PATH,"atomic_coordination_number.pyx")]),
              ]
CMDCLASS = {'build_ext' : build_ext}
            
# get packages and remove everything that is not fullrmc
PACKAGES = get_packages(path=PACKAGE_PATH, exclude=(os.path.join(PACKAGE_NAME,"docs"),))
for package in PACKAGES.keys():
    if PACKAGE_NAME not in package:
        PACKAGES.pop(package)

# get package data
PACKAGE_DATA = find_package_data(where=os.path.join(PACKAGE_NAME, "Examples"), 
                                 relativePath="Examples",
                                 package='fullrmc', 
                                 showIgnored=False)
                               
# create meta data
metadata = dict(# package
                name             = PACKAGE_NAME,
                packages         = PACKAGES.keys(),
                package_dir      = PACKAGES,
                # package data
                package_data     = PACKAGE_DATA,
                # info
                version          = PACKAGE_INFO['__version__'] ,
                author           = "Bachir AOUN",
                author_email     = "bachir.aoun@e-aoun.com",
                # Description
                description      = "\n".join(DESCRIPTION),
                long_description = "\n".join(LONG_DESCRIPTION),
                # Extensions
                ext_modules      = EXTENSIONS,
                cmdclass         = CMDCLASS,
                # online
                url              = "http://bachiraoun.github.io/fullrmc/index.html",
                download_url     = "https://github.com/bachiraoun/fullrmc",
                # Licence and classifiers
                license          = 'GNU',
                classifiers      = [_f for _f in CLASSIFIERS.split('\n') if _f],
                platforms        = ["Windows", "Linux", "Mac OS-X", "Unix"],
                # Dependent packages (distributions)
                install_requires = ["pysimplelog>=0.1.7",
                                    "pdbParser>=0.1.2",
                                    "matplotlib>=1.4" ], # it also needs numpy and cython, but this is left out for the user to install.
                setup_requires   = [''], 
                )

# setup
setup(**metadata)


    
