# USAGE: python setup.py build_ext --inplace

# standard libraries imports
import os
import shutil
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# external libraries imports
import numpy as np

# fullrmc imports
from fullrmc.Core.Collection import get_path

# general variables
fullrmc_PATH     = get_path("fullrmc")
CURRENT_PATH     = os.path.abspath(os.path.dirname(__file__))
EXTENSIONS_PATH  = os.path.join(fullrmc_PATH, "Extensions")
DESTINATION_PATH = os.path.join(fullrmc_PATH, "Core")




##########################################################################################
################################# CREATE EXTENSIONS LIST #################################

c_cpp = [# get_reciprocal_basis
         #Extension('get_reciprocal_basis',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"get_reciprocal_basis.c")]),
         ## transform_coordinates
         #Extension('transform_coordinates',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"transform_coordinates.c")]),
         ## pair_distribution_histogram
         #Extension('pair_distribution_histogram',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"pair_distribution_histogram.c")]),
         ## distances
         #Extension('distances',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"distances.c")]),
         ## bonds
         #Extension('bonds',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"bonds.c")]),
         ## angles
         #Extension('angles',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"angles.c")]),
         ## improper_angles
         #Extension('improper_angles',
         #include_dirs=[np.get_include()],
         #sources = [os.path.join(EXTENSIONS_PATH,"improper_angles.c")]),
         ## atomic_coordination_number
         #Extension('atomic_coordination_number',
         #include_dirs=[np.get_include()],
         #language="c++",
         #sources = [os.path.join(EXTENSIONS_PATH,"atomic_coordination_number.cpp")]),
        ]

pyx = [# boundary conditions collection
       Extension('boundary_conditions_collection',
       include_dirs=[np.get_include()],
       sources = [os.path.join(EXTENSIONS_PATH,"boundary_conditions_collection.pyx")]),
       # reciprocal space
       Extension('reciprocal_space',
       include_dirs=[np.get_include()],
       sources = [os.path.join(EXTENSIONS_PATH,"reciprocal_space.pyx")]),
       # pairs distances
       Extension('pairs_distances',
       include_dirs=[np.get_include()],
       language="c",
       sources = [os.path.join(EXTENSIONS_PATH,"pairs_distances.pyx")]),
       # pairs histograms
       Extension('pairs_histograms',
       include_dirs=[np.get_include()],
       language="c",
       sources = [os.path.join(EXTENSIONS_PATH,"pairs_histograms.pyx")]),
       # vdw
       Extension('vdw',
       include_dirs=[np.get_include()],
       language="c",
       sources = [os.path.join(EXTENSIONS_PATH,"vdw.pyx")]),
       # bonds
       Extension('bonds',
       include_dirs=[np.get_include()],
       sources = [os.path.join(EXTENSIONS_PATH,"bonds.pyx")]),
       # angles
       Extension('angles',
       include_dirs=[np.get_include()],
       sources = [os.path.join(EXTENSIONS_PATH,"angles.pyx")]),
       # improper angles
       Extension('improper_angles',
       include_dirs=[np.get_include()],
       sources = [os.path.join(EXTENSIONS_PATH,"improper_angles.pyx")]),
       # atomic coordination number
       Extension('atomic_coordination_number',
       include_dirs=[np.get_include()],
       language="c++",
       sources = [os.path.join(EXTENSIONS_PATH,"atomic_coordination_number.pyx")]),
       
       ############# TESTS ########## 
       
       ]




##########################################################################################
################################### COMPILE EXTENSIONS ###################################
def compile(extensions, extensionsPath, destinationPath, cmdclass=None):
    # find maximum name length
    prefix = "*** *** Compiling "
    names  = [len(E.name) for E in extensions]
    maxLen = len(prefix)
    if len(names):
        maxLen += max( names )
    # compile extensions
    for E in extensions:
        m = prefix+"%s"%E.name
        m = "="*maxLen + "\n" + m + "\n" + "="*maxLen
        print "\n\n"+m
        # try to run setup
        try:
            if cmdclass is None:
                setup( ext_modules  = [E],
                       author       = "Bachir Aoun",
                       author_email = "fullrmc@gmail.com",
                       url          = "https://github.com/bachiraoun/fullrmc",
                       download_url = "http://bachiraoun.github.io/fullrmc/")
            else:
                setup( ext_modules  = [E], cmdclass=cmdclass,
                       author       = "Bachir Aoun",
                       author_email = "fullrmc@gmail.com",
                       url          = "https://github.com/bachiraoun/fullrmc",
                       download_url = "http://bachiraoun.github.io/fullrmc/")
        # intercept errors
        except (Exception, SystemExit) as e:
            m  = "\n\n--- --- COMPILATION ERROR --- ---\n"
            m += "--- Compilation error of %s :\n%s"%(E.name,e)
            print m
        # move compiled file to fullrmc.Core
        else:
            ldir = os.listdir(extensionsPath)
            if (E.name+".so" in ldir):
                fname = E.name+".so"
            elif (E.name+".pyd" in ldir):
                fname = E.name+".pyd"
            else:
                m  = "\n\n--- --- COMPILATION WARNING --- ---\n"
                m += "--- Extension not found: '%s' .so or .pyd extension is not created in '%s'"%(E.name, extensionsPath)
                print m
                continue
            # try to move compiled file to fullrmc core directory
            try:
                shutil.move(os.path.join(extensionsPath,fname), os.path.join(destinationPath,fname))
                m  = "\n\n+++ +++  COMPILATION INFORMATION +++ +++\n"
                m += "+++ Compiled file '%s' moved to '%s'"%(fname, destinationPath)
                print m
            # intercept error
            except Exception as e:
                m  = "\n\n--- --- COMPILATION WARNING --- ---\n"
                m += "--- Unable to copy compiled file '%s'. Try recompiling as admin (%s)"%(fname,e)
                print m
            
    
# Compile c and cpp files
compile(extensions      = c_cpp, 
        extensionsPath  = ".", 
        destinationPath = DESTINATION_PATH, 
        cmdclass=None)

# Compile cython files
compile(extensions      = pyx, 
        extensionsPath  = ".", 
        destinationPath = DESTINATION_PATH, 
        cmdclass={'build_ext': build_ext} )
        
