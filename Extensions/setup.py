# USAGE: python setup.py build_ext --inplace

# standard libraries imports
import os
import shutil
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# external libraries imports
import numpy as np

# get fullrmc path
try:
    from fullrmc.Core.Collection import get_path
    fullrmc_PATH = get_path("fullrmc")
# this only means that fullrmc has never been compiled and hopefully 
# it's been pulled from github and correctly handled.
except: 
    setupPath = os.path.split( os.path.abspath(__file__) )[0]
    fullrmc_PATH = os.path.split( setupPath )[0]

# general variables
EXTENSIONS_PATH  = os.path.abspath(os.path.dirname(__file__))
DESTINATION_PATH = os.path.join(fullrmc_PATH, "Core")


##########################################################################################
################################### CHANGE COMPILER  #####################################
#os.environ["CC"] = "/usr/local/Cellar/gcc/6.1.0/bin/gcc-6"


##########################################################################################
################################## COMPILER ARGUMENTS ####################################
EXTRA_COMPILE_ARGS = []
EXTRA_LINK_ARGS    = []
#### uncomment the following to compile with openmp allowing multithreading
EXTRA_COMPILE_ARGS.append("-fopenmp")
EXTRA_LINK_ARGS.append("-fopenmp")


##########################################################################################
################################# CREATE EXTENSIONS LIST #################################

c_cpp = []

pyx = [### boundary conditions collection
       Extension('boundary_conditions_collection',
                 include_dirs=[np.get_include()],
                 language="c",
                 sources = [os.path.join(EXTENSIONS_PATH,"boundary_conditions_collection.pyx")]),
       ### reciprocal space
       Extension('reciprocal_space',
                 include_dirs=[np.get_include()],
                 language="c",
                 sources = [os.path.join(EXTENSIONS_PATH,"reciprocal_space.pyx")]),
       ### pairs distances
       Extension('pairs_distances',
                 include_dirs=[np.get_include()],
                 language="c",
                 extra_compile_args = EXTRA_COMPILE_ARGS,
                 extra_link_args    = EXTRA_LINK_ARGS,
                 sources = [os.path.join(EXTENSIONS_PATH,"pairs_distances.pyx")]),
       ### pairs histograms
       Extension('pairs_histograms',
                 include_dirs=[np.get_include()],
                 language="c",
                 extra_compile_args = EXTRA_COMPILE_ARGS,
                 extra_link_args    = EXTRA_LINK_ARGS,
                 sources = [os.path.join(EXTENSIONS_PATH,"pairs_histograms.pyx")]),
       ### atomic_distances
       Extension('atomic_distances',
                 include_dirs=[np.get_include()],
                 language="c",
                 extra_compile_args = EXTRA_COMPILE_ARGS,
                 extra_link_args    = EXTRA_LINK_ARGS,
                 sources = [os.path.join(EXTENSIONS_PATH,"atomic_distances.pyx")]),
       #### coordination number
       Extension('atomic_coordination',
                 include_dirs=[np.get_include()],
                 language="c",
                 extra_compile_args = EXTRA_COMPILE_ARGS,
                 extra_link_args    = EXTRA_LINK_ARGS,
                 sources = [os.path.join(EXTENSIONS_PATH,"atomic_coordination.pyx")]),
       ### bonds
       Extension('bonds',
                 include_dirs=[np.get_include()],
                 language="c",
                 sources = [os.path.join(EXTENSIONS_PATH,"bonds.pyx")]),
       ### angles
       Extension('angles',
                 language="c",
                 include_dirs=[np.get_include()],
                 sources = [os.path.join(EXTENSIONS_PATH,"angles.pyx")]),
       ### improper angles
       Extension('improper_angles',
                 include_dirs=[np.get_include()],
                 language="c",
                 sources = [os.path.join(EXTENSIONS_PATH,"improper_angles.pyx")]),
       ]




##########################################################################################
################################### COMPILE EXTENSIONS ###################################
COMPILATION_INFO = []
COMPILATION_INFO.append( "\n" )
COMPILATION_INFO.append( "======================== COMPILATION SUMMARY ======================== " )
COMPILATION_INFO.append( "===================================================================== " )

def compile_extention(E, extensionsPath, destinationPath, cmdclass, infoList):
    # try to run setup
    try:
        setup( ext_modules  = [E], cmdclass=cmdclass,
               author       = "Bachir Aoun",
               author_email = "fullrmc@gmail.com",
               url          = "https://github.com/bachiraoun/fullrmc",
               download_url = "http://bachiraoun.github.io/fullrmc/")
    # intercept errors
    except (Exception, SystemExit) as e:
        # check fopenmp flag
        compArgs = [arg for arg in E.extra_compile_args if "fopenmp" not in arg]
        linkArgs = [arg for arg in E.extra_link_args    if "fopenmp" not in arg]
        if len(E.extra_compile_args)!=len(compArgs) or len(E.extra_link_args)!=len(linkArgs):
            E.extra_compile_args = compArgs
            E.extra_link_args    = linkArgs
            # try to compile without openmp
            t = "\n--- --- COMPILATION ERROR --- ---\n"
            m = "--- Info --> Compilation of '%s' with fopenmp failed: %s\n             Trying to compile without fopenmp ..."%(E.name,e)
            print t+m
            infoList.append(m)
            compile_extention( E               = E, 
                               extensionsPath  = extensionsPath, 
                               destinationPath = destinationPath, 
                               cmdclass        = cmdclass,
                               infoList        = infoList)
        else:                   
            t = "\n--- --- COMPILATION ERROR --- ---\n"
            m = "--- Error --> Compilation of '%s': %s"%(E.name,e)
            infoList.append(m)
            print t+m
    # move compiled file to fullrmc.Core
    else:
        ldir = os.listdir(extensionsPath)
        if (E.name+".so" in ldir):
            fname = E.name+".so"
        elif (E.name+".pyd" in ldir):
            fname = E.name+".pyd"
        else:
            t = "\n--- --- COMPILATION WARNING --- ---\n"
            m = "--- Warning -->  Extension not found: '%s' .so or .pyd extension is not created in '%s'"%(E.name, extensionsPath)
            infoList.append(m)
            print t+m
            return
        # try to move compiled file to fullrmc core directory
        try:
            shutil.move(os.path.join(extensionsPath,fname), os.path.join(destinationPath,fname))
            t = "\n+++ +++  COMPILATION INFORMATION +++ +++\n"
            if "-fopenmp" in E.extra_compile_args or "-fopenmp" in E.extra_link_args:
                m = "+++ Info --> Compiled file '%s' with fopenmp flag moved to '%s'"%(fname, destinationPath)
            else:
                m = "+++ Info --> Compiled file '%s' moved to '%s'"%(fname, destinationPath)
            infoList.append(m)
            print t+m
        # intercept error
        except Exception as e:
            t = "\n--- --- COMPILATION WARNING --- ---\n"
            m = "--- Warning --> Unable to copy compiled file '%s'. Try recompiling as admin (%s)"%(fname,e)
            infoList.append(m)
            print t+m
                
                     
def compile_all(extensions, extensionsPath, destinationPath, cmdclass, infoList):
    if cmdclass is None:
        cmdclass = {}
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
        compile_extention( E               = E, 
                           extensionsPath  = extensionsPath, 
                           destinationPath = destinationPath, 
                           cmdclass        = cmdclass,
                           infoList        = infoList)
        
            
# change directory to extensions' one
try:
    os.chdir(EXTENSIONS_PATH) 
except:
    print "Unable to change directory to '%s'. \
Extension are built in place and need to be move \
by hand to '%s'"%(EXTENSIONS_PATH, DESTINATION_PATH)
    
# Compile c and cpp files
compile_all(extensions      = c_cpp, 
            extensionsPath  = EXTENSIONS_PATH, 
            destinationPath = DESTINATION_PATH, 
            cmdclass        = None,
            infoList        = COMPILATION_INFO)

# Compile cython files
compile_all(extensions      = pyx, 
            extensionsPath  = EXTENSIONS_PATH, 
            destinationPath = DESTINATION_PATH, 
            cmdclass        = {'build_ext': build_ext},
            infoList        = COMPILATION_INFO )
        
print "\n".join(COMPILATION_INFO)
