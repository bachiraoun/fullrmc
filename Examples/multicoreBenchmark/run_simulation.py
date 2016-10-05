##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys, time, multiprocessing
# external libraries imports
import numpy as np

# fullrmc library imports
from fullrmc.Globals import LOGGER
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint    
    

##########################################################################################
##################################  SHUT DOWN LOGGING  ###################################
LOGGER.set_minimum_level(sys.maxint, stdoutFlag=True, fileFlag=True)


##########################################################################################
#####################################  USER VARIBLES  ####################################
# to get real number of physical cores one should use 
# https://pypi.python.org/pypi/cpu_cores instead of multiprocessing.cpu_count()
NCORES = range(1,multiprocessing.cpu_count()+1)# [1,2,3,4]
NSTEPS = 1000


##########################################################################################
###############################  DEFINE run_engine METHOD  ###############################
# dirname
DIR_PATH = os.path.dirname( os.path.realpath(__file__) )

# engine file names
expFileName    = "thf_pdf.exp"
pdbFileName    = "thf.pdb" 

# engine variables
expPath        = os.path.join(DIR_PATH, expFileName)
pdbPath        = os.path.join(DIR_PATH, pdbFileName)
    
# create engine method
def run_engine(PDF=True, IMD=True, B=True, BA=True, IA=True, 
               molecular=True, nsteps=10000, ncores=1):
    ENGINE = Engine(path=None)
    ENGINE.set_pdb(pdbPath)
    # create experimental constraints
    if PDF:
        C = PairDistributionConstraint(experimentalData=expPath, weighting="atomicNumber")
        ENGINE.add_constraints(C)
    # create and define molecular constraints
    if IMD:
        C = InterMolecularDistanceConstraint(defaultDistance=1.5)
        ENGINE.add_constraints(C)
    if B:
        C = BondConstraint()
        ENGINE.add_constraints(C)
        C.create_bonds_by_definition( bondsDefinition={"THF": [('O' ,'C1' , 1.29, 1.70),
                                                               ('O' ,'C4' , 1.29, 1.70),
                                                               ('C1','C2' , 1.29, 1.70),
                                                               ('C2','C3' , 1.29, 1.70),
                                                               ('C3','C4' , 1.29, 1.70),
                                                               ('C1','H11', 0.58, 1.15),('C1','H12', 0.58, 1.15),
                                                               ('C2','H21', 0.58, 1.15),('C2','H22', 0.58, 1.15),
                                                               ('C3','H31', 0.58, 1.15),('C3','H32', 0.58, 1.15),
                                                               ('C4','H41', 0.58, 1.15),('C4','H42', 0.58, 1.15)] })
    if BA:
        C = BondsAngleConstraint()
        ENGINE.add_constraints(C)
        C.create_angles_by_definition( anglesDefinition={"THF": [ ('O'  ,'C1' ,'C4' , 95 , 135),
                                                                  ('C1' ,'O'  ,'C2' , 95 , 135),
                                                                  ('C4' ,'O'  ,'C3' , 95 , 135),
                                                                  ('C2' ,'C1' ,'C3' , 90 , 120),
                                                                  ('C3' ,'C2' ,'C4' , 90 , 120),
                                                                  # H-C-H angle
                                                                  ('C1' ,'H11','H12', 95 , 125),
                                                                  ('C2' ,'H21','H22', 95 , 125),
                                                                  ('C3' ,'H31','H32', 95 , 125),
                                                                  ('C4' ,'H41','H42', 95 , 125),
                                                                  # H-C-O angle
                                                                  ('C1' ,'H11','O'  , 100, 120),
                                                                  ('C1' ,'H12','O'  , 100, 120),
                                                                  ('C4' ,'H41','O'  , 100, 120),
                                                                  ('C4' ,'H42','O'  , 100, 120),                                                                           
                                                                  # H-C-C
                                                                  ('C1' ,'H11','C2' , 80, 123),
                                                                  ('C1' ,'H12','C2' , 80, 123),
                                                                  ('C2' ,'H21','C1' , 80, 123),
                                                                  ('C2' ,'H21','C3' , 80, 123),
                                                                  ('C2' ,'H22','C1' , 80, 123),
                                                                  ('C2' ,'H22','C3' , 80, 123),
                                                                  ('C3' ,'H31','C2' , 80, 123),
                                                                  ('C3' ,'H31','C4' , 80, 123),
                                                                  ('C3' ,'H32','C2' , 80, 123),
                                                                  ('C3' ,'H32','C4' , 80, 123),
                                                                  ('C4' ,'H41','C3' , 80, 123),
                                                                  ('C4' ,'H42','C3' , 80, 123) ] })
    if IA:
        C = ImproperAngleConstraint()    
        ENGINE.add_constraints(C)
        C.create_angles_by_definition( anglesDefinition={"THF": [ ('C2','O','C1','C4', -15, 15),
                                                                  ('C3','O','C1','C4', -15, 15) ] })
    # initialize constraints data
    ENGINE.initialize_used_constraints()
    # run engine
    if molecular:
        ENGINE.set_groups_as_molecules()
        print 'molecular, %s atoms, %s steps, %2s cores'%(ENGINE.numberOfAtoms, nsteps, ncores),
        tic = time.time()
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, restartPdb=None, ncores=ncores)
        elapsed = float(time.time()-tic)/float(nsteps)
        print ' -- > %s seconds per step'%(elapsed,)
    else:
        ENGINE.set_groups_as_atoms() 
        print 'atomic   , %s atoms, %s steps, %2s cores'%(ENGINE.numberOfAtoms, nsteps, ncores),
        tic = time.time()
        ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, restartPdb=None, ncores=ncores)
        elapsed = float(time.time()-tic)/float(nsteps)
        print ' -- > %s seconds per step'%(elapsed,)
        # return elapsed time
    return elapsed


    

##########################################################################################
#####################################  RUN BENCHMARK  ####################################
BENCHMARK = {}
BENCHMARK["PDF"]                  = {'atomic':[[],[]], 'molecular':[[],[]]}
BENCHMARK["interatomic distance"] = {'atomic':[[],[]], 'molecular':[[],[]]}
BENCHMARK["bonds"]                = {'atomic':[[],[]], 'molecular':[[],[]]}
BENCHMARK["bond_angles"]          = {'atomic':[[],[]], 'molecular':[[],[]]}
BENCHMARK["improper angles"]      = {'atomic':[[],[]], 'molecular':[[],[]]}
BENCHMARK["all"]                       = {'atomic':[[],[]], 'molecular':[[],[]]}

# run benchmarks
for cname in BENCHMARK.keys():
    # set flags
    PDF=False;IMD=False;B=False;BA=False;IA=False
    if cname == "PDF":
       PDF = True 
    elif cname == "interatomic distance":
       IMD = True 
    elif cname == "bonds":
       B = True 
    elif cname == "bond angles":
       BA = True 
    elif cname == "improper angles":
       IA = True 
    elif cname == "all":
        PDF=True;IMD=True;B=True;BA=True;IA=True
    print 'running PDF=%5s, IMD=%5s, B=%5s, BA=%5s, IA=%5s'%(PDF, IMD, B, BA, IA)
    print '========================================================='
    # run cores atomic rmc    
    for N in sorted(NCORES):
        T = run_engine(PDF=PDF, IMD=IMD, B=B, BA=BA, IA=IA, molecular=False, nsteps=NSTEPS, ncores=N)
        BENCHMARK[cname]['atomic'][0].append(N)
        BENCHMARK[cname]['atomic'][1].append(T)
    # run cores molecular rmc   
    for N in sorted(NCORES):    
        T = run_engine(PDF=PDF, IMD=IMD, B=B, BA=BA, IA=IA, molecular=True,  nsteps=NSTEPS, ncores=N)
        BENCHMARK[cname]['molecular'][0].append(N)
        BENCHMARK[cname]['molecular'][1].append(T)
    
    
##########################################################################################
##########################################  PLOT  ########################################
import matplotlib.pyplot as plt

for constraint, values in BENCHMARK.items():
    # create figure
    fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0.1})
    fig.patch.set_facecolor('white')
    fig.suptitle(constraint)
    axes[1].set_xlabel("number of cores")
    axes[0].set_ylabel("time per step (sec.)")
    axes[1].set_ylabel("time per step (sec.)")
    axes[0].set_xlim([0,max(NCORES)+1])
    axes[1].set_xlim([0,max(NCORES)+1])
    # plot atomic
    axes[0].plot(values['atomic'][0], values['atomic'][1], '-o', label="atomic")
    axes[0].legend(frameon=False)
    axes[0].set_ylim([ min(values['atomic'][1])-0.1*min(values['atomic'][1]), 
                       max(values['atomic'][1])+0.1*max(values['atomic'][1])] )
    # plot molecular
    axes[1].plot(values['molecular'][0], values['molecular'][1], '-o', label="molecular")
    axes[1].legend(frameon=False) 
    axes[1].set_ylim([ min(values['molecular'][1])-0.1*min(values['molecular'][1]), 
                       max(values['molecular'][1])+0.1*max(values['molecular'][1])] )
# show plot
plt.show()
exit()






