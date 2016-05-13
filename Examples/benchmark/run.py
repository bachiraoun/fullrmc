##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
# standard libraries imports
import os, sys, time

# external libraries imports
import numpy as np
import matplotlib.pyplot as plt

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
#####################################  CREATE ENGINE  ####################################
# parameters
NSTEPS  = 10000
pdbPath = 'system.pdb'
expData = 'experimental.gr'
# initialize engine
ENGINE = Engine(pdb=pdbPath, constraints=None)
# create constraints
PDF_CONSTRAINT = PairDistributionConstraint(engine=None, experimentalData=expData, weighting="atomicNumber")
EMD_CONSTRAINT = InterMolecularDistanceConstraint(engine=None)
B_CONSTRAINT   = BondConstraint(engine=None)
BA_CONSTRAINT  = BondsAngleConstraint(engine=None)
IA_CONSTRAINT  = ImproperAngleConstraint(engine=None)
# add constraints to engine
ENGINE.add_constraints([PDF_CONSTRAINT, EMD_CONSTRAINT, B_CONSTRAINT, BA_CONSTRAINT, IA_CONSTRAINT])
# initialize constraints definitions
B_CONSTRAINT.create_bonds_by_definition( bondsDefinition={"THF": [('O' ,'C1' , 1.22, 1.70),
                                                                  ('O' ,'C4' , 1.22, 1.70),
                                                                  ('C1','C2' , 1.25, 1.90),
                                                                  ('C2','C3' , 1.25, 1.90),
                                                                  ('C3','C4' , 1.25, 1.90),
                                                                  ('C1','H11', 0.58, 1.22),('C1','H12', 0.58, 1.22),
                                                                  ('C2','H21', 0.58, 1.22),('C2','H22', 0.58, 1.22),
                                                                  ('C3','H31', 0.58, 1.22),('C3','H32', 0.58, 1.22),
                                                                  ('C4','H41', 0.58, 1.22),('C4','H42', 0.58, 1.22)] })
BA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"THF": [ ('O'  ,'C1' ,'C4' , 105, 125),
                                                                      ('C1' ,'O'  ,'C2' , 100, 120),
                                                                      ('C4' ,'O'  ,'C3' , 100, 120),
                                                                      ('C2' ,'C1' ,'C3' , 95 , 115),
                                                                      ('C3' ,'C2' ,'C4' , 95 , 115),
                                                                      # H-C-H angle
                                                                      ('C1' ,'H11','H12', 98 , 118),
                                                                      ('C2' ,'H21','H22', 98 , 118),
                                                                      ('C3' ,'H31','H32', 98 , 118),
                                                                      ('C4' ,'H41','H42', 98 , 118),
                                                                      # H-C-O angle
                                                                      ('C1' ,'H11','O'  , 100, 120),
                                                                      ('C1' ,'H12','O'  , 100, 120),
                                                                      ('C4' ,'H41','O'  , 100, 120),
                                                                      ('C4' ,'H42','O'  , 100, 120),                                                                           
                                                                      # H-C-C
                                                                      ('C1' ,'H11','C2' , 103, 123),
                                                                      ('C1' ,'H12','C2' , 103, 123),
                                                                      ('C2' ,'H21','C1' , 103, 123),
                                                                      ('C2' ,'H21','C3' , 103, 123),
                                                                      ('C2' ,'H22','C1' , 103, 123),
                                                                      ('C2' ,'H22','C3' , 103, 123),
                                                                      ('C3' ,'H31','C2' , 103, 123),
                                                                      ('C3' ,'H31','C4' , 103, 123),
                                                                      ('C3' ,'H32','C2' , 103, 123),
                                                                      ('C3' ,'H32','C4' , 103, 123),
                                                                      ('C4' ,'H41','C3' , 103, 123),
                                                                      ('C4' ,'H42','C3' , 103, 123) ] })
IA_CONSTRAINT.create_angles_by_definition( anglesDefinition={"THF": [ ('C2','O','C1','C4', -15, 15),
                                                                      ('C3','O','C1','C4', -15, 15) ] })


##########################################################################################
####################################  DIFFERENT RUNS  ####################################
def run(nsteps, groups=None, pdf=False, vdw=False, bond=False, angle=False, improper=False, message=""):                                                                   
    # reset pdb
    ENGINE.set_pdb(pdbPath)
    ENGINE.set_groups(groups)
    # initialize constraints data
    PDF_CONSTRAINT.set_used(pdf)
    EMD_CONSTRAINT.set_used(vdw)
    B_CONSTRAINT.set_used(bond)
    BA_CONSTRAINT.set_used(angle)
    IA_CONSTRAINT.set_used(improper)
    # run
    ENGINE.initialize_used_constraints()
    tic = time.time()
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps)
    spentTime = time.time()-tic
    # log
    message += "constraints --> ("
    if pdf:
        message += "pdf,"
    if vdw:
        message += "vdw,"    
    if bond:
        message += "bond,"    
    if angle:
        message += "angle,"      
    if improper:
        message += "improper," 
    message += ") steps (%i) time (%i sec) \n"%(nsteps, spentTime)
    LOGGER.info("generated:%i - tried:%i - accepted:%i"%(ENGINE.generated, ENGINE.tried, ENGINE.accepted))
    LOGGER.info(message)
    return float(spentTime)/float(nsteps), ENGINE.tried , ENGINE.accepted

def benchmark_constraints(groupsList):
    print "================ Benchmark constraints ================"   
    # run benchmark
    benchmark = {}
    accepted  = {}
    tried     = {}
    for GN in groupsList:
        print "++++++++ %i atoms per group"%GN   
        groups = [np.array(item, dtype=np.int32) for item in zip( *[range(idx,ENGINE.numberOfAtoms,GN) for idx in range(GN)] )]
        benchmark[GN] = {}
        tried[GN]     = {}
        accepted[GN]  = {}
        print "---- No constraints"
        benchmark[GN]['0001_none'],     tried[GN]['0001_none'] ,     accepted[GN]['0001_none']     = run(nsteps=NSTEPS, groups=groups, pdf=False, vdw=False, bond=False, angle=False, improper=False, message="atoms (%i) "%GN)
        print "---- PairDistributionConstraint"
        benchmark[GN]['0002_pdf'],      tried[GN]['0002_pdf'] ,      accepted[GN]['0002_pdf']      = run(nsteps=NSTEPS, groups=groups, pdf=True,  vdw=False, bond=False, angle=False, improper=False, message="atoms (%i) "%GN)
        print "---- InterMolecularDistanceConstraint"
        benchmark[GN]['0003_vdw'],      tried[GN]['0003_vdw'] ,      accepted[GN]['0003_vdw']      = run(nsteps=NSTEPS, groups=groups, pdf=False, vdw=True,  bond=False, angle=False, improper=False, message="atoms (%i) "%GN)
        print "---- BondConstraint"
        benchmark[GN]['0004_bond'],     tried[GN]['0004_bond'] ,     accepted[GN]['0004_bond']     = run(nsteps=NSTEPS, groups=groups, pdf=False, vdw=False, bond=True,  angle=False, improper=False, message="atoms (%i) "%GN)
        print "---- BondsAngleConstraint"
        benchmark[GN]['0005_angle'],    tried[GN]['0005_angle'] ,    accepted[GN]['0005_angle']    = run(nsteps=NSTEPS, groups=groups, pdf=False, vdw=False, bond=False, angle=True,  improper=False, message="atoms (%i) "%GN)
        print "---- ImproperAngleConstraint"
        benchmark[GN]['0006_improper'], tried[GN]['0006_improper'] , accepted[GN]['0006_improper'] = run(nsteps=NSTEPS, groups=groups, pdf=False, vdw=False, bond=False, angle=False, improper=True,  message="atoms (%i) "%GN)
        print "---- All constraints"
        benchmark[GN]['0007_all'],      tried[GN]['0007_all'] ,      accepted[GN]['0007_all']      = run(nsteps=NSTEPS, groups=groups, pdf=True,  vdw=True,  bond=True,  angle=True,  improper=True,  message="atoms (%i) "%GN)
    
    # plot
    plt.figure()
    atoms = sorted(benchmark.keys())
    bench = [np.array(atoms)]
    accep = [np.array(atoms)]
    tri   = [np.array(atoms)]
    header = "groups "
    for key in sorted(benchmark[atoms[0]].keys()):
        times = np.array( [ benchmark[n][key] for n in atoms] )
        header += " %s"%key.split("_")[1]
        bench.append(times)
        tri.append( np.array( [ tried[n][key] for n in atoms] ) )
        accep.append( np.array( [ accepted[n][key] for n in atoms] ) )
        plt.plot(atoms, times, label=key.split("_")[1])
    # annotate tried(accepted)
    for i, txt in enumerate( accep[-1] ):
        plt.gca().annotate(str(tri[-1][i])+"("+str(txt)+")", (bench[0][i],bench[-1][i]))
    # show plot
    plt.legend()
    plt.title("Constraints Benchmark")
    plt.xlabel("number of atoms")
    plt.ylabel("time per step (s)")
    # save
    np.savetxt(fname="benchmark_constraints_time.dat",     X=np.transpose(bench), fmt='%.10f', delimiter='    ', newline='\n', header=header)
    np.savetxt(fname="benchmark_constraints_tried.dat",    X=np.transpose(tri),   fmt='%.10f', delimiter='    ', newline='\n', header=header)
    np.savetxt(fname="benchmark_constraints_accepted.dat", X=np.transpose(accep), fmt='%.10f', delimiter='    ', newline='\n', header=header)


def benchmark_nsteps(constraint, groupSize=13, stepsList=range(5000,105000,5000)):
    print "================ Benchmark number of steps ================"   
    CS = {'pdf':False, 'vdw':False, 'bond':False, 'angle':False, 'improper':False}
    if not constraint in CS.keys():
        if constraint == "all":
            for k in CS.keys():
                CS[k]=True
        else:
            assert constraint == "none"
    print  "++++++++ Used constraints are %s"%CS  
    # set groups
    groups = [np.array(item, dtype=np.int32) for item in zip( *[range(idx,ENGINE.numberOfAtoms,groupSize) for idx in range(groupSize)] )]
    # run benchmark
    benchmark = []
    accepted  = []
    tried     = []
    stepsList = sorted(stepsList)
    for ns in stepsList:
        print "---- %i steps"%ns
        bench, tri, accep = run(nsteps=ns, groups=groups, pdf=CS['pdf'], vdw=CS['vdw'], bond=CS['bond'], angle=CS['angle'], improper=CS['improper'], message="atoms (%i) "%groupSize)
        benchmark.append( bench )
        tried.append( tri )
        accepted.append( accep )
    
    # create data
    benchmark = [np.array(stepsList), np.array(benchmark)]
    tried     = [np.array(stepsList), np.array(tried)]
    accepted  = [np.array(stepsList), np.array(accepted)]
    # plot
    plt.figure()
    plt.plot(benchmark[0], benchmark[1])
    # annotate tried(accepted)
    for i, txt in enumerate( accepted[-1] ):
        plt.gca().annotate(str(tried[-1][i])+"("+str(txt)+")", (benchmark[0][i],benchmark[-1][i]))
    # show plot
    plt.title("Number of steps Benchmark")
    plt.xlabel("number of steps")
    plt.ylabel("time per step (s)")
    # save
    np.savetxt(fname='benchmark_%sSteps_%iGroupSize_time.dat'%(constraint,groupSize),     X=np.transpose(benchmark), fmt='%.10f', delimiter='    ', newline='\n', header="steps timePerStep(s)")
    np.savetxt(fname='benchmark_%sSteps_%iGroupSize_tried.dat'%(constraint,groupSize),    X=np.transpose(tried),     fmt='%.10f', delimiter='    ', newline='\n', header="steps timePerStep(s)")
    np.savetxt(fname='benchmark_%sSteps_%iGroupSize_accepted.dat'%(constraint,groupSize), X=np.transpose(accepted),  fmt='%.10f', delimiter='    ', newline='\n', header="steps timePerStep(s)")
    
def load_and_plot_constraints_benchmark():
    benchmark = np.loadtxt(fname='benchmark_constraints_time.dat')
    tried     = np.loadtxt(fname='benchmark_constraints_tried.dat')
    accepted  = np.loadtxt(fname='benchmark_constraints_accepted.dat')
    # plot
    keys = open('benchmark_constraints_time.dat').readlines()[0].split("#")[1].split()
    minY = 0
    maxY = 0
    for idx in range(1,len(keys)): 
        style = '-'
        if keys[idx] == 'all':
            style = '.-'
        # mean accepted
        meanAccepted = int( sum(benchmark[:,0]*accepted[:,idx])/sum(benchmark[:,0]) )
        plt.plot(benchmark[:,0], benchmark[:,idx], style, label=keys[idx]+" (%i)"%meanAccepted)
        minY = min(minY, min(benchmark[:,idx]) )
        maxY = max(minY, max(benchmark[:,idx]) )
    # annotate tried(accepted) 
    for i, txt in enumerate( accepted[:,-1] ):
        T = 100*float(tried[i,-1])/float(tried[1,1])
        A = 100*float(accepted[i,-1])/float(tried[1,1])
        plt.gca().annotate( "%.2f%% (%.2f%%)"%(T,A),  #"%i (%i)"%( int(tried[i,-1]),int(txt) ), 
                            xy = (benchmark[i,0],benchmark[i,-1]),
                            rotation=90,
                            horizontalalignment='center',
                            verticalalignment='bottom')     
    # show plot
    plt.legend(frameon=False, loc='upper left')
    plt.xlabel("Number of atoms per group")
    plt.ylabel("Time per step (s)")
    plt.gcf().patch.set_facecolor('white')
    # set fig size
    #figSize = plt.gcf().get_size_inches()
    #figSize[1] = figSize[1]+figSize[1]/2.
    #plt.gcf().set_size_inches(figSize, forward=True)
    plt.ylim((None, maxY+0.3*(maxY-minY)))
    # save
    plt.savefig("benchmark_constraint.png")
    # plot
    plt.show()
    
def load_and_plot_steps_benchmark(constraint="all", groupSize=13):
    benchmark = np.loadtxt(fname='benchmark_%sSteps_%iGroupSize_time.dat'%(constraint,groupSize) )
    tried     = np.loadtxt(fname='benchmark_%sSteps_%iGroupSize_tried.dat'%(constraint,groupSize) )
    accepted  = np.loadtxt(fname='benchmark_%sSteps_%iGroupSize_accepted.dat'%(constraint,groupSize) )
    # plot benchmark
    plt.plot(benchmark[:,0], benchmark[:,1])
    minY = min(benchmark[:,1]) 
    maxY = max(benchmark[:,1]) 
    # annotate tried(accepted) 
    for i, txt in enumerate( accepted[:,-1] ):
        T = 100*float(tried[i,-1])/float(benchmark[i,0])
        A = 100*float(accepted[i,-1])/float(benchmark[i,0])
        plt.gca().annotate( "%.2f%% (%.2f%%)"%(T,A),  #str(int(tried[i,-1]))+" ("+str(int(txt))+")", 
                            xy = (benchmark[i,0],benchmark[i,-1]),
                            rotation=90,
                            horizontalalignment='center',
                            verticalalignment='bottom')     
    # show plot
    plt.legend(frameon=False, loc='upper left')
    plt.xlabel("Number of steps")
    plt.ylabel("Time per step (s)")
    plt.gcf().patch.set_facecolor('white')
    # set fig size
    #figSize = plt.gcf().get_size_inches()
    #figSize[1] = figSize[1]+figSize[1]/2.
    #plt.gcf().set_size_inches(figSize, forward=True)
    plt.ylim((None, maxY+0.3*(maxY-minY)))
    # save
    plt.savefig("benchmark_steps.png")
    # plot
    plt.show()
    
    

##########################################################################################
#####################################  RUN BENCHMARKS  ################################### 
benchmark_constraints( range(1,30,1) )
benchmark_nsteps(constraint = 'all', groupSize=13, stepsList=range(5000,105000,5000))
# show benchmark plots
plt.show()

##########################################################################################
####################################  PLOT BENCHMARKS  ###################################  
#load_and_plot_steps_benchmark()
#load_and_plot_constraints_benchmark()







        



        
