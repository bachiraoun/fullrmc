##########################################################################################
##############################  IMPORTING USEFUL DEFINITIONS  ############################
import time, multiprocessing
from collections import OrderedDict
import numpy as np


##########################################################################################
#####################################  USER VARIBLES  ####################################
# to get real number of physical cores one should use 
# https://pypi.python.org/pypi/cpu_cores instead of multiprocessing.cpu_count()
CORES            = range(1,multiprocessing.cpu_count()+1)# [1,2,3,4]
AVERAGE          = 100
NUMBER_OF_ATOMS  = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]
HISTOGRAMS       = True
PAIRS_DISTANCES  = True
ATOMIC_DISTANCES = True


##########################################################################################
##################################### INIT VARIABLES #####################################
CORES            = sorted(CORES)
NUMBER_OF_ATOMS  = sorted(NUMBER_OF_ATOMS)
DATA             = {}


##########################################################################################
####################################### HISTOGRAMS #######################################
print 'peformance is computed as the mean time of %s runs'%AVERAGE

if HISTOGRAMS:
    from fullrmc.Core.pairs_histograms import single_pairs_histograms
    
    print 'testing pairs_histograms computations'
    print '=====================================' 
    atomIndex = np.int32(100)
    DATA["histograms"] = OrderedDict()
    for SIZE in NUMBER_OF_ATOMS:
        DATA["histograms"][SIZE] = [[],[]]
        distances = 1000*np.random.random((SIZE,)).astype(np.float32)
        
        moleculeIndex = []
        molIdx = 0
        while len(moleculeIndex)<len(distances):
            add = min( len(distances)/1000, len(distances)-len(moleculeIndex) )
            moleculeIndex.extend( [molIdx]*add )
            molIdx +=1
        moleculeIndex = np.array(moleculeIndex, dtype=np.int32)
        
        elementIndex = []
        elIdx = 0
        while len(elementIndex)<len(distances):
            add = min( len(distances)/100, len(distances)-len(elementIndex) )
            elementIndex.extend( [elIdx]*add )
            elIdx +=1
        elementIndex = np.array(elementIndex, dtype=np.int32)
        
        
        minDistance = np.float32(1)
        maxDistance = np.float32(500)
        bin = np.float32(0.1)
        histSize = np.int32( (maxDistance-minDistance)/bin + 10 )
        
        for n in CORES:
            mtime = []
            for _ in range(AVERAGE):
                hintra = np.zeros((elIdx,elIdx,histSize), dtype=np.float32)
                hinter = np.zeros((elIdx,elIdx,histSize), dtype=np.float32)
                tic = time.time()              
                single_pairs_histograms( atomIndex     = atomIndex, 
                                         distances     = distances,
                                         moleculeIndex = moleculeIndex,
                                         elementIndex  = elementIndex,
                                         hintra        = hintra,
                                         hinter        = hinter,
                                         minDistance   = minDistance,
                                         maxDistance   = maxDistance,
                                         bin           = bin, 
                                         allAtoms      = True,
                                         ncores        = np.int32(n))
                mtime.append(time.time()-tic)
            print "histograms: %s atoms, ncores %s --> %.10f "%(SIZE, n, sum(mtime)/len(mtime))
            DATA["histograms"][SIZE][0].append(n)
            DATA["histograms"][SIZE][1].append(sum(mtime)/len(mtime)) 
        
    
##########################################################################################    
#################################### PAIRS DISTANCES #####################################
if PAIRS_DISTANCES:
    from fullrmc.Core.pairs_distances import pairs_distances_to_indexcoords, pairs_distances_to_multi_points
    
    print '\n'
    print 'testing pairs_distances computations'
    print '====================================' 
    
    DATA["distance using index"]  = OrderedDict()
    DATA["distance to multiple points"] = OrderedDict()
    for SIZE in NUMBER_OF_ATOMS:
        DATA["distance using index"][SIZE] = [[],[]]
        DATA["distance to multiple points"][SIZE] = [[],[]]
        atomIndex = np.int32(0)
        coords = np.random.random((SIZE, 3)).astype(np.float32)
        points = np.random.random((3,100)).astype(np.float32)
        
        basis = np.ones((3, 3)).astype(np.float32)
        
        for n in CORES:
            mtime = []
            for _ in range(AVERAGE):
                tic = time.time()
                pairs_distances_to_indexcoords( atomIndex = atomIndex,
                                                coords    = coords,
                                                basis     = basis,
                                                isPBC     = True,
                                                allAtoms  = True,
                                                ncores    = np.int32(n))
            mtime.append(time.time()-tic)
            print "distance using index: %s atoms, ncores %s --> %.10f "%(SIZE, n, sum(mtime)/len(mtime))
            DATA["distance using index"][SIZE][0].append(n)
            DATA["distance using index"][SIZE][1].append(sum(mtime)/len(mtime)) 
        
        for n in CORES:
            mtime = []
            for _ in range(AVERAGE):
                tic = time.time()
                pairs_distances_to_multi_points( points    = points,
                                                 coords    = coords,
                                                 basis     = basis,
                                                 isPBC     = True,
                                                 ncores    = np.int32(n))
            mtime.append(time.time()-tic)
            print "distance to multiple points: %s atoms, ncores %s --> %.10f "%(SIZE, n, sum(mtime)/len(mtime))
            DATA["distance to multiple points"][SIZE][0].append(n)
            DATA["distance to multiple points"][SIZE][1].append(sum(mtime)/len(mtime)) 
    
    
##########################################################################################    
#################################### ATOMIC DISTANCES ####################################
if ATOMIC_DISTANCES:
    from fullrmc.Core.atomic_distances import single_atomic_distances_dists
    
    print '\n'
    print 'testing atomic_distances computations'
    print '=====================================' 
    
    atomIndex = np.int32(0)
    distances = 1000*np.random.random((SIZE, )).astype(np.float32)
    
    
    DATA["intermolecular distances"] = OrderedDict()
    for SIZE in NUMBER_OF_ATOMS:
        DATA["intermolecular distances"][SIZE] = [[],[]]
        
        moleculeIndex = []
        molIdx = 0
        while len(moleculeIndex)<len(distances):
            add = min( len(distances)/1000, len(distances)-len(moleculeIndex) )
            moleculeIndex.extend( [molIdx]*add )
            molIdx +=1
        moleculeIndex = np.array(moleculeIndex, dtype=np.int32)
        
        elementIndex = []
        elIdx = 0
        while len(elementIndex)<len(distances):
            add = min( len(distances)/100, len(distances)-len(elementIndex) )
            elementIndex.extend( [elIdx]*add )
            elIdx +=1
        elementIndex = np.array(elementIndex, dtype=np.int32)
        
        lowerLimit = np.ones((elIdx,elIdx,1), dtype=np.float32)
        upperLimit = 3*np.ones((elIdx,elIdx,1), dtype=np.float32)
        
        for n in CORES:
            mtime = []
            for _ in range(AVERAGE):
                dintra = np.zeros((elIdx,elIdx,1), dtype=np.float32)
                dinter = np.zeros((elIdx,elIdx,1), dtype=np.float32)
                nintra = np.zeros((elIdx,elIdx,1), dtype=np.int32)
                ninter = np.zeros((elIdx,elIdx,1), dtype=np.int32)
            
                tic = time.time()
                single_atomic_distances_dists( atomIndex             = atomIndex, 
                                               distances             = distances,
                                               moleculeIndex         = moleculeIndex,
                                               elementIndex          = elementIndex,
                                               dintra                = dintra,
                                               dinter                = dinter,
                                               nintra                = nintra,
                                               ninter                = ninter,
                                               lowerLimit            = lowerLimit,
                                               upperLimit            = upperLimit,
                                               interMolecular        = True,
                                               intraMolecular        = True,
                                               countWithinLimits     = True,
                                               reduceDistanceToUpper = False,
                                               reduceDistanceToLower = False,
                                               reduceDistance        = False,
                                               allAtoms              = True, 
                                               ncores                = np.int32(n) )
            mtime.append(time.time()-tic)
            print "intermolecular distances: %s atoms, ncores %s --> %.10f "%(SIZE, n, sum(mtime)/len(mtime))
            DATA["intermolecular distances"][SIZE][0].append(n)
            DATA["intermolecular distances"][SIZE][1].append(sum(mtime)/len(mtime)) 
            


##########################################################################################
########################################## PLOT ##########################################            
import matplotlib.pyplot as plt
colormap = plt.cm.jet 
colors = [colormap(i) for i in np.linspace(0, 1,len(NUMBER_OF_ATOMS))]

for kdict, vdict in DATA.items():
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.suptitle(kdict)
    maxs = []
    mins = []
    for i, k in enumerate(vdict.keys()):
        v = vdict[k]
        maxs.append(max(v[1]))
        mins.append(min(v[1]))
        plt.plot(v[0], v[1], marker='o', color=colors[i], label="%s atoms"%k) 
    plt.gca().set_xlabel("number of cores")
    plt.gca().set_ylabel("time (s)")
    plt.gca().legend(frameon=False, ncol=np.ceil(len(NUMBER_OF_ATOMS)/6.).astype(int) )
    plt.gca().set_xlim([0,CORES[-1]+1])
    mins = min(mins)
    maxs = max(maxs)
    plt.gca().set_ylim([mins-0.1*mins, maxs+0.1*maxs])
plt.show()
exit()





