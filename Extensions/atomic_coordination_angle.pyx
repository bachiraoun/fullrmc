from libc.math cimport sqrt, abs
from libcpp.vector cimport vector
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from cython.parallel import prange

# declare types
NUMPY_FLOAT32 = np.float32
NUMPY_INT32   = np.int32
ctypedef np.float32_t C_FLOAT32
ctypedef np.int32_t   C_INT32

# declare constants
cdef C_FLOAT32 FLOAT32_ZERO    = 0.0
cdef C_FLOAT32 BOX_LENGTH      = 1.0
cdef C_FLOAT32 HALF_BOX_LENGTH = 0.5
cdef C_FLOAT32 FLOAT32_ONE     = 1.0
cdef C_INT32   INT32_ONE       = 1
cdef C_INT32   INT32_ZERO      = 0


cdef extern from "math.h":
    C_FLOAT32 floor(C_FLOAT32 x)
    C_FLOAT32 ceil(C_FLOAT32 x)
    C_FLOAT32 sqrt(C_FLOAT32 x)

cdef inline C_FLOAT32 round(C_FLOAT32 num):
    return floor(num + HALF_BOX_LENGTH) if (num > FLOAT32_ZERO) else ceil(num - HALF_BOX_LENGTH)
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_shell_coordination_atoms( C_INT32 atomIndex, 
                                     ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                     ndarray[C_FLOAT32, ndim=2] basis not None,
                                     ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                     ndarray[C_INT32, ndim=1] indexes not None,
                                     C_FLOAT32 lowerLimit,
                                     C_FLOAT32 upperLimit):         
    # declare variables
    cdef C_INT32 loopIdx, startIndex, endIndex
    cdef C_INT32 int32Var
    cdef C_FLOAT32 float32Var
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz, distance
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # cast arguments
    lowerLimit = <C_FLOAT32>lowerLimit
    upperLimit = <C_FLOAT32>upperLimit
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # get atom molecule and types and atom data
    atomMoleculeIndex = moleculeIndex[atomIndex]
    # get atom data
    neighbours  = [] 
    neighCoords = [] 
    # loop
    for int32Var from INT32_ZERO <= int32Var < <C_INT32>len(indexes):
        loopIdx = indexes[int32Var]
        # if same atom skip
        if loopIdx == atomIndex: continue
        # if intra skip
        if moleculeIndex[loopIdx] == atomMoleculeIndex: continue
        # calculate difference
        box_dx = boxCoords[loopIdx,0]-atomBox_x
        box_dy = boxCoords[loopIdx,1]-atomBox_y
        box_dz = boxCoords[loopIdx,2]-atomBox_z
        box_dx -= round(box_dx)
        box_dy -= round(box_dy)
        box_dz -= round(box_dz)
        # get real difference
        real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]
        # calculate distance         
        distance = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
        # check limits
        if distance<lowerLimit:
            continue
        if distance>upperLimit:
            continue
        # append atom to shell
        neighbours.append(loopIdx)
        neighCoords.append( [real_dx,real_dy,real_dz] )
    # return data
    return neighbours, neighCoords


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atomic_coordination_angle( C_INT32 atomIndex, 
                                      ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                      ndarray[C_FLOAT32, ndim=2] basis not None,
                                      ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                      ndarray[C_INT32, ndim=1] typesIndex not None,
                                      dict typesDefinition,
                                      dict typeIndexesLUT,
                                      list coordAngData):    
    cdef C_INT32 shellIdx,
    cdef C_INT32 int32Var
    # get atom molecule and types and atom data
    atomTypeIndex    = typesIndex[atomIndex]
    atomCnDef        = typesDefinition[atomTypeIndex]
    firstNeigType    = atomCnDef['firstTypeIdx']
    firstNeigLower   = atomCnDef['firstLowerLimit']
    firstNeigUpper   = atomCnDef['firstUpperLimit']
    secondNeigType   = atomCnDef['secondTypeIdx']
    secondNeigLower  = atomCnDef['secondLowerLimit']
    secondNeigUpper  = atomCnDef['secondUpperLimit']
    lowerAngles      = atomCnDef['angleLowerLimit']
    upperAngles      = atomCnDef['angleUpperLimit']
    # reset coordination angle data for atom atomIndex
    atomFirstNeighs  = coordAngData[atomIndex]['firstNeighs']  = {}
    atomSecondNeighs = coordAngData[atomIndex]['secondNeighs'] = {}
    # loop over shells
    print atomCnDef, atomTypeIndex
    endIndex = <C_INT32>len(firstNeigType)
    for shellIdx from INT32_ZERO <= shellIdx < endIndex:
        typeIndex = firstNeigType[shellIdx]
        print firstNeigLower[shellIdx], firstNeigUpper[shellIdx]
        firstNeighs, firstCoords = single_shell_coordination_atoms(atomIndex=atomIndex,
                                                                   boxCoords=boxCoords,
                                                                   basis=basis,
                                                                   moleculeIndex=moleculeIndex,
                                                                   indexes=typeIndexesLUT[typeIndex],
                                                                   lowerLimit=firstNeigLower[shellIdx],
                                                                   upperLimit=firstNeigUpper[shellIdx]) 
        typeIndex = secondNeigType[shellIdx]
        print secondNeigLower[shellIdx], secondNeigLower[shellIdx]
        secondNeighs, secondCoords = single_shell_coordination_atoms(atomIndex=atomIndex,
                                                                     boxCoords=boxCoords,
                                                                     basis=basis,
                                                                     moleculeIndex=moleculeIndex,
                                                                     indexes=typeIndexesLUT[typeIndex],
                                                                     lowerLimit=secondNeigLower[shellIdx],
                                                                     upperLimit=secondNeigUpper[shellIdx])
        # set first atoms in coordination number
        atomFirstNeighs[shellIdx] = firstNeighs
        for int32Var in firstNeighs:
            #atomFirstNeighs[int32Var] = shellIdx
            coordAngData[int32Var]['neighbouring'][atomIndex] = shellIdx
        # set second atoms in coordination number
        atomSecondNeighs[shellIdx] = secondNeighs
        for int32Var in secondNeighs:
            #atomSecondNeighs[int32Var] = shellIdx
            coordAngData[int32Var]['neighbouring'][atomIndex] = shellIdx 

    print #coordAngData[atomIndex]
    exit()           
    ## compute squared deviations
    #cdef ndarray[C_INT32,  mode="c", ndim=1] deviations = np.zeros(len(atomCnDef['minNumOfNeig']), dtype=NUMPY_INT32)
    #for shellIdx in atomNeighbours.values():
    #    deviations[shellIdx] += 1
    #for shellIdx from INT32_ZERO <= shellIdx < <C_INT32>len(atomCnDef['minNumOfNeig']):
    #    if deviations[shellIdx] < atomCnDef['minNumOfNeig'][shellIdx]:
    #        deviations[shellIdx] = deviations[shellIdx]-atomCnDef['minNumOfNeig'][shellIdx]
    #    elif deviations[shellIdx] > atomCnDef['maxNumOfNeig'][shellIdx]:
    #        deviations[shellIdx] = deviations[shellIdx]-atomCnDef['maxNumOfNeig'][shellIdx]
    #    else:
    #        deviations[shellIdx] = 0
    #coordAngData[atomIndex]['deviations'] = deviations
    #coordAngData[atomIndex]['squaredDeviations'] = np.sum(coordAngData[atomIndex]['deviations']**2)    
    #return coordAngData
   
def multiple_atomic_coordination_angle( ndarray[C_INT32, ndim=1] indexes not None,
                                        ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                        ndarray[C_FLOAT32, ndim=2] basis not None,
                                        ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                        ndarray[C_INT32, ndim=1] typesIndex not None,
                                        dict typesDefinition,
                                        dict typeIndexesLUT,
                                        list coordAngData): 
    for i in indexes:
    #for idx in prange(len(indexes)):
        coordAngData = single_atomic_coordination_angle( atomIndex = i, 
                                                         boxCoords = boxCoords,
                                                         basis = basis,
                                                         moleculeIndex = moleculeIndex,
                                                         typesIndex = typesIndex,
                                                         typesDefinition = typesDefinition,
                                                         typeIndexesLUT = typeIndexesLUT,
                                                         coordAngData = coordAngData)
    return coordAngData                                                     
                                                          
                                                          
                                                          
def full_atomic_coordination_angle( ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                    ndarray[C_FLOAT32, ndim=2] basis not None,
                                    ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                    ndarray[C_INT32, ndim=1] typesIndex not None,
                                    dict typesDefinition,
                                    dict typeIndexesLUT,
                                    list coordAngData): 
    
    # get number of atoms
    cdef numberOfAtoms = <C_INT32>boxCoords.shape[0]
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(numberOfAtoms, dtype=NUMPY_INT32)
    return multiple_atomic_coordination_angle(indexes=indexes,
                                              boxCoords = boxCoords,
                                              basis = basis,
                                              moleculeIndex = moleculeIndex,
                                              typesIndex = typesIndex,
                                              typesDefinition = typesDefinition,
                                              typeIndexesLUT=typeIndexesLUT,
                                              coordAngData = coordAngData)
                                                                                    
                                                          

def atom_coordination_angle_data( C_INT32 atomIndex, 
                                  ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                  ndarray[C_FLOAT32, ndim=2] basis not None,
                                  ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                  ndarray[C_INT32, ndim=1] typesIndex not None,
                                  dict typesDefinition):
    # declare variables
    cdef C_INT32 loopIdx, startIndex, endIndex
    cdef C_INT32 shellIdx
    cdef C_INT32 int32Var
    cdef C_FLOAT32 float32Var
    cdef C_FLOAT32 box_dx, box_dy, box_dz
    cdef C_FLOAT32 real_dx, real_dy, real_dz, distance
    cdef C_FLOAT32 atomBox_x, atomBox_y, atomBox_z
    # get point coordinates
    atomBox_x = boxCoords[atomIndex,0]
    atomBox_y = boxCoords[atomIndex,1]
    atomBox_z = boxCoords[atomIndex,2]
    # get atom molecule and types and atom data
    atomMoleculeIndex = moleculeIndex[atomIndex]
    thisAtomTypeIndex = typesIndex[atomIndex]
    # initiate lists
    cdef vector[C_INT32] neighboursOfAtom, neighboursShellIdx, neighbours
    # if atom has no coordination neighbours definition simply return
    if not typesDefinition.has_key(thisAtomTypeIndex): 
        return neighboursOfAtom, neighboursShellIdx, neighbours
    # get this atom properties
    thisAtomCnDef      = typesDefinition[thisAtomTypeIndex]
    thisAtomShellsType = thisAtomCnDef['neigTypeIdx']
    thisAtomMinShells  = thisAtomCnDef['lowerLimit']
    thisAtomMaxShells  = thisAtomCnDef['upperLimit']
    # loop
    for loopIdx from INT32_ZERO <= loopIdx < <C_INT32>boxCoords.shape[0]:
        # if same atom skip
        if loopIdx == atomIndex: continue
        # if intra skip
        if moleculeIndex[loopIdx] == atomMoleculeIndex: continue
        # get loop atom properties
        loopAtomTypeIndex  = typesIndex[loopIdx]
        loopAtomCnDef      = typesDefinition[loopAtomTypeIndex]
        loopAtomShellsType = loopAtomCnDef['neigTypeIdx']
        loopAtomMinShells  = loopAtomCnDef['lowerLimit']
        loopAtomMaxShells  = loopAtomCnDef['upperLimit']
        # check if neighbour type is in definition
        computeThisAtom = loopAtomTypeIndex in thisAtomShellsType
        computeLoopAtom = thisAtomTypeIndex in loopAtomShellsType
        if not computeThisAtom and not computeLoopAtom: continue
        # calculate distance
        box_dx = (boxCoords[loopIdx,0]-atomBox_x)%1
        box_dy = (boxCoords[loopIdx,1]-atomBox_y)%1
        box_dz = (boxCoords[loopIdx,2]-atomBox_z)%1
        box_dx = abs(box_dx)
        box_dy = abs(box_dy)
        box_dz = abs(box_dz)
        if box_dx > HALF_BOX_LENGTH: box_dx = BOX_LENGTH-box_dx
        if box_dy > HALF_BOX_LENGTH: box_dy = BOX_LENGTH-box_dy
        if box_dz > HALF_BOX_LENGTH: box_dz = BOX_LENGTH-box_dz
        real_dx = box_dx*basis[0,0] + box_dy*basis[1,0] + box_dz*basis[2,0]
        real_dy = box_dx*basis[0,1] + box_dy*basis[1,1] + box_dz*basis[2,1]
        real_dz = box_dx*basis[0,2] + box_dy*basis[1,2] + box_dz*basis[2,2]        
        distance = <C_FLOAT32>sqrt(real_dx*real_dx + real_dy*real_dy + real_dz*real_dz)
        # find shells of this atoms
        if computeThisAtom:
            for shellIdx from <C_INT32>np.searchsorted(thisAtomMinShells, distance) > shellIdx >= INT32_ZERO:
                # check if within shell limits
                if distance>thisAtomMaxShells[shellIdx]:
                    continue
                # check shell type
                if loopAtomTypeIndex != thisAtomShellsType[shellIdx]:
                    continue
                # atom must be counted
                neighboursOfAtom.push_back(atomIndex)
                neighboursShellIdx.push_back(shellIdx)
                neighbours.push_back(loopIdx)
                
        # find shells of loop atoms
        if computeLoopAtom:
            for shellIdx from <C_INT32>np.searchsorted(loopAtomMinShells, distance) > shellIdx >= INT32_ZERO:
                # check if within shell limits
                if distance>loopAtomMaxShells[shellIdx]:
                    continue
                # check shell type
                if thisAtomTypeIndex != loopAtomShellsType[shellIdx]:
                    continue
                # atom must be counted
                neighboursOfAtom.push_back(loopIdx)
                neighboursShellIdx.push_back(shellIdx)
                neighbours.push_back(atomIndex)
    
    return neighboursOfAtom, neighboursShellIdx, neighbours

    
    
    

                                        
                                       