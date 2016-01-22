"""
This is a C++ compiled Cython generated module to calculate atomic coordination number constraints. 
It contains the following methods.

**single_shell_coordination_number**: It computes the neighbours indexes around a certain atom 
  within a lower and upper shell limits.   
    :Arguments:
       #. atomIndex (int32): The atom index to compute the neighbour atom indexes around.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. indexes (int32 array): The indexes of atoms to find neighbours from.
       #. lowerLimit (float32 array): The coordination number shell lower limit or minimum distance.
       #. upperLimit (float32 array): The coordination number shell upper limit or maximum distance.
                                      
    :Returns:
       #. neighbours (python List): List of neighbours indexes.
       
           
**single_atomic_coordination_number**: It computes the coordination number around a certain atom
  in all defined shells.
    :Arguments:
       #. atomIndex (int32): The atom index to compute the neighbour atom indexes around.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. typesIndex (int32 array): The atoms's types array, assigning a type for every atom.
       #. typesDefinition (python dictionary): The coordination number definition per atom type.
       #. typeIndexesLUT (python dictionary): The dictionary of all types grouped in key: list of atoms indexes.
       #. coordNumData (python List): The coordination number data.

    :Returns:
       #. coordNumData (python List): The coordination number data updated.
       
**multiple_atomic_coordination_number**: It computes the coordination number of multiple atoms.
    :Arguments:
       #. indexes (int32): The list of atoms to compute their coordination number.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. typesIndex (int32 array): The atoms's types array, assigning a type for every atom.
       #. typesDefinition (python dictionary): The coordination number definition per atom type.
       #. typeIndexesLUT (python dictionary): The dictionary of all types grouped in key: list of atoms indexes.
       #. coordNumData (python List): The coordination number data.

    :Returns:
       #. coordNumData (python List): The coordination number data updated.

       
**full_atomic_coordination_number**: It computes the coordination number of all system's atoms.
    :Arguments:
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. typesIndex (int32 array): The atoms's types array, assigning a type for every atom.
       #. typesDefinition (python dictionary): The coordination number definition per atom type.
       #. typeIndexesLUT (python dictionary): The dictionary of all types grouped in key: list of atoms indexes.
       #. coordNumData (python List): The coordination number data.

    :Returns:
       #. coordNumData (python List): The coordination number data updated.
 
 
**atom_coordination_number_data**: It cocd mputes all coordination number informations and atoms
  involved in a given atom coordination number calculation.
    :Arguments:
       #. atomIndex (int32): The atom index to compute the neighbour atom indexes around.
       #. boxCoords (float32 array): The whole system box coordinates.
       #. basis (float32 array): The box vectors.
       #. moleculeIndex (int32 array): The molecule's index array, assigning a molecule index for every atom.
       #. typesIndex (int32 array): The atoms's types array, assigning a type for every atom.
       #. typesDefinition (python dictionary): The coordination number definition per atom type.
       
    :Returns:
       #. atoms (python List): The List of atoms involved in a way or another with the coordination number of atomIndex.
       #. shells (python List): The List of shells indexes corresponding to the list of atoms.
       #. neighbours (python List): The List of atoms found neighbouring atoms in corresponding shells.          
"""

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
def single_shell_coordination_number( C_INT32 atomIndex, 
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
    neighbours = [] 
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
    # return data
    return neighbours    


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.always_allow_keywords(False)
def single_atomic_coordination_number( C_INT32 atomIndex, 
                                       ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                       ndarray[C_FLOAT32, ndim=2] basis not None,
                                       ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                       ndarray[C_INT32, ndim=1] typesIndex not None,
                                       dict typesDefinition,
                                       dict typeIndexesLUT,
                                       list coordNumData):    
    cdef C_INT32 shellIdx,
    cdef C_INT32 int32Var
    # get atom molecule and types and atom data
    atomTypeIndex     = typesIndex[atomIndex]
    atomCnDef         = typesDefinition[atomTypeIndex]
    atomShellsType    = atomCnDef['neigTypeIdx']
    atomNeighbours    = coordNumData[atomIndex]['neighbours'] = {}
    atomMinShells     = atomCnDef['lowerLimit']
    atomMaxShells     = atomCnDef['upperLimit']
    # loop over shells
    endIndex = <C_INT32>len(atomShellsType)
    for shellIdx from INT32_ZERO <= shellIdx < endIndex:
        typeIndex = atomShellsType[shellIdx]
        neighbours = single_shell_coordination_number(atomIndex=atomIndex,
                                                      boxCoords=boxCoords,
                                                      basis=basis,
                                                      moleculeIndex=moleculeIndex,
                                                      indexes=typeIndexesLUT[typeIndex],
                                                      lowerLimit=atomMinShells[shellIdx],
                                                      upperLimit=atomMaxShells[shellIdx])                            
        # set atoms in coordination number
        for int32Var in neighbours:
            atomNeighbours[int32Var] = shellIdx
            # update neighbouring
            coordNumData[int32Var]['neighbouring'][atomIndex] = shellIdx
                
    # compute squared deviations
    cdef ndarray[C_INT32,  mode="c", ndim=1] deviations = np.zeros(len(atomCnDef['minNumOfNeig']), dtype=NUMPY_INT32)
    for shellIdx in atomNeighbours.values():
        deviations[shellIdx] += 1
    for shellIdx from INT32_ZERO <= shellIdx < <C_INT32>len(atomCnDef['minNumOfNeig']):
        if deviations[shellIdx] < atomCnDef['minNumOfNeig'][shellIdx]:
            deviations[shellIdx] = deviations[shellIdx]-atomCnDef['minNumOfNeig'][shellIdx]
        elif deviations[shellIdx] > atomCnDef['maxNumOfNeig'][shellIdx]:
            deviations[shellIdx] = deviations[shellIdx]-atomCnDef['maxNumOfNeig'][shellIdx]
        else:
            deviations[shellIdx] = 0
    coordNumData[atomIndex]['deviations'] = deviations
    coordNumData[atomIndex]['squaredDeviations'] = np.sum(coordNumData[atomIndex]['deviations']**2)    
    return coordNumData
   
def multiple_atomic_coordination_number( ndarray[C_INT32, ndim=1] indexes not None,
                                         ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                         ndarray[C_FLOAT32, ndim=2] basis not None,
                                         ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                         ndarray[C_INT32, ndim=1] typesIndex not None,
                                         dict typesDefinition,
                                         dict typeIndexesLUT,
                                         list coordNumData): 
    for i in indexes:
    #for idx in prange(len(indexes)):
        coordNumData = single_atomic_coordination_number( atomIndex = i, 
                                                          boxCoords = boxCoords,
                                                          basis = basis,
                                                          moleculeIndex = moleculeIndex,
                                                          typesIndex = typesIndex,
                                                          typesDefinition = typesDefinition,
                                                          typeIndexesLUT = typeIndexesLUT,
                                                          coordNumData = coordNumData)
    return coordNumData                                                     
                                                          
                                                          
                                                          
def full_atomic_coordination_number( ndarray[C_FLOAT32, ndim=2] boxCoords not None,
                                     ndarray[C_FLOAT32, ndim=2] basis not None,
                                     ndarray[C_INT32, ndim=1] moleculeIndex not None,
                                     ndarray[C_INT32, ndim=1] typesIndex not None,
                                     dict typesDefinition,
                                     dict typeIndexesLUT,
                                     list coordNumData): 
    
    # get number of atoms
    cdef numberOfAtoms = <C_INT32>boxCoords.shape[0]
    # get indexes
    cdef ndarray[C_INT32,  mode="c", ndim=1] indexes = np.arange(numberOfAtoms, dtype=NUMPY_INT32)
    return multiple_atomic_coordination_number(indexes=indexes,
                                               boxCoords = boxCoords,
                                               basis = basis,
                                               moleculeIndex = moleculeIndex,
                                               typesIndex = typesIndex,
                                               typesDefinition = typesDefinition,
                                               typeIndexesLUT=typeIndexesLUT,
                                               coordNumData = coordNumData)
                                                                                    
                                                          

def atom_coordination_number_data( C_INT32 atomIndex, 
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

    
    
    

                                        
                                       