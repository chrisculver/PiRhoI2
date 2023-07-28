import numpy as np
import math
import cmath
import copy
import os
import h5py

def parse_corr_file(fileBase, i, j, NT):
    values=[]

    corrFile = os.path.join(fileBase,"corr_op.{}_op.{}.dat".format(i,j))
    conjugateCorr = False
    if not os.path.exists(corrFile):
        corrFile = os.path.join(fileBase,"corr_op.{}_op.{}.dat".format(j,i))
        if not os.path.exists(corrFile):
            raise RuntimeError("Couldn't find any file to fill data for corr i={},j={}".format(i,j))
        conjugateCorr=True    

    with open(corrFile, 'r') as data:
        
        for line in data:
            cols=line.split(' ')
            values.append(float(cols[0])+float(cols[1])*1j)
    
    NC=len(values)//NT 
    if len(values) % NT != 0:
        raise RuntimeError("Number of values not divisible by NT, NT is wrong!")

    values = np.reshape(values, (NC,NT))
    if conjugateCorr:
        values=np.conjugate(values)

    return values


def log_effective_mass(corr):
    avg=np.mean(corr, axis=0)
    res=[]

    for t in range(len(avg)-1):
        try:
            res.append(math.log(avg[t]/avg[t+1]))
        except ValueError:
            res.append(-1j*math.pi + math.log(-avg[t]/avg[t+1]))
        except: 
            raise
    
    return res

def normalize_corr_matrix(corrMatrix, t0):
    res = np.zeros(corrMatrix.shape,dtype=complex)

    for cfg in range(len(corrMatrix)):
        for t in range(len(corrMatrix[0])):
            for o1 in range(len(corrMatrix[0,0])):
                for o2 in range(len(corrMatrix[0,0,0])):
                    res[cfg,t,o1,o2]=corrMatrix[cfg,t,o1,o2]/cmath.sqrt(corrMatrix[cfg,t0,o1,o1]*corrMatrix[cfg,t0,o2,o2])


    return res


def is_hermitian(corrMatrix, atol=1e-8):
    hermConj = np.array([[mat.conjugate().transpose() for mat in timeSlice] for timeSlice in corrMatrix])
    
    hermitian = np.allclose(corrMatrix, hermConj)

    if not hermitian:
        print("WARNING: Non hermitian correlator matrix to atol={}...".format(atol))
        print("         Checking ensemble average, c[t,i,j]")
        avgCorrMatrix=np.mean(corrMatrix,axis=0)
        avgHermConj=np.mean(hermConj,axis=0)

        for t in range(len(avgCorrMatrix)):
            close = np.isclose(avgCorrMatrix[t],avgHermConj[t],atol=atol)
            for i in range(len(avgCorrMatrix[0])):
                for j in range(len(avgCorrMatrix[0,0])):
                    if close[i,j]==False:
                        print("c[{:d},{:d},{:d}]={:.4e}, c^d[{:d},{:d},{:d}]={:.4e}".format(
                            t,i,j,avgCorrMatrix[t,i,j],
                            t,i,j,avgHermConj[t,i,j]))

    return hermitian

    


def non_hermitian_configs(corrMatrix):
    for cfg in range(len(corrMatrix)):
        ts=[]
        for t in range(len(corrMatrix[0])):
            mat=corrMatrix[cfg,t]
            hcMat=mat.conjugate().transpose()
            if not np.allclose(mat,hcMat):
                ts.append(t)
        if ts!=[]:
            print("cfg #{} for ts={} non-hermitian to 1e-8".format(cfg,ts))



def non_hermitian_all_ts(corrMatrix):
    cfgs=[]
    for cfg in range(len(corrMatrix)):
        ts=[]
        for t in range(len(corrMatrix[0])):
            mat=corrMatrix[cfg,t]
            hcMat=mat.conjugate().transpose()
            if not np.allclose(mat,hcMat):
                ts.append(t)
        if ts==[t for t in range(len(corrMatrix[0]))]:
            print("cfg #{} non-hermitian to 1e-8".format(cfg,ts))
            cfgs.append(cfg)
    return cfgs

def non_hermitian_tolerance(mat):
    for atol in [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]:
        if np.allclose(mat,mat.conjugate().transpose(),atol=atol):
            print("Reached tolerance at {:.2e}".format(atol))
            return None

    print("No tolerance reached down to 1e-1")


def x_within_sigma_of_y(x,xerr,y):
    if (x-xerr)<y<(x+xerr):
        return True
    return False

def check_offdiagonal_elems(matrix, err):
    nonZeroReal=0
    nonZeroIm=0
    N=1.0*len(matrix)*len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if not x_within_sigma_of_y(matrix[i,j].real,np.sqrt(err[i,j].real),0):
                nonZeroReal+=1
            if not x_within_sigma_of_y(matrix[i,j].imag,np.sqrt(err[i,j].imag),0):
                nonZeroIm+=1
    
    return nonZeroReal*1./N, nonZeroIm*1./N



def niFile_to_dict(fileName,irrep):
    opFile = h5py.File(fileName, 'r')
    opTypes=opFile[irrep].keys()
    opData={}
    for op in opTypes:
        opData[op]=[]
        for elem in opFile[irrep][op]:
            opData[op].append({'en': elem[0], 'mult': elem[1]})
    return opData