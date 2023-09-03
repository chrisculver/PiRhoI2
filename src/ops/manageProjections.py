import FiniteVolumeGroups as fvg
from src.ops.quarkIsospin import *
from src.ops.projection_utils import *

import sympy as sp
import numpy as np

import time            
            

class TimerError(Exception):
    """A custom exception for timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        
    def start(self,message):
        if self._start_time is not None:
            raise TimerError("Timer is running...")
        self._start_time = time.perf_counter()
        self.message = message
    
    def stop(self):
        if self._start_time is None:
            raise TimerError("Timer is not running...")
        
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(self.message + ": took  {:0.4f} seconds".format(elapsed_time))
            
    
class GenericProjectors:
        
    def rotate_basis(self):
        for b in self.basis:
            op = sp.simplify(b)
            op = self.rotationPrepFunction(op)
            for g in self.group.elements:
                rotatedOp = op.subs(rotation_group_subs(g)).doit()
                self.rotatedBasis.append(rotatedOp)
    
    def create_coefMat(self):
        coefMat = []
        for o in self.rotatedBasis:
            row = []
            polyOp = sp.Poly(o, *self.elementals)
            for elem in self.elementals:
                row.append(polyOp.as_expr().coeff(elem))
            coefMat.append(row)
        return coefMat
        
    def newRepMatrix(self, i):
        dim=len(self.basis)
        self.repDim = dim
        res=np.zeros((dim,dim))
        groupDim = len(self.group.elements)
        for r in range(dim):
            for c in range(dim):
                res[r,c]=np.dot(self.coefMat[r*groupDim+i],self.coefMat[c*groupDim])
        return np.transpose(res)
        
    def projector_matrix(self, irrep, row):
        total = np.zeros([self.repDim,self.repDim],dtype=float)
        for i in range(len(self.group.elements)):
            elem = self.group.elements[i]
            total=total+elem.irreps[irrep][row,row]*np.transpose(self.repMatrices[i])

        return total*float(len(self.group.elements[0].irreps[irrep])/len(self.group.elements))


    def run(self):
        t = Timer()
        if(self.timed):
            t.start("Full rotate basis")
        self.basis = [e.subs(self.symbolToVec) for e in self.basis]
        #print('new basis size {}'.format(len(self.basis)))
        self.rotatedBasis=[]
        self.rotate_basis()
    
        if(self.timed):
            t.stop()
            t.start("Full cleanup")
        #print('new rotated basis has len {}'.format(len(self.rotatedBasis)))
        self.vecToSymbol = momentumSubs(self.rotatedBasis)
        self.symbolToVec = {v: k for k,v in self.vecToSymbol.items()}
            
        self.rotatedBasis = [e.subs(self.vecToSymbol) for e in self.rotatedBasis]
        self.rotatedBasis = [i for i in self.rotatedBasis if i != 0]
        self.basis = [e.subs(self.vecToSymbol) for e in self.basis]
        self.elementals = get_elementals(self.rotatedBasis)
        if(self.timed):
            t.stop()
            t.start("Full coef mat")
        self.coefMat = self.create_coefMat()
        if(self.timed):
            t.stop()
            t.start("rep matrices to ops")
        self.repMatrices = []
        for i in range(len(self.group.elements)):
            self.repMatrices.append(self.newRepMatrix(i))
        self.projectors = {}
        self.ops = 0
        for irrep in self.group.elements[0].irreps.keys():
            projMat = self.projector_matrix(irrep,0)
            self.projectors[irrep] = projMat
            self.ops += len(self.group.elements[0].irreps[irrep])*np.linalg.matrix_rank(np.matrix(projMat,dtype=float))
        
        if(self.ops!=len(self.basis)):
            print('ERROR: wrong number of operators across irreps!!!')
            
        if(self.timed):
            t.stop()
            


class ManageRhoPi(GenericProjectors):
    
    def __init__(self, group, twoMomenta,timed=False):
        self.rhopiSeed = twoMesons(2,2,
                                  MesonData(vectors,spP1,1),
                                  MesonData(pseudoScalars,spP2,1))
        self.timed=timed
        self.group = group
        self.momSeed = twoMomenta
        
        self.rhopiVec = sp.Matrix([self.rhopiSeed.subs(self.rhopiDirSubs(pol)) for pol in ['x','y','z']])
        self.basis = []
        self.rotatedBasis = []
        self.rotate_basis()
        
        self.vecToSymbol = momentumSubs(self.rotatedBasis)
        self.symbolToVec = {v: k for k,v in self.vecToSymbol.items()}
        
        self.rotatedBasis = [e.subs(self.vecToSymbol) for e in self.rotatedBasis]
        self.rotatedBasis = [i for i in self.rotatedBasis if i != 0]
        self.basis = [e.subs(self.vecToSymbol) for e in self.basis]
        self.elementals = get_elementals(self.rotatedBasis)

        
        self.coefMat = self.create_coefMat()
        #print('first coef matrix with tiny basis made')
        self.basis = np.matmul(
                                np.matrix(sp.Matrix(self.coefMat).rref()[0],dtype=float),
                               self.elementals
                                ).tolist()[0]
        self.basis = [i for i in self.basis if i!=0]
        self.basis = [e.subs(self.symbolToVec) for e in self.basis]
        #print('new basis size {}'.format(len(self.basis)))
        self.rotatedBasis=[]
        self.rotate_basis2()
        #print('new rotated basis has len {}'.format(len(self.rotatedBasis)))
        self.vecToSymbol = momentumSubs(self.rotatedBasis)
        self.symbolToVec = {v: k for k,v in self.vecToSymbol.items()}
        
        self.rotatedBasis = [e.subs(self.vecToSymbol) for e in self.rotatedBasis]
        self.rotatedBasis = [i for i in self.rotatedBasis if i != 0]
        self.basis = [e.subs(self.vecToSymbol) for e in self.basis]
        self.elementals = get_elementals(self.rotatedBasis)
        #print("basis=",self.basis)
        #print("rotBasis=",self.rotatedBasis)
        #print("elementals=",self.elementals)
        self.coefMat = self.create_coefMat()
        
        self.repMatrices = []
        for i in range(len(self.group.elements)):
            self.repMatrices.append(self.newRepMatrix(i))
        
        self.projectors = {}
        self.ops = 0
        for irrep in group.elements[0].irreps.keys():
            projMat = self.projector_matrix(irrep,0)
            self.projectors[irrep] = projMat
            self.ops += len(group.elements[0].irreps[irrep])*np.linalg.matrix_rank(np.matrix(projMat,dtype=float))
        
        if(self.ops!=len(self.basis)):
            print('ERROR: wrong number of operators across irreps!!!')
        
        
        
        
    def rhopiDirSubs(self,pol): 
        return {sp.Function('rho_i^'+c): sp.Function('rho_'+pol+'^'+c) 
                for c in ['-','0','+']}
    
    def rotate_basis(self):
        for i,label in enumerate(['x','y','z']):
            seed = self.rhopiSeed.subs(self.rhopiDirSubs(label)).doit()
            self.basis.append(seed.subs(twomomenta_subs(self.momSeed)))
            
            for g in self.group.elements:
                op = self.rhopiVec
                op = self.prep_rhopi_rotation1(op)
                rotatedOp1 = (op.subs(rotation_group_subs(g)).doit())[i]
                rotatedOp = self.prep_rhopi_rotation2(rotatedOp1).subs(twomomenta_subs(self.momSeed))
                rotatedOp = rotatedOp.subs(rotation_group_subs(g)).doit()
                self.rotatedBasis.append(rotatedOp)
                
    def rotate_basis2(self):
        for b in self.basis:
            #print('rotating basis element b={}'.format(b))
            for g in self.group.elements:
                op = self.prep_rhopi_rotation3(b)
                #print('prepped op={}'.format(op))
                rotatedOp = op.subs(rotation_group_subs(g)).doit()
                #print('rotated op={}'.format(rotatedOp))
                rotatedOp = self.rotatePolarizedRho(sp.expand(rotatedOp),g)
                #print("rotated pol rho={}".format(rotatedOp))
                self.rotatedBasis.append(rotatedOp)
                
    def prep_rhopi_rotation1(self, expr):
        return detR*vecRotation*expr
    
    def prep_rhopi_rotation2(self,expr):
        args=unique_arguments(expr)
        momentaRotations = {p: vecRotation*p for p in args}
        return expr.subs(momentaRotations)
        
    def prep_rhopi_rotation3(self,expr):
        args=unique_arguments(expr)
        momentaRotations = {p: vecRotation*p for p in args}
        return detR*expr.subs(momentaRotations)
        

            
    def rotatePolarizedRho(self,expr,g):
        #print("rotatingPolarizedRho")
        #print(expr)
        #print(g.rotation)
        directions = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
        res = 0.0
        if isinstance(expr, sp.core.mul.Mul):
            newT=1.0
            for elem in expr.args:
                newRhoString = ''
                oldRho = ''
                if 'rho' in str(elem):
                    #print('elem={}'.format(elem))
                    signFlip = 1
                    oldRho=str(elem).split('(')[0]
                    
                    elemStr = str(elem).split('(')[0]
                    startString = str(elemStr).split('_')[0]+'_'
                    pol = str(elemStr).split('_')[1][0]
                    endString = '^'+str(elemStr).split('^')[1]
        
                    newPol = np.matmul(g.rotation,directions[pol])
        
                    signFlip = False if (newPol==abs(newPol)).all() else True
                    newDir = ''
                    if(abs(newPol)==[1, 0, 0]).all():
                        newDir='x'
                    elif(abs(newPol)==[0, 1, 0]).all():
                        newDir='y'
                    elif(abs(newPol)==[0, 0, 1]).all():
                        newDir='z'
                    else:
                        newDir='?'
                        print('rho rotated weird')
                    newRhoString = startString + newDir + endString
    
                    elem=elem.subs({sp.Function(oldRho): sp.Function(newRhoString)}).doit()
                    if(signFlip):
                        newT *= -1.0
        
                newT *= elem
            #print("newT=",newT)
            return newT 
                
        
        elif isinstance(expr, sp.core.add.Add):
            for t in expr.args:
                newT = 1.0
                #print('t={}'.format(t))
                for elem in t.args:
                    newRhoString = ''
                    oldRho = ''
                    if 'rho' in str(elem):
                        #print('elem={}'.format(elem))
                        signFlip = 1
                        oldRho=str(elem).split('(')[0]
                        
                        elemStr = str(elem).split('(')[0]
                        startString = str(elemStr).split('_')[0]+'_'
                        pol = str(elemStr).split('_')[1][0]
                        endString = '^'+str(elemStr).split('^')[1]
            
                        newPol = np.matmul(g.rotation,directions[pol])
            
                        signFlip = False if (newPol==abs(newPol)).all() else True
                        newDir = ''
                        if(abs(newPol)==[1, 0, 0]).all():
                            newDir='x'
                        elif(abs(newPol)==[0, 1, 0]).all():
                            newDir='y'
                        elif(abs(newPol)==[0, 0, 1]).all():
                            newDir='z'
                        else:
                            newDir='?'
                            print('rho rotated weird')
                        newRhoString = startString + newDir + endString
        
                        elem=elem.subs({sp.Function(oldRho): sp.Function(newRhoString)}).doit()
                        if(signFlip):
                            newT *= -1.0
        
                    newT *= elem
        
                res += newT
            return res    
        else:
            raise ValueError("")