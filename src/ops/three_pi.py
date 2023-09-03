from src.ops.quarkIsospin import *
from src.ops.projection_utils import *
from src.ops.manageProjections import Timer

import numpy as np
import sympy as sp
import copy
import math
import itertools


def rref(mat):
    return np.array(sp.Matrix(mat).rref()[0],dtype=float)


class SimplePi:
    def __init__(self,charge,mom):
        self.charge=charge
        if not isinstance(mom,np.ndarray):
            raise TypeError('Only support numpy at the moment')
        self.mom=mom
        
    def to_sympy(self):
        momStr=str(int(self.mom[0]))+str(int(self.mom[1]))+str(int(self.mom[2]))
        momSP=sp.Symbol(momStr.replace('-','m'))
        return sp.Function('pi^'+self.charge)(momSP)
    
    def spatial_rotate(self,Rg):
        rotated=copy.deepcopy(self)
        rotated.mom=np.matmul(Rg,rotated.mom)
        return rotated
        
    def __str__(self):
        return "pi^"+self.charge+"("+str(int(self.mom[0]))+','+str(int(self.mom[1]))+','+str(int(self.mom[2]))+")"
        
    def __eq__(self,other):
        return (self.charge==other.charge) and (self.mom[0]==other.mom[0] and self.mom[1]==other.mom[1] and self.mom[2]==other.mom[2])
        
class ThreePiElemental:
    def __init__(self,coef,pions):
        if len(pions)!=3:
            raise ValueError('Need three pions')
        self.coef=coef
        self.pions=pions
        
    def to_sympy(self):
        expr=self.coef
        for p in self.pions:
            expr*=p.to_sympy()
        return expr
    
    def spatial_rotate(self,g):
        rotated=copy.deepcopy(self)
        rotated.coef=g.identifier['parity']*rotated.coef
        for i,p in enumerate(rotated.pions):
            rotated.pions[i]=p.spatial_rotate(g.rotation)
        return rotated
        
    def __str__(self):
        s=str(self.coef)+"*"
        for e in self.pions[:-1]:
            s+=str(e)+"*"
        s+=str(self.pions[-1])
        return s
    
    def same_pions_as(self,other):
        perms=[[0,1,2],
               [0,2,1],
               [1,0,2],
               [1,2,0],
               [2,0,1],
               [2,1,0]]
        same=False
        for p in perms:
            if self.pions[0]==other.pions[p[0]] and self.pions[1]==other.pions[p[1]] and self.pions[2]==other.pions[p[2]]:
                same=True
                break
        return same
        
    def __rmul__(self,other):
        if isinstance(other,float):
            res=copy.deepcopy(self)
            res.coef*=other
            return res
        
    def __add__(self,other):
        if isinstance(other,ThreePiElemental):
            if math.isclose(self.coef,0.) and math.isclose(other.coef,0.):
                return 0.
            elif math.isclose(other.coef,0.):
                return ThreePiOp([self])
            elif math.isclose(self.coef,0.):
                return ThreePiOp([other])
            else:
                return ThreePiOp([self,other])
            
        if isinstance(other,float):
            if not math.isclose(other,0.):
                raise ValueError('Adding constant value of {} to operator!?!'.format(other))
            elif math.isclose(self.coef,0.):
                return 0.
            else:
                return ThreePiOp([self])
        
    __radd__ = __add__
    
    def __eq__(self,other):
        return (self.coef==other.coef) and self.same_pions_as(other)

class ThreePiOp:
    def __init__(self,elementals):
        self.elementals=elementals
        
    def to_sympy(self):
        expr=0
        for e in self.elementals:
            expr+=e.to_sympy()
        return expr
    
    def spatial_rotate(self,g):
        rotated=copy.deepcopy(self)
        for i,e in enumerate(rotated.elementals):
            rotated.elementals[i]=e.spatial_rotate(g)
        return rotated
            
    def coefficients(self,basis):
        row=[]
        for e in basis:
            found=False
            for e2 in self.elementals:
                if e.same_pions_as(e2):
                    row.append(e2.coef)
                    found=True
                    break
            if not found:
                row.append(0)
        return row
        
    def __str__(self):
        if len(self.elementals)==0:
            return "0"
            
        s=""
        for e in self.elementals[:-1]:
            s+=str(e)+"+"
        s+=str(self.elementals[-1])
        return s
    
    def __add__(self,other):
        res=copy.deepcopy(self)
        if isinstance(other,ThreePiElemental):
            if math.isclose(other.coef,0.):
                return self
            
            found=False
            for i,e in enumerate(res.elementals):
                if e.same_pions_as(other):
                    res.elementals[i].coef+=other.coef
                    found=True
                    break
            if not found:
                res.elementals.append(other)
            if found:
                res.elementals = [e for e in res.elementals if e.coef!=0]
                        
            return res
        
        elif isinstance(other,ThreePiOp):
            for e in other.elementals:
                res=res+e
                
            return res
        
        elif isinstance(other,float):
            if math.isclose(other,0.):
                return self
            else:
                raise ValueError('Trying to add {} to op'.format(other))
        
        else:
            raise TypeError('Unknown add for ThreePiOp w/ {}'.format(type(other)))
    
    __radd__ = __add__
    
    def __sub__(self, other):
        return self+(-1.0*other)
    
    
    def __rmul__(self,other):
        if isinstance(other,float):
            if math.isclose(other,0.):
                return 0.
            res=copy.deepcopy(self)
            for i in range(len(res.elementals)):
                res.elementals[i].coef*=other
            return res
        else:
            raise TypeError('Unknown mul for ThreePiOp w/ {}'.format(type(other)))
    
    __mul__=__rmul__
    
    def __eq__(self,other):
        #TODO: Also needs to permute all pions in other.elementals...
        if isinstance(other,ThreePiOp):
            difference=self.to_sympy()-other.to_sympy()
            if difference==0:
                return True
            else:
                return False

        if isinstance(other,ThreePiElemental):
            t = ThreePiOp([other])
            return self==t
            
        if isinstance(other,int) and other==0:
            return self.elementals==[]
        
        else:
            raise TypeError('The other thing isnt a three-pi op its a {}'.format(type(other)))
    
    
    def same_elementals(self,other):
        if isinstance(other,ThreePiOp):
            same=True
            for e in self.elementals:
                found=False
                for ep in other.elementals:
                    if e.same_pions_as(ep):
                        found=True
                        break
                if not found:
                    same=False
                
            return same
        
        else:
            raise TypeError('The other is not a three-pi-op...')
    
    
def found_in(elementals, perms):
    for p in perms:
        if elementals==p:
            return True
    return False


def all_elementals(basis):
    elems=[]
    for op in basis:
        for e in op.elementals:
            contains=False
            for e2 in elems:
                if e.same_pions_as(e2):
                    contains=True
                    break
            if not contains:
                elems.append(copy.deepcopy(e))
    
    for i in range(len(elems)):
        elems[i].coef=1.
    
    return elems

def coefficient_matrix(basis, all_elementals):
    coefMat=[]
    for b in basis:
        row=[]
        for e in all_elementals:
            found=False
            for e2 in b.elementals:
                if e.same_pions_as(e2):
                    row.append(e2.coef/e.coef)
                    found=True
                    break
            if not found:
                row.append(0)
        coefMat.append(row)
    return coefMat




    
class ThreePiFixedSeed:
    def __init__(self,group,momenta,timed=False):
        self.group=group
        self.momenta=momenta
        self.timed=timed
        
        #Using the seeds, find a list of independent basis vectors.
        t=Timer()
        if(self.timed):
            t.start("Rotate isospin seeds to create basis")
            
        self.generate_first_basis()
        rb1=self.rotate_basis()
        self.elementals=all_elementals(rb1)
        cm1=coefficient_matrix(rb1, self.elementals)
        rrefCM=np.array(sp.Matrix(cm1).rref()[0],dtype=float)
        
        self.basis = np.matmul(rrefCM, self.elementals)
        self.basis = [b for b in self.basis if b!=0]
        
        self.vecToSymbol = momentumSubs(self.basis)

        if(self.timed):
            t.stop()
    
    def run(self):
        t=Timer()
        if(self.timed):
            t.start("Finishing calculation")
        #Now that I've found my basis vectors, do the group theory stuff.
        
        #self.elementals=all_elementals(self.rotatedBasis)
        self.coefMat=self.generate_coefficient_matrix()
        self.repMatrices=self.generate_rep_matrices()
        self.projectors=self.generate_projector_matrices()
        
        numOps=0
        for irrep in self.group.elements[0].irreps.keys():
            numOps+=len(self.group.elements[0].irreps[irrep])*np.linalg.matrix_rank(np.matrix(self.projectors[irrep],dtype=float))
        if(numOps!=len(self.basis)):
            raise ValueError('Num ops!=basis size')
        if(self.timed):
            t.stop()
        

        
    def generate_projector_matrices(self):
        pMats={}
        for irrep in self.group.elements[0].irreps.keys():
            pMats[irrep] = np.array(self.projector_matrix(irrep,0),dtype=float)
        return pMats
        
    def generate_rep_matrices(self):
        repMats=[]
        for i in range(len(self.group.elements)):
            repMats.append(self.newRepMatrix(i))
        return repMats
        
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
        
        
    def rotate_basis(self):
        rotated=[]
        for i in self.basis:
            for g in self.group.elements:
                rotated.append(i.spatial_rotate(g))
        return rotated

    
    def generate_coefficient_matrix(self):
        rotatedBasis = self.rotate_basis()
        coefMat=[]
        for bRot in rotatedBasis:
            mat=[]
            for b in self.basis:
                mat.append(b.coefficients(self.elementals))
            mat.append(bRot.coefficients(self.elementals))
            coefMat.append(np.transpose(np.array(sp.Matrix(np.transpose(mat)).rref()[0],dtype=float).tolist())[len(self.basis),0:len(self.basis)])
        return coefMat   







        
    def generate_first_basis(self):
        p1=self.momenta[0]
        p2=self.momenta[1]
        p3=self.momenta[2]
        self.basis=[]
    
        threepiSeeds = [threeMesons(2,2,iso,
                                        MesonData(pseudoScalars,spP1,1),
                                        MesonData(pseudoScalars,spP2,1),
                                        MesonData(pseudoScalars,spP3,1)) 
                             for iso in [1,2]]

        for i,op in enumerate(threepiSeeds):
            if(i==0):
                threepiSeeds[i]=sp.expand(sp.simplify(op*2/sp.sqrt(2)))
            if(i==1):
                threepiSeeds[i]=sp.expand(sp.simplify(6*op/sp.sqrt(6)))
        
        
        for seed in threepiSeeds:
            elementals=0.
            for e in sp.expand(seed).args:
                if len(e.args)==3:
                    coef=1
                    pi1=e.args[0]
                    pi2=e.args[1]
                    pi3=e.args[2]
                else:
                    coef=e.args[0]
                    pi1=e.args[1]
                    pi2=e.args[2]
                    pi3=e.args[3]

                elemental = ThreePiElemental(coef,
                              [SimplePi(pi1.name.split('^')[1],p1),
                              SimplePi(pi2.name.split('^')[1],p2),
                              SimplePi(pi3.name.split('^')[1],p3)],
                              )
                elementals=elementals+elemental

            self.basis.append(elementals)
            
    
    def is_equivalent_basis(self,otherBasis):
        # make a dummy object
        tp=ThreePiFixedSeed(self.group,[np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])])
        # reset the basis of this object
        tp.basis=self.basis+otherBasis
        rb=tp.rotate_basis()
        elementals=all_elementals(rb)
        coefMat=coefficient_matrix(rb,elementals)
        rrefCM=np.array(sp.Matrix(coefMat).rref()[0],dtype=float)
        basis=np.matmul(rrefCM, all_elementals(rb))
        basis=[b for b in basis if b!=0]
        return len(basis)==len(self.basis)
        

    