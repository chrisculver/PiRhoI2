from src.ops.manageProjections import *
from src.ops.three_pi import *

import math
import sympy as sp

def mom2(p,L,eta):
    return 4*math.pi*math.pi*(p[0]*p[0]+p[1]*p[1]+(p[2]*p[2]/(eta*eta)))/(L**2)

def EN_rhopi(p1,p2,mrho,mpi,L,eta):
    return (math.sqrt(mrho*mrho + mom2(p1,L,eta))
          +math.sqrt(mpi*mpi + mom2(p2,L,eta)))/mpi
    
def EN_sigmapi(p1,p2,msigma,mpi,L,eta):
    return (math.sqrt(msigma*msigma + mom2(p1,L,eta))
          +math.sqrt(mpi*mpi + mom2(p2,L,eta)))/mpi
    
def EN_threepi(p1,p2,p3,mpi,L,eta):
    return (math.sqrt(mpi*mpi + mom2(p1,L,eta))
          +math.sqrt(mpi*mpi + mom2(p2,L,eta))
          +math.sqrt(mpi*mpi + mom2(p3,L,eta)))/mpi

# for momenta combinations:
#   if (en < cutoff) and (seed not in existing basis):
#     ManageOps(momenta)

def indicesUpTo(N):
    res = []
    for i in range(-N,N+1):
        for j in range(-N,N+1):
            for k in range(-N,N+1):
                res.append([i,j,k])
    return res

def float_mom(p):
    return [p[0]*1.0, p[1]*1.0, p[2]*1.0]

def get_rhopi_results(group,pTot, N, enCUT, mrho, mpi, L,eta):
    rhopiResults = []

    for p1 in indicesUpTo(N):
        p1=sp.Matrix(float_mom(p1))
        p2 = sp.Matrix(float_mom(pTot-p1))
        en=EN_rhopi(p1,p2,mrho,mpi,L,eta)
        if(en<enCUT):
            rhopi = ManageRhoPi(group,[p1,p2])
            compute = True
            for newOp in rhopi.basis:
                for results in rhopiResults:
                    if newOp in results['full'].basis:
                        compute=False
                        break
            
            if(compute):
                rhopiResults.append({'E':en, 'seed':[p1,p2], 'full':rhopi})
    return rhopiResults


def get_threepi_results(group,pTot,N,enCUT,mpi,L,eta,timed=False):
    threepiResults = []
    #timer=Timer()
    for p1 in indicesUpTo(N):
        p1=np.array(p1)
        for p2 in indicesUpTo(N):
            if (p1==np.array([0,0,0])).all() and (p2==np.array([0,0,0])).all():
                continue
            p2=np.array(p2)
            p3=np.array([pTot[0],pTot[1],pTot[2]])-p1-p2
            en=EN_threepi(p1,p2,p3,mpi,L,eta)
            if(en<enCUT):
                #print('[{},{},{}]'.format(p1,p2,p3))
                #timer.start('running fixed seed')
                threepi = ThreePiFixedSeed(group,[p1,p2,p3])
                #threepi.run()
                #timer.stop()
                compute=True
                #timer.start('checking other results')
                for results in threepiResults:
                    #as long as one of the existing basis is equivalent to the new one
                    #don't add it.
                    
                    if threepi.is_equivalent_basis(results['full'].basis):
                        compute=False
                        break
                        
                #timer.stop()
                if(compute):
                    #if(timed):
                    #    print("New ops at E={}, seed=[{},{},{}])".format(en,p1,p2,p3))
                    threepi.run()
                    threepiResults.append({'E':en, 'seed':[p1,p2,p3],'full':threepi})

    return threepiResults


def seedsToNumpy(results):
    res=[]
    for ops in results:
        mom=[np.array(ops['seed'][0]).astype(np.float64),np.array(ops['seed'][1]).astype(np.float64)]
        res.append(mom)
    return res

# this should go in MANAGER...
def getOperators(result,irrep):
    mat = sp.Matrix(result.projectors[irrep]).rref()[0]
    ops = np.matmul(np.array(mat).astype(np.float64),result.basis)
    return [o for o in ops if o!=0]

def momXZ(p):
    return sp.Matrix([p[2],p[1],p[0]])

def getQuarkSubs(allMomenta):
    quarkSubs = {}
    for p in allMomenta:
        quarkSubs[sp.Function('pi^+')(p)]=sp.Function('u5d')(p)
        quarkSubs[sp.Function('pi^0')(p)]=sp.Function('d5d')(p)-sp.Function('u5u')(p)
        quarkSubs[sp.Function('pi^-')(p)]=sp.Function('d5u')(p)
        quarkSubs[sp.Function('sigma')(p)]=sp.Function('dId')(p)+sp.Function('uIu')(p)
        for d in ['x','y','z']:
            quarkSubs[sp.Function('rho_'+d+'^+')(p)]=sp.Function('u'+d+'d')(p)
            quarkSubs[sp.Function('rho_'+d+'^0')(p)]=sp.Function('d'+d+'d')(p)-sp.Function('u'+d+'u')(p)
            quarkSubs[sp.Function('rho_'+d+'^-')(p)]=sp.Function('d'+d+'u')(p)

    return quarkSubs


def cppMom(m,swapXZ=False):
    tmp=m.replace('m','-')
    for i in ['0','1','2','3','4','5','6','7','8','9']:
        tmp=tmp.replace(i,i+' ')
    dirs=tmp.split(' ')
    if swapXZ:
        tmp=dirs[2]+' '+dirs[1]+' '+dirs[0]
    else:
        tmp=dirs[0]+' '+dirs[1]+' '+dirs[2]
    return tmp

gammaCPP={'x':'1','y':'2','z':'3','I':'6','5':'5'}
gammaCPPswapXZ={'x':'3','y':'2','z':'1','I':'6','5':'5'}

#def opToCPP(op):
#    allMomenta=unique_arguments(op)
#    quarkSubs=getQuarkSubs(allMomenta)
#    
#    o = sp.sstr(sp.expand(op.subs(quarkSubs))).replace('\n','')
#    o = o.replace('(0)','(000)')
#    o = o.replace(' ','')
#    o=['+'+t for t in o.split('+')]
#    for i,t in enumerate(o):
#        if t[0:2]=='+-':
#            o[i]=t[1:]
#            
#    o2=[]
#    for t in o:
#        tt=['-'+t for t in t.split('-')]
#        for s in tt:
#            if(s!='-'):
#                if s[0:2]=='-+':
#                    o2.append(s[1:])
#                else:
#                    o2.append(s)
#                    
#    s=''
#    for t in o2:
#        lst=t.split('*')
#        if(lst[0][0]=='-'):
#            s += '+' + lst[0] + '|'
#        else:
#            s += lst[0] + '|'
#        for i,m in enumerate(lst[1:]):
#            s+=m[0]+','+gammaCPP[m[1]]+',\delta_{ii},'+cppMom(m[4:-1])+','+m[2]
#            if(i!=len(lst[1:])-1):
#                s+='|'
#    
#    return s


def opToCPP(op,swapXZ=False):
    allMomenta=unique_arguments(op)
    quarkSubs=getQuarkSubs(allMomenta)
    
    o = sp.sstr(sp.expand(op.subs(quarkSubs))).replace('\n','')
    o = o.replace('(0)','(000)')
    o = o.replace(' ','')
    o=['+'+t for t in o.split('+')]
    for i,t in enumerate(o):
        if t[0:2]=='+-':
            o[i]=t[1:]
            
    o2=[]
    for t in o:
        tt=['-'+t for t in t.split('-')]
        for s in tt:
            if(s!='-'):
                if s[0:2]=='-+':
                    o2.append(s[1:])
                else:
                    o2.append(s)
                    
    s=''
    for t in o2:
        lst=t.split('*')
        del1 = -1
        copy = -1
        for idx,elem in enumerate(lst):
            if elem=='':
                del1=idx
                copy=idx-1
        if(copy!=-1):
            lst.pop(del1)
            lst.pop(del1)
            lst.append(lst[copy])
        #for elem in lst:
        #    print(elem)
        
        if(lst[0][0]=='-'):
            s += '+' + lst[0] + '|'
        else:
            s += lst[0] + '|'
        for i,m in enumerate(lst[1:]):
            if(swapXZ):
                s+=m[0]+','+gammaCPPswapXZ[m[1]]+',\\delta_{ii},'+cppMom(m[4:-1],swapXZ)+','+m[2]
            else:
                s+=m[0]+','+gammaCPP[m[1]]+',\\delta_{ii},'+cppMom(m[4:-1])+','+m[2]
            if(i!=len(lst[1:])-1):
                s+='|'
                
    if s[0]=='+':
        s=s[1:]
    return s