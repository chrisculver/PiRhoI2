import lsqfit 
import gvar 
import numpy as np
from src.analysis.stats import get_raw_model_prob, model_avg

def closest_E(params, ref):
    ens=[val.mean for name,val in params.items() if name[0]=='E']
    ensShift=[np.abs(val.mean-ref) for name,val in params.items() if name[0]=='E']
    ers=[val.sdev for name,val in params.items() if name[0]=='E']
    
    idx=np.argmin(ensShift)
    return gvar.gvar(ens[idx],ers[idx])

def remove_bad_fits(allFits, fitinfos, model_probs, energies, mpi, probCut=0.001, sdevCut=1):
    model_probs /= np.sum(model_probs)

    badFits=[]
    for i,fit in enumerate(allFits):
        if model_probs[i] < probCut:
            badFits.append(i)
        elif energies[i].sdev/mpi > sdevCut:
            badFits.append(i)
        elif fit.chi2>1000:
            badFits.append(i)

    badFits

    allFits2=[]
    model_probs2=[]
    energies2=[]
    fitinfos2=[]

    for i in range(len(allFits)):
        if i not in badFits:
            model_probs2.append(model_probs[i])
            energies2.append(energies[i])
            allFits2.append(allFits[i])
            fitinfos2.append(fitinfos[i])

    return allFits2, fitinfos2, model_probs2, energies2


def remove_low_probs(energies, probs, probCut=0.001):
    probs /= np.sum(probs)
    badIndices=[]
    for i,p in enumerate(probs):
        if p < probCut:
            badIndices.append(i)

    energies2=[]
    probs2=[]
    for i in range(len(probs)):
        if i not in badIndices:
            energies2.append(energies[i])
            probs2.append(probs[i])
    return energies2, probs2


def perform_many_fits(data, en, fitTypes, t0, tiMax, mpi, debug=False):
    NT = len(data[0])
    ts=np.array([t for t in range(NT)])
    corr = np.real(data[0][:,en,en])
    corrCov = np.real(data[1][:,:,en,en])

    fits=[]
    model_probs=[]
    energies=[]
    fitinfos=[]
    referenceEnergy=None
    for k,ft in fitTypes.items():
        for ti in range(t0, tiMax+1):
            for tf in range(ti+len(ft['p0'])+1, NT):
                fit=lsqfit.nonlinear_fit(
                    data=(ts[ti:tf+1],corr[ti:tf+1],corrCov[ti:tf+1,ti:tf+1]),
                    fcn=ft['fcn'],
                    p0=ft['p0']
                )
                if debug:
                    print("ti={}, tf={}, type={}".format(ti,tf,k))
                    print("fitres={}, chi2={}, dof={}".format(fit.p, fit.chi2, fit.dof))

                enVal=None
                if k=='single':
                    enVal=fit.p['E0']
            
                else:
                    if not referenceEnergy:
                        referenceEnergy = np.mean([e.mean for e in energies])
                    enVal=closest_E(fit.p, referenceEnergy)

                fits.append(fit)
                model_probs.append(get_raw_model_prob(fit, IC="AIC", N_cut=(NT-tf)+ti-t0))
                energies.append(enVal)
                fitinfos.append({'type': k, 'ti': ti, 'tf': tf})

    return fits,fitinfos,model_probs,energies  



def perform_good_fits_on_resamples(en, samples, cov, allFitInfos, fitTypes, t0, NT):
    ts=np.array([t for t in range(NT)])
    allMeans=[]
    for sample in samples:
        energies=[]
        model_probs=[]
        referenceEnergy=None
        for fitInfo in allFitInfos:
            ft=fitTypes[fitInfo['type']]
            ti=fitInfo['ti']
            tf=fitInfo['tf']
            fit=lsqfit.nonlinear_fit(
                data=(ts[ti:tf+1],sample[ti:tf+1,en,en],cov[ti:tf+1,ti:tf+1,en,en]),
                fcn=ft['fcn'],
                p0=ft['p0']
                )
            
            enVal=None
            if fitInfo['type']=='single':
                enVal=fit.p['E0']
            
            else:
                if not referenceEnergy:
                    referenceEnergy = np.mean([e.mean for e in energies])
                enVal=closest_E(fit.p, referenceEnergy)


            model_probs.append(get_raw_model_prob(fit, IC="AIC", N_cut=(NT-tf)+ti-t0))
            energies.append(enVal)

        mAvg=model_avg(energies, model_probs)
        allMeans.append(mAvg.mean)
    return allMeans










def min_chi2_fit(energies, allFits):
    bestFitIdx=0
    bestFitChi2=1e10
    for i,fit in enumerate(allFits):
        if fit.chi2 < bestFitChi2:
            bestFitIdx=0
            bestFitChi2=fit.chi2
    
    return energies[bestFitIdx]


def perform_model_avg(data, en, fitTypes, t0, tiMax, mpi):
    #print("WARNING: Deprecated, see L4848")
    NT = len(data[0])
    ts=np.array([t for t in range(NT)])
    corr = np.real(data[0][:,en,en])
    corrCov = np.real(data[1][:,:,en,en])

    model_probs=[]
    energies=[]
    referenceEnergy=None
    for k,ft in fitTypes.items():
        for ti in range(t0, tiMax+1):
            for tf in range(ti+len(ft['p0'])+1, NT):
                fit=lsqfit.nonlinear_fit(
                    data=(ts[ti:tf+1],corr[ti:tf+1],corrCov[ti:tf+1,ti:tf+1]),
                    fcn=ft['fcn'],
                    p0=ft['p0']
                )
                
                enVal=None
                if k=='single':
                    enVal=fit.p['E0']
            
                else:
                    if not referenceEnergy:
                        referenceEnergy = np.mean([e.mean for e in energies])
                    enVal=closest_E(fit.p, referenceEnergy)
                

                if enVal.sdev/mpi < 1:
                    model_probs.append(get_raw_model_prob(fit, IC="AIC", N_cut=(NT-tf)+ti-t0))
                    energies.append(enVal)

    energies, model_probs = remove_low_probs(energies, model_probs)

    return model_avg(energies, model_probs)


def perform_best_fit(data, en, fitTypes, t0, tiMax, mpi):
    #print("WARNING: Deprecated, see L4848")
    NT = len(data[0])
    ts=np.array([t for t in range(NT)])
    corr = np.real(data[0][:,en,en])
    corrCov = np.real(data[1][:,:,en,en])

    bestFit=None
    minChi2=1e8
    referenceEnergy=None
    energies=[]
    for k,ft in fitTypes.items():
        for ti in range(t0, tiMax+1):
            for tf in range(ti+len(ft['p0'])+1, NT):
                fit=lsqfit.nonlinear_fit(
                    data=(ts[ti:tf+1],corr[ti:tf+1],corrCov[ti:tf+1,ti:tf+1]),
                    fcn=ft['fcn'],
                    p0=ft['p0']
                )
                
                enVal=None
                if k=='single':
                    enVal=fit.p['E0']
            
                else:
                    if not referenceEnergy:
                        referenceEnergy = np.mean([e.mean for e in energies])
                    enVal=closest_E(fit.p, referenceEnergy)
                
                if enVal.sdev/mpi<1:
                    energies.append(enVal)

                if fit.chi2 < minChi2:
                    bestFit=fit
                    minChi2=fit.chi2

    if not referenceEnergy:
        referenceEnergy = np.mean([e.mean for e in energies])

    return closest_E(bestFit.p, referenceEnergy)









def single_exp(t,p):
    return p["A0"]*np.exp(-p["E0"]*t)
def double_exp(t,p):
    return p["A0"]*np.exp(-p["E0"]*t)+p["A1"]*np.exp(-p["E1"]*t)
def triple_exp(t,p):
    return p["A0"]*np.exp(-p["E0"]*t)+p["A1"]*np.exp(-p["E1"]*t)+p["A2"]*np.exp(-p["E2"]*t)

def p0_single():
    return { "A0": 1.0, "E0": 0.7 }

def p0_double():
    return { "A0": 1.0, "E0": 0.7, "A1": 0.1, "E1": 2.0 }

def p0_triple():
    return { "A0": 1.0, "E0": 0.7, "A1": 0.1, "E1": 2.0, "A2": 1e-4, "E2": 0.1  }