import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gvar
import math
import numpy as np

def model_average_summary_plot(pivotedCorr, en, energies, model_probs, mAvg, mpi, emYlim=None):
    NT=len(pivotedCorr[0])

    corr = lambda en: np.real(pivotedCorr[0][:,en,en])
    corrCov = lambda en: np.real(pivotedCorr[1][:,:,en,en])
    corrErr = lambda en: np.array([gvar.gvar(corr(en)[t],math.sqrt(np.abs(corrCov(en)[t,t]))) for t in range(NT)])
    effMass = lambda en: np.array([gvar.log(corrErr(en)[t]/corrErr(en)[t+1]) for t in range(NT-1)])/mpi
    fig = plt.figure(figsize=(14,6))

    gs = gridspec.GridSpec(2,2, width_ratios=[2,1])

    axTopRight = fig.add_subplot(gs[0,1])
    axBotRight = fig.add_subplot(gs[1,1])
    axLeft = fig.add_subplot(gs[:,0])


    ts=[t for t in range(len(effMass(en)))]
    axLeft.fill_between(ts, [(mAvg.mean-mAvg.sdev)/mpi for t in range(len(effMass(en)))],[(mAvg.mean+mAvg.sdev)/mpi for t in range(len(effMass(en)))],
                    color='lightgray', alpha=0.5)
    axLeft.errorbar(ts, [e.mean for e in effMass(en)], 
                yerr=[e.sdev for e in effMass(en)],  
                linestyle="None", marker="o", lw=1, color='black')
    axLeft.plot(ts, [mAvg.mean/mpi for t in range(len(effMass(0)))], color='gray')
    axLeft.set_ylabel('$aE~\\left[m_{\\pi}\\right]$')
    axLeft.set_xlabel('t')

    if emYlim is not None:
        axLeft.set_ylim(emYlim[0],emYlim[1])

    axTopRight.errorbar([i for i in range(len(energies))], [e.mean/mpi for e in energies], yerr=[e.sdev/mpi for e in energies],
        linestyle="None", marker=".", lw=1, color='black')
    axTopRight.errorbar([len(energies)+1], [mAvg.mean/mpi], yerr=[mAvg.sdev/mpi],
        linestyle="None", marker=".", lw=1, color='red')

    axTopRight.set_xlim(-0.5,len(model_probs)+1.5)

    yScale=1.5
    axTopRight.set_ylim((mAvg.mean-yScale*mAvg.sdev)/mpi, (mAvg.mean+yScale*mAvg.sdev)/mpi)


    axTopRight.tick_params(
        axis='x',
        labelbottom=False,
    )
    axTopRight.set_ylabel('$aE~\\left[m_{\\pi}\\right]$')

    axBotRight.scatter([i for i in range(len(energies))], [p for p in model_probs/np.sum(model_probs)])
    axBotRight.tick_params(
        axis='x',
        labelbottom=False,
    )
    axBotRight.set_xlabel('fit')
    axBotRight.set_ylabel('$p$')

    axBotRight.set_xlim(-0.5,len(model_probs)+1.5)






def adjustTicks(ticksPos):
    for i in range(len(ticksPos)):
        for j in range(len(ticksPos)):
            e0=ticksPos[i]
            e1=ticksPos[j]
            if e0!=e1:
                if np.isclose(e0,e1, atol=0.05):
                    e0lte1=e0<e1
                    if e0lte1:
                        ticksPos[i]-=0.03
                        ticksPos[j]+=0.03
                    else: 
                        ticksPos[i]+=0.03
                        ticksPos[j]-=0.03

def get_ni_ens(opData, optype): 
    return [opData[optype][i]['en'] for i in range(len(opData[optype]))]
def get_ni_mult(opData, optype):
    return [opData[optype][i]['mult'] for i in range(len(opData[optype]))]

def plot_ni_levels(opData):
    plt.hlines(get_ni_ens(opData, 'rhopi'),xmin=-11,xmax=100, linestyle=(0, (3, 5, 1, 5)),color='red', label=r'$\\rho\\pi')
    #plt.hlines(get_ni_ens(opData, 'sigmapi'),xmin=-11,xmax=100, linestyle=(0, (1,1)),color='blue')
    plt.hlines(get_ni_ens(opData, 'threepi'),xmin=-11,xmax=100, linestyle=(0, (5,5)),color='green', label=r'$\\pi\\pi\\pi')
    ax=plt.gca()
    ax2=ax.twinx()
    colors=['red' for i in range(len(get_ni_ens(opData, 'rhopi')))]
    #colors=colors+['blue' for i in range(len(get_ni_ens(opData, 'sigmapi')))]
    colors=colors+['green' for i in range(len(get_ni_ens(opData, 'threepi')))]

    ticksPos=get_ni_ens(opData, 'rhopi')+get_ni_ens(opData, 'threepi')
    adjustTicks(ticksPos)

    ax2.set_yticks(ticksPos)
    ax2.set_yticklabels(get_ni_mult(opData, 'rhopi')+get_ni_mult(opData, 'threepi'))
    ax2.set_ylim(ax.get_ylim())
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), colors):
        ticklabel.set_color(tickcolor)


def plot_fit_results(allEnergies,fitType,mpi):
    for idx, energies in allEnergies[fitType].items():
        xpos=[]
        yvals=[]
        yerrs=[]
        for i,energy in enumerate(energies):
            if energy.sdev/mpi<0.5:
                xpos.append(idx + 0.05*i)
                yvals.append(energy.mean/mpi)
                yerrs.append(energy.sdev/mpi)
        plt.errorbar(xpos, yvals, yerr=yerrs, linestyle="None", marker="o", lw=1)
    plt.gca().set_prop_cycle(None)
    for idx, energies in allEnergies[fitType].items():
        xpos=[]
        yvals=[]
        yerrs=[]
        for i,energy in enumerate(energies):
            if energy.sdev/mpi>=0.5:
                xpos.append(idx + 0.05*i)
                yvals.append(energy.mean/mpi)
                yerrs.append(0)
        plt.errorbar(xpos, yvals, yerr=yerrs, linestyle="None", marker="s", lw=1)



def plot_t0td_fit_results(allEnergies,fitType,mpi):
    idx=0
    ax=plt.gca()
    tickPos=[]
    tickLabel=[]

    for t0td, energies in allEnergies[fitType].items():
        xpos=[]
        yvals=[]
        yerrs=[]
        for i,energy in enumerate(energies):
            if energy.sdev/mpi<0.5:
                xpos.append(idx + 0.05*i)
                yvals.append(energy.mean/mpi)
                yerrs.append(energy.sdev/mpi)
        plt.errorbar(xpos, yvals, yerr=yerrs, linestyle="None", marker="o", lw=1)
        tickPos.append(idx)
        idx+=1
        tickLabel.append(t0td)
    
    ax.set_xticks(tickPos)
    ax.set_xticklabels(tickLabel)
    
    plt.gca().set_prop_cycle(None)

    idx=0
    for t0td, energies in allEnergies[fitType].items():
        
        xpos=[]
        yvals=[]
        yerrs=[]
        for i,energy in enumerate(energies):
            if energy.sdev/mpi>=0.5:
                xpos.append(idx + 0.05*i)
                yvals.append(energy.mean/mpi)
                yerrs.append(0)
        plt.errorbar(xpos, yvals, yerr=yerrs, linestyle="None", marker="s", lw=1)
        idx+=1
