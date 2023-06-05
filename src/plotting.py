import matplotlib.pyplot as plt
import numpy as np

def nonint_pltData(lst,irrep):
    res = []
    for ops in lst:
        n = np.linalg.matrix_rank(np.matrix(ops['full'].projectors[irrep],dtype=float))
        if(n>0):
            res.append([ops['E'],n])
    return res


def nonint_plt(rhopiResults,threepiResults,irrep,basisCut=0,xRange=[2.9,7.1]):
    fig, ax = plt.subplots()
    colors=['red','green']
    styles=[(0,(3,5,1,5,1,5)),(0,(5,5))]
    axTOP = ax.twiny()

    xTopTicks=[]
    xTopLabels=[]
    for i,ops in enumerate([rhopiResults,threepiResults]):
        data=np.array(nonint_pltData(ops,irrep))
        if(len(data)!=0):
            ax.vlines(data[:,:1],0,1,
                    colors=colors[i],
                    linestyles=styles[i], lw=4)
            for n in data:
                xTopTicks.append(n[0])
                xTopLabels.append(str(int(n[1])))
        else:
            xTopTicks.append(0)
            xTopLabels.append('0')
#sort them
    xTopTicks, xTopLabels = (list(t) for t in zip(*sorted(zip(xTopTicks,xTopLabels))))
#apply a gap if the ticks are realy close
    changed=True
    while(changed):
        for i in range(len(xTopTicks)-1):
            changed=False
            if (xTopTicks[i+1]-xTopTicks[i])<0.03:
                xTopTicks[i]-=0.03
                xTopTicks[i+1]+=0.03
                changed=True
    
    
    ax.vlines(5.0,0,1,colors='black',lw=4)
    ax.vlines(basisCut,0,1,colors='grey',lw=4,linestyles=(0,(10,1)))

    plt.rcParams['ytick.labelleft']=plt.rcParams['ytick.left']=False
    #plt.rcParams["figure.figsize"]=(20,5)

    axTOP.set_xticks(xTopTicks)
    axTOP.set_xticklabels(xTopLabels,fontsize=20)
    ax.tick_params(axis='x',labelsize=24)

    ax.set_xlim(xRange)
    axTOP.set_xlim(xRange)
    ax.set_ylim([0.1,0.9])
    axTOP.set_ylim([0.1,0.9])
#plt.text(3,0.8,ensemble+'-'+irrep,fontsize=20)
    
    return plt