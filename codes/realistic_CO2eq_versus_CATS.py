import numpy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


years=[2019,2020,2021,2022,2023,2024,2025]

## For real observations
#ucep_co2=[np.nan,55.9,56.3,55.3,56.2,57.7,np.nan]
ucep_co2=[np.nan,np.nan,np.nan,55.3,56.2,57.7,np.nan] # from 2025 report: Emissions Gap Report 2025: Off target
## For CAT
optim_co2=   [54.169,52.043,54.141,54.838,55.508,55.636,55.545] # Optimistc pathway
pledges_co2= [54.169,52.043,54.141,54.838,55.522,55.701,55.835] # Long-term pledge
target_co2=  [54.169,52.043,54.141,54.838,55.522,55.701,55.835] # 2030 & 2035 target
current_co2= [54.169,52.043,54.141,54.838,55.592,56.428,57.066] # Current policy action
cats_co2=[optim_co2, pledges_co2, target_co2, current_co2]
cats_label=['Optimistic','Pledges and targets','2030 & 2035 targets','Current policy actions']
colors=['royalblue','lightskyblue','lightsalmon','orangered']
lws=[1,1,1.5,1]
zorders=[2,1.5,0.9,0.5]


plt.close()
fig,ax1=plt.subplots(1,1,figsize=(5,3))
for i, cat_ts in enumerate(cats_co2):
    y=cat_ts
    ax1.plot(years,y,color=colors[i],lw=lws[i],label=cats_label[i],zorder=zorders[i],marker='o',ms=2)
## For the United Nation report
ax1.plot(years,ucep_co2,color='k',marker='o',label='UNEP Emissions Gap Report 2025',ms=2)
## Set legend
ax1.legend(bbox_to_anchor=(0,0.7), ncol=1, loc='lower left', frameon=False,
                columnspacing=0.5,handletextpad=0.5, labelspacing=0.2, reverse=True,fontsize=9)
ax1.set_ylabel("Total GHG emissions\n(GtCO2e/year)")
ax1.set_yticks([52,53,54,55,56,57,58])
## Save figures
for ax in [ax1]:
    for j in ['right', 'top']:
        ax.spines[j].set_visible(False)
        ax.tick_params(axis='x', which='both',length=2)
        ax.tick_params(axis='y', which='both',length=2)
## Save fig
fig_name = 'emission_compare'
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0) # hspace is the vertical
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)

