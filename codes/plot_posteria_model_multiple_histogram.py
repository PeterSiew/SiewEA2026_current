import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import pandas as pd
import multiprocessing
import scipy; from scipy import stats
from importlib import reload
import os

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)


if __name__ == "__main__":
    
    Y_predict_groups=np.load("/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_predict_fig3.npy",allow_pickle=True).item()
    groups=[*Y_predict_groups]

    model_multiple=20 # default in our study
    model_multiples=[40,30,20,10]
    colors=['darkred','red','orange','gold']
    predict_param_fn='parameters_LHC_1000_8param_range' # std_obs*20 (is this default?)
    rcp_titles={'rcp26':'RCP26','rcp45':'RCP45','rcp60':'RCP60','rcp85':'RCP85'}
    #cat_titles={'optimistic':'Optimistic',"longpledge":'Long-term pledge','shortpledge':'Short-term target','policyhigh':'Current actions'}
    cat_titles={'optim':'Optimistic',"pledge":'Pledges and targets','target':'2030 & 2035 targets','current':'Current policy actions'}
    titles={**rcp_titles,**cat_titles}

    mm_weights={}
    for mm in model_multiples:
        weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,mm)) 
        mm_weights[mm]=weights

    #ipdb.set_trace()
    plt.close()
    bin_no=20
    bin_no=30 # it won't change anything
    bw_method='scott'
    fig, axs = plt.subplots(2,4,figsize=(7,3))
    axs=axs.flatten()
    ### Pick up the year 2300
    for i, group in enumerate(groups):
        Y_predict_2300=Y_predict_groups[group][:,:,-1] # Only pick up 2300. #no. of GCM x parameters
        Y_predict_2300_reshape=Y_predict_2300.reshape(-1) # considers both GCM x parameters
        hist_plot=Y_predict_2300_reshape
        ## For prior
        ## Add histogram
        #n, bins, patches = axs[i].hist(hist_plot, bins=bin_no, density=True, color='gray', edgecolor='gray')
        ## Add KDE for prior
        kde = stats.gaussian_kde(hist_plot,bw_method=bw_method)
        xx = np.linspace(np.min(hist_plot)-np.std(hist_plot), np.max(hist_plot)+np.std(hist_plot), 500)
        axs[i].plot(xx, kde(xx), color='k',lw=1,zorder=0)
        ## Add text for prior
        q17=np.percentile(hist_plot,17,axis=0)
        q83=np.percentile(hist_plot,83,axis=0)
        axs[i].annotate("%s—%s"%(round(q17,1),round(q83,1)),xy=(0.50,0.4),xycoords='axes fraction', fontsize=7,color='k')
        ## For posteria
        for j, mm in enumerate(model_multiples):
            weights=mm_weights[mm]
            #np.tile(weights,Y_predict_2300.shape[0])
            weights_reshape=np.vstack(([weights]*Y_predict_2300.shape[0])).reshape(-1)
            ## Add histogram
            #n, bins, patches = axs[i].hist(hist_plot, bins=bin_no, density=True, weights=weights_reshape, color=colors[j],alpha=0.5, edgecolor=colors[j])
            ## Add KDE for posteria
            kde = stats.gaussian_kde(hist_plot,weights=weights_reshape,bw_method=bw_method) #bw_method ==> smaller is closer to histogram
            xx = np.linspace(np.min(hist_plot)-np.std(hist_plot), np.max(hist_plot)+np.std(hist_plot), 500)
            axs[i].plot(xx, kde(xx), color=colors[j], lw=1, zorder=j+1)
            ## add the likely range as texts on the right
            #if (group=='rcp85') & (mm==20) & False: 
            #ipdb.set_trace()
            q17=np.percentile(hist_plot,17,axis=0,weights=weights_reshape,method="inverted_cdf")
            q83=np.percentile(hist_plot,83,axis=0,weights=weights_reshape,method="inverted_cdf")
            ## alternative method to obtain q17 and q83 (very similar to obrain directly from historgram - for testing
            if (group=='rcp85') & (mm==20) & False: 
                cdf = scipy.integrate.cumulative_trapezoid(kde(xx), xx, initial=0) # initial=0 means adding 0 in the front to make the same size as xx
                q17_idx=np.abs(cdf-0.17).argmin(); q17_new=xx[q17_idx]
                q83_idx=np.abs(cdf-0.83).argmin(); q83_new=xx[q83_idx]
                print(group,mm)
                print(q17,q17_new)
                print(q83,q83_new)
            axs[i].annotate("%s—%s"%(round(q17.item(),1),round(q83.item(),1)),xy=(0.5,0.4+(j+1)*0.1),xycoords='axes fraction', fontsize=7,color=colors[j])
        ### Set legend
        if i==0:
            axs[i].plot([-100,-200],[-100,-200],color='k',lw=1,label='Prior')
            for j, mm in enumerate(model_multiples):
                axs[i].plot([-100,-200],[-100,-200],color=colors[j],lw=1,label='*%s'%mm)
                axs[i].legend(bbox_to_anchor=(0.01,1.2), ncol=5, loc='lower left', frameon=True, columnspacing=0.8,handletextpad=0.5, labelspacing=0.5, reverse=False)
        axs[i].set_title(titles[group],size=9,loc='left')
        axs[i].set_xlim(-20,220)
        axs[i].set_ylim(-0.002,0.09)
        axs[i].set_yticks([0,0.02,0.04,0.06,0.08])
        axs[i].set_xticks([0,100,200])
        if i in [4,5,6,7]:
            axs[i].set_xlabel('Sea level (cm)', fontsize=9)
        if i in [0,4]:
            axs[i].set_ylabel('Probability density', fontsize=9)
    ### Set axis
    for ax in axs:
        for j in ['right', 'top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2,direction='out')
    ### Save figures
    fig_name = 'histograms_prior_posteria_model_multiple'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.7,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)



