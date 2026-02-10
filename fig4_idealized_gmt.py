import xarray as xr
import numpy as np; np.set_printoptions(legacy='1.25') # Don't print np.flaot64 
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
#import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import pandas as pd
import multiprocessing
import scipy; from scipy import stats
from sklearn.linear_model import Ridge
from importlib import reload
import os

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)


if __name__ == "__main__":

    ## Choose what regions
    region='greenland'
    region='global'

    ## Set training models and years
    train_models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var','MIROC5_rcp2685mean_pad2600_8var','MIROC5_rcp85cooling_pad2600_8var']
    train_years={model:range(2015,2601) for model in train_models}
   
    ## Set predict models and years
    predict_param_fn='parameters_LHC_5000_8param_range' 
    predict_param_fn='parameters_LHC_1000_8param_range' 
    predict_params=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s.npy'%predict_param_fn)
    model_multiple=20 # this is the standard
    weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
    param_no=len(predict_params)
    #predict_years=range(2015,2301)

    ### Plotting idealized GMT (Figure 5)
    plotting_fig5=True 

    if plotting_fig5: 
        if False: # only stablaization up to year 2301 (Figure 5B)
            predict_models=['stable2050_policyhigh_nopad2300','stable2100_policyhigh_nopad2300',
                            'stable2150_policyhigh_nopad2300', 'stable2200_policyhigh_nopad2300',
                            'stable2250_policyhigh_nopad2300','stable2300_policyhigh_nopad2300'] # The last one is the same as current policy
            predict_years=range(2015,2301)
            ABCD=['A','B','C','D']
        else: ### New extended to year 4000 (Figure 5C)
            #predict_models=['stable2050to4000_current_nopad2300','stable2100to4000_current_nopad2300',
            #                'stable2150to4000_current_nopad2300', 'stable2200to4000_current_nopad2300',
            #                'stable2250to4000_current_nopad2300','stable2300to4000_current_nopad2300'] # The last one is similar to current policy - but extended to year 4000
            #predict_years=range(2015,4001)
            predict_models=['stable2050to3000_current_nopad2300','stable2100to3000_current_nopad2300',
                            'stable2150to3000_current_nopad2300', 'stable2200to3000_current_nopad2300',
                            'stable2250to3000_current_nopad2300','stable2300to3000_current_nopad2300'] 
            predict_years=range(2015,3001)
            ABCD=['A','B']

        predict_models_ens=['1']*len(predict_models)
        labels=['Stable in 2050', 'Stable in 2100',
                'Stable in 2150', 'Stable in 2200',
                'Stable in 2250', 'No stabalization']
        labels=['2050', '2100', '2150', '2200', '2250', 'Current Policy']
        labels=['2050', '2100', '2150', '2200', '2250', '2300']
        models_colors=['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f']
        models_colors=['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','orangered']
        models_colors=['gainsboro', 'lightgray', 'darkgrey', 'grey', 'dimgrey', 'black']
        models_colors=['#f7fcb9','#fed976','#fd8d3c','#e31a1c','#800026','#662506']

    ### Start the algorithm here (this is universial for Figures 3 and 5)
    ### Read GMT for training models
    ## train_ens has to be 1, because PISM forced by MIROC5 en1
    gmt_ens=[1]*len(train_models)
    gmt_datas,gmt_cum_datas,time_since_last_changes,gmt_cum_gradient_datas=functions_read.read_gmts(train_models,gmt_ens)
    ## Xs, Ys are for training; pism_slrs are the actual simualted SLR
    Xs,Ys,pism_slrs,records=functions_read.read_training_XY(train_models,train_years,gmt_datas,gmt_cum_datas,time_since_last_changes,slr_relative_yr=2015)

    ### Do the training 
    ### Standardize the training data (Xs to X_train_standard)
    X_mean=Xs.mean(axis=0)
    X_std=Xs.std(axis=0)
    X_train_standard=(Xs-X_mean)/X_std
    ## Standardize Ys to Y
    Y_mean=Ys.mean(axis=0)
    Y_std=Ys.std(axis=0)
    Y_train_standard=(Ys-Y_mean)/Y_std
    ### Do the training 
    regress = Ridge(alpha=700).fit(X_train_standard,Y_train_standard) # the best result based on figS1
    #print(regress.coef_)

    ### Read GMT for predict models (make sure the model name includes the en - some model has en1 to en4., CSIRO-Mk3-6-0_rcp85)
    gmt_datas={}
    gmt_cum_datas={}
    time_since_last_changes={}
    gmt_cum_gradient_datas={}
    for predict_model, en in zip(predict_models, predict_models_ens):
        gmt_data,gmt_cum_data,time_since_last_change,gmt_cum_gradient_data=functions_read.read_gmts([predict_model],[en])
        gmt_datas[predict_model+'_en%s'%en]=gmt_data[predict_model]
        gmt_cum_datas[predict_model+'_en%s'%en]=gmt_cum_data[predict_model]
        time_since_last_changes[predict_model+'_en%s'%en]=time_since_last_change[predict_model]
        gmt_cum_gradient_datas[predict_model+'_en%s'%en]=gmt_cum_gradient_data[predict_model]


    ### Do the prediction using the trained models (Ridge Regression - regress) via multi-processing
    predict_models_new=[*gmt_datas] # this one further includes the ensemble number
    argus=[]
    for model in predict_models_new:
        argu=(model,predict_params,predict_years,gmt_datas[model],gmt_cum_datas[model],X_mean,X_std,Y_mean,Y_std,regress)
        argus.append(argu)
    if True: # Multiprocessing result
        pool_no=50
        pool = multiprocessing.Pool(pool_no)
        results=pool.starmap(functions_read.do_prediction,argus)
    else: # debugging mode
        results=[]
        for argu in argus:
            result=functions_read.do_prediction(argu)
            results.append(result)
    ### Put the results into correct format
    Y_predicts={}
    for i, model in enumerate(predict_models_new):
        Y_predict=results[i]
        Y_predicts[model]=Y_predict
    Y_predict_all=[Y_predicts[model][j][year] for model in predict_models_new for j in range(predict_params.shape[0]) for year in predict_years]
    Y_predict_reshape=np.array(Y_predict_all).reshape(len(predict_models_new),predict_params.shape[0],len(predict_years)) # no. of models x no. of parameters x years

    ### find the posteria (with history matching) from prior distribution
    slr_prior_qs={model:{} for model in predict_models}
    slr_post_qs={model:{} for model in predict_models}
    for i, model in enumerate(predict_models):
        slr_prior_qs[model][0.5]=np.percentile(Y_predict_reshape[i],50,axis=0,weights=weights,method="inverted_cdf")
        slr_prior_qs[model][0.17]=np.percentile(Y_predict_reshape[i],17,axis=0,weights=weights,method="inverted_cdf")
        slr_prior_qs[model][0.83]=np.percentile(Y_predict_reshape[i],83,axis=0,weights=weights,method="inverted_cdf")
        slr_post_qs[model][0.5]=np.percentile(Y_predict_reshape[i],50,axis=0,weights=weights,method="inverted_cdf")
        slr_post_qs[model][0.17]=np.percentile(Y_predict_reshape[i],17,axis=0,weights=weights,method="inverted_cdf")
        slr_post_qs[model][0.83]=np.percentile(Y_predict_reshape[i],83,axis=0,weights=weights,method="inverted_cdf")

    for i, model in enumerate(predict_models):
        slr_last_year=slr_post_qs[model][0.5][-1]
        print(model,'SLR at yr 3000 is: ',slr_last_year)
    ### Print the 2300 SLR rise (medians) for all scenarios, and make the supplementary figure (the expotential decrease of SLR)
    if True:
        #stable_years={'stable2050to4000_current_nopad2300':2050, 'stable2100to4000_current_nopad2300':2100,
        #              'stable2150to4000_current_nopad2300':2150, 'stable2200to4000_current_nopad2300':2200,
        #              'stable2250to4000_current_nopad2300':2250, 'stable2300to4000_current_nopad2300':2300} 
        stable_years={'stable2050to3000_current_nopad2300':2050, 'stable2100to3000_current_nopad2300':2100,
                      'stable2150to3000_current_nopad2300':2150, 'stable2200to3000_current_nopad2300':2200,
                      'stable2250to3000_current_nopad2300':2250, 'stable2300to3000_current_nopad2300':2300} 
        testing_years=[2300,3000]
        plt.close()
        fig,axs=plt.subplots(len(testing_years),1,figsize=(3,1.5*len(testing_years)))
        if len(testing_years)==1:
            axs=[axs]
        for i, test_year in enumerate(testing_years):
            year_idx=predict_years.index(test_year)
            percentage_changes=[]
            for j, model in enumerate(predict_models):
                slr_model=slr_post_qs[model][0.5][year_idx]
                #slr_2300stable=slr_post_qs['stable2300to3000_current_nopad2300'][0.5][year_idx]
                #percentage_change=(slr_model-slr_2300stable)/slr_2300stable*100
                ## Set the old as 2050
                slr_2050stable=slr_post_qs['stable2050to3000_current_nopad2300'][0.5][year_idx]
                percentage_change=(slr_model-slr_2050stable)/slr_2050stable*100
                percentage_changes.append(percentage_change)
            print("Percentage changes in year %s is %s"%(test_year,percentage_changes))
            ###
            ### Start plotting for each testing year
            #years=[2050,2100,2150,2200,2250,2300]
            x=range(len(predict_models))
            axs[i].plot(x,percentage_changes,marker='X',markersize=2)
            axs[i].set_xticks(x)
            #axs[i].set_yticks([0,-25,-50,-75])
            #axs[i].set_yticks([0,-25,-50])
            axs[i].set_yticks([0,50,100,150])
            axs[i].set_ylim(-5,150)
            axs[i].axhline(y=0,color='lightgray',linestyle='--',lw=0.5)
            axs[i].grid()
            #axs[i].axhline(y=100,color='lightgray',linestyle='--',lw=0.5)
            #axs[i].axhline(y=200,color='lightgray',linestyle='--',lw=0.5)
            #axs[i].set_title(test_year,loc='left')
            axs[i].annotate('Year: %s'%test_year,xy=(0.01,0.8),xycoords='axes fraction', fontsize=10)
            #if i==2:
            axs[i].set_ylabel("% change of\nsea level")
            if i==len(testing_years)-1:
                axs[i].set_xlabel("Year of GMT stabalisation")
                axs[i].set_xticklabels([stable_years[m] for m in predict_models])
            else:
                axs[i].set_xticklabels([])
        for ax in axs:
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        ## Save figures
        fig_name = 'figSX_expotential_SLR_decrease'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.3) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

    ###
    ### Start the plotting of Figure 5
    if plotting_fig5:
        ### Start plotting
        plt.close()
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(4,5))
        ## ax1 - the GMT showing the stabalization
        for i, model in enumerate(predict_models):
            #years=range(2015,2301)
            gmt=[gmt_datas[model+'_en1'][year] for year in predict_years]
            #ax1.plot(predict_years,gmt,color=models_colors[i],linewidth=8-i*1.3,zorder=i)
            ax1.plot(predict_years,gmt,color=models_colors[i],linewidth=2,zorder=-i)
            ax1.plot([-200,-201],[-200,-201],color=models_colors[i],label=labels[i],linewidth=2,zorder=i)
        #ax1.set_xlim(predict_years[0],predict_years[-1])
        ax1.set_ylim(0.5,5.5); ax1.set_yticks([1,2,3,4,5])
        ax1.set_ylabel('Global mean\ntemperature (K)')
        ## ax2 - slr gmt
        max_q=0.83; min_q=0.17
        for i, model in enumerate(predict_models):
            slr_median=slr_post_qs[model][0.5][:]
            slr_max=slr_post_qs[model][max_q][:]
            slr_min=slr_post_qs[model][min_q][:]
            ax2.plot(predict_years,slr_median,color=models_colors[i],lw=2,zorder=3,label=labels[i])
            ax2.fill_between(predict_years,slr_min,slr_max,fc=models_colors[i],zorder=1,color=models_colors[i], alpha=0.3, linewidth=0)
            ## Error bar for the last year
            median=slr_median[-1]
            ymin=slr_min[-1]
            ymax=slr_max[-1]
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            ax2.errorbar(predict_years[-1]+20*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='_',elinewidth=2,ms=3)
            ax2.errorbar(predict_years[-1]+20*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='o',elinewidth=0,ms=3)
        for ax in [ax1,ax2]:
            ax.set_xlim(predict_years[0],predict_years[-1]+130)
            ax.set_xticks([2200,2400,2600,2800,3000])
        ax2.set_ylabel('Sea-level contribution\n(cm)')
        ax2.legend(bbox_to_anchor=(0,0.4), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2, reverse=False)
        #ax2.set_xticks([2015]+[i for i in range(2050,2301,50)])
        if False: ## False for supp. figure - simulations extended to year 4000
            ax2.set_xticks([i for i in range(2050,2301,50)])
            ax2.set_ylim(0,60)
        ## Setup axis
        for ax in [ax1,ax2]:
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        for ax in [ax1,ax2]:
            #for yr in [2050,2100,2150,2200,2250]:
            for yr in [2300]:
                ax.axvline(x=yr,color='lightgray',linestyle='--',lw=0.9)
                pass
        ## Set axis title
        titles=['Stabalized warming profiles', 'SLR response up to year 4000']
        titles=['','','','']
        for i, ax in enumerate([ax1,ax2]):
            ax.annotate(r'$\bf({%s})$ %s'%(ABCD[i],titles[i]),xy=(-0.20,0.98),xycoords='axes fraction', fontsize=10)
        # Save figures
        fig_name = 'fig4_slrs_timeseries_idealized'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.25) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)


