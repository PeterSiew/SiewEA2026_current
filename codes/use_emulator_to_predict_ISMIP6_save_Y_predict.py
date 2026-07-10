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
import random

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import fig2_fig3_GMT_SLR_fractional_idealized as fig23; reload(fig23)
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)


if __name__ == "__main__":

    ## Choose what regions
    region='greenland'
    region='global'

    ## Set training models and years
    train_models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var','MIROC5_rcp2685mean_pad2600_8var','MIROC5_rcp85cooling_pad2600_8var']
    #train_years={model:range(2015,2101) for model in train_models} # This result in higher SLR
    train_years={model:range(2015,2601) for model in train_models}
   
    ## Set predict params
    predict_param_fn='parameters_LHC_1000_8param_range'  # default
    predict_param_fn='parameters_LHC_1000_8param_range_ISMIP' # new one
    predict_params=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s.npy'%predict_param_fn)
    if False: ## Reading the weights (we don't need the weights here, as here we only create the uncalibrated projections
        model_multiple=20 # this is the standard
        weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
        param_no=len(predict_params)

    #################################################################################
    ### Define predict models and predict years

    if False: ## CMIP6 SSP585 up to years 2300 (not used)
        predict_models1=['CNRM-ESM2-1_ssp585_pad2300','IPSL-CM6A-LR_ssp585_pad2300','MPI-ESM1-2-HR_ssp585_pad2300','UKESM1-0-LL_ssp585_pad2300']
        predict_models2=['CESM2_ssp245_pad2300','IPSL-CM6A-LR_ssp245_pad2300','MPI-ESM1-2-HR_ssp245_pad2300']
        predict_models3=['CESM2_ssp126_pad2300','MPI-ESM1-2-HR_ssp126_pad2300']
        ## Add them together
        predict_models=predict_models1+predict_models2+predict_models3
        predict_models_ens=[0]*len(predict_models)
        predict_years=range(2015,2301)

    if False:  ## CMIP5 RCP85 up to 2100
        ## We don't include HadGEM back as it starts from 1860 rather than 1850
        predict_models=['ACCESS1-3_rcp85_pad2300','CSIRO-Mk3-6-0_rcp85_pad2300','IPSL-CM5A-MR_rcp85_pad2300', 
                        'MIROC5_rcp85_pad2300','NorESM1-M_rcp85_pad2300','MIROC5_rcp26_pad2300']
        predict_models_ens=[1]*len(predict_models)
        predict_years=range(2015,2101)

    if True:  ## CMIP6 SSP585 up to 2100
        ## Not sure if MPI-ESM1 is correct ==> LR or HR? ==> HR is correct
        ## Remove CESM2-WACCM_ssp585 as it is not used for year 2100 prediction in Golzer et al.
        predict_models1=['CESM2_ssp585_pad2300','CNRM-CM6-1_ssp585_pad2300','CNRM-ESM2-1_ssp585_pad2300',
                        'EC-Earth3_ssp585_pad2300','IPSL-CM6A-LR_ssp585_pad2300','MPI-ESM1-2-HR_ssp585_pad2300','NorESM2-MM_ssp585_pad2300',
                        'UKESM1-0-LL_ssp585_pad2300']
        predict_models2=['CESM2_ssp245_pad2300','MPI-ESM1-2-HR_ssp245_pad2300','NorESM2-MM_ssp245_pad2300','UKESM1-0-LL_ssp245_pad2300']
        predict_models3=['CESM2_ssp126_pad2300','EC-Earth3_ssp126_pad2300','MPI-ESM1-2-HR_ssp126_pad2300']
        ## Adding them all together
        predict_models=predict_models1+predict_models2+predict_models3
        predict_models_ens=[0]*len(predict_models)
        predict_years=range(2015,2101)


    colors=["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99"]
    colors=['gray']*100
    models_colors={model:colors[i] for i, model in enumerate(predict_models)}

    #**********************************************************************************************************#
        
    ### Start the algorithm here (this is universial for Figures 3 - mian and Figure 5 - idazlied GMT)
    ### Read GMT for training models
    ## train_ens has to be 1, because PISM forced by MIROC5 en1
    print("Read training data")
    gmt_ens=[1]*len(train_models)
    gmt_datas,gmt_cum_datas,time_since_last_changes,gmt_cum_gradient_datas=functions_read.read_gmts(train_models,gmt_ens)
    ## Xs, Ys are for training; pism_slrs are the actual simualted SLR
    Xs,Ys,pism_slrs,records=functions_read.read_training_XY(train_models,train_years,gmt_datas,gmt_cum_datas,time_since_last_changes,slr_relative_yr=2015)
    print("Read training datai - Finish")

    ### Do the training 
    print("Start training")
    ### Standardize the training data (Xs to X_train_standard)
    X_mean=Xs.mean(axis=0)
    X_std=Xs.std(axis=0)
    X_train_standard=(Xs-X_mean)/X_std
    ## Standardize Ys to Y
    Y_mean=Ys.mean(axis=0)
    Y_std=Ys.std(axis=0)
    Y_train_standard=(Ys-Y_mean)/Y_std
    ### Do the training 
    ## Do the Rdige Regression 
    ridge_alpha=700
    regress = Ridge(alpha=ridge_alpha,fit_intercept=True).fit(X_train_standard,Y_train_standard) # the optimal alpha=700 based on figS1
    print("Regression X1-X10 coefficients + intercept at alpha=%s"%ridge_alpha)
    print(regress.coef_, regress.intercept_)


    ### Prediction phase-read GMT for predict models (make sure the model name includes the en - some model has en1 to en4., CSIRO-Mk3-6-0_rcp85)
    ### The training is done - so we can create new dict for gmt_datas
    gmt_datas={}
    gmt_cum_datas={}
    time_since_last_changes={}
    gmt_cum_gradient_datas={}
    print("Read GMT for predict models")
    for predict_model, en in zip(predict_models, predict_models_ens):
        gmt_data,gmt_cum_data,time_since_last_change,gmt_cum_gradient_data=functions_read.read_gmts([predict_model],[en])
        gmt_datas[predict_model+'_en%s'%en]=gmt_data[predict_model]
        gmt_cum_datas[predict_model+'_en%s'%en]=gmt_cum_data[predict_model]
        time_since_last_changes[predict_model+'_en%s'%en]=time_since_last_change[predict_model]
        gmt_cum_gradient_datas[predict_model+'_en%s'%en]=gmt_cum_gradient_data[predict_model]
    print("Read GMT for predict models - Finish")

    ### Do the prediction 
    print("Start prediction")
    predict_models_new=[*gmt_datas] # this one further includes the ensemble number
    argus=[]
    for model in predict_models_new:
        argu=(model,predict_params,predict_years,gmt_datas[model],gmt_cum_datas[model],X_mean,X_std,Y_mean,Y_std,regress)
        argus.append(argu)
    if True: # Multiprocessing result
        pool_no=55
        pool_no=100
        pool = multiprocessing.Pool(pool_no)
        results=pool.starmap(functions_read.do_prediction,argus)
    else: # debugging mode
        results=[]
        for argu in argus:
            result=functions_read.do_prediction(*argu)
            results.append(result)
    ### Put the results into correct format
    Y_predicts={}
    for i, model in enumerate(predict_models_new):
        Y_predict=results[i]
        Y_predicts[model]=Y_predict
    Y_predict_all=[Y_predicts[model][j][year] for model in predict_models_new for j in range(predict_params.shape[0]) for year in predict_years]
    # no. of models x no. of parameters x years after reshape
    Y_predict_reshape=np.array(Y_predict_all).reshape(len(predict_models),len(predict_params),len(predict_years)) 
    print("Start prediction - Finish")
    save_Y_predict=True
    if save_Y_predict: # Save the 
        print("Save Y_predict_reshape for ISMIP6 (CMIP5 & CMIP6)")
        save_folder='/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_predict_ISMIP6/%s'%predict_param_fn
        os.makedirs(save_folder,exist_ok=True)
        for i, model in enumerate(predict_models_new):
            st_yr=predict_years[0]
            end_yr=predict_years[-1]
            save_data=Y_predict_reshape[i] # parameter no. x year
            np.save("%s/%s_%s_%s"%(save_folder,model,predict_years[0],predict_years[-1]),save_data) 
        print("Saving Done")

    ## We don't need to do history calibration here (we do calibration in the "plot_boxplots" scirpt by random drawing according to the weights
    if False: 
        ## Further reshape them into no. of model*no. of param, and then years
        Y_predict_reshape_new=np.array(Y_predict_reshape).reshape(len(predict_models)*len(predict_params),len(predict_years)) 
        Y_predict_weights=np.tile(weights,len(predict_models))
        slr_prior_qs, slr_post_qs = {}, {}
        ### get Prior
        slr_prior_qs[0.5]=np.percentile(Y_predict_reshape_new,50,axis=0)
        slr_prior_qs[0.17]=np.percentile(Y_predict_reshape_new,17,axis=0)
        slr_prior_qs[0.83]=np.percentile(Y_predict_reshape_new,83,axis=0)
        ## Posterio after weight adjustment
        slr_post_qs[0.5]=np.percentile(Y_predict_reshape_new,50,axis=0,weights=Y_predict_weights,method="inverted_cdf")
        slr_post_qs[0.17]=np.percentile(Y_predict_reshape_new,17,axis=0,weights=Y_predict_weights,method="inverted_cdf")
        slr_post_qs[0.83]=np.percentile(Y_predict_reshape_new,83,axis=0,weights=Y_predict_weights,method="inverted_cdf")

    ## Start plotting
    if True: 
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(5,2))
        ## Plot ISMIP6
        ys=[]
        models_in_legend=[]
        for i, model in enumerate(predict_models):
            #color=models_colors[model]
            color=models_colors[model]
            for j, param in enumerate(predict_params):
                ts=Y_predict_reshape[i][j] # the first parameter
                ax1.plot(predict_years,ts,color=color,alpha=0.3,zorder=0.05,lw=0.1,label=None)
                ys.append(ts)
            models_in_legend.append(model)
        ## Plot the average
        ax1.plot(predict_years,np.median(ys,axis=0),color='k',zorder=3,lw=2,label="Ensemble average (%s)"%len(ys))
        ## Create model legend
        for i, model in enumerate(predict_models):
            color=models_colors[model]
            ax1.plot([-10],[-10],color=color,zorder=2.5,lw=2,label=model)
        ## Error bar for the last year
        for i, model in enumerate(predict_models):
            slr_last=Y_predict_reshape[i,:,-1]
            median=np.median(slr_last)
            #ymin=np.min(slr_last)
            #ymax=np.max(slr_last)
            ymin=np.percentile(slr_last,1)
            ymax=np.percentile(slr_last,99)
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            ax1.errorbar(predict_years[-1]+2*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[model],fmt='_',elinewidth=2,ms=3)
            ax1.errorbar(predict_years[-1]+2*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[model],fmt='o',elinewidth=0,ms=3)
        ## Set legend
        ax1.set_ylim(-5,150); ax1.set_xlim(2015,2320)
        ax1.set_xlim(2015,2120); ax1.set_ylim(-2,20)
        ax1.set_ylim(-5,100); ax1.set_xlim(2015,2320)
        ax1.set_ylim(-5,35); ax1.set_xlim(2015,2120)
        ax1.set_ylabel('Sea-level contribution \n (cm)')
        ax1.axhline(y=0,color='k',linestyle='--',lw=0.9)
        ax1.legend(bbox_to_anchor=(0.01,0.7), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2,fontsize=9)
        ## Save
        for j in ['right', 'top']:
            ax1.spines[j].set_visible(False)
            ax1.tick_params(axis='x', which='both',length=2)
            ax1.tick_params(axis='y', which='both',length=2)
        ##
        fig_name = 'PISM_emulate_ISMIP6_GolzerEA2025'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
        ###
        ### Plot the GMT timeseries
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(5,2))
        for i, model in enumerate(predict_models):
            en=predict_models_ens[i]
            gmt_data=gmt_datas[model+"_en%s"%en]
            year=[*gmt_data]
            gmt_ts=[gmt_data[yr] for yr in year]
            ax1.plot(year,gmt_ts,color=models_colors[model],label=model)
        ##
        ax1.legend(bbox_to_anchor=(0.01,0.7), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2,fontsize=9)
        ## Save
        for j in ['right', 'top']:
            ax1.spines[j].set_visible(False)
            ax1.tick_params(axis='x', which='both',length=2)
            ax1.tick_params(axis='y', which='both',length=2)
        ##
        fig_name = 'GMT_ts_timeseries'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
