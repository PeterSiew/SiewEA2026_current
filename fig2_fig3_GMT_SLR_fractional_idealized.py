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
   
    ## Set predict models and years
    predict_param_fn='parameters_LHC_2000_8param_range' # std_obs*10 
    predict_param_fn='parameters_LHC_1000_8param_range' # std_obs*20 (is this default?)
    predict_params=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s.npy'%predict_param_fn)
    ## Reading the weights
    model_multiple=20 # this is the standard
    weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
    param_no=len(predict_params)
    predict_years=range(2015,2301)

    ### Plotting idealized GMT (Figure 5)
    plotting_fig2_gmt=True
    plotting_fig3=True
    plotting_table1=True
    plotting_tableS1_training_parameters=True
    plotting_tableS2_extended_RCPs_GCMs=True

    ### Figure 3 - Get all predicting models for RCP and CATs scenaris up to 2300
    if plotting_fig3: 
        ### Read all extended RCPs
        folder='/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/'
        file_list=os.listdir(folder)
        file_selects=[]
        models=[]
        ens=[]
        for file in file_list:
            if ('rcp' in file) & ('_nopad2300_' in file) & ('CMIP5mean' not in file):
                model='_'.join(file.split('_')[2:5])
                models.append(model)
                en=file.split('_')[5][-1]
                ens.append(en)
                file_selects.append(file) # For record (the size is 43: 8 rcp26; 23 rcp45; 2 rcp60; 10 rcp85)
        predict_models_rcps=models
        predict_models_rcps_ens=ens
        ###
        ### Get policy-driven snenarios
        #random_numbers=range(0,100) # very simular results as 500
        #cats_name=['optimistic','longpledge','shortpledge','policyhigh']
        cats_name=['optim','pledge','target','current']
        cats_full_dist={'optim':600,'pledge':1200,'target':600,'current':1200}
        #cats_random_draw={'optim':600,'pledge':1200,'target':600,'current':1200} # produce the same result as below
        cats_random_draw={'optim':300,'pledge':300,'target':300,'current':300}
        predict_en=1 # the same for all random CAT timeseries
        models=[]
        ens=[]
        for i, cat in enumerate(cats_name):
            random.seed(i)
            random_numbers=sorted(random.sample(range(cats_full_dist[cat]),cats_random_draw[cat])) # Only randomly draw 300 sample
            #print(cat,random_numbers)
            for no in random_numbers:
                #model='random%s'%no +'_'+cat+'_nopad2300'
                model='distno%s'%no +'_'+cat+'_nopad2300' # only use the no pad one
                models.append(model)
                ens.append(predict_en)
        predict_models_cats=models
        predict_models_cats_ens=ens
        ## Comebine the predict models for RCPs and CATs
        predict_models=predict_models_rcps+predict_models_cats
        predict_models_ens=predict_models_rcps_ens+predict_models_cats_ens
        ## Others
        models_colors=['royalblue','lightskyblue','lightsalmon','orangered']

    #ipdb.set_trace()

    ### Plot the GCM table S2 (only for extended RCPs)
    if plotting_tableS2_extended_RCPs_GCMs:
        reload(fig23); fig23.create_tableS2(predict_models_rcps,predict_models_rcps_ens)

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

    ### Create table and plots
    plotting_training_domain_supp=False
    if plotting_training_domain_supp:
        reload(fig23); fig23.plotting_training_domain(Xs,Ys,records)
    ### Plot the parameter table S1
    if plotting_tableS1_training_parameters:
        reload(fig23);fig23.create_training_params_tableS1(pism_slrs)

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
    if True: # Do the Rdige Regression 
        regress = Ridge(alpha=700,fit_intercept=True).fit(X_train_standard,Y_train_standard) # the optimal alpha=700 based on figS1
        print(regress.coef_, regress.intercept_)
    else: # Do the ANN - but without loops
        from sklearn.neural_network import MLPRegressor
        solver='adam'; ann_shuffle=True; ann_nodes=(10,10) 
        #ann_batch_size=20; l2_coeff=0.001; learn_rate=0.05 # This largely underestimates in RCP8.5 case (this should not be used)
        ann_batch_size=32; l2_coeff=0.0001; learn_rate=0.01 # This correspoinds to Figure S2 for "leave-one-simulation-out"
        loops=range(0,5)
        regress={}
        for loop in loops:
            reg_loop=MLPRegressor(hidden_layer_sizes=ann_nodes, learning_rate_init=learn_rate,solver=solver,early_stopping=True,n_iter_no_change=50,
                    learning_rate='constant',batch_size=ann_batch_size,max_iter=1500, alpha=l2_coeff, shuffle=ann_shuffle, validation_fraction=0.2,
                    tol=1e-6,verbose=False,random_state=loop).fit(X_train_standard,Y_train_standard)
            regress[loop]=reg_loop
    print("Start training - Finish")

    ### Read GMT for predict models (make sure the model name includes the en - some model has en1 to en4., CSIRO-Mk3-6-0_rcp85)
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

    if plotting_fig2_gmt:
        rcp_models_label=['RCP26','RCP45','RCP60','RCP85']
        #cat_models_label=['Optimistic','Long-term pledge','Short-term target','Current actions']
        cat_models_label=['Optimistic','Pledges and targets','2030 & 2035 targets','Current policy actions']
        reload(fig23);fig23.plotting_figure2_gmt(predict_models_rcps,predict_models_cats,models_colors,rcp_models_label,cat_models_label,gmt_datas,gmt_cum_datas)

    ### Do the prediction using the trained models (regress) via multi-processing
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
        #results=pool.starmap(functions_read.do_prediction_ANN,argus)
    else: # debugging mode
        results=[]
        for argu in argus:
            result=functions_read.do_prediction(*argu)
            #result=functions_read.do_prediction_ANN(*argu)
            results.append(result)
    ### Put the results into correct format
    Y_predicts={}
    for i, model in enumerate(predict_models_new):
        Y_predict=results[i]
        Y_predicts[model]=Y_predict
    Y_predict_all=[Y_predicts[model][j][year] for model in predict_models_new for j in range(predict_params.shape[0]) for year in predict_years]
    Y_predict_reshape=np.array(Y_predict_all).reshape(len(predict_models_new),predict_params.shape[0],len(predict_years)) # no. of models x no. of parameters x years
    print("Start prediction - Finish")

    ### Seperate the resulting data into groups (4 RCPs and 4 CATs)
    ### This part should also go to multi-processing to save time
    ## the weight used to define prior and posteria
    rcps=['rcp26','rcp45','rcp60','rcp85']
    #cats=['optimistic','longpledge','shortpledge','policyhigh']
    cats=['optim','pledge','target','current']
    rcps_cats=rcps+cats
    Y_predict_groups={group:[] for group in rcps_cats}
    Y_predict_weights={group:[] for group in rcps_cats}
    for i, model in enumerate(predict_models):
        group=model.split('_')[1]
        Y_predict_groups[group].append(Y_predict_reshape[i])
    Y_predict_groups_new={group:[] for group in rcps_cats}
    for group in rcps_cats:
        Y_predict_groups[group]=np.array(Y_predict_groups[group]) # [group(e.g.,rcp26')] has a size of GCMs,params,years
        ## Further reshape them into no. of model*no. of param, and then years
        Y_predict_groups_new[group]=np.array(Y_predict_groups[group]).reshape(len(Y_predict_groups[group])*predict_params.shape[0],len(predict_years)) 
        Y_predict_weights[group]=np.tile(weights,Y_predict_groups[group].shape[0])
        #Y_predict_weights[group]=np.vstack(([weights]*Y_predict_groups[group].shape[0])).reshape(-1) # Same as above line
    save_Y_predict=True
    save_Y_predict=False
    if save_Y_predict: ### Save the Y_predict_file (prior - no weight adjustment) for testing the posteria for different mm
        print("Save Y_predict_main_figure")
        # the Y_predict_groups_new can also be saved if you want
        np.save("/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_predict_fig3",Y_predict_groups) 
        print("Saving Done")
    ######
    ### Calculate the prior and posteri - mean, median and percentiles of the data (without KDE) for creating Figure 3 - main figure
    slr_prior_qs={group:{} for group in rcps_cats}
    slr_post_qs={group:{} for group in rcps_cats}
    for group in rcps_cats: 
        print(group)
        print("Getting the percentile with and without history matching")
        ## Prior
        #slr_prior_qs[group]['mean']=Y_predict_groups_new[group].mean(axis=0)
        #slr_prior_qs[group]['variance']=Y_predict_groups_new[group].std(axis=0)**2
        #slr_prior_qs[group][0.1]=np.percentile(Y_predict_groups_new[group],10,axis=0)
        #slr_prior_qs[group][0.9]=np.percentile(Y_predict_groups_new[group],90,axis=0)
        slr_prior_qs[group][0.5]=np.percentile(Y_predict_groups_new[group],50,axis=0)
        slr_prior_qs[group][0.17]=np.percentile(Y_predict_groups_new[group],17,axis=0)
        slr_prior_qs[group][0.83]=np.percentile(Y_predict_groups_new[group],83,axis=0)
        ## Posterio after weight adjustment
        #slr_post_qs[group]['mean']=np.average(Y_predict_groups_new[group],axis=0,weights=Y_predict_weights[group])
        #slr_post_qs[group]['variance']=Y_predict_groups_new[group].std(axis=0)**2
        #slr_post_qs[group][0.1]=np.percentile(Y_predict_groups_new[group],10,axis=0,weights=Y_predict_weights[group],method="inverted_cdf")
        #slr_post_qs[group][0.9]=np.percentile(Y_predict_groups_new[group],90,axis=0,weights=Y_predict_weights[group],method="inverted_cdf")
        slr_post_qs[group][0.5]=np.percentile(Y_predict_groups_new[group],50,axis=0,weights=Y_predict_weights[group],method="inverted_cdf")
        slr_post_qs[group][0.17]=np.percentile(Y_predict_groups_new[group],17,axis=0,weights=Y_predict_weights[group],method="inverted_cdf")
        slr_post_qs[group][0.83]=np.percentile(Y_predict_groups_new[group],83,axis=0,weights=Y_predict_weights[group],method="inverted_cdf")

    ###
    ###
    if plotting_table1:
        reload(fig23);fig23.plotting_table1(slr_prior_qs,slr_post_qs,rcp_models_label,cat_models_label,rcps,cats,predict_years)

    ###
    ### Do the plotting of Figure 3 - the SLR main figure
    if True & plotting_fig3:
        plt.close()
        #fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,2.5))
        fig,ax_all=plt.subplots(3,2,figsize=(8,7),height_ratios=[3,2,2])
        ax1,ax2,ax3,ax4,ax5,ax6=ax_all.flatten()
        ## for RCPS projection (ax1)
        max_q=0.83; min_q=0.17
        for i, model in enumerate(rcps):
            slr_median=slr_post_qs[model][0.5][:]
            #slr_mean=slr_post_qs[model]['mean'][:]
            slr_max=slr_post_qs[model][max_q][:]
            slr_min=slr_post_qs[model][min_q][:]
            ax1.plot(predict_years,slr_median,color=models_colors[i],lw=2,zorder=3,label=rcp_models_label[i]) # mean
            ax1.fill_between(predict_years,slr_min,slr_max,fc=models_colors[i],zorder=1,color=models_colors[i], alpha=0.3, linewidth=0)
            ## Error bar for the last year
            median=slr_median[-1]
            #mean=slr_mean[-1]
            ymin=slr_min[-1]
            ymax=slr_max[-1]
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            ax1.errorbar(predict_years[-1]+10*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='_',elinewidth=2,ms=3)
            ax1.errorbar(predict_years[-1]+10*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='o',elinewidth=0,ms=3)
            ## Prior next to the posteri
            median=slr_prior_qs[model][0.5][:][-1]
            #mean=slr_prior_qs[model]['mean'][:][-1]
            ymax=slr_prior_qs[model][max_q][:][-1]
            ymin=slr_prior_qs[model][min_q][:][-1]
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            eb_alpha=0.3
            ax1.errorbar(predict_years[-1]+10*(i+1)-5,median,yerr=[[yerr_min],[yerr_max]],color='gray',fmt='_',elinewidth=2,ms=3,alpha=eb_alpha)
            ax1.errorbar(predict_years[-1]+10*(i+1)-5,median,yerr=[[yerr_min],[yerr_max]],color='gray',fmt='o',elinewidth=0,ms=3,alpha=eb_alpha)
        ## For CATs projection (ax2)
        for i, model in enumerate(cats):
            slr_median=slr_post_qs[model][0.5][:]
            #slr_mean=slr_post_qs[model]['mean'][:]
            slr_max=slr_post_qs[model][max_q][:]
            slr_min=slr_post_qs[model][min_q][:]
            ax2.plot(predict_years,slr_median,color=models_colors[i],lw=2,zorder=3,label=cat_models_label[i])
            ax2.fill_between(predict_years,slr_min,slr_max,fc=models_colors[i],zorder=1,color=models_colors[i], alpha=0.3, linewidth=1, edgecolor='none')
            ## Error bar for the last year
            median=slr_median[-1]
            #mean=slr_mean[-1]
            ymin=slr_min[-1]
            ymax=slr_max[-1]
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            ax2.errorbar(predict_years[-1]+10*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='_',elinewidth=2,ms=3)
            ax2.errorbar(predict_years[-1]+10*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='o',elinewidth=0,ms=3)
            ## Prior next to the posteri
            median=slr_prior_qs[model][0.5][:][-1]
            #mean=slr_prior_qs[model]['mean'][:][-1]
            ymax=slr_prior_qs[model][max_q][:][-1]
            ymin=slr_prior_qs[model][min_q][:][-1]
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            ax2.errorbar(predict_years[-1]+10*(i+1)-5,median,yerr=[[yerr_min],[yerr_max]],color='gray',fmt='_',elinewidth=2,ms=3,alpha=eb_alpha)
            ax2.errorbar(predict_years[-1]+10*(i+1)-5,median,yerr=[[yerr_min],[yerr_max]],color='gray',fmt='o',elinewidth=0,ms=3,alpha=eb_alpha)
        for i, ax in enumerate([ax1,ax2]):
            ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]); ax.set_ylim(0,130)
            #ax.set_yticks([0,10,20,30,40,50,60,70,80,90]); ax2.set_ylim(0,80)
            ax.set_xlim(predict_years[0],predict_years[-1]+45)
            ax.legend(bbox_to_anchor=(0.01,0.6), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2, reverse=True)
            if i==0:
                ax.set_ylabel('Sea-level contribution (cm)')
            if i==1:
                ax.set_yticklabels([])
        ## Set axis title
        ax1.annotate(r'$\bf({A})$',xy=(-0.17,1.07),xycoords='axes fraction', fontsize=11)
        ax1.set_title('Extended RCP Scenarios')
        ax2.annotate(r'$\bf({D})$',xy=(-0.08,1.07),xycoords='axes fraction', fontsize=11)
        ax2.set_title('Climate Policy Actions')
    ### (middle panel - ax3, ax4)
    ### Compute the variance of each component and fractional variance)
    if True:
        ## 1) Get the varaince of RCPs/CATs - i) mean across GCMs in each RCP, ii) weighted mean across parameters in each RCP iii) std across RCPs
        ## 2) Get the variance across parameters - i) mean across GCMs in each RCP ii) Mean across RCPs iii) weighted std across parameters
        ## 3) Get the varaiance across GCMs - i) std across GCMs first, ii) weighted mean across parameters iii) mean across RCPs
        ## 3) This can also be i) Weighted mean across parameters, std across GCMs, mean across RCPs (this is surprisinly the same as before)
        ## With history matching (has weight adjustment) - default
        ## RCPs
        rcps_scenario_std=np.array([np.average(Y_predict_groups[rcp].mean(axis=0),weights=weights,axis=0) for rcp in rcps]).std(axis=0)
        values=np.array([Y_predict_groups[rcp].mean(axis=0) for rcp in rcps]).mean(axis=0) # mean across GCMs and then RCP (shape is param x years)
        values_mean=np.average(values,weights=weights,axis=0)
        rcps_param_std=np.sqrt(np.average((values-values_mean)**2,weights=weights,axis=0))
        rcps_gcm_std=np.array([np.average(Y_predict_groups[rcp].std(axis=0),weights=weights,axis=0) for rcp in rcps]).mean(axis=0)
        #rcps_gcm_std=np.array([np.average(Y_predict_groups[rcp],axis=1,weights=weights).std(axis=0) for rcp in rcps]).mean(axis=0)
        ## CATs
        cats_scenario_std=np.array([np.average(Y_predict_groups[cat].mean(axis=0),weights=weights,axis=0) for cat in cats]).std(axis=0)
        values=np.array([Y_predict_groups[cat].mean(axis=0) for cat in cats]).mean(axis=0) # mean across GCMs and then RCP (shape is param x years)
        values_mean=np.average(values,weights=weights,axis=0)
        cats_param_std=np.sqrt(np.average((values-values_mean)**2,weights=weights,axis=0))
        cats_gcm_std=np.array([np.average(Y_predict_groups[cat].std(axis=0),weights=weights,axis=0) for cat in cats]).mean(axis=0)
        ### The parameter std without history matching (for the grey shading in Fig. 4A bars)
        rcps_param_std_no_histmatch=np.array([Y_predict_groups[rcp].mean(axis=0) for rcp in rcps]).mean(axis=0).std(axis=0)
        cats_param_std_no_histmatch=np.array([Y_predict_groups[cat].mean(axis=0) for cat in cats]).mean(axis=0).std(axis=0)
        green_color='forestgreen'
        if False: # For plotting the supp - fractioanl variance without history matching (gray instead of green)
            rcps_param_std=rcps_param_std_no_histmatch
            cats_param_std=cats_param_std_no_histmatch
            green_color='lightgray'
        ### Calculate the variances and realtive variance of each component
        ## For RCPs
        rcps_total_var = rcps_scenario_std**2+ rcps_param_std**2 + rcps_gcm_std**2
        rcps_scenario_frac = (rcps_scenario_std**2/rcps_total_var)*100
        rcps_param_frac = (rcps_param_std**2/rcps_total_var)*100
        rcps_gcm_frac = (rcps_gcm_std**2/rcps_total_var)*100
        ## For CATs
        cats_total_var = cats_scenario_std**2+cats_param_std**2 + cats_gcm_std**2
        cats_scenario_frac = (cats_scenario_std**2/cats_total_var)*100
        cats_param_frac = (cats_param_std**2/cats_total_var)*100
        cats_gcm_frac = (cats_gcm_std**2/cats_total_var)*100
        ###
        ### Start the figure
        #fig = plt.figure(figsize=(7,5)); #ax1 = plt.subplot(2,1,1); #ax2 = plt.subplot(2,2,3); #ax3 = plt.subplot(2,2,4)
        years_sel=[2050,2100,2150,2200,2250,2300]
        yr_idx=[predict_years.index(yr) for yr in years_sel]
        ## RCPS
        gcm_1=(rcps_gcm_std**2)[yr_idx]
        scenario_1=(rcps_scenario_std**2)[yr_idx]
        param_1=(rcps_param_std**2)[yr_idx]
        param_1_no_histmatch=(rcps_param_std_no_histmatch**2)[yr_idx]
        ## CATS
        gcm_2=np.array(cats_gcm_std**2)[yr_idx]
        scenario_2=np.array(cats_scenario_std**2)[yr_idx]
        param_2=np.array(cats_param_std**2)[yr_idx]
        param_2_no_histmatch=np.array(cats_param_std_no_histmatch**2)[yr_idx]
        ## For RCPs (plotting ax3)
        bar_width=0.5
        x_adj=bar_width
        x=np.arange(len(years_sel))
        x_adj=0
        hatch='xxx'
        hatch=None
        ax3.bar(x-x_adj*0.5,gcm_1,bar_width,color='royalblue',edgecolor='k',hatch=hatch,lw=0,label='Climate model forcings')
        ax3.bar(x-x_adj*0.5,scenario_1,bar_width,bottom=gcm_1,color='orange',edgecolor='k',hatch=hatch,lw=0,label='RCP scenarios')
        ax3.bar(x-x_adj*0.5,param_1,bar_width,bottom=gcm_1+scenario_1,edgecolor='k',hatch=hatch,color=green_color,lw=0,label='Ice-sheet model parameters\n(with history-calibration)')
        ax3.bar(x-x_adj*0.5,param_1_no_histmatch,bar_width,bottom=gcm_1+scenario_1+param_1,edgecolor='k',hatch=hatch,color='lightgray',lw=0,label='Ice-sheet model parameters\n(without history-calibration)')
        #ax3.bar([-100],[-100],bar_width,edgecolor='k',hatch=hatch,color='white',label='Projection - RCP emission scenarios (left)',lw=0)
        ## For CATs (plotting ax4)
        ax4.bar(x+x_adj*0.5,gcm_2,bar_width,color='royalblue',edgecolor='k',hatch=hatch,lw=0,label='GMT predictions')
        ax4.bar(x+x_adj*0.5,scenario_2,bar_width,bottom=gcm_2,color='orange',edgecolor='k',hatch=hatch,lw=0,label='Policy actions')
        ax4.bar(x+x_adj*0.5,param_2,bar_width,bottom=gcm_2+scenario_2,color=green_color,edgecolor='k',hatch=hatch,lw=0,label='Ice-sheet model parameters\n(with history-calibration)')
        ax4.bar(x+x_adj*0.5,param_2_no_histmatch,bar_width,bottom=gcm_2+scenario_2+param_2,edgecolor='k',hatch=hatch,color='lightgray',lw=0,label='Ice-sheet model parameters\n(without history-calibration)')
        #ax4.bar([-100],[-100],bar_width,edgecolor='k',hatch=hatch,color='white',label='Prediction - Policy actions (right)',lw=0)
        for i, ax in enumerate([ax3,ax4]):
            ax.set_ylim(0,1000)
            ax.set_xlim(x[0]-0.5,x[-1]+0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(years_sel)
            if i==0:
                #ax.set_ylabel('Variance (cm)')
                ax.set_ylabel(r'Variance (cm$^{2}$)')
            if i==1:
                ax.set_yticklabels([])
            frame=ax.legend(bbox_to_anchor=(-0.01,0.99), ncol=1, loc='upper left', frameon=False, columnspacing=0.2,
                                handletextpad=0.4, labelspacing=0.5, fontsize=8.5, reverse=True)
        ax3.annotate(r'$\bf({B})$',xy=(-0.17,1.07),xycoords='axes fraction', fontsize=11)
        ax4.annotate(r'$\bf({E})$',xy=(-0.06,1.07),xycoords='axes fraction', fontsize=11)
    ###
    ### the loweest panel - (ax5 and ax6), fractional uncertainty
    if True: 
        ## RCPs: ax5
        ## Plot GCM fraction
        base=0; top=rcps_gcm_frac
        ax5.fill_between(predict_years, base, top, fc='royalblue',color='royalblue',label='Climate model forcing')
        ## Plot RCP fraction
        base=top; top=base+rcps_scenario_frac
        ax5.fill_between(predict_years, base, top, fc='orange',color='orange',label='RCP emission scenarios')
        ## Plot parameter fraction
        base=top; top=base+rcps_param_frac
        ax5.fill_between(predict_years, base, top, fc=green_color,color=green_color,label='PISM parameters')
        ## CATs: ax6
        ## Plot GCM fraction
        base=0; top=cats_gcm_frac
        ax6.fill_between(predict_years, base, top, fc='royalblue',color='royalblue',label='GMT predictions')
        ## Plot CAT scenario fraction
        base=top; top=base+cats_scenario_frac
        ax6.fill_between(predict_years, base, top, fc='orange',color='orange',label='Policy actions')
        ## Plot parameter fraction
        base=top; top=base+cats_param_frac
        ax6.fill_between(predict_years, base, top, fc=green_color,color=green_color,label='PISM parameters')
        for i, ax in enumerate([ax5,ax6]):
            ax.set_xlim(predict_years[0],predict_years[-1])
            ax.set_ylim(0,100)
            if i==0:
                ax.set_ylabel('Fractional\nvariance (%)')
            if i==1:
                ax.set_yticklabels([])
        ax5.annotate(r'$\bf({C})$', xy=(-0.17,1.07),xycoords='axes fraction', fontsize=11)
        ax6.annotate(r'$\bf({F})$' ,xy=(-0.05,1.07),xycoords='axes fraction', fontsize=11)
        ## For all panels together
        for ax in ax_all.flatten():
            #ax.axvline(x=2015,color='k',linestyle='--',lw=0.9)
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        ## Save figure
        fig_name = 'fig3_main_figure'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.3) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

    ###
    ###
    ###
    print_everything=True
    if print_everything:
        print("Start to print valused used in the manuscript")
        if True: ## Print the percentage change from history-calibration to no history-calibration for four-RCP secenarios (averaged) in 2300
            percentage_changes=[]
            for rcp in ['rcp85','rcp60','rcp45','rcp26']:
                old=slr_prior_qs[rcp][0.83][-1]-slr_prior_qs[rcp][0.17][-1]
                new=slr_post_qs[rcp][0.83][-1]-slr_post_qs[rcp][0.17][-1]
                perc_change=(new-old)/old*100
                percentage_changes.append(perc_change)
            print("the 2300-SLE pecentage change from prior  to posteri for 17-83 pecentage averaged over the four RCPs is  %s"%np.mean(percentage_changes))
            print("")
        if True: # Print the fractional uncertainty of RCP scnearios in years 2050, 2100 and 2300
            years_sel=[2050,2100,2300]
            yr_idx=[predict_years.index(yr) for yr in years_sel]
            print('RCP scenario uncertainty in 2050 is: ',rcps_scenario_frac[yr_idx[0]])
            print('RCP scenario uncertainty in 2100 is: ',rcps_scenario_frac[yr_idx[1]])
            print('RCP scenario uncertainty in 2300 is: ',rcps_scenario_frac[yr_idx[2]])
            print("")
        if True: # Print the total variance decrease from RCP to CAT
            years_sel=[2100,2200,2300]
            yr_idx=[predict_years.index(yr) for yr in years_sel]
            percentage_change=(cats_total_var[yr_idx]-rcps_total_var[yr_idx])/rcps_total_var[yr_idx]*100
            print("the total variance changes (in percentage) from RCPs to CATs in years %s are %s"%(years_sel, percentage_change))
            print("")
        if True: ## The the CATs fractions in 2300
            print('CAT-GMT prediction,fractional uncertainty in 2300 is: ', cats_gcm_frac[-1])
            print('CAT-Policy action,fractional uncertainty in 2300 is: ', cats_scenario_frac[-1])
            print('CAT-PISM parameter,fractional uncertainty in 2300 is:', cats_param_frac[-1])
            print("")



def plotting_table1(slr_prior_qs,slr_post_qs,rcp_models_label,cat_models_label,rcps,cats,predict_years):

    from matplotlib.font_manager import FontProperties
    import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
    idx2100=predict_years.index(2100)
    idx2200=predict_years.index(2200)
    idx2300=predict_years.index(2300)
    df = pd.DataFrame()
    if True: # Reverse the order for RCP and CAT
        rcps=rcps[::-1]
        cats=cats[::-1]
        rcp_models_label=rcp_models_label[::-1]
        cat_models_label=cat_models_label[::-1]
    df['Year']=["Extended RCP Scenarios"]+rcp_models_label+['Climate Policy Actions']+cat_models_label
    #ipdb.set_trace()
    ## 2100
    df['2100']=['']+[str(round(slr_post_qs[rcp][0.17][idx2100],1))+', '+str(round(slr_post_qs[rcp][0.5][idx2100],1))+', '+str(round(slr_post_qs[rcp][0.83][idx2100],1))+"\n" +"("+ str(round(slr_prior_qs[rcp][0.17][idx2100],1))+", "+str(round(slr_prior_qs[rcp][0.5][idx2100],1))+", "+str(round(slr_prior_qs[rcp][0.83][idx2100],1))+")" for rcp in rcps] \
    +['']+[str(round(slr_post_qs[cat][0.17][idx2100],1))+', '+str(round(slr_post_qs[cat][0.5][idx2100],1))+', '+str(round(slr_post_qs[cat][0.83][idx2100],1))+"\n" +"("+ str(round(slr_prior_qs[cat][0.17][idx2100],1))+", "+str(round(slr_prior_qs[cat][0.5][idx2100],1))+", "+str(round(slr_prior_qs[cat][0.83][idx2100],1))+")" for cat in cats] 
    ## 2200
    df['2200']=['']+[str(round(slr_post_qs[rcp][0.17][idx2200],1))+', '+str(round(slr_post_qs[rcp][0.5][idx2200],1))+', '+str(round(slr_post_qs[rcp][0.83][idx2200],1))+"\n" +"("+ str(round(slr_prior_qs[rcp][0.17][idx2200],1))+", "+str(round(slr_prior_qs[rcp][0.5][idx2200],1))+", "+str(round(slr_prior_qs[rcp][0.83][idx2200],1))+")" for rcp in rcps] \
    +['']+[str(round(slr_post_qs[cat][0.17][idx2200],1))+', '+str(round(slr_post_qs[cat][0.5][idx2200],1))+', '+str(round(slr_post_qs[cat][0.83][idx2200],1))+"\n" +"("+ str(round(slr_prior_qs[cat][0.17][idx2200],1))+", "+str(round(slr_prior_qs[cat][0.5][idx2200],1))+", "+str(round(slr_prior_qs[cat][0.83][idx2200],1))+")" for cat in cats] 
    ## 2300
    df['2300']=['']+[str(round(slr_post_qs[rcp][0.17][idx2300],1))+', '+str(round(slr_post_qs[rcp][0.5][idx2300],1))+', '+str(round(slr_post_qs[rcp][0.83][idx2300],1))+"\n" +"("+ str(round(slr_prior_qs[rcp][0.17][idx2300],1))+", "+str(round(slr_prior_qs[rcp][0.5][idx2300],1))+", "+str(round(slr_prior_qs[rcp][0.83][idx2300],1))+")" for rcp in rcps] \
    +['']+[str(round(slr_post_qs[cat][0.17][idx2300],1))+', '+str(round(slr_post_qs[cat][0.5][idx2300],1))+', '+str(round(slr_post_qs[cat][0.83][idx2300],1))+"\n" +"("+ str(round(slr_prior_qs[cat][0.17][idx2300],1))+", "+str(round(slr_prior_qs[cat][0.5][idx2300],1))+", "+str(round(slr_prior_qs[cat][0.83][idx2300],1))+")" for cat in cats] 
    ###
    ## Create the table
    plt.close()
    fig = plt.figure(figsize=(7,4))
    ax=fig.gca()
    ax.axis('off')
    r,c = df.shape
    # ensure consistent background color
    #ax.table(cellColours=[['lightgray']] + [['none']], bbox=[0,0,1,1])
    #ax.table(cellColours=[['gray']] + [['green']]+[['black']], bbox=[0,0,1,1])
    # plot the real table
    col_widths=[0.35,0.2,0.2,0.2]
    table = ax.table(cellText=np.vstack([df.columns, df.values]), 
                     bbox=[0, 0, 1, 1], cellLoc='center', rowLoc='center',colWidths=col_widths)
    # need to draw here so the text positions are calculated
    #fig.canvas.draw()
    ## Do merge cells
    reload(fig23)
    fig23.mergecells1(table, [(1,1), (1,0), (1,2), (1,3)])
    fig23.mergecells1(table, [(6,0), (6,1), (6,2), (6,3)])
    ## For particular cells
    #row,col=list(table.properties()['celld'].keys())[-1]
    ## Seth the width
    #for i in range(0,10):
    #    table.properties()["celld"][i,0].set_width(0.3)
    ## set the font position of a particular cell
    table.properties()["celld"][1,0].set_text_props(ha="left")
    table.properties()["celld"][6,0].set_text_props(ha="left")
    #table.get_celld()[(1,0)]._text.set_horizontalalignment('left')
    ## Bold the cell
    table.get_celld()[(1,0)].set_text_props(fontproperties=FontProperties(weight='bold'))
    table.get_celld()[(6,0)].set_text_props(fontproperties=FontProperties(weight='bold'))
    #table.properties()["celld"][1,0]._loc='left'
    ## Set colors
    #table[(1, 0)].set_facecolor("lightgray")
    ## Set fontsize
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    #table.auto_set_column_width(col=list(range(len(df.columns))))
    #table.auto_set_column_width(col=[0])
    #table.auto_set_row_height(range(10))
    ### Save figures
    fig_name = 'table1_fig3_slr_values'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)


def create_training_params_tableS1(pism_slrs):

    import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
    np.set_printoptions(suppress=True)

    ### This is the table S1

    models_name=['MIROC5 RCP85','MIROC5 RCP26', 'MIROC5 RCP2685', 'MIROC5 RCP85-cooling']

    models=[] # 1st column
    ensembles=[]
    phimins=[]
    siaas=[]
    ddf_ices=[]
    ddf_snows=[]
    t_thresholds=[]
    t_stds=[]
    refreeze_percents=[]
    ssaes=[]
    for i, model in enumerate([*pism_slrs]):
        ## 1st column (model)
        models.append(models_name[i])
        param_size=pism_slrs[model].params.size
        ## 2nd column (ensemble numbers)
        new_item='\n'.join([str(i+1) for i in range(param_size)])
        ensembles.append(new_item)
        ## 3rd column (1st param: phi_min)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[0]),1)) for i in range(0,param_size)]
        #aaa=[pism_slrs[model].params[i].item().split('-')[0] for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        phimins.append(new_item)
        ## 4th column (2st param:SIA_A)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[1]),6)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        siaas.append(new_item)
        ## 5th column (3rd param:DDF_ice)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[2]),4)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        ddf_ices.append(new_item)
        ## 6th column (4rd param:DDF_snow)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[3]),4)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        ddf_snows.append(new_item)
        ## 7th column (5rd param:T melt threshold)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[4]),1)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        t_thresholds.append(new_item)
        ## 8th column (6rd param:T std)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[5]),2)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        t_stds.append(new_item)
        ## 9th column (7rd param:Refreeze percentage)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[6]),2)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        refreeze_percents.append(new_item)
        ## 10th column (8rd param:Refreeze percentage)
        aaa=[str(round(float(pism_slrs[model].params[i].item().split('-')[7]),2)) for i in range(0,param_size)]
        new_item='\n'.join(aaa)
        ssaes.append(new_item)


    df = pd.DataFrame()
    df['Models']=models
    df['Ensembles']=ensembles
    ### PDD
    df['DDF_ice']=ddf_ices
    df['DDF_snow']=ddf_snows
    df['T_thresholds']=t_thresholds
    df['T_stds']=t_stds
    df['Refreeze_ratio']=refreeze_percents
    ### Ice-flow
    df['Phimins']=phimins
    df['SIA_a']=siaas
    df['SSA_e']=ssaes


    column_names={'Models':'PISM forced by',
                'Ensembles':'Ensemble\nmembers',
                ##
                'DDF_ice':r'$K_{ice}$',
                'DDF_snow':r'$K_{snow}$',
                'T_thresholds':r'$T_{melt}$',
                'T_stds':r'$T_{std}$',
                'Refreeze_ratio':r'$R_{refreeze}$',
                ##
                'Phimins':r'$\phi_{min}$',
                'SIA_a':r'$E_{SIA}$',
                'SSA_e':r'$E_{SSA}$'}
    new_column_names=[column_names[name] for name in df.columns]

    plt.close()
    fig = plt.figure(figsize=(19,34))
    ax=fig.gca()
    ax.axis('off')
    r,c = df.shape
    # ensure consistent background color
    #ax.table(cellColours=[['lightgray']] + [['none']], bbox=[0,0,1,1])
    #ax.table(cellColours=[['gray']] + [['green']]+[['black']], bbox=[0,0,1,1])
    # plot the real table
    #col_widths=[0.35,0.2,0.2,0.2]
    #table = ax.table(cellText=np.vstack([df.columns, df.values]), 
    #                 bbox=[0, 0, 1, 1], cellLoc='center', rowLoc='center')
    table = ax.table(cellText=np.vstack([new_column_names, df.values]), 
                     bbox=[0, 0, 1, 1], cellLoc='center', rowLoc='center')
    cell_dict = table.get_celld()
    row,col=[*table.get_celld()][-1]
    #row,col=[*table.properties()['celld']][-1]
    for i in range(row+1):
        for j in range(col+1):
            if i==0:
                cell_dict[(i, j)].set_height(0.1)
            if i==1:
                cell_dict[(i, j)].set_height(1.5)
            if i==2:
                cell_dict[(i, j)].set_height(0.4)
            if i==3:
                cell_dict[(i, j)].set_height(0.35)
            if i==4:
                cell_dict[(i, j)].set_height(0.26)

    ## Set fontsize
    #table.auto_set_font_size(True)
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    #table.auto_set_column_width(col=[0])
    ### Save figures
    fig_name = 'tableS1_training_parameters'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

def create_tableS2(predict_models,predict_models_ens):

    ### List of GCM for RCP predictions (Table S2)

    import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"

    scenarios=['CMIP5 Extended\nRCP scenarios','RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5']
    models_rcp26=''
    models_rcp45=''
    models_rcp60=''
    models_rcp85=''
    models_rcp26_no=0
    models_rcp45_no=0
    models_rcp60_no=0
    models_rcp85_no=0

    ### Remove the duplicate and count the number
    models_counts=[]
    predict_models_new=sorted(list(set(predict_models)))
    for i, model in enumerate(predict_models_new):
        count=predict_models.count(model)
        models_counts.append(str(count))

    for i, model in enumerate(predict_models_new):
        if 'rcp26' in model:
            models_rcp26=models_rcp26+"\n"+model.split('_')[0]+" ("+models_counts[i]+")"
            models_rcp26_no=models_rcp26_no+int(models_counts[i])
        if 'rcp45' in model:
            models_rcp45=models_rcp45+"\n"+model.split('_')[0]+" ("+models_counts[i]+")"
            models_rcp45_no=models_rcp45_no+int(models_counts[i])
        if 'rcp60' in model:
            models_rcp60=models_rcp60+"\n"+model.split('_')[0]+" ("+models_counts[i]+")"
            models_rcp60_no=models_rcp60_no+int(models_counts[i])
        if 'rcp85' in model:
            models_rcp85=models_rcp85+"\n"+model.split('_')[0]+" ("+models_counts[i]+")"
            models_rcp85_no=models_rcp85_no+int(models_counts[i])
    models_names=np.array(['Models\n(number of\nensemble members)',models_rcp26,models_rcp45,models_rcp60,models_rcp85])
    models_number=np.array(['Total\nnumbers',models_rcp26_no,models_rcp45_no,models_rcp60_no,models_rcp85_no])
    #ipdb.set_trace()

    ### Start plotting
    plt.close()
    fig = plt.figure(figsize=(3,6))
    ### Create table
    ax=fig.gca()
    ax.axis('off')
    table = ax.table(cellText=np.column_stack([scenarios, models_names, models_number]), 
                     bbox=[0, 0, 1, 1], cellLoc='center', rowLoc='center')
    cell_dict = table.get_celld()
    row,col=[*table.get_celld()][-1]
    ### Set row height
    if True: 
        for i in range(row+1):
            for j in range(col+1):
                if i==0:
                    cell_dict[(i, j)].set_height(0.1)
                if i==1:
                    cell_dict[(i, j)].set_height(0.22)
                if i==2:
                    cell_dict[(i, j)].set_height(0.32)
                if i==3:
                    cell_dict[(i, j)].set_height(0.08)
                if i==4:
                    cell_dict[(i, j)].set_height(0.22)
    ### Set font size
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(col+1)))
    ### Save figures
    fig_name = 'tableS2_RCP_GCMs'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

def mergecells1(table, cells):

    import matplotlib as mpl
    '''
    Merge N matplotlib.Table cells

    Parameters
    -----------
    table: matplotlib.Table
        the table
    cells: list[set]
        list of sets od the table coordinates
        - example: [(0,1), (0,0), (0,2)]

    Notes
    ------
    https://stackoverflow.com/a/53819765/12684122
    '''
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i+1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i+1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda v: v[1]))
        edges = ['BTL'] + ['BT' for i in range(len(cells) - 2)] + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda h: h[0]))
        edges = ['TRL'] + ['RL' for i in range(len(cells) - 2)] + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        table[cell[0], cell[1]].visible_edges = e

    txts = [table[cell[0], cell[1]].get_text() for cell in cells]
    tpos = [np.array(t.get_position()) for t in txts]

    # transpose the text of the left cell
    trans = (tpos[-1] - tpos[0])/2
    # didn't had to check for ha because I only want ha='center'
    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))
    for txt in txts[1:]:
        txt.set_visible(False)



def plotting_figure2_gmt(predict_models_rcps,predict_models_cats,models_colors,rcp_models_label,cat_models_label,gmt_datas,gmt_cum_datas):

    import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"

    models=[*gmt_datas]
    rcps=['rcp26','rcp45','rcp60','rcp85']
    #cats=['optimistic','longpledge','shortpledge','policyhigh']
    cats=['optim','pledge','target','current']
    groups=rcps+cats
    gmt_data_groups={group:{} for group in groups}
    for model in models:
        if 'rcp26' in model:
            gmt_data_groups['rcp26'][model]=gmt_datas[model]
        elif 'rcp45' in model:
            gmt_data_groups['rcp45'][model]=gmt_datas[model]
        elif 'rcp60' in model:
            gmt_data_groups['rcp60'][model]=gmt_datas[model]
        elif 'rcp85' in model:
            gmt_data_groups['rcp85'][model]=gmt_datas[model]
        elif 'optim' in model:
            gmt_data_groups['optim'][model]=gmt_datas[model]
        elif 'pledge' in model:
            gmt_data_groups['pledge'][model]=gmt_datas[model]
        elif 'target' in model:
            gmt_data_groups['target'][model]=gmt_datas[model]
        elif 'current' in model:
            gmt_data_groups['current'][model]=gmt_datas[model]

    ###
    plt.close()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,2))
    ## for rcps projection
    years=range(2015,2301)
    models_colors=['royalblue','lightskyblue','lightsalmon','orangered']
    ### Plot RCPs
    for i, rcp in enumerate(rcps):
        models=gmt_data_groups[rcp].keys()
        gmt_models=[]
        for model in models:
            gmt_model=[gmt_data_groups[rcp][model][yr] for yr in years]
            gmt_models.append(gmt_model)
            ax1.plot(years,gmt_model,color=models_colors[i],lw=0.03,zorder=i*-1)
        ## Plot the average
        ax1.plot(years,np.mean(gmt_models,axis=0),color=models_colors[i],lw=2,zorder=i*-1,label=rcp_models_label[i])
    ### Plot CATs
    for i, cat in enumerate(cats):
        models=list(gmt_data_groups[cat].keys())
        #print(len(models))
        if False: ## Further draw 100 samples for plotting the cat GMT (no replacement)
            random.seed(i)
            #models=random.sample(models,300)
            models=random.sample(models,100)
        gmt_models=[]
        #ipdb.set_trace()
        ### R
        for model in models:
            gmt_model=[gmt_data_groups[cat][model][yr] for yr in years]
            gmt_models.append(gmt_model)
            ax2.plot(years,gmt_model,color=models_colors[i],lw=0.015,zorder=i*-1)
        ## Plot the average
        ax2.plot(years,np.mean(gmt_models,axis=0),color=models_colors[i],lw=2,zorder=(i*-1+10),label=cat_models_label[i])
        if cat in ['optim','current']:
            years_sel=[2100,2300]
            yr_idx=[years.index(yr) for yr in years_sel]
            print("We are in the 'plotting_figure2_gmt' function")
            print(cat)
            print("The avreagerd GMT in years 2100 and 2300 are")
            print(np.mean(gmt_models,axis=0)[yr_idx])
            print("")
    ### Set axi
    for ax in [ax1,ax2]:
        ax.set_yticks([0,2,4,6,8,10,12,14,16])
        ax.set_ylim(0,15)
        ax.set_xlim(years[0],years[-1])
        ax.legend(bbox_to_anchor=(0.01,0.5), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2, reverse=True)
        #ax.axhline(y=0,color='k',linestyle='--',lw=0.9)
        for j in ['right', 'top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
    for ax in [ax2]:
        #ax.axvline(x=2100,color='k',linestyle='--',lw=0.9)
        ax.plot([2100,2100],[0,6],lw=0.9,ls='--',color='k')
    ### Set labells
    #ax1.annotate(r'$\bf({A})$ Projection - RCP emission scenarios',xy=(-0.18,1.07),xycoords='axes fraction', fontsize=10)
    ax1.annotate(r'$\bf({A})$',xy=(-0.18,1.07),xycoords='axes fraction', fontsize=10)
    ax1.set_title('Extended RCP Scenarios')
    #ax2.annotate(r'$\bf({B})$ Prediction - policy actions',xy=(-0.09,1.07),xycoords='axes fraction', fontsize=10)
    ax2.annotate(r'$\bf({B})$',xy=(-0.09,1.07),xycoords='axes fraction', fontsize=10)
    ax2.set_title('Climate Policy Actions')
    ax1.set_ylabel("Global mean\ntemperature (K)")
    if False: ## Plot Berkerley GMT observations
        lable_name='Berkeley Earth Surface Temperatures'
        save_name='BEST'
        gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_en%s_%s.npy'%('tas',save_name,'1','global'))
        gmt_ts_obs=gmt_read[:,1]
        gmt_ts_obs=xr.DataArray(gmt_ts_obs,dims=['time'],coords={'time':gmt_read[:,0]})
        gmt_ts_obs=gmt_ts_obs.sel(time=slice(2015,2021))
        for ax in [ax1,ax2]:
            ax.plot(gmt_ts_obs.time, gmt_ts_obs, linestyle='-', lw=1, color='k', label='Observations',zorder=5)
    ### Save figures
    fig_name = 'fig2_corresponding_GMT'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

def plotting_training_domain(Xs,Ys,records):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    row=10
    plt.close()
    fig, axs = plt.subplots(row,1,figsize=(3,3*row))
    params_unique=list(set(records[:,1]))[0:row]
    for i, param in enumerate(params_unique):
        #ipdb.set_trace()
        idx=records[:,1]==param
        ## Y is delta SLR; X is GMT (t-1)
        #axs[i,0].scatter(Xs[idx,8],Ys[idx],s=0.1)
        ## Y is delta SLR; X is SLR (t-1)
        #axs[i,1].scatter(Xs[idx,9],Ys[idx],s=0.1)
        cs=axs[i].scatter(Xs[idx,8],Xs[idx,9],c=Ys[idx],s=0.1, vmin=-np.min(Ys[idx]),vmax=np.max(Ys[idx]),cmap='coolwarm')
        #axs[i,0].colorbar(sc)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, orientation='vertical', extend='both')
        model=records[:,0][idx][0]
        axs[i].set_title(model+'\n'+param)
    ### Save figures
    fig_name = 'training_domain'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.4,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

