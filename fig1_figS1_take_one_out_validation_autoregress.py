import xarray as xr
import numpy as np
import matplotlib.pyplot as plt # import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import datetime as dt
import ipdb
from importlib import reload
import scipy
import multiprocessing
import os
### For SKlearn
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
### For pytorch
import torch
device=torch.device("cpu"); torch.set_num_threads(1) # https://github.com/pytorch/pytorch/issues/17199

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import sys; sys.path.insert(0, '/Users/home/siewpe/codes/greenland_emulator')
import tools
import fig1_figS1_take_one_out_validation_autoregress as take_one_autoregress
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)

if __name__ == "__main__":

    reload(take_one_autoregress)
    reload(functions_read)

    region='greenland'
    region='global'
    regress_tool='sklearn_ann'
    regress_tool='sklearn_ridge'
    relative_to_2015=True
    ###
    if regress_tool=='sklearn_ann': # ANN - multi-ensembles
        ann_loops=[0,1,2,3,4]
        #ann_batch_size, l2_coeff, learn_rate, ridge_alpha
        #parameter_sets=[(200,0.001,0.001,'none'),(300,0.001,0.002,'non'),(400,0.001,0.003,'none')]
        ## Finalized
        parameter_sets=[(20,0.001,0.05,'none')] # This is indeed better than the Ridge Regression. Small batch size helps (10 batch size won't be better)
        parameter_sets=[(32,0.0001,0.01,'none')] # This produces similar result in the main figure compared to Ridge Regression
        figure_alphas_rmse=False
    elif regress_tool=='sklearn_ridge': # Ridge Regression
        ann_loops=['none']
        parameter_sets=[('none','none','none',700)] # Set ridge=700 ### This shows the best result in Figure S1
        ###
        ### Get the Figure S1
        figure_alphas_rmse=True
        figure_alphas_rmse=False
    if figure_alphas_rmse: ### For Figure S1
        parameter_sets=[('none','none','none',0.1),('none','none','none',1),('none','none','none',10),('none','none','none',20),('none','none','none',40),
                        ('none','none','none',100),('none','none','none',200),('none','none','none',300),('none','none','none',400),
                        ('none','none','none',500),('none','none','none',600),('none','none','none',700),('none','none','none',800),
                        ('none','none','none',900),('none','none','none',1000),('none','none','none',1500),('none','none','none',2500),
                        ('none','none','none',2500),('none','none','none',4000),('none','none','none',8000),('none','none','none',16000)]
    ensemble_models=[]
    for ps in parameter_sets:
        for loop in ann_loops:
            ps_new=list(ps)
            ps_new.append(loop)
            ensemble_models.append(ps_new)

    ## To make it the same as ISMIP6 (this one is correct; because the taining set used MIROC5_26 before 2015
    #train_years=range(1985,2301) # The simulation itself starts from 1985 (not 1986)
    train_years={'MIROC5_rcp85_pad':range(2015,2301),'MIROC5_rcp26_pad':range(2015,2301),'MIROC5_rcp2685mean_pad':range(2015,2301),'MIROC5_rcp2685_cooling_pad':range(2015,2301),
                'MIROC5_rcp85_pad2600':range(2015,2601),
                'MIROC5_rcp85_pad2600_8var':range(2015,2601),
                'MIROC5_rcp26_pad2600_8var':range(2015,2601),
                'MIROC5_rcp2685mean_pad2600_8var':range(2015,2601),
                'MIROC5_rcp85cooling_pad2600_8var':range(2015,2601)}
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad','MIROC5_rcp85_pad2600']
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad','MIROC5_rcp2685mean_pad']
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad','MIROC5_rcp2685mean_pad','MIROC5_rcp2685_cooling_pad','MIROC5_rcp85_pad2600']
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad','MIROC5_rcp2685mean_pad','MIROC5_rcp2685_cooling_pad']
    #models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad','MIROC5_rcp85_pad2600','MIROC5_rcp2685_cooling_pad']
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad']
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad','MIROC5_rcp85_pad2600']
    ## New simulations
    models=['MIROC_85_2600_8var', 'MIROC_26_2600_8var','MIROC_2685mean_2600_8var','MIROC_85cooling_2600_8var']
    models=['MIROC_85_2600_8var', 'MIROC_26_2600_8var']
    models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var', 'MIROC5_rcp2685mean_pad2600_8var']
    ## Just between two
    models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var']
    labels=['MIROC5_rcp85_2600_8var', 'MIROC5_rcp26_2600_8var']
    ## Default (all four)
    models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var','MIROC5_rcp2685mean_pad2600_8var','MIROC5_rcp85cooling_pad2600_8var']
    labels=[r'$\bf{(A)}$ PISM forced by RCP85', r'$\bf{(B)}$ PISM forced by RCP26',r'$\bf{(C)}$ PISM forced by RCP2685', r'$\bf{(D)}$ PISM forced by RCP85-cooling']
    ## Final

    ### Read training data - Xs and Ys; train_models and gmt_models are the same
    train_models=models; gmt_ens=[1]*len(train_models)
    #Xs,Ys,model_datas,gmt_datas,gmt_cum_datas,time_since_last_changes,records=functions_read.read_training_XY(train_models,train_years,gmt_models,region=region)
    gmt_datas,gmt_cum_datas,time_since_last_changes,gmt_cum_gradient_datas=functions_read.read_gmts(train_models,gmt_ens)
    ## Xs, Ys are for training 
    Xs,Ys,pism_slrs,records=functions_read.read_training_XY(train_models,train_years,gmt_datas,gmt_cum_datas,time_since_last_changes,slr_relative_yr=2015)

    ### Take-one-forcing-out for validation
    Y_tests={}; X_test_standards={}; Y_train_means={}; Y_train_stds={}
    slr_predicts={}
    slr_trues={}
    jobs=[]; manager=multiprocessing.Manager(); return_dict=manager.dict()
    for model in models:
        if True: # "take-one-model-out" for validation
            test_bool = records[:,0]==model
            test_idx=test_bool.nonzero()[0]
            train_bool = ~test_bool
            train_idx=train_bool.nonzero()[0]
        X_train=Xs[train_idx,:]; Y_train=Ys[train_idx]
        X_test=Xs[test_idx,:]; Y_test=Ys[test_idx]
        ## Standardize X_train and Y_train
        X_train_mean=X_train.mean(axis=0); X_train_sd=X_train.std(axis=0)
        X_train_standard=(X_train-X_train_mean)/X_train_sd
        Y_train_mean=Y_train.mean(axis=0); Y_train_sd=Y_train.std(axis=0)
        Y_train_standard=(Y_train-Y_train_mean)/Y_train_sd
        if False: ## Fill 0 for nan values (no nan values)
            X_train_standard[np.isnan(X_train_standard)]=0
            Y_train_standard[np.isnan(Y_train_standard)]=0
        ## Apply the mean and SD to testing data
        X_test_standard=(X_test-X_train_mean)/X_train_sd
        ## Save for later use
        X_test_standards[model]=X_test_standard
        Y_train_means[model]=Y_train_mean
        Y_train_stds[model]=Y_train_sd
        Y_tests[model]=Y_test # It doesn't need to be standadized because this is the true
        ### Start ANN training
        slr_predict_ens={}
        records_test = records[test_idx,:]
        #years=train_years[model][1:]
        years=train_years[model]
        param_size=pism_slrs[model].params.size
        for i, en_model in enumerate(ensemble_models):
            #ann_batch_size, l2_coeff, learn_rate, ridge_alpha, loop = en_model
            dict_key=model+';'+str(en_model)
            if True: ## Do multi-processing
                proc=multiprocessing.Process(target=take_one_autoregress.ANN_function_auto,
                        args=(model,en_model,X_train_standard,Y_train_standard,X_test_standard,X_train_sd,X_train_mean,Y_train_sd,Y_train_mean,
                            records_test,years,param_size,dict_key,return_dict),kwargs=dict(regress_tool=regress_tool))
                proc.start(); jobs.append(proc)
            else: ## Without multi-processing (for debugging)
                take_one_autoregress.ANN_function_auto(model,en_model,X_train_standard,Y_train_standard,X_test_standard,X_train_sd,
                                            X_train_mean,Y_train_sd,Y_train_mean,records_test,years,param_size,dict_key,return_dict,regress_tool=regress_tool)
        slr_trues[model]=pism_slrs[model].sel(time=slice(2016,':')).values # save the Y_test in this way...easier (equaivlent to Y_tests which starts from all years)
    ###
    ### Send model and ensemble_models together for multiprocessing (5 models x 10 members)
    for proc1 in jobs:
        proc1.join()
    ### Reaad the return_key
    slr_predicts_temp={model:{} for model in models}
    for dict_key in return_dict.keys():
        model,en_model=dict_key.split(';')
        slr_predicts_temp[model][en_model]=return_dict[dict_key]
    ### Average over ANN ensemble models
    slr_predicts={}
    for model in models:
        slr_predicts[model]=np.array([slr_predicts_temp[model][str(en_model)] for en_model in ensemble_models]).mean(axis=0) 

    ###
    ### Plot the figure - take-one-out-validation
    plt.close()
    #fig,axs=plt.subplots(1,len(models),figsize=(len(models)*3,3))
    fig,axs=plt.subplots(2,len(models),figsize=(len(models)*3*0.9,5*0.9),height_ratios=[2,1])
    axs=axs.flatten()
    for i, model in enumerate(models):
        ### Scatter plot (upper panels)
        years=train_years[model][1:]
        Y_true = slr_trues[model].reshape(-1)
        Y_predict= slr_predicts[model].reshape(-1) ## Plotting the average of the ensembles
        axs[i].scatter(Y_true,Y_predict,color='k', zorder=-5, s=0.01) # Plot all dots
        axs[i].axhline(y=0,color='lightgray',linestyle='--')
        axs[i].axvline(x=0,color='lightgray',linestyle='--')
        ymin=np.min([Y_true,Y_predict])
        ymax=np.max([Y_true,Y_predict])
        axs[i].plot([ymin,ymax],[ymin,ymax],linestyle='--',color='lightgray')
        axs[i].annotate(labels[i],xy=(-0.17,1.05),xycoords='axes fraction', fontsize=11)
        #axs[i].set_title(labels[i],loc='left',size=10)
        axs[i].set_xlabel('Simulated sea level (cm)')
        if i==0:
            axs[i].set_ylabel('Emulation sea level (cm)')
        ## Add RMSE (this RMSE will be adjusted with relative_to_2015 or not)
        rmse=tools.rmse_nan(Y_true,Y_predict)
        axs[i].annotate("RMSE=%s"%(str(round(rmse,3))),xy=(0.04,0.95),xycoords='axes fraction', fontsize=10,color='k') # idx 1 is RMSE; 0 is MSE
        # Add correlation
        corr=scipy.stats.pearsonr(Y_true,Y_predict)[0]
        axs[i].annotate("r=%s"%(str(round(corr,3))),xy=(0.04,0.88),xycoords='axes fraction', fontsize=10,color='k')
        ## Set xticks
        #axs[i].set_yticks(axs[i].get_xticks())
        axs[i].set_yticks(range(0,190,50))
        axs[i].set_xticks(range(0,190,50))
        axs[i].set_xlim(-5,185); axs[i].set_ylim(-5,185)
    ### Timeseries validation rather than scatter plot (lower panels)
    for i, model in enumerate(models):
        years=train_years[model][1:]
        params=pism_slrs[model].params
        for j, param in enumerate(params): 
            axs[i+len(models)].plot(years,slr_trues[model][j], color='gray', lw=0.3)
            axs[i+len(models)].plot(years,slr_predicts[model][j], color='salmon', lw=0.3)
        axs[i+len(models)].plot(years,slr_trues[model].mean(axis=0), color='black', lw=3, label='Simulated')
        axs[i+len(models)].plot(years,slr_predicts[model].mean(axis=0), color='red', lw=2, label='Emulated')
        #axs[i+len(models)].annotate("%s"%(model),xy=(0.01,0.9),xycoords='axes fraction', fontsize=10, color='black')
        axs[i+len(models)].legend(bbox_to_anchor=(0.01,0.5), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.3, labelspacing=0.2, fontsize=10)
        axs[i+len(models)].set_ylim(-10,185)
        axs[i+len(models)].set_xlim(2015,2600)
        axs[i+len(models)].set_xticks([2015,2200,2400,2600])
        axs[i+len(models)].set_xlabel('Year')
        axs[i+len(models)].axhline(y=0,color='lightgray',linestyle='--')
        if i==0:
            axs[i+len(models)].set_ylabel('Sea-level contribution (cm)')
    for ax in axs:
        for j in ['right', 'top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
    #fig_name = 'take_one_model_out_validation_ts_auto_regressive'
    fig_name = 'fig1_validation_scatter_ts_combine'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.20,hspace=0.3) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)


    ### Figure S1 - The ridge regression figure
    if figure_alphas_rmse: 
        rmses={model:{} for model in models}
        alphas=[]
        for model in models:
            en_models=[*slr_predicts_temp[model]]
            for en_model in en_models:
                alpha=float(en_model.split(',')[3])
                if alpha not in alphas:
                    alphas.append(alpha)
                Y_predict=slr_predicts_temp[model][en_model].reshape(-1) # turn params x years into 1d
                Y_true=slr_trues[model].reshape(-1)
                rmse=tools.rmse_nan(Y_true,Y_predict)
                rmses[model][alpha]=rmse
        alphas=[i for i in np.sort(alphas)]
        ### Get the average RMSE for all model in each alpha
        rmse_means={}
        for alpha in alphas:
            rmse_mean=np.mean([rmses[model][alpha] for model in models])
            rmse_means[alpha]=rmse_mean
        ### Start plotting
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,1))
        bar_width=0.1
        x=np.arange(len(alphas))
        x_adjs=np.linspace(-0.2,0.2,len(models))
        for i, model in enumerate(models):
            rmse_models=[rmses[model][alpha] for alpha in alphas]
            ax1.bar(x+x_adjs[i], rmse_models, bar_width, color='k')
        ax1.bar(x+x_adjs[-1]+np.diff(x_adjs).mean(), [rmse_means[alpha] for alpha in alphas], bar_width, color='red')
        print(rmse_means)
        for i, xx in enumerate(x):
            #axs[i].annotate(labels[i],xy=(-0.17,1.05),xycoords='axes fraction', fontsize=11)
            ax1.annotate(round(rmse_means[alphas[i]],3), xy=(x[i]-0.5,rmse_means[alphas[i]]+5), xycoords='data',fontsize=5,color='red',rotation=45)
        yticks=[0,5,10,15]
        ax1.set_yticks(yticks)
        #ax1.bar(x-0.4, rmse_test, bar_width, color='r')
        ax1.set_xticks(x)
        alphas_label=[int(i) if i>=1 else i  for i in alphas]
        ax1.set_xticklabels(alphas_label,rotation=45,size=8.5)
        ax1.set_xlabel(r'Lambda ($\lambda$)')
        ax1.set_ylabel('RMSE (cm)')
        for ax in [ax1]:
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        ## Save figure
        fig_name = 'figS1_supplementary'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.001)




def ANN_function_auto(model,en_model,X_train_standard,Y_train_standard,X_test_standard,X_train_sd,X_train_mean,Y_train_sd,Y_train_mean,records_test,years,param_size,dict_name,return_dict,regress_tool=None):

    print("Start auto-regress prediction for %s %s"%(model,en_model))
    ann_batch_size, l2_coeff, learn_rate, ridge_alpha, loop = en_model # 20, 0.001, 0.05

    if regress_tool=='sklearn_ann': ## ANN 
        solver='adam'; ann_shuffle=True; ann_nodes=(10,10) 
        regress=MLPRegressor(hidden_layer_sizes=ann_nodes, learning_rate_init=learn_rate,solver=solver,early_stopping=True,n_iter_no_change=50,
                learning_rate='constant',batch_size=ann_batch_size,max_iter=1500, alpha=l2_coeff, shuffle=ann_shuffle, validation_fraction=0.2,
                tol=1e-6,verbose=False,random_state=loop).fit(X_train_standard,Y_train_standard)
    elif regress_tool=='sklearn_ridge': ## Ridge Regression
        #from sklearn.ensemble import RandomForestRegressor
        #regress = LinearRegression().fit(X_train_standard,Y_train_standard)
        if False: # Add a constant manually
            #mean=0; std_dev=1; constant=np.random.normal(mean, std_dev, X_train_standard.shape[0]).reshape(-1,1)
            constant=np.ones(X_train_standard.shape[0]).reshape(-1,1)
            X_train_standard_constants=np.hstack((constant,X_train_standard))
            regress = Ridge(alpha=ridge_alpha,fit_intercept=False).fit(X_train_standard_constants,Y_train_standard)
        else: # The constant is within the SK-learn function
            regress = Ridge(alpha=ridge_alpha).fit(X_train_standard,Y_train_standard)
            #regress = Ridge(alpha=0.000001).fit(X_train_standard,Y_train_standard) # This will be equilavent to linear regression
            #regress = LinearRegression().fit(X_train_standard,Y_train_standard)
        print(regress.coef_, regress.intercept_)

    if True: ### Start auto-regresstive prediction for the tessting model
        slr_saves={}
        slr_saves_std={}
        for year in years[1:]: # start from 2016
            #print(year)
            idx=records_test[:,2]==str(year)
            X_test_new=X_test_standard[idx] # size of 48 x no. of predictors 
            if year==2016: # the slr in previous year is 0 (but it is standardized)
                pass
            else: # Replace the slr for the last column (the new SLR used for prediction)
                X_test_new[:,-1]=slr_saves_std[year-1]
            ### In long-term, just put slr_lag-1 = 0  for year 2016, but to standardize it
            if False: # Add a constant manually
                constant=np.ones(X_test_new.shape[0]).reshape(-1,1)
                X_test_new_constants=np.hstack((constant,X_test_new))
                delta_slr_std=regress.predict(X_test_new_constants)
            else: # The functions has the constant
                delta_slr_std=regress.predict(X_test_new)
            delta_slr=delta_slr_std*Y_train_sd+Y_train_mean
            slr_last_year=X_test_new[:,-1]*X_train_sd[-1]+X_train_mean[-1] # size of 48
            slr_this_year=slr_last_year+delta_slr
            slr_saves[year]=slr_this_year
            ## Re-standardize the slr
            slr_saves_std[year]=(slr_this_year-X_train_mean[-1])/X_train_sd[-1]
        slr_predict=[]
        for j in range(param_size):
            slr_temp=[slr_saves[year][j] for year in years[1:]]
            slr_predict.append(slr_temp)
        slr_predict=np.array(slr_predict)

    if False:  ## Check - using the functions are identical as the original one
        X_test_full=X_test_standard*X_train_sd+X_train_mean
        idx=records_test[:,-1]=='2016'
        predict_params=X_test_full[idx,0:8]
        #records_test[0:len(years)-1]
        gmt_2016_2600=X_test_full[0:len(years)-1,8] # actually 2016 is storing 2015 gmt because of the read_X_Y_training funrction
        gmt_data={}
        for i, year in enumerate(years[0:-1]): # year 2015 to 2599
            gmt_data[year]=gmt_2016_2600[i]
        gmt_cum_data=None
        predict_years=years # range(2015,2601)
        X_mean=X_train_mean
        X_std=X_train_sd
        Y_mean=Y_train_mean
        Y_std=Y_train_sd
        slr_predict_temp=functions_read.do_prediction(model,predict_params,predict_years,gmt_data,gmt_cum_data,X_mean,X_std,Y_mean,Y_std,regress)
        slr_predict_new=[]
        for i in range(param_size):
            slr_predict_new.append([slr_predict_temp[i][year] for year in years[1:]]) # Crop out the years 2015
        slr_predict_new=np.array(slr_predict_new)
        slr_predict=slr_predict_new


    # slr_predict has a shape of no.of.ensemble*years (48*585 for RCP85)
    return_dict[dict_name]=slr_predict
