import xarray as xr
import numpy as np
import matplotlib.pyplot as plt; import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import datetime as dt
import ipdb
from importlib import reload
import scipy
import multiprocessing
### For SKlearn
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/'); sys.path.insert(0, '/Users/home/siewpe/codes/greenland_emulator')
import tools
#import functions_read_training_data as functions_read
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)

if __name__ == "__main__":

    reload(functions_read)

    ### This aims to predict ERA5 (1992-2021) GMT with the 4 new training mdoels and autoregresstive appraoch.
    ### Create 1000 ensembles within the current parameters range (select the ensemble with the correct range


    ### Set models for training
    train_years={'MIROC5_rcp85_pad2600_8var':range(2015,2601),
                'MIROC5_rcp26_pad2600_8var':range(2015,2601),
                'MIROC5_rcp2685mean_pad2600_8var':range(2015,2601),
                'MIROC5_rcp85cooling_pad2600_8var':range(2015,2601)}
    train_models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var','MIROC5_rcp2685mean_pad2600_8var','MIROC5_rcp85cooling_pad2600_8var']

    ### Setup predict models (we have three types: pad, nopad, pad2600)
    ## pad: up to 2300 with constraint forcing after 2100 (all available CMIP5 models have this)
    ## pad2600: up to 2600 with increasing forcing after 2100  (all available CMIP5 models have this)
    ## nopad2300: up to 2300 with transisent forcing after 2100 (only a few models have this - no model has all four scnaerios
    if False: # For testing on simulations
        predict_models=['MIROC5_rcp26_pad2600','MIROC5_rcp45_pad2600','MIROC5_rcp60_pad2600','MIROC5_rcp85_pad2600']; predict_years=range(2015,2601)
        predict_models=['CESM1-CAM5_rcp26_pad2300','CESM1-CAM5_rcp45_pad2300','CESM1-CAM5_rcp60_pad2300', 'CESM1-CAM5_rcp85_pad2300']; predict_years=range(2015,2301) 
        # CESM1-CAM5 has no RCP85 
        predict_models=['CESM1-CAM5_rcp26_nopad2300','CESM1-CAM5_rcp45_nopad2300','CESM1-CAM5_rcp60_nopad2300', 'CCSM4_rcp85_nopad2300']; predict_years=range(2015,2301) 
        predict_models_ens=['1']*len(predict_models)
        plot_imbie=False; calculate_and_save_weights=False
        colors=['forestgreen','gold','darkorange','red']
    else: ## to create weighting to ensemble
        predict_models=['BEST']; predict_years=range(1992,2021)
        predict_models_ens=['1']
        plot_imbie=True; calculate_and_save_weights=True
        colors=['gray']
        ## Make the train_years and predict_years consistent
        if False: 
            ## this is not necessary in the auto-regressive appraoch, as it is predicting the delta SLR in each timestep)
            ### better to keep train_models 2015-2601 so that the training Ridge Regression coeffs are the same for all cases 
            train_years={'MIROC5_rcp85_pad2600_8var':range(1992,2601),
                        'MIROC5_rcp26_pad2600_8var':range(1992,2601),
                        'MIROC5_rcp2685mean_pad2600_8var':range(1992,2601),
                        'MIROC5_rcp85cooling_pad2600_8var':range(1992,2601)}

    ## Predict prameters 
    #predict_param_fn='parameters_LHC_1000_new_simulation_range' # old one
    #predict_param_fn='parameters_LHC_5000_8param_range' # new range using 8 parameters
    predict_param_fn='parameters_LHC_1000_8param_range' # new range using 8 parameters
    predict_params=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s.npy'%predict_param_fn)

    ### Read training data
    gmt_ens=[1]*len(train_models)
    gmt_datas,gmt_cum_datas,time_since_last_changes,gmt_cum_gradient_datas=functions_read.read_gmts(train_models,gmt_ens)
    ## Xs, Ys are for training; pism_slrs are the actual simualted SLR
    Xs,Ys,pism_slrs,records=functions_read.read_training_XY(train_models,train_years,gmt_datas,gmt_cum_datas,time_since_last_changes,slr_relative_yr=train_years[train_models[0]][0])

    ### Start and finishing the trining first - before prediction 
    ### Standardize the training data (Xs to X_train_standard)
    X_mean=Xs.mean(axis=0)
    X_std=Xs.std(axis=0)
    X_train_standard=(Xs-X_mean)/X_std
    ## Standardize Ys to Y
    Y_mean=Ys.mean(axis=0)
    Y_std=Ys.std(axis=0)
    Y_train_standard=(Ys-Y_mean)/Y_std
    ### Do the training 
    regress = Ridge(alpha=700).fit(X_train_standard,Y_train_standard)
    print(regress.coef_, regress.intercept_)

    ### Read GMT for predict models (make sure the model name includes the en - some model has en1 to en4., CSIRO-Mk3-6-0_rcp85)
    ### Researt the gmt_datas as the training has finished
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

    ### Do the prediction
    predict_models_new=[*gmt_datas] # this one further includes the ensemble number
    slr_predicts={model:{} for model in predict_models_new}
    for i, model in enumerate(predict_models_new):
        print(model)
        for j, param in enumerate(predict_params):
            slrs={}; slrs[predict_years[0]]=0
            for year in predict_years[1:]: # start from 1993 in this case (predict_years start from 1992)
                X9=gmt_datas[model][year-1]
                #X10=gmt_cum_datas[model][year-1]
                X11=slrs[year-1]
                X_predict=np.concatenate((param,[X9],[X11]))
                ### Standardize it
                X_predict_std=(X_predict-X_mean)/X_std
                ### do the prediction
                delta_slr_std=regress.predict(X_predict_std.reshape(1,-1))
                delta_slr=delta_slr_std*Y_std+Y_mean
                slrs[year]=slrs[year-1]+delta_slr.item()
            slr_predicts[model][j]=slrs

    ### Plot the SLR predictions
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(4,2))
    yss={}
    yss_mean={}
    for i, model in enumerate(predict_models_new):
        ys=[]
        for j, param in enumerate(predict_params):
            x=predict_years
            y=[slr_predicts[model][j][yr] for yr in x]
            ys.append(y)
            ax1.plot(x,y,color=colors[i],lw=0.1,alpha=0.5)
            #ax1.scatter(x[-1]+i*20,y[-1],color=colors[i],s=5)
        yss[model]=ys
        yss_mean[model]=np.mean(ys,axis=0) # Save it for view
        ax1.plot(x,yss_mean[model],color=colors[i],lw=0.5,label='historical emulation')
    ax1.axhline(y=0,color='darkgray',linestyle='--',lw=0.9)
    ### Ploting IMBIE
    if plot_imbie: 
        predict_slr_relative_yr=predict_years[0]
        ## Read the data
        data=np.genfromtxt('/Users/home/siewpe/codes/greenland_emulator/save_obsSLR/imbie_obs/imbie_greenland_2021_mm.csv',delimiter=',') # Latest dataset (1992-2020)
        ## column 0 (idx) is the monthly data
        obs_years=data[:,0][1:]
        obs_years=range(1992,2021)
        ## column 3 (idx) is the cumulative mass balance in mm ==> *0.1 to cm
        ice_loss=data[:,3][1:]*0.1 
        ice_loss=xr.DataArray(ice_loss,dims='time')
        ## column 4 is the cumulative mass balance uncertainty in mm 
        uncertainty=data[:,4][1:]*0.1 
        uncertainty=xr.DataArray(uncertainty,dims='time')
        ## Calculate the annual-mean
        ice_loss=ice_loss.coarsen(time=12).mean().assign_coords({'time':obs_years})
        uncertainty=uncertainty.coarsen(time=12).mean().assign_coords({'time':obs_years})
        ## Set them relative to a year
        # The first year has to be 1992. Need to check - why it is not equal to 0? BEcause we take the annual-mean, 1992 is not equal to 0 anymore
        ice_loss_first=ice_loss.sel(time=predict_slr_relative_yr) 
        ice_loss=ice_loss-ice_loss_first
        ax1.plot(obs_years,ice_loss,'k',zorder=200,lw=2,label='IMBIE')
        if False: ## Get and plot the uncertainty
            loss_min=ice_loss-uncertainty 
            loss_max=ice_loss+uncertainty
            ax1.plot(obs_years,loss_min,'k',linestyle='-',zorder=200,lw=1)
            ax1.plot(obs_years,loss_max,'k',linestyle='-',zorder=200,lw=1)
        if calculate_and_save_weights: # Calculate the difference between obs and each ensemble member (according to previous works)
            ## Calculate the weights
            diff=yss['BEST_en1']-ice_loss.values
            std_obs=uncertainty.values
            #std_model=std_obs*10 # larger this number the weights of the max are smaller (more evenly distributed - won't give too much weights to a single ensemble)
            model_multiple=40
            model_multiple=30
            model_multiple=20 # Default
            model_multiple=10
            print("model_multiple: ",model_multiple)
            std_model=std_obs*model_multiple
            std_total=(std_obs**2+std_model**2)**0.5
            #error_sum_en1=((diff[0]/std_total)**2).sum(axis=1) 
            error_sum=((diff/std_total)**2).sum(axis=1) # along the year, get the weight of each ensemble
            sj=np.exp(-0.5*error_sum) # sj is always positive
            weights=sj/np.sum(sj)
            ## Save the weight files
            np.save('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights'%(predict_param_fn,model_multiple),weights)
            print("Save for weights for %s using %s"%(predict_param_fn,predict_models))
            ## Plot the highest weight ensemble and loweest weight in two difference colors
            sort_idx=np.argsort(weights) # from small to large
            for i in sort_idx[0:200]: # smallest weights
                ax1.plot(predict_years,yss['BEST_en1'][i],color='royalblue',lw=1,zorder=10)
            for i in sort_idx[-200:]: # largest weights
                ax1.plot(predict_years,yss['BEST_en1'][i],color='red',lw=1,zorder=10)
    ## Plot the relative year
    ax1.axvline(x=predict_slr_relative_yr,color='k',linestyle='--',lw=0.9)
    ## Get the legend
    ax1.legend(bbox_to_anchor=(0.05,0.7), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.2, labelspacing=0.1, fontsize=10)
    ### Set the ticks
    ax1.set_ylabel("Sea-level contribution\n(cm)")
    ax1.set_xlabel("Year")
    ax1.set_yticks(range(-3,8))
    ax1.set_xticks(range(1992,2022,2))
    ax1.set_xticklabels(range(1992,2022,2),rotation=45)
    for ax in [ax1]:
        for j in ['right', 'top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
    ### Save the figure
    fig_name = 'SLR_prediction_test'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.05)


            

