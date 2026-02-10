import xarray as xr
import numpy as np
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
#import matplotlib.pyplot as plt; import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import datetime as dt
import ipdb
from importlib import reload
import scipy
import os
from sklearn.neural_network import MLPRegressor
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import sys; sys.path.insert(0, '/Users/home/siewpe/codes/greenland_emulator')

def read_gmts(gmt_models,gmt_ens):

    ### Read GMT
    gmt_datas={model:{} for model in gmt_models}
    gmt_cum_datas={model:{} for model in gmt_models}
    time_since_last_change={model:{} for model in gmt_models}
    gmt_cum_gradient_datas={model:{} for model in gmt_models}
    models_done=[] # Use to record which model has be run
    tas_or_tos='tas'
    region='greenland'
    region='global'
    for i, model in enumerate(gmt_models):
        gmt_en=gmt_ens[i]
        if model in models_done:
            continue
        models_done.append(model)
        if model=='MIROC5_rcp2685mean_pad2600_8var':
            years=list(gmt_datas['MIROC5_rcp85_pad2600_8var'].keys()) ### Select all vaialble years from GMT
            gmts=[]
            for year in years:
                gmt=gmt_datas['MIROC5_rcp85_pad2600_8var'][year]*0.5+gmt_datas['MIROC5_rcp26_pad2600_8var'][year]*0.5
                gmts.append(gmt)
            gmts=xr.DataArray(gmts,dims=['time'],coords={'time':years})
            gmts_cum=gmts.cumulative("time").sum() 
            X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3
            X8=xr.DataArray(X8,dims=['time'],coords={'time':years})
            gmt_cum_data_gradient=gmts_cum.differentiate("time") #idx4
            for year in years:
                gmt_datas[model][year]=gmts.sel(time=year).item()
                gmt_cum_datas[model][year]=gmts_cum.sel(time=year).item()
                time_since_last_change[model][year]=X8.sel(time=year).item()
                gmt_cum_gradient_datas[model][year]=gmt_cum_data_gradient.sel(time=year).item()
            continue # it stopes here can go for the next model
        if model=='MIROC5_rcp85cooling_pad2600_8var':
            years=list(gmt_datas['MIROC5_rcp85_pad2600_8var'].keys()) 
            gmts=[]
            for year in years:
                if year<=2100:
                    gmt=gmt_datas['MIROC5_rcp85_pad2600_8var'][year]
                else: # After 2100 the gmt is fixed at 2015 
                    gmt=gmt_datas['MIROC5_rcp85_pad2600_8var'][2015]
                gmts.append(gmt)
            gmts=xr.DataArray(gmts,dims=['time'],coords={'time':years})
            gmts_cum=gmts.cumulative("time").sum() 
            X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3
            X8=xr.DataArray(X8,dims=['time'],coords={'time':years})
            gmt_cum_data_gradient=gmts_cum.differentiate("time") #idx4
            for year in years:
                gmt_datas[model][year]=gmts.sel(time=year).item()
                gmt_cum_datas[model][year]=gmts_cum.sel(time=year).item()
                time_since_last_change[model][year]=X8.sel(time=year).item()
                gmt_cum_gradient_datas[model][year]=gmt_cum_data_gradient.sel(time=year).item()
            continue
        ## Training models (train_en has to be 1 - by design)
        if model=='MIROC5_rcp85_pad2600_8var':
            model_read='MIROC5_rcp85_pad2600'
            gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model_read,gmt_en,region))
        elif model=='MIROC5_rcp26_pad2600_8var':
            model_read='MIROC5_rcp26_pad2600'
            gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model_read,gmt_en,region))
        ## Predict models
        ## just a different directoary
        #elif 'random' in model: # just a different directory
        #    gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/random/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model,gmt_en,region))
        elif 'distno' in model: # just a different directory
            gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policy_full600/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model,gmt_en,region))
        elif 'stable' in model: # just a different directory
            #gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policyhigh_stable/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model,gmt_en,region))
            gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policy_current_stable/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model,gmt_en,region))
        #elif 'q0.5_optimistic_enlarge' in model: 
        #    gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/enlarge/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model,gmt_en,region))
        else: # For all other gmt_models
            model_read=model
            gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_en%s_%s.npy'%(tas_or_tos,model_read,gmt_en,region))
        gmt=xr.DataArray(gmt_read[:,1],dims=['time'],coords={'time':gmt_read[:,0]})
        cum_gmt=xr.DataArray(gmt_read[:,2],dims=['time'],coords={'time':gmt_read[:,0]})
        last_change=xr.DataArray(gmt_read[:,3],dims=['time'],coords={'time':gmt_read[:,0]})
        cum_gmt_gradient=xr.DataArray(gmt_read[:,4],dims=['time'],coords={'time':gmt_read[:,0]})
        gmt_years=gmt.time.values # this is the years from original gmt
        for i, year in enumerate(gmt_years):
            gmt_datas[model][year]=gmt.sel(time=year).item()
            gmt_cum_datas[model][year]=cum_gmt.sel(time=year).item()
            time_since_last_change[model][year]=last_change.sel(time=year).item()
            gmt_cum_gradient_datas[model][year]=cum_gmt_gradient.sel(time=year).item()

    return gmt_datas, gmt_cum_datas, time_since_last_change, gmt_cum_gradient_datas


def read_training_XY(train_models,train_years,gmt_datas,gmt_cum_datas,time_since_last_change,slr_relative_yr=2015):

    ### Use what TAS
    #only_tas=False; both_tas_and_tos=True
    only_tas=True; both_tas_and_tos=False

    ## sea leve rise scalar file names
    filenames={'ctrlProj':'1990-2100-ctrlProj',
            'MIROC5_rcp26_pad':'1990-2300-MIROC_26',
            'MIROC5_rcp85_pad':'1990-2300-MIROC_85',
            'MIROC5_rcp2685mean_pad':'1990-2300-MIROC_2685mean',
            'MIROC5_rcp2685_cooling_pad':'1990-2300-MIROC_2685mean-coolAO', # Both atmosphere and ocean and set to 2015 value after 2100
            'MIROC5_rcp85_2km':'1990-2300-2km-MIROC_85',
            'HadGEM2-ES_rcp85':'1990-2300-HadGEM2_85',
            ## New simulation
            'MIROC5_rcp85_pad2600':'1990-2600-5km-MIROC_85',
            ## Super new
            'MIROC5_rcp85_pad2600_8var':'1990-2600-5km-MIROC_85',
            'MIROC5_rcp26_pad2600_8var':'1990-2600-5km-MIROC_26',
            'MIROC5_rcp2685mean_pad2600_8var':'1990-2600-5km-MIROC_2685mean',
            'MIROC5_rcp85cooling_pad2600_8var':'1990-2600-5km-MIROC_85-cooling'}
    foldernames={'ctrlProj':'ctrlProj',
            'MIROC5_rcp26_pad':'MIROC_26',
            'MIROC5_rcp85_pad':'MIROC_85',
            'MIROC5_rcp2685mean_pad':'MIROC_2685mean',
            'MIROC5_rcp2685_cooling_pad':'MIROC_2685_cooling',
            'MIROC5_rcp85_2km':'MIROC_85_2km',
            'NorESM1-M_rcp85':'NorESM_85',
            'HadGEM2-ES_rcp85':'HadGEM2_85',
            ## New simulation
            'MIROC5_rcp85_pad2600':'MIROC_85_2600',
            ## SUper New
            'MIROC5_rcp85_pad2600_8var':'MIROC_85_2600_8var',
            'MIROC5_rcp26_pad2600_8var':'MIROC_26_2600_8var',
            'MIROC5_rcp2685mean_pad2600_8var':'MIROC_2685mean_2600_8var',
            'MIROC5_rcp85cooling_pad2600_8var':'MIROC_85_2600_8var_cooling'}
    models_years={'ctrlProj':range(1985,2101),
                'MIROC5_rcp85_pad':range(1985,2301),'MIROC5_rcp26_pad':range(1985,2301),'MIROC5_rcp2685mean_pad':range(1985,2301),'MIROC5_rcp2685_cooling_pad':range(1985,2301),
                'HadGEM2-ES_rcp85':range(1985,2301),'NorESM1-M_rcp85':range(1985,2301),
                'MIROC5_rcp85_2km':range(1985,2301),
                ## New simulation
                'MIROC5_rcp85_pad2600':range(1985,2601),
                ## Super new
                'MIROC5_rcp85_pad2600_8var':range(1985,2601),
                'MIROC5_rcp26_pad2600_8var':range(1985,2601),
                'MIROC5_rcp2685mean_pad2600_8var':range(1985,2601),
                'MIROC5_rcp85cooling_pad2600_8var':range(1985,2601)}

    ### Read Sea level rise and patermeters
    relative_to_control=False
    slr_datas={}
    slr_ratio=100 # m to cm
    for i, model in enumerate(train_models):
        path='/data/ungol_03/shared-data/PISM_GIS/outputs/%s/'%foldernames[model]
        file_list=os.listdir(path)
        file_list=[i for i in file_list if 'timeser-ens-%s'%filenames[model] in i]
        new_time = models_years[model]
        datas=[]
        params=[]
        for fn in file_list:
            data=xr.open_dataset(path+fn,use_cftime=True)['sea_level_rise_potential']
            if relative_to_control: # Read in the control and substract that
                control_path='/data/ungol_03/shared-data/PISM_GIS/outputs/ctrlProj48/'
                control_fn=fn.replace("timeser-ens-1990-2300-%s"%model, "timeser-ens-1990-2300-ctrlProj")
                data_control=xr.open_dataset(control_path+control_fn,use_cftime=True)['sea_level_rise_potential']
                data=data_control-data
            datas.append(data)
            #param=fn.split('-')[-5:]
            param=fn.split('-')[-8:]
            param[-1]=param[-1][0:-3] # take out the .nc in the last part
            params.append('-'.join(param))
        datas=xr.concat(datas,dim='params')
        datas=datas.assign_coords({'params':params})
        datas=datas.assign_coords({'time':new_time})
        # Select dates
        if False:
            datas=datas.sel(time=slice(str(train_years[0]),str(train_years[-1])))
        else: # Select all years (starting from 1985)
            pass 
        if not relative_to_control: # Let not relative to control, then relative to the first year date, and then inverse it
            #datas_first_year=datas.isel(time=0)
            datas_first_year=datas.sel(time=slr_relative_yr)
            datas=datas-datas_first_year; datas=datas*-1
            datas=datas*slr_ratio 
        slr_datas[model]=datas
        #np.save('./parameters_48',slr_datas[model].params.values)

    ### Put the predictors and predictands into X and Y correctly
    X1s, X2s, X3s, X4s, X5s = [], [], [], [], []
    X6s, X7s, X8s, X9s, X10s = [], [], [], [], []
    X11s, X12s, X13s, X14s, X15s = [], [], [], [], []
    Ys = []
    records = []
    for i, model in enumerate(train_models):
        params=slr_datas[model].params.values
        for j, param in enumerate(params):
            for year in train_years[model][1:]: # Start from year 2016
                ## Read parameters
                #X1,X2,X3,X4,X5=param.item().split('-')
                X1,X2,X3,X4,X5,X6,X7,X8=param.item().split('-')
                X1s.append(float(X1))
                X2s.append(float(X2))
                X3s.append(float(X3))
                X4s.append(float(X4))
                X5s.append(float(X5))
                X6s.append(float(X6))
                X7s.append(float(X7))
                X8s.append(float(X8))
                ## Read GMT as X9 in the previous year
                X9=gmt_datas[model][year-1]
                X9s.append(X9)
                ## Read Cumulative GMT as X10
                #X10=gmt_cum_datas[model][year-1]
                #X10s.append(X10)
                ## Read SLR in lag -1 as X11
                X11=slr_datas[model].sel(params=param).sel(time=year-1) 
                X11s.append(X11)
                ## Create X8 manually (last time since GMT changes)
                #X8=time_since_last_change[model][year]
                #X8=gmt_cum_gradient_datas[model][year-1]
                ## Read Y
                #Y=slr_datas[model].sel(params=param).sel(time=year)
                Y=slr_datas[model].sel(params=param).sel(time=year) - slr_datas[model].sel(params=param).sel(time=year-1) # delta SLR from current - last year
                Ys.append(Y.values.item())
                ## Read records
                records.append((model,param,year))

    ### Turn all X and Y into numpy arrays
    records=np.array(records)
    Ys=np.array(Ys)
    ### Combine X
    #Xs=np.column_stack((X1s,X2s,X3s,X4s,X5s,X6s,X7s,X8s,X9s,X10s,X11s)) # Both GMT and cumulative GMT in previous year
    Xs=np.column_stack((X1s,X2s,X3s,X4s,X5s,X6s,X7s,X8s,X9s,X11s)) # No cumulative GMT (only GMT)
    #Xs=np.column_stack((X1s,X2s,X3s,X4s,X5s,X6s,X7s,X8s,X11s)) # No GMT and cumulative GMT

    return Xs, Ys, slr_datas, records

def do_prediction_ANN(model,predict_params,predict_years,gmt_data,gmt_cum_data,X_mean,X_std,Y_mean,Y_std,regress):

    print(model)

    loops=[*regress.keys()]

    slrs={}
    for p, param in enumerate(predict_params):
        slrs[p]={}
        for y, year in enumerate(predict_years):
            if y==0:
                slrs[p][year]=0
            else:
                slrs[p][year]=None

    for y, year in enumerate(predict_years[1:]): # start from 2016 or the yr after relative yr
        X9=gmt_data[year-1] # GMT is always the same for each param in a single model
        X_predicts=[]
        for p, param in enumerate(predict_params):
            X11=slrs[p][year-1]
            X_predict=np.concatenate((param,[X9],[X11])) # not including cumulative GMT
            X_predicts.append(X_predict)
        X_predicts=np.array(X_predicts)
        X_predicts_std=(X_predicts-X_mean)/X_std
        ## Do the regress together with all 1000 parameters
        delta_slrs={}
        for loop in loops:
            regress_loop=regress[loop]
            delta_slr_std=regress_loop.predict(X_predicts_std)
            delta_slr_temp=delta_slr_std*Y_std+Y_mean
            delta_slrs[loop]=delta_slr_temp
        ## Average across the loops
        delta_slr=np.array([delta_slrs[loop] for loop in loops]).mean(axis=0)
        for p, param in enumerate(predict_params):
            slrs[p][year]=slrs[p][year-1]+delta_slr[p]

    slr_predicts=slrs
    return slr_predicts

def do_prediction(model,predict_params,predict_years,gmt_data,gmt_cum_data,X_mean,X_std,Y_mean,Y_std,regress):

    print(model)
    slrs={}
    for p, param in enumerate(predict_params):
        slrs[p]={}
        for y, year in enumerate(predict_years):
            if y==0:
                slrs[p][year]=0
            else:
                slrs[p][year]=None

    for y, year in enumerate(predict_years[1:]): # start from 2016 or the yr after relative yr
        X9=gmt_data[year-1] # GMT is always the same for each param in a single model
        X_predicts=[]
        for p, param in enumerate(predict_params):
            X11=slrs[p][year-1]
            X_predict=np.concatenate((param,[X9],[X11])) # not including cumulative GMT
            X_predicts.append(X_predict)
        X_predicts=np.array(X_predicts)
        X_predicts_std=(X_predicts-X_mean)/X_std
        ## Do the regress together with all 1000 parameters
        delta_slr_std=regress.predict(X_predicts_std)
        delta_slr=delta_slr_std*Y_std+Y_mean
        for p, param in enumerate(predict_params):
            slrs[p][year]=slrs[p][year-1]+delta_slr[p]

    slr_predicts=slrs
    #slr_preddicts has a dict of [param_idx][year]
    return slr_predicts

def do_prediction_old(model,predict_params,predict_years,gmt_data,gmt_cum_data,X_mean,X_std,Y_mean,Y_std,regress):

    print(model)
    slr_predicts={}
    for j, param in enumerate(predict_params): # Can we to all params all together rather than looping like this
        slrs={}; slrs[predict_years[0]]=0
        for k, year in enumerate(predict_years[1:]): # start from 2016 or the yr after relative yr
            X9=gmt_data[year-1]
            #X10=gmt_cum_data[year-1]
            X11=slrs[year-1]
            if False: # Adjust the refreezing paramter (idx:6) to reduce by 0.1% (0.001) per year. Dont't adjust the org. param value
                refreeze_new=param[6]*(0.99**k)
                new_param=param.copy()
                new_param[6]=refreeze_new
                X_predict=np.concatenate((new_param,[X9],[X11])) # not including cumulative GMT
            else:
                X_predict=np.concatenate((param,[X9],[X11])) # not including cumulative GMT
                #X_predict=np.concatenate((param,[X9],[X10],[X11])) # including cumulative GMT
                #X_predict=np.concatenate((param,[X11])) # excluding both GMT and cumulative GMT
            ### Standardize it using the training X_mean and X_std
            X_predict_std=(X_predict-X_mean)/X_std
            ### do the prediction (the intercept is within the SK-Learn function)
            delta_slr_std=regress.predict(X_predict_std.reshape(1,-1))
            delta_slr=delta_slr_std*Y_std+Y_mean
            slrs[year]=slrs[year-1]+delta_slr.item()
        slr_predicts[j]=slrs

    return slr_predicts
