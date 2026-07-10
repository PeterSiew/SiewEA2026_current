import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import datetime as dt
import ipdb
from importlib import reload
import scipy
from scipy import stats
import os
import itertools

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)

if __name__ == "__main__":


    ### Read MIROC5 GMT
    ### Read training data - Xs and Ys; train_models and gmt_models are the same
    if True: # This is the filter (smooth GMT)
        models=['MIROC5_rcp26_pad2600_8var','MIROC5_rcp85_pad2600_8var']
        train_years={model:range(2015,2601) for model in models}
        models_ens=[1]*len(models)
        gmt_datas,gmt_cum_datas,time_since_last_changes,gmt_cum_gradient_datas=functions_read.read_gmts(models,models_ens)
        #Xs,Ys,pism_slrs,records=functions_read.read_training_XY(models,train_years,gmt_datas,gmt_cum_datas,time_since_last_changes,slr_relative_yr=2015)

    tas_folder='/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_map_data/MIROC5/'
    vars=['tas','pr']; Amons=['Amon','Amon']
    vars=['tas','tos','pr']; Amons=['Amon','Omon','Amon']
    vars=['tas']; Amons=['Amon']
    exps=['hist','rcp26','rcp85']
    exps=['rcp26','rcp85']
    filenames={'hist':'%var_%Amon_MIROC5_historical_r1i1p1_185001-201212.nc',
            'rcp26':'%var_%Amon_MIROC5_rcp26_r1i1p1_200601-210012.nc',
            'rcp85':'%var_%Amon_MIROC5_rcp85_r1i1p1_200601-210012.nc'}
    ### Read data
    tas_datas={var:{} for var in vars}
    for i, var in enumerate(vars):
        for exp in exps:
            filename=filenames[exp]
            filename=filename.replace("%var",var)
            filename=filename.replace("%Amon",Amons[i])
            ## Read the file
            data=xr.open_dataset(tas_folder+filename)[var]
            if exp=='hist':
                data=data.sel(time=slice("1850-01-01","2005-12-31"))
            elif exp in ['rcp26','rcp85']:
                data=data.sel(time=slice("2006-01-01","2100-12-31"))
            ## Calculate the annual-mean
            if var=='pr':
                pass
            data=data.coarsen(time=12).mean()
            years=data.time.dt.year
            data=data.assign_coords({'time':years})
            tas_datas[var][exp]=data.compute()
    ###
    ### Calculate the global and Greenland domain
    regions=['global','greenland']
    #lat1=60; lat2=83; lon1=290; lon2=340 # Small area within Greenland
    domains={'global':(-90,90,0,360),'greenland':(55,85,260,360)} # lat1, lat2, lon1, lon2
    tas_tss={var:{exp:{} for exp in exps} for var in vars}
    for var in vars:
        for exp in exps:
            data=tas_datas[var][exp]
            for region in regions:
                lat1,lat2,lon1,lon2=domains[region]
                ## For Amon variables
                if var in ['tas','pr']:
                    lons=data.lon.values; lats=data.lat.values
                    ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
                    ts=xr.DataArray(ts,dims=['time'],coords={'time':data.time.values})
                    tas_tss[var][exp][region]=ts
                if var in ['tos']:
                    pass

    ### Extract MIROC5 forcing files (these are already in Greenland domain)
    """ 
    For MIROC5 RCP85 exp
    -surface_anomaly_file ./inputs/atm_forcing/atm-MIROC5_rcp85-combined-NEW.nc 
    -ocean th -ocean_th_file ./inputs/ocn_forcing/MIROC-ESM-CHEM_rcp85-comb-nonan-NEW.nc
    ## For MIROC RCP26 rcp
    -surface_anomaly_file ./inputs/atm_forcing/atm-MIROC5_rcp26-combined-NEW.nc
    -ocean_th_file ./inputs/ocn_forcing/MIROC-ESM-CHEM_rcp26-comb-nonan-NEW.nc
    """
    vars=['atm','ocn']
    forcing_folders={'atm':'/data/ungol_03/shared-data/PISM_GIS/inputs/atm_forcing/',
            'ocn':'/data/ungol_03/shared-data/PISM_GIS/inputs/ocn_forcing/'}
    filenames={'atm':'atm-MIROC5_%rcp-combined-NEW.nc',
                'ocn':'MIROC-ESM-CHEM_%rcp-comb-nonan-NEW.nc'}
    vars_names={'atm':'ice_surface_temp_anomaly',
            'ocn':'theta_ocean'}
    exps=['rcp26','rcp85']
    ### Read data
    years=range(1950,2101)
    forcing_datas={var:{} for var in vars}
    for i, var in enumerate(vars):
        for exp in exps:
            print(var,exp)
            folder=forcing_folders[var]
            filename=filenames[var].replace('%rcp',exp)
            ## Read the file
            data=xr.open_dataset(folder+filename,decode_times=False,chunks={'time':5})
            data=data[vars_names[var]]
            ## Assign the correct time
            data=data.assign_coords({'time':years})
            ## Start from 2015
            data=data.sel(time=slice(2015,2100))
            ## Put zero data to nan
            data=xr.where(data!=0,data,np.nan)
            ## Do the slicing to save memory
            if False:
                data=data.isel(y=slice(None, None, 3)).isel(x=slice(None, None, 3))
            forcing_datas[var][exp]=data.compute()
    ###
    ### Average the Greenland domain
    forcing_tss={var:{} for var in vars}
    for i, var in enumerate(vars):
        for exp in exps:
            data=forcing_datas[var][exp]
            data_ts=data.mean(dim='x').mean(dim='y')
            forcing_tss[var][exp]=data_ts

    ipdb.set_trace()

    ### 
    ### Start plotting
    plt.close()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3))
    exp_colors={'rcp26':'green','rcp85':'red'}
    years_sel=range(2015,2101)
    for i, exp in enumerate(exps):
        ## Get the GMT
        if False: # Use the smooth GMT
            model=models[i]
            gmt_ts=[gmt_datas[model][year] for year in years_sel]
        else: # use the non-smooth GMT (this GMT is derived here)
            gmt_ts=tas_tss['tas'][exp]['global'].sel(time=years_sel)
        ## ax1: GMT versus Greenland-atm-forcing
        atm_ts=forcing_tss['atm'][exp].sel(time=years_sel)
        corr=round(tools.correlation_nan(gmt_ts,atm_ts),2)
        ax1.scatter(gmt_ts,atm_ts,color=exp_colors[exp],s=5,label="MIROC5 %s (r=%s)"%(exp,corr))
        ax1.set_ylabel("Ice surface\ntemperature anomaly (K)")
        ## ax2:GMT versus Greenland-ocean-forcing
        ocn_ts=forcing_tss['ocn'][exp].sel(time=years_sel)
        corr=round(tools.correlation_nan(gmt_ts,ocn_ts),2)
        ax2.scatter(gmt_ts,ocn_ts,color=exp_colors[exp],s=5,label="MIROC5 %s (r=%s)"%(exp,corr))
        ax2.set_ylabel("Ocean thermal forcing at\neffective depth (K)")
    ## Fix the axis
    for ax in [ax1,ax2]:
        ax.set_xlabel("Global-mean temperature (GMT, K)")
        ax.legend(bbox_to_anchor=(-0.05,0.8), ncol=1, loc='lower left', frameon=False, columnspacing=0.2,handletextpad=0.2, labelspacing=0.1, reverse=True)
        for j in ['right', 'top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
    ## Save figure
    fig_name = 'gmt_as_proxy_atm_ocn'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)



    ### Plto the GMT versus rate of change of SLR
    if False:
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(3,3))
        years_sel=range(2015,2601)
        for i, exp in enumerate(exps):
            model=models[i]
            gmt_ts=[gmt_datas[model][year] for year in years_sel]
            #gmt_ts=tas_tss['tas'][exp]['global'].sel(time=years_sel)
            #slr_diff=pism_slrs[model].mean(dim='params').sel(time=years_sel).differentiate("time")
            slr_diff=pism_slrs[model].mean(dim='params').sel(time=years_sel)
            ax1.scatter(gmt_ts,slr_diff,color=exp_colors[exp],s=5)
        ## Save figure
        fig_name = 'gmt_versus_slr_rate'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.4) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
