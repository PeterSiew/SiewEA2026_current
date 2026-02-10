import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
from importlib import reload
import pandas as pd

import sys
import os
import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
from scipy.signal import detrend
import scipy

if __name__ == "__main__":

    #models=['IPSL-CM5A-LR_rcp26','IPSL-CM5A-LR_rcp45','IPSL-CM5A-LR_rcp60','IPSL-CM5A-LR_rcp85']

    ### This script create individual GMT timeseries, their mean and plot the figures

    folder='/Users/home/siewpe/codes/greenland_emulator/save_gmt/gmt_ts_download/CMIP5/'
    file_list=os.listdir(folder)
    file_list_new=[]
    for file in file_list:
        if ('rcp' in file) & ('i1p1' in file):
            file_list_new.append(file)
    file_list_new=sorted(file_list_new)
    models=[]
    ensembles=[]
    for file in file_list_new:
        model=file.split('_')[3] 
        rcp=file.split('_')[4] 
        models.append(model+'_'+rcp)
        en=file.split('_')[5].replace('r','').replace('i1p1.dat','') # the en might have 2 digits (e.g., 11,12)
        ensembles.append(en)

    region='global'; var='tas'
    clim_st_yr=1850; clim_end_yr=1900 #ini_st_year=1850# but some models have no 1850. e.g., HadGEM2-ES starts at 1859. 
    final_st_yr=1850; #final_end_yr=2300 
    time_pad_st=2101; time_pad_end=2300 # time_pad_end could be longer if want to filter by moving windows
    #### Remember to select there
    create_pad_till_2600=True
    create_pad_till_2600=False
    ####***
    if create_pad_till_2600: # Mostly for the MIROC5 training simulation that extends up to 2600
        time_pad_st=2101; time_pad_end=2600 # just for temporary
    filter_window=51

    gmt_anoms={}
    models_ens_pad,  models_ens_nopad= [], []
    for i, model in enumerate(models):
        en=ensembles[i]
        #print(model,en)
        ### Read the monthly data
        path=folder+'global_tas_Amon_'+model+'_r%si1p1.dat'%en
        data_raw=np.genfromtxt(path)
        years=data_raw[:,0]
        gmt=data_raw[:,1:13].reshape(-1)
        gmt=xr.DataArray(gmt,dims=['time'])

        ## Put them into annual-mean
        gmt=gmt.coarsen(time=12).mean().assign_coords(time=years)

        if gmt.time.values[0]!=1850:
            print(model, en, ' has no 1850 year data')
            continue

        ## Relative to PI-period (1850-1900)
        gmt_clim=gmt.sel(time=slice('%s'%clim_st_yr,'%s'%clim_end_yr)).mean(dim='time')
        gmt_anom=gmt-gmt_clim
        gmt_anoms[model+'_'+str(en)]=gmt_anom # save a backup for a non-filter version

        ## Create a version with padding after 2100 (even for GMT with 2300 data - so for all data)
        gmt_early=gmt_anom.sel(time=slice(final_st_yr,time_pad_st-1))
        ending_value=gmt_early.isel(time=-1).values.item() # 2100 data
        time_pad=np.arange(time_pad_st,time_pad_end+1) # model has time_pad_st=2101
        pad_data=xr.DataArray([ending_value]*len(time_pad),dims=['time'],coords={'time':time_pad})
        gmt_pad=xr.concat([gmt_early,pad_data], dim='time')
        ## Do filtering on the pad timeseries
        gmt_pad_filter=scipy.signal.savgol_filter(gmt_pad,filter_window,3)
        gmt_pad=gmt_pad.copy(data=gmt_pad_filter)

        ## Create a version without padding (only for GMT extended to 2300)
        gmt_nopad=gmt_anom
        if gmt_nopad.time[-1]==time_pad_end: ##!! There won't be nopad data for create_pad_till_2600 as no GCM extends simulation up to 2600
            gmt_nopad_filter=scipy.signal.savgol_filter(gmt_nopad,filter_window,3)
            gmt_nopad=gmt_nopad.copy(data=gmt_nopad_filter)
        else: # None if no 2300 data
            gmt_nopad=None

        ### Loop via the pad and nopad versions
        ### Save the timeseries into correct formate
        pad_saves=['pad2300','nopad2300'] # constant forcing versus extended forcing until 2300
        if create_pad_till_2600:
            pad_saves=['pad2600','nopad2600'] # nopad2600 basically won't be save, as gmt_nopad is always None (no 2600 GCM simulations)
        for j, gmt_ts in enumerate([gmt_pad, gmt_nopad]):
            if (j==1) & (gmt_ts is None): # skip this if the gmt_nopad is None. gmt_ts must be None if gmt_nopad2600
                print(model,en, ' has no 2300 GMT - cannot create nopad GMT - skip')
                continue
            elif (j==1) & (gmt_ts is not None): # This one and the previous one is mutually exclusive
                print(model,en, ' has 2300 GMT - can create nopad GMT')
            else:
                print(model,en, ' has 2100 GMT - can create pad GMT') # All data must have 2100 GMT
            if j==0:
                models_ens_pad.append([model,en])
            elif j==1: # model that has no 2300 data wont go here becuase of the "continue" appearing before
                 models_ens_nopad.append([model,en])
            years=gmt_ts.time # idx0
            gmt_data=gmt_ts #idx1
            gmt_cum_data=gmt_ts.cumulative("time").sum() #idx2 (the cumulative data starts at 1990)
            if j==0: # when pad_save=pad
                X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3
            elif j==1: # when pad_save=no_pad
                X8=np.array([1 if year<=2100 else 1 for year in years]) #idx3
            gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
            save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
            np.save('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_%s_en%s_%s.npy'%(var,model,pad_saves[j],en,region),save_data)

    ### Plot the saved timeseries (also create the interpolated-MIROC and CMIP5-mean data)
    if True:
        fig, axs = plt.subplots(2,1,figsize=(4,5))
        pad_saves=['pad2300','nopad2300'] # constant forcing versus extended forcing
        if create_pad_till_2600:
            pad_saves=['pad2600','nopad2600'] 
        rcp_colors={'rcp85':'red','rcp60':'tomato','rcp45':'orange','rcp26':'green'}
        ipdb.set_trace()
        for i, models_ens in enumerate([models_ens_pad, models_ens_nopad]): ## go via the models with pad first, and then model with nopad (it doesn't work for 2600 as there is no nopad model)
            models=np.array(models_ens)[:,0]; ens=np.array(models_ens)[:,1]
            tss={}
            for j, model in enumerate(models):
                en=ens[j]
                gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_%s_en%s_%s.npy'%(var,model,pad_saves[i],en,region))
                if True: # Plot the GMT
                    gmt=xr.DataArray(gmt_read[:,1],dims=['time'],coords={'time':gmt_read[:,0]})
                elif False: # plot the cumulative GMT
                    gmt=xr.DataArray(gmt_read[:,2],dims=['time'],coords={'time':gmt_read[:,0]}) #
                #print(model,en,gmt[-1])
                rcp=model.split('_')[-1]
                if rcp not in tss:
                    tss[rcp]=[]
                tss[rcp].append(gmt)
                axs[i].plot(gmt.time,gmt,color=rcp_colors[rcp],alpha=0.3,lw=0.3)
                if (i==0) & (model=='MIROC5_rcp85') & (en=='1'): # MIROC5 only happens in pad but not nopad (it has no 2300)
                    #print(i,model,en,gmt[-1])
                    axs[0].plot(gmt.time,gmt,color='royalblue',alpha=1,lw=1,zorder=10,label='MIROC5 RCP85')
                    axs[0].plot(gmt.time,gmt,color='k',alpha=1,lw=2,zorder=8)
                    pass
            ### plot rcp-mean and save the RCP-mean for pad and nopad data
            for rcp in sorted(tss.keys()):
                gmt_mean=xr.concat(tss[rcp],dim='rcp').mean(dim='rcp')
                axs[i].plot(gmt_mean.time,gmt_mean,color=rcp_colors[rcp],alpha=1,lw=2,label='%s (%s models)'%(rcp,len(tss[rcp])))
                axs[i].plot(gmt_mean.time,gmt_mean,color='k',alpha=1,lw=1)
                ### Save the timeseries for CMIP mean - for each RCP
                years=gmt_mean.time # idx0
                gmt_data=gmt_mean #idx1
                gmt_cum_data=gmt_mean.cumulative("time").sum() #idx2 (the cumulative data starts at 1990)
                if i==0: #pad
                    X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3
                elif i==1: #nopad (always 1)
                    X8=np.array([1 if year<=2100 else 1 for year in years]) #idx3
                gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
                save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
                var='tas'; modell='CMIP5mean_%s'%rcp; en='1'; region='global'# the mean of all models and ensembles
                np.save('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_%s_en%s_%s.npy'%(var,modell,pad_saves[i],en,region),save_data)
                # Get the name of models in a RCP i # [m for m in  models if 'rcp85' in m]
        ###
        if False: ## Include MIROC5 interpolartion (using the raw timeseries - nonfilter)
            #gmt_read=np.load('./save_gmt/tas_ts_create/old_cmip5/gmt_tas_MIROC5_85_en1_global.npy') # the same - confimed
            #gmt_aa=xr.DataArray(gmt_read[:,1],dims=['time'],coords={'time':gmt_read[:,0]})
            #axs[0].plot(gmt_aa.time,gmt_aa,color='k',zorder=9,lw=3)
            gmt=gmt_anoms['MIROC5_rcp85_1'] # THis is the non-filtered version
            x=gmt.sel(time=slice(2080,2100)).time.values
            y=gmt.sel(time=slice(2080,2100)).values
            slope,intercept, _, _, _ = scipy.stats.linregress(x,y)
            new_x=range(2101,2301)
            new_y=new_x*slope+intercept
            new_y=xr.DataArray(new_y, dims=['time'], coords={'time':new_x})
            new_gmt=xr.concat([gmt.sel(time=slice(1850,2100)),new_y], dim='time')
            ## plot the new gmt (old + interpolated)
            axs[1].plot(new_gmt.time,new_gmt,color='royalblue',alpha=1,lw=2,zorder=10,linestyle='-',label='MIROC5 r1i1p1 RCP85 interpolate')
            # filter the timeseries to check
            new_gmt_filter=scipy.signal.savgol_filter(new_gmt,filter_window,3)
            new_gmt_filter=new_gmt.copy(data=new_gmt_filter)
            axs[1].plot(new_gmt_filter.time,new_gmt_filter,color='blue',alpha=1,lw=1,zorder=30,linestyle='-')
            ## Save the file by calculating the ratio
            ratio=new_gmt/new_gmt.sel(time=2100).values
            save_array=np.array([new_gmt.time.values.astype(np.int64),np.around(new_gmt.values,5),np.round(ratio.values,5)]).T
            miroc5_extrapolate_gmt=new_gmt
            header='Year GMT ratio'
            fmt='%d','%1.5f','%1.5f'
            fn='gmt_tas_MIROC5_rcp85_nopad_extended_en1_global.txt'
            np.savetxt('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/%s'%fn, save_array, fmt=fmt, header=header)
            ## The filter GMT should be saved for training
        if False: # Include IPSL forcing up to 2200
            ### Read IPSL-CM6A-LR hist and SSP585 tas map file and create the timeseries
            path='/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_map_data/IPSL-CM6A-LR/*.nc'
            data_raw=xr.open_mfdataset(path,use_cftime=True)['tas']
            new_time=range(1850,2301)
            ## To monthly mean
            data=data_raw.coarsen(time=12).mean().assign_coords(time=new_time)
            ## Relative to 1960-1989
            data_clim=data.sel(time=slice(1850,1900)).mean(dim='time')
            data_anom=data-data_clim
            data=data_anom
            ## Create timtseries
            lons=data.lon.values; lats=data.lat.values
            lat1=-90; lat2=90; lon1=0; lon2=360 # Assume data is from -90 to +90
            gmt_ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
            gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':new_time})
            gmt_ts=gmt_ts.sel(time=slice(1850,2200))
            ipsl_gmt_ts=gmt_ts
            axs[1].plot(ipsl_gmt_ts.time,ipsl_gmt_ts,color='violet',alpha=1,lw=1,zorder=30,linestyle='-',label='IPSL-CM6A-LR SSP585')
        for ax in axs:
            ax.set_ylabel("GMT relative to\n1850-1900 (K)")
            #ax.set_yticks(range(0,20,2)); ax.set_ylim(-1,16)
            ax.axvline(x=2100,color='lightgray',linestyle='--')
            ax.grid()
            ax.legend(bbox_to_anchor=(0.02,0.5), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.2, labelspacing=0.1, fontsize=8)
        ## Save figure
        fig_name = 'IPSL_time_pad_check'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

