import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import matplotlib
from importlib import reload
import pandas as pd
import cartopy.crs as ccrs
import random

import sys
import os
import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
from scipy.signal import detrend
import scipy


if __name__ == "__main__":

    if True: ### Create the timeseries
        ## This data is already relative to 1850-1900 (according to the email-coversation)
        path='/Users/home/siewpe/codes/greenland_emulator/save_gmt/CATs/'
        scenarios={'current':['CPP_low','CPP_high'],
                    'target':['Targets_2030_and_2035'],
                    'pledge':['Pledges_and_Targets','Pledges_low'],
                    'optim':['Optimistic_scenario']}
        filename='GMT_all_scenarios_Nov2025.xlsx' # This file is sent on 18 Nov 2025
        data_tss={}
        for scenario in scenarios:
            tss=[]
            for scen in scenarios[scenario]:
                data_raw= pd.read_excel(path+filename,header=None,sheet_name=scen)
                ts=data_raw.values[13:,7:]
                tss.append(ts)
            data_tss[scenario]=np.array(np.vstack(tss),dtype='float')
        raw_years=np.arange(1750,2105+1)

        ### Scenarios
        final_st_yr=1850; final_end_yr=2300 
        filter_window=51
        en='1'; region='global'; var='tas'
        for scenario in scenarios:
            print(scenario)
            scen_tss=data_tss[scenario] # 600 or 1200 x all years
            ## Loop via all 600 or 1200 timeseries
            for j, gmt_ts in enumerate(scen_tss): 
                gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':raw_years})
                gmt_ts=gmt_ts.sel(time=slice(final_st_yr,2100))
                ## Create pad
                time_pad=np.arange(2101,final_end_yr+1) 
                ending_value=gmt_ts.isel(time=-1).values.item() # 2100 data
                pad_data=xr.DataArray([ending_value]*len(time_pad),dims=['time'],coords={'time':time_pad})
                gmt_ts_pad=xr.concat([gmt_ts,pad_data], dim='time')
                gmt_ts_pad=gmt_ts_pad.sel(time=slice(final_st_yr,final_end_yr))
                ## Create nopad (extrapolate till 2300)
                x=gmt_ts.sel(time=slice(2080,2100)).time.values
                y=gmt_ts.sel(time=slice(2080,2100)).values
                slope,intercept, _, _, _ = scipy.stats.linregress(x,y)
                new_x=range(2101,final_end_yr+1)
                new_y=new_x*slope+intercept
                new_y=xr.DataArray(new_y, dims=['time'], coords={'time':new_x})
                gmt_ts_nopad=xr.concat([gmt_ts,new_y], dim='time')
                ## Create them for saving
                ## Save the timeseries for both pad and nopad
                gmt_tss=[gmt_ts_pad,gmt_ts_nopad] # both from 1850 to 2300
                pad_saves=['pad2300','nopad2300']
                for i, pad_save in enumerate(pad_saves):
                    gmt_ts=gmt_tss[i]
                    if True: ## Do the filter after padding (it doesn't make much difference after year 2015 - but just to be more consitent)
                        ### Without filtering, the GMTs would be exactly equal to the one publlished online on CAT
                        gmt_ts_filter=scipy.signal.savgol_filter(gmt_ts,filter_window,3)
                        gmt_ts=xr.DataArray(gmt_ts_filter,dims=['time'],coords={'time':gmt_ts.time.values})
                    ## Set the format
                    gmt_years=gmt_ts.time #idx0
                    gmt_data=gmt_ts #idx1
                    gmt_cum_data=gmt_ts.cumulative("time").sum() #idx2
                    if pad_save=='pad2300':
                        X8=np.array([1 if year<=2100 else year-2100 for year in gmt_years]) #idx3
                    elif pad_save=='nopad2300':
                        X8=np.array([1 if year<=2100 else 1 for year in gmt_years]) #idx3
                    gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
                    ## Save the data
                    save_data=np.column_stack((gmt_years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
                    save_name='distno'+str(j)+'_'+scenario+'_'+pad_save
                    np.save('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policy_full600/gmt_%s_%s_en%s_%s.npy'%(var,save_name,en,region),np.float64(save_data))


    if True: ### Plotting the data after creating the timeseries
        plt.close()
        fig,axs=plt.subplots(2,1,figsize=(5,5))
        if True: ## Plot Berkerley GMST observations (we should change to use GSAT instead of GMST - 20th v3 reanalysis + ERA5)
            lable_name='Berkeley Earth Surface Temperatures'
            save_name='BEST' # This is also set to be relative to 1850-1900
            gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_%s_%s_en%s_%s.npy'%('tas',save_name,'1','global'))
            gmt_ts_obs=gmt_read[:,1]
            gmt_ts_obs=xr.DataArray(gmt_ts_obs,dims=['time'],coords={'time':gmt_read[:,0]})
            gmt_ts_obs=gmt_ts_obs.sel(time=slice(1850,2021))
            axs[0].plot(gmt_ts_obs.time, gmt_ts_obs, linestyle='-', lw=1, color='k', label='BEST',zorder=5)
            axs[1].plot(gmt_ts_obs.time, gmt_ts_obs, linestyle='-', lw=1, color='k', label='BEST',zorder=5)
        ## Plot CAT
        pad_saves=['pad2300','nopad2300']
        #cat_colors={'policyhigh':'red','shortpledge':'orange','longpledge':'yellow','optimistic':'green'}
        cat_colors={'current':'red','target':'orange','pledge':'yellow','optim':'green'}
        cat_numbers={'current':1200,'target':600,'pledge':1200,'optim':600}
        cat_names=['current','target','pledge','optim']
        gmt_ts_save={p:{} for p in pad_saves}
        for i, pad_save in enumerate(pad_saves):
            for cat in cat_names:
                gmt_tss=[]
                for no in range(cat_numbers[cat]):
                    save_name='distno'+str(no)+'_'+cat+'_'+pad_save
                    gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policy_full600/gmt_%s_%s_en%s_%s.npy'%('tas',save_name,'1','global'))
                    gmt_ts=gmt_read[:,1]
                    gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
                    gmt_tss.append(gmt_ts)
                gmt_ts_mean=xr.concat(gmt_tss,dim='no').median(dim='no')
                gmt_ts_83=xr.concat(gmt_tss,dim='no').quantile(0.83,dim='no')
                gmt_ts_17=xr.concat(gmt_tss,dim='no').quantile(0.17,dim='no')
                gmt_ts_save[pad_save][cat]=gmt_ts_mean
                axs[i].plot(gmt_ts_mean.time,gmt_ts_mean,color=cat_colors[cat],ls='-',label='%s (q50)'%cat,lw=2)
                axs[i].plot(gmt_ts_83.time,gmt_ts_83,color=cat_colors[cat],ls='--',lw=1)
                axs[i].plot(gmt_ts_17.time,gmt_ts_17,color=cat_colors[cat],ls='--',lw=1)
        axs[0].set_title('pad2300 - constant GMT up to 2300')
        axs[1].set_title('nopad2300 - increasing GMT up to 2300')
        if False: ## Plot action tracker file (public online)
            hist_years=[1990,2000,2010,2020]; hist_gmt=np.array([0.6252,0.7616,0.9899,1.2359])
            axs[0].scatter(hist_years,hist_gmt,s=10,color='brown',zorder=10,label='CAT historical period online')
        ## Add vertical line
        for ax in axs:
            ax.axvline(x=2015,color='k',linestyle='--',lw=0.9,label='2015')
            ax.axvline(x=2100,color='k',linestyle='-.',lw=0.9,label='2100')
            ax.axhline(y=0,color='lightgray',linestyle='--',zorder=-1)
            ax.set_ylim(-0.5,10)
        ## Add legend
        axs[0].legend(bbox_to_anchor=(0.01,0.4),ncol=1,loc='lower left',frameon=False,columnspacing=0.5,handletextpad=0.2,labelspacing=0.1,fontsize=7)
        ## Save figure
        fig_name='CAT_Nov2025_timeseries_check'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0.3) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)




def testing():

    ### Read file
    path='/Users/home/siewpe/codes/greenland_emulator/save_gmt/CATs/'
    filename='CAT_GMT_extended_2025Nov.csv' # This file is sent on 18 Nov 2025
    data=np.genfromtxt(path+filename,delimiter=',',dtype='str')
    ## The timeseries here is relative to 1750. Not 1850-1900 yet
    data_ts=np.array(data[1:,7:],dtype='float') # skip the first row, and the timeseries start in column idx=7. It has a shape of 4800 x 356 (year 1750 to 2105)
    years=np.arange(1750,2105+1)
    ###
    pi_years=np.arange(1750,1850)
    pi_years=np.arange(1850,1890)
    pi_years=np.arange(1850,1901) # Default
    pi_years_idx=np.isin(years,pi_years).nonzero()[0]
    online_years=[1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100]
    online_years_idx=np.isin(years,online_years).nonzero()[0]
    scenarios=data[1:,4] #1 is to skip the first row. It has a shape of 4800 (8 scenarios and 600 samples each)
    set(scenarios)

    idx=scenarios=='Targets_2030_and_2035'
    idx=scenarios=='Targets_2030_only'
    idx=scenarios=='Optimistic_low'
    idx=scenarios=='Pledges_low'
    idx=scenarios=='CPP_low'
    idx=scenarios=='Optimistic_scenario'
    idx=scenarios=='Pledges_and_Targets'
    idx=scenarios=='CPP_high'
    if False: # Cor CPP_combined
        idx1=(scenarios=='CPP_high').nonzero()[0]
        idx2=(scenarios=='CPP_low').nonzero()[0]
        idx=np.append(idx1,idx2)
        idx=np.sort(idx)
    scen_ts=data_ts[idx,:]
    scen_median=np.median(scen_ts,axis=0)
    #scen_median=np.mean(scen_ts,axis=0)
    if True: # Relative to 1850-1900
        scen_pi_avg=scen_ts[:,pi_years_idx].mean(axis=1) # the PI average over 1950-1900
        scen_ts_new=scen_ts-scen_pi_avg[:,np.newaxis]
        scen_median_new=np.median(scen_ts_new,axis=0)
        #scen_median_new=np.mean(scen_ts_new,axis=0)
    print(scen_median_new[online_years_idx])
    print(scen_median[online_years_idx])
    #np.percentile(data_ts[idx,:],axis=0)

    if True: ### Plot the historgram of the GMT in 2100
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(4,4))
        bin_no=20
        # index -6 is year 2100
        n, bins, patches = ax1.hist(scen_ts_new[:,-5], bins=bin_no, density=False, color='royalblue',alpha=0.5, edgecolor='royalblue')
        ### Save figures
        fig_name = 'CAT_new_GMT'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.6) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
