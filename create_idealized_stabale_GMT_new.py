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

    ### Read all current policies, and get the median
    #cat='policyhigh';
    cat='current';
    gmt_tss=[]
    for no in range(0,1200):
        gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policy_full600/gmt_tas_distno%s_%s_nopad2300_en1_global.npy'%(no,cat))
        gmt_ts=gmt_read[:,1]
        gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
        gmt_tss.append(gmt_ts)
    gmt_ts=xr.concat(gmt_tss,dim='no').median(dim='no')
    sudden_cooling=True # sudden cooling at 2degree when the temperature keep fixed (don't use this anymore - not very interesting)
    sudden_cooling=False # default. The temp keeps fixed at years 2000, 2050, 2100, 2150...
    year4000_extend=True # The pad will go until year-4000
    year4000_extend=False # The pad is only up to year-2300
    year3000_extend=False
    year3000_extend=True

    
    ### temperature starts to be stalized
    #temps=[2,3,4,5]
    years_stable=[2000,2050,2100,2150,2200,2250,2300]
    for year_stable in years_stable:
        gmt_ts_old=gmt_ts.sel(time=slice(0,year_stable-1))
        gmt_ts_discard=gmt_ts.sel(time=slice(year_stable,2301)) # discard this
        if sudden_cooling: #sudden cooling at 0 degree
            pad_data=xr.DataArray([0]*len(gmt_ts_discard),dims=['time'],coords={'time':gmt_ts_discard.time})
        else:  # Default without sudden_colling
            pad_data=xr.DataArray([gmt_ts_old.isel(time=-1)]*len(gmt_ts_discard),dims=['time'],coords={'time':gmt_ts_discard.time})
            if year4000_extend: ### Long extend period - just for testing
                #pad_data=xr.DataArray([gmt_ts_old.isel(time=-1)]*(3001-year_stable),dims=['time'],coords={'time':range(year_stable,3001)})
                pad_data=xr.DataArray([gmt_ts_old.isel(time=-1)]*(4001-year_stable),dims=['time'],coords={'time':range(year_stable,4001)})
            if year3000_extend: ### Long extend period - just for testing
                pad_data=xr.DataArray([gmt_ts_old.isel(time=-1)]*(3001-year_stable),dims=['time'],coords={'time':range(year_stable,3001)})
        ### Add them together
        gmt_ts_new=xr.concat([gmt_ts_old,pad_data], dim='time')
        ### Save the GMT
        years=gmt_ts_new.time #idx0
        gmt_data=gmt_ts_new #idx1
        gmt_cum_data=gmt_data.cumulative("time").sum() #idx2
        X8=np.array([1 if year<=year_stable else year-year_stable for year in years]) #idx3: the stable year can be 2150, for example
        gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
        ## Save the data
        save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
        var='tas';  en='1'; region='global'
        pad_save='nopad2300' # this file is modified from the nopad file, although the GMT is fixed after a certain point
        if sudden_cooling:
            save_name='stable'+str(year_stable)+'cooling0''_'+cat+'_'+pad_save
        else:
            save_name='stable'+str(year_stable)+'_'+cat+'_'+pad_save
            if year4000_extend:
                save_name='stable'+str(year_stable)+'to4000'+'_'+cat+'_'+pad_save
            if year3000_extend:
                save_name='stable'+str(year_stable)+'to3000'+'_'+cat+'_'+pad_save
        np.save('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policy_current_stable/gmt_%s_%s_en%s_%s.npy'%(var,save_name,en,region),np.float64(save_data))

    if False:
        ### Plot the saved timeseries
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(5,2))
        ### Plot policy high
        gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/gmt_tas_q0.5_policyhigh_nopad2300_en1_global.npy')
        gmt_ts=gmt_read[:,1]
        gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
        ax1.plot(gmt_ts.time,gmt_ts,color='r',zorder=10)
        ### Plot stablising exps
        #for temp in temps:
        for year_stable in years_stable:
            if sudden_cooling: ## Plot sudden cooling
                save_name='stable'+str(year_stable)+'cooling0'+'_'+cat+'_'+pad_save
                gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policyhigh_stable/gmt_%s_%s_en%s_%s.npy'%('tas',save_name,'1','global'))
                gmt_ts=gmt_read[:,1]
                gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
                ax1.plot(gmt_ts.time,gmt_ts,color='k')
            else: # Default
                save_name='stable'+str(year_stable)+'_'+cat+'_'+pad_save
                if year4000_extend:
                    save_name='stable'+str(year_stable)+'to4000'+'_'+cat+'_'+pad_save
                gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policyhigh_stable/gmt_%s_%s_en%s_%s.npy'%('tas',save_name,'1','global'))
                gmt_ts=gmt_read[:,1]
                gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
                ax1.plot(gmt_ts.time,gmt_ts,color='k')

        ## Save figure
        fig_name='idealized_GMT_scenarios'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0.6) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
