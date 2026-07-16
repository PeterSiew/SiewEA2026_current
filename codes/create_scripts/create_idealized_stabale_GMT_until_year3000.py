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
from scipy.signal import detrend
import scipy


if __name__ == "__main__":

    ### Read all current policies, and get the median (or the mean)
    cat='current';
    gmt_tss=[]
    for no in range(0,1200):
        ## This is the no-pad timeseries, so linear trends increase
        gmt_read=np.load('../save_gmt/tas_ts_create/policy_full600/'+
                                'gmt_tas_distno%s_%s_nopad2300_en1_global.npy'%(no,cat))
        gmt_ts=gmt_read[:,1]
        gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
        gmt_tss.append(gmt_ts)
    gmt_ts_mean=xr.concat(gmt_tss,dim='no').mean(dim='no')
    #ipdb.set_trace()
    sudden_cooling=True # sudden cooling at 2degree when the temperature keep fixed (don't use this anymore - not very interesting)
    sudden_cooling=False # default. The temp keeps fixed at years 2000, 2050, 2100, 2150...
    stabalization=True # default
    yr_extend=3000 # the profile is pad up to this year (could be 4000 or 5000)

    ### Creat stabaliztion profiles
    years_stable=[2000,2050,2100,2150,2200,2250,2300]
    for year_stable in years_stable:
        gmt_ts_old=gmt_ts_mean.sel(time=slice(0,year_stable-1))
        gmt_ts_discard=gmt_ts_mean.sel(time=slice(year_stable,2301)) # discard this
        if stabalization: # Default - stabalization profile after years stable
            pad_data=xr.DataArray([gmt_ts_old.isel(time=-1)]*(yr_extend+1-year_stable),dims=['time'],coords={'time':range(year_stable,yr_extend+1)})
        elif sudden_cooling: #sudden cooling at 0 degree
            pad_data=xr.DataArray([0]*len(gmt_ts_discard),dims=['time'],coords={'time':gmt_ts_discard.time})
        ### Add them together
        gmt_ts_new=xr.concat([gmt_ts_old,pad_data], dim='time')
        ### Save the GMT
        years=gmt_ts_new.time #idx0
        gmt_data=gmt_ts_new #idx1
        #gmt_cum_data=gmt_data.cumulative("time").sum() #idx2
        #X8=np.array([1 if year<=year_stable else year-year_stable for year in years]) #idx3: the stable year can be 2150, for example
        #gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
        ## Save the data
        #save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
        save_data=np.column_stack((years.values,gmt_data.values))
        var='tas';  en='1'; region='global'
        pad_save='nopad2300' # this file is modified from the nopad file, although the GMT is fixed after a certain point
        if stabalization:
            save_name='stable'+str(year_stable)+'to%s'%yr_extend+'_'+cat+'_'+pad_save
            #ipdb.set_trace()
        elif sudden_cooling:
            save_name='stable'+str(year_stable)+'cooling0''_'+cat+'_'+pad_save
        np.save('../save_gmt/tas_ts_create/policy_current_stable_yr3000/gmt_%s_%s_en%s_%s.npy'%(var,save_name,en,region),np.float64(save_data))

    if True:
        ### Plot the saved timeseries
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(5,2))
        ax1.plot(gmt_ts_mean.time,gmt_ts_mean,color='blue',zorder=10) # Plot the original policy high
        ### Plot stablising exps
        for year_stable in years_stable:
            if stabalization: # default
                save_name='gmt_tas_stable%sto%s_current_nopad2300_en1_global.npy'%(year_stable,yr_extend)
                gmt_read=np.load('../save_gmt/tas_ts_create/policy_current_stable_yr3000/%s'%save_name)
            elif sudden_cooling: ## Plot sudden cooling
                #save_name='stable'+str(year_stable)+'_'+cat+'_'+pad_save
                #gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/policyhigh_stable/gmt_%s_%s_en%s_%s.npy'%('tas',save_name,'1','global'))
                pass
            gmt_ts=gmt_read[:,1]
            gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
            ax1.plot(gmt_ts.time,gmt_ts,color='k')
        ## Save figure
        fig_name='idealized_GMT_scenarios'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0.6) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
