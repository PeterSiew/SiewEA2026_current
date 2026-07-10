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
from statsmodels.tsa.api import Holt
from scipy.interpolate import CubicHermiteSpline
from scipy.signal import detrend
import scipy
import sys
import os


if __name__ == "__main__":

    ### Read all current policies, and get the median
    models_colors={'optim':'royalblue',
                   'pledge':'lightskyblue',
                   'target':'lightsalmon',
                   'current':'orangered'}
    cats=['optim','pledge','target','current']
    cats_full_dist={'optim':600,'pledge':1200,'target':600,'current':1200}
    gmt_tss_stable={cat:None for cat in cats}
    gmt_tss_damped={cat:None for cat in cats}
    gmt_tss_overshoot={cat:None for cat in cats} 
    gmt_tss_linear={cat:None for cat in cats} 
    for i, cat in enumerate(cats):
        ### Get the mean of the CAT scenario
        gmt_ts_mean=[]
        for no in range(cats_full_dist[cat]):
            gmt_read=np.load('../save_gmt/tas_ts_create/policy_full600/'+'gmt_tas_distno%s_%s_nopad2300_en1_global.npy'%(no,cat))
            gmt_ts=gmt_read[:,1]
            gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':gmt_read[:,0]})
            gmt_ts_mean.append(gmt_ts)
        ## Take the average of the policy scenarios
        gmt_ts_mean=xr.concat(gmt_ts_mean,dim='en').mean(dim='en')
        #if cat=='current':
            #ipdb.set_trace()
        gmt_tss_linear[cat]=gmt_ts_mean
        ## 1. Create the stabalized profile (already created) from 2100 to 2300 (directly read the nopad one)
        if True:  
            gmt_ts_keep=gmt_ts_mean.sel(time=slice(1850,2100))
            time_pad=range(2101,2301)
            ## Replace 2100 afterwards to a flat line (stabalized)
            gmt2100=gmt_ts_mean.sel(time=2100).item()
            pad_data=xr.DataArray([gmt2100]*len(time_pad),dims=['time'],coords={'time':time_pad})
            gmt_combine=xr.concat([gmt_ts_keep,pad_data], dim='time')
            gmt_tss_stable[cat]=gmt_combine
        ## 2. Modify the nopad (with linear trends) to get the damped trends
        if True:  
            gmt_linear=gmt_ts_mean.sel(time=slice(2080,2100)) # Use the trends from 2100-2150 to get the damped trend
            model_damped=Holt(gmt_linear.values, damped_trend=True).fit(damping_trend=0.995)
            gmt_damped=model_damped.forecast(len(range(2101,2301)))
            gmt_damped=xr.DataArray(gmt_damped,dims=['time'],coords={'time':range(2101,2301)})
            gmt_combine=xr.concat([gmt_ts_mean.sel(time=slice(1850,2100)),gmt_damped], dim='time')
            gmt_tss_damped[cat]=gmt_combine
        ## 3.Create the overshoot (old)
        if False: 
            gmt_ts_keep=gmt_ts_mean.sel(time=slice(1850,2100))
            ## Get the slope
            #current_slope = np.mean(np.diff(gmt_ts_keep.sel(time=slice(2080,2100))))
            current_slope= gmt_ts_mean.differentiate("time").sel(time=2100)
            ## Define the Scenario Control Points; # We define years of Start, Peak, Floor
            ## d-slope control slopes: [Match current, 0 at peak, 0 at floor]
            x_nodes=[2100,2200,2300]; y_nodes=[gmt_ts_mean.sel(time=2100),gmt_ts_mean.sel(time=2200).values,2]; d_nodes = [current_slope,0,0]
            x_nodes=[2100,2200,2300]; y_nodes=[gmt_ts_mean.sel(time=2100),2,2]; d_nodes = [current_slope,0,0]
            x_nodes=[2100,2150,2300]; y_nodes=[gmt_ts_mean.sel(time=2100),gmt_ts_mean.sel(time=2150),gmt_ts_mean.sel(time=2100)]; d_nodes = [current_slope,0,0]
            #y_nodes = [gmt_ts_keep.values[-1], gmt_ts_keep.values[-1]*1.3, gmt_ts_keep.values[-1]]
            #y_nodes = [gmt_ts_keep.values[-1], gmt_ts_keep.values[-1]*1.3, 2]
            ## Create the Smooth Curve
            spline=CubicHermiteSpline(x_nodes, y_nodes, d_nodes)
            new_time=np.arange(2101,2301)
            gmt_new=spline(new_time)
            gmt_new=xr.DataArray(gmt_new,dims=['time'],coords={'time':new_time})
            ## Combine the old and new parts
            gmt_ts_overshoot=xr.concat([gmt_ts_keep,gmt_new],dim='time')
            gmt_tss_overshoot[cat]=gmt_ts_overshoot
            #ipdb.set_trace()
        ## 3. Create the new overshoot
        if True: 
            gmt_ts_keep=gmt_ts_mean.sel(time=slice(1850,2100))
            ## Set 2300 to be 2dec C
            gmt_2300=xr.DataArray([2],dims=['time'],coords={'time':[2300]})
            gmt_combine=xr.concat([gmt_ts_keep,gmt_2300],dim='time')
            gmt_interpolate=gmt_combine.interp(time=range(1850,2301))
            ## Calculate the rolling value
            gmt_interpolate=gmt_interpolate.rolling(time=20, center=True, min_periods=1).mean()
            gmt_tss_overshoot[cat]=gmt_interpolate
    #ipdb.set_trace()

    if True: # Save the GMT timeseries
        var='tas'
        region='global'
        methods=['extra2300linear','extra2300stabalization','extra2300damped','extra2300overshoot']
        model_folder="policy_stable_damped_overshoot"
        en='1'
        gmts_save=[gmt_tss_linear,gmt_tss_stable,gmt_tss_damped,gmt_tss_overshoot]
        for cat in cats:
            for i, method in enumerate(methods):
                model=method+'_%s'%cat
                gmt=gmts_save[i][cat]
                years=gmt.time # idx0
                gmt_data=gmt
                gmt_cum_data=gmt.cumulative("time").sum() #idx2 (the cumulative data starts at 1990)
                X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3 
                gmt_cum_data_gradient=gmt.differentiate("time") #idx4
                save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
                np.save('../save_gmt/tas_ts_create/%s/gmt_%s_%s_en%s_%s.npy'%(model_folder,var,model,en,region),save_data)


    if True: ### Plot the timeseries
        plt.close()
        fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,2))
        ax_all=[ax1,ax2,ax3]
        titles=['Damped trends','Stabilized','Overshoot']
        cat_models_label=['Optimistic','Pledges and targets','2030 & 2035 targets','Current policy actions']
        lw=2
        for i, cat in enumerate(cats):
            ## The original no-pad timeseries
            gmt_linear=gmt_tss_linear[cat]
            ## Start plotting the damped trend (left)
            gmt_damped=gmt_tss_damped[cat]
            ax1.plot(gmt_damped.time,gmt_damped,color=models_colors[cat],zorder=10,ls='-',lw=lw,label=cat_models_label[i]) # Plot the original policy high
            ax1.plot(gmt_linear.time,gmt_linear,color=models_colors[cat],zorder=10,ls='--',lw=lw/2.0) # Plot the original policy high
            ## Start plot the stabalize profile (middle)
            gmt_stable=gmt_tss_stable[cat]
            ax2.plot(gmt_stable.time,gmt_stable,color=models_colors[cat],zorder=10,ls='-',lw=lw) # Plot the original policy high
            ax2.plot(gmt_linear.time,gmt_linear,color=models_colors[cat],zorder=10,ls='--',lw=lw/2.0) # Plot the original policy high
            ## Start plot overshopping scenario (right)
            gmt_overshoot=gmt_tss_overshoot[cat]
            ax3.plot(gmt_overshoot.time,gmt_overshoot,color=models_colors[cat],zorder=10,ls='-',lw=lw) # Plot the original policy high
            ax3.plot(gmt_linear.time,gmt_linear,color=models_colors[cat],zorder=10,ls='--',lw=lw/2.0) # Plot the original policy high
        ## Set legend
        ax1.legend(bbox_to_anchor=(-0,0.5), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,
                    handletextpad=0.5, labelspacing=0.2,fontsize=9,reverse=True)
        ## set axis
        for ax in ax_all:
            ax.set_xticks([2100,2200,2300])
            ax.axvline(x=2100,color='lightgray',linestyle='--',lw=1)
            ax.axvline(x=2200,color='lightgray',linestyle='--',lw=1)
            ax.set_yticks([0,1,2,3,4,5])
            ax.set_ylim([0.8,6])
            ax.set_xlim([2015,2300])
        for i, ax in enumerate(ax_all):
            #ax.axvline(x=2015,color='k',linestyle='--',lw=0.9)
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
            ax.set_title(titles[i],loc='left')
            ax.set_xlabel("Year")
            if i==0:
                ax.set_ylabel("Global-mean\ntemperature (K)")
        ## Save figure
        fig_name='stabalize_damped_overshoot'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.2,hspace=0.6) # hspace is the vertical
        plt.savefig('../../graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
