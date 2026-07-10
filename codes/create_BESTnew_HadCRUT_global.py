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
import pandas as pd


if __name__ == "__main__":

    #Estimated Jan 1951-Dec 1980 monthly absolute temperature for BEST
    #Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
    #best_month_anoms={1:12.23,2:12.44,3:13.06,4:13.97,5:14.95,6:15.67,7:15.95,8:15.79,9:15.19,10:14.26,11:13.24,12:12.49}

    # Move can move to 1850-1900 to make it consistent with IPCC and Climate Action Tracker (but some models have no 1850)
    ##! This works. Results are pretty much the same using 1850-1900 compared to 1960-1990
    clim_st_yr=1850; clim_end_yr=1900; ini_st_year=1850# but some models have no 1850. e.g., HadGEM2-ES starts at 1859. 
    final_st_yr=1850; final_end_yr=2300 

    ### Read BEST dataset (1-d timeseries from the globa-mean file)
    models=['BESTnew','HadCRUT'][::-1]
    for model in models:
        temps_absolute=[]
        if model=='BESTnew':
            new_time = pd.date_range(start='1850-01-01', end='2024-12-01', freq='MS')
            #% Global Average Temperature Anomaly with Sea Ice Temperature Inferred from Air Temperatures
            folder="/mnt/data/data_a/Berkeley_SAT_land/global_timeseries/"; filename="Land_and_Ocean_complete.txt"
            data=np.loadtxt(folder+filename, usecols=(0, 1, 2))
            for row in data:
                year=row[0]
                month=row[1]
                temp_anom=row[2]
                #temp_abs=temp_anom+best_month_anoms[month]+273.15 # Actually it will be the same no matter we can back the climatology or not
                temp_abs=temp_anom+273.15 # this is not important
                temps_absolute.append(temp_abs)
        elif model=='HadCRUT':
            new_time = pd.date_range(start='1850-01-01', end='2025-12-01', freq='MS')
            folder="/mnt/data/data_a/HadCRUT.5.1.0.0/"; filename="HadCRUT.5.1.0.0.analysis.summary_series.global.monthly.csv"
            data=np.loadtxt(folder+filename, usecols=(0, 1), skiprows=1, dtype=str, delimiter=',')
            for row in data:
                year=row[0].split('-')[0]
                mon=row[0].split('-')[1]
                temp_abs=row[1] # this is the anoamly relative to 1961-1990
                temps_absolute.append(float(temp_abs))
        ## Monthly to annual
        best_gmt=xr.DataArray(temps_absolute,dims=['time'],coords={'time':new_time})
        best_gmt=best_gmt.coarsen(time=12).mean()
        ## This is now the raw Annual-mean GMT from 1850 to 2023 (the climatology is added back)
        best_gmt=best_gmt.assign_coords({'time':best_gmt.time.dt.year.values})

        ## 1. Relative to clim year
        gmt_ts=best_gmt-best_gmt.sel(time=slice(clim_st_yr,clim_end_yr)).mean()

        ## 2. Do the running average or filtering
        if True: # do the filtering (this is the default)
            filter_window=51
            gmt_ts_filter=scipy.signal.savgol_filter(gmt_ts,filter_window,3)
            gmt_ts_filter=gmt_ts.copy(data=gmt_ts_filter)
            gmt_ts=gmt_ts_filter
            add_fn=''
        else: # Save an additional version for non-filter data
            pass
            add_fn='nofilter'

        ## 3. Finally select the year
        gmt_ts=gmt_ts.sel(time=slice(final_st_yr,final_end_yr))

        ## 4. save the data
        if True: ### Save the files
            var='tas'
            en='1'
            region='global'
            years=gmt_ts.time # idx0
            gmt_data=gmt_ts
            #gmt_cum_data=gmt_ts.cumulative("time").sum() #idx2 (the cumulative data starts at 1990)
            #X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3 (this is not useful in our case as the BEST stops in 2023)
            #gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
            #save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
            save_data=np.column_stack((years.values,gmt_data.values))
            np.save('../save_gmt/tas_ts_create/%s/gmt_%s_%s%s_en%s_%s.npy'%(model,var,model,add_fn,en,region),save_data)


    ### Compareing the New BEST and old BEST
    add_fn='nofilter'; models=['HadCRUT','BESTnew']
    models=['HadCRUT','BEST','BESTnew']; add_fn=''
    colors=['black','red','orange']
    plt.close()
    fig,ax1=plt.subplots(1,1,figsize=(5,2))
    for i, model in enumerate(models):
        ### Plot BEST and BESTnew and compare them
        gmt_read=np.load('../save_gmt/tas_ts_create/%s/gmt_%s_%s%s_en%s_%s.npy'%(model,var,model,add_fn,en,region))
        ## Plot the GMT
        gmt=xr.DataArray(gmt_read[:,1],dims=['time'],coords={'time':gmt_read[:,0]})
        #ipdb.set_trace()
        print(model,gmt)
        ax1.plot(gmt.time,gmt,label=model, color=colors[i])
    ax1.legend(bbox_to_anchor=(0.02,0.5), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.2, labelspacing=0.1, fontsize=8)
    ## Save
    fig_name = 'GMT_comparsion_BEST_BESTnew'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)


