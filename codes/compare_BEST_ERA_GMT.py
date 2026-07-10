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

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools

if __name__ == "__main__":

    ### Read BEST and HARDCUT
    var='tas'
    en='1'
    region='global'
    models=['HadCRUT','BESTnew']; add_fn='nofilter' # use the new BEST file (definded from the provided timeseries)
    models=['HadCRUT','BEST']; add_fn='nofilter' # use the old BEST file (defined from the lat-lon regrid data)
    model_tss={}
    for i, model in enumerate(models):
        ### Plot BEST and BESTnew and compare them
        gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/%s/gmt_%s_%s%s_en%s_%s.npy'%(model,var,model,add_fn,en,region))
        gmt=xr.DataArray(gmt_read[:,1],dims=['time'],coords={'time':gmt_read[:,0]})
        model_tss[model]=gmt

    ipdb.set_trace()

    ### Read BEST and ERA5 2d data
    models=['ERA5']
    paths={'ERA5':'/mnt/data/data_a/ERA5/T2M/T2M_monthly.nc'}
    vars_nc={'ERA5':'t2m'}
    region_domains={'ERA5':(-90, 90, 0, 360)}
    st_yr='1940'; end_yr='2023'
    for model in models:
        data=xr.open_dataset(paths[model])
        data=data[vars_nc[model]]
        if model=='ERA5': ### Reverse ERA5 lat (90 to -90) before doing the average
            # reverse the latitude if it is ERA5
            data=data.isel(latitude=slice(None, None, -1)) 
            data=data.rename({'latitude':'lat', 'longitude':'lon'})
        ### 3. Calculate annual-mean from monthly data
        data=data.coarsen(time=12).mean()
        years=data.time.dt.year
        data=data.assign_coords({'time':years})
        if True: ### 4. Select the time
            data=data.sel(time=slice('%s-01-01'%st_yr,'%s-12-31'%end_yr))
        ### 5. Calculate the area average (both GMTs and Greenland)
        #print('number of nan data: ', np.isnan(data).sum().item())
        lat1,lat2,lon1,lon2=region_domains[model]
        lons=data.lon.values; lats=data.lat.values
        gmt_ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
        gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':data.time})
        model_tss[model]=gmt_ts

    models=['HadCRUT','BESTnew','ERA5']
    models=['HadCRUT','BEST','ERA5']
    ## Relative to 1980-2000
    for model in models:
        gmt_ts=model_tss[model]
        gmt_ts=gmt_ts-gmt_ts.sel(time=slice(1980,2000)).mean()
        model_tss[model]=gmt_ts

    #########################
    model_colors={'ERA5':'k','BESTnew':'green', 'HadCRUT':'red'}
    model_labels={'ERA5':'ERA5','BESTnew':'BEST', 'HadCRUT':'HadCRUT5'}
    if True: 
        model_colors={'ERA5':'k','BEST':'green', 'HadCRUT':'red'}
        model_labels={'ERA5':'ERA5','BEST':'BEST', 'HadCRUT':'HadCRUT5'}
    ipdb.set_trace()
    ### Plot the timeseries
    plt.close()
    fig, ax1=plt.subplots(1,1,figsize=(4,2))
    for model in models:
        ax1.plot(model_tss[model].time,model_tss[model],color=model_colors[model],label=model_labels[model],lw=1)
    ax1.legend(bbox_to_anchor=(0.05,0.6), ncol=1, loc='lower left', frameon=True, 
                        columnspacing=1.5,handletextpad=0.5, labelspacing=0.2,fontsize=10)
    ax1.set_ylabel("GMT (K)\nrelative to 1980-2000")
    ax1.axhline(y=0,color='lightgray',linestyle='--',lw=0.9,zorder=100)
    ## Save figure
    fig_name = 'timeseries_compartion_ERA5_BEST'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.02)
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)


def test():

    # for testing the new and old BEST timeseries
    tas_or_tos='tas'
    model_read='BEST'
    gmt_en='1'
    region='global'
    gmt_read=np.load('/Users/home/siewpe/codes/greenland_emulator/save_gmt/tas_ts_create/BEST/gmt_%s_%s_en%s_%s.npy'
                                                    %(tas_or_tos,model_read,gmt_en,region))
    ## Finish reading - starting assigning the GMT data
    gmt=xr.DataArray(gmt_read[:,1],dims=['time'],coords={'time':gmt_read[:,0]})
    cum_gmt=xr.DataArray(gmt_read[:,2],dims=['time'],coords={'time':gmt_read[:,0]})
    last_change=xr.DataArray(gmt_read[:,3],dims=['time'],coords={'time':gmt_read[:,0]})
    cum_gmt_gradient=xr.DataArray(gmt_read[:,4],dims=['time'],coords={'time':gmt_read[:,0]})
    gmt_years=gmt.time.values # this is the years from original gmt

    ipdb.set_trace()

