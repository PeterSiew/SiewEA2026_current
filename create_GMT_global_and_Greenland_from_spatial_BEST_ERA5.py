import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import matplotlib
from importlib import reload
import pandas as pd
import cartopy.crs as ccrs

import sys
import os
import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
from scipy.signal import detrend
import scipy

if __name__ == "__main__":

    if False:
        ## For predicting models with data up to 2300 (just for testing - not used anymore)
        models=['CCSM4_26','CCSM4_60','CCSM4_85']; before2016_miroc26en1=False; time_pad_st=2101 # for testing
        models=['IPSL-CM5A-LR_26','IPSL-CM5A-LR_60','IPSL-CM5A-LR_85']; before2016_miroc26en1=False; time_pad_st=2101 # for testing

    ## For validating models appraoch
    models=['MIROC5_26','HadGEM2-ES_85', 'NorESM1-M_85']; before2016_miroc26en1=False; time_pad_st=2101 
    models=['MIROC5_26', 'MIROC5_45', 'MIROC5_60', 'MIROC5_85']; before2016_miroc26en1=False; time_pad_st=2101 # Actually it doesn't matter to replace or not


    ## For real prediction (the data before 2016 really should come from that models)
    models=['CCSM4_26','CCSM4_45','CCSM4_60','CCSM4_85', # CCSM4 data has some problem
            'IPSL-CM5A-LR_26','IPSL-CM5A-LR_45','IPSL-CM5A-LR_60','IPSL-CM5A-LR_85',
            'IPSL-CM5A-MR_26','IPSL-CM5A-MR_45','IPSL-CM5A-MR_60','IPSL-CM5A-MR_85',
            'CESM1-CAM5_26', 'CESM1-CAM5_45', 'CESM1-CAM5_60', 'CESM1-CAM5_85',
            'MIROC5_26', 'MIROC5_45', 'MIROC5_60', 'MIROC5_85',
            'NorESM1-M_26', 'NorESM1-M_45', 'NorESM1-M_60', 'NorESM1-M_85',
            'NorESM1-ME_26', 'NorESM1-ME_45', 'NorESM1-ME_60', 'NorESM1-ME_85',
            'GISS-E2-H_26', 'GISS-E2-H_45', 'GISS-E2-H_60', 'GISS-E2-H_85']; before2016_miroc26en1=False; time_pad_st=2101; create_CMIP5_model_rcp_mean=True

    ## For observations
    # The Best global TAS is replaced by berkeryearth_global_tas.py (this creates a slighly larger GMT, which is more consistent with Greeland ice loss)
    models=['ERA5']; before2016_miroc26en1=False; time_pad_st=np.nan # ERA5 has no 1850. Produce nan
    models=['BEST']; before2016_miroc26en1=False; time_pad_st=np.nan # We don't do padding (i.e, extend up to 2300) for obs because now it can be directly smoothed by filter

    ensembles=range(1,2)  # Only the first enseble
    #rolling_year=1 # it doesn't help to set like this
    rolling_year=30 # default
    filter_window=51
    regions=['global','greenland']
    if True: # the lastest solution
        # Move can move to 1850-1900 to make it consistent with IPCC and Climate Action Tracker (but some models have no 1850)
        ##! This works. Results are pretty much the same using 1850-1900 compared to 1960-1990
        clim_st_yr=1850; clim_end_yr=1900; ini_st_year=1850# but some models have no 1850. e.g., HadGEM2-ES starts at 1859. 
        final_st_yr=1850; final_end_yr=2300 
    else: # old method
        clim_st_yr=1960; clim_end_yr=1989; ini_st_year=1960 # According to ISMIP6 paper (defaut).
        final_st_yr=1990; final_end_yr=2300 ##! Should start from 2015 to 2300 to make less confusion (actually no because we want to predict ERA5 1992-2021)

    ###
    if True: # TAS
        var='tas'
        model_hist_files={
                'MIROC5_hist':['tas_Amon_MIROC5_historical_r1i1p1_185001-201212.nc'],
                'HadGEM2-ES_hist':['tas_Amon_HadGEM2-ES_historical_r1i1p1_185912-188411.nc',
                            'tas_Amon_HadGEM2-ES_historical_r1i1p1_188412-190911.nc','tas_Amon_HadGEM2-ES_historical_r1i1p1_190912-193411.nc',
                            'tas_Amon_HadGEM2-ES_historical_r1i1p1_193412-195911.nc','tas_Amon_HadGEM2-ES_historical_r1i1p1_195912-198411.nc',
                            'tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc','tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc'],
                'IPSL-CM5A-LR_hist':['tas_Amon_IPSL-CM5A-LR_historical_r1i1p1_185001-200512.nc'],
                'IPSL-CM5A-MR_hist':['tas_Amon_IPSL-CM5A-MR_historical_r1i1p1_185001-200512.nc'],
                'CCSM4_hist':['tas_Amon_CCSM4_historical_r1i1p1_185001-200512.nc'],
                'NorESM1-M_hist':['tas_Amon_NorESM1-M_historical_r1i1p1_185001-200512.nc'],
                'CESM1-CAM5_hist':['tas_Amon_CESM1-CAM5_historical_r1i1p1_185001-200512.nc'],
                'NorESM1-ME_hist':['tas_Amon_NorESM1-ME_historical_r1i1p1_185001-200512.nc'],
                'GISS-E2-H_hist':['tas_Amon_GISS-E2-H_historical_r1i1p1_185001-190012.nc','tas_Amon_GISS-E2-H_historical_r1i1p1_190101-195012.nc',
                                  'tas_Amon_GISS-E2-H_historical_r1i1p1_195101-200512.nc'],
                'ERA5_hist':['T2M_monthly.nc'],
                'BEST_hist':['Land_and_Ocean_LatLong1_regrid_1x1_0to360.nc']
                }

        model_rcp_files={
                'MIROC5_26':['tas_Amon_MIROC5_rcp26_r1i1p1_200601-210012.nc'],
                'MIROC5_45':['tas_Amon_MIROC5_rcp45_r1i1p1_200601-210012.nc'],
                'MIROC5_60':['tas_Amon_MIROC5_rcp60_r1i1p1_200601-210012.nc'],
                'MIROC5_85':['tas_Amon_MIROC5_rcp85_r1i1p1_200601-210012.nc'],
                'HadGEM2-ES_85':['tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc','tas_Amon_HadGEM2-ES_rcp85_r1i1p1_203012-205511.nc',
                            'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_205512-208011.nc','tas_Amon_HadGEM2-ES_rcp85_r1i1p1_208012-209912.nc',
                            'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_209912-212411.nc'],
                'IPSL-CM5A-LR_26':['tas_Amon_IPSL-CM5A-LR_rcp26_r1i1p1_200601-230012.nc'],
                'IPSL-CM5A-LR_45':['tas_Amon_IPSL-CM5A-LR_rcp45_r1i1p1_200601-230012.nc'],
                'IPSL-CM5A-LR_60':['tas_Amon_IPSL-CM5A-LR_rcp60_r1i1p1_200601-210012.nc'],
                'IPSL-CM5A-LR_85':['tas_Amon_IPSL-CM5A-LR_rcp85_r1i1p1_200601-230012.nc'],
                'CCSM4_26':['tas_Amon_CCSM4_rcp26_r1i1p1_200601-210012.nc','tas_Amon_CCSM4_rcp26_r1i1p1_210101-230012.nc'], # CCSM4 TAS are nan after 2100 over Greenland
                'CCSM4_45':['tas_Amon_CCSM4_rcp45_r1i1p1_200601-210012.nc','tas_Amon_CCSM4_rcp45_r1i1p1_210101-229912.nc'],
                'CCSM4_60':['tas_Amon_CCSM4_rcp60_r1i1p1_200601-210012.nc','tas_Amon_CCSM4_rcp60_r1i1p1_210101-230012.nc'],
                'CCSM4_85':['tas_Amon_CCSM4_rcp85_r1i1p1_200601-210012.nc','tas_Amon_CCSM4_rcp85_r1i1p1_210101-230012.nc'],
                'IPSL-CM5A-MR_26':['tas_Amon_IPSL-CM5A-MR_rcp26_r1i1p1_200601-210012.nc'],
                'IPSL-CM5A-MR_45':['tas_Amon_IPSL-CM5A-MR_rcp45_r1i1p1_200601-210012.nc'],
                'IPSL-CM5A-MR_60':['tas_Amon_IPSL-CM5A-MR_rcp60_r1i1p1_200601-210012.nc'],
                'IPSL-CM5A-MR_85':['tas_Amon_IPSL-CM5A-MR_rcp85_r1i1p1_200601-210012.nc'],
                'NorESM1-M_26':['tas_Amon_NorESM1-M_rcp26_r1i1p1_200601-210012.nc'],
                'NorESM1-M_45':['tas_Amon_NorESM1-M_rcp45_r1i1p1_200601-210012.nc'],
                'NorESM1-M_85':['tas_Amon_NorESM1-M_rcp85_r1i1p1_200601-210012.nc'],
                'NorESM1-M_60':['tas_Amon_NorESM1-M_rcp60_r1i1p1_200601-210012.nc'],
                'CESM1-CAM5_26':['tas_Amon_CESM1-CAM5_rcp26_r1i1p1_200601-210012.nc'],
                'CESM1-CAM5_45':['tas_Amon_CESM1-CAM5_rcp45_r1i1p1_200601-210012.nc'],
                'CESM1-CAM5_60':['tas_Amon_CESM1-CAM5_rcp60_r1i1p1_200601-210012.nc'],
                'CESM1-CAM5_85':['tas_Amon_CESM1-CAM5_rcp85_r1i1p1_200601-210012.nc'],
                'NorESM1-ME_26':['tas_Amon_NorESM1-ME_rcp26_r1i1p1_200601-206012.nc','tas_Amon_NorESM1-ME_rcp26_r1i1p1_206101-210112.nc'],
                'NorESM1-ME_45':['tas_Amon_NorESM1-ME_rcp45_r1i1p1_200601-210212.nc'],
                'NorESM1-ME_60':['tas_Amon_NorESM1-ME_rcp60_r1i1p1_200601-205012.nc','tas_Amon_NorESM1-ME_rcp60_r1i1p1_205101-210112.nc'],
                'NorESM1-ME_85':['tas_Amon_NorESM1-ME_rcp85_r1i1p1_200601-204412.nc','tas_Amon_NorESM1-ME_rcp85_r1i1p1_204501-210012.nc'],
                'GISS-E2-H_26':['tas_Amon_GISS-E2-H_rcp26_r1i1p1_200601-205012.nc','tas_Amon_GISS-E2-H_rcp26_r1i1p1_205101-210012.nc'],
                'GISS-E2-H_45':['tas_Amon_GISS-E2-H_rcp45_r1i1p1_200601-205012.nc','tas_Amon_GISS-E2-H_rcp45_r1i1p1_205101-210012.nc'],
                'GISS-E2-H_60':['tas_Amon_GISS-E2-H_rcp60_r1i1p1_200601-205012.nc','tas_Amon_GISS-E2-H_rcp60_r1i1p1_205101-210012.nc'],
                'GISS-E2-H_85':['tas_Amon_GISS-E2-H_rcp85_r1i1p1_200601-205012.nc','tas_Amon_GISS-E2-H_rcp85_r1i1p1_205101-210012.nc'],
                'ERA5':['T2M_monthly.nc'],
                'BEST':['Land_and_Ocean_LatLong1_regrid_1x1_0to360.nc']}
    else: # TOS (which is not very useful)
        var='tos'
        model_hist_files={'MIROC_26':['tos_Omon_MIROC5_historical_r1i1p1_185001-201212.nc'],
                'MIROC_85':['tos_Omon_MIROC5_historical_r1i1p1_185001-201212.nc'],
                'NorESM_85':['tos_Omon_NorESM1-M_historical_r1i1p1_185001-200512.nc'],
                'HadGEM2_85':['tos_Omon_HadGEM2-ES_historical_r1i1p1_185912-195911.nc','tos_Omon_HadGEM2-ES_historical_r1i1p1_195912-200512.nc']}
        model_rcp_files={'MIROC_26':['tos_Omon_MIROC5_rcp26_r1i1p1_200601-210012.nc'],
                'MIROC_85':['tos_Omon_MIROC5_rcp85_r1i1p1_200601-210012.nc'],
                'NorESM_85':['tos_Omon_NorESM1-M_rcp85_r1i1p1_200601-210012.nc'],
                'HadGEM2_85':['tos_Omon_HadGEM2-ES_rcp85_r1i1p1_200512-209912.nc','tos_Omon_HadGEM2-ES_rcp85_r1i1p1_209912-219911.nc']}

    ###
    gmt_tss={model:{en:{} for en in ensembles} for model in models}
    gmt_tss_unsmooth={model:{en:{} for en in ensembles} for model in models}
    gmt_tss_filter={model:{en:{} for en in ensembles} for model in models}
    gmt_tss_rolling={model:{en:{} for en in ensembles} for model in models}
    gmt_ts_raw={model:{en:{} for en in ensembles} for model in models}
    datas_anoms={model:{en:{} for en in ensembles} for model in models}
    for mm, model in enumerate(models):
        if model=='ERA5':
            path='/mnt/data/data_a/ERA5/T2M/'
            var='t2m'
        elif model=='BEST':
            path='/mnt/data/data_a/Berkeley_SAT_land/monthly_global/'
            var='temperature'
        else: # For other models
            path='/Users/home/siewpe/codes/greenland_emulator/save_gmt/map_data/%s/'%model.split('_')[0]
        for en in ensembles:
            for region in regions:
                print(var,model,en,region)
                if region=='global': # Global
                    lat1=-90; lat2=90; lon1=0; lon2=360 # Assume data is from -90 to +90
                elif region=='greenland':
                    #lat1=60; lat2=83; lon1=290; lon2=340 # Small area within Greenland
                    lat1=55; lat2=85; lon1=260; lon2=360 # Standard area
                else:
                    pass

                ### 1. Read historical (<=2004) data
                model_mod=model.split('_')[0]+'_hist'
                hist_files=model_hist_files[model_mod]
                ## replace ensemble no from r1i1p1 to the tartget ensemble number
                hist_files=[i.replace('r1i1p1','r%si1p1'%en) for i in hist_files]
                hist_datas=[]
                for file in hist_files:
                    hist_data= xr.open_dataset(path+file)[var]
                    hist_datas.append(hist_data)
                hist_datas=xr.concat(hist_datas,dim='time')
                ## Select 1950 Jan to 2005 Dec 
                hist_datas=hist_datas.sel(time=slice('%s-01-01'%ini_st_year,'2005-12-30'))

                ### 2. Read future (>=2005) data
                rcp_files=model_rcp_files[model]
                rcp_files=[i.replace('r1i1p1','r%si1p1'%en) for i in rcp_files]
                rcp_datas=[]
                for file in rcp_files:
                    rcp_data= xr.open_dataset(path+file)[var]
                    rcp_datas.append(rcp_data)
                rcp_datas=xr.concat(rcp_datas,dim='time')
                ## Select 2006 Jan to 2100 Dec 
                rcp_datas=rcp_datas.sel(time=slice('2006-01-01','2100-12-30')) 
                #rcp_datas=rcp_datas.sel(time=slice('2006-01-01','2300-12-30')) # Allow selection up to year 2300
                dumplicate_index=rcp_datas.time.to_index().duplicated()
                rcp_datas=rcp_datas.isel(time=~dumplicate_index)
                ## Combine hist and RCP data
                datas=xr.concat([hist_datas,rcp_datas],dim='time')
                #print(model,hist_datas.shape,rcp_datas.shape) # To make 

                ### 2.5
                ### Reverse ERA5 lat (90 to -90) before doing the average
                if model=='ERA5': ## BEST already has the name lat and lon
                    # reverse the latitude if it is ERA5
                    datas=datas.isel(latitude=slice(None, None, -1)) 
                    datas=datas.rename({'latitude':'lat', 'longitude':'lon'})

                if 'HadGEM2-ES' in model: # HadGEM2 starts in 1959-Dec 
                    datas=datas.sel(time=slice('1860-01-01','2100-12-30')) 

                ### 3. Calculate annual-mean from monthly data
                datas=datas.coarsen(time=12).mean()
                years=datas.time.dt.year
                datas=datas.assign_coords({'time':years})

                ### 4. Relative to 1960-1989 as 2d-map climatology
                ##! maybe calcuclated weight-area average first and then removing climatology? - but this can build an anomaly map
                datas_clim=datas.sel(time=slice('%s'%clim_st_yr,'%s'%clim_end_yr)).mean(dim='time')
                if region=='global':
                    if datas.sel(time=slice('%s'%clim_st_yr,'%s'%clim_end_yr)).shape[0]!=51:
                        print(model, ', This model does not have full 1850-1900 data')
                datas_anom=datas-datas_clim
                datas_anoms[model][en][region]=datas_anom
                data=datas_anom
                #print(var,model,(data==0).sum().item())

                ### 5. Calculate the area average (both GMTs and Greenland)
                if var in ['tas','t2m', 'temperature']: # t2m for ERA5; temperature for BEST
                    #print('number of nan data: ', np.isnan(data).sum().item())
                    lons=data.lon.values; lats=data.lat.values
                    gmt_ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
                    gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':years})
                elif (var=='tos') & ("MIROC" in model): # MIROC has no land mask. Need to fill 0 with nan
                    mask1=data!=0
                    mask2=(data.lat>lat1) & (data.lat<lat2) & (data.lon>lon1) & (data.lon<lon2)
                    mask=mask1 & mask2 # True are the values needed
                    data_temp=np.where(mask,data.values,np.nan)
                    data=data.copy(data=data_temp, deep=True) # Put back into numpy
                    lat_weight=np.cos(data.lat*np.pi/180)
                    lat_weight_mask=xr.where(mask,lat_weight,np.nan)
                    lat_weight_mask_sum=lat_weight_mask.sum(dim='rlat').sum(dim='rlon') # the sum of the weight for needed value
                    gmt_ts=(data*lat_weight_mask).sum(dim='rlat').sum(dim='rlon')/lat_weight_mask_sum
                elif (var=='tos') & (model=='HadGEM2_85'): # HadGEM2 data has land mask (np.nan) and in lat-lon grid
                    lons=data.lon.values; lats=data.lat.values
                    gmt_ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
                    gmt_ts=xr.DataArray(gmt_ts,dims=['time'],coords={'time':years})
                elif (var=='tos') & ("NorESM_85" in model): # NorESM has land mask too, in i, j grid
                    mask1=~np.isnan(data) # Select all non-nan values
                    mask2=(data.lat>lat1) & (data.lat<lat2) & (data.lon>lon1) & (data.lon<lon2)
                    mask=mask1 & mask2 # True are the values needed
                    data_temp=np.where(mask,data.values,np.nan)
                    data=data.copy(data=data_temp, deep=True) 
                    lat_weight=np.cos(data.lat*np.pi/180)
                    lat_weight_mask=xr.where(mask,lat_weight,np.nan)
                    lat_weight_mask_sum=lat_weight_mask.sum(dim='i').sum(dim='j') 
                    gmt_ts=(data*lat_weight_mask).sum(dim='i').sum(dim='j')/lat_weight_mask_sum
                else:
                    ipdb.set_trace()

                ### 6. Fill in 2100 data from 2101 to 2350 ## Pad more year so that running-mean can be calculated (only for models, but for ERA5)
                if model in ['BEST','ERA5']:
                    pass
                    gmt_ts=gmt_ts
                else:
                    time_pad_end=2350
                    time_pad=np.arange(time_pad_st,time_pad_end+1) # model has time_pad_st=2101; obs has time_pad_st=2024
                    ending_value=gmt_ts.isel(time=-1).values.item() # 2100 data
                    pad_data=xr.DataArray([ending_value]*len(time_pad),dims=['time'],coords={'time':time_pad})
                    gmt_ts=xr.concat([gmt_ts,pad_data], dim='time')
                    gmt_ts_raw[model][en][region]=gmt_ts # use to store the raw gmt for aggregating MIROC_26

                ### 7. Replace everything before 2016 as MIROC26 (only for SOME models, not for ERA5)
                if (model!='MIROC_26') & before2016_miroc26en1: # Set up False to manually pass this
                    gmt_ts_early=gmt_ts_raw['MIROC5_26'][1][region].sel(time=slice(ini_st_year,2015))
                    #print(model,gmt_ts_early)
                    gmt_ts_later=gmt_ts.sel(time=slice(2016,time_pad_end))
                    gmt_ts=xr.concat([gmt_ts_early,gmt_ts_later],dim='time')
                    print("Use MIROC6_en1 to replace any data before 2016")

                ### 7. Do the running average or filtering
                # Save one for the unsmooth version
                gmt_ts_filter=scipy.signal.savgol_filter(gmt_ts,filter_window,3)
                gmt_ts_filter=gmt_ts.copy(data=gmt_ts_filter)
                gmt_ts_rolling=gmt_ts.rolling(time=rolling_year,center=True).mean() # this number is the rolling period

                ### 8. Finally select the year
                ### Save all versions for comparitions
                gmt_tss_unsmooth[model][en][region]=gmt_ts.sel(time=slice(final_st_yr,final_end_yr))
                gmt_tss_filter[model][en][region]=gmt_ts_filter.sel(time=slice(final_st_yr,final_end_yr))
                gmt_tss_rolling[model][en][region]=gmt_ts_rolling.sel(time=slice(final_st_yr,final_end_yr))

                ### 9. Choose the final version as gmt_tss
                #gmt_tss[model][en][region]=gmt_tss_rolling[model][en][region]
                gmt_tss[model][en][region]=gmt_tss_filter[model][en][region]


    if True: ### Save the files
        if var in ['t2m','temperature']:
            var='tas'
        for model in models:
            for en in ensembles:
                for region in regions:
                    years=gmt_tss[model][en][region].time # idx0
                    gmt_data=gmt_tss[model][en][region] #idx1
                    gmt_cum_data=gmt_tss[model][en][region].cumulative("time").sum() #idx2 (the cumulative data starts at 1990)
                    X8=np.array([1 if year<=2100 else year-2100 for year in years]) #idx3
                    gmt_cum_data_gradient=gmt_cum_data.differentiate("time") #idx4
                    save_data=np.column_stack((years.values,gmt_data.values,gmt_cum_data.values,X8,gmt_cum_data_gradient.values))
                    np.save('./save_gmt/tas_ts_create/gmt_%s_%s_en%s_%s.npy'%(var,model,en,region),save_data)



    ### Start plotting the timeseies
    models_colors={'ctrlProj':"k",'MIROC_85':'red','MIROC_26':'green','HadGEM2_85':'orange','NorESM_85':'brown','MIROC_2685mean':'blue','MIROC_2685_cooling':'violet',
            'MIROC_85_2km':'salmon','MIROC_45':'violet','MIROC_60':'violet',
            'IPSLCM5ALR_26':'gainsboro','IPSLCM5ALR_45':'silver','IPSLCM5ALR_60':'gray','IPSLCM5ALR_85':'black',
            'CCSM4_26':'gainsboro','CCSM4_45':'silver','CCSM4_60':'gray','CCSM4_85':'black','ERA5':'k','BEST':'k'}
    models_colors={'85':'red','60':'tomato','45':'orange','26':'green','ERA5':'k','BEST':'k'}
    plt.close(); fig,(ax1,ax2)=plt.subplots(2,1,figsize=(4,4))
    for model in models:
        en=ensembles[0] # Only plotting the first ensemble
        #en=ensembles[1] # Only plotting the first ensemble
        years=gmt_tss[model][en]['global'].time.values
        if model in ['ERA5','BEST']:
            model_mod=model.split('_')[0] 
            ax1.set_xlim(1990,2024)
            ax2.set_xlim(1990,2024)
        else:
            model_mod=model.split('_')[1] 
        ax1.plot(years, gmt_tss[model][en]['global'], color=models_colors[model_mod], lw=1, label=model)
        ax1.plot(years, gmt_tss[model][en]['global'], color='k', lw=1)
        ax1.plot(years, gmt_tss_unsmooth[model][en]['global'], color=models_colors[model_mod], lw=1)
        #ax1.plot(years, gmt_tss_filter[model][en]['global'], color=models_colors[model_mod], lw=1, label=model, ls='--')
        ax2.plot(years, gmt_tss[model][en]['greenland'], color='k', lw=1)
        ax2.plot(years, gmt_tss_unsmooth[model][en]['greenland'], color=models_colors[model_mod], lw=1)
        #ax2.plot(years, gmt_tss_filter[model][en]['greenland'], color=models_colors[model_mod], lw=1, ls='--')
    ax1.legend(bbox_to_anchor=(0.02,0.5), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.2, labelspacing=0.1, fontsize=8)
    ax1.set_title('GMT global %s'%var)
    ax2.set_title('GMT Greenland %s'%var)
    axs=(ax1,ax2)
    for ax in axs:
        ax.axhline(y=0,color='lightgray',linestyle='--')
        ax.axvline(x=2014,color='k',linestyle='--',lw=1)
        ax.axvline(x=2015,color='k',linestyle='--',lw=1)
    #ax1.set_ylim(-1,7)
    #ax1.set_ylim(-1,10)
    #ax1.annotate(r"$\rho$=%s (All)"%(str(corr)),xy=(0.01,0.95), xycoords='axes fraction', fontsize=10)
    fig_name = 'gmt_timeseries_greenland_global'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0.4) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

    if True: # Start plotting the maps (it doesn't work for ERA5)
        row=1; col = len(models)
        row=len(models); col = 1
        grid = row*col
        en=ensembles[0] # Only plotting the first ensemble
        plotting_grids=[datas_anoms[model][en]['greenland'].sel(time=2050) for model in models]
        plotting_grids=[pt.rename({'lat':'latitude', 'lon':'longitude'}) for pt in plotting_grids]
        #plotting_grids_latlon=[pt.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2)) for pt in plotting_grids]
        #[np.isnan(pt).sum().item() for pt in plotting_grids_latlon]
        shading_grids = plotting_grids
        #mapcolor_grid = ['jet'] * grid
        mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        mapcolor_grid = [cmap] * grid
        #mapcolor_grid = ['jet'] * grid
        shading_level = np.linspace(-2,2,11)
        shading_level = np.linspace(-5,5,11)
        shading_level = np.linspace(-12,12,11)
        shading_level_grid = [shading_level] * grid
        contour_grids = None; contour_clevels = None
        clabels_row = [''] * grid
        top_title = [''] * col
        left_title = [''] * row
        leftcorner_text = None
        leftcorner_text=models
        #projection=ccrs.PlateCarree(central_longitude=0); xsize=4; ysize=1
        projection=ccrs.Mercator(central_longitude=0); xsize=8; ysize=2
        xlim = [-110,10]
        ylim = (55,85)
        ylim = (50,90)
        pval_map = None
        pval_hatches = False
        fill_continent=True
        fill_continent=False
        region_boxes = None
        lat1=55; lat2=85; lon1=260; lon2=360 # Standard area of Greenland
        region_boxes = [tools.create_region_box(lat1, lat2, lon1, lon2)]*len(models)
        lat1=60; lat2=83; lon1=290; lon2=340 # Small area within Greenland
        region_boxes_extra = [tools.create_region_box(lat1, lat2, lon1, lon2)]*len(models)
        tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row,
                    top_titles=top_title, 
                    left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=True,
                    region_boxes=region_boxes, region_boxes_extra=region_boxes_extra, leftcorner_text=leftcorner_text,
                    ylim=ylim, xlim=xlim, quiver_grids=None,
                    pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=True,
                    contour_map_grids=contour_grids, contour_clevels=contour_clevels)
        fig_name = 'ESFG_download_TAS_maps_models'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

