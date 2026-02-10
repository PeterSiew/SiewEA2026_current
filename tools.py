import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy; import cartopy.crs as ccrs
import cartopy.feature as cf
import datetime as dt
import ipdb
from scipy import stats
import matplotlib

import tools


def read_data(var, months=None, limit_year=False, limit_lat=True, lon_lat_name_change=True, slicing=False, compute=False, open_single_file=True, reverse_lat=True, varname=None, decode_times=True):

    ### Some data has new time for adjustment
    new_time = None
    ### MERRA2 in czheng folder
    atmo_vars1 = [v+h for v in ['H', 'Q', 'T', 'U', 'V'] for h in ['850', '500', '250']]
    atmo_vars2 = ['QV2M', 'TQV', 'PS', 'U2M', 'V2M']
    atmo_vars3 = ['U10M', 'V10M']
    if var in atmo_vars1+atmo_vars2+atmo_vars3:
        path= '/dx03/czheng/MERRA2/dx14/%s_19800101-20230601.nc'%var
    elif var in ['SLP', 'T2M', 'FRSNO']: #FRSNO is the snow cover change from MERR2
        path = '/dx03/czheng/MERRA2/dx14/%s_19800101-20230601.nc'%var
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='D') 
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='D') 
    # MERRA2 in pyfsiew folder
    elif var in ['UFLXPHI', 'VFLFXPHI', 'THF', 'OMEGA500', 'DQVDT_DYN']: # Wm-2, Wm-1 and Wm-2, kg m-2 s-1 (convergence is +ve)
        path = '/dx13/pyfsiew/MERRA2/%s_19800101-20211231.nc'%var
    elif var in ['H500_regrid_1x1', 'U850_regrid_1x1', 'V850_regrid_1x1', 'T850_regrid_1x1', 'H850_regrid_1x1',
            'PS_regrid_1x1', 'THF_regrid_1x1', 'LWGAB_regrid_1x1',
            'surface_net_flux_regrid_1x1']:
        path = '/dx13/pyfsiew/MERRA2/%s.nc'%var
        new_time = pd.date_range(start='1980-01-01', end='2021-12-31', freq='D') 
    elif var in ['SLP_regrid_1x1']: # This is the daily data
        path = '/dx13/pyfsiew/MERRA2/%s.nc'%var
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='D') 
    elif var in ['SLP_regrid_1x1_monthly', 'SLP_regrid_1x1_monthly_0to360']: # The monthly data
        path = '/dx13/pyfsiew/MERRA2/%s.nc'%var
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='MS') 
    elif var in ['T2M_regrid_1x1_monthly']:
        path = '/mnt/data/data_a/MERRA2/%s.nc'%var
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='MS') 
    elif var in ['U10M_regrid_1x1_monthly', 'V10M_regrid_1x1_monthly']:
        path = '/dx13/pyfsiew/MERRA2/%s.nc'%var
        new_time = pd.date_range(start='1987-01-01', end='2019-12-01', freq='MS') 
    # For temporary (for the forced versus unforced response paper)
    # MERRA2
    elif var =='obs1_psl_en':
        path = '/dx13/pyfsiew/MERRA2/SLP_regrid_1x1_monthly.nc'
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='MS') 
    elif var =='obs1_tas_en':
        path = '/dx13/pyfsiew/MERRA2/T2M_regrid_1x1_monthly.nc'
        new_time = pd.date_range(start='1980-01-01', end='2023-06-01', freq='MS') 
    # ERA5
    elif var =='obs2_psl_en': 
        path = '/mnt/data/data_a/ERA5/MSLP/MSLP_monthly.nc'
        new_time = pd.date_range(start='1940-01-01', end='2023-12-01', freq='MS') 
        open_single_file=False
    elif var =='obs2_tas_en': 
        path = '/dx15/pyfsiew/ERA5/T2M/T2M_monthly.nc'
        new_time = pd.date_range(start='1940-01-01', end='2023-12-01', freq='MS') 
    # JRA55
    elif var =='obs3_psl_en': 
        path = '/dx15/pyfsiew/JRA55/prmsl_1979-2023_monthly.nc'
        new_time = pd.date_range(start='1979-01-01', end='2023-12-01', freq='MS') 
    elif var =='obs3_tas_en': 
        path = '/dx15/pyfsiew/JRA55/tmp_1979-2023_monthly.nc'
        new_time = pd.date_range(start='1979-01-01', end='2023-12-01', freq='MS') 
    # NCEP R1
    elif var == 'obs4_psl_en':
        path = '/dx13/pyfsiew/NCEP_NCAR_reanalysis1/slp.mon.mean.nc' 
        new_time = pd.date_range(start='1948-01-01', end='2023-06-01', freq='MS') 
    elif var == 'obs4_tas_en':
        path = '/dx13/pyfsiew/NCEP_NCAR_reanalysis1/slp.mon.mean.nc' 
        new_time = pd.date_range(start='1948-01-01', end='2023-06-01', freq='MS') 
    # NSIDC sea ice (different algorithem)
    elif var =='obs1_hi_en':
        path = '/dx15/pyfsiew/PIOMAS/piomas_himonth_1979to2020.nc'
        new_time = pd.date_range(start='1979-01-01', end='2020-12-01', freq='MS') 
    elif var =='obs1_sic_en':
        path = '/mnt/data/data_b/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_monthly_nh_197811to202309_regrid_1x1_0to360.nc'# NSIDC CDR algorithm
        new_time = pd.date_range(start='1978-11-01', end='2023-09-01', freq='MS') 
    elif var =='obs2_sic_en': 
        #path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/nsidcbt_seaice_conc_monthly_nh_197811to202309_regrid_1x1.nc' # NSIDC-BT algorithm
        #new_time = pd.date_range(start='1978-11-01', end='2023-09-01', freq='MS')  
        path = '/mnt/data/data_b/PIOMAS_20C/piomas20c.area.1901.2010.v1.0_regrid.nc'# just for temporary
        new_time = pd.date_range(start='1901-01-01', end='2010-12-01', freq='MS') 
    elif var =='obs3_sic_en':
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/nsidcnt_seaice_conc_monthly_nh_197811to202309_regrid_1x1.nc' # NSIDC-CNT alogrithem
        new_time = pd.date_range(start='1978-11-01', end='2023-09-01', freq='MS') 
    elif var =='obs4_sic_en':
        path = '/mnt/data/data_b/PIOMAS_20C/piomas20c_nisidc_1901jan_to_2022dec.nc'
        new_time = pd.date_range(start='1901-01-01', end='2022-12-01', freq='MS') 
    # MERRA2 V850 daily
    elif var =='obs1_V850daily_en':
        path = '/dx13/pyfsiew/MERRA2/V850_regrid_1x1.nc'
        new_time = pd.date_range(start='1980-01-01', end='2021-12-31', freq='D') 
    elif var =='obs1_T850daily_en':
        path = '/dx13/pyfsiew/MERRA2/T850_regrid_1x1.nc'
        new_time = pd.date_range(start='1980-01-01', end='2021-12-31', freq='D') 
    # ERA5 file
    elif var == 'ERA5_T2M_daily_regrid_1x1': # daily
        path = '/mnt/data/data_a/ERA5/T2M_daily/T2M_daily-1940Jan_2024Mar_1x1.nc'
    elif var == 'ERA5_MSLP_monthly': # Monthly
        path = '/mnt/data/data_a/ERA5/MSLP/MSLP_monthly.nc'
        open_single_file = False
    elif var == 'ERA5_T2M_monthly': # Monthly
        path = '/mnt/data/data_a/ERA5/T2M/T2M_monthly.nc'
        open_single_file = False
    # NOAA/CIRES/DOE 20th Century Reanalysis (V3)
    elif var == 'noaa_20CR_slp': # 1836-01-01 to 2015-12-01
        path = '/dx13/pyfsiew/NOAA_20th_century_reanalysis_v3/prmsl.mon.mean.nc'
        open_single_file = False
    elif 'noaa_20CR_slp_en' in var: # man-made bootstrap samples
        en = var.replace('noaa_20CR_slp_en', '')
        path = '/dx13/pyfsiew/NOAA_20th_century_reanalysis_v3/bootstrap/noaa_20CR_slp_en%s.nc'%en
    elif var == 'noaa_20CR_slp_extend': # After 2015 Dec, extended by ERA5 data
        path = '/dx13/pyfsiew/NOAA_20th_century_reanalysis_v3/prmsl.mon.mean.extended.nc'
        open_single_file = False
    elif var == 'noaa_20CR_psl_daily': 
        path = '/dx13/pyfsiew/NOAA_20th_century_reanalysis_v3/psl_daily/prmsl.*.nc'
        new_time = pd.date_range(start='1806-01-01', end='2015-12-31', freq='D') 
        open_single_file = False
    # Best SAT (Berkerly)
    elif var == 'BESTSAT_daily_regrid_1x1':
        path = '/mnt/data/data_a/Berkeley_SAT_land/BESTSAT_daily-1880Jan_2022July_1x1.nc'
    # Sea surface temperature SST data monthly (noaa.oisst.v2.highres)
    elif var == 'noaa_sst_monthly': # Monthly
        path = '/dx13/pyfsiew/noaa_monthly_mean_sst/sst.mon.mean.nc'
    # Satellite sea ice cocnetration data
    elif var == 'cdr_seaice_conc': # Daily
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_19790101-20211231.nc'
    elif var == 'cdr_seaice_conc_monthly': # Monthly (directly downloaded from NSIDC - but regrid)
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_monthly_nh_197811to202309_regrid_1x1.nc'
    elif var == 'cdr_seaice_conc_monthly_raw': # Monthly (directly downloaded from NSIDC - raw)
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/monthly_raw_data/seaice_conc_monthly_nh_197811_202309_v04r00.nc'
    elif var == 'cdr_seaice_conc_monthly_0.25x0.25': # Monthly (directly downloaded from NSIDC - raw)
        path = '/mnt/data/data_b/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_monthly_nh_197811to202309_regrid_0.25x0.25.nc'
    # HadISST sea ice
    elif var == 'HadISST_SIC':
        path = '/dx13/pyfsiew/HadiSST_SIC/HadISST_ice.nc'
        new_time = pd.date_range(start='1870-01-01', end='2023-01-01', freq='MS') 
    # Satellite ice move data from Pahtfinder v4
    elif var == 'icemove_u': # Unit is in cm/s
        path = '/dx13/pyfsiew/seaice_drift_pathfinder/icemove_u_19790101-20201231.nc'
    elif var == 'icemove_v':
        path = '/dx13/pyfsiew/seaice_drift_pathfinder/icemove_v_19790101-20201231.nc'
    # PIOMAS 20C reanalysis (1900 to 2100)
    elif var =='piomas_20CR':
        path = '/dx15/pyfsiew/PIOMAS_20C/piomas20c.area.1901.2010.v1.0_regrid.nc'
        new_time = pd.date_range(start='1901-01-01', end='2010-12-01', freq='MS') 
    # CIEC5 ice model data by Robin Claney in daily resolution
    elif var == 'cice5_sic': # (0-1) Multiple to 100 to %
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/aice_d_1979to2018.nc'
    elif var == 'cice5_sic_monthly': # (0-1) Multiple to 100 to %
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/aice_d_1979to2018_monthly.nc'
    elif var == 'cice5_sic_dynam': # %/day
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/daidtd_d_1979to2018.nc'
    elif var == 'cice5_sic_thermo': # %/day
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/daidtt_d_1979to2018.nc'
    elif var == 'cice5_thickness': # unit is m 
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/hi_d_1979to2018.nc'
    elif var == 'cice5_thickness_dynam': # unit is cm/day
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/dvidtd_d_1979to2018.nc'
    elif var == 'cice5_thickness_thermo': # unit is cm/day
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/dvidtt_d_1979to2018.nc'
    elif var == 'cice5_icemove_u': # Unit is in m/s
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/uu_d_1979to2018.nc'
    elif var == 'cice5_icemove_v':# Unit is in m/s (unilike pathfinder)
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/vv_d_1979to2018.nc'
    elif var == 'cice5_thickness_thermo_bottom': # unit is cm/day
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/meltb_d_1979to2018.nc'
    elif var == 'cice5_thickness_thermo_top': 
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/meltt_d_1979to2018.nc'
    elif var == 'cice5_thickness_thermo_lateral': 
        path = '/dx13/pyfsiew/CICE5_sea_ice_model/netcdf/meltl_d_1979to2018.nc'
    ### Mdoels
    # CMIP5 LENS
    # NCAR-CESM1 (40 members)# Skip the first enesmble (which starts at 1850 rather than 1920)
    elif 'cesm1_sic_en' in var: 
        en=var.replace('cesm1_sic_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/CESM/sic/sic_OImon_CESM1-CAM5_historical_rcp85_r%si1p1_192001-210012_regrid.nc'%en
        new_time = pd.date_range(start='1920-01-01', end='2100-12-01', freq='MS') 
    elif 'cesm1_snc_en' in var: 
        en=var.replace('cesm1_snc_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/CESM/snc/snc_LImon_CESM1-CAM5_historical_rcp85_r%si1p1_192001-210012.nc'%en
        new_time = pd.date_range(start='1920-01-01', end='2100-12-01', freq='MS') 
    elif 'cesm1_psl_en' in var: 
        en=var.replace('cesm1_psl_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/CESM/psl/psl_Amon_CESM1-CAM5_historical_rcp85_r%si1p1_192001-210012.nc'%en
    elif 'cesm1_tas_en' in var:
        en=var.replace('cesm1_tas_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/CESM/tas/tas_Amon_CESM1-CAM5_historical_rcp85_r%si1p1_192001-210012.nc'%en
        ipdb.set_trace()
    elif 'cesm1_psl_daily_en' in var: 
        en=var.replace('cesm1_psl_daily_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/CESM/psl_daily/psl_day_CESM1-CAM5_historical_rcp85_r%si1p1_19200101-21001231.nc'%en
        new_time = pd.date_range(start='1920-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
    elif 'cesm1_tas_daily_en' in var: 
        en=var.replace('cesm1_tas_daily_en', '')
        path = '/mnt/data/data_a/CMIP5_LENS/CESM/tas_daily/tas_day_CESM1-CAM5_historical_rcp85_r%si1p1_19200101-21001231.nc'%en
        if en=='1':
            new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='D') 
        else:
            new_time = pd.date_range(start='1920-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
    # CanESM2 (50 members)
    elif 'canesm2_sic_en' in var: 
        en = var.replace('canesm2_sic_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/canesm2/sic/sic_OImon_CanESM2_historical_rcp85_r%si1p1_195001-210012.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS') 
    elif 'canesm2_psl_en' in var: 
        en = var.replace('canesm2_psl_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/canesm2/psl/psl_Amon_CanESM2_historical_rcp85_r%si1p1_195001-210012.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS') 
    elif 'canesm2_snc_en' in var: 
        en = var.replace('canesm2_snc_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/canesm2/snc/snc_LImon_CanESM2_historical_rcp85_r%si1p1_195001-210012.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS') 
    elif 'canesm2_tas_en' in var: 
        en = var.replace('canesm2_tas_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/canesm2/tas/tas_Amon_CanESM2_historical_rcp85_r%si1p1_195001-210012.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS') 
    elif 'canesm2_tas_daily_en' in var: 
        en = var.replace('canesm2_tas_daily_en','')
        path = '/mnt/data/data_a/CMIP5_LENS/CanESM2/tas_daily/tas_day_CanESM2_historical_rcp85_r%si1p1_19500101-21001231.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time=new_time[mask]
    elif 'canesm2_psl_daily_en' in var: 
        en = var.replace('canesm2_psl_daily_en','')
        path = '/dx15/pyfsiew/CMIP5_LENS/canesm2/psl_daily/psl_day_CanESM2_historical_rcp85_r%si1p1_19500101-21001231.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time=new_time[mask]
    elif 'canesm2_tas_daily_fake_en' in var:  #15 ensembles
        en = var.replace('canesm2_tas_daily_fake_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/canesm2/tas_daily_PI/tas_day_CanESM2_piControl_fakeen%s_19790101-20221231.nc'%en
    # GFDL-ESM2M (30 members)
    elif 'gfdlesm2m_tas_daily_en' in var: # 
        en=var.replace('gfdlesm2m_tas_daily_en', '')
        path='/mnt/data/data_a/CMIP5_LENS/GFDL-ESM2M/tas_daily/tas_day_GFDL-ESM2M_historical_rcp85_r%si1p1_19500101-21001231.nc'%en
        new_time = pd.date_range(start='1950-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
    # GFDL-CM3 (20 members)
    elif 'gfdlcm3_tas_daily_en' in var:
        en=var.replace('gfdlcm3_tas_daily_en', '')
        path = '/mnt/data/data_a/CMIP5_LENS/GFDL-CM3/tas_daily/tas_day_GFDL-CM3_historical_rcp85_r%si1p1_19200101-21001231.nc'%en
        new_time = pd.date_range(start='1920-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
    elif 'gfdlcm3_psl_daily_en' in var:
        en=var.replace('gfdlcm3_psl_daily_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/gfdl_cm3/psl_daily/psl_day_GFDL-CM3_historical_rcp85_r%si1p1_19200101-21001231.nc'%en
        new_time = pd.date_range(start='1920-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
   # CSIRO-MK360
    elif 'mk360_tas_daily_en' in var:
        en = var.replace('mk360_tas_daily_en','')
        path='/mnt/data/data_a/CMIP5_LENS/CSIRO-Mk3-6-0/tas_daily/tas_day_CSIRO-Mk3-6-0_historical_rcp85_r%si1p1_185001-210012.nc'%en
        new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time=new_time[mask]
    elif 'mk360_psl_daily_en' in var:
        en = var.replace('mk360_psl_daily_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/CSIRO-Mk3-6-0/psl_daily/psl_day_CSIRO-Mk3-6-0_historical_rcp85_r%si1p1_185001-210012.nc'%en
        new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='D') 
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time=new_time[mask]
    elif 'mk360_snc_en' in var:
        en = var.replace('mk360_snc_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/CSIRO-Mk3-6-0/snc/snc_LImon_CSIRO-Mk3-6-0_historical_rcp85_r%si1p1_185001-210012.nc'%en
        new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='MS') 
    # EC-Earth-LENS (16 members) It has 29 Feb
    elif 'ecearth_tas_daily_en' in var:
        en = var.replace('ecearth_tas_daily_en','')
        path='/mnt/data/data_a/CMIP5_LENS/EC-EARTH/tas_daily/tas_day_EC-EARTH_historical_rcp85_r%si1p1_1860101-21001231.nc'%en
        new_time = pd.date_range(start='1860-01-01', end='2100-12-31', freq='D') 
    elif 'ecearth_psl_daily_en' in var:
        en = var.replace('ecearth_psl_daily_en','')
        path='/dx15/pyfsiew/CMIP5_LENS/ec-earth/psl_daily/psl_day_EC-EARTH_historical_rcp85_r%si1p1_1860101-21001231.nc'%en
        new_time = pd.date_range(start='1860-01-01', end='2100-12-31', freq='D') 
    # MPI-ESM-LR (100 members)
    elif 'MPI-ESM-LR_psl_en' in var: 
        en=var.replace('MPI-ESM-LR_psl_en', '')
        path = '/dx15/pyfsiew/CMIP5_LENS/MPI-ESM-LR/psl/psl_Amon_MPI-ESM_historical_rcp85_r%si1p1_185001-209912.nc'%en
        new_time = pd.date_range(start='1850-01-01', end='2099-12-01', freq='MS') 
    elif 'MPI-ESM-LR_sic_en' in var: # (0-100)
        en = var.replace('MPI-ESM-LR_sic_en', '').zfill(3)
        path = '/dx15/pyfsiew/CMIP5_LENS/MPI-ESM-LR/sic/sic_OImon_MPI-ESM_*_r%si*_regrid.nc'%en
        open_single_file = False
        new_time = pd.date_range(start='1850-01-01', end='2099-12-01', freq='MS') 
   # CMIP5 Pre-industrial control daily
    elif 'canesm2_tas_daily_PI' in var: 
        path = '/mnt/data/data_a/CMIP5_LENS/CanESM2/tas_daily_PI/tas_day_CanESM2_piControl_r1i1p1_*-*.nc'
        new_time = xr.cftime_range(start='2015-01-01', end='3110-12-31', freq='D')
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time=new_time[mask]
        open_single_file = False
    elif 'cesm1_tas_daily_PI' in var: # Not used (time timing has problem)
        path = '/mnt/data/data_a/CMIP5_LENS/CESM/tas_daily_PI/b.e11.B1850C5CN.f09_g16.005.cam.h1.TREFHT*.nc'
        #new_time = xr.cftime_range(start='0402-01-01', end='2200-12-31', freq='D')
        #mask = ~((new_time.month==2) & (new_time.day==29)); new_time=new_time[mask]
        new_time=var
        open_single_file = False
    ### CMIP6 LENS
    # NCAR-CESM2 (50 enesmbles)
    elif ('CESM2' in var) & ('en' in var):
        int_years = ['1001','1021','1041','1061','1081','1101','1121','1141','1161','1181']+['1231']*10+['1251']*10+['1281']*10+['1301']*10 + ['1011','1031','1051','1071','1091','1111','1131','1151','1171','1191']+['1231']*10+['1251']*10+['1281']*10+['1301']*10
        nums = ['001','002','003','004','005','006','007','008','009','010']*6+['011','012','013','014','015','016','017','018','019','020']*4
        ens = {str(i+1):int_years[i]+'.'+nums[i] for i in range(len(int_years))}
        ens['avg']='enavg'
        new_time = pd.date_range(start='1850-01-01', end='2100-12-01', freq='MS') 
        open_single_file = False
        # Atmospheric data
        if 'CESM2_psl_en' in var:
            en = var.replace('CESM2_psl_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/psl/b.e21.*.f09_g17.LE2-%s.cam.h0.PSL.*.nc'%(ens[en])
        elif 'CESM2_sst_en' in var:
            en = var.replace('CESM2_sst_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/sst/b.e21.*.f09_g17.LE2-%s.cam.h0.SST.*.nc'%(ens[en])
        elif 'CESM2_surlatent_en' in var:
            en = var.replace('CESM2_surlatent_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/LHFLX/b.e21.*.f09_g17.LE2-%s.cam.h0.LHFLX.*.nc'%(ens[en])
        elif 'CESM2_sursensible_en' in var:
            en = var.replace('CESM2_sursensible_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/SHFLX/b.e21.*.f09_g17.LE2-%s.cam.h0.SHFLX.*.nc'%(ens[en])
        elif 'CESM2_tas_en' in var: # THREFHT=T2M or TAS
            en = var.replace('CESM2_tas_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/TREFHT/b.e21.*.f09_g17.LE2-%s.cam.h0.TREFHT.*.nc'%(ens[en])
        ## Land data
        elif 'CESM2_rainland_en' in var:
            en = var.replace('CESM2_rainland_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/land/RAIN/b.e21.*.f09_g17.LE2-%s.clm2.h0.RAIN.*.nc'%(ens[en])
        elif 'CESM2_snowland_en' in var:
            en = var.replace('CESM2_snowland_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/land/SNOW/b.e21.*.f09_g17.LE2-%s.clm2.h0.SNOW.*.nc'%(ens[en])
        elif 'CESM2_qsoilland_en' in var:
            en = var.replace('CESM2_qsoilland_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/land/QSOIL/b.e21.*.f09_g17.LE2-%s.clm2.h0.QSOIL.*.nc'%(ens[en])
        elif 'CESM2_qrunoffland_en' in var:
            en = var.replace('CESM2_qrunoffland_en', '')
            ## This QRUNOFF has landunit
            #path = '/mnt/data/data_b/CMIP6_LENS/CESM2/land/QRUNOFF/b.e21.*.f09_g17.LE2-%s.clm2.h2.QRUNOFF.*.nc'%(ens[en]) 
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/land/QRUNOFF_coupler/b.e21.*.f09_g17.LE2-%s.clm2.h0.QRUNOFF_TO_COUPLER.*.nc'%(ens[en]) 
        ## Land ice data
        elif 'CESM2_SMB_en' in var:
            en = var.replace('CESM2_SMB_en', '')
            path = '/mnt/data/data_b/CMIP6_LENS/CESM2/land_ice/smb/b.e21.*.f09_g17.LE2-%s.cism.h.smb.*.nc'%(ens[en])
            new_time = pd.date_range(start='1850-01-01', end='2100-12-01', freq='YS')  # This one is yearly mean
        ## Sea ice data
        elif 'CESM2_sic_en' in var: # (0 to 1)
            en = var.replace('CESM2_sic_en', '')
            path = '/mnt/data/data_b/CMIP6/CESM2/sic/b.e21.*.f09_g17.LE2-%s.cice.h.aice.*_regrid.nc'%(ens[en])
        elif 'CESM2_sicraw_en' in var: # (0 to 1)
            en = var.replace('CESM2_sicraw_en', '')
            #path = '/dx13/pyfsiew/LENS/CESM2/sic_raw/b.e21.*.f09_g17.LE2-%s.cice.h.aice.*.nc'%(ens[en])
            path = '/mnt/data/data_b/CMIP6/CESM2/sic_raw/b.e21.*.f09_g17.LE2-%s.cice.h.aice.*.nc'%(ens[en])
        elif 'CESM2_hi_en' in var: # (0 to 1)
            en = var.replace('CESM2_hi_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/hi/b.e21.*.f09_g17.LE2-%s.cice.h.hi.*_regrid.nc'%(ens[en])
        elif 'CESM2_hiraw_en' in var: 
            en = var.replace('CESM2_hiraw_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/hi/b.e21.*.f09_g17.LE2-%s.cice.h.hi.*.nc'%(ens[en])
        elif 'CESM2_hs_en' in var: #  snow thickness on ice
            en = var.replace('CESM2_hs_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/hs/b.e21.*.f09_g17.LE2-%s.cice.h.hs.*.nc'%(ens[en])
        elif 'CESM2_sicdynam_en' in var: # Still in ocean grid
            en = var.replace('CESM2_sicdynam_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/daidtd/b.e21.*.f09_g17.LE2-%s.cice.h.daidtd.*.nc'%(ens[en])
        elif 'CESM2_sicthermo_en' in var: 
            en = var.replace('CESM2_sicthermo_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/daidtt/b.e21.*.f09_g17.LE2-%s.cice.h.daidtt.*.nc'%(ens[en])
        elif 'CESM2_iceoceanflux_en' in var: 
            en = var.replace('CESM2_iceoceanflux_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/fhocn/b.e21.*.f09_g17.LE2-%s.cice.h.fhocn.*.nc'%(ens[en])
        elif 'CESM2_sitempbot_en' in var: 
            en = var.replace('CESM2_sitempbot_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/sitempbot/b.e21.*.f09_g17.LE2-%s.cice.h.sitempbot.*.nc'%(ens[en])
        elif 'CESM2_fswabs_en' in var:  # Shortwave over sea ice?
            en = var.replace('CESM2_fswabs_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/fswabs/b.e21.*.f09_g17.LE2-%s.cice.h.fswabs.*.nc'%(ens[en])
        elif 'CESM2_iceu_en' in var: 
            en = var.replace('CESM2_iceu_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/ice_u/b.e21.*.f09_g17.LE2-%s.cice.h.uvel.*.nc'%(ens[en])
        elif 'CESM2_icev_en' in var: 
            en = var.replace('CESM2_icev_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/ice_v/b.e21.*.f09_g17.LE2-%s.cice.h.vvel.*.nc'%(ens[en])
        elif 'CESM2_isenbot_en' in var:
            en = var.replace('CESM2_isenbot_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/siflsensupbot/b.e21.*.f09_g17.LE2-%s.cice.h.siflsensupbot.*.nc'%(ens[en])
        elif 'CESM2_isentop_en' in var:
            en = var.replace('CESM2_isentop_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/siflsenstop/b.e21.*.f09_g17.LE2-%s.cice.h.siflsenstop.*.nc'%(ens[en])
        elif 'CESM2_iconductbot_en' in var:
            en = var.replace('CESM2_iconductbot_en', '')
            path = '/dx13/pyfsiew/LENS/CESM2/siflcondbot/b.e21.*.f09_g17.LE2-%s.cice.h.siflcondbot.*.nc'%(ens[en])
        ## Ocean data
        elif 'CESM2_otemp_en' in var: # Still in ocean grid 
            en = var.replace('CESM2_otemp_en', '')
            path = '/mnt/data/data_b/CMIP6/CESM2/OTEMP/b.e21.*.f09_g17.LE2-%s.pop.h.TEMP.*.nc'%(ens[en])
        elif 'CESM2_ou_en' in var: #  (ocean current on the 30cm)
            en = var.replace('CESM2_ou_en', '')
            path = '/dx02/pyfsiew/CESM2_LENS_ocean/UVEL/b.e21.*.f09_g17.LE2-%s.pop.h.UVEL.*.nc'%(ens[en])
        elif 'CESM2_ov_en' in var: # (ocean current on the 30cm)
            en = var.replace('CESM2_ov_en', '')
            path = '/dx02/pyfsiew/CESM2_LENS_ocean/VVEL/b.e21.*.f09_g17.LE2-%s.pop.h.VVEL.*.nc'%(ens[en])
        elif 'CESM2_ow_en' in var: # (ocean current on the 30cm)
            en = var.replace('CESM2_ow_en', '')
            path = '/dx02/pyfsiew/CESM2_LENS_ocean/WVEL/b.e21.*.f09_g17.LE2-%s.pop.h.WVEL.*.nc'%(ens[en])
        elif 'CESM2_out_en' in var: #  ocean u heat transport
            en = var.replace('CESM2_out_en', '')
            path = '/dx02/pyfsiew/CESM2_LENS_ocean/UET/b.e21.*.f09_g17.LE2-%s.pop.h.UET.*.nc'%(ens[en])
        elif 'CESM2_ovt_en' in var: #  ocean u heat transport
            en = var.replace('CESM2_ovt_en', '')
            path = '/dx02/pyfsiew/CESM2_LENS_ocean/VNT/b.e21.*.f09_g17.LE2-%s.pop.h.VNT.*.nc'%(ens[en])
        elif 'daily' in var:
            #new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='D') 
            #mask = ~((new_time.month==2)&(new_time.day==29)); new_time=new_time[mask]
            new_time = None
            if 'CESM2_V850daily_en' in var:
                en = var.replace('CESM2_V850daily_en', '')
                path = '/dx01/pyfsiew/CMIP6_LENS_daily/CESM2/v850_daily/b.e21.*.f09_g17.LE2-%s.cam.h1.V850.*.nc'%(ens[en])
            elif 'CESM2_T850daily_en' in var:
                en = var.replace('CESM2_T850daily_en', '')
                path = '/dx01/pyfsiew/CMIP6_LENS_daily/CESM2/t850_daily/b.e21.*.f09_g17.LE2-%s.cam.h1.T850.*.nc'%(ens[en])
   # MIROC6 (50 members)
    elif ('MIROC6' in var) & ('en' in var):
        new_time = pd.date_range(start='1850-01-01', end='2100-12-01', freq='MS') 
        open_single_file = False
        if 'MIROC6_psl_en' in var:
            path = '/dx13/pyfsiew/LENS/MIROC6/psl/psl_Amon_MIROC6_*_r%si1p1f1_gn_*.nc'%(var.replace('MIROC6_psl_en', ''))
        elif 'MIROC6_sic_en' in var: # (0-100)
            path = '/dx13/pyfsiew/LENS/MIROC6/sic/siconc_SImon_MIROC6_*_r%si1p1f1_gn_*_regrid.nc'%(var.replace('MIROC6_sic_en', ''))
        elif 'MIROC6_sicraw_en' in var: # (0-100)
            path = '/mnt/data/data_b/MIROC6/sic_raw/siconc_SImon_MIROC6_*_r%si1p1f1_gn_*.nc'%(var.replace('MIROC6_sicraw_en', ''))
        elif 'MIROC6_sst_en' in var: 
            path = '/dx13/pyfsiew/LENS/MIROC6/sst/tos_Omon_MIROC6_*_r%si1p1f1_gn_*_regrid.nc'%(var.replace('MIROC6_sst_en', ''))
        elif 'MIROC6_tas_en' in var: 
            path = '/dx13/pyfsiew/LENS/MIROC6/tas/tas_Amon_MIROC6_*_r%si1p1f1_gn_*.nc'%(var.replace('MIROC6_tas_en', ''))
        elif 'MIROC6_vas_en' in var: 
            path = '/dx13/pyfsiew/LENS/MIROC6/vas/vas_Amon_MIROC6_*_r%si1p1f1_gn_*.nc'%(var.replace('MIROC6_vas_en', ''))
        if 'daily' in var:
            new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='D') 
            #mask = ~((new_time.month==2)&(new_time.day==29));  new_time=new_time[mask]
            if 'MIROC6_V850daily_en' in var:
                path = '/dx13/pyfsiew/LENS/MIROC6/v850_daily/va_day_MIROC6_*_r%si1p1f1_gn_*.nc'%(var.replace('MIROC6_V850daily_en', ''))
            elif 'MIROC6_T850daily_en' in var:
                path = '/dx13/pyfsiew/LENS/MIROC6/t850_daily/ta_day_MIROC6_*_r%si1p1f1_gn_*.nc'%(var.replace('MIROC6_T850daily_en', ''))
    # CanESM5 (50 members)
    elif ('CanESM5' in var) & ('en' in var) & ('CanESM5-1' not in var) & ('CanESM5_hist' not in var):
        rs=list(range(1,26))+list(range(1,26)); ps=[1]*25+[2]*25
        ens = {str(i+1):'r%si1p%sf1'%(rs[i],ps[i]) for i in range(len(rs))} # the first 25 is p1 and the last 25 is p2
        ens['avg'] = 'ravgi1p1f1'
        new_time = pd.date_range(start='1850-01-01', end='2100-12-01', freq='MS') 
        open_single_file = False
        if 'CanESM5_psl_en' in var:
            en = var.replace('CanESM5_psl_en', '')
            path = '/dx13/pyfsiew/LENS/CanESM5/psl/psl_Amon_CanESM5_*_%s_gn_*.nc'%(ens[en])
        elif 'CanESM5_sic_en' in var:
            en = var.replace('CanESM5_sic_en', '')
            path = '/dx13/pyfsiew/LENS/CanESM5/sic/siconc_SImon_CanESM5_*_%s_gn_*_regrid.nc'%(ens[en])
        elif 'CanESM5_sicraw_en' in var:
            en = var.replace('CanESM5_sicraw_en', '')
            path = '/mnt/data/data_b/CanESM5/sic_raw/siconc_SImon_CanESM5_*_%s_gn_*.nc'%(ens[en])
        elif 'CanESM5_sst_en' in var:
            en = var.replace('CanESM5_sst_en', '')
            path = '/dx13/pyfsiew/LENS/CanESM5/sst/tos_Omon_CanESM5_*_%s_gn_*_regrid.nc'%(ens[en])
        elif 'CanESM5_tas_en' in var:
            en = var.replace('CanESM5_tas_en', '')
            path = '/dx13/pyfsiew/LENS/CanESM5/tas/tas_Amon_CanESM5_*_%s_gn_*.nc'%(ens[en])
        elif 'daily' in var:
            new_time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='D') 
            mask = ~((new_time.month==2)&(new_time.day==29));  new_time=new_time[mask]
            if 'CanESM5_V850daily_en' in var:
                en = var.replace('CanESM5_V850daily_en', '')
                path = '/dx13/pyfsiew/LENS/CanESM5/v850_daily/va_day_CanESM5_*_%s_gn_*.nc'%(ens[en])
            elif 'CanESM5_T850daily_en' in var:
                en = var.replace('CanESM5_T850daily_en', '')
                path = '/dx13/pyfsiew/LENS/CanESM5/t850_daily/ta_day_CanESM5_*_%s_gn_*.nc'%(ens[en])
    # ACCESS-ESM1-5 (40 ensembles)
    elif ('ACCESS-ESM1-5' in var) & ('en' in var):
        new_time = pd.date_range(start='1850-01-01', end='2100-12-01', freq='MS') 
        open_single_file = False
        if 'ACCESS-ESM1-5_psl_en' in var:
            path='/dx13/pyfsiew/LENS/ACCESS-ESM1-5/psl/psl_Amon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_psl_en',''))
        elif 'ACCESS-ESM1-5_tas_en' in var:
            path='/dx13/pyfsiew/LENS/ACCESS-ESM1-5/tas/tas_Amon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_tas_en',''))
        elif 'ACCESS-ESM1-5_vas_en' in var: 
            path='/dx13/pyfsiew/LENS/ACCESS-ESM1-5/vas/vas_Amon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_vas_en',''))
        elif 'ACCESS-ESM1-5_sic_en' in var: # (0-100)
            path = '/mnt/data/data_b/CMIP6/ACCESS-ESM1-5/sic/siconc_SImon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_sic_en',''))
        elif 'ACCESS-ESM1-5_sicraw_en' in var: # (0-100)
            path = '/mnt/data/data_b/CMIP6/ACCESS-ESM1-5/sic_raw/siconc_SImon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_sicraw_en',''))
        elif 'ACCESS-ESM1-5_hi_en' in var: # 
            path = '/dx13/pyfsiew/LENS/ACCESS-ESM1-5/hi/sithick_SImon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*_regrid.nc'%(var.replace('ACCESS-ESM1-5_hi_en',''))
        elif 'ACCESS-ESM1-5_sst_en' in var: 
            path='/dx13/pyfsiew/LENS/ACCESS-ESM1-5/sst/tos_Omon_ACCESS-ESM1-5_*_r%si1p1f1_gn_*_regrid.nc'%(var.replace('ACCESS-ESM1-5_sst_en',''))
        elif 'daily' in var:
            new_time = pd.date_range(start='1950-01-01', end='2100-12-31', freq='D') 
            if 'ACCESS-ESM1-5_V850daily_en' in var: 
                path='/dx13/pyfsiew/LENS/ACCESS-ESM1-5/v850_daily/va_day_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_V850daily_en',''))
            elif 'ACCESS-ESM1-5_T850daily_en' in var: 
                path='/dx13/pyfsiew/LENS/ACCESS-ESM1-5/t850_daily/ta_day_ACCESS-ESM1-5_*_r%si1p1f1_gn_*.nc'%(var.replace('ACCESS-ESM1-5_T850daily_en',''))
    ### DAMIP (detection and attribution MIP)
    # CESM2
    elif 'CESM2_histGHG_tas_daily_en' in var: 
        en = var.replace('CESM2_histGHG_tas_daily_en', '').zfill(3)
        path='/dx02/pyfsiew/CMIP6_DAMIP/CESM2/b.e21.B1850cmip6.f09_g17.CESM2-SF-GHG.%s.cam.h1.TREFHT.*.nc'%en
        open_single_file = False
    elif 'CESM2_histaer_tas_daily_en' in var: 
        en = var.replace('CESM2_histaer_tas_daily_en', '').zfill(3)
        path='/dx02/pyfsiew/CMIP6_DAMIP/CESM2/b.e21.B1850cmip6.f09_g17.CESM2-SF-AAER.%s.cam.h1.TREFHT.*.nc'%en
        open_single_file = False
    elif 'CESM2_histBMB_tas_daily_en' in var: 
        en = var.replace('CESM2_histBMB_tas_daily_en', '').zfill(3)
        path='/dx02/pyfsiew/CMIP6_DAMIP/CESM2/b.e21.B1850cmip6.f09_g17.CESM2-SF-BMB.%s.cam.h1.TREFHT.*.nc'%en
        open_single_file = False
    elif 'CESM2_histEE_tas_daily_en' in var: # The ensemble is from 101 to 115
        en = var.replace('CESM2_histEE_tas_daily_en', '').zfill(3)
        path='/dx02/pyfsiew/CMIP6_DAMIP/CESM2/b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.%s.cam.h1.TREFHT.*.nc'%(int(en)+100)
        open_single_file = False
    # For CanESM5
    elif 'CanESM5_histGHG_tas_daily_en' in var: # members 1-10
        en = var.replace('CanESM5_histGHG_tas_daily_en', '')
        path='/dx02/pyfsiew/CMIP6_DAMIP/CanESM5/tas_day_CanESM5_hist-GHG_r%si1p1f1_gn_18500101-20201231.nc'%en
        open_single_file = False
    elif 'CanESM5_histaer_tas_daily_en' in var: # members 1-25
        en = var.replace('CanESM5_histaer_tas_daily_en', '')
        path='/dx02/pyfsiew/CMIP6_DAMIP/CanESM5/tas_day_CanESM5_hist-aer_r%si1p1f1_gn_18500101-20201231.nc'%en
        open_single_file = False
    elif 'CanESM5_histnat_tas_daily_en' in var: # members  1-25
        en = var.replace('CanESM5_histnat_tas_daily_en', '')
        path='/dx02/pyfsiew/CMIP6_DAMIP/CanESM5/tas_day_CanESM5_hist-nat_r%si1p1f1_gn_18500101-20201231.nc'%en
        open_single_file = False
    ### Observational large ensembles (created by Peter)
    elif 'olens_slp_en' in var: # Start from 1976-01-30 (DJF-mean) to 2016-01-30 (DJF-mean)
        en = var.replace('olens_slp_en', '').zfill(4)
        path = '/dx13/pyfsiew/olens/slp/slp_member%s.nc'%en
    elif 'olens_noiseonly_slp_en' in var:
        en = var.replace('olens_noiseonly_slp_en', '').zfill(4)
        path = '/dx13/pyfsiew/olens/slp/slp_noiseonly_member%s.nc'%en
        #new_time = pd.date_range(start='1921-01-01', end='2015-12-01', freq='MS') 
    ### CMIP6 Pre-industrial control
    # CESM2 (2000-year)
    elif 'CESM2_PI_psl' in var: # Not single file
        path = '/dx13/pyfsiew/CMIP6_PI/CESM2/psl/*'; open_single_file=False; 
        new_time = xr.cftime_range(start='0001-01-01', end='2000-12-01', freq='MS')
    elif var=='CESM2_PI_sic': # Not single file  Unit (0-1). In atmospheric grid already
        path = '/dx13/pyfsiew/CMIP6_PI/CESM2/sic/*'; open_single_file=False;
        new_time = xr.cftime_range(start='0001-01-01', end='2000-12-01', freq='MS')
    elif var=='CESM2_PI_SST': 
        path = '/dx13/pyfsiew/CMIP6_PI/CESM2/SST/*'; open_single_file=False;
        new_time = xr.cftime_range(start='0001-01-01', end='2000-12-01', freq='MS')
   # MIROC6 (3200-01-01 to 3999-12-01: 800 years)
    elif 'MIROC6_PI_psl' in var: 
        path = '/dx13/pyfsiew/CMIP6_PI/MIROC6/psl/*'; open_single_file=False; 
        new_time = xr.cftime_range(start='0001-01-01', end='0800-12-01', freq='MS')
    elif 'MIROC6_PI_sic' in var: 
        path = '/dx13/pyfsiew/CMIP6_PI/MIROC6/sic/*'; open_single_file=False; 
        new_time = xr.cftime_range(start='0001-01-01', end='0800-12-01', freq='MS')
    # CanESM5 (5201-01-01 to 6200-01-01: 1000-year
    elif 'CanESM5_PI_psl' in var: 
        path = '/dx13/pyfsiew/CMIP6_PI/CanESM5/psl/*'; open_single_file=False; 
        new_time = xr.cftime_range(start='0001-01-01', end='1000-12-01', freq='MS')
    elif 'CanESM5_PI_sic' in var: # siconc - regrid (0-100)
        path = '/dx13/pyfsiew/CMIP6_PI/CanESM5/sic/*regrid.nc'; open_single_file=False; 
        new_time = xr.cftime_range(start='0001-01-01', end='1000-12-01', freq='MS')
    ### Nuding data by Ding et al. 2022 (CESM1)
    elif var in ['ding22_Z525']:
        new_time = pd.date_range(start='1979-01-01', end='2020-12-31', freq='MS') 
        path = '/dx15/pyfsiew/ding22_nudge/extract/Z3_525_enavg_1x1.nc'
    elif var in ['ding22_T860']:
        path = '/dx15/pyfsiew/ding22_nudge/extract/T_860_enavg.nc'
        new_time = pd.date_range(start='1979-01-01', end='2020-12-31', freq='MS') 
    elif var in ['ding22_aice_avg_sic_en']: # the unit is (0-100)
        path = '/dx15/pyfsiew/ding22_nudge/extract/aice_enavg.nc'
        new_time = pd.date_range(start='1979-01-01', end='2020-12-31', freq='MS') 
    elif var in ['ding22_TREFHT']: # This is the same as T2M or SAT
        path = '/dx15/pyfsiew/ding22_nudge/extract/TREFHT_enavg_1x1.nc'
        new_time = pd.date_range(start='1979-01-01', end='2020-12-31', freq='MS') 
    elif var in ['ding22_PS']: # 
        path = '/dx15/pyfsiew/ding22_nudge/extract/PS_enavg_1x1.nc'
        new_time = pd.date_range(start='1979-01-01', end='2020-12-31', freq='MS') 
    ### Nudging data by Ding et al. 2023 (CESM2)
    elif 'ding23_aice_avg_sic_en' in var: # the unit is (0-1)
        en = var.replace('ding23_aice_en','')
        #path ='/dx15/pyfsiew/ding23_nudge_cesm2/icefrac.monthly.1979-2020.mem%s.nc'%en
        path ='/dx15/pyfsiew/ding23_nudge_cesm2/icefrac.monthly.1979-2020.memavg.nc'
        new_time = pd.date_range(start='1979-01-01', end='2020-12-31', freq='MS') 
    ### NCAR CICE5+POP2, forced by JRA55
    elif var in ['FOSI_aice']: #(0-1)
        path = '/dx15/pyfsiew/NCAR-FOSI/aice_regrid.nc'
        new_time = pd.date_range(start='1958-01-01', end='2018-12-01', freq='MS') 
    elif var == 'FOSI_hi': #(0-1)
        path = '/dx15/pyfsiew/NCAR-FOSI/hi_regrid.nc'
        new_time = pd.date_range(start='1958-01-01', end='2018-12-01', freq='MS') 
    elif var in ['FOSI_sic_dynam']:
        path = '/dx15/pyfsiew/NCAR-FOSI/daidtd_regrid.nc'
        new_time = pd.date_range(start='1958-01-01', end='2018-12-01', freq='MS') 
    elif var in ['FOSI_sic_thermo']:
        path = '/dx15/pyfsiew/NCAR-FOSI/daidtt_regrid.nc'
        new_time = pd.date_range(start='1958-01-01', end='2018-12-01', freq='MS') 
    ### AMIP-exp
    # Cheng AMIP-control (WACCAM6: climatological SST, sea ice and fixed GHGs)
    elif var in ['Cheng2023_amip_control']:
        path='/dx11/czheng/Nonlinear_WACCM6/217years/BKS_SIC_100.PSL.2000-2216.monthly.nc'
        new_time = pd.date_range(start='2000-01-01', end='2217-12-01', freq='MS') 
    # CAM5 control in Jie et al. 2017 (CAM5: Climatological SST sea ice and fixed GHGs) in pre-industrial ages
    elif var in ['Jie2017_amip_control']: # This is a CAM5 control (forced by clim-SST and SIC)
        #path='/dx15/pyfsiew/jie2017/AMIP-climSST.PSL.000101-260012.nc'
        path='/dx15/pyfsiew/jie2017/AMIP-climSST.PSL.2x2.000101-260012.nc'  # Smaller region
        new_time = xr.cftime_range(start='0001-01-01', end='2600-12-01', freq='MS')
    elif var in ['Jie2017_amip_fullSST_control']: # 400-year long # AMIP forced by PI-SST
        path='/dx15/pyfsiew/jie2017/AMIP-fullSST.PSL.040101-060012.nc' 
        new_time = pd.date_range(start='2001-01-01', end='2200-12-01', freq='MS') 
    elif var == 'amip_CAM5_slp_control': # Forced by monthly clim-SST and SIC in NCAR CVWDG
        path='/dx15/pyfsiew/ncar_cam5_amip_control/f.e11.F1850C5CN.f09_f09.001.cam.h0.PSL.*.nc'
        new_time = xr.cftime_range(start='0001-01-01', end='2600-12-01', freq='MS')
        open_single_file=False
    elif var == 'amip_CAM6_slp_control': # Forced by monthly clim-SST and SIC in NCAR CVWDG
        path='/dx15/pyfsiew/ncar_cam6_amip_control/f.e21.F1850.f09_f09_mg17.SSTICE_CMIP6-piControl.001.cam.h0.PSL.*.nc'
        new_time = xr.cftime_range(start='0001-01-01', end='0999-12-01', freq='MS')
        open_single_file=False
    # NOAA FACTs AMIP forced by obs sea ice, obs SST and obs GHGs
    # monthly data
    elif 'amip_CAM5_slp_en' in var: # 40 members (Jan 1900 to Feb 2022)
        en = var.replace('amip_CAM5_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/amip_facts/ESRL-CAM5/psl_ESRL-CAM5_amip_obs_rf_ens%s.nc'%en
    elif 'amip_ECHAM5_slp_en' in var: # 50 members (Jan 1979 to Feb 2021) (lat goes from 90 to -90)
        en = var.replace('amip_ECHAM5_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/amip_facts/ECHAM5/psl/psl_ECHAM5_amip_obs_rf_ens%s.nc'%en
    elif 'amip_GFSv2_slp_en' in var: # 50 members (Jan 1979 to Feb 2018)
        en = var.replace('amip_GFSv2_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/amip_facts/ESRL-GFSv2/psl_ESRL-GFSv2_amip_obs_rf_ens%s.nc'%en
    # daily data
    elif 'amip_ECHAM5_daily_tas_en' in var: # 50 members (Jan 1979 to Feb 2021)
        en = var.replace('amip_ECHAM5_daily_tas_en', '').zfill(2)
        path='/dx15/pyfsiew/amip_facts/ECHAM5_daily/tas/tas_ECHAM5_amip_obs_rf_ens%s_*.nc'%en 
        new_time = pd.date_range(start='1979-01-01', end='2021-02-28', freq='D') 
        open_single_file=False
    # FACTS AMIP forced by obs SST, GHGs and climatological SIC
    # Monthly data
    elif 'amip_ECHAM5_climsic_slp_en' in var: # 30 members (Jan 1979 to Feb 2018)
        en = var.replace('amip_ECHAM5_climsic_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/amip_facts/ECHAM5_climsic/psl/psl_ECHAM5_amip_clim_polar_ens%s.nc'%en
    # Daily
    elif 'amip_ECHAM5_daily_climsic_tas_en' in var: # 
        en = var.replace('amip_ECHAM5_daily_climsic_tas_en', '').zfill(2)
        path='/dx15/pyfsiew/amip_facts/ECHAM5_daily_climsic/tas/tas_ECHAM5_amip_clim_polar_ens%s_*.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2018-02-28', freq='D') 
        open_single_file=False
    # WACCM6 data (provided by Yu-chiao Ling) 30 members (Jan 1979 to Dec 2014) 
    # EXP1 (with observed sea ice)
    elif 'amip_WACCM6_slp_en' in var:
        en = var.replace('amip_WACCM6_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/liang_greenice/waccm6/slp_monthly/exp1_ens%s_waccm_whoi_ncar.cam.h0.197901-201412.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2014-12-01', freq='MS') 
    elif 'amip_WACCM6_icefrac_en' in var:
        en = var.replace('amip_WACCM6_icefrac_en', '').zfill(2)
        path='/dx15/pyfsiew/liang_greenice/waccm6/icefrac_monthly/exp1_ens%s_waccm_whoi_ncar.cam.h0.197901-201412.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2014-12-01', freq='MS') 
    elif 'amip_WACCM6_u820_en' in var:
        en = var.replace('amip_WACCM6_u820_en', '').zfill(2)
        path='/dx15/pyfsiew/liang_greenice/waccm6/u820_monthly/exp1_ens%s_waccm_whoi_ncar.cam.h0.197901-201412.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2014-12-01', freq='MS') 
    # Daily data
    elif 'amip_WACCM6_daily_tas_en' in var: 
        en = var.replace('amip_WACCM6_daily_tas_en', '').zfill(2)
        path='/mnt/data/data_a/liang_greenice/waccm6/tas_daily/exp1_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%en
        open_single_file=False
        new_time = pd.date_range(start='1979-01-01', end='2014-12-31', freq='D') 
        mask = ~((new_time.month==2)&(new_time.day==29));  new_time=new_time[mask]
    elif 'amip_WACCM6_daily_slp_en' in var: 
        en = var.replace('amip_WACCM6_daily_slp_en', '').zfill(2)
        path='/mnt/data/data_a/liang_greenice/waccm6/slp_daily/exp1_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%en
        open_single_file=False
        new_time = pd.date_range(start='1979-01-01', end='2014-12-31', freq='D') 
        mask = ~((new_time.month==2)&(new_time.day==29));  new_time=new_time[mask]
    # Forced by climatology SIC (Exp2)
    elif 'amip_WACCM6_climsic_slp_en' in var: # 30 members (Jan 1979 to Dec 2014) (Provided by Liang)
        en = var.replace('amip_WACCM6_climsic_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/liang_greenice/waccm6/slp_monthly/exp2_ens%s_waccm_whoi_ncar.cam.h0.197901-201412.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2014-12-01', freq='MS') 
    elif 'amip_WACCM6_climsic_icefrac_en' in var: # 30 members (Jan 1979 to Dec 2014) (Provided by Liang)
        en = var.replace('amip_WACCM6_climsic_icefrac_en', '').zfill(2)
        path='/dx15/pyfsiew/liang_greenice/waccm6/icefrac_monthly/exp2_ens%s_waccm_whoi_ncar.cam.h0.197901-201412.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2014-12-01', freq='MS') 
    elif 'amip_WACCM6_climsic_u820_en' in var: # 30 members (Jan 1979 to Dec 2014) (Provided by Liang)
        en = var.replace('amip_WACCM6_climsic_u820_en', '').zfill(2)
        path='/dx15/pyfsiew/liang_greenice/waccm6/u820_monthly/exp2_ens%s_waccm_whoi_ncar.cam.h0.197901-201412.nc'%en
        new_time = pd.date_range(start='1979-01-01', end='2014-12-01', freq='MS') 
    # Daily data
    elif 'amip_WACCM6_climsic_daily_tas_en' in var: 
        en = var.replace('amip_WACCM6_climsic_daily_tas_en', '').zfill(2)
        path='/mnt/data/data_a/liang_greenice/waccm6/tas_daily/exp2_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%en
        open_single_file=False
        new_time = pd.date_range(start='1979-01-01', end='2014-12-31', freq='D') 
        mask = ~((new_time.month==2)&(new_time.day==29));  new_time=new_time[mask]
    # NCAR AMIP-GOGA (observed sea ice, obs SST and obs GHGs)
    elif 'amip_CAM6_slp_en' in var: # 10 members (Jan 1979 to Feb 2018)
        en = var.replace('amip_CAM6_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/ncar_cam6_amip_goga/f.*.*_BGC.f09_f09.*.ersstv5.goga.*%s.cam.h0.PSL.*-*.nc'%en
        open_single_file=False
        new_time = pd.date_range(start='1880-01-01', end='2021-04-01', freq='MS') 
    elif 'amip_CAM6_toga_slp_en' in var: # 10 members 
        en = var.replace('amip_CAM6_toga_slp_en', '').zfill(2)
        path='/dx15/pyfsiew/ncar_cam6_amip_toga/f.e21.*.f09_f09.*.ersstv5.toga.*%s.cam.h0.PSL.*.nc'%en
        open_single_file=False
        new_time = pd.date_range(start='1880-01-01', end='2019-12-01', freq='MS') 

    #############
    # For full lat and lon ranges, the best chucks of time is 500
    if open_single_file:
        var3d = xr.open_dataset(path, chunks={'time':500})
        #var3d = xr.open_dataset(path, chunks='auto')
        #var3d = xr.open_dataset(path, chunks={})
    else:
        var3d = xr.open_mfdataset(path, chunks={'time':500}, decode_times=decode_times)
    var3d = var3d.squeeze() # Remove the single dimension
    var3d = var3d.drop('lat_bnds') if 'lat_bnds' in var3d else var3d # They can be 3d
    var3d = var3d.drop('lon_bnds') if 'lon_bnds' in var3d else var3d
    var3d = var3d.drop('zlon_bnds') if 'zlon_bnds' in var3d else var3d
    var3d=var3d.drop('zlon') if 'zlon' in var3d else var3d
    ### Just for temporary
    #var3d=var3d.set_coords(("TLAT", "TLONG")) if 'TLAT' in var3d else var3d ### Just for temporary
    vars=['cesm2_ow_en'] # Vertical velocity in ocean
    #var3d = var3d.drop('lev') if 'lev' in var3d else var3d
    ### Get the variable with 3-D dimension
    var_names = [i for i in var3d.data_vars]
    dims = [len(var3d[i].dims) for i in var_names]
    var_idx = [i for i, d in enumerate(dims) if d in [3,4]][-1] # the last index
    var_name = var_names[var_idx]
    if varname is not None:
        var_name = varname
    var3d = var3d[var_name]


    # Replace the time coords with new_time, # The V4 sea ice data (cdr_seaice_conc) can decode the time itself
    if '19800101-20211231' in path:
        start_date = '1980-01-01' # 10957 days after 1950-01-01 is this
        new_time = pd.date_range(start=start_date, periods=var3d.time.size, freq='D') # The new time has a dimension of 15431
        var3d = var3d.assign_coords({'time':new_time})
    if not new_time is None:
        var3d = var3d.assign_coords({'time':new_time})

    if slicing:
        var3d = var3d[:, ::3, ::3]
    if False: # Remove 29 Feb
        mask = ~((var3d.time.dt.month==2) & (var3d.time.dt.day==29))
        var3d = var3d.sel(time=mask)
    # Already force to synchonize years of the data
    if months == 'DJF':
        var3d = var3d.sel(time=slice('1980-12-01', '2021-02-28'))
        mon_mask = var3d.time.dt.month.isin([1,2,12])
    elif months == 'DJFM':
        var3d = var3d.sel(time=slice('1980-12-01', '2021-03-31'))
        mon_mask = var3d.time.dt.month.isin([1,2,3,12])
    elif months == 'NDJFM':
        var3d = var3d.sel(time=slice('1980-11-01', '2021-03-31'))
        mon_mask = var3d.time.dt.month.isin([1,2,3,11,12])
    elif months == 'ONDJFMA':
        var3d = var3d.sel(time=slice('1980-10-01', '2021-04-30'))
        mon_mask = var3d.time.dt.month.isin([1,2,3,4,10,11,12])
    elif months == 'all':
        var3d = var3d.sel(time=slice('1980-01-01', '2021-12-31')) if limit_year else var3d
        mon_mask = var3d.time.dt.month.isin([1,2,3,4,5,6,7,8,9,10,11,12])
    var3d = var3d.sel(time=mon_mask)
    if lon_lat_name_change: # Change the lon and lat name
        if ('lat' in var3d.dims) & ('lon' in var3d.dims): 
            var3d = var3d.rename({'lat':'latitude', 'lon':'longitude'})
    if reverse_lat: # If the latitudes start from 90N, reverse (ERA5, NCEP R2, ECHAM5 AMIP)
        if var3d.latitude[0].item()>var3d.latitude[-1].item():
            var3d=var3d.isel(latitude=slice(None, None, -1)) 
    if limit_lat:
        var3d = var3d.sel(latitude=slice(20,90))
    if compute:
        var3d = var3d.compute()
    if var == 'cdr_seaice_conc': # Mask the missing sea ice data in the satellite period
        var3d = tools.mask_missing_ice_value(var3d, days_plus_one=False)


    return var3d

def read_timeseries(var, months='DJF', lag=0, remove_winter_mean=True):

    # The raw ts can have full month (Jan-Dec) or NDJFM data
    ts_raw = xr.open_dataset('/dx13/pyfsiew/MERRA2/timeseries/%s.nc'%var)[var]

    if months == 'DJF':
        ts_sel = ts_raw.sel(time=slice('1980-12-01', '2021-02-28'))
        mon_mask = ts_sel.time.dt.month.isin([12,1,2])
        ts_sel = ts_sel.sel(time=mon_mask)
    elif months == 'DJFM':
        ts_sel = ts_raw.sel(time=slice('1980-12-01', '2021-03-31'))
        mon_mask = ts_sel.time.dt.month.isin([12,1,2,3])
        ts_sel = ts_sel.sel(time=mon_mask)
    elif months == 'NDJFMA':
        ts_sel = ts_raw.sel(time=slice('1980-11-01', '2021-04-30'))
        mon_mask = ts_sel.time.dt.month.isin([11,12,1,2,3,4])
        ts_sel = ts_sel.sel(time=mon_mask)
    elif months == 'NDJFM':
        ts_sel = ts_raw.sel(time=slice('1980-11-01', '2021-03-31'))
        mon_mask = ts_sel.time.dt.month.isin([11,12,1,2,3])
        ts_sel = ts_sel.sel(time=mon_mask)
    elif months == 'ONDJFMA':
        ts_sel = ts_raw.sel(time=slice('1980-10-01', '2021-04-30'))
        mon_mask = ts_sel.time.dt.month.isin([10,11,12,1,2,3,4])
        ts_sel = ts_sel.sel(time=mon_mask)
    elif months == 'all': # Don't do any selection
        ts_sel = ts_raw

    if remove_winter_mean:
        ts_sel = tools.remove_winter_mean(ts_sel, months=months)

    if lag==0:
        pass
    else:
        lag_time= ts_sel.time.to_index() + pd.Timedelta(days=lag)
        ts_sel = ts_raw.sel(time=lag_time)

    return ts_sel

def remove_winter_mean(data, months=None):

    # Remove the interannual variability of data. It works for 29 Feb
    # It works for both map data and timeseries
    # The input data must be DJF/NDJFM/ONDJFMA data

    if months=='DJF':
        st_mon='12'; end_mon='02'; end_day ='28'
    if months=='DJFM':
        st_mon='12'; end_mon='03'; end_day ='31'
    if months=='NDJFMA':
        st_mon='11'; end_mon='03'; end_day ='31'
    elif months=='ONDJFMA':
        st_mon='10'; end_mon='04'; end_day='30'

    group_no = np.zeros(data.time.size)
    group_no = xr.DataArray(group_no, dims=['time'], coords={'time':data.time}, name='group_no')
    for year in range(1980, 2021+1):
        start_date = '%s-%s-01'%(year, st_mon)
        end_date = '%s-%s-%s'%(year+1, end_mon, end_day)
        time_sel = pd.date_range(start=start_date, end=end_date, freq='D')
        mask = data.time.isin(time_sel)
        group_no[mask] = year

    # Remove the interannual variability
    data_groups = data.groupby(group_no) # Grouping can be done because data and group_no have the same time dimension
    data_new = data_groups - data_groups.mean(dim='time')
    data_new = data_new.reset_coords('group_no', drop=True)

    return data_new

def mask_missing_ice_value(ice_data, days_plus_one=False):

    # Work for both timeseries or 3d var data
    # The problem is the data in these data have some 0 values (probably because part of the BKS box fall into the ocean grid), inregularly. So we should mask all these dates

    # 1987-12-03 to 1988-01-13: 42 days

    # Satellite daily product have the following dates sea ice missing
    missing_dates = xr.open_dataset('/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_missing_days.nc')['ice_missing_dates']
    if days_plus_one: # Also mask the the next days of those missing dates (useful because we sometimes calculated the delta change)
        missing_dates_plus = missing_dates + pd.Timedelta(days=1) # Missing days + 1 should also be masked
        missing_dates = np.concatenate([missing_dates, missing_dates_plus])

    missing_idx = np.in1d(ice_data.time, missing_dates) 
    ice_data[missing_idx] = np.nan

    return ice_data

##################################################################################3333

def pick_extreme_sea_ice(ice_ts, method='peter2022', high_or_low='low', sd=1):

    # The input is BKS timeseries DJF daily data 
    # The picking criterion is according to Cheng et al. 2021

    if high_or_low == 'low':
        mask = ice_ts < (ice_ts.std() * -1 * sd)
    elif high_or_low == 'high':
        mask = ice_ts > (ice_ts.std() * 1 * sd )

    masked_days = ice_ts.sel(time=mask).time.values
    #masked_days = mask.where(mask, drop=True).time.values

    if method == 'chengEA2021':
        days_needed = []
        days_sel = []
        for day in masked_days:
            if day in days_needed:
                continue
            days_needed = np.array([day+pd.Timedelta(days=i).to_numpy() for i in [1,2,3,4]])
            if np.all(np.in1d(days_needed,masked_days)):
                days_sel.append(day)
        days_sel = np.array(days_sel)

        days_sel_final = [days_sel[0]]
        for i, day in enumerate(days_sel[1:]):
            day_diff = (day - days_sel_final[-1]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
            if day_diff > 5:
                days_sel_final.append(day)
        days_sel_final= np.array(days_sel_final)
        ice_ts = ice_ts.sel(time=days_sel_final)

    elif method == 'peter2022': # The days should be th than the adjacent two points
        days_sel = []
        for day in masked_days:
            neighbour_days = np.array([day+pd.Timedelta(days=i).to_numpy() for i in [-2,-1,1,2]])
            #neighbour_days = np.array([day+pd.Timedelta(days=i).to_numpy() for i in [-1,1]])
            # if not all neighdays days are avaiable, skip this day
            if ~np.all(np.in1d(neighbour_days, ice_ts.time.values)):
                continue
            if high_or_low == 'low':
                # The day is more negative than the neighbouring 4 days
                test_boolean1 = ice_ts.sel(time=neighbour_days) > ice_ts.sel(time=day)
                # All neighbour days are always more negative than -1 SD
                test_boolean2 = ice_ts.sel(time=neighbour_days) < (ice_ts.std() * -1 * sd)
            elif high_or_low == 'high':
                test_boolean1 = ice_ts.sel(time=neighbour_days) < ice_ts.sel(time=day)
                test_boolean2 = ice_ts.sel(time=neighbour_days) > (ice_ts.std() * 1 * sd)
            if np.all(test_boolean1.values.tolist() + test_boolean2.values.tolist()):
                days_sel.append(day)

        ice_ts = ice_ts.sel(time=days_sel)

    return ice_ts


def remove_seasonal_cycle_and_detrend(data, detrend=False):

    # Jan to Dec daily data is required because all data is required to smooth the seasonal cycle
    # Remove the daily climatology (smoothed by 30-day seasonal cycle)
    # If Detrend is True, it is better to have data in all months so that the detrending is more physical
    # This detrend methods also works for both data with or without 29 Feb
    # It is supposed to handle nan data (for sea ice data - should mask the missing values before throwing into this function)
    # It doesn't work with monthly data

    # Create the daily group
    monday_str = data.indexes['time'].strftime('%m-%d')
    monday_str_xr = xr.DataArray(monday_str, dims=['time'], coords={'time':data.time}, name='mon_day_str')
    data_groupby = data.groupby(monday_str_xr)
    # Smoooth the climatology with a 30-day running mean. Pad the climatology at the beginning and end, and then compute the running mean
    climatology = data_groupby.mean(dim='time')
    len_climatology = climatology.mon_day_str.shape[0]
    starting_15 = climatology.isel(mon_day_str=slice(0,15))
    ending_15= climatology.isel(mon_day_str=slice(-15,len_climatology)) # The data might have 29 Feb (366 size) or without (365 size)
    climatology_pad = xr.concat([ending_15, climatology, starting_15], dim='mon_day_str') #this pad has a size of 365 (or 365) + 15 + 15
    climatology_rolling = climatology_pad.rolling(mon_day_str=31, center=True).mean().isel(mon_day_str=slice(15,15+len_climatology))
    data_anom = data_groupby - climatology_rolling
    data_anom_detrend = data_anom if not detrend else None

    if detrend:
        data_anom_detrends = []
        for mon_day in list(data_groupby.groups.keys()):
            # Extract the data in that mon_day_str
            mask = data_anom.mon_day_str == mon_day
            anom_day= data_anom.sel(time=mask)
            new_time = range(0, len(anom_day.time)) # Usually it is 0...42 because we have 42 years (except 29 Feb)
            x = xr.DataArray(new_time, dims=['time'], coords={'time':new_time})
            y = anom_day.assign_coords(time=new_time)
            xmean=x.mean(dim='time'); ymean=y.mean(dim='time')
            # Operate the linear 
            slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time')
            intercept = ymean - xmean*slope  
            # Remove the linear trend by removing y by the predicted y
            y_detrend = y - (x*slope + intercept)
            # Add back the time into the y data
            y_detrend= y_detrend.assign_coords(time=anom_day.time)
            # Rechuck at this point after grouping them
            #y_detrend = y_detrend.chunk(chunks={'time':len(anom_day.time)})
            data_anom_detrends.append(y_detrend)

        # Concat the data again
        data_anom_detrend = xr.concat(data_anom_detrends, dim='time').sortby('time')

    return data_anom_detrend.reset_coords('mon_day_str', drop=True)

def remove_seasonal_cycle_simple(data, detrend=False):

    # This also works for monhtly data
    # Just simply remove the daily climatology mean (without smoothing the seasonal cycle)
    # It doesn't need Jan to Dec data being here

    # Create the daily/monthly group
    monday_str = data.indexes['time'].strftime('%m-%d')
    monday_str_xr = xr.DataArray(monday_str, dims=['time'], coords={'time':data.time}, name='mon_day_str')
    groupby_monsdays = data.groupby(monday_str_xr)
    climatology = groupby_monsdays.mean(dim='time')
    data_anom = groupby_monsdays - climatology
    data_anom_detrend = data_anom if not detrend else None

    if detrend:
        data_anom_detrends = []
        for mon_day in list(groupby_monsdays.groups.keys()):
            # Extract the data in that mon_day_str
            mask = data_anom.mon_day_str == mon_day
            anom_day= data_anom.sel(time=mask)
            new_time = range(0, len(anom_day.time)) # Usually it is 0...42 because we have 42 years (except 29 Feb)
            x = xr.DataArray(new_time, dims=['time'], coords={'time':new_time})
            y = anom_day.assign_coords(time=new_time)
            xmean=x.mean(dim='time'); ymean=y.mean(dim='time')
            # Operate the linear 
            slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time')
            intercept = ymean - xmean*slope  
            # Remove the linear trend by removing y by the predicted y
            y_detrend = y - (x*slope + intercept)
            # Add back the time into the y data
            y_detrend= y_detrend.assign_coords(time=anom_day.time)
            data_anom_detrends.append(y_detrend)
        data_anom_detrend = xr.concat(data_anom_detrends, dim='time').sortby('time')

    return data_anom_detrend.reset_coords('mon_day_str', drop=True)

def remove_seasonal_cycle_not_used(data, days=151):

    # Obtain the anomalies by removing seasonal cycle
    # data is the 3d array with dims: time x lat x lon
    # 151 days is the number of days in Jan,Feb,Mar,Nov,Dec (omitting 29 Feb)
    # This function does not work with 29 Feb so it is not used currently

    # reshape it with n winters x days
    winters = int(data.time.shape[0] / days)
    lat_shape = int(data.latitude.shape[0])
    lon_shape = int(data.longitude.shape[0])
    # To re-shape the array so that seasonal cycle can be calculated in a easy way
    data_reshape = np.reshape(data.values, (winters, days, lat_shape, lon_shape))
    climatology = data_reshape.mean(axis=0)
    data_anom_temp = data_reshape - climatology
    # Reshape back the data to normal/value
    data_anom = np.reshape(data_anom_temp, ((winters*days), lat_shape, lon_shape))
    # Put data_anom back to a xarray format
    data_anom = xr.DataArray(data_anom, dims=data.dims, coords=data.coords)
    return data_anom


def filter_data(data, filter_type='high'):

    # Should work for both timeseries and lat,lon,time data
    # It has to contain continous Jan - Dec data
    # This only works for daily data (see Blackmon and Lau 1979). For twice-daily data: Another set of high-pass filter is used
    
    if filter_type == 'high': # it is actually the band-pass filter from 2.5-6 days
        a1toa10 = [-0.0728693709, -0.2885051308, 0.0973270826, 0.0395130908, 0.0283273699, #a1 to a5
                    0.0331625327, -0.0708879974, -0.0022652475, 0.0030189695, 0.0070759754] #a6 to a 10
        a0 = 0.4522054510
    elif filter_type == 'low': # > 10 days pass 
        a1toa10 = [0.1974416342, 0.1576890490, 0.1028784073, 0.0462514755, 0,
                  -0.0281981813, -0.0368362395, -0.0300256308, -0.0151817136, 0]
        a0 = 0.2119623984

    weight = a1toa10[::-1] + [a0] + a1toa10
    weight = xr.DataArray(weight, dims=['windows'])

    # construct here creates a new dimension named "windows" (i.e., lat x lon x time x windows). Each time has a 21 windows dimensions.
    # Multipy that 21 windows with the weight (which also has the dimension 'windows'. The windows dimension will be removed after the multiple
    data_filter = data.rolling(time=len(weight), center=True).construct('windows').dot(weight)

    # Cannot use dropna because some data has nan lat lon in some regions (e.g., V850 or Q850)
    return data_filter


def latlon_map_plotting(map2d, title="", shading_level=11, cmap='PuBu', fig_name='save', unit='', projection=ccrs.PlateCarree(central_longitude=0), low_lat=60, region_boxes=None, xlim=None, ylim=None):

    # Plot the climatology
    lons = map2d.longitude.values
    lats = map2d.latitude.values

    plt.close()
    fig = plt.figure(figsize=(6,3))

    ax1 = fig.add_subplot(1,1,1, projection=projection)
    cs = ax1.contourf(lons, lats, map2d, shading_level, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
    #cs1 = ax1.contour(lons, lats, map2d, shading_level, transform=ccrs.PlateCarree(), colors='k')
    #g1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='darkgrey', linestyle='-',
     #                       xlocs=[-180, -120, -60, 0, 60, 120, 180], ylocs=[-90,-60,-30,0,30,60,90])

    xlim = xlim if not xlim is None else [-180,180]
    ylim = ylim if not ylim is None else [0,90]
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_ylim(ylim[0],ylim[1])

    if projection == ccrs.NorthPolarStereo():
        map_extents = {'left':-180, 'right':180, 'bottom':low_lat, 'top':90}
        ax1.set_extent([map_extents['left'], map_extents['right'], map_extents['bottom'], map_extents['top']], ccrs.PlateCarree())

    if not region_boxes is None:
        lons_l, lats_l = region_boxes[0], region_boxes[1]
        ax1.plot(lons_l, lats_l, transform=ccrs.PlateCarree(), color='k', linewidth=1)

    ax1.coastlines(color='darkgray', linewidth=1)
    ax1.set_title(title, size=15)
    # Set the coloarbar
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    cax = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(cs, cax=cax, orientation='vertical', fraction=0.035)
    cbar.ax.set_ylabel(unit, rotation=0)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=200)


def map_grid_plotting(map_grids, rows, cols, mapcolors, clevels_grids, clabels_rows, top_titles=[], left_titles=[], region_boxes=None, region_boxes_extra=None, pval_map=None, sig=0.05, xlim=None, ylim=None,ind_titles=None, projection=ccrs.NorthPolarStereo(), pval_hatches=None, colorbar=True, indiv_colorbar=None, pltf=None, ax_all=None, map_extents=None, xsize=2.5, ysize=2.5, transpose=False, contour_map_grids=None, contour_clevels=None, gridline=True, leftcorner_text=None, leftcorner_text_pos=(0.02,0.98),circle=False, shading_extend='both',coastlines=True, mask_ocean=False, lat_ticks=None, lon_ticks=None, freetext=None, freetext_pos=None, quiver_grids=None, contour_color='k', contour_lw=0.5, set_xylim=None, fill_continent=False, box_outline_gray=False, set_extent=True, region_box_color='red',region_box_extra_color='k'):

    gidx= np.empty((rows,cols)).astype('int')
    for i in range(rows):
        for j in range(cols):
            gidx[i,j] = i*cols+j
    if transpose:
        rows, cols = cols, rows
        gidx = gidx.T
        left_titles, top_titles = top_titles, left_titles

    # use the Catropy
    plt.close() if pltf is None else ''
    fig = plt.figure(figsize=(cols*xsize, rows*ysize)) if pltf is None else pltf
    k=0
    for i in range(rows):
        for j in range(cols):
            if len(map_grids) <= gidx[i,j]: # If the numbers inside the map_grids is less than rows*cols
                continue
            ax = fig.add_subplot(rows, cols, k+1, projection=projection) if ax_all is None else ax_all[gidx[i,j]]
            xlim = (-180,180) if xlim is None else xlim
            ylim = (0,90) if ylim is None else ylim
            if set_extent:
                ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], ccrs.PlateCarree()) 
            if not set_xylim is None:
                xlims = set_xylim[gidx[i,j]][0]
                ylims = set_xylim[gidx[i,j]][1]
                ax.set_xlim(xlims[0], xlims[1])
                ax.set_ylim(ylims[0], ylims[1])
            if circle:
                theta = np.linspace(0,2*np.pi,100); center,radius=[0.5,0.5],0.5 ;verts=np.vstack([np.sin(theta),np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            if not map_grids is None:
                if not map_grids[gidx[i,j]] is None:
                    lons = map_grids[gidx[i,j]].longitude.values
                    lats = map_grids[gidx[i,j]].latitude.values
                    cs = ax.contourf(lons, lats, map_grids[gidx[i,j]], clevels_grids[gidx[i,j]],
                    cmap=mapcolors[gidx[i,j]],transform=ccrs.PlateCarree(), extend=shading_extend)
                    # extend can be 'neither', 'both', 'min' or 'max'
            if not contour_map_grids is None:
                if not contour_map_grids[gidx[i,j]] is None:
                    lons = contour_map_grids[gidx[i,j]].longitude.values
                    lats = contour_map_grids[gidx[i,j]].latitude.values
                    # To remove the 0 line in contour level
                    idx_0 = np.where(contour_clevels[gidx[i,j]]==0)[0] # return the index array of 0
                    if len(idx_0) == 1:  #if there is 0 in the contour levels - then delete it 
                        clevels_new = np.delete(contour_clevels[gidx[i,j]], idx_0[0])
                    else: # No change
                        clevels_new = contour_clevels[gidx[i,j]]
                    csf = ax.contour(lons, lats, contour_map_grids[gidx[i,j]], clevels_new, colors=contour_color, 
                                    linewidths=contour_lw, transform=ccrs.PlateCarree())
                    #ax.clabel(csf, inline=True, fontsize=2, fmt='%1.0f')
            if (not pval_map is None) & (not pval_hatches is None):
                if not pval_map[gidx[i,j]] is None:
                    lons = pval_map[gidx[i,j]].longitude.values
                    lats = pval_map[gidx[i,j]].latitude.values
                    ax.contourf(lons, lats, pval_map[gidx[i,j]], pval_hatches[gidx[i,j]][0], 
                            hatches=pval_hatches[gidx[i,j]][1],
                            colors='none', extend='neither', transform=ccrs.PlateCarree(),zorder=10)
            if not quiver_grids is None:
                if not quiver_grids[gidx[i,j]] is None:
                    lons = quiver_grids[gidx[i,j]][0].longitude.values
                    lats = quiver_grids[gidx[i,j]][0].latitude.values
                    x_vector = quiver_grids[gidx[i,j]][0] 
                    y_vector = quiver_grids[gidx[i,j]][1] 
                    # smaller scale, longer arrow. If scale_units is "inches" & the scale is 2, a value of 1 
                    # has a length of 0.5 inches. If scale is 5, a value of 1 has a length of 0.2 inches
                    # Regrid_shape is important. Larger means denser
                    transform = ccrs.PlateCarree()
                    Q= ax.quiver(lons, lats, x_vector.values, y_vector.values, headwidth=3, headlength=3, 
                           headaxislength=2, units='width', scale_units='inches', pivot='middle', color='black', 
                           #width=0.005, scale=200, transform=transform, regrid_shape=30, zorder=2) 
                           width=0.004, scale=200, transform=transform, regrid_shape=50, zorder=2) 
                           #width=0.004, scale=2000, transform=transform, regrid_shape=20, zorder=2) 
                           #width=0.004, scale=2000e6, transform=transform, regrid_shape=20, zorder=2) 
                           #width=0.004, scale=2000e8, transform=transform, regrid_shape=15, zorder=2) 
                    #qk = ax.quiverkey(Q, 0.7, 0.25, 10, "10", labelpos='E', coordinates='figure')
                    #qk = ax.quiverkey(Q, 0.7, 0.25, 300, "300", labelpos='E', coordinates='figure')
                    qk = ax.quiverkey(Q, 0.7, 0.15, 10, "10", labelpos='E', coordinates='figure')
            if not region_boxes is None:
                if not region_boxes[gidx[i,j]] is None:
                    lons_l, lats_l = region_boxes[gidx[i,j]][0], region_boxes[gidx[i,j]][1]
                    ax.plot(lons_l, lats_l, transform=ccrs.PlateCarree(), color=region_box_color, linewidth=1)
            if not region_boxes_extra is None:
                if not region_boxes_extra[gidx[i,j]] is None:
                    lons_l, lats_l = region_boxes_extra[gidx[i,j]][0], region_boxes_extra[gidx[i,j]][1]
                    ax.plot(lons_l, lats_l, transform=ccrs.PlateCarree(), color=region_box_extra_color, linewidth=1)
            if not leftcorner_text is None:
                if not leftcorner_text[gidx[i,j]] is None:
                    t1 = ax.annotate(leftcorner_text[gidx[i,j]], xy=leftcorner_text_pos, xycoords='axes fraction',
                        fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white', 
                        alpha=1, pad=0.001), zorder=100)
                    #t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
            if gridline:
                g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.2,
                        color='lightgray', linestyle='-',
                            xlocs=[-180, -120, -60, 0, 60, 120, 180], ylocs=[-90,-60,-30,0,30,60,90])
                            #xlocs=[-180, -120, -60, 0, 60, 120], ylocs=[-60,0,30,60,70])
                #g1 = ax.gridlines(draw_labels=False, linewidth=1, color='darkgrey', linestyle='-', alpha=0.5)
                #g1.n_steps = 90
                #g1.xlocator = matplotlib.ticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
                #g1.ylocator = matplotlib.ticker.FixedLocator([0,30,60,90])
            if fill_continent:
                import cartopy.feature as cfeature
                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',  # can be 110m
                                    facecolor='lightgrey'), zorder=-10)
                #ax.add_feature(cfeature.LAND,zorder=1)
            #if box_outline_gray:
                #import cartopy.feature as cfeature
                #ax.outline_patch.set_edgecolor('black')
                #ax.add_feature(cfeature.BORDERS, linewidth=1)
            if not lat_ticks is None:
                ax.set_yticks(lat_ticks)
            if not lon_ticks is None:
                ax.set_xticks(lon_ticks)
            if mask_ocean:
                ax.add_feature(cartopy.feature.OCEAN,facecolor=(0.5,0.5,0.5), zorder=1)
            if coastlines:
                #ax.coastlines(color='darkgray', linewidth=1)
                #ax.coastlines(resolution='10m', color='k', linewidth=1) # Can be 110m
                #ax.coastlines(resolution='50m', color='darkgray', linewidth=0.5)
                ax.coastlines(resolution='10m', color='gray', linewidth=0.3)
                #coast = cf.GSHHSFeature(scale='h')
                #ax.add_feature(coast)
            if i==0:
                ax.set_title(top_titles[j], size=10, y=0.95)
            if j==0:
                ax.text(-0.1, 0.35, '%s'%left_titles[i], rotation='vertical', transform=ax.transAxes)
                #ax.text(-0.15, 0.5, '%s'%left_titles[i], rotation='horizontal', transform=ax.transAxes)
            if not ind_titles is None:
                ax.set_title(ind_titles[gidx[i,j]], loc='left', size=9)
            if not freetext is None:
                if not freetext[gidx[i,j]] is None:
                    ax.annotate(freetext[gidx[i,j]], xy=(freetext_pos[gidx[i,j]][0],
                            freetext_pos[gidx[i,j]][1]), xycoords='axes fraction', size=10)
            if colorbar: # Vertical colobar at the right if not transpose
                colorbar_grids_plotting(fig, ax, cs, clabels_rows, i, j, rows, cols, transpose=transpose)
            if not indiv_colorbar is None: # Setup an indivudal colorbars
                if indiv_colorbar[gidx[i,j]]:
                    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
                    # size is the "thickness" of the coloar, pad is the distance from
                    cax = make_axes_locatable(ax).append_axes("bottom", size="10%", pad=0.05, axes_class=plt.Axes) 
                    cbar = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.set_xticks(cbar.ax.get_xticks())
                    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=0)
                    #cbar.ax.set_xlabel(clabels_rows[gidx[i,j]], rotation=0, horizontalalignment='right', position=(1,15))
            k=k+1

def colorbar_grids_plotting(fig, ax, cs, clabels_rows, i, j, rows, cols, transpose=False):
    # Adjust the "subplots" before creating the colorbar axes. Note that the colorbar axis dones't belong to the "subplots" object
    cb_vert_hori = "horizontal" if transpose else "vertical"
    ax_pos = ax.get_position()
    if (not transpose) & (j==(cols-1)): # the last column
        # x-start, y-start, xwidth, ywidth
        xstart = ax_pos.x1+(ax_pos.x1-ax_pos.x0)*0.05
        #ystart = ax_pos.y0+(ax_pos.y1-ax_pos.y0)*0.3 # Figure 3A
        ystart = ax_pos.y0+(ax_pos.y1-ax_pos.y0)*0.05 # 0.3
        xwidth = (ax_pos.x1-ax_pos.x0) * 0.1
        ywidth = (ax_pos.y1-ax_pos.y0)*0.9
        #ywidth = (ax_pos.y1-ax_pos.y0)*0.7 # For figure 3A
        cax = fig.add_axes([xstart, ystart, xwidth, ywidth])
        #cax = fig.add_subplot(2,6,12)
    elif transpose & (i==(rows-1)): # the last row
        xstart = ax_pos.x0+(ax_pos.x1-ax_pos.x0)*0.3
        ystart = ax_pos.y0-(ax_pos.y1-ax_pos.y0)*0.1
        xwidth = (ax_pos.x1-ax_pos.x0)*0.8
        ywidth = (ax_pos.y1-ax_pos.y0)*0.05 
        cax = fig.add_axes([xstart, ystart, xwidth, ywidth])
    else:
        cax = None
    tick_labels_rotation='vertical' if transpose else 'horizontal'
    if not cax is None:
        cbar = fig.colorbar(cs, cax=cax, orientation=cb_vert_hori)
        #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=tick_labels_rotation)
        #cbar.formatter.set_powerlimits((0, 0))
        #tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
        #cbar.locator = tick_locator
        #cbar.update_ticks()
        cbar.ax.tick_params(labelsize=10)
        cax.set_xlabel(clabels_rows[j], fontsize=10) if transpose else cax.set_xlabel(clabels_rows[i], fontsize=10)
        # The position of the colorbar ticks label
        #[(t.set_x(2),t.set_horizontalalignment('right')) for t in cbar.ax.get_yticklabels()] 
        [(t.set_x(2.5),t.set_horizontalalignment('right')) for t in cbar.ax.get_yticklabels()] 

def create_region_box(lat1, lat2, lon1, lon2):
    box_lon, box_lat = [lon1, lon2, lon2, lon1, lon1], [lat2, lat2, lat1, lat1, lat2]
    lons_l = np.array([np.linspace(j, box_lon[k+1], 30) for k, j  in enumerate(box_lon[0:-1])]).reshape(-1)
    lats_l = np.array([np.linspace(j, box_lat[k+1], 30) for k, j  in enumerate(box_lat[0:-1])]).reshape(-1)
    region_boxes = (lons_l, lats_l)
    return region_boxes

def map_fill_white_gap(latlon_map):

    # new xarray
    extra_lon_first = latlon_map.isel(longitude=0) # append the last longitude
    extra_lon_first = xr.DataArray(extra_lon_first.values.reshape(extra_lon_first.values.shape[0], -1), 
                    dims=['latitude', 'longitude'],
                    coords={'latitude':extra_lon_first.latitude, 'longitude':[-180.00]})
    extra_lon_last = latlon_map.isel(longitude=-1) # append the last longitude
    extra_lon_last = xr.DataArray(extra_lon_last.values.reshape(extra_lon_last.values.shape[0], -1), 
                    dims=['latitude', 'longitude'],
                    coords={'latitude':extra_lon_last.latitude, 'longitude':[180]})
    # minimal is required
    latlon_map_new= xr.concat([extra_lon_first, latlon_map, extra_lon_last], dim='longitude', coords='minimal') 

    return latlon_map_new

def scatter_plotting(x, y, ax, xlabel=None, ylabel=None, clabel='', mcolor=None, color_normalize=True, regress_line=False, anom=False, texts=None, legend_no = '', tsize=12, single_color='k', markersize=10, s_alpha=1, xlim=None, ylim=None, text_xadjust=0):

    if mcolor is None:
        ax.scatter(x, y, s=markersize, c=single_color, alpha=s_alpha) 
    elif isinstance(mcolor.values, np.ndarray): # Colors are some values. Plot the color bar
        mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#f4a582', '#d6604d', '#b2182b']
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        if color_normalize:
            mcolor = (mcolor - mcolor.mean()) / mcolor.std()
        cs = ax.scatter(x, y, c=mcolor, s=markersize, vmin=-3, vmax=3, cmap=cmap)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, orientation='vertical', extend='both')
        cbar.set_label(clabel, size=tsize)
    else:
        raise ValeError('check this line')

    if not texts is None:
        for i in range(len(x)):
            ax.annotate(texts[i], (x[i]+text_xadjust, y[i]), fontsize=7)

    if regress_line:
        m, __, __, __, __ = stats.linregress(x,y)
        slope = round(m, 3)
        fit = np.polyfit(x, y ,1)
        fit_fn = np.poly1d(fit) 
        corr = round(stats.pearsonr(x, y)[0], 3)
        xx = [np.min(x), np.max(x)]
        ax.plot(xx, fit_fn(xx), '-.', color='red', label=legend_no + 'corr: ' + str(corr) + '/ slope: '+str(slope), linewidth=0)
        ax.legend()

    if anom:
        ax.axvline(x=0, color='lightgray', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)

    if xlim != None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim != None:
        ax.set_ylim(ylim[0],ylim[1])
    if not xlabel is None:
        ax.set_xlabel(xlabel, size=tsize)
    if not ylabel is None:
        ax.set_ylabel(ylabel, size=tsize)

def timeseries_plotting(times, timeseries, legend_label=[None], vertical_lines=[None], xlims=None, ylims=None, fname='', pltf=None, ax=None, grid=False, xsize=8, ysize=3, xticklabels=None, xticks=None, colors=['k'], zero_hline=False, linewidth=2, linestyles=None, markersize=2, xlabel=None, ylabel=None, title=None, yticks=None, yticklabels=None, legend_label_set=True, savefig=True):

    plt.close() if pltf is None else ''
    fig = plt.figure(figsize=(xsize, ysize)) if pltf is None else pltf
    ax = fig.add_subplot(111) if ax is None else ax
    linestyles = ['solid'] * len(times) if linestyles is None else linestyles
    # Plotting the timeseries
    for i, ts in enumerate(timeseries):
        ax.plot(times[i], ts, marker='o', markersize=markersize, linestyle=linestyles[i], lw=linewidth, label=legend_label[i], color=colors[i])
        #ax.plot(times[i], ts, marker='None', markersize=0, lw=linewidth, color=colors[i])
    if not ylims is None:
        ax.set_ylim(ylims[0], ylims[1])
    if not vertical_lines[0] is None:
        for vl in vertical_lines:
            ax.axvline(x=vl, color='k', ls='--', lw=0.2)
    if legend_label_set:
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 1.3), frameon=False, columnspacing=1.5, handletextpad=0.2)
    if zero_hline:
        ax.axhline(y=0, color='gray', ls='--', lw=1)
    if not ylabel is None:
        #ax.set_ylabel(ylabel, rotation='0')
        ax.set_ylabel(ylabel)
    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not xticks is None:
        ax.set_xticks(xticks)
    if not xticklabels is None:
        ax.set_xticklabels(xticklabels)
    if not yticks is None:
        ax.set_yticks(yticks)
    if not yticklabels is None:
        ax.set_yticklabels(yticks)
    if not xlims is None:
        #ax.set_xlim(xlims[0], xlims[1])
        ax.set_xlim(times[0], times[-1])
    if not title is None:
        ax.set_title(title, size=10, loc='left')
    ax.grid() if grid else ''

    if savefig:
        plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=200)

def linregress_xarray(y, x, null_hypo=0):
	# y is usually the 3darray
	# x is the timeseries
	# x and y should have the same time dimension so that they can align together!!!!!!!!! Very important here
	# null_hypo is the null hypothsis of the slope 
	"""
	Input: Two xr.Datarrays of any dimensions with the first dim being time. 
	Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
	Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
	for y with respect to x.
	Output: Covariance, correlation, regression slope and intercept, p-value, and standard error on regression
	between the two datasets along their aligned time dimension.  
	Lag values can be assigned to either of the data, with lagx shifting x, and lagy shifting y, with the specified lag amount. 
	""" 
	#3. Compute data length, mean and standard deviation along time axis for further use: 
	size  = x.time.shape[0]
	xmean = x.mean(dim='time')
	ymean = y.mean(dim='time')
	xstd  = x.std(dim='time')
	ystd  = y.std(dim='time')

	#4. Compute covariance along time axis
	cov = ((x-xmean)*(y-ymean)).sum(dim='time', skipna=True)/size

	#5. Compute correlation along time axis
	cor = cov/(xstd*ystd)

	#6. Compute regression slope and intercept:
	slope     = cov/(xstd**2)
	intercept = ymean - xmean*slope  

	#7. Compute P-value and standard error
	#Compute t-statistics
	tstats = cor*np.sqrt(size-2)/np.sqrt(1-cor**2)
	stderr = slope/tstats

	if null_hypo!=0:
		# Calculate the standard error manually
		predicted_y = x*slope+intercept
		stderr_new = np.sqrt(((((predicted_y-y)**2).sum(dim='time'))/(size-2)) / (((x-xmean)**2).sum(dim='time')))
		tstats = (slope - null_hypo) / stderr_new

	pval = stats.t.sf(np.abs(tstats), size-2)*2
	pval = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

	results_ds = xr.Dataset()
	results_ds['covariance'] = cov
	results_ds['correlation'] = cor
	results_ds['slope'] = slope
	results_ds['intercept'] = intercept
	results_ds['pvalues'] = pval
	results_ds['standard_error'] = stderr

	#return cov,cor,slope,intercept,pval,stderr
	return results_ds

def multiple_linear_regression(y, x):
    # Provide a vectorized way for multiple linear regression
    # y = b1x1 + b2x2 + b3x3 + b4x4... + c 
    # y can be a two dimensional array and the least square solution is calculated in each column (a vectorized way)
    # x is (x1, x2, x3, x4). in each column

    # Add a constant 
    x_c = np.c_[x, np.ones(x.shape[0])]
    #x_c = np.column_stack((x, np.ones(x.shape[0])))
    betas = np.linalg.lstsq(x_c, y, rcond=None)[0]
    predict_y = np.dot(x_c, betas)

    residual = predict_y - y

    return betas, residual

def partial_correlation_new(x, y, z):

    # if y is a 3d array, reshape it into 2d frist

    # x and y is mostly a single array. If you want to vectorize the computuation. Put the arrway Y in 2d
    # z is a multidimension array (can also be single)

    # Regress X onto Z (X = m1Z1 + m2Z2 + m3Z3 ... + c) and find the residual 
    _, res1 = tools.multiple_linear_regression(x, z)

    # Regress y onto Z
    _, res2 = tools.multiple_linear_regression(y, z)

    res1_mean = np.mean(res1)
    res2_mean = np.mean(res2)

    # If the residual is 2d, reshape it back to 3d (lat, lon, time) and then calculate the correlation between them
    correlation = np.sum((res1-res1_mean)*(res2-res2_mean)) / np.sum(((res1-res1_mean) **2)*np.sum( (res2-res2_mean)**2))**0.5

    return correlation

def correlation_nan(x, y, remove_mean=False, weight=None):

    x = np.array(x); y=np.array(y)

    if weight is not None:
        x=x*weight
        y=y*weight

    # ASsume both timeseries have nan values
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if remove_mean:
        x=x-x.mean()
        y=y-y.mean()

    return stats.pearsonr(x,y)[0]

def rmse_nan(x, y):

    x = np.array(x); y=np.array(y)

    # ASsume both timeseries have nan values
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    return np.sqrt(np.mean((x-y)**2))

def linregress_nan(x, y):

    x = np.array(x); y=np.array(y)

    # ASsume both timeseries have nan values
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    return stats.linregress(x,y)[0:2]


def return_lag_date(node=1, max_day=20, add_one_day=True):

    import sys; sys.path.insert(0, '/home/pyfsiew/codes/som/')
    import fig9_total_ice_change_persist_days as fig9

    ################################# Form the composite with lead and lag
    # Read the saved dates in various regimes
    folder = '/home/pyfsiew/codes/som/node_dates/'
    m,n,seed = 3,3,0
    m,n,seed = 3,3,1
    node_date= np.load(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), allow_pickle='TRUE').item()
    ts_date = node_date[node]

    # persist_date stores the date with persistence. Lag date is the day according to the persitence
    days = range(1,max_day)
    persist_date = {}
    for day in days:
        persist_date[day] = fig9.persistent_count(ts_date, day, next_day_interval=1)

    lag_date = {day:[] for day in days}
    for day in days:
        for dd in range(day,days[-1],1):
            dates = np.array(persist_date[dd])
            if len(dates)==0:
                pass
            else:
                lag_date[day].append(dates[:,day-1])
    for day in days:
        lag_date[day]= [j for i in lag_date[day] for j in i]
    if add_one_day:
        for day in days:
            add_date= [i+pd.Timedelta(days=1) for i in lag_date[day]]
            lag_date[day] = lag_date[day] + add_date

    ### Aggregate the days into Triad 1, 2, 3 lags
    lag_date_combine = {}
    lag_date_combine[1] = [lag_date[1], lag_date[2], lag_date[3]]
    lag_date_combine[2] = [lag_date[4], lag_date[5], lag_date[6]]
    lag_date_combine[3] = [lag_date[7], lag_date[8], lag_date[9]]
    lag_date_combine[4] = [lag_date[day] for day in range(10,max_day)] 
    for day in [1,2,3,4]:
        lag_date_combine[day] = [j for i in lag_date_combine[day] for j in i]

    return lag_date, lag_date_combine, node_date


def pattern_correlation(Ymap, Xmap):

    # This function calculate the pattern correlation and regression (e.g., to obtain the timeseries from EOF)
    # Xmap is the reference map (i.e., projecting Ymap onto Xmap). Xmap is usually a the cluster centroid. It doesn't matter for pattern correlation

    Y = Ymap.values.flatten()
    X = Xmap.values.flatten()
    regression_coeff = ((X-X.mean())*(Y-Y.mean())).sum() / ((X-X.mean())**2).sum()
    correlation = ((X-X.mean())*(Y-Y.mean())).sum() / (((X-X.mean())**2).sum()*((Y-Y.mean())**2).sum())**0.5
    #slope, intercept, r, p, se = linregress(X, Y)
    return regression_coeff, correlation
