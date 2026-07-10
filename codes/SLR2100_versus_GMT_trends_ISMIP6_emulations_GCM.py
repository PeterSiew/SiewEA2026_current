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


    ### Get ALL GMTs
    predict_years=range(2015,2101)

    ## CMIP5-RCP85 up to 2100: Need to add HadGEM back (it starts from 1860 rather than 1850)
    predict_models_rcp85=['ACCESS1-3_rcp85_pad2300','CSIRO-Mk3-6-0_rcp85_pad2300','IPSL-CM5A-MR_rcp85_pad2300', 
                    'MIROC5_rcp85_pad2300','NorESM1-M_rcp85_pad2300']
    predict_models_rcp26=['MIROC5_rcp26_pad2300']
    ens_rcp85=[1]*len(predict_models_rcp85)
    ens_rcp26=[1]*len(predict_models_rcp26)

    ## CMIP6-SSP585 up to 2100
    predict_models_ssp585=['CESM2_ssp585_pad2300','CNRM-CM6-1_ssp585_pad2300','CNRM-ESM2-1_ssp585_pad2300',
                    'EC-Earth3_ssp585_pad2300','IPSL-CM6A-LR_ssp585_pad2300','MPI-ESM1-2-HR_ssp585_pad2300','NorESM2-MM_ssp585_pad2300',
                    'UKESM1-0-LL_ssp585_pad2300']
    predict_models_ssp245=['CESM2_ssp245_pad2300','MPI-ESM1-2-HR_ssp245_pad2300','NorESM2-MM_ssp245_pad2300','UKESM1-0-LL_ssp245_pad2300']
    predict_models_ssp126=['CESM2_ssp126_pad2300','EC-Earth3_ssp126_pad2300','MPI-ESM1-2-HR_ssp126_pad2300']
    ens_ssp585=[0]*len(predict_models_ssp585)
    ens_ssp245=[0]*len(predict_models_ssp245)
    ens_ssp126=[0]*len(predict_models_ssp126)

    ## Adding them all together
    predict_models=predict_models_rcp85+predict_models_rcp26+predict_models_ssp585+predict_models_ssp245+predict_models_ssp126
    predict_models_ens=ens_rcp85+ens_rcp26+ens_ssp585+ens_ssp245+ens_ssp126
    predict_years=range(2015,2101)

    ### Read the GMT for the following models
    gmt_datas={}
    gmt_cum_datas={}
    time_since_last_changes={}
    gmt_cum_gradient_datas={}
    print("Read GMT for predict models")
    for model, en in zip(predict_models, predict_models_ens):
        gmt_data,gmt_cum_data,time_since_last_change,gmt_cum_gradient_data=functions_read.read_gmts([model],[en])
        gmt_datas[model+'_en%s'%en]=gmt_data[model]
        #gmt_cum_datas[predict_model+'_en%s'%en]=gmt_cum_data[predict_model]
        #time_since_last_changes[predict_model+'_en%s'%en]=time_since_last_change[predict_model]
        #gmt_cum_gradient_datas[predict_model+'_en%s'%en]=gmt_cum_gradient_data[predict_model]
    print("Read GMT for predict models - Finish")

    ### Read the emulation of these models
    #predict_param_fn='parameters_LHC_1000_8param_range_ISMIP' 
    predict_param_fn='parameters_LHC_1000_8param_range' 
    model_multiple=20 # this is the standard
    weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
    save_Y_folder="/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_predict_ISMIP6/%s"%predict_param_fn
    predict_models_new=[*gmt_datas] # Have an additioanl name of ensemble
    emulator_tss_p50={}
    emulator_tss_p25={}
    emulator_tss_p75={}
    rng = np.random.default_rng()
    for model in predict_models_new:
        ts=np.load("%s/%s_%s_%s.npy"%(save_Y_folder,model,predict_years[0],predict_years[-1])) 
        #ipdb.set_trace()
        ts=ts[:,-1] # etract the last year
        if True: # Do the weighting
            ts_weight=rng.choice(ts,size=10000,p=weights)
        else: # Not do the weighting
            ts_weight=ts
        emulator_tss_p50[model]=np.percentile(ts_weight,50).item()
        emulator_tss_p25[model]=np.percentile(ts_weight,1).item()
        emulator_tss_p75[model]=np.percentile(ts_weight,99).item()

    ### Read the direction simulation from ISMIP
    save_Y_folder="/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_direct_ISMIP6"
    ismip_tss_p50={}
    ismip_tss_p25={}
    ismip_tss_p75={}
    for i, model in enumerate(predict_models_new):
        model_new=model.split("_")
        model_new='_'.join(model_new[0:2])
        ts=np.load("%s/%s_%s.npy"%(save_Y_folder,model_new,predict_years[-1]))
        ipdb.set_trace()
        #print(model,ts.size)
        ismip_tss_p50[model]=np.percentile(ts,50).item()
        ismip_tss_p25[model]=np.percentile(ts,1).item()
        ismip_tss_p75[model]=np.percentile(ts,99).item()

    #ipdb.set_trace()
    ###
    st_yr=predict_years[0]; end_yr=predict_years[-1]
    st_yr=1900
    st_yr=1850
    xs=[]
    emulator_ys_p50=[]
    emulator_ys_p25=[]
    emulator_ys_p75=[]
    ismip_ys_p50=[]
    ismip_ys_p25=[]
    ismip_ys_p75=[]
    for i, model in enumerate(predict_models_new):
        ## Get the GMT (x-axis)
        gmt_ts=[gmt_datas[model][yr] for yr in range(st_yr,end_yr)]
        result=stats.linregress(range(len(gmt_ts)),gmt_ts)
        #xs.append(gmt_ts[-1]) # Only include the last year (2100)
        #xs.append(np.mean(gmt_ts))
        xs.append(result[0].item()*10) # The GMT trend from 1850 to 2100
        ## For emulator
        emulator_ys_p50.append(emulator_tss_p50[model])
        emulator_ys_p25.append(emulator_tss_p25[model])
        emulator_ys_p75.append(emulator_tss_p75[model])
        ## For ISMIP6
        ismip_ys_p50.append(ismip_tss_p50[model])
        ismip_ys_p25.append(ismip_tss_p25[model])
        ismip_ys_p75.append(ismip_tss_p75[model])

    ## Plot the scatter-plot
    ### Plot the scatter plot relationship between GMT and emulated SLR in 2100
    markers=[ # 1-7: The "Big" Fills
    'o', 's', 'D', '^', 'v', 'p', '*', 
    # 8-14: The "Alternate" Fills & Polygons
    'h', 'H', 'd', '<', '>', 'P', 'X',
    # 15-21: The "Thin" Glyphs & Strokelines
    '+', 'x', '1', '2', '3', '4', '$f$']
    plt.close()
    fig, ax1=plt.subplots(1,1,figsize=(4,4))
    xs=np.array(xs)
    ms=30
    for i, x in enumerate(xs):
        #x=np.array(xs)
        model=predict_models_new[i]
        model_new=model.split("_"); model_new='_'.join(model_new[0:2])
        ax1.scatter(x,ismip_ys_p50[i],s=ms,color='black',marker=markers[i],label=model_new)
        ax1.scatter(x,emulator_ys_p50[i],s=ms,color='red',marker=markers[i],zorder=200)
    ## Plot slopes for ISMIP6 (black) p50
    slope, intercept, r, p, se = stats.linregress(xs, ismip_ys_p50)
    ax1.plot(xs,xs*slope+intercept,lw=2,color='black',label=r'ISMIP6 ($y$=%s$x$ + %s)'%(round(slope,1),round(intercept,1)))
    if False:
        ## Plot slopes for ISMIP6 (black) p01
        slope, intercept, r, p, se = stats.linregress(xs, ismip_ys_p25)
        ax1.plot(xs,xs*slope+intercept,lw=0.5,color='black')
        ## Plot slopes for ISMIP6 (black) p99
        slope, intercept, r, p, se = stats.linregress(xs, ismip_ys_p75)
        ax1.plot(xs,xs*slope+intercept,lw=0.5,color='black')
    ## Slope for Emulations  (red) p50
    slope, intercept, r, p, se = stats.linregress(xs, emulator_ys_p50)
    ax1.plot(xs,xs*slope+intercept,lw=2,color='red',label=r'Our emulation ($y$=%s$x$ + %s)'%(round(slope,1),round(intercept,1)))
    if False:
        ## Slope for Emulations (red) p01
        slope, intercept, r, p, se = stats.linregress(xs, emulator_ys_p25)
        ax1.plot(xs,xs*slope+intercept,color='red',lw=0.5,linestyle='-')
        ## Slope for Emulations (red) p99
        slope, intercept, r, p, se = stats.linregress(xs, emulator_ys_p75)
        ax1.plot(xs,xs*slope+intercept,color='red',lw=0.5,linestyle='-')
    ## Set label
    ax1.set_xlabel("Trends of GMT from %s to 2100 (K/decade)"%st_yr)
    ax1.set_ylabel("Sea-level contribution at 2100 (cm)")
    ax1.set_ylim(0,25)
    #ax1.axhline(y=0,color='lightgray',linestyle='--',lw=0.9,zorder=100)
    #ax1.axvline(x=0,color='lightgray',linestyle='--',lw=0.9,zorder=100)
    ax1.legend(bbox_to_anchor=(1.05,-0.15), ncol=1, loc='lower left', frameon=True, columnspacing=1.5,handletextpad=0.5, labelspacing=0.2,fontsize=10)
    ###
    for j in ['right', 'top']:
        ax1.spines[j].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    ## Save figure
    fig_name = 'scatter_plot_GMT_versus_emulated_SLR'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.02)
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

