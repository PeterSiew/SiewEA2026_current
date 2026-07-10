import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import datetime as dt
import ipdb
from importlib import reload
import scipy
import os
import itertools

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)


if __name__ == "__main__":

    if False:
        ## CMIP5 RCP85 from Goelzer et al. 2025
        models=['ACCESS1.3_rcp85','CSIRO-Mk3.6.0_rcp85','HadGEM2-ES_rcp85','IPSL-CM5A-MR_rcp85','MIROC5_rcp85','NorESM1-M_rcp85']
        models=['ACCESS1.3_rcp85','CSIRO-Mk3.6.0_rcp85','IPSL-CM5A-MR_rcp85','MIROC5_rcp85','NorESM1-M_rcp85']
        exps=['scalars']
        new_time=range(2015,2101)
    if False:
        ## CMIP6 SSP585 from Goelzer et al. paper
        ## Full
        models=['CESM2_ssp585','CESM2-Leo_ssp585','CESM2-WACCM_ssp585','CNRM-CM6-1_ssp585','CNRM-ESM2-1_ssp585','EC-Earth3_ssp585',
                         'IPSL-CM6A-LR_ssp585','MPI-ESM1_ssp585','NorESM2-MM_ssp585','UKESM1-0-LL_ssp585','UKESM1-0-LL-Robin_ssp585']
        ## Select
        models=['CESM2_ssp585','CNRM-CM6-1_ssp585','CNRM-ESM2-1_ssp585','EC-Earth3_ssp585',
                         'IPSL-CM6A-LR_ssp585','MPI-ESM1-2-HR_ssp585','NorESM2-MM_ssp585','UKESM1-0-LL_ssp585']
        new_time=range(2015,2101)
        exps=['scalars'] # It just pass through
    if False:
        ## CMIP6 SSP126 from Goelzer et al. paper
        models=['CESM2_ssp126','EC-Earth3_ssp126','MPI-ESM1-2-HR_ssp126']
        new_time=range(2015,2101)
        exps=['scalars'] # It just pass through
    if False: # SSP245 exps for repeating 2300
        models=['CESM2_ssp245','IPSL-CM6A-LR_ssp245','MPI-ESM1-2-HR_ssp245']
        new_time=range(2015,2301)
        exps=['CISM16t','CISM16tc']
    if False:
        ## CMIP6 SSP585 for reapting 2100 exps up to 2300
        models=['CNRM-ESM2-1_ssp585','IPSL-CM6A-LR_ssp585','MPI-ESM1-2-HR_ssp585','UKESM1-0-LL_ssp585']
        new_time=range(2015,2301)

    #########################################

    if False: # CMIP6-ISMIP6 2300 simulations (let'us not to include this as it only consists of a single model - Norce CISM)
        models1=['CESM2_ssp126','MPI-ESM1-2-HR_ssp126']
        models2=['CESM2_ssp245','IPSL-CM6A-LR_ssp245','MPI-ESM1-2-HR_ssp245']
        models3=['CNRM-ESM2-1_ssp585','IPSL-CM6A-LR_ssp585','MPI-ESM1-2-HR_ssp585','UKESM1-0-LL_ssp585']
        models=models1+models2+models3
        models_forcings={model:model for model in models}
        exps=['CISM16t','CISM16tc'] # need to the repeating experiments
        new_time=range(2015,2301)
        ts_folder="/mnt/data/data_b/Greenland_ISMIP_2300/timeseries/"
        CMIP_en='0'

    if True: ## CMIP5-ISMIP6 2100 simulatiosn (Figure S6; left panel)
        ## Not including "exp08":'HadGEM2-ES_rcp85' here because we don't have this GMT
        models=['exp05','exp06','exp09','exp10','expa01','expa02','expa03','exp07']
        models_forcings={'exp05':'MIROC5_rcp85','exp06':'NorESM1-M_rcp85','exp07':'MIROC5_rcp26',
                         'exp09':'MIROC5_rcp85','exp10':'MIROC5_rcp85',
                         'expa01':'IPSL-CM5A-MR_rcp85','expa02':'CSIRO-Mk3-6-0_rcp85','expa03':'ACCESS1-3_rcp85'}
        exps=['scalars']
        new_time=range(2015,2101)
        #ts_folder='/mnt/data/data_b/ISMIP6_sea_level_rise_scalar/timeseries/'
        ts_folder="/mnt/data/data_b/Greenland_ISMIP6_2100_2300_scalar/ISMIP6_sea_level_rise_scalar/timeseries/"
        CMIP_en='1'
        legend='ISMIP6 forced by CMIP5 (Goelzer et al. 2020)'

    if False: # CMIP6-ISMIP6 2100 simulations (Figure S6; right panel)
        models1=['CESM2_ssp585','CNRM-CM6-1_ssp585','CNRM-ESM2-1_ssp585','EC-Earth3_ssp585',
                'IPSL-CM6A-LR_ssp585','MPI-ESM1-2-HR_ssp585','NorESM2-MM_ssp585','UKESM1-0-LL_ssp585']
        models2=['CESM2_ssp245','MPI-ESM1-2-HR_ssp245','NorESM2-MM_ssp245','UKESM1-0-LL_ssp245']
        models3=['CESM2_ssp126','EC-Earth3_ssp126','MPI-ESM1-2-HR_ssp126']
        models=models1+models2+models3
        models_forcings={model:model for model in models}
        exps=['scalars'] # It just pass through (select from all experiments)
        new_time=range(2015,2101)
        #ts_folder="/mnt/data/data_b/Greenland_ISMIP_2300/timeseries/"
        ts_folder="/mnt/data/data_b/Greenland_ISMIP6_2100_2300_scalar/Greenland_ISMIP_2300/timeseries/"
        CMIP_en='0'
        legend='ISMIP6 forced by CMIP6 (Goelzer et al. 2025)'

    ### Start reading ISMIP6 simulations (not our emulations)
    models=np.array(models)
    colors=["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99"]
    colors=["gray"]*100
    models_colors={model:colors[i] for i, model in enumerate(models)}
    years=new_time
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ###
    filenames=os.listdir(ts_folder)
    filenames_plot=[]
    filenames_not_plot=[]
    models_plot=[]
    models_not_plot=[] # those without correct time
    tss=[]
    for filename in filenames:
        #ipdb.set_trace()
        ## if the filename has any of the model we want - continue
        model_test=[model in filename for model in models]
        if np.sum(model_test)==0: 
            continue
        else:# if there is one True
            pass
        ## if the filename has any of the exp we require
        exp_test=[exp in filename for exp in exps]
        if np.sum(exp_test)==0: 
            continue
        else:
            pass
        ## Start reading the NC file
        ts=xr.open_dataset(ts_folder+filename,decode_times=time_coder)['sle']*100*-1
        ## if the time-dimentions are not equal to time (e.g., 86 for 2015-2100 projections)
        model=models[model_test].item()
        if ts.time.size==len(years):
            filenames_plot.append(filename)
            models_plot.append(model)
            pass
        else: ## These are the models and files that do not have the correct time
            filenames_not_plot.append((filename,ts.time.size))
            models_not_plot.append(model)
            continue
        #print(ts.time.size, ts.time[0].values, ts.time[-1].values)
        #ts=ts.sel(time=slice("2016-01-01","2101-01-01"))
        ts=ts.assign_coords({'time':new_time})
        ts=ts-ts.isel(time=0)
        if False: # find the control file (the control change is very small)
            pass
            #scalars_mm_GIS_ILTS_PIK_SICOPOLIS1_ctrl_proj.nc
            #strings=np.array(filename.split('_'))[[0,1,3,4,5,6]]
            #control_filename="_".join(strings)+'_ctrl_proj.nc'
            control_filename=filename.replace("_cr_","_").replace(model,"ctrl_proj")
            ts_control=xr.open_dataset(ts_folder+control_filename,decode_times=time_coder)['sle']*100*-1
            ts_control=ts_control.assign_coords({'time':new_time})
            ts_control=ts_control-ts_control.isel(time=0)
            ## Calculate the difference
            ts=ts+ts_control
        tss.append(ts)


    if True: ### Start plotting the ISMIP6 timeseies figures (not used in the appeal email, just for testing)
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(5,2))
        ## Plot ISMIP6
        ys=[]
        slr_lasts={model:[] for model in models}
        for i, ts in enumerate(tss):
            model=models_plot[i]
            color=models_colors[model]
            ax1.plot(years,ts.values,color=color,alpha=0.7,zorder=2.5,lw=0.8,label=None)
            ys.append(ts.values)
            slr_lasts[model].append(ts.values[-1])
        ## Plot the average
        ax1.plot(years,np.median(ys,axis=0),color='k',zorder=3,lw=2,label="Ensemble average (%s)"%len(ys))
        ## Create model legend
        for i, model in enumerate(models):
            color=models_colors[model]
            ax1.plot([-10],[-10],color=color,zorder=2.5,lw=2,label=model +" (%s)"%models_forcings[model]+" (%s)"%len(slr_lasts[model]))
        ## Error bar for the last year
        for i, model in enumerate(models):
            slr_last=slr_lasts[model]
            median=np.median(slr_last)
            ymin=np.percentile(slr_last,1)
            ymax=np.percentile(slr_last,99)
            yerr_min = np.array(median)-np.array(ymin)
            yerr_max = np.array(ymax)-np.array(median)
            ax1.errorbar(years[-1]+2*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[model],fmt='_',elinewidth=2,ms=3)
            ax1.errorbar(years[-1]+2*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[model],fmt='o',elinewidth=0,ms=1.5)
        ## Set legend
        ax1.set_xlim(2015,2120); ax1.set_ylim(-2,10)
        ax1.set_ylim(-5,100); ax1.set_xlim(2015,2320)
        ax1.set_xlim(2015,2120); ax1.set_ylim(-5,35)
        ax1.set_xlim(2015,2320); ax1.set_ylim(-5,150);
        ax1.set_ylabel('Sea-level contribution \n (cm)')
        ax1.axhline(y=0,color='k',linestyle='--',lw=0.9,zorder=-100)
        ax1.legend(bbox_to_anchor=(0.01,0.7), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2,fontsize=9)
        ## Save
        for j in ['right', 'top']:
            ax1.spines[j].set_visible(False)
            ax1.tick_params(axis='x', which='both',length=2)
            ax1.tick_params(axis='y', which='both',length=2)
        ##
        fig_name = 'PISM_imsip6_comparison_Goelzer2025'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

    ### Group ISMIP data by model forcing (e.g., MIROC5 RCP85, RCP26) for plotting the boxplot figures for projection at 2100 or 2300
    #models_forcings_list=list(set([models_forcings[model] for model in models]))
    models_forcings_list=list(dict.fromkeys([models_forcings[model] for model in models])) # Same as above - but preserving orders
    ismip6_tss={mfl:[] for mfl in models_forcings_list}
    for i, ts in enumerate(tss):
        ## Create ISMIP6 groups
        model=models_plot[i]
        model_forcing=models_forcings[model]
        ismip6_tss[model_forcing].append(ts.sel(time=new_time[-1]).item()) # Get the last year
    if False: # Save the ISMIP6 direct ts (only the last year 2100) - for plotting "SLR2100 versus GMT_trends_ISMIP6_emulations_GCM.py"
        save_folder='/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_direct_ISMIP6'
        for model_forcing in models_forcings_list:
            save_data=ismip6_tss[model_forcing]
            np.save("%s/%s_%s"%(save_folder,model_forcing,new_time[-1]),save_data) 



    ### Reading our emulation for plotting (these emulations are create by "use_emulator_to_predict_ISMIP6.py"
    ## Create out emulator prediction groups
    predict_param_fn='parameters_LHC_1000_8param_range' 
    ## Reading the weights
    model_multiple=20 # this is the standard
    weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
    save_Y_folder="/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_predict_ISMIP6/%s"%predict_param_fn
    emulator_tss={}
    for model_forcing in models_forcings_list:
        ts=np.load("%s/%s_pad2300_en%s_%s_%s.npy"%(save_Y_folder,model_forcing,CMIP_en,new_time[0],new_time[-1]))[:,-1] # Get the last year
        emulator_tss[model_forcing]=ts
    ## Read emulatotions for adjusted range towards ISMIP6
    predict_param_fn='parameters_LHC_1000_8param_range_ISMIP' # The parameter aims to shift towards ISMIP6
    ## Reading the weights
    model_multiple=20 # this is the standard
    weights_new=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
    save_Y_folder="/Users/home/siewpe/codes/greenland_emulator/save_Y_predicts/Y_predict_ISMIP6/%s"%predict_param_fn
    emulator_tss_new={}
    for model_forcing in models_forcings_list:
        ts=np.load("%s/%s_pad2300_en%s_%s_%s.npy"%(save_Y_folder,model_forcing,CMIP_en,new_time[0],new_time[-1]))[:,-1] # Get the last year
        emulator_tss_new[model_forcing]=ts


    ###
    ## Plot the boxplot-last year of projection
    rng=np.random.default_rng(seed=10)
    models_forcings_list_new=list(ismip6_tss.keys()) 
    plt.close()
    fig, ax1 = plt.subplots(1,1, figsize=(4,len(models_forcings_list_new)*0.7)) # Just to crop out the first ax (ax1)
    ## Create the boxplots
    colors = ('black', 'orange', 'red', 'skyblue', 'blue')
    bp_width=0.06
    xx = np.arange(len(models_forcings_list_new))
    xx1=xx+0.3
    xx2=xx+0.15
    xx3=xx
    xx4=xx-0.15
    xx5=xx-0.3
    ## Create weighted boxplot for our emulation based on history calibration (but it doesn't work with "ALL" because the weights here are not aggregated)
    boxplots1=[ismip6_tss[mfl] for mfl in models_forcings_list_new] # black boxplot
    boxplots2=[emulator_tss[mfl] for mfl in models_forcings_list_new] # orange
    boxplots3=[rng.choice(emulator_tss[mfl],size=10000,p=weights) for mfl in models_forcings_list_new] #read
    boxplots4=[emulator_tss_new[mfl] for mfl in models_forcings_list_new] #lightblue
    boxplots5=[rng.choice(emulator_tss_new[mfl],size=10000,p=weights_new) for mfl in models_forcings_list_new] #blue 
    # Start plotting 
    fliers=True
    bp1=ax1.boxplot(boxplots1,positions=xx1,showfliers=fliers,vert=False,widths=bp_width,capwidths=bp_width,
                                                                    whis=[1,99],patch_artist=True,label=legend)
    bp2=ax1.boxplot(boxplots2,positions=xx2,showfliers=fliers,vert=False,widths=bp_width,capwidths=bp_width,
                                                                    whis=[1,99],patch_artist=True,label='Our emulation (default range)')
    bp3=ax1.boxplot(boxplots3,positions=xx3,showfliers=fliers,vert=False,widths=bp_width,capwidths=bp_width,
                                                                    whis=[1,99],patch_artist=True,label='Our emulation (default range; with history-calibration)')
    bp4=ax1.boxplot(boxplots4,positions=xx4,showfliers=fliers,vert=False,widths=bp_width,capwidths=bp_width,
                                                                    whis=[1,99],patch_artist=True,label='Our emulation (new range)')
    bp5=ax1.boxplot(boxplots5,positions=xx5,showfliers=fliers,vert=False,widths=bp_width,capwidths=bp_width,
                                                                    whis=[1,99],patch_artist=True,label='Our emulation (new range; with history-calibration)')
    ## Plot no. of model/paramter ensemble for ISMIP
    #boxplots1_size=[len(ismip6_tss[mfl]) for mfl in models_forcings_list_new]
    x_adj=0.5; y_adj=-0.03; fs=6
    for i, mfl in enumerate(models_forcings_list_new):
        ## Annatote all boxplot
        ## bp1 (ISMIP6)
        ismip6_size=str(len(ismip6_tss[mfl]))
        x_loc=np.percentile(ismip6_tss[mfl],99)+x_adj; y_loc=xx[i]+0.3+y_adj
        ax1.annotate("(s:%s)"%ismip6_size, (x_loc, y_loc), fontsize=6)
        ## bp2
        emulation_size=str(len(emulator_tss[mfl]))
        x_loc=np.percentile(emulator_tss[mfl],99)+x_adj; y_loc=xx[i]+0.15+y_adj
        ax1.annotate("(s:%s)"%emulation_size, (x_loc, y_loc), fontsize=6)
        ## bp3
        x_loc=np.percentile(rng.choice(emulator_tss[mfl],size=10000,p=weights),99)+x_adj; y_loc=xx[i]+0+y_adj
        ax1.annotate("(s:%s)"%emulation_size, (x_loc, y_loc), fontsize=6)
        ## bp4
        x_loc=np.percentile(emulator_tss_new[mfl],99)+x_adj; y_loc=xx[i]-0.15+y_adj
        ax1.annotate("(s:%s)"%emulation_size, (x_loc, y_loc), fontsize=6)
        ## bp5
        x_loc=np.percentile(rng.choice(emulator_tss_new[mfl],size=10000,p=weights_new),99)+x_adj; y_loc=xx[i]-0.3+y_adj
        ax1.annotate("(s:%s)"%emulation_size, (x_loc, y_loc), fontsize=6)
    bps=[bp1,bp2,bp3,bp4,bp5]
    for i, bp in enumerate(bps):
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], color=colors[i], lw=1.5) 
        for box in bp['boxes']:
            box.set(facecolor=colors[i])
        plt.setp(bp['medians'], color='white', lw=1)
        plt.setp(bp['fliers'], marker='o', markersize=0, markerfacecolor=colors[i], markeredgecolor=colors[i]) # set size=0; not showing outliers
    ## Set axis labels
    ax1.set_ylim(xx[0]-0.5, xx[-1]+0.5)
    ax1.set_yticks(xx)
    #ylabel_text=[mfl.replace('_','\n') for mfl in models_forcings_list_new]
    ylabel_text=[mfl.replace('_','_')+"" for i, mfl in enumerate(models_forcings_list_new)] # do nothing
    ax1.set_yticklabels(ylabel_text,rotation=0)
    ax1.set_xlabel('Sea-level contribution at %s (cm)'%new_time[-1])
    ax1.axvline(x=0,color='gray',linestyle='--',lw=0.9,zorder=100)
    for x in xx:
        ax1.axhline(y=x-0.5,color='k',linestyle='-',lw=0.9,zorder=100)
    ## Set legend
    ax1.legend(bbox_to_anchor=(-0.5,0.99), ncol=1, loc='lower left', frameon=False, columnspacing=1.5,handletextpad=0.5, labelspacing=0.2,fontsize=9.5)

    #ax1.invert_yaxis()
    ## Remvoe the box lines
    for j in ['right', 'top', 'left']:
        ax1.spines[j].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=0)
    ## Save figure
    fig_name = 'boxplots_comparion_ISMIP6_emulator'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.02)
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

