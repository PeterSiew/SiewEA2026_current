import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import datetime as dt
import ipdb
from importlib import reload
import scipy
import os

import sys; sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools


if __name__ == "__main__":

    ## sea leve rise scalar file names
    filenames={'ctrlProj':'1990-2100-ctrlProj',
            'MIROC5_rcp85_pad':'1990-2300-MIROC_85',
            'MIROC5_rcp26_pad':'1990-2300-MIROC_26',
            'MIROC5_rcp2685mean_pad':'1990-2300-MIROC_2685mean',
            ## New
            'MIROC5_rcp85_pad2600':'1990-2600-5km-MIROC_85',
            'MIROC5_rcp85_pad2600_hiMelt':'1990-2600-5km_hiMelt-MIROC_85',
            ## Super new
            'MIROC5_rcp85_pad2600_8var':'1990-2600-5km-MIROC_85',
            'MIROC5_rcp26_pad2600_8var':'1990-2600-5km-MIROC_26',
            'MIROC5_rcp2685mean_pad2600_8var':'1990-2600-5km-MIROC_2685mean',
            'MIROC5_rcp85cooling_pad2600_8var':'1990-2600-5km-MIROC_85-cooling'}
    foldernames={'ctrlProj':'ctrlProj',
            'MIROC5_rcp26_pad':'MIROC_26',
            'MIROC5_rcp85_pad':'MIROC_85',
            'MIROC5_rcp2685mean_pad':'MIROC_2685mean',
            ## New
            'MIROC5_rcp85_pad2600':'MIROC_85_2600',
            'MIROC5_rcp85_pad2600_hiMelt':'MIROC_85_2600_hiMelt',
            ## SUper New
            'MIROC5_rcp85_pad2600_8var':'MIROC_85_2600_8var',
            'MIROC5_rcp26_pad2600_8var':'MIROC_26_2600_8var',
            'MIROC5_rcp2685mean_pad2600_8var':'MIROC_2685mean_2600_8var',
            'MIROC5_rcp85cooling_pad2600_8var':'MIROC_85_2600_8var_cooling'}
    models_years={'ctrlProj':range(1985,2101),
            'MIROC5_rcp26_pad':range(1985,2301),
            'MIROC5_rcp85_pad':range(1985,2301),
            'MIROC5_rcp2685mean_pad':range(1985,2301),
            ## new
            'MIROC5_rcp85_pad2600':range(1985,2601),
            'MIROC5_rcp85_pad2600_hiMelt':range(1985,2601),
            ## Super new
            'MIROC5_rcp85_pad2600_8var':range(1985,2601),
            'MIROC5_rcp26_pad2600_8var':range(1985,2601),
            'MIROC5_rcp2685mean_pad2600_8var':range(1985,2601),
            'MIROC5_rcp85cooling_pad2600_8var':range(1985,2601)}

    models=['MIROC5_rcp85_pad2600']
    models=['MIROC5_rcp85_pad2600_hiMelt']
    models=['MIROC5_rcp26_pad','MIROC5_rcp85_pad', 'MIROC5_rcp2685mean_pad']
    models=['MIROC5_rcp85_pad2600','MIROC5_rcp85_pad2600_hiMelt', 'MIROC5_rcp85_pad2600_8var']
    ### The latest setup run 
    models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var','MIROC5_rcp2685mean_pad2600_8var','MIROC5_rcp85cooling_pad2600_8var']
    labels=['PISM forced by RCP85','PISM forced by RCP26', 'PISM forced by RCP2685', 'PISM forced by RCP85-cooling']
    colors=['red','green','orange','gray']


    relative_to_control=False
    slr_ratio=100 # m to cm
    slr_relative_yr=2015

    ### Read Sea level rise and patermeters
    slr_datas={}
    for i, model in enumerate(models):
        path='/data/ungol_03/shared-data/PISM_GIS/outputs/%s/'%foldernames[model]
        file_list=os.listdir(path)
        file_list=[i for i in file_list if 'timeser-ens-%s'%filenames[model] in i]
        #if model=='MIROC5_rcp85_pad2600_8var':
        #    file_list=file_list[0:48] # all ensmebles
        new_time = models_years[model]
        datas=[]
        params=[]
        for fn in file_list:
            data=xr.open_dataset(path+fn)['sea_level_rise_potential']
            if relative_to_control: # Read in the control and substract that
                control_path='/data/ungol_03/shared-data/PISM_GIS/outputs/ctrlProj48/'
                control_fn=fn.replace("timeser-ens-1990-2300-%s"%model, "timeser-ens-1990-2300-ctrlProj")
                data_control=xr.open_dataset(control_path+control_fn)['sea_level_rise_potential']
                data=data_control-data
            datas.append(data)
            param=fn.split('-')[-5:]
            param[-1]=param[-1][0:-3]
            params.append('-'.join(param))
        datas=xr.concat(datas,dim='params')
        datas=datas.assign_coords({'params':params})
        datas=datas.assign_coords({'time':new_time})
        # Select dates
        if not relative_to_control: # Let not relative to control, then relative to the first year date, and then inverse it
            #datas_first_year=datas.isel(time=0)
            #datas_first_year=datas.sel(time=2015)
            datas_first_year=datas.sel(time=slr_relative_yr)
            datas=datas-datas_first_year; datas=datas*-1
            datas=datas*slr_ratio 
        slr_datas[model]=datas
        #np.save('./parameters_48',model_datas[model].params.values)

    ### Start plotting the supplemenatary figure
    plot_imbie=True
    predict_slr_relative_yr=2015
    ###
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(4,2))
    for i, var in enumerate(models):
        params=slr_datas[var].params
        for param in params:
            x=slr_datas[var].time
            y=slr_datas[var].sel(params=param)
            ax1.plot(x,y,color=colors[i],lw=0.5,alpha=0.5)
            ax1.scatter(x[-1]+10+i*15,y[-1],color=colors[i],s=3)
        ax1.plot(x,slr_datas[var].mean(dim='params'),color=colors[i],lw=3,zorder=10,label="%s (%s)"%(labels[i],len(params)))
    ### Ploting IMBIE
    if plot_imbie: 
        data=np.genfromtxt('/Users/home/siewpe/codes/greenland_emulator/save_obsSLR/imbie_obs/imbie_greenland_2021_mm.csv',delimiter=',') # Latest dataset (1992-2020)
        obs_years=data[:,0][1:]
        ice_loss=data[:,3][1:]*0.1 # from mm to cm (this is the cumulative mass balance in mm)
        ice_loss=xr.DataArray(ice_loss,dims='time')
        uncertainty=data[:,4][1:]*0.1 
        uncertainty=xr.DataArray(uncertainty,dims='time')
        ## Calculate the annual-mean
        obs_years=range(1992,2021)
        ice_loss=ice_loss.coarsen(time=12).mean().assign_coords({'time':obs_years})
        uncertainty=uncertainty.coarsen(time=12).mean().assign_coords({'time':obs_years})
        ### Set them relative to a year
        if predict_slr_relative_yr is not None:
            # The first year has to be 1992. Need to check - why it is not equal to 0?
            ice_loss_first=ice_loss.sel(time=predict_slr_relative_yr) 
        else:
            ice_loss_first=ice_loss.isel(time=0)
        ax1.axvline(x=predict_slr_relative_yr,color='k',linestyle='--',lw=0.9)
        ax1.axvline(x=2100,color='k',linestyle='--',lw=0.9)
        ax1.axhline(y=0,color='k',linestyle='--',lw=0.9)
        ice_loss=ice_loss-ice_loss_first
        loss_min=ice_loss-uncertainty*2
        loss_max=ice_loss+uncertainty*2
        ice_loss_imbie=ice_loss
        ax1.plot(obs_years,ice_loss,'k',zorder=200,lw=2,label='IMBIE')
        #ax1.plot(obs_years,loss_min,'k',linestyle='--',zorder=200)
        #ax1.plot(obs_years,loss_max,'k',linestyle='--',zorder=200)
    ###
    ax1.legend(bbox_to_anchor=(-0.15,1), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.2, labelspacing=0.1, fontsize=10)
    ax1.set_ylabel("Sea-level contribution\n(cm)")
    #ax1.set_ylim(-10,220); ax1.set_xlim(1991,2650)
    #ax1.set_xlim(1991,2025); ax1.set_ylim(-10,10)
    for j in ['right', 'top']:
        ax1.spines[j].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    fig_name = 'PISM_New_Greenland_2600'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

    ### Compare "MIROC5_rcp85_pad2600_8var" with the ISMIP6 exp05 (forced by MIROC5 8.5) in 2100
    if True: 
        # data is download here: https://zenodo.org/records/3939037
        # mm is the Greenland-wide; exp05_05 is the MIRCO5 8.5; is the control removed
        ismip6_models =['AWI-ISSM1','AWI-ISSM2','AWI-ISSM3','BGC-BISICLES','GSFC-ISSM',
                'ILTS_PIK-SICOPOLIS1','ILTS_PIK-SICOPOLIS2','IMAU-IMAUICE1','IMAU-IMAUICE2',
                'JPL-ISSM','JPL-ISSMPALEO','LSCE-GRISLI','MUN-GSM1','MUN-GSM2',
                'NCAR-CISM','UAF-PISM1','UAF-PISM2','UCIJPL-ISSM1','UCIJPL-ISSM2','VUB-GISM','VUW-PISM']
        path='~/codes/greenland_emulator/ISMIP6_Greenland/ISMIP6_sea_level_rise_scalar/v7_CMIP5_pub/'
        ismip_slrs=[]
        data_controls={}
        for model in ismip6_models :
            institute,version=model.split('-')
            data=xr.open_dataset('%s/%s/%s/exp05_05/scalars_mm_cr_GIS_%s_%s_exp05.nc'%(path,institute,version,institute,version)) # MIROC5_85 forced (strongest)
            data=data['sle'].values*-1*100
            data=xr.DataArray(data,dims=['time'],coords={'time':range(2015,2101)})
            ## For the control
            data_control=xr.open_dataset('%s/%s/%s/ctrl_proj_05/scalars_mm_GIS_%s_%s_ctrl_proj.nc'%(path,institute,version,institute,version)) 
            data_control=data_control['sle']*-1*100
            data_control=data_control-data_control.isel(time=0)
            data_control=xr.DataArray(data_control.values,dims=['time'],coords={'time':range(2015,2101)})
            #data_controls[model]=data_control
            ###
            #ismip_slrs.append(data+data_control)
            ismip_slrs.append(data)
        ismip_slrs=xr.concat(ismip_slrs,dim='models')
    ## Start plotting
    plt.close()
    fig,ax1=plt.subplots(1,1,figsize=(5,2))
    ## Plot ISMIP6
    ys=[]
    for model in range(ismip_slrs.models.size):
        y=ismip_slrs[model]
        ax1.plot(ismip_slrs.time,y,color='blue',alpha=0.3,zorder=2.5,lw=0.8)
        ys.append(y)
    ax1.plot(ismip_slrs.time,np.array(ys).mean(axis=0),color='blue',zorder=3,lw=2,label='ISMIP6 models forced by MIROC5 RCP85 (exp05)')
    ## Plot Nicks experiemnt
    param_no=slr_datas['MIROC5_rcp85_pad2600_8var'].params.size
    ys=[]
    for en in range(param_no):
        y=slr_datas['MIROC5_rcp85_pad2600_8var'].isel(params=en).sel(time=slice(2015,2100))
        ax1.plot(range(2015,2101),y,color='red',alpha=0.5,zorder=1,lw=0.5)
        ys.append(y)
    ax1.plot(range(2015,2101),np.array(ys).mean(axis=0),color='red',zorder=2.1,lw=2,label='PISM forced by MIROC5 RCP85')
    ax1.legend(bbox_to_anchor=(0.01,0.7), ncol=1, loc='lower left', frameon=False, columnspacing=0.5,handletextpad=0.5, labelspacing=0.2,fontsize=9)
    ##
    for j in ['right', 'top']:
        ax1.spines[j].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    ##
    ax1.set_xlim(2015,2100)
    ax1.set_ylabel('Sea-level contribution \n (cm)')
    ## Save
    fig_name = 'PISM_imsip6_comparison'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)




    ### Plotting indivudal models
    if False:
        if False: ## hi-MElt
            var='MIROC5_rcp26_pad'; label='MIROC5 RCP26'
            var='MIROC5_rcp85_pad2600_hiMelt'; label='MIROC5 RCP85 hiMelt'
            params=slr_datas[var].params
            for param in params:
                x=slr_datas[var].time
                y=slr_datas[var].sel(params=param)
                ax1.plot(x,y,color='orangered',lw=0.5,alpha=0.5)
                ax1.scatter(x[-1]+10,y[-1],color='orangered',s=3)
            ax1.plot(x,slr_datas[var].mean(dim='params'),color='red',lw=2,label=label)
        if False: ## Old 2600 simulation
            var='MIROC5_rcp85_pad'; label='MIROC5 RCP85'
            var='MIROC5_rcp85_pad2600'; label='MIROC5 rcp85 pad 2600'
            params=slr_datas[var].params
            for param in params:
                x=slr_datas[var].time
                y=slr_datas[var].sel(params=param)
                ax1.plot(x,y,color='royalblue',lw=0.5,alpha=0.5)
                ax1.scatter(x[-1]+20,y[-1],color='royalblue',s=3)
            ax1.plot(x,slr_datas[var].mean(dim='params'),color='blue',lw=2,label=label)
        if False: ## Old 2600 simulation
            var='MIROC5_rcp2685mean_pad'; label='MIROC5 RCP2685mean'
            var='MIROC5_rcp85_pad2600_8var'; label='MIROC5 RCP85 8var'
            params=slr_datas[var].params
            for param in params:
                x=slr_datas[var].time
                y=slr_datas[var].sel(params=param)
                ax1.plot(x,y,color='green',lw=0.5,alpha=0.5)
                ax1.scatter(x[-1]+30,y[-1],color='green',s=3)
            ax1.plot(x,slr_datas[var].mean(dim='params'),color='forestgreen',lw=2,label=label)
        if False:
            slr_cal_mean=slr_datas['MIROC5_rcp26_pad']*0.5+slr_datas['MIROC5_rcp85_pad']*0.5
            params=slr_cal_mean.params
            for param in params:
                x=slr_cal_mean.time
                y=slr_cal_mean.sel(params=param)
                ax1.plot(x,y,color='gray',lw=0.5,alpha=0.5)
                ax1.scatter(x[-1]+30,y[-1],color='gray',s=3)
            ax1.plot(x,slr_datas[var].mean(dim='params'),color='black',lw=2,label=label)
        ###
    if False: ## If the 2686_mean equal to the mean of 26 and 85 SLR forcing
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(2,2))
        params=slr_datas['MIROC5_rcp2685mean_pad'].params
        slr_cal_mean=slr_datas['MIROC5_rcp26_pad']*0.5+slr_datas['MIROC5_rcp85_pad']*0.5
        for param in params:
            aa=slr_cal_mean.sel(params=param)
            bb=slr_datas['MIROC5_rcp2685mean_pad'].sel(params=param)
            ax1.scatter(aa,bb,s=3)
        ax1.plot([-10,40],[-10,40])
        ax1.set_xlabel('Real RCP2685mean')
        ax1.set_ylabel('Calculated RCP2685mean')
        fig_name = 'scatter_plot_PISM_2685mean'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
