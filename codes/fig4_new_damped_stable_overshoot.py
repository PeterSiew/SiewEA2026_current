import xarray as xr
import numpy as np; np.set_printoptions(legacy='1.25') # Don't print np.flaot64 
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
#import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
#import matplotlib; matplotlib.rcParams['font.sans-serif'] = "Ubuntu"
import pandas as pd
import multiprocessing
import scipy; from scipy import stats
from sklearn.linear_model import Ridge
from importlib import reload
import os

import functions_read_gmt_training_data_seperate as functions_read; reload(functions_read)


if __name__ == "__main__":

    ### Plotting idealized GMT (Figure 4)
    plotting_fig4=True 
    if plotting_fig4: 
        cats=['optim','pledge','target','current']
        methods=['extra2300damped','extra2300stabalization','extra2300overshoot','extra2300linear']
        predict_models_A=["%s_%s"%(method,cat) for method in methods for cat in cats]
        predict_years_A=[range(2015,2301)]*len(predict_models_A)

        if True:
            predict_models_B=['stable2050to3000_current_nopad2300','stable2100to3000_current_nopad2300',
                            'stable2150to3000_current_nopad2300', 'stable2200to3000_current_nopad2300',
                            'stable2250to3000_current_nopad2300','stable2300to3000_current_nopad2300'] 
            predict_years_B=[range(2015,3001)]*len(predict_models_B)

        predict_models=predict_models_A+predict_models_B
        predict_years=predict_years_A+predict_years_B
        predict_models_ens=['1']*len(predict_models)


    ## Set training models and years
    ## Choose what regions
    region='greenland'
    region='global'
    train_models=['MIROC5_rcp85_pad2600_8var', 'MIROC5_rcp26_pad2600_8var','MIROC5_rcp2685mean_pad2600_8var','MIROC5_rcp85cooling_pad2600_8var']
    train_years={model:range(2015,2601) for model in train_models}
   

    ### Start the algorithm here (this is universial for Figures 3 and 5)
    ### Read GMT for training models
    ## train_ens has to be 1, because PISM forced by MIROC5 en1
    gmt_ens=[1]*len(train_models)
    gmt_datas,gmt_cum_datas,time_since_last_changes,gmt_cum_gradient_datas=functions_read.read_gmts(train_models,gmt_ens)
    ## Xs, Ys are for training; pism_slrs are the actual simualted SLR
    Xs,Ys,pism_slrs,records=functions_read.read_training_XY(train_models,train_years,gmt_datas,gmt_cum_datas,time_since_last_changes,slr_relative_yr=2015)

    ### Do the training 
    ### Standardize the training data (Xs to X_train_standard)
    X_mean=Xs.mean(axis=0)
    X_std=Xs.std(axis=0)
    X_train_standard=(Xs-X_mean)/X_std
    ## Standardize Ys to Y
    Y_mean=Ys.mean(axis=0)
    Y_std=Ys.std(axis=0)
    Y_train_standard=(Ys-Y_mean)/Y_std
    ### Do the training 
    regress = Ridge(alpha=700).fit(X_train_standard,Y_train_standard) # the best result based on figS1
    print('regress coeff:',regress.coef_)

    ### Start prediction here
    ## Set predict models and years
    predict_param_fn='parameters_LHC_1000_8param_range' 
    predict_params=np.load('../save_parameters/%s.npy'%predict_param_fn)
    model_multiple=20 # this is the standard
    weights=np.load('../save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,model_multiple)) 
    print('max weight:',weights.max())
    param_no=len(predict_params)
    ### Read GMT for predict models (make sure the model name includes the en - some model has en1 to en4., CSIRO-Mk3-6-0_rcp85)
    gmt_datas={}
    #gmt_cum_datas={}; time_since_last_changes={}; gmt_cum_gradient_datas={}
    for predict_model, en in zip(predict_models, predict_models_ens):
        gmt_data,gmt_cum_data,time_since_last_change,gmt_cum_gradient_data=functions_read.read_gmts([predict_model],[en])
        gmt_datas[predict_model+'_en%s'%en]=gmt_data[predict_model]
        #gmt_cum_datas[predict_model+'_en%s'%en]=gmt_cum_data[predict_model]
        #time_since_last_changes[predict_model+'_en%s'%en]=time_since_last_change[predict_model]
        #gmt_cum_gradient_datas[predict_model+'_en%s'%en]=gmt_cum_gradient_data[predict_model]


    ### Do the prediction using the trained models (Ridge Regression - regress) via multi-processing
    predict_models_new=[*gmt_datas] # this one further includes the ensemble number
    argus=[]
    for i, model in enumerate(predict_models_new):
        argu=(model,predict_params,predict_years[i],gmt_datas[model],None,X_mean,X_std,Y_mean,Y_std,regress)
        argus.append(argu)
    if True: # Multiprocessing result
        pool_no=6
        pool = multiprocessing.Pool(pool_no)
        results=pool.starmap(functions_read.do_prediction,argus)
    else: # debugging mode
        results=[]
        for argu in argus:
            result=functions_read.do_prediction(argu)
            results.append(result)
    ### Put the results into correct format
    Y_predicts={}
    for i, model in enumerate(predict_models_new):
        Y_predict=results[i]
        Y_predicts[model]=Y_predict
    Y_predict_reshape={}
    for i, model in enumerate(predict_models_new):
        model_short=predict_models[i]
        Y_predict_temp=[Y_predicts[model][j][year] for j in range(param_no) for year in predict_years[i]]
        ## no. of param x no. of year
        Y_predict_reshape[model_short]=np.array(Y_predict_temp).reshape(param_no,len(predict_years[i]))

    ### find the posteria (with history matching) from prior distribution
    slr_prior_qs={model:{} for model in predict_models}
    slr_post_qs={model:{} for model in predict_models}
    for i, model in enumerate(predict_models):
        slr_prior_qs[model][0.5]=np.percentile(Y_predict_reshape[model],50,axis=0,method="inverted_cdf")
        slr_prior_qs[model][0.17]=np.percentile(Y_predict_reshape[model],17,axis=0,method="inverted_cdf")
        slr_prior_qs[model][0.83]=np.percentile(Y_predict_reshape[model],83,axis=0,method="inverted_cdf")
        slr_post_qs[model][0.5]=np.percentile(Y_predict_reshape[model],50,axis=0,weights=weights,method="inverted_cdf")
        slr_post_qs[model][0.17]=np.percentile(Y_predict_reshape[model],17,axis=0,weights=weights,method="inverted_cdf")
        slr_post_qs[model][0.83]=np.percentile(Y_predict_reshape[model],83,axis=0,weights=weights,method="inverted_cdf")
    #for cat in cats: print(cat,slr_post_qs['extra2300linear_%s'%cat][0.5][-1])
    #for cat in cats: print(cat,slr_prior_qs['extra2300linear_%s'%cat][0.5][-1])

    if True: 
        ## print the GMT in 2300 for all extrapolation methods
        print("")
        print("the GMT in 2300 for the four scenarios")
        print('linear:',gmt_datas['extra2300linear_current_en1'][2300])
        print('damped:',gmt_datas['extra2300damped_current_en1'][2300])
        print('statbalization,',gmt_datas['extra2300stabalization_current_en1'][2300])
        print('overshoot',gmt_datas['extra2300overshoot_current_en1'][2300])
        print("")
        ## print the percentage change of the sea-level for all method in year 2300
        cat='current'
        for method in methods:
            print("the % change of SLR compared to the linear method")
            print(cat,method)
            old=slr_prior_qs['extra2300linear_%s'%cat][0.5][-1]
            new=slr_prior_qs['%s_%s'%(method,cat)][0.5][-1]
            percentage_change=(new-old)/old*100
            print(percentage_change)
            print("")

    #ipdb.set_trace()
    if plotting_fig4: # upper panel is the GMT; lower panel is the sea-level response
        ## Create the color for predict_models
        #models_colors=['black']*100
        colors_dict={'optim':'royalblue','pledge':'lightskyblue','target':'lightsalmon','current':'orangered'}
        models_colors=[]
        models_zorders=[]
        for model in predict_models:
            #models_colors=['#f7fcb9','#fed976','#fd8d3c','#e31a1c','#800026','#662506']
            if 'optim' in model:
                models_colors.append(colors_dict['optim'])
                models_zorders.append(0)
            elif 'pledge' in model:
                models_colors.append(colors_dict['pledge'])
                models_zorders.append(0)
            elif 'target' in model:
                models_colors.append(colors_dict['target'])
                models_zorders.append(0)
            elif ('extra2300' in model) & ('current' in model):
                models_colors.append(colors_dict['current'])
                models_zorders.append(0)
            elif 'stable2050to3000' in model:
                models_colors.append('#f7fcb9')
                models_zorders.append(10)
            elif 'stable2100to3000' in model:
                models_colors.append('#fed976')
                models_zorders.append(9)
            elif 'stable2150to3000' in model:
                models_colors.append('#fd8d3c')
                models_zorders.append(8)
            elif 'stable2200to3000' in model:
                models_colors.append('#e31a1c')
                models_zorders.append(7)
            elif 'stable2250to3000' in model:
                models_colors.append('#800026')
                models_zorders.append(6)
            elif 'stable2300to3000' in model:
                models_colors.append('#662506')
                models_zorders.append(5)
        ## Start plotting by looping models
        max_q=0.83; min_q=0.17
        labels=['']*100
        titles=['Damped trend','Stabilized','Overshoot','Stabilized to year 3000']
        col_nums={'linear':[0,1,2],'damped':[0],'stabalization':[1],'overshoot':[2],'stable2':[3]}
        cat_models_label={'optim':'Optimistic','pledge':'Pledges and targets','target':'2030 & 2035 targets','current':'Current policy actions'}
        plt.close()
        #fig,axs=plt.subplots(2,3,figsize=(9,3.5))
        fig,axs=plt.subplots(2,4,figsize=(11,3.5))
        axs_flatten=axs.flatten()
        for i, model in enumerate(predict_models):
            years=predict_years[i]
            ## ax1: for the GMT
            gmt=[gmt_datas[model+'_en1'][year] for year in years]
            if 'damped' in model:
                col_no=col_nums['damped']
                ls='-'; lw=2
            elif 'stabalization' in model:
                col_no=col_nums['stabalization']
                ls='-'; lw=2
            elif 'overshoot' in model:
                col_no=col_nums['overshoot']
                ls='-'; lw=2
            elif 'linear' in model:
                col_no=col_nums['linear']
                ls='--'; lw=0.7
            elif 'stable2' in model:
                col_no=col_nums['stable2']
                ls='-'; lw=2
            else:
                pass
            for col in col_no:
                axs[0,col].plot(years,gmt,color=models_colors[i],linewidth=lw,zorder=models_zorders[i],ls=ls)
            ##
            ## ax2 - slr gmt
            slr_median=slr_post_qs[model][0.5][:]
            slr_max=slr_post_qs[model][max_q][:]
            slr_min=slr_post_qs[model][min_q][:]
            for col in col_no:
                axs[1,col].plot(years,slr_median,color=models_colors[i],linewidth=lw,ls=ls,zorder=models_zorders[i],label=labels[i])
            if False: ## Error bar for the last year 
                ## don't show thoe whole range due to parameters
                axs[1,col].fill_between(years,slr_min,slr_max,fc=models_colors[i],zorder=models_zorders[i],color=models_colors[i], alpha=0.3, linewidth=0)
                ## Show the error bar at the nd
                median=slr_median[-1]
                ymin=slr_min[-1]
                ymax=slr_max[-1]
                yerr_min = np.array(median)-np.array(ymin)
                yerr_max = np.array(ymax)-np.array(median)
                for col in col_no:
                    axs[1,col].errorbar(years[-1]+20*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='_',elinewidth=2,ms=3)
                    axs[1,col].errorbar(years[-1]+20*(i+1),median,yerr=[[yerr_min],[yerr_max]],color=models_colors[i],fmt='o',elinewidth=0,ms=3)
        ## Set axis
        for i in [0,1,2]:
            axs[0,i].set_ylim(1,5.5)
            axs[1,i].set_ylim(0,50)
            axs[0,i].set_xlim(2015,2300)
            axs[1,i].set_xlim(2015,2300)
            axs[0,i].set_xticks([2100,2200,2300])
            axs[1,i].set_xticks([2100,2200,2300])
            axs[0,i].set_xticklabels([])
            axs[0,i].set_yticks([1,2,3,4,5])
        for i in [3]:
            axs[0,i].set_xlim(2015,3000)
            axs[1,i].set_xlim(2015,3000)
            axs[0,i].set_xticks([2200,2400,2600,2800,3000])
            axs[1,i].set_xticks([2200,2400,2600,2800,3000])
            axs[0,i].set_xticklabels([])
            axs[0,i].set_xticklabels([])
            axs[0,i].set_ylim(1,5.5)
            axs[1,i].set_ylim(0,160)
            axs[0,i].set_yticks([1,2,3,4,5])
        ## For indivudal axis
        axs[0,0].set_ylabel('Global mean\ntemperature (K)')
        axs[1,0].set_ylabel('Sea-level\ncontribution (cm)')
        ABC=['A','B','C','D']
        ## For titles
        for i, title in enumerate(titles):
            axs[0,i].set_title(r'$\bf{(%s)}$'%ABC[i]+' '+title)
        ## Set legend
        ## Only set legend for first row first column
        ## For first three columns
        for cat in cats:
            axs[1,0].plot([-100,-100],[-100,-100],color=colors_dict[cat],lw=2,label=cat_models_label[cat])
        axs[1,0].legend(bbox_to_anchor=(-0.02,0.45), ncol=1, loc='lower left', frameon=False,
                        columnspacing=0.5,handletextpad=0.5, labelspacing=0.2, reverse=True,fontsize=9)
        ## For last column - stabalization until year-3000
        for i, model in enumerate(predict_models):
            if model not in predict_models_B:
                continue
            axs[1,3].plot([-100,100],[-100,100],color=models_colors[i],lw=2,label=model[6:10])
        axs[1,3].legend(bbox_to_anchor=(0,0.3), ncol=1, loc='lower left', frameon=False,
                        columnspacing=0.5,handletextpad=0.5, labelspacing=0.2, reverse=True,fontsize=9)
        ## Save figures
        for ax in axs_flatten:
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        fig_name = 'fig4_new_extrapolation_stable_damped_overshoot'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.18,hspace=0.2) # hspace is the vertical
        plt.savefig(f"../graphs/{dt.date.today()}_{fig_name}.png", bbox_inches='tight', dpi=500, pad_inches=0.01)


    ### Plot the percentage increase for the stabalization up to year-3000 (Figure SX)
    if True:
        slr3000_2050stable=slr_post_qs['stable2050to3000_current_nopad2300'][0.5][-1]
        slr3000_2300stable=slr_post_qs['stable2300to3000_current_nopad2300'][0.5][-1]
        print("SLR at yr3000 for stabalization at 2050: ",slr3000_2050stable)
        print("SLR at yr3000 for stabalization at 2300 ",slr3000_2300stable)
        percent_increase=(slr3000_2300stable-slr3000_2050stable)/slr3000_2050stable*100
        print("the percentage increase is: ",percent_increase)
        print("")
        stable_years={'stable2050to3000_current_nopad2300':2050, 'stable2100to3000_current_nopad2300':2100,
                      'stable2150to3000_current_nopad2300':2150, 'stable2200to3000_current_nopad2300':2200,
                      'stable2250to3000_current_nopad2300':2250, 'stable2300to3000_current_nopad2300':2300} 
        testing_years=[2300,3000]
        plt.close()
        fig,axs=plt.subplots(len(testing_years),1,figsize=(3,1.5*len(testing_years)))
        if len(testing_years)==1:
            axs=[axs]
        #ipdb.set_trace()
        for i, test_year in enumerate(testing_years):
            year_idx=predict_years[-1].index(test_year)
            percentage_changes=[]
            for j, model in enumerate(predict_models_B):
                slr_model=slr_post_qs[model][0.5][year_idx]
                ## Set the old as 2300
                #slr_2300stable=slr_post_qs['stable2300to3000_current_nopad2300'][0.5][year_idx]
                #percentage_change=(slr_model-slr_2300stable)/slr_2300stable*100
                ## Set the old as 2050
                slr_2050stable=slr_post_qs['stable2050to3000_current_nopad2300'][0.5][year_idx]
                percentage_change=(slr_model-slr_2050stable)/slr_2050stable*100
                percentage_changes.append(round(percentage_change,2))
            print("SLR percentage changes in test_year %s are %s for GMT stabalizations in %s"%(test_year,percentage_changes,predict_models_B))
            print("")
            ###
            ### Start plotting for each testing year
            #years=[2050,2100,2150,2200,2250,2300]
            x=range(len(predict_models_B))
            axs[i].plot(x,percentage_changes,marker='X',markersize=2)
            axs[i].set_xticks(x)
            #axs[i].set_yticks([0,-25,-50,-75])
            #axs[i].set_yticks([0,-25,-50])
            axs[i].set_yticks([0,50,100,150])
            axs[i].set_ylim(-5,160)
            axs[i].axhline(y=0,color='lightgray',linestyle='--',lw=0.5)
            axs[i].grid()
            #axs[i].axhline(y=100,color='lightgray',linestyle='--',lw=0.5)
            #axs[i].axhline(y=200,color='lightgray',linestyle='--',lw=0.5)
            #axs[i].set_title(test_year,loc='left')
            axs[i].annotate('Year: %s'%test_year,xy=(0.01,0.8),xycoords='axes fraction', fontsize=10)
            #if i==2:
            axs[i].set_ylabel("% change of\nsea level")
            if i==len(testing_years)-1:
                axs[i].set_xlabel("Year of GMT stabalisation")
                axs[i].set_xticklabels([stable_years[m] for m in predict_models_B])
            else:
                axs[i].set_xticklabels([])
        for ax in axs:
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        ## Save figures
        fig_name = 'figSX_expotential_SLR_decrease'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1,hspace=0.3) # hspace is the vertical
        plt.savefig(f"../graphs/{dt.date.today()}_{fig_name}.png", bbox_inches='tight', dpi=500, pad_inches=0.01)
