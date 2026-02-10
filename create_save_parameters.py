import numpy as np
from multiprocessing import Process; import multiprocessing
import ipdb
import random
from scipy import stats; import scipy

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
import matplotlib.pyplot as plt
import datetime as dt


if True: # Read the weights and plot prior and posteri parameters set

    predict_param_fn='parameters_LHC_1000_8param_range'
    predict_params=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s.npy'%predict_param_fn)
    mm=20
    weights=np.load('/Users/home/siewpe/codes/greenland_emulator/save_parameters/%s_modelmultiple%s_weights.npy'%(predict_param_fn,mm)) 

    if True: ### New 8-parameter simulations (MIROC85_2600_8var)
        ### Copy from below
        X1_min=0.1; X1_max=30
        X2_min=0.000025; X2_max=0.00008
        X3_min=0.005; X3_max=0.015
        X4_min=0.002; X4_max=0.007
        X5_min=270; X5_max=273 
        X6_min=2; X6_max=5 
        X7_min=0.2; X7_max=0.6
        X8_min=0.5; X8_max=1
        xlims=[(X1_min,X1_max),(X2_min,X2_max),(X3_min,X3_max),(X4_min,X4_max),(X5_min,X5_max),(X6_min,X6_max),(X7_min,X7_max),(X8_min,X8_max)]
        X_names=['Phimins','SIA_a','DDF_ice','DDF_snow','T_thresholds','T_stds','Refreeze_ratio','SSA_e']
        X_names_symbols={
                'DDF_ice':r'$K_{ice}$',
                'DDF_snow':r'$K_{snow}$',
                'T_thresholds':r'$T_{melt}$',
                'T_stds':r'$T_{std}$',
                'Refreeze_ratio':r'$R_{refreeze}$',
                ##
                'SIA_a':r'$E_{SIA}$',
                'SSA_e':r'$E_{SSA}$',
                'Phimins':r'$\phi_{min}$'}

    # Plot the histogram of predict_params
    param_no=8
    plt.close()
    fig,axs=plt.subplots(2,int(param_no/2),figsize=(9,4))
    axs=axs.flatten()
    bin_no=20
    bin_no=10
    bw_method=0.1
    bw_method='scott'
    for i in range(param_no):
        ## Prior
        param_values=predict_params[:,i]
        n, bins, patches = axs[i].hist(param_values, bins=bin_no, density=True, color='gray', edgecolor='gray',alpha=0.5)
        # Add KDE for prior
        kde = stats.gaussian_kde(param_values,bw_method=bw_method)
        xx = np.linspace(np.min(param_values)-np.std(param_values), np.max(param_values)+np.std(param_values), 1000)
        axs[i].plot(xx, kde(xx), color='k', lw=2,zorder=10)
        ## Posteri
        n, bins, patches = axs[i].hist(param_values, bins=bin_no, density=True, weights=weights, color='royalblue',alpha=0.5, edgecolor='royalblue')
        ## Add KDE of posteri
        kde = stats.gaussian_kde(param_values,weights=weights,bw_method=bw_method) #bw_method ==> smaller is closer to histogram
        axs[i].plot(xx, kde(xx), color='blue', lw=2,zorder=11)
        ## Set xticks
        #axs[i].set_xticks([np.min(param_values),np.max(param_values)])
        axs[i].set_xticks([xlims[i][0],xlims[i][1]])
        X_name=X_names[i]
        axs[i].set_title(X_names_symbols[X_name],loc='left')
        if i in [0,4]:
            axs[i].set_ylabel("Probability density")
        if False: ## Add lines of 25,50,75th percentile
            cdf = scipy.integrate.cumulative_trapezoid(kde(xx), xx, initial=0) # initial=0 means adding 0 in the front to make the same size as xx
            q10_idx= np.abs(cdf-0.1).argmin()
            q50_idx= np.abs(cdf-0.5).argmin()
            q90_idx= np.abs(cdf-0.9).argmin()
            axs[i].axvline(x=xx[q50_idx],color='b',linestyle='--',lw=0.9)
            axs[i].axvline(x=xx[q10_idx],color='b',linestyle='--',lw=0.9)
            axs[i].axvline(x=xx[q90_idx],color='b',linestyle='--',lw=0.9)
        if False: # Manual-way to calculate the new KDE after weighting
            new_counts=n*np.diff(bins)*500
            new_counts_int=np.array([int(round(c)) for c in new_counts])
            centres=(bins[:-1]+bins[1:])/2
            new_param_values=np.repeat(centres,new_counts_int)
            kde = stats.gaussian_kde(new_param_values,bw_method='scott')
            xx = np.linspace(np.min(new_param_values)-np.std(new_param_values), np.max(new_param_values)+np.std(new_param_values), 1000)
            #np.trapz(kde(xx[0:403]),x=xx[0:403])
    if True:
        for ax in axs:
            for j in ['right', 'top']:
                ax.spines[j].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2,direction='in')
    ### Save figures
    fig_name = 'histograms_parameters'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)


if False: # Create parameters by the Latin-Hyper-Cube
    if False: ### Default range set up nick in the old simulation (MIROC85, MIROC26, MIROC_2685mean)
        X1_min=10; X1_max=30 # smaller X1
        X2_min=0.000025; X2_max=0.00004 # bigger X2
        X3_min=0.005; X3_max=0.015
        X4_min=0.002; X4_max=0.007
        X5_min=270; X5_max=273 # smaler X5
        save_add='org_range'
    if False: ### Default range set up nick in the new simulation (MIROC_85_2600)
        X1_min=0.1; X1_max=30 # smaller X1
        X2_min=0.000025; X2_max=0.00008 # bigger X2
        X3_min=0.005; X3_max=0.015
        X4_min=0.002; X4_max=0.007
        X5_min=270; X5_max=273 # smaler X5
        save_add='new_simulation_range'
    if True: ### New 8-parameter simulations (MIROC85_2600_8var)
        X1_min=0.1; X1_max=30  # Phi-min
        X2_min=0.000025; X2_max=0.00008 # E-SIA
        X3_min=0.005; X3_max=0.015 # Ki
        X4_min=0.002; X4_max=0.007 # Ks
        X5_min=270; X5_max=273 # T-melt threshold
        X6_min=2; X6_max=5 # T-std
        X7_min=0.2; X7_max=0.6 # R-refreeze
        X8_min=0.5; X8_max=1 # E-SSA
        save_add='8param_range'

    from scipy.stats import qmc
    np.set_printoptions(suppress=True)

    errors=[]
    #for seed in range(0,100):
    seed=1
    no_sample=1000 
    no_sample=5000 
    print('Seed:',seed)
    print('Parameter samples:',no_sample)
    if False:
        a_bounds = [X1_min, X2_min, X3_min, X4_min, X5_min]
        b_bounds = [X1_max, X2_max, X3_max, X4_max, X5_max]
    else:
        a_bounds = [X1_min, X2_min, X3_min, X4_min, X5_min, X6_min, X7_min, X8_min]
        b_bounds = [X1_max, X2_max, X3_max, X4_max, X5_max, X6_max, X7_max, X8_max]

    ##
    sampler = qmc.LatinHypercube(d=len(a_bounds),optimization="random-cd",seed=seed)
    sample = sampler.random(n=no_sample)
    error=qmc.discrepancy(sample)
    sample_scaled=qmc.scale(sample, a_bounds, b_bounds)
    #sample_scaled=np.sort(sample_scaled,axis=0) # We can't sort this. 
    #errors.append(error)
    #min_error=np.min(errors)
    #idx=errors.index(min_error) # Results show that 44 is the smallest
    #ipdb.set_trace()
    np.save('/Users/home/siewpe/codes/greenland_emulator/save_parameters/parameters_LHC_%s_%s.npy'%(no_sample,save_add),sample_scaled)

if False: # Change the original parameter set (with hypens between parameters to more readable array)
    #predict_params=np.load('./parameters/parameters_48_raw.npy') # Default
    params=np.load('./save_parameters/parameters_48_raw_MIROC5_rcp85_pad2600.npy')
    X1s, X2s, X3s, X4s, X5s = [], [], [], [], []
    for j, param in enumerate(params):
        X1,X2,X3,X4,X5=param.item().split('-')
        # Calculate the mean of X2,X3,X4,X5
        X1s.append(np.array(X1,dtype=float).item())
        X2s.append(np.array(X2,dtype=float).item())
        X3s.append(np.array(X3,dtype=float).item())
        X4s.append(np.array(X4,dtype=float).item())
        X5s.append(np.array(X5,dtype=float).item())
    Xs=np.array([X1s,X2s,X3s,X4s,X5s]).transpose()
    np.save('./save_parameters/parameters_48_MIROC5_rcp85_pad2600.npy',Xs)


### The below parts are not used anymore

if False: #  Create testing parameters with given range by random draws (not used anymore - as we don't do random draw)
    if True: # Default range set up nick
        X1_min=10; X1_max=30 # smaller X1
        X2_min=0.000025; X2_max=0.00004 # bigger X2
        X3_min=0.005; X3_max=0.015
        X4_min=0.002; X4_max=0.007
        X5_min=270; X5_max=273 # smaler X5
    else: # new range
        X1_min=10/4; X1_max=30*4
        X2_min=0.000025/4; X2_max=0.00004*4
        X3_min=0.005/4; X3_max=0.015*4
        X4_min=0.002/4; X4_max=0.007*4
        X5_min=260; X5_max=283
    ### Create random param
    interval=20; np.set_printoptions(suppress=True,precision=8) # no. of internal within the drawing space
    X1_range=np.linspace(X1_min,X1_max,interval)
    X2_range=np.linspace(X2_min,X2_max,interval)
    X3_range=np.linspace(X3_min,X3_max,interval)
    X4_range=np.linspace(X4_min,X4_max,interval)
    X5_range=np.linspace(X5_min,X5_max,interval)
    Xs=[]
    X1_draws,X2_draws,X3_draws,X4_draws,X5_draws=[],[],[],[],[]
    draw_no=10000
    for i in range(0,draw_no):
        X1_draw=np.array(random.choice(X1_range)); #X1_draws.append(X1_draw.item())
        X2_draw=np.array(random.choice(X2_range)); #X2_draws.append(X2_draw.item())
        X3_draw=np.array(random.choice(X3_range)); #X3_draws.append(X3_draw.item())
        X4_draw=np.array(random.choice(X4_range)); #X4_draws.append(X4_draw.item())
        X5_draw=np.array(random.choice(X5_range)); #X5_draws.append(X5_draw.item())
        #param_draw=[np.array2string(X1_draw)+'-'+np.array2string(X2_draw)+'-'+np.array2string(X3_draw)+'-'+np.array2string(X4_draw)+'-'+np.array2string(X5_draw)]
        param_draw=[X1_draw.item(),X2_draw.item(),X3_draw.item(),X4_draw.item(),X5_draw.item()]
        Xs.append(param_draw)
    Xs=np.array(Xs)
    np.save('./save_parameters/parameters_%s.npy'%draw_no,Xs)

if False:
    
    ### Read the parameters and make the histogram (old one - the one with smallest RMSE without weights)
    #predict_params=np.load('./save_parameters/parameters_48.npy') # Default
    predict_params=np.load('./save_parameters/parameters_10000.npy') 
    ### Read predict data
    X1s, X2s, X3s, X4s, X5s = [], [], [], [], []
    for j, param in enumerate(predict_params):
        X1,X2,X3,X4,X5=param
        ## Read parameters
        X1s.append(np.array(X1,dtype=float).item())
        X2s.append(np.array(X2,dtype=float).item())
        X3s.append(np.array(X3,dtype=float).item())
        X4s.append(np.array(X4,dtype=float).item())
        X5s.append(np.array(X5,dtype=float).item())

    predict_params=np.load('./save_parameters/parameters_1000fit_10000.npy') 
    X1s_sel, X2s_sel, X3s_sel, X4s_sel, X5s_sel = [], [], [], [], []
    for j, param in enumerate(predict_params):
        X1_sel,X2_sel,X3_sel,X4_sel,X5_sel=param
        ## Read parameters
        X1s_sel.append(np.array(X1_sel,dtype=float).item())
        X2s_sel.append(np.array(X2_sel,dtype=float).item())
        X3s_sel.append(np.array(X3_sel,dtype=float).item())
        X4s_sel.append(np.array(X4_sel,dtype=float).item())
        X5s_sel.append(np.array(X5_sel,dtype=float).item())

    plt.close()
    fig,axs=plt.subplots(1,5,figsize=(10,2))
    Xs=[X1s,X2s,X3s,X4s,X5s]
    Xs_sel=[X1s_sel,X2s_sel,X3s_sel,X4s_sel,X5s_sel]
    for i, X in enumerate(Xs):
        n, bins, patches = axs[i].hist(X,20)
        X_sel=Xs_sel[i]
        axs[i].hist(X_sel,bins)
        axs[i].set_xticks([np.min(X),np.max(X)])
        #for X_indiv in X_sel:
        #    axs[i].axvline(x=X_indiv,color='k',linestyle='--',lw=0.05)
    # Save figures
    fig_name = 'histograms_parameters'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.6) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)
    


if False: # Vary one column while keeping other the means

    #predict_params=np.load('./parameters/parameters_48.npy') # Default
    predict_params=np.load('./parameters/parameters_100.npy') # Default
    param_no=predict_params.shape[1]
    Xs=predict_params
    Xs_mean = Xs.mean(axis=0)

    vary_indices=range(param_no)
    for vary_ind in vary_indices:
        Xs_new=np.zeros(Xs.shape)
        for i in range(Xs.shape[0]):
            for j in range(Xs.shape[1]):
                if j==vary_ind: # Allow varying
                    Xs_new[i,j]=Xs[i,j] 
                else: # Take from the mean
                    Xs_new[i,j]=Xs_mean[j] 
        # Save Xs_new
        print(vary_ind)
        #np.save('./parameters/parameters_48vary%s.npy'%vary_ind,Xs_new)
        ipdb.set_trace()
        np.save('./parameters/parameters_100vary%s.npy'%vary_ind,Xs_new)

