import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime as dt
import ipdb
from importlib import reload


    
def weighted_area_average(var3d, lat1, lat2, lon1, lon2, lons, lats, lon_reverse=False, return_extract3d=False):

    # Put the var3d into a mask array and mask the invalid regions # useful for sea ice. But it won't change other data
    var3d = np.ma.masked_invalid(var3d)

    # lat1 and lon1 are the smaller index
    lon1_idx = (np.abs(lons-lon1)).argmin()
    lon2_idx = (np.abs(lons-lon2)).argmin()
    lat1_idx = (np.abs(lats-lat1)).argmin()
    lat2_idx = (np.abs(lats-lat2)).argmin()

    if (lat2_idx < lat1_idx):
        raise ValueError('lat1 and lat2 are wrong')
    if lon2_idx < lon1_idx:
        raise ValueError('lon1 and lon2 are wrong')

    lats_extract = lats[lat1_idx:lat2_idx+1]
    lats_mask = np.in1d(lats, lats_extract)
    lons_extract = lons[lon1_idx:lon2_idx+1]
    lons_mask = np.in1d(lons, lons_extract)
    if lon_reverse==True:
        lons_mask = ~lons_mask
        #print('longitugde reverse region are choosen')
    var3d_v = var3d[:, lats_mask, :]
    var3d_v = var3d_v[:, :, lons_mask]

    np.set_printoptions(suppress=True) # force not to print the "interger" form
    #print('Latitude extraction is from:')
    #print(lats[lats_mask])
    #print('Longitude extraction is from:')
    #print(lons[lons_mask])

    if return_extract3d:
        return var3d_v, lats[lats_mask], lons[lons_mask]

    # Here is the correct one. We donn't count the masked grids (e.g., ICE value) for the area-weighted
    # This method has been tested so that it provides a same value as the NAO
    # The new method is the same as the new old one if the data is not masked
    # The new method will create a masked value of around 1.4x than the older method for the BKS masked sea ice
    # - since the masked value (e.g., the land is not counted)
    weight_lat = np.cos(lats_extract*np.pi/180)
    lon_shape = var3d_v.shape[2] # time, lat, lon
    weight_map = np.tile(weight_lat, (lon_shape,1)).T#Make the 1dweight_lat to be a 2d-and then transpse it to go back to lat,lon
    if True: # This allows different days having a different mask (useful for ice-drift data)
        mask_copy = var3d_v.mask 
        weight_map_3d = np.repeat(weight_map[np.newaxis, :, :], var3d_v.shape[0], axis=0) # Copy mask the mask to a 3d-dim
        weight_map_mask = np.ma.array(weight_map_3d, mask=mask_copy) # Create a 3d-mask weight array
        var3d_weight = var3d_v * weight_map_mask # Apply the 3d weight masked array to the values
        var3d_area_mean=np.ma.sum(np.ma.sum(var3d_weight, axis=1),axis=1)/np.ma.sum(np.ma.sum(weight_map_mask, axis=1), axis=1)

    if False: # Capture the mask of the last day data from the data - Assume everyday (month) has the same mask
        mask_copy = var3d_v[-1].mask 
        # Check if every day has the same mask
        # for i in range(var3d_v.shape[0]): print(var3d_v[i].mask.sum()); #diff_mask_index = [i for i in range(var3d_v.shape[0]) if var3d_v[i].mask.sum()!=var3d_v[-1].mask.sum()]
        weight_map_mask = np.ma.array(weight_map, mask=mask_copy) # Apply the mask to the weighted mask
        # Mutltiply each grid by by weighted value
        var3d_weight = var3d_v * weight_map_mask[np.newaxis, :, :] 
        # calculate the area weigh mean # Remember that the weight_map should also contain the mask!!!
        var3d_area_mean = np.ma.sum(np.ma.sum(var3d_weight, axis=1), axis=1) / np.ma.sum(weight_map_mask)

    return var3d_area_mean

