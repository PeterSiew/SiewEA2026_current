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
import tools_greenland


def create_region_box(lat1, lat2, lon1, lon2):
    box_lon, box_lat = [lon1, lon2, lon2, lon1, lon1], [lat2, lat2, lat1, lat1, lat2]
    lons_l = np.array([np.linspace(j, box_lon[k+1], 30) for k, j  in enumerate(box_lon[0:-1])]).reshape(-1)
    lats_l = np.array([np.linspace(j, box_lat[k+1], 30) for k, j  in enumerate(box_lat[0:-1])]).reshape(-1)
    region_boxes = (lons_l, lats_l)
    return region_boxes


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


