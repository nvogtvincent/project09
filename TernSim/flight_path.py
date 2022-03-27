#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares flight routes across CMIP6 ensemble
@author: Noam Vogt-Vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import os
from netCDF4 import Dataset
from datetime import timedelta
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import cmasher as cmr
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = {'terns_per_release': 2000,
         'fig_fn': 'tern_routes.png',

         # Plotting variables
         'res'           : 2, # (degrees)
         'bounds'        : {'lon_min' : -90,
                            'lon_max' : +60,
                            'lat_min' : -75,
                            'lat_max' : +70},
         'fig_fn_1'        : 'histogram.png',
         'fig_fn_2'        : 'limits.png',

         'source_bnds'   : {'lon_min' : -60,
                            'lon_max' : -40,
                            'lat_min' : -70,
                            'lat_max' : -70},
         'sink_lat'      : 60.,
         'target_coords' : {'lon'     : -20,
                            'lat'     : 65},
         'cmap1'          : cmr.flamingo_r,
         'cmap2'          : cmr.fusion_r
         }

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ_DATA/',
        'figs': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/'}

# FILE HANDLES
fh = {'traj': sorted(glob(dirs['traj'] + '*.nc')),
      'fig': dirs['figs'] + param['fig_fn']}

# Make a list of model names
model_names = []

get_model_name = lambda file_name : file_name.split('/')[-1].split('_')[0]
get_scen_name = lambda file_name : file_name.split('/')[-1].split('_')[1]

for i in range(len(fh['traj'])):
    model_name = get_model_name(fh['traj'][i])
    if model_name not in model_names:
        model_names.append(model_name)

# Check all files are present
for i in range(len(model_names)):
    for scenario in ['HISTORICAL', 'SSP245', 'SSP585']:
        check_name = dirs['traj'] + model_names[i] + '_' + scenario + '_TRAJ.nc'
        if check_name not in fh['traj']:
            raise FileNotFoundError(check_name + ' does not exist!')

print('Model names:')
print(model_names)

##############################################################################
# PROCESS DATA                                                               #
##############################################################################

# Define years for averages
hist_years = np.arange(1950, 2000)
scen_years = np.arange(2050, 2100)

# Create grids for binning
x_bnd = np.arange(param['bounds']['lon_min'],
                  param['bounds']['lon_max']+param['res'],
                  param['res'])

y_bnd = np.arange(param['bounds']['lat_min'],
                  param['bounds']['lat_max']+param['res'],
                  param['res'])

grid = np.zeros((3, len(y_bnd)-1, len(x_bnd)-1), dtype=np.float64) # HIST/245/585

n_traj_tot = [0, 0, 0]

# Grid the model output
for model_i, model in enumerate(model_names):
    for scenario_i, scenario in enumerate(['HISTORICAL', 'SSP245', 'SSP585']):
        output_name = dirs['traj'] + model + '_' + scenario + '_TRAJ.nc'
        print('Gridding ' + scenario + ' ' + model + '...')

        with Dataset(output_name, mode='r') as nc:
            t0 = nc.variables['time'][:, 0]
            nyr = int(len(t0)/(param['terns_per_release']*3))

            if scenario_i == 0:
                release_year = 2015-nyr

            else:
                if nyr != 86:
                    raise ValueError('Unexpected number of years in scenario data!')
                else:
                    release_year = 2101-nyr

            year_list = np.arange(release_year, release_year+nyr)
            year_list = np.repeat(year_list, param['terns_per_release']*3)

            assert len(year_list) == len(t0)

            if scenario_i == 0:
                traj_idx = np.where(np.isin(year_list, hist_years))[0]
            else:
                traj_idx = np.where(np.isin(year_list, scen_years))[0]

            # Load and grid data
            lon = nc.variables['lon'][traj_idx, :]
            lat = nc.variables['lat'][traj_idx, :]
            n_traj = np.shape(lon)[0]+1 # Number of unique terns gathered
            n_traj_tot[scenario_i] += n_traj

            tid = np.zeros_like(lon) # Trajectory identifier
            tid[:] = np.arange(np.shape(lon)[0]).reshape(-1, 1)+1
            tid = np.ma.masked_array(tid, mask=lon.mask).compressed()
            lon = lon.compressed()
            lat = lat.compressed()

            # Split arrays if necessary
            n_splits = n_traj*len(x_bnd)*len(y_bnd)/5e8
            n_splits = 1 if n_splits < 1 else int(np.ceil(n_splits))

            lon = np.array_split(lon, n_splits)
            lat = np.array_split(lat, n_splits)
            tid = np.array_split(tid, n_splits)

            # Now calculate the number of unique trajectories passing through each cell, and normalise
            for i in range(len(lon)):
                grid_ = np.histogramdd(np.array([lon[i], lat[i], tid[i]]).T, bins=[x_bnd, y_bnd, np.arange(tid[i].min(), tid[i].max()+2, 1)-0.5])[0]
                grid_[grid_ > 1] = 1
                grid[scenario_i, :, :] += np.sum(grid_, axis=2).T

for i in range(3):
    grid[i, :, :] /= n_traj_tot[i]

# Now calculate differences
# grid[0] = HIST, grid[1] = SSP245, grid[2] = SSP585, grid[3] = SSP245-HIST, grid[4] = SSP585-HIST
grid = np.concatenate([grid,
                       (grid[1,:,:]-grid[0,:,:]).reshape(1, np.shape(grid)[1], np.shape(grid)[2]),
                       (grid[2,:,:]-grid[0,:,:]).reshape(1, np.shape(grid)[1], np.shape(grid)[2])],
                       axis=0)

##############################################################################
# PLOT DATA                                                                  #
##############################################################################

##############################################################################
# HISTORICAL TERN LOCATIONS ##################################################
##############################################################################

f = plt.figure(constrained_layout=True, figsize=(27, 20))
gs = GridSpec(2, 6, figure=f, width_ratios=[2, 1, 0.05, 0.1, 1, 0.05])
ax = []
ax.append(f.add_subplot(gs[:, 0], projection = ccrs.Robinson(central_longitude=-30)))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.Robinson(central_longitude=-30)))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.Robinson(central_longitude=-30)))
ax.append(f.add_subplot(gs[0, 4], projection = ccrs.Robinson(central_longitude=-30)))
ax.append(f.add_subplot(gs[1, 4], projection = ccrs.Robinson(central_longitude=-30)))
ax.append(f.add_subplot(gs[:, 2]))
ax.append(f.add_subplot(gs[:, 5]))
label_list = ['a', 'b', 'd', 'c', 'e']

data_crs = ccrs.PlateCarree()

f.subplots_adjust(hspace=0.08, wspace=0.08)

gl = []
hist = []

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='gray',
                                        zorder=1)

for i in range(5):
    ax[i].set_aspect(1)
    ax[i].set_extent([-85, 25, -90, 90], crs=data_crs)

    # Add start/finish line
    start_line = ax[i].plot(np.array([param['source_bnds']['lon_min'],
                                      param['source_bnds']['lon_max']]),
                            np.array([param['source_bnds']['lat_min'],
                                      param['source_bnds']['lat_max']]),
                            'k-', linewidth=2, transform=data_crs,
                            zorder=100)

    start_bnds = ax[i].scatter(np.array([param['source_bnds']['lon_min'],
                                         param['source_bnds']['lon_max']]),
                               np.array([param['source_bnds']['lat_min'],
                                         param['source_bnds']['lat_max']]),
                               c='k', marker='.', s = 100,
                               zorder=100,
                               transform=data_crs)

    target_pnt = ax[i].scatter(param['target_coords']['lon'],
                               param['target_coords']['lat'],
                               c='k', marker='+', s=100,
                               transform=data_crs,
                               zorder=100)

    # Add cartographic features
    gl.append(ax[i].gridlines(crs=data_crs, draw_labels=True,
                              linewidth=0.5, color='black', linestyle='--', zorder=11))
    gl[i].xlocator = mticker.FixedLocator(np.arange(-210, 210, 60))
    gl[i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
    gl[i].xlabels_top = False
    gl[i].ylabels_right = False
    gl[i].ylabels_left = False if i > 0 else True
    gl[i].ylabel_style = {'size': 18}
    gl[i].xlabel_style = {'size': 18}

    ax[i].add_feature(land_10m)

    # Plot the colormesh
    if i <= 2:
        cmap = param['cmap1']
        hist.append(ax[i].contourf(0.5*(x_bnd[:-1]+x_bnd[1:]), 0.5*(y_bnd[:-1]+y_bnd[1:]), 100*grid[i, :, :],
                                   cmap=cmap, levels=np.linspace(2, 20, num=10), transform=data_crs,
                                   zorder=2, extend='max'))
    else:
        cmap = param['cmap2']
        hist.append(ax[i].contourf(0.5*(x_bnd[:-1]+x_bnd[1:]), 0.5*(y_bnd[:-1]+y_bnd[1:]), 100*np.ma.masked_where(grid[i, :, :] == 0, grid[i, :, :]),
                             cmap=cmap, levels=np.linspace(-4, 4, num=17), transform=data_crs,
                             zorder=2, extend='both'))

    xpos = -122 if i == 0 else -120

    ax[i].text(xpos, -85, label_list[i], transform=data_crs, fontsize=30, va='bottom', ha='left', fontweight='bold')

cb1 = plt.colorbar(hist[0], cax=ax[5])
cb1.set_label('Percentage of vTerns passing through cell on northbound migration', size=18)

cb2 = plt.colorbar(hist[3], cax=ax[6])
cb2.set_label('Change in vTern percentage passing through cell for Scenario vs Historical', size=18)

