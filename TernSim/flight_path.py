#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares flight routes across CMIP6 ensemble
@author: Noam Vogt-Vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cmocean.cm as cm
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
         'res'           : 5, # (degrees)
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
         'cmap1'          : cmr.torch_r,
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

# Grid the model output
for model_i, model in enumerate(model_names):
    for scenario_i, scenario in enumerate(['HISTORICAL', 'SSP245', 'SSP585']):
        output_name = dirs['traj'] + model_names[i] + '_' + scenario + '_TRAJ.nc'

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
            lon = nc.variables['lon'][traj_idx, :].compressed()
            lat = nc.variables['lat'][traj_idx, :].compressed()

            grid[scenario_i, :, :] += np.histogram2d(lon, lat,
                                                     bins=[x_bnd, y_bnd],
                                                     normed=True)[0].T

# Remove grids with extremely small (insiginificant) tern numbers
grid[grid < 1e-4] = 0

##############################################################################
# PLOT DATA                                                                  #
##############################################################################

##############################################################################
# HISTORICAL TERN LOCATIONS ##################################################
##############################################################################

# Plotting ocean trajectories
f, ax = plt.subplots(1, 1, figsize=(15, 10),
                     subplot_kw={'projection': ccrs.Robinson(central_longitude=-30)})
data_crs = ccrs.PlateCarree()

f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
ax.set_aspect(1)
ax.set_extent([-95, 35, -90, 90], crs=data_crs)

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='black')

# Add start/finish line
start_line = ax.plot(np.array([param['source_bnds']['lon_min'],
                                param['source_bnds']['lon_max']]),
                      np.array([param['source_bnds']['lat_min'],
                                param['source_bnds']['lat_max']]),
                      'k-', linewidth=2, transform=data_crs,
                      zorder=100)

start_bnds = ax.scatter(np.array([param['source_bnds']['lon_min'],
                                   param['source_bnds']['lon_max']]),
                         np.array([param['source_bnds']['lat_min'],
                                   param['source_bnds']['lat_max']]),
                         c='k', marker='.', s = 100,
                         zorder=100,
                         transform=data_crs)

target_pnt = ax.scatter(param['target_coords']['lon'],
                         param['target_coords']['lat'],
                         c='k', marker='+', s=100,
                         transform=data_crs,
                         zorder=100)


# Set up the colorbar
pos_cax = f.add_axes([ax.get_position().x1+0.015,ax.get_position().y0-0.0,0.015,ax.get_position().height])

# Add cartographic features
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=0.5, color='black', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 60))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
gl.xlabels_top = False
gl.ylabels_right = False

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='gray',
                                        zorder=1)
ax.add_feature(land_10m)

# Plot the colormesh
cmap1 = param['cmap1']

hist = ax.pcolormesh(x_bnd, y_bnd, np.ma.masked_where(grid[0]==0, grid[0]),
                     cmap=cmap1, vmin=0, vmax=np.max(grid[0]),
                     transform=ccrs.PlateCarree(), zorder=2,
                     alpha=0.8)

cb = plt.colorbar(hist, cax=pos_cax)
cb.set_label('Simulated Arctic Tern locations on northbound migration, 1950-1999', size=12)

##############################################################################
# PAST/FUTURE DIFFERENCES (SSP245) ###########################################
##############################################################################

# Plotting ocean trajectories
f, ax = plt.subplots(1, 1, figsize=(15, 10),
                     subplot_kw={'projection': ccrs.Robinson(central_longitude=-30)})
data_crs = ccrs.PlateCarree()

f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
ax.set_aspect(1)
ax.set_extent([-95, 35, -90, 90], crs=data_crs)

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='black')

# Add start/finish line
start_line = ax.plot(np.array([param['source_bnds']['lon_min'],
                                param['source_bnds']['lon_max']]),
                      np.array([param['source_bnds']['lat_min'],
                                param['source_bnds']['lat_max']]),
                      'k-', linewidth=2, transform=data_crs,
                      zorder=100)

start_bnds = ax.scatter(np.array([param['source_bnds']['lon_min'],
                                   param['source_bnds']['lon_max']]),
                         np.array([param['source_bnds']['lat_min'],
                                   param['source_bnds']['lat_max']]),
                         c='k', marker='.', s = 100,
                         zorder=100,
                         transform=data_crs)

target_pnt = ax.scatter(param['target_coords']['lon'],
                         param['target_coords']['lat'],
                         c='k', marker='+', s=100,
                         transform=data_crs,
                         zorder=100)


# Set up the colorbar
pos_cax = f.add_axes([ax.get_position().x1+0.015,ax.get_position().y0-0.0,0.015,ax.get_position().height])

# Add cartographic features
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=0.5, color='black', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 60))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
gl.xlabels_top = False
gl.ylabels_right = False

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='gray',
                                        zorder=1)
ax.add_feature(land_10m)

# Plot the colormesh
cmap2 = param['cmap2']
grid10 = grid[1]/grid[0]

hist = ax.pcolormesh(x_bnd, y_bnd, np.ma.masked_where(grid10==0, grid10),
                     cmap=cmap2, norm=colors.LogNorm(vmin=0.1, vmax=10),
                     transform=ccrs.PlateCarree(), zorder=2, alpha=0.8)

cb = plt.colorbar(hist, cax=pos_cax)
cb.set_label('Normalised tern trajectory difference for SSP245 vs Historical, [2050-2099]/[1950-1999]', size=12)

##############################################################################
# PAST/FUTURE DIFFERENCES (SSP585) ###########################################
##############################################################################

# Plotting ocean trajectories
f, ax = plt.subplots(1, 1, figsize=(15, 10),
                     subplot_kw={'projection': ccrs.Robinson(central_longitude=-30)})
data_crs = ccrs.PlateCarree()

f.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
ax.set_aspect(1)
ax.set_extent([-95, 35, -90, 90], crs=data_crs)

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='black')

# Add start/finish line
start_line = ax.plot(np.array([param['source_bnds']['lon_min'],
                                param['source_bnds']['lon_max']]),
                      np.array([param['source_bnds']['lat_min'],
                                param['source_bnds']['lat_max']]),
                      'k-', linewidth=2, transform=data_crs,
                      zorder=100)

start_bnds = ax.scatter(np.array([param['source_bnds']['lon_min'],
                                   param['source_bnds']['lon_max']]),
                         np.array([param['source_bnds']['lat_min'],
                                   param['source_bnds']['lat_max']]),
                         c='k', marker='.', s = 100,
                         zorder=100,
                         transform=data_crs)

target_pnt = ax.scatter(param['target_coords']['lon'],
                         param['target_coords']['lat'],
                         c='k', marker='+', s=100,
                         transform=data_crs,
                         zorder=100)


# Set up the colorbar
pos_cax = f.add_axes([ax.get_position().x1+0.015,ax.get_position().y0-0.0,0.015,ax.get_position().height])

# Add cartographic features
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=0.5, color='black', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 60))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 30))
gl.xlabels_top = False
gl.ylabels_right = False

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='gray',
                                        zorder=1)
ax.add_feature(land_10m)

# Plot the colormesh
cmap2 = param['cmap2']
grid20 = grid[2]/grid[0]

hist = ax.pcolormesh(x_bnd, y_bnd, np.ma.masked_where(grid20==0, grid20),
                     cmap=cmap2, norm=colors.LogNorm(vmin=0.1, vmax=10),
                     transform=ccrs.PlateCarree(), zorder=2, alpha=0.8)

cb = plt.colorbar(hist, cax=pos_cax)
cb.set_label('Normalised tern trajectory difference for SSP585 vs Historical, [2050-2099]/[1950-1999]', size=12)
plt.savefig(fh['fig'], dpi=300)