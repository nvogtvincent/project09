#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script visualises tern trajectory data as a histogram with the average
year for a particular cell
@author: Noam Vogt-Vincent
"""

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
import cartopy.crs as ccrs
import os
from netCDF4 import Dataset

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = {# Set these flags to True/False depending on whether inputs are used
         'hist_avail'    : False,
         'scen_avail'    : True,

         # Set these variables to the model/scenario names
         'model'         : 'UKESM1-0-LL',
         'scenario'      : 'SSP585',       # Ignored if scen_avail == False
         'hist_fn'       : '',
         'scen_fn'       : 'UKESM1-0-LL_SSP585_TRAJ.nc',

         # Plotting variables
         'res'           : 2, # (degrees)
         'bounds'        : {'lon_min' : -90,
                            'lon_max' : +65,
                            'lat_min' : -75,
                            'lat_max' : +70},
         'fig_fn'        : 'sample_output.png',

         'source_bnds'   : {'lon_min' : -50,
                            'lon_max' : -10,
                            'lat_min' : -70,
                            'lat_max' : -70},
         'sink_lat'      : 60.,
         'target_coords' : {'lon'     : -20,
                            'lat'     : 65}
         }

# DIRECTORIES
dirs  = {'script': os.path.dirname(os.path.realpath(__file__)),
         'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ_DATA/',
         'figs': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/'}

# FILE HANDLES
fh = {'hist': dirs['traj'] + param['model'] + '_HISTORICAL_TRAJ.nc',
      'scen': dirs['traj'] + param['model'] + '_' + param['scenario'] + '_TRAJ.nc',
      'fig' : dirs['figs'] + param['fig_fn']}


### INSERT CODE HERE TO COMBINE THE TIME SERIES

##############################################################################
# HISTOGRAM                                                                  #
##############################################################################

print('Creating histogram...')
print('')

# Load data
with Dataset(fh['scen'], mode='r') as nc:
    data = {'lon' : np.array(nc.variables['lon'][:]),
            'lat' : np.array(nc.variables['lat'][:]),
            'time': np.array(nc.variables['time'][:], dtype=np.int64),}

# Convert the starting times to years
# 1. Set the starting year
param['s_year'] = '1850' if param['hist_avail'] else '2015'
param['s_year'] = np.datetime64(param['s_year'] + '-01-01')

# 2. Convert time to years
# Note: third line is stupid but necessary because of the awful numpy datetime
data['s_time'] = data['time'][:, 0]*np.timedelta64(1, 's')
data['s_time'] = data['s_time'] + param['s_year']
data['s_time'] = data['s_time'].astype('datetime64[Y]').astype(int) + 1970

# Generate the binning grid
param['Llon'] = param['bounds']['lon_max'] - param['bounds']['lon_min']
param['Llat'] = param['bounds']['lat_max'] - param['bounds']['lat_min']

# Check that the lon/lat range is divisible by the resolution
if param['Llon']%param['res']:
    print('Longitude not divisible by resolution. Adjusting lon_max.')
    param['Llon'] = param['res']*np.ceil(param['Llon']/param['res'])
    param['bounds']['lon_max'] = param['bounds']['lon_min'] + param['Llon']

if param['Llat']%param['res']:
    print('Latitude not divisible by resolution. Adjusting lat_max.')
    param['Llat'] = param['res']*np.ceil(param['Llat']/param['res'])
    param['bounds']['lat_max'] = param['bounds']['lat_min'] + param['Llat']

lon = np.linspace(param['bounds']['lon_min'],
                  param['bounds']['lon_max'],
                  num=int(param['Llon']/param['res'])+1)
lat = np.linspace(param['bounds']['lat_min'],
                  param['bounds']['lat_max'],
                  num=int(param['Llat']/param['res'])+1)

# Transform the particle lat, lon and time into flattened arrays
data['s_time'] = np.repeat(data['s_time'][:, np.newaxis],
                           np.shape(data['lat'][1]),
                           axis=1)

data['s_time'] = data['s_time'].flatten()
data['lon'] = data['lon'].flatten()
data['lat'] = data['lat'].flatten()

hist_yn = np.histogram2d(x=data['lon'],
                         y=data['lat'],
                         bins=[lon, lat],
                         weights=data['s_time'])[0]
hist_yn = np.ma.masked_values(hist_yn, 0)

hist_n = np.histogram2d(x=data['lon'],
                         y=data['lat'],
                         bins=[lon, lat])[0]
hist_n = np.ma.masked_values(hist_n, 0)

hist = hist_yn/hist_n

##############################################################################
# MEAN TRAJECTORIES                                                          #
##############################################################################

print('Creating trajectories...')
print('')

# Calculate the mean longitude for each latitude bin, for each decade
# Results in a ND x NL x 3 array:
# ND: Number of decades
# NL: Number of latitude bins - 1
# 3 : Quartiles (25/50/75)
param['y_range'] = np.max(data['s_time']) - np.min(data['s_time'])
param['n_dec'] = int(np.ceil(param['y_range']/10))
param['first_dec'] = int(np.floor(np.min(data['s_time']/10))*10)
param['dec_list'] = np.arange(start=param['first_dec'],
                              stop=param['first_dec']+(param['n_dec']+1)*10,
                              step=10)
mean_traj = np.zeros([param['n_dec'], len(lat)-1, 3])

for deci in range(param['n_dec']):
    # Filter data for decade only
    dec_lon = data['lon'][((data['s_time'] >= param['dec_list'][deci])*
                           (data['s_time'] < param['dec_list'][deci+1]))]
    dec_lat = data['lat'][((data['s_time'] >= param['dec_list'][deci])*
                           (data['s_time'] < param['dec_list'][deci+1]))]
    for lati in range(len(lat)-1):
        # Filter longitudes by latitude
        lat_lon = dec_lon[(dec_lat >= lat[lati])*(dec_lat < lat[lati+1])]

        if not lat_lon.size:
            # Write a NaN if no trajectories for that latitude
            mean_traj[deci, lati, :] = np.nan
        else:
            mean_traj[deci, lati, 0] = np.quantile(lat_lon, 0.25)
            mean_traj[deci, lati, 1] = np.quantile(lat_lon, 0.55)
            mean_traj[deci, lati, 2] = np.quantile(lat_lon, 0.75)


##############################################################################
# PLOTTING                                                                   #
##############################################################################

# Plot the histogram

data_crs = ccrs.PlateCarree()
proj_crs = ccrs.Robinson(central_longitude=-20)

f1 = plt.figure(figsize=(18,9))
ax1 = plt.subplot(111, projection=proj_crs)
f1.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

ax1.set_global()
ax1.coastlines()
ax1.stock_img()

plotHist = ax1.pcolormesh(lon, lat, hist.T, cmap=cm.matter,
                          transform=data_crs)
start_line = ax1.plot(np.array([param['source_bnds']['lon_min'],
                                param['source_bnds']['lon_max']]),
                      np.array([param['source_bnds']['lat_min'],
                                param['source_bnds']['lat_max']]),
                      'k-', linewidth=2, transform=data_crs)

start_bnds = ax1.scatter(np.array([param['source_bnds']['lon_min'],
                                   param['source_bnds']['lon_max']]),
                         np.array([param['source_bnds']['lat_min'],
                                   param['source_bnds']['lat_max']]),
                         c='k', marker='.', s = 100,
                         transform=data_crs)

finish_line = ax1.plot(np.array([-180., 180.]),
                       np.array([param['sink_lat'], param['sink_lat']]),
                       'k-', linewidth=2, transform=data_crs)

target_pnt = ax1.scatter(param['target_coords']['lon'],
                         param['target_coords']['lat'],
                         c='r', marker='*', s=100,
                         transform=data_crs)

# Set up the colorbar
axpos = ax1.get_position()
pos_x = axpos.x0+axpos.width + 0.02
pos_y = axpos.y0
cax_width = 0.02
cax_height = axpos.height

pos_cax = f1.add_axes([pos_x, pos_y, cax_width, cax_height])

cb = plt.colorbar(plotHist, cax=pos_cax)
cb.set_label('Average year', size=12)
ax1.set_aspect('auto', adjustable=None)
ax1.margins(x=-0.01, y=-0.01)

# Save figure
plt.savefig(fh['fig'], dpi=300)

# Plot the line graph

f2 = plt.figure(figsize=(18,9))
ax2 = plt.subplot(111, projection=proj_crs)
f2.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

ax2.set_global()
ax2.coastlines()
ax2.stock_img()

for deci in range(param['n_dec']):
    ax2.plot(mean_traj[deci, :, 1],
             (lat[1:] + lat[:-1])/2,
             transform=data_crs)