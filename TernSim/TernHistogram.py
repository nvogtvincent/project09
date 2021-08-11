#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script visualises tern trajectory data as a histogram with the average
year for a particular cell
@author: Noam Vogt-Vincent
"""

import numpy as np
import os
from parcels import (FieldSet, ParticleSet, JITParticle, ErrorCode, Variable)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime

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
         'scen_fn'       : 'UKESM1-0-LL_SSP585_TRAJ.nc'
         }

# DIRECTORIES
dirs  = {'script': os.path.dirname(os.path.realpath(__file__)),
         'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ_DATA/'}

# FILE HANDLES
fh = {'hist':  dirs['traj'] + param['model'] + '_HISTORICAL_TRAJ.nc',
      'scen':  dirs['traj'] + param['model'] + '_' + param['scenario'] + '_TRAJ.nc'}


### INSERT CODE HERE TO COMBINE THE TIME SERIES

##############################################################################
# PROCESS INPUTS                                                             #
##############################################################################


print('Processing inputs...')
print('')

with Dataset(fh['scen'], mode='r') as nc:
    data = {'lon' : np.array(nc.variables['lon'][:]),
            'lat' : np.array(nc.variables['lat'][:]),
            'time': np.array(nc.variables['time'][:], dtype=np.int64),}

# Convert the starting times to years
# 1. Set the starting year
param['s_year'] = '1850' if param['hist_avail'] else '2015'
param['s_year'] = np.datetime64(param['s_year'] + '-01-01')

# 2. Convert start times to start years
# Note: third line is stupid but necessary because of the awful numpy datetime
data['s_time'] = data['time'][:, 0]*np.timedelta64(1, 's')
data['s_time'] = data['s_time'] + param['s_year']
data['s_time'] = data['s_time'].astype('datetime64[Y]').astype(int) + 1970

