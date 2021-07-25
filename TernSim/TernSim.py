#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles departing from a source location and flying with
an airspeed towards a destination location.
@author: Noam Vogt-Vincent
"""
import ternmethods as tm
import numpy as np
import os
import matplotlib.pyplot as plt
import cmocean.cm as cm
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar, Variable)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

dirs  = {'script': os.path.dirname(os.path.realpath(__file__)),
         'model': os.path.dirname(os.path.realpath(__file__)) + '/MODEL_DATA/',
         'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ_DATA/'}

param = {'model_name'        : 'UKESM1-0-LL',
         'scenarios'         : ['SSP245',
                                'SSP585'],

         'release_start_day' : 80,
         'release_end_day'   : 100,
         'number_of_releases': 3,
         'terns_per_release' : 5,
         'release_lat'       : -70.,
         'release_lon_range' : [-50., -10.],       # [min, max]
         'target_lat'        : 60.,
         'target_lon'        : -20.,
         'airspeed'          : 10.,                # (m s-1)
         'fly_frac'          : 0.6,                # Fraction of day in flight
         'mode'              : 'traj'   ,          # See notes below
         'parcels_dt'        : timedelta(hours=1), # Parcels solver dt
         'out_dt'            : timedelta(days=1),  # Only used if mode == traj
         'var_name'          : ['uas', 'vas'],     # [zonal, meridional]
         'coordinate_name'   : ['lon', 'lat'],     # [lon, lat]

         'debug'             : False,              # Toggle to skip simulations
         'first_sim'         : 1                   # Only used if debug == True
         }

# MODE NOTES:
# 'traj' : Full trajectory is recorded
# 'time' : Only time to reach destination is recorded
##############################################################################
# PROCESS INPUTS                                                             #
##############################################################################

print('Processing inputs...')
print('')

# FIND FILE NAMES
param['n_scen'] = len(param['scenarios'])

fh = {'u_hist' : (dirs['model'] + param['var_name'][0] + '_' +
                  param['model_name'] + '_HISTORICAL.nc'),
      'v_hist' : (dirs['model'] + param['var_name'][1] + '_' +
                  param['model_name'] + '_HISTORICAL.nc'),
      'u_scen' : [],
      'v_scen' : []}

for i in range(param['n_scen']):
    fh['u_scen'].append(dirs['model'] + param['var_name'][0] + '_' +
                        param['model_name'] + '_' + param['scenarios'][i] +
                        '.nc')
    fh['v_scen'].append(dirs['model'] + param['var_name'][1] + '_' +
                        param['model_name'] + '_' + param['scenarios'][i] +
                        '.nc')

fh['traj_hist'] = dirs['traj'] + param['model_name'] + '_HISTORICAL_TRAJ.nc'
fh['traj_scen'] = []

for i in range(param['n_scen']):
    fh['traj_scen'].append(dirs['model'] + param['model_name'] + '_' +
                           param['scenarios'][i] + '_TRAJ.nc')


# GENERATE INITIAL POSITIONS
release = {'lon' : {'basis': np.linspace(param['release_lon_range'][0],
                                         param['release_lon_range'][1],
                                         num = param['terns_per_release'])},
           'lat' : {'basis': np.linspace(param['release_lat'],
                                         param['release_lat'],
                                         num = param['terns_per_release'])},}

# FIND START AND END TIMES
with Dataset(fh['u_hist'], mode='r') as nc:
    param['calendar'] = nc.variables['time'].calendar

    param['Ystart'] = {'hist' : num2date(nc.variables['time'][0],
                                         nc.variables['time'].units,
                                         calendar=param['calendar']).year}

    param['Yend']   = {'hist' : num2date(nc.variables['time'][-1],
                                         nc.variables['time'].units,
                                         calendar=param['calendar']).year}

    param['time_offset'] = (num2date(nc.variables['time'][0],
                                     nc.variables['time'].units) -
                            datetime(year  = param['Ystart']['hist'],
                                     month = 1,
                                     day   = 1)).total_seconds()

    param['run_time'] = {'hist' : (param['Yend']['hist'] -
                                   param['Ystart']['hist'] + 1)}
    param['run_time']['hist'] *= 3600*24*360
    param['run_time']['hist'] -= 4*param['time_offset']
    param['run_time']['hist'] = timedelta(seconds=param['run_time']['hist'])

with Dataset(fh['u_scen'][0], mode='r') as nc:
    param['Ystart']['scen'] = num2date(nc.variables['time'][0],
                                       nc.variables['time'].units,
                                       calendar=param['calendar']).year

    param['Yend']['scen'] = num2date(nc.variables['time'][-1],
                                       nc.variables['time'].units,
                                       calendar=param['calendar']).year

    param['run_time']['scen'] = (param['Yend']['scen'] -
                                 param['Ystart']['scen'] + 1)
    param['run_time']['scen'] *= 3600*24*360
    param['run_time']['scen'] -= 4*param['time_offset']
    param['run_time']['scen'] = timedelta(seconds=param['run_time']['scen'])

release = tm.prepare_release(release, param)

# GENERATE THE FLYING VECTOR FIELD
fh['fly_field'] = dirs['model'] + 'fly_field.nc'
fly_field = tm.genTargetField(param['target_lon'],
                              param['target_lat'],
                              param['airspeed'],
                              fh)

##############################################################################
# TERN KERNELS                                                               #
##############################################################################

class arcticTern(JITParticle):
    flight_time = Variable('flight_time', dtype=np.float32, initial=0.)
    release_time = Variable('release_time', dtype=np.int32, initial=0.)
    time_of_day = Variable('time_of_day', dtype=np.float32, initial=0.,
                           to_write=False)

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    particle.delete()

def fly(particle, fieldset, time):
    # Find the year at the start
    if particle.flight_time == 0.:
        particle.release_time = time

    # Remove the tern if they reach the end latitude
    if particle.lat > fieldset.target_lat:
        particle.delete()

    if particle.time_of_day > fieldset.night_time:
        # Assume birds are static for a certain proportion of the day
        ustatic = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        vstatic = fieldset.V[time, particle.depth, particle.lat, particle.lon]
        particle.lon -= ustatic * particle.dt
        particle.lat -= vstatic * particle.dt

    # Update the particle age
    particle.flight_time += particle.dt
    particle.time_of_day += particle.dt
    if particle.time_of_day >= 86400.:
        particle.time_of_day = 0

##############################################################################
# RELEASE THE TERNS                                                          #
##############################################################################

if not param['debug']:
    param['first_sim'] = 0  # Set first simulation to default of debug = False

for i in range(param['first_sim'], param['n_scen'] + 1):
    # Create fieldset
    if i == 0:
        print('Setting up historical simulation...')
        filenames = {'U': fh['u_hist'],
                     'V': fh['v_hist']}
    else:
        print('Setting up ' + param['scenarios'][i-1] + ' simulation...')
        filenames = {'U': fh['u_scen'][i-1],
                     'V': fh['v_scen'][i-1]}

    print('')

    variables = {'U': param['var_name'][0],
                 'V': param['var_name'][1]}

    dimensions = {'U': {'lon': param['coordinate_name'][0],
                        'lat': param['coordinate_name'][1],
                        'time': 'time'},
                  'V': {'lon': param['coordinate_name'][0],
                        'lat': param['coordinate_name'][1],
                        'time': 'time'}}

    wind_fieldset = FieldSet.from_netcdf(filenames,
                                         variables,
                                         dimensions)

    flight_fieldset = FieldSet.from_data(data = {'U': fly_field['u'],
                                                 'V': fly_field['v']},
                                         dimensions = {'lon': fly_field['lon'],
                                                       'lat': fly_field['lat']})

    fieldset = FieldSet(U=wind_fieldset.U+flight_fieldset.U,
                        V=wind_fieldset.V+flight_fieldset.V)

    # Add constants
    fieldset.add_constant('target_lat', param['target_lat'])
    fieldset.add_constant('night_time', 86400.*param['fly_frac'])

    # Create particleset
    if i == 0:
        pset = ParticleSet.from_list(fieldset=fieldset,
                                     pclass=arcticTern,
                                     lon = release['lon']['hist'],
                                     lat = release['lat']['hist'],
                                     time = release['time']['hist'])
        traj_file = fh['traj_hist']
    else:
        pset = ParticleSet.from_list(fieldset=fieldset,
                                     pclass=arcticTern,
                                     lon = release['lon']['scen'],
                                     lat = release['lat']['scen'],
                                     time = release['time']['scen'])
        traj_file = fh['traj_scen'][i-1]

    if param['mode'] == 'traj':
        traj_file = pset.ParticleFile(name=traj_file,
                                      outputdt=param['out_dt'],
                                      write_ondelete=False)

    elif param['mode'] == 'time':
        traj_file = pset.ParticleFile(name=traj_file,
                                      write_ondelete=True)
    else:
        raise NotImplementedError('Mode not understood!')

    # Run the simulation
    print('Set-up complete!')
    print('Releasing the birds!')
    if i == 0:
        pset.execute((pset.Kernel(AdvectionRK4) +
                      pset.Kernel(fly)),
                     runtime=param['run_time']['hist'],
                     dt = param['parcels_dt'],
                     recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                     output_file=traj_file)
    else:
        pset.execute((pset.Kernel(AdvectionRK4) +
                      pset.Kernel(fly)),
                     runtime=param['run_time']['scen'],
                     dt = param['parcels_dt'],
                     recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                     output_file=traj_file)

    print('')
    print('Simulation complete!')
    print('Exporting netcdf...')
    traj_file.export()

    print('Export complete!')
    print('')

    if i < param['n_scen']:
        print('Moving to the next simulation...')
        print('')
    else:
        print('Simulations complete!')
        print('The terns will miss you!')

