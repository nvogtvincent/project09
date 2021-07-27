#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for TernSim
@author: Noam Vogt-Vincent
"""

import numpy as np
from geographiclib.geodesic import Geodesic
from netCDF4 import Dataset
import os


def prepare_release(release, param):
    if param['calendar'] != '360_day':
        raise NotImplementedError('This calendar is not 360_day!')

    # HISTORICAL RUN
    # Firstly generate the times in a year
    release_times = np.linspace(param['release_start_day'],
                                param['release_end_day'],
                                num = param['number_of_releases'])
    release_times *= 24*3600    # Convert to seconds

    # Now generate year start times
    release['time'] = {'hist': np.arange(param['Ystart']['hist'],
                                         param['Yend']['hist'] + 1,
                                         1)}
    release['time']['hist'] -= param['Ystart']['hist']
    release['time']['hist'] *= 24*3600*360  # Convert to seconds

    # Now combine
    release['time']['hist'] = (np.tile(release_times,
                                       reps=len(release['time']['hist'])) +
                               np.repeat(release['time']['hist'],
                                         repeats=len(release_times)))

    # Now add the offset (since parcels makes t=0 = first frame, not 00:00)
    release['time']['hist'] -= param['time_offset']

    # Now multiply by the number of particles per release
    release['lon']['hist'] = np.tile(release['lon']['basis'],
                                     reps=len(release['time']['hist']))
    release['lat']['hist'] = np.tile(release['lat']['basis'],
                                     reps=len(release['time']['hist']))
    release['time']['hist'] = np.repeat(release['time']['hist'],
                                        repeats=param['terns_per_release'])

    # SCENARIO RUNS
    # Now generate year start times
    release['time']['scen'] = np.arange(param['Ystart']['scen'],
                                        param['Yend']['scen'] + 1,
                                        1)
    release['time']['scen'] -= param['Ystart']['scen']
    release['time']['scen'] *= 24*3600*360  # Convert to seconds

    # Now combine
    release['time']['scen'] = (np.tile(release_times,
                                       reps=len(release['time']['scen'])) +
                               np.repeat(release['time']['scen'],
                                         repeats=len(release_times)))

    # Now add the offset (since parcels makes t=0 = first frame, not 00:00)
    release['time']['scen'] -= param['time_offset']

    # Now multiply by the number of particles per release
    release['lon']['scen'] = np.tile(release['lon']['basis'],
                                     reps=len(release['time']['scen']))
    release['lat']['scen'] = np.tile(release['lat']['basis'],
                                     reps=len(release['time']['scen']))
    release['time']['scen'] = np.repeat(release['time']['scen'],
                                        repeats=param['terns_per_release'])

    return release


def genTargetField(end_lon, end_lat, speed, fh):
    # Check if field already exists
    if os.path.isfile(fh['fly_field']):
        print('Tern velocity file found, using these values.')
        print('')
        with Dataset(fh['fly_field'], mode='r') as nc:
            fly_field = {'u' : np.array(nc.variables['u'][:]),
                         'v' : np.array(nc.variables['v'][:]),
                         'lon' : np.array(nc.variables['lon'][:]),
                         'lat' : np.array(nc.variables['lat'][:])}
    else:
        print('Calculating tern velocity vectors...')
        target_lon = np.linspace(-179.5, 179.5, num=360)
        target_lat = np.linspace(-89.5, 89.5, num=180)
        target_lon, target_lat = np.meshgrid(target_lon, target_lat)
        u = np.zeros_like(target_lon)
        v = np.zeros_like(target_lon)

        for i in range(180):
            for j in range(360):
                bearing = Geodesic.WGS84.Inverse(target_lat[i, j],
                                                 target_lon[i, j],
                                                 end_lat,
                                                 end_lon)['azi1']*np.pi/180
                if target_lat[i, j] < 0:
                    if bearing > np.pi/4:
                        bearing = np.pi/4
                    elif bearing < -np.pi/4:
                        bearing = -np.pi/4

                u[i, j] = speed*np.sin(bearing)
                v[i, j] = speed*np.cos(bearing)

        fly_field = {'u' : u,
                     'v' : v,
                     'lon' : target_lon[0, :],
                     'lat' : target_lat[:, 0]}

        print('Complete.')
        print('Exporting to netcdf...')
        with Dataset(fh['fly_field'], mode='w') as nc:
            # Create the dimensions
            nc.createDimension('lon', len(fly_field['lon']))
            nc.createDimension('lat', len(fly_field['lat']))

            nc.createVariable('lon', 'f4', ('lon'), zlib=True)
            nc.variables['lon'].long_name = 'longitude'
            nc.variables['lon'].units = 'degrees_east'
            nc.variables['lon'].standard_name = 'longitude'
            nc.variables['lon'][:] = fly_field['lon']

            nc.createVariable('lat', 'f4', ('lat'), zlib=True)
            nc.variables['lat'].long_name = 'latitude'
            nc.variables['lat'].units = 'degrees_north'
            nc.variables['lat'].standard_name = 'latitude'
            nc.variables['lat'][:] = fly_field['lat']

            nc.createVariable('u', 'f4', ('lat', 'lon'), zlib=True)
            nc.variables['u'].long_name = 'u_flight_velocity'
            nc.variables['u'].units = 'm s-1'
            nc.variables['u'].standard_name = 'u_vel'
            nc.variables['u'][:] = fly_field['u']

            nc.createVariable('v', 'f4', ('lat', 'lon'), zlib=True)
            nc.variables['v'].long_name = 'v_flight_velocity'
            nc.variables['v'].units = 'm s-1'
            nc.variables['v'].standard_name = 'v_vel'
            nc.variables['v'][:] = fly_field['v']

            nc.target_lon = str(target_lon)
            nc.target_lat = str(target_lat)
            nc.speed      = str(speed) + ' m s-1'
        print('')

    return fly_field
