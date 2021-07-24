#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for TernSim
@author: Noam Vogt-Vincent
"""

import numpy as np
from datetime import timedelta, datetime
from geographiclib.geodesic import Geodesic

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


def genTargetField(end_lon, end_lat, speed):
    print('Calculating tern velocity vectors...')
    target_lon = np.linspace(0.5, 359.5, num=360)
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
                 'lon' : target_lon,
                 'lat' : target_lat}

    print('Complete.')
    print('')

    return fly_field





# def starting_time(pos_array, start_day, nyears):
#     npart = np.shape(pos_array)[0]
#     pos = np.zeros((npart * nyears, 3))

#     for year in range(nyears):
#         day = (360.*year) + start_day
#         sec = day*86400.
#         pos[year*npart:(year+1)*npart, :2] = pos_array
#         pos[year*npart:(year+1)*npart, 2] = sec

#     return pos