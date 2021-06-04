#!/bin/bash

D_GRID=/gws/pw/j05/cop26_hackathons/bristol/project09/regridding/etopo60grid
cdo remapnn,${D_GRID} ${1} outfile.nc
mv outfile.nc ETOPO_${1}

