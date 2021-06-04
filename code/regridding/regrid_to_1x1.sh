#!/bin/bash

cdo setctomiss,0 ${1} out.nc
bash /gws/pw/j05/cop26_hackathons/bristol/project09/regridding/regrid_cdo_ETOPO_GENERIC.sh out.nc
mv ETOPO_out.nc /gws/pw/j05/cop26_hackathons/bristol/project09/data/ETOPO_${2}.nc
rm out.nc
