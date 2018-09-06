from stompy.model.suntans import sun_driver

import six
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy.spatial import proj_utils, field, wkb2shp
from stompy.model.delft import dfm_grid
import stompy.model.delft.io as dio
from stompy.model import otps
from stompy import utils

from stompy.grid import unstructured_grid

from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv
##
six.moves.reload_module(dfm)
six.moves.reload_module(dfm_grid)
six.moves.reload_module(drv)

##

read_otps.OTPS_DATA='../derived'

use_temp=False
use_salt=True

# sun001: 25 layers, 1.05 stretch, stairstep
# sun002: 1.08 stretch, no stairstep, no mpi
# sun003: zero salt/temp, and then static HYCOM
run_dir='runs/sun003'
run_start=np.datetime64("2017-06-01")
run_stop =np.datetime64("2017-06-03")

model=drv.SuntansModel()
model.projection="EPSG:26910"
model.num_procs=1
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.load_template("sun-template.dat")

model.set_run_dir(run_dir,mode='pristine')
model.config['Nkmax']=25
# would like to relax this ASAP
model.config['stairstep']=0

dt_secs=60 
model.config['dt']=dt_secs
# quarter-hour map output:
model.config['ntout']=int(15*60/dt_secs)
# daily restart file
model.config['ntoutStore']=int(86400/dt_secs)
model.config['mergeArrays']=0
model.config['rstretch']=1.08

# these are scaled down by 1e-3 for debugging
if use_temp:
    model.config['gamma']=0.00021
else:
    model.config['gamma']=0.0

if use_salt:
    model.config['beta']=0.00077
else:
    model.config['beta']=0.0

model.run_start=run_start
model.run_stop=run_stop

# dest_grid="../derived/matched_grid_v01.nc"
dest_grid="../ragged/grid_v01.nc"
assert os.path.exists(dest_grid),"Grid %s not found"%dest_grid
model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid))

model.add_gazetteer("linear_features.shp")

# spatially varying
ocean_bc=drv.MultiBC(drv.OTPSStageBC,name='Ocean',otps_model='wc')
model.add_bcs(ocean_bc)

model.write()

#----------
# HYCOM
utils.path("../")
from sfb_dfm_utils import ca_roms, coamps, hycom
hycom_lon_range=[-124.7, -121.7 ]
# hycom_lon_range=[-124.7+360, -121.7+360]
hycom_lat_range=[36.2, 38.85]
coastal_pad=0 # np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]
coastal_files=hycom.fetch_range(hycom_lon_range,hycom_lat_range,coastal_time_range)

# For starters, set IC and BC to a static hycom field.
hycom_ds=xr.open_dataset(coastal_files[1])

# -----
# map each cell to a hycom
cc=model.grid.cells_center()
cc_ll=model.native_to_ll(cc)

dlat=np.median(np.diff(hycom_ds.lat.values))
dlon=np.median(np.diff(hycom_ds.lon.values))
lat_i = utils.nearest(hycom_ds.lat.values,cc_ll[:,1],max_dx=1.2*dlat)
lon_i = utils.nearest(hycom_ds.lon.values,cc_ll[:,0],max_dx=1.2*dlon)

# make this positive:down to match hycom and make the interpolation
sun_z = -model.ic_ds.z_r.values

default_s=33.4 # would be nice to pull a nominal shallow value from HYCOM
assert ('time', 'Nk', 'Nc') == model.ic_ds.salt.dims,"Workaround is fragile"

for c in range(model.grid.Ncells()):
    sun_s=default_s
    if lat_i[c]<0 or lon_i[c]<0:
        print("Cell %d does not overlap HYCOM grid"%c)
    else:
        # top to bottom, depth positive:down
        s_profile=hycom_ds.salinity.isel(lon=lon_i[c],lat=lat_i[c])
        s_profile=s_profile.values
        valid=np.isfinite(s_profile)
        if not np.any(valid):
            print("Cell %d is dry in HYCOM grid"%c)
        else:
            # could add bottom salinity if we really cared.
            sun_s=np.interp( sun_z,
                             hycom_ds.depth.values[valid], s_profile[valid] )
            # if Nk wasn't broken:
            #model.ic_ds.salt.isel(time=0,Nc=c).values[:]=sun_s
    model.ic_ds.salt.values[0,:,c]=sun_s

model.write_ic_ds()

#----

# Copy IC values to the boundary conditions for testing
for ci,c in enumerate(utils.progress(model.bc_ds.cellp.values)):
    ic_salt = model.ic_ds.salt.values[0,:,c]
    for ti in range(model.bc_ds.dims['Nt']):
        model.bc_ds.S.isel(Ntype3=ci,Nt=ti).values[:]=ic_salt
model.write_bc_ds()

##---
model.partition()

model.run_simulation()

