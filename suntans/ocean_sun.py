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

# If true, check for newer grid, and regenerate grid with bathy
# on demand.
update_grid=True

use_temp=False
use_salt=False

run_dir='runs/sun001'
run_start=np.datetime64("2017-06-01")
run_stop =np.datetime64("2017-06-04")

model=drv.SuntansModel()
model.projection="EPSG:26910"
model.num_procs=4
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.load_template("sun-template.dat")

model.set_run_dir(run_dir,mode='pristine')
model.config['Nkmax']=25
# would like to relax this ASAP
model.config['stairstep']=1

dt_secs=30
model.config['dt']=dt_secs
# quarter-hour map output:
model.config['ntout']=int(15*60/dt_secs)
# daily restart file
model.config['ntoutStore']=int(86400/dt_secs)
model.config['mergeArrays']=0
model.config['rstretch']=1.05

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

model.partition()

model.run_simulation()


