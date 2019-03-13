"""
Combining ocean_sun and sfbay_sun for the merged grid.
"""
import six
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from stompy.model import otps
from stompy import utils, filters

import logging as log

utils.path("../")
from stompy.io.local import hycom
cache_dir='cache'

from stompy.grid import unstructured_grid

from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv

##
six.moves.reload_module(dfm)
six.moves.reload_module(drv)

read_otps.OTPS_DATA='../derived'

use_temp=True
use_salt=True
ocean_method='hycom'

# merge_001: initial trial of combined grid
run_dir='/opt/sfb_ocean/suntans/runs/merge_001'
run_start=np.datetime64("2017-06-15 00:00:00")
run_stop =np.datetime64("2017-06-16 06:00:00")

model=drv.SuntansModel()
model.projection="EPSG:26910"
model.num_procs=4
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.load_template("sun-template.dat")

model.set_run_dir(run_dir,mode='askclobber')
model.config['Nkmax']=35
model.config['stairstep']=0

dt_secs=5.0
model.config['dt']=dt_secs
# 5 minute map output:
model.config['ntout']=int(5*60/dt_secs)
# model.config['ntout']=1 # for debugging
model.config['ntoutStore']=int(86400/dt_secs) # daily
model.config['mergeArrays']=0
model.config['rstretch']=1.1
model.config['Cmax']=30.0 # volumetric is a better test, this is more a backup.

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

grid_dir="grid-merged"
dest_grid=os.path.join(grid_dir,"spliced_grids_01_bathy.nc")
model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid))
model.add_gazetteer(os.path.join(grid_dir,"linear_features.shp"))

## 

# spatially varying
hycom_ll_box=[-124.9, -121.7, 35.9, 39.0]

ocean_salt_bc=drv.HycomMultiScalarBC(name='Ocean',scalar='salinity',cache_dir=cache_dir,ll_box=hycom_ll_box)
ocean_temp_bc=drv.HycomMultiScalarBC(name='Ocean',scalar='temperature',cache_dir=cache_dir,ll_box=hycom_ll_box,)
model.add_bcs([ocean_salt_bc,ocean_temp_bc]) 

if ocean_method=='eta':
    ocean_bc=drv.MultiBC(drv.OTPSStageBC,name='Ocean',otps_model='wc')
elif ocean_method=='flux':
    ocean_bc=drv.MultiBC(drv.OTPSFlowBC,name='Ocean',otps_model='wc')
elif ocean_method=='velocity':
    ocean_bc=drv.MultiBC(drv.OTPSVelocityBC,name='Ocean',otps_model='wc')
elif ocean_method=='hycom':
    # explicity give bounds to make sure we always download the same
    # subset.
    ocean_bc=drv.HycomMultiVelocityBC(ll_box=hycom_ll_box,
                                      name='Ocean',cache_dir=cache_dir)
    model.add_bcs(drv.MultiBC(drv.OTPSVelocityBC,name='Ocean',otps_model='wc',mode='add'))

model.add_bcs(ocean_bc)

model.write()

##

# Initial condition

# Start with USGS Polaris 
if 1: 
    import polaris_ic
    polaris_ic.set_ic_from_usgs_sfbay(model,
                                      scalar='salt',
                                      usgs_scalar='Salinity',
                                      ocean_surf=34.0,ocean_grad=0.0,
                                      clip=[0,34],
                                      cache_dir=cache_dir)
    polaris_ic.set_ic_from_usgs_sfbay(model,
                                      scalar='temp',
                                      usgs_scalar='Temperature',
                                      ocean_grad=0.0,
                                      clip=[5,30],
                                      cache_dir=cache_dir)

if 1: # Include HYCOM initial condition
    import hycom_ic
    hycom_ic.set_ic_from_hycom(model,hycom_ll_box,cache_dir,default_s=None,default_T=None)

model.write_ic_ds()
model.partition()
model.run_simulation()

