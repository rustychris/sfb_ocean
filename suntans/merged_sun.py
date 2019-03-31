"""
Combining ocean_sun and sfbay_sun for the merged grid.
"""
import six
import shutil
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
try:
    cache_dir=os.path.join(os.path.dirname(__file__),'cache')
except NameError:
    cache_dir='cache'

from stompy.grid import unstructured_grid

from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv

##
six.moves.reload_module(dfm)
six.moves.reload_module(drv)

## 
# rough command line interface
import argparse
parser = argparse.ArgumentParser(description='Set up and run SF Bay SUNTANS simulations.')
parser.add_argument("-s", "--start", help="Date of simulation start",
                    default="2017-06-15")
parser.add_argument("-e", "--end", help="Date of simulation stop",
                    default="2017-06-22")
parser.add_argument("-d", "--dir", help="Run directory",
                    default="runs/bay007")
parser.add_argument("-r", "--resume", help="Resume from run",
                    default=None)
parser.add_argument("-g","--write-grid", help="Write grid to ugrid")

if __name__=='__main__':
    args=parser.parse_args()
else:
    # For manually running the script.
    # args=parser.parse_args(["-g","grid-connectivity.nc"])
    raise Exception("Update args")


##

read_otps.OTPS_DATA='../derived'

use_temp=True
use_salt=True
ocean_method='hycom'
grid_dir="grid-merged"
# moving to have all of the elevation offset stuff more manual, less
# magic
z_offset_manual=-5
drv.SuntansModel.sun_bin_dir="/home/rusty/src/suntans/main"
# AVOID anaconda mpi (at least if suntans is compiled with system mpi)
drv.SuntansModel.mpi_bin_dir="/usr/bin/"

# merge_001: initial trial of combined grid

if not args.resume:
    # HYCOM experiments change just before 2017-06-15
    model=drv.SuntansModel()
    model.num_procs=16
    model.load_template("sun-template.dat")

    model.config['Nkmax']=35
    model.config['stairstep']=0
    model.dredge_depth=-2 + z_offset_manual # 2m below the offset of -5m.

    dt_secs=5.0
    model.config['dt']=dt_secs
    # 30 minute map output:
    model.config['ntout']=int(30*60/dt_secs)
    # model.config['ntout']=1 # for debugging
    model.config['ntoutStore']=int(86400/dt_secs) # daily
    model.config['mergeArrays']=0
    model.config['rstretch']=1.1
    model.config['Cmax']=30.0 # volumetric is a better test, this is more a backup.

    # esp. with edge depths, seems better to use z0B so that very shallow
    # edges can have the right drag.
    model.config['CdB']=0
    model.config['z0B']=0.001

    model.use_edge_depths=True

    if use_temp:
        model.config['gamma']=0.00021
    else:
        model.config['gamma']=0.0

    if use_salt:
        model.config['beta']=0.00077
    else:
        model.config['beta']=0.0

    model.run_start=np.datetime64(args.start)

    # This grid comes in with NAVD88 elevation
    dest_grid=os.path.join(grid_dir,"spliced_grids_01_bathy.nc")
    grid=unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid)
    # which are modified before giving to the model
    grid.cells['depth'] += z_offset_manual
    grid.edges['edge_depth'] += z_offset_manual
    model.set_grid(grid)

    # make sure edge depths actually were included
    edge_depths=model.grid.edges['edge_depth']
    assert edge_depths.min()<0.0,"Looks like edge depths were not set on %s"%dest_grid
else:
    old_model=drv.SuntansModel.load(args.resume)
    model=old_model.create_restart(symlink=True)
    model.dredge_depth=None # no need to dredge grid if a restart

    edge_depths=model.grid.edges['edge_depth']
    assert edge_depths.min()<0.0,"Looks like edge depths were not found in restart"

# common to restart and initial run:
model.projection="EPSG:26910"
model.run_stop=np.datetime64(args.end)

# run_dir='/opt/sfb_ocean/suntans/runs/merge_001-20170601'
model.set_run_dir(args.dir,mode='pristine')
    
model.add_gazetteer(os.path.join(grid_dir,"linear_features.shp"))
model.add_gazetteer(os.path.join(grid_dir,"point_features.shp"))

dt_secs=float(model.config['dt'])

if args.resume is None:
    # Setup profile output -- should move into sun_driver.py once working
    model.config['ProfileVariables']='husT'
    model.config['ntoutProfs']=int(900/dt_secs) # 15 minute data
    model.config['numInterpPoints']=1
    model.config['DataLocations']='profile_locs.dat'
    model.config['NkmaxProfs']=0 # all layers
    model.config['ProfileDataFile']="profdata.dat"

    mon_points=model.match_gazetteer(type='monitor')
    xys=[ np.array(feat['geom']) for feat in mon_points]

    valid_xys=[xy for xy in xys if model.grid.select_cells_nearest(xy,inside=True) is not None]
    np.savetxt( os.path.join(model.run_dir,model.config['DataLocations']),
                np.array(valid_xys) )

## 

# spatially varying
hycom_ll_box=[-124.9, -121.7, 35.9, 39.0]

if ocean_method=='eta':
    ocean_bc=drv.MultiBC(drv.OTPSStageBC,name='Ocean',otps_model='wc')
    ocean_offset_bc=drv.StageBC(name='Ocean',mode='add',z=z_offset_manual)
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

ocean_salt_bc=drv.HycomMultiScalarBC(name='Ocean',scalar='salinity',cache_dir=cache_dir,ll_box=hycom_ll_box)
ocean_temp_bc=drv.HycomMultiScalarBC(name='Ocean',scalar='temperature',cache_dir=cache_dir,ll_box=hycom_ll_box,)
model.add_bcs([ocean_salt_bc,ocean_temp_bc]) 

import sfb_common
sfb_common.add_delta_bcs(model,cache_dir)
sfb_common.add_usgs_stream_bcs(model,cache_dir)  # disable if no internet
sfb_common.add_potw_bcs(model,cache_dir)

import coamps_sfei_wind
print("Adding WIND")
coamps_sfei_wind.add_wind(model,cache_dir)

##

def set_ic(model):
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

    if 1: # set freesurface
        model.ic_ds.eta.isel(time=0).values[...]=z_offset_manual

if __name__=='__main__':
    if args.write_grid:
        model.grid.write_ugrid(args.write_grid,overwrite=True)
    else:
        print("Num procs A: ",model.num_procs)
        model.write()
        try:
            script=__file__
        except NameError:
            print("__file__ not defined.  cannot copy script")
            script=None 
        if script:
            shutil.copy(script,model.run_dir)
           
        if args.resume is None:
            set_ic(model)
            model.write_ic_ds()
        print("Num procs B: ",model.num_procs)
        model.partition()
        print("Num procs C: ",model.num_procs)
        model.run_simulation()

