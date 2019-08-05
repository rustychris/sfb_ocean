"""
Testing just in the LSB sub grid
"""

from stompy.model.suntans import sun_driver

import shutil
import six
import sys
import os
import datetime

import numpy as np
import xarray as xr
from stompy.spatial import proj_utils, field, wkb2shp
from stompy.model.delft import dfm_grid
import stompy.model.delft.io as dio
from stompy.model import otps
from stompy import utils, filters

import logging as log

from stompy.io.local import hycom
try:
    here=os.path.dirname(__file__)
except NameError:
    here="."
cache_dir=os.path.join(here,'cache')
os.path.exists(cache_dir) or os.makedirs(cache_dir)

from stompy.grid import unstructured_grid

from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv

##
def set_bathy(g_in,g_out):
    from stompy.grid import depth_connectivity
    import bathy
    
    assert g_in!=g_out
    shallow_thresh=-1
    g=unstructured_grid.UnstructuredGrid.from_ugrid(g_in)
    dem=bathy.dem()
    z_cell_mean=depth_connectivity.cell_mean_depth(g,dem)

    e2c=g.edge_to_cells().copy()
    nc1=e2c[:,0]
    nc2=e2c[:,1]
    nc1[nc1<0]=nc2[nc1<0]
    nc2[nc2<0]=nc1[nc2<0]
    # starting point for edges is shallower of the neighboring cells
    z_edge=np.maximum(z_cell_mean[nc1],z_cell_mean[nc2])
    # only worry about connectivity when the edge is starting above
    # the threshold
    shallow=(z_edge>shallow_thresh)
    # centers='centroid' seemed to be losing a lot of connectivity.
    z_edge_conn=depth_connectivity.edge_connection_depth(g,dem,
                                                         edge_mask=shallow,
                                                         centers='lowest')
    valid=np.isfinite(z_edge_conn)
    z_edge[valid]=z_edge_conn[valid]
    # edge-based is better at getting the unresolved channels connected
    # leads to alligator teeth in some places.
    # only use edge connectivity approach down to edge_thresh
    z_cell_edgeminthresh=[ min(max(shallow_thresh,
                                   z_edge[ g.cell_to_edges(c) ].min()),
                               z_cell_mean[c])
                           for c in range(g.Ncells()) ]
    g.add_cell_field('z_bed',np.asarray(z_cell_edgeminthresh),
                     on_exists='overwrite')
    rough='z0B'
    if rough in g.edges.dtype.names:
        missing=g.edges[rough]==0
        g.edges[rough][missing]=0.002

        
    ec=g.edge_to_cells().copy()
    nc1=ec[:,0]
    nc2=ec[:,1]
    nc1[nc1<0]=nc2[nc1<0] ; nc2[nc2<0]=nc1[nc2<0]
    edge_z=np.maximum( g.cells['z_bed'][nc1],
                       g.cells['z_bed'][nc2] )
    g.add_edge_field('edge_z_bed',edge_z,on_exists='overwrite')
        
    g.write_ugrid(g_out,overwrite=True)
    return g
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
    args=parser.parse_args(["-g","grid-connectivity.nc"])

use_temp=True
use_salt=True

run_dir=args.dir # '/opt/sfb_ocean/suntans/runs/bay003'

if not args.resume:
    model=drv.SuntansModel()
    model.run_start=np.datetime64(args.start)
    model.num_procs=4
    model.use_edge_depths=True
    model.load_template("sun-template.dat")
    # this won't transfer to the full domain... :-(
    model.config['Nkmax']=11 # with 11, the first layer here is 0.437, compared to 0.438 to in the real deal.
    model.config['stairstep']=1
    dt_secs=20.0
    model.config['dt']=dt_secs
    model.config['ntout']=int(1*3600/dt_secs) # hourly map output
    model.config['ntoutStore']=int(86400/dt_secs) # # daily restart file
    model.config['mergeArrays']=0
    model.config['rstretch']=1.125
    model.config['Cmax']=30.0 # volumetric is a better test, this is more a backup.
    # esp. with edge depths, seems better to use z0B so that very shallow
    # edges can have the right drag.
    model.config['CdB']=0
    model.config['z0B']=0.002
    if use_temp:
        model.config['gamma']=0.00021
    else:
        model.config['gamma']=0.0

    if use_salt:
        model.config['beta']=0.00077
    else:
        model.config['beta']=0.0

else:
    old_model=drv.SuntansModel.load(args.resume)
    model=old_model.create_restart(symlink=True)

model.run_stop=np.datetime64(args.end)

model.projection="EPSG:26910"
model.sun_bin_dir="/home/rusty/src/suntans/main"

model.set_run_dir(run_dir,mode='pristine')

dt_secs=float(model.config['dt'])

z_offset_manual=-5 # m

model.z_offset=0
model.dredge_depth=-2 + z_offset_manual # 2m below the offset of -5m.

if args.resume is None:
    src_grid="grid-lsb/lsb_subgrid-edit22.nc"
    dest_grid=src_grid.replace(".nc","-bathy.nc")
    assert os.path.exists(src_grid),"Grid %s not found"%src_grid
    assert dest_grid != src_grid

    if utils.is_stale(dest_grid,[src_grid,"bathy.py"]):
        set_bathy(src_grid,dest_grid)

    g=unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid)
    g.cells['z_bed'] += z_offset_manual
    g.edges['edge_z_bed'] += z_offset_manual
    model.set_grid(g)
else:
    print("Grid comes from restart")
    model.dredge_depth=None # no need to dredge grid if a restart
    
model.add_gazetteer("grid-lsb/linear_features.shp")

##

ocean_salt_bc=drv.ScalarBC(name='ocean',scalar='salinity',value=25)
ocean_temp_bc=drv.ScalarBC(name='ocean',scalar='temperature',value=10)
msl_navd88=0.94 # m

ocean_bc=drv.NOAAStageBC(name='ocean',
                         station=9414575, # Coyote - will come in as MSL
                         # station=9414750, # Alameda.
                         cache_dir=cache_dir,
                         filters=[dfm.Lowpass(cutoff_hours=1.5)])
ocean_offset_bc=drv.StageBC(name='ocean',mode='add',z=z_offset_manual + msl_navd88)

model.add_bcs([ocean_bc,ocean_salt_bc,ocean_temp_bc,ocean_offset_bc]) 

import sfb_common
# sfb_common.add_usgs_stream_bcs(model,cache_dir)  # disable if no internet
# sfb_common.add_potw_bcs(model,cache_dir)

if __name__=='__main__':
    if args.write_grid:
        model.grid.write_ugrid(args.write_grid,overwrite=True)
    else:
        model.write()
        try:
            shutil.copy(__file__,model.run_dir)
        except NameError:
            print("__file__ not defined.  cannot copy script")

        # temperature is not getting set in point sources.

        if args.resume is None:
            model.ic_ds.salt.values[:]=25
            model.ic_ds.temp.values[:]=10
            model.write_ic_ds()
        else:
            print("Restart -- not setting IC")

        # Even with restart, this does some work
        model.partition()

        model.run_simulation()

