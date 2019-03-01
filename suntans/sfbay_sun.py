"""
Port SF Bay DFM to suntans, for testing ahead of merging with
ocean domain.
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

# bay006: setup on linuxmodeling

if __name__=='__main__':
    args=parser.parse_args()
else:
    # For manually running the script.
    args=parser.parse_args(["-g","grid-connectivity.nc"])

use_temp=True
use_salt=True

# bay003: add perimeter freshwater sources
# bay004: debugging missing san_jose
# bay005: Nkmax=30, stairstep=1
run_dir=args.dir # '/opt/sfb_ocean/suntans/runs/bay003'

if not args.resume:
    model=drv.SuntansModel()
    model.run_start=np.datetime64(args.start)
    model.num_procs=16
    model.use_edge_depths=True
    model.load_template("sun-template.dat")
    model.config['Nkmax']=30
    model.config['stairstep']=1
    dt_secs=5.0
    model.config['dt']=dt_secs
    model.config['ntout']=int(12*3600/dt_secs) # 12-hour map output
    model.config['ntoutStore']=int(86400/dt_secs) # # daily restart file
    model.config['mergeArrays']=0
    model.config['rstretch']=1.1
    model.config['Cmax']=30.0 # volumetric is a better test, this is more a backup.
    # esp. with edge depths, seems better to use z0B so that very shallow
    # edges can have the right drag.
    model.config['CdB']=0
    model.config['z0B']=0.001
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

model.set_run_dir(run_dir,mode='askclobber')

dt_secs=float(model.config['dt'])

model.z_offset=-5
model.dredge_depth=-2

if args.resume is None:
    src_grid="grid-sfbay/sfbay-grid-20190228.nc"
    dest_grid=src_grid.replace(".nc","-bathy.nc")
    assert os.path.exists(src_grid),"Grid %s not found"%src_grid
    assert dest_grid != src_grid

    if utils.is_stale(dest_grid,[src_grid,"bathy.py"]):
        g_src=unstructured_grid.UnstructuredGrid.from_ugrid(src_grid)
        import bathy
        dem=bathy.dem()
        # Add some deep bias by choosing min depth of nodes
        node_depths=dem(g_src.nodes['x'])
        cell_depths=dem(g_src.cells_centroid())
        for c in range(g_src.Ncells()):
            nodes=g_src.cell_to_nodes(c)
            cell_depths[c]=min(cell_depths[c],node_depths[nodes].min())

        while 1: # fill a few nans..
            bad=np.nonzero(np.isnan(cell_depths))[0]
            if len(bad)==0:
                break
            print("Fill %d nans"%len(bad))
            for i in bad:
                nbrs=g_src.cell_to_cells(i)
                cell_depths[i]=np.nanmean(cell_depths[nbrs])

        assert np.all(np.isfinite(cell_depths)),"Whoa hoss - got some nan depth"
        g_src.add_cell_field('depth',cell_depths,on_exists='overwrite')
        if 'depth' in g_src.nodes.dtype.names:
            g_src.delete_node_field('depth')

        # Also set edge depths
        #  First step: edges take shallower of neighboring cells.
        de=np.zeros(g_src.Nedges(),np.float64)
        e2c=g_src.edge_to_cells()
        c1=e2c[:,0].copy() ; c2=e2c[:,1].copy()
        c1[c1<0]=c2[c1<0]
        c2[c2<0]=c1[c2<0]
        de=np.maximum(g_src.cells['depth'][c1],g_src.cells['depth'][c2])
        #  Second step: emulate levees from connectivity
        from stompy.grid import depth_connectivity
        edge_depths=depth_connectivity.edge_connection_depth(g_src,dem,edge_mask=None,centers='centroid')
        invalid=np.isnan(edge_depths)
        edge_depths[invalid]=de[invalid]
        de=np.maximum(de,edge_depths)

        assert np.all(np.isfinite(de)),"Whoa hoss - got some nan depth on edges"
        g_src.add_edge_field('edge_depth',de,on_exists='overwrite')

        g_src.write_ugrid(dest_grid,overwrite=True)

    g=unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid)

    if 1: # override some levee elevations
        # This is very ugly.  Would be better to add gate/structure entries
        # to the gazetteer, and for suntans provide the option to represent
        # gates as closed edges
        override_fn="grid-sfbay/edge-depth-override.shp"
        overrides=wkb2shp.shp2geom(override_fn)
        de=g.edges['edge_depth'].copy()
        # Do this once for min_depth, once for max_depth
        new_depths={}
        for field in ['min_depth','max_depth']:
            plis=[]
            for feat_i,feat in enumerate(overrides):
                if np.isfinite(feat[field]):
                    xy=np.array(feat['geom'].coords)
                    z=feat[field]*np.ones(xy.shape[0])
                    xyz=np.c_[xy,z]
                    pli_feat=[str(feat_i),xyz] # good enough??
                    plis.append(pli_feat)
            new_de=dio.pli_to_grid_edges(g,plis)
            new_depths[field]=np.where(np.isnan(new_de),
                                       de, new_de)
            print("%s: %d pli features, new depths: %s"%
                  (field,len(plis),new_de[np.isfinite(new_de)]))

        de=np.maximum(de,new_depths['min_depth'])
        de=np.minimum(de,new_depths['max_depth'])
        
        g.add_edge_field('edge_depth',de,on_exists='overwrite')

    model.set_grid(g)
else:
    print("Grid comes from restart")
    model.dredge_depth=None # no need to dredge grid if a restart
    
model.add_gazetteer("grid-sfbay/linear_features.shp")
model.add_gazetteer("grid-sfbay/point_features.shp")

##
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
ocean_salt_bc=drv.ScalarBC(name='ocean',scalar='salinity',value=34)
ocean_temp_bc=drv.ScalarBC(name='ocean',scalar='temperature',value=10)
ocean_bc=drv.NOAAStageBC(name='ocean',station=9415020,cache_dir=cache_dir,
                         filters=[dfm.Lowpass(cutoff_hours=1.5)])
model.add_bcs([ocean_bc,ocean_salt_bc,ocean_temp_bc]) 

# Delta inflow
# SacRiver, SJRiver
sac_bc=dfm.NwisFlowBC(name='SacRiver',station=11455420,cache_dir=cache_dir,
                      filters=[dfm.Lowpass(cutoff_hours=3)])
sj_bc =dfm.NwisFlowBC(name='SJRiver',station=11337190,cache_dir=cache_dir,
                      filters=[dfm.Lowpass(cutoff_hours=3)])

sac_salt_bc=drv.ScalarBC(name='SacRiver',scalar='salinity',value=0.0)
sj_salt_bc =drv.ScalarBC(name='SJRiver',scalar='salinity',value=0.0)
sac_temp_bc=drv.ScalarBC(name='SacRiver',scalar='temperature',value=20.0)
sj_temp_bc =drv.ScalarBC(name='SJRiver',scalar='temperature',value=20.0)

model.add_bcs([sac_bc,sj_bc,sac_salt_bc,sj_salt_bc,sac_temp_bc,sj_temp_bc])

#
if 1: # disable if no internet
    # USGS gauged creeks
    for station,name in [ (11172175, "COYOTE"),
                          (11169025, "SCLARAVCc"), # Alviso Sl / Guad river
                          (11180700,"UALAMEDA"), # Alameda flood control
                          (11458000,"NAPA") ]:
        Q_bc=dfm.NwisFlowBC(name=name,station=station,cache_dir=cache_dir)
        salt_bc=drv.ScalarBC(name=name,scalar='salinity',value=0.0)
        temp_bc=drv.ScalarBC(name=name,scalar='temperature',value=20.0)

        model.add_bcs([Q_bc,salt_bc,temp_bc])
else:
    print("Disabling USGS gauged inputs")
#

# WWTP discharging into sloughs
potw_dir="../sfbay_potw"
potw_ds=xr.open_dataset( os.path.join(potw_dir,"outputs","sfbay_delta_potw.nc"))

# the gazetteer uses the same names for potws as the source data
# omits some of the smaller sources, and this does not include any
# benthic discharges
for potw_name in ['sunnyvale','san_jose','palo_alto',
                  'lg','sonoma_valley','petaluma','cccsd','fs','ddsd',
                  'ebda','ebmud','sf_southeast']:
    # This has variously worked and not worked with strings vs bytes.
    # Brute force and try both.
    try:
        Q_da=potw_ds.flow.sel(site=potw_name)
    except KeyError:
        Q_da=potw_ds.flow.sel(site=potw_name.encode())
        
    # Have to seek back in time to find a year that has data for the
    # whole run
    offset=np.timedelta64(0,'D')
    while model.run_stop > Q_da.time.values[-1]+offset:
        offset+=np.timedelta64(365,'D')
    if offset:
        print("Offset for POTW %s is %s"%(potw_name,offset))

    # use the geometry to decide whether this is a flow BC or a point source
    hits=model.match_gazetteer(name=potw_name)
    if hits[0]['geom'].type=='LineString':
        print("%s: flow bc"%potw_name)
        Q_bc=drv.FlowBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)])
    else:
        print("%s: source bc"%potw_name)
        Q_bc=drv.SourceSinkBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)])
        
    salt_bc=drv.ScalarBC(parent=Q_bc,scalar='salinity',value=0.0)
    temp_bc=drv.ScalarBC(parent=Q_bc,scalar='temperature',value=20.0)
    model.add_bcs([Q_bc,salt_bc,temp_bc])

def set_usgs_sfbay_ic(model):
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
            # while developing, initialize everywhere to 34ppt, so we can see rivers
            # coming in
            if 0:
                model.ic_ds.salt.values[:]=34
                model.ic_ds.temp.values[:]=10
            if 1: # for real runs, initialize with Polaris cruise
                set_usgs_sfbay_ic(model)    
            model.write_ic_ds()
        else:
            print("Restart -- not setting IC")

        # Even with restart, this does some work
        model.partition()

        model.run_simulation()

