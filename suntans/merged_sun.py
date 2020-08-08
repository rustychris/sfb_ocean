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
from stompy import utils, filters
from stompy.spatial import field

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
parser.add_argument("-n", "--dryrun", help="Do not actually partition or run the simulation",
                    action='store_true')
parser.add_argument("--ocean",help="Set ocean forcing method",
                    default="velocity-hycom+otps")
parser.add_argument("-g","--write-grid", help="Write grid to ugrid")

if __name__=='__main__':
    args=parser.parse_args()
else:
    # For manually running the script.
    # args=parser.parse_args(["-g","grid-connectivity.nc"])
    # args=parser.parse_args(["-s","2017-06-01","-e","2017-06-05","-d","test-met5-KtoC"])
    args=parser.parse_args(["-s","2017-06-01T00:00:00",
                            "-e","2017-07-01T12:00:00",
                            "-d","/opt2/sfb_ocean/suntans/runs/merge_016-20170601"])
    #raise Exception("Update args")

##

read_otps.OTPS_DATA='../derived'

use_temp=True
use_salt=True
ocean_method=args.ocean # 'hycom'
grid_dir="grid-merge-suisun"
# moving to have all of the elevation offset stuff more manual, less
# magic.  This gives the difference between model 0 and NAVD88.
z_offset_manual=-5 # m
# at Point Reyes, via tidesandcurrents.noaa.gov.  I.e. MSL is 0.94 NAVD88,
# or (-5+0.94) m in the model datum.
msl_navd88=0.94 # m

import local_config
drv.SuntansModel.sun_bin_dir=local_config.sun_bin_dir # "/home/rusty/src/suntans/main"
# AVOID anaconda mpi (at least if suntans is compiled with system mpi)
drv.SuntansModel.mpi_bin_dir=local_config.mpi_bin_dir # "/usr/bin/"

if not args.resume:
    # HYCOM experiments change just before 2017-06-15
    model=drv.SuntansModel()
    model.num_procs=4
    model.load_template("sun-template.dat")

    model.config['Nkmax']=60
    model.config['stairstep']=1
    model.dredge_depth=-2 + z_offset_manual # 2m below the offset of -5m.

    # 2019-07-03: trying 20 - hold on tight!
    dt_secs=20.0
    model.config['dt']=dt_secs
    # had been ramping at 86400, but don't linger so much...
    model.config['thetaramptime']=43200
    
    model.config['ntout']=int(86400/dt_secs) # daily
    model.config['ntaverage']=int(30*60/dt_secs) # 30 minutes
    model.config['ntoutStore']=int(86400/dt_secs) # daily
    model.config['calcaverage']=1
    model.config['averageNetcdfFile']="average.nc"

    # 40 days per average file (i.e. one per month)
    model.config['nstepsperncfile']=int( 40*86400/(int(model.config['ntaverage'])*dt_secs) )
    model.config['mergeArrays']=1
    model.config['metmodel']=5 # wind, temperature nudging
    
    model.config['rstretch']=1.125 # about 1.7m surface layer thickness
    model.config['Cmax']=30.0 # volumetric is a better test, this is more a backup.

    # esp. with edge depths, seems better to use z0B so that very shallow
    # edges can have the right drag.
    model.config['CdB']=0
    model.config['z0B']=0.002

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
    dest_grid=os.path.join(grid_dir,"spliced-bathy.nc")
    grid=unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid)
    # older iterations called bed elevation 'depth', but now I want to call it
    # z_bed

    try:
        grid.cells['z_bed']
    except ValueError:
        raise Exception("Now requiring grid to come in with cell z_bed field")
    
    if 'edge_z_bed' in grid.edges.dtype.names:
        log.info("Grid came in with edge depths")
    else:
        ec=grid.edge_to_cells().copy()
        nc1=ec[:,0]
        nc2=ec[:,1]
        nc1[nc1<0]=nc2[nc1<0] ; nc2[nc2<0]=nc1[nc2<0]
        edge_z=np.maximum( grid.cells['z_bed'][nc1],
                           grid.cells['z_bed'][nc2] )
        grid.add_edge_field('edge_z_bed',edge_z)
    # make sure edge depths actually were included, not just a bunch of zeros.
    assert grid.edges['edge_z_bed'].min()<0.0,"Looks like edge depths were not set on %s"%dest_grid
    
    # Apply offset before handing to model instance.
    grid.cells['z_bed'] += z_offset_manual
    grid.edges['edge_z_bed'] += z_offset_manual
        
    model.set_grid(grid)

    # make sure edge depths actually were included
    edge_z=model.grid.edges['edge_z_bed']
    assert edge_z.min()<0.0,"Looks like edge depths were not set on %s"%dest_grid
else:
    old_model=drv.SuntansModel.load(args.resume)
    model=old_model.create_restart(symlink=True)
    model.dredge_depth=None # no need to dredge grid if a restart

    edge_z=model.grid.edges['edge_z_bed']
    assert edge_z.min()<0.0,"Looks like edge depths were not found in restart"

# common to restart and initial run:
model.projection="EPSG:26910"
model.run_stop=np.datetime64(args.end)

# run_dir='/opt/sfb_ocean/suntans/runs/merge_001-20170601'
model.set_run_dir(args.dir,mode='askclobber')
    
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
ocean_bcs=[]

if ocean_method=='eta-otps':
    ocean_bc=drv.MultiBC(drv.OTPSStageBC,name='Ocean',otps_model='wc')
    ocean_offset_bc=drv.StageBC(name='Ocean',mode='add',water_level=z_offset_manual)
    ocean_bcs+= [ocean_bc,ocean_offset_bc]
elif ocean_method=='flux-otps':
    ocean_bc=drv.MultiBC(drv.OTPSFlowBC,name='Ocean',otps_model='wc')
    ocean_bcs.append(ocean_bc)
elif ocean_method=='velocity-otps':
    ocean_bc=drv.MultiBC(drv.OTPSVelocityBC,name='Ocean',otps_model='wc')
    ocean_bcs.append(ocean_bc)
elif ocean_method.startswith('velocity-hycom'):
    # explicity give bounds to make sure we always download the same
    # subset.
    # z_offset here is the elevation of MSL in the model vertical datum.
    # NAVD88 is offset by z_offset_manual, but MSL is a bit higher than
    # that.
    ocean_bc=drv.HycomMultiVelocityBC(ll_box=hycom_ll_box,
                                      name='Ocean',cache_dir=cache_dir,
                                      z_offset=z_offset_manual+msl_navd88)
    ocean_bcs.append(ocean_bc)
    
    if ocean_method=='velocity-hycom+otps':
        log.info("Including OTPS in addition to HYCOM")
        ocean_tidal_bc=drv.MultiBC(drv.OTPSVelocityBC,name='Ocean',otps_model='wc',mode='add')        
        ocean_bcs.append(ocean_tidal_bc)
    else:        
        log.info("Will *not* add OTPS to HYCOM")

    for ocean_shore in ["Ocean-north-shore",
                        "Ocean-south-shore"]:
        ocean_shore_bc=drv.MultiBC(drv.OTPSStageBC,name=ocean_shore,otps_model='wc')
        offset_bc=drv.StageBC(name=ocean_shore,mode='add',water_level=z_offset_manual+msl_navd88)
        shore_salt_bc=drv.HycomMultiScalarBC(name=ocean_shore,parent=ocean_shore_bc,
                                             scalar='salinity',cache_dir=cache_dir,ll_box=hycom_ll_box)
        shore_temp_bc=drv.HycomMultiScalarBC(name=ocean_shore,parent=ocean_shore_bc,
                                             scalar='temperature',cache_dir=cache_dir,ll_box=hycom_ll_box)
        ocean_bcs += [ocean_shore_bc,offset_bc,shore_salt_bc,shore_temp_bc]

ocean_salt_bc=drv.HycomMultiScalarBC(name='Ocean',scalar='salinity',cache_dir=cache_dir,ll_box=hycom_ll_box)
ocean_temp_bc=drv.HycomMultiScalarBC(name='Ocean',scalar='temperature',cache_dir=cache_dir,ll_box=hycom_ll_box,)
ocean_bcs+=[ocean_salt_bc,ocean_temp_bc]

model.add_bcs(ocean_bcs)

import sfb_common
sfb_common.add_delta_bcs(model,cache_dir)
if 0:
    sfb_common.add_usgs_stream_bcs(model,cache_dir)  # disable if no internet
else:
    # HERE
    #  1. done - copied watershed shapefile into per-grid folder
    #  2. done - edit watershed shapefile to match up with merge-suisun grid
    #  3. Update sfb_common.add_scaled_watersheds to use updated model
    #     interface
    #    for now, keep this shapefile separate from gazetteer, though it
    flow_locations_shp=os.path.join(grid_dir,'watershed_inflow_locations.shp')
    inventory_shp=os.path.join("../sfb_dfm_utils/usgs_inventory.shp")
    sfb_common.add_scaled_streamflow(model,flow_locations_shp,inventory_shp,
                                     cache_dir=cache_dir)

    
sfb_common.add_potw_bcs(model,cache_dir)


##
from stompy.io.local import coamps
import coamps_sfei_wind
six.moves.reload_module(coamps_sfei_wind)
log.info("Adding WIND")
coamps_sfei_wind.add_wind_preblended(model,cache_dir)

assert (np.diff(model.met_ds.nt.values)/np.timedelta64(1,'s')).min() > 0

##

# Air temp for met model
import coamps_temp
six.moves.reload_module(coamps_temp)
if 1:
    # turns out grnd_sea_temp is (a) in Kelvin, and (b)
    # likely has a lot of bleed from land into water.

    # land=1, sea=0
    land_sea_raw=field.GdalGrid('coamps_land_sea')
    # Reproject to UTM
    land_sea=land_sea_raw.warp("EPSG:26910")

    coamps_temp.add_coamps_fields(model,cache_dir,
                                  [('grnd_sea_temp','Tair') ],
                                  mask_field=land_sea)
    model.met_ds.Tair.values[:] -= 273.15 # Kelvin to Celsius
    min_Tair=model.met_ds.Tair.values.min()
    
    if not (min_Tair>=0):
        log.error("min Tair value: %s"%(min_Tair))
    assert min_Tair>=0.0,"Maybe bad K->C conversion"
    assert np.isfinite(min_Tair),"Bad values in Tair"
    model.config['metmodel']=5 # nudge
## 

if 1:
    # update x_rain, y_rain
    x1,x2,y1,y2=model.grid.bounds()

    for v in ['rain','x_rain','y_rain','z_rain','Nrain']:
        if v in model.met_ds:
            del model.met_ds[v]

    pnts=np.array( [ [x1,y1],
                     [x1,y2],
                     [x2,y2],
                     [x2,y1] ] )
    model.met_ds['x_rain']=('Nrain',), pnts[:,0]
    model.met_ds['y_rain']=('Nrain',), pnts[:,1]
    model.met_ds['z_rain']=('Nrain',), 0*pnts[:,0]
    rain=np.zeros( (model.met_ds.dims['nt'],
                    model.met_ds.dims['Nrain']), np.float64)
    if 0:
        # test with constant 150mm/month evaporation
        rain[:,:] = -5.8e-8
    else:
        # because of the need to do some filling, don't fetch this straight
        # from cimis, but use the pre-processed data via sfb_common.
        cimis=sfb_common.cimis_net_precip(cache_dir=cache_dir)
        # deal with difference in time values
        met_dn=utils.to_dnum(model.met_ds.nt.values)
        rain_dn=utils.to_dnum(cimis.time.values)
        rain_interp=np.interp(met_dn,rain_dn,cimis.net_rain.values)
        rain[:,:]=rain_interp[:,None]
        # convert mm/hr to m/s
        rain *= 1. / 1000. / 3600.
        # create rain data array.
    model.met_ds['rain']=('nt','Nrain'), rain
    
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

    if 0: # set freesurface - skip if using Ocean-north-shore and Ocean-south-shore, this
        model.ic_ds.eta.isel(time=0).values[...]=z_offset_manual+msl_navd88

    if 1: # include roughness
        if 'z0B' in model.grid.edges.dtype.names:
            edge_z0B=model.grid.edges['z0B'].reshape([1,-1])
        else:
            # depth-based
            edge_z_bed=model.grid.edges['edge_z_bed']
            edge_z0B=np.interp(edge_z_bed,
                               z_offset_manual +
                               np.array([ -20, -10,   -5,   -3,   0]),
                               np.array([1e-5,1e-4,0.001,0.005,0.01]))
            edge_z0B=edge_z0B.reshape([1,-1])
        log.info('Writing roughness to initial condition')
        model.ic_ds['z0B']=('time','Ne'),edge_z0B

if __name__=='__main__':
    if args.write_grid:
        model.grid.write_ugrid(args.write_grid,overwrite=True)
    else:
        model.write()
        try:
            script=__file__
        except NameError:
            log.warning("__file__ not defined.  cannot copy script")
            script=None 
        if script:
            shutil.copy(script,model.run_dir)
           
        if args.resume is None:
            set_ic(model)
            model.write_ic_ds()

        if not args.dryrun:
            model.partition()
            model.run_simulation()

