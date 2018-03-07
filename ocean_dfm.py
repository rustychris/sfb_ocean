"""
Driver script for coastal-ocean scale DFM runs, initially
to support microplastics project
"""
import logging
log=logging.getLogger('ocean_dfm')
log.setLevel(logging.INFO)

import subprocess
import copy
import os
import sys
import glob
import shutil
import datetime

import six

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy.spatial import proj_utils, field

from stompy.model.delft import dfm_grid
from stompy.model import otps

from stompy import filters, utils
from stompy.spatial import wkb2shp

import stompy.model.delft.io as dio
from stompy.grid import unstructured_grid

import sfb_dfm_utils

## 
dfm_bin_dir="/opt/software/delft/dfm/r53925-opt/bin"

utm2ll=proj_utils.mapper('EPSG:26910','WGS84')
ll2utm=proj_utils.mapper('WGS84','EPSG:26910')

## 
mdu=dio.MDUFile('template.mdu')

# short_test_01: straight up waterlevel, 100% from OTPS
# short_test_02: fix bathy, tried riemann, dirichlet
# short_test_03: add DC offset, start comparing Point Reyes
#                clip bathy to -4, not -10
#                add salinity to BC and mdu.  initial=34
#                bump up to 6 days
# short_test_04: Adding COAMPS wind
# short_test_05: Convert to more complete DFM script
# short_test_06: Try 3D, 10 layers
# short_test_07: ragged boundary
# short_test_08: Adding SF Bay
# medium_09: Longer run, with temperature, and ill-behaved velocity
# medium_10: add sponge layer diffusion for scalars, too
# short_test_11: return to ROMS-only domain, and shorter duration
# short_test_12: attempt z layers
# short_test_13: bring the full domain back
# short_test_14: bring back depth-average advected velocity, and then depth-varying velocity
# medium_15: longer test.  runs, but doesn't look good against observations.
# medium_16: try forcing more velocity than freesurface.
# short_17: return to ocean only, single cores
# short_18: more aggressive sponge layer, no wind.
# short_19: add wind back in
# short_20: and switch to HYCOM
# short_21: now OTPS velocities, no freesurface BCs
# short_22: reduce to a SINGLE velocity BC to see if it is showing up correctly.
# short_23: apply at most one BC to each cell
# short_24: apply constant flows with "label" values
# short_25: check for z-layer issue
run_name="short_25"

include_fresh=False # or True
layers='z' # or 'sigma'
grid='ragged_coast' # 'rectangle_coast' 'ragged_full', ...
nprocs=1 # 16
mdu['physics','Temperature']=1
mdu['physics','Salinity']=1
use_wind=False
# not implemented otps_flux=True # use fluxes rather than velocities from OTPS

coastal_source=['otps','hycom'] # 'roms'

# ROMS has some wacky data, especially at depth.  set this to True to zero out
# data below a certain depth (written as positive-down sounding)
coastal_max_sounding=20000 # allow all depths
set_3d_ic=True
extrap_bay=False # for 3D initial condition whether to extrapolated data inside the Bay.

##

run_base_dir=os.path.join('runs',run_name)
os.path.exists(run_base_dir) or os.makedirs(run_base_dir)

mdu.set_filename(os.path.join(run_base_dir,run_name+".mdu"))

mdu['geometry','Kmx']=20
if layers=='sigma':
    mdu['geometry','SigmaGrowthFactor']=1

run_start=ref_date=np.datetime64('2017-07-01')
run_stop=np.datetime64('2017-08-01')

mdu.set_time_range(start=run_start,
                   stop=run_stop,
                   ref_date=ref_date)

if 1: # clean out most of the run dir:
    # rm -r *.pli *.tim *.t3d *.mdu FlowFM.ext *_net.nc DFM_* *.dia *.xy* initial_conditions_*
    patts=['*.pli','*.tim','*.t3d','*.mdu','FlowFM.ext','*_net.nc','DFM_*', '*.dia',
           '*.xy*','initial_conditions*','dflowfm-*.log']
    for patt in patts:
        matches=glob.glob(os.path.join(run_base_dir,patt))
        for m in matches:
            if os.path.isfile(m):
                os.unlink(m)
            elif os.path.isdir(m):
                shutil.rmtree(m)
            else:
                raise Exception("What is %s ?"%m)


if layers=='z':
    mdu['geometry','Layertype']=2 # z layers
    if 0: # works, but uniform layers
        mdu['geometry','StretchType']=0 # uniform
        mdu['geometry','StretchCoef']=""
    else:
        mdu['geometry','StretchType']=2 # exponential
        # surface percentage, ignored, bottom percentage
        # mdu['geometry','StretchCoef']="0.002 0.02 0.8"
        # This gives about 2m surface cells, and O(500m) deep layers.
        mdu['geometry','StretchCoef']="0.0003 0.02 0.7"
        mdu['numerics','Zwsbtol'] = 0.0 # that's the default anyway...
        # This is the safer of the options, but gives a stairstepped bed.
        mdu['numerics','Keepzlayeringatbed']=1
        # This helps with reconstructing the z-layer geometry, better than
        # trying to duplicate dflowfm layer code.
        mdu['output','FullGridOutput']    = 1
elif layers=='sigma':
    mdu['geometry','Layertype']=1 # sigma
    if 1: # 
        mdu['geometry','StretchType']=1 # user defined 
        # These must sum exactly to 100.
        mdu['geometry','StretchCoef']="8 8 7 7 6 6 6 6 5 5 5 5 5 5 5 5 2 2 1 1"
    else:
        mdu['geometry','StretchType']=0 # uniform
else:
    raise Exception("bad layer choice '%s'"%layers)
    

old_bc_fn = os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])
## 

from sfb_dfm_utils import ca_roms, coamps, hycom

# Get the ROMS inputs:
coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]
if 'roms' in coastal_source:
    coastal_files=ca_roms.fetch_ca_roms(coastal_time_range[0],coastal_time_range[1])
elif 'hycom' in coastal_source:
    # As long as these are big enough, don't change (okay if too large),
    # since the cached data relies on the ll ranges matching up.
    hycom_lon_range=[-124.7, -121.7 ]
    hycom_lat_range=[36.2, 38.85]

    coastal_files=hycom.fetch_range(hycom_lon_range,hycom_lat_range,coastal_time_range)
else:
    coastal_files=None

##
if grid=='rectangle_coast': # rectangular subset
    ugrid_file='derived/matched_grid_v00.nc'

    if not os.path.exists(ugrid_file):
        g=ca_roms.extract_roms_subgrid()
        ca_roms.add_coastal_bathy(g)
        g.write_ugrid(ugrid_file)
    else:
        g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
    coastal_bc_coords=None 
    # should get some coordinates if I return to this grid
    raise Exception("Probably ought to fill in coastal_bc_coords for this grid")
elif grid=='ragged_coast': # ragged edge
    ugrid_file='derived/matched_grid_v01.nc'
    
    if not os.path.exists(ugrid_file):
        poly=wkb2shp.shp2geom('grid-poly-v00.shp')[0]['geom']
        g=ca_roms.extract_roms_subgrid_poly(poly)
        ca_roms.add_coastal_bathy(g)
        g.write_ugrid(ugrid_file)
    else:
        g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
        g_shp='derived/matched_grid_v01.shp'
        if not os.path.exists(g_shp):
            g.write_edges_shp(g_shp)
    coastal_bc_coords=[ [450980., 4291405.], # northern
                        [595426., 4037083.] ] # southern
elif grid=='ragged_splice': # Spliced grid generated in splice_grids.py
    ugrid_file='spliced_grids_01_bathy.nc'
    g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
    # define candidates based on start/end coordinates
    coastal_bc_coords=[ [450980., 4291405.], # northern
                        [595426., 4037083.] ] # southern
else:
    raise Exception("Unknown grid %s"%grid)
## 

# Identify ocean boundary edges
# Limit the boundary edges to edges which have a real cell on the other
# side in the ROMS output

if coastal_files is not None:
    # Used to choose the candidate subset of edges based on some stuff in
    # the grid, but to be more flexible about choices of coastal ocean 
    # data, instead rely on a coordinate pair defining the section of
    # grid boundary to be forced by coastal sources

    if coastal_bc_coords is not None:
        candidate_nodes=g.select_nodes_boundary_segment(coastal_bc_coords)
        candidates=[ g.nodes_to_edge( [a,b] )
                     for a,b in zip(candidate_nodes[:-1],
                                    candidate_nodes[1:]) ]
        candidates=np.array(candidates)
    else:
        candidates=None # !? danger will robinson.
        
    ca_roms.annotate_grid_from_data(g,coastal_files,candidate_edges=candidates)

    boundary_edges=np.nonzero( g.edges['src_idx_out'][:,0] >= 0 )[0]

## 

# To get lat/lon info, and later used for the initial condition
src=xr.open_dataset(coastal_files[0])

# May move more of this to sfb_dfm_utils in the future
Otps=otps.otps_model.OTPS('/home/rusty/src/otps/OTPS2', # Locations of the OTPS software
                          '/opt/data/otps') # location of the data

# xy for boundary edges:
boundary_out_lats=src.lat.values[ g.edges['src_idx_out'][boundary_edges,0] ]
boundary_out_lons=(src.lon.values[ g.edges['src_idx_out'][boundary_edges,1] ] + 180) % 360 - 180
boundary_out_ll=np.c_[boundary_out_lons,boundary_out_lats]

z_harmonics = Otps.extract_HC( boundary_out_ll )
u_harmonics = Otps.extract_HC( boundary_out_ll, quant='u')
v_harmonics = Otps.extract_HC( boundary_out_ll, quant='v')

pad=np.timedelta64(2,'D')
otps_times=np.arange(run_start-pad, run_stop+pad,
                     np.timedelta64(600,'s'))
otps_water_level=otps.reconstruct(z_harmonics,otps_times)
otps_u=otps.reconstruct(u_harmonics,otps_times)
otps_v=otps.reconstruct(v_harmonics,otps_times)
# convert cm/s to m/s
otps_u.result[:] *= 0.01 
otps_v.result[:] *= 0.01

##

coastal_boundary_data=ca_roms.extract_data_at_boundary(coastal_files,g,boundary_edges)

## 

os.path.exists(old_bc_fn) and os.unlink(old_bc_fn)

# for adding in MSL => NAVD88 correction.  Just dumb luck that it's 1.0
# Seems that both CA ROMS and HYCOM are relative to MSL.
dfm_zeta_offset=1.0

# average tidal prediction across boundary at start of simulation
# plus above DC offset
zeta_ic = dfm_zeta_offset + np.interp(utils.to_dnum(run_start),
                                      utils.to_dnum(otps_water_level.time),
                                      otps_water_level.result.mean(dim='site'))
mdu['geometry','WaterLevIni'] = zeta_ic

import common
from common import write_pli, write_tim, write_t3d

if 'edge_depth' in g.edges.dtype.names:
    edge_depth=g.edges['edge_depth']
else:
    edge_depth=g.nodes['depth'][ g.edges['nodes'] ].mean(axis=1)
                                                         
for ji,j in enumerate(boundary_edges):
    src_name='oce%05d'%j
    print(src_name)
    
    depth=edge_depth[j]

    if 1: # bring in OTPS harmonics:
        water_level=dfm_zeta_offset + otps_water_level.result.isel(site=ji)

        veloc_u=otps_u.result.isel(site=ji)
        veloc_v=otps_v.result.isel(site=ji)
        veloc_uv=xr.DataArray(np.c_[veloc_u.values,veloc_v.values],
                              coords=[('time',veloc_u.time),('comp',['e','n'])])
        veloc_uv.name='uv'

        # inward-positive
        veloc_normal=(g.edges['bc_norm_in'][j,0]*veloc_u + g.edges['bc_norm_in'][j,1]*veloc_v)
        
    if 1: # Coastal model:        
        coastal_dt=np.median( np.diff(coastal_boundary_data.time.values) )
        coastal_dt_h= coastal_dt / np.timedelta64(3600,'s')

        if 0: # Add Coastal model zeta to waterlevel
            coastal_water_level=coastal_boundary_data.zeta.isel(boundary=ji)

            if coastal_dt_h<12:
                # 36h cutoff with 6h ROMS data
                # Note that if the HYCOM fetch switches to finer resolution,
                # it's unclear whether we want to filter it further or not, since
                # it will be non-tidal.
                # This will have some filtfilt trash at the end, probably okay
                # at the beginning
                coastal_water_level.values[:] = filters.lowpass(coastal_water_level.values,
                                                                cutoff=36.,order=4,
                                                                dt=coastal_dt_h)

            # As far as I know, ROMS and HYCOM zeta are relative to MSL
            coastal_interp=np.interp( utils.to_dnum(water_level.time),
                                      utils.to_dnum(coastal_water_level.time),
                                      coastal_water_level.values )
            water_level.values += coastal_interp
            
        if 1: # salinity, temperature
            if 1: # proper spatial variation:
                salinity_3d=coastal_boundary_data.isel(boundary=ji).salt
                temperature_3d=coastal_boundary_data.isel(boundary=ji).temp
            else: # spatially constant
                salinity_3d=coastal_boundary_data.salt.mean(dim='boundary')
                temperature_3d=coastal_boundary_data.temp.mean(dim='boundary')

            if coastal_dt_h<12:
                for zi in range(len(salinity_3d.depth)):
                    salinity_3d.values[:,zi] = filters.lowpass(salinity_3d.values[:,zi],
                                                               cutoff=36,order=4,dt=coastal_dt_h)
                    temperature_3d.values[:,zi] = filters.lowpass(temperature_3d.values[:,zi],
                                                                  cutoff=36,order=4,dt=coastal_dt_h)

        if 0: # 3D velocity
            coastal_u=coastal_boundary_data.isel(boundary=ji).u
            coastal_v=coastal_boundary_data.isel(boundary=ji).v

            for zi in range(len(coastal_u.depth)):
                if coastal_max_sounding < coastal_u.depth[zi]:
                    coastal_u.values[:,zi]=0.0
                    coastal_v.values[:,zi]=0.0
                else:
                    if coastal_dt_h<12:
                        coastal_u.values[:,zi] = filters.lowpass(coastal_u.values[:,zi],
                                                                 cutoff=36,order=4,dt=coastal_dt_h)
                        coastal_v.values[:,zi] = filters.lowpass(coastal_v.values[:,zi],
                                                                 cutoff=36,order=4,dt=coastal_dt_h)
            coastal_uv=xr.DataArray( np.array([coastal_u.values,coastal_v.values]).transpose(1,2,0),
                                     coords=[('time',coastal_u.time),
                                             ('depth',coastal_u.depth),
                                             ('comp',['e','n'])])
            # inward facing normal for this boundary edge
            inward_nx,inward_ny = g.edges['bc_norm_in'][j]
            coastal_normal=xr.DataArray( inward_nx*coastal_u.values+inward_ny*coastal_v.values,
                                         coords=[('time',coastal_u.time),
                                                 ('depth',coastal_u.depth)])

    if 0: # depth-varying from tidal model+ROMS
        # veloc_uv: has the tidal time scale
        # roms_u,roms_v: has the vertical variation
        # Used to prescribe u,v vector velocity, but I think it's supposed to just
        # be normal velocity.
        veloc_3d=xr.DataArray( np.zeros( (len(veloc_normal),len(coastal_normal.depth)) ),
                               dims=['time','depth'],
                               coords={'time':veloc_normal.time,
                                       'depth':coastal_normal.depth} )
        # This will broadcast tidal velocity over depth
        veloc_3d+=veloc_normal
        if 0: # not yet adding this in
            # And this is supposed grab nearest-in-time ROMS velocity to add in.
            # Note that nearest just pulls an existing record, but retains the
            # original time value, so grab the values directly
            veloc_3d.values+=coastal_normal.sel(time=veloc_3d.time,method='nearest').values
            veloc_3d.name='un'

    # Include velocity in riemann BC:
    #   from page 124 of the user manual:
    #   zeta = 2*zeta_b - sqrt(H/g)*u - zeta_0
    #   zeta_0 is initial water level, aka zeta_ic
    #   if zeta is what we want the water level to be,
    #   and zeta_b is what we give to DFM, then
    #   zeta_b=0.5*( zeta+zeta_0 + sqrt(H/g)*u)
    riemann=0.5*(water_level + zeta_ic + np.sqrt(np.abs(depth)/9.8)*veloc_normal)
                
    assert np.all( np.isfinite(water_level.values) ) # sanity check

    forcing_data=[]

    if int(mdu['physics','Salinity']):
        forcing_data.append( ('salinitybnd',salinity_3d,'_salt') )

    if int(mdu['physics','Temperature']):
        forcing_data.append( ('temperaturebnd',temperature_3d,'_temp') )
        
    if 0: # riemann only
        # This works pretty well, good agreement at Point Reyes.
        forcing_data.append( ('riemannbnd',riemann,'_rmn') )

    if 0: # Riemann only in shallow areas:
        if depth<-200:
            forcing_data.append( ('waterlevelbnd',water_level,'_ssh') )
        else:
            forcing_data.append( ('riemannbnd',riemann,'_rmn') )

    if 1: # waterlevel in shallow areas, velocity elsewhere
        # waterlevel in shallow areas has not been that great..
        # forcing large fluxes through shallow areas to make up for
        # large scale volume errors.
        # For short_21, use entirely velocities
        #if (depth>-200) or (ji%10==0):
        #if (depth<-200) and (ji%10==0):
        if False:
            forcing_data.append( ('waterlevelbnd',water_level,'_ssh') )
        else:
            # This maybe was causing issues around 3.5 hours in
            # forcing_data.append( ('velocitybnd',veloc_3d,'_uv3') )
            # Try the simpler forcing -- this is getting some memory
            # errors -- somewhere in the processing of velocity bcs
            # it overwrites some geometry link info
            # forcing_data.append( ('velocitybnd',veloc_uv,'_uv') )

            # Maybe it was wrong to use vector velocity with
            # velocitybnd.
            # try just normal velocity?
            forcing_data.append( ('velocitybnd',veloc_normal,'_un') )
            
    if 0: # included advected velocity -- cannot coexist with velocitybnd
        forcing_data.append( ('uxuyadvectionvelocitybnd',veloc_3d,'_uv3') )

    for quant,da,suffix in forcing_data:
        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=%s"%quant,
                   "FILENAME=%s%s.pli"%(src_name,suffix),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

        feat_suffix=write_pli(g,run_base_dir,src_name,j,suffix)
        
        if 'depth' in da.dims:
            write_t3d(da,suffix,feat_suffix,edge_depth[j],
                      quantity=quant.replace('bnd',''),
                      mdu=mdu)
        else:
            write_tim(da,suffix,feat_suffix,mdu=mdu)
            
## 

# Write spatially-variable horizontal eddy viscosity field
ca_roms.add_sponge_layer(mdu,run_base_dir,g,boundary_edges,
                         sponge_visc=5000,
                         background_visc=10,
                         sponge_L=50000,quantity='viscosity')

# For diffusivity, have to use smaller background number so that
# freshwater inflows are not driven by diffusion
ca_roms.add_sponge_layer(mdu,run_base_dir,g,boundary_edges,
                         sponge_visc=5000,
                         background_visc=0.001,
                         sponge_L=50000,quantity='diffusivity')

##

if 1:
    obs_shp_fn = "inputs-static/observation-points.shp"
    # Observation points taken from shapefile for easier editing/comparisons in GIS
    obs_pnts=wkb2shp.shp2geom(obs_shp_fn)
    obs_fn='observation_pnts.xyn'
    
    with open(os.path.join(run_base_dir,obs_fn),'wt') as fp:
        for idx,row in enumerate(obs_pnts):
            xy=np.array(row['geom'])
            fp.write("%12g %12g '%s'\n"%(xy[0], xy[1], row['name']))
    mdu['output','ObsFile'] = obs_fn

##

dredge_depth=-1
# sfb_dfm_v2_base_dir="../../sfb_dfm_v2"
adjusted_pli_fn = 'nudged_features.pli'

if include_fresh: 
    # ---------SF FRESH
    if 0: # BAHM data
        # SF Bay Freshwater and POTW, copied from sfb_dfm_v2:
        # features which have manually set locations for this grid
        # Borrow files from sfb_dfm_v2 -- should switch to submodules

        if 1: # Transcribe to shapefile for debugging/vis
            from shapely import geometry
            from stompy.spatial import wkb2shp
            adj_pli_feats=dio.read_pli(adjusted_pli_fn)
            names=[feat[0] for feat in adj_pli_feats]
            geoms=[geometry.Point(feat[1].mean(axis=0)) for feat in adj_pli_feats]
            wkb2shp.wkb2shp('derived/input_locations.shp',geoms,fields={'name':names},
                            overwrite=True)

        # kludge - wind the clock back a bit:
        print("TOTAL KLUDGE ON FRESHWATER")
        from sfb_dfm_utils import sfbay_freshwater

        # This will pull freshwater data from 2012, where we already
        # have a separate run which kind of makes sense
        time_offset=np.datetime64('2012-01-01') - np.datetime64('2017-01-01') 

        sfbay_freshwater.add_sfbay_freshwater(run_base_dir,
                                              run_start,run_stop,ref_date,
                                              adjusted_pli_fn,
                                              freshwater_dir='sfbay_freshwater',
                                              grid=g,
                                              dredge_depth=dredge_depth,
                                              old_bc_fn=old_bc_fn,
                                              all_flows_unit=False,
                                              time_offset=time_offset)
    else: # watershed scaling
        # Developing freshwater inputs for periods not covered by BAHM, but covered by
        # a subset of USGS gages.
        from sfb_dfm_utils import sfbay_scaled_watersheds
        sfbay_scaled_watersheds.add_sfbay_freshwater(mdu,
                                                     'inputs-static/watershed_inflow_locations.shp',
                                                     g,dredge_depth)

##

if include_fresh: # POTWs
    # The new-style boundary inputs file (FlowFM_bnd_new.ext) cannot represent
    # sources and sinks, so these come in via the old-style file.
    potw_dir='sfbay_potw'
    from sfb_dfm_utils import sfbay_potw

    sfbay_potw.add_sfbay_potw(run_base_dir,
                              run_start,run_stop,ref_date,
                              potw_dir,
                              adjusted_pli_fn,
                              g,dredge_depth,
                              old_bc_fn,
                              all_flows_unit=False,
                              time_offset= np.datetime64('2016-01-01') - np.datetime64('2017-01-01') )
if include_fresh: # DELTA
    # Delta boundary conditions
    # may need help with inputs-static
    from sfb_dfm_utils import delta_inflow

    delta_inflow.add_delta_inflow(run_base_dir,
                                  run_start,run_stop,ref_date,
                                  static_dir=os.path.join("inputs-static"),
                                  grid=g,dredge_depth=dredge_depth,
                                  old_bc_fn=old_bc_fn,
                                  all_flows_unit=False,
                                  time_offset= np.datetime64('2016-01-01') - np.datetime64('2017-01-01'))

# ---------- END SF FRESH, POTW, DELTA

##

if 1:
    mdu['geometry','NetFile'] = os.path.basename(ugrid_file).replace('.nc','_net.nc')
    dfm_grid.write_dfm(g,os.path.join(run_base_dir,mdu['geometry','NetFile']),
                       overwrite=True)

# This step is pretty slow the first time around.
if use_wind:
    coamps.add_coamps_to_mdu(mdu,run_base_dir,g,use_existing=True)


##

# Setting a full 3D initial condition requires a partitioned
# run, so go ahead partition now:


dfm_output_count=0
def dflowfm(mdu_fn,args=['--autostartstop']):
    global dfm_output_count
    
    cmd=[os.path.join(dfm_bin_dir,"dflowfm")] + args
    if mdu_fn is not None:
        cmd.append(os.path.basename(mdu_fn))

    if nprocs>1:
        cmd=["%s/mpiexec"%dfm_bin_dir,"-n","%d"%nprocs] + cmd

    # This is more backwards compatible than 
    # passing cwd to subprocess()
    pwd=os.getcwd()
    dfm_output_count+=1
    log_file= os.path.join(run_base_dir,'dflowfm-log-%d.log'%dfm_output_count)
    with open(log_file, 'wt') as fp:
        log.info("Command '%s' logging to %s"%(cmd,log_file))
        try:
            os.chdir(run_base_dir)
            res=subprocess.call(cmd,stderr=subprocess.STDOUT,stdout=fp)
        finally:
            os.chdir(pwd)
            log.info("Command '%s' completed"%cmd)
    return res


def partition_grid(clear_old=True):
    if nprocs<=1:
        return

    if clear_old:
        # Anecdotally it might help to clear the old netcdf files?
        # Just trying to chase down how some bad values crept into these.
        grid_fn=mdu.filepath(['geometry','NetFile'])
        for p in range(nprocs):
            gridN_fn=grid_fn.replace('_net.nc','_%04d_net.nc')
            if os.path.exists(gridN_fn):
                os.unlink(gridN_fn)
    
    dflowfm(None,["--partition:ndomains=%d"%nprocs,mdu['geometry','NetFile']])
    
        
def partition_mdu(mdu_fn):
    if nprocs<=1:
        return
    
    # similar, but for the mdu:
    cmd="%s/generate_parallel_mdu.sh %s %d 6"%(dfm_bin_dir,os.path.basename(mdu_fn),nprocs)
    pwd=os.getcwd()
    try:
        os.chdir(run_base_dir)
        res=subprocess.call(cmd,shell=True)
    finally:
        os.chdir(pwd)

## 

# Need a partitioned grid for setting up 3D initial conditions
partition_grid()

##

if set_3d_ic:
    if nprocs<=1:
        map_fn=os.path.join(run_base_dir,
                            'DFM_OUTPUT_%s-tmp'%run_name,
                            '%s-tmp_map.nc'%run_name)
        map_fns=[map_fn]
    else:
        map_fns=[os.path.join(run_base_dir,
                              'DFM_OUTPUT_%s-tmp'%run_name,
                              '%s-tmp_%04d_map.nc'%(run_name,n))
                 for n in range(nprocs)]
        map_fn=map_fns[0]

    if 1: # clear old maps
        # Used to allow re-using this run, but that has been a constant source of pain,
        # or at least perceived pain.
        for fn in map_fns:
            os.path.exists(fn) and os.unlink(fn)
            
    if not os.path.exists(map_fn):
        # Very short run just to get a map file
        # 
        temp_mdu=copy.deepcopy(mdu)
        temp_mdu.set_time_range(start=run_start,
                                stop=run_start+np.timedelta64(60,'s'),
                                ref_date=ref_date)
        temp_mdu_fn=os.path.join(run_base_dir,run_name+"-tmp.mdu")
        temp_mdu.write(temp_mdu_fn)
        partition_mdu(temp_mdu_fn)

        dflowfm(temp_mdu_fn)

    if nprocs<=1:
        ic_fns=[os.path.join(run_base_dir,'initial_conditions_map.nc')]
    else:
        ic_fns=[os.path.join(run_base_dir,'initial_conditions_%04d_map.nc'%n)
                for n in range(nprocs)]

    # For forcing this to run -- include this while things are changing a lot
    [(os.path.exists(f) and os.unlink(f)) for f in ic_fns]

    # The map file should always exist now that we do a short run to
    # create it above, but if one wanted to skip the 3D IC, will
    # leave this test in.

    if os.path.exists(map_fn):
        if not os.path.exists(ic_fns[0]):
            # Get a baseline, global 2D salt field from Polaris/Peterson data
            if extrap_bay:
                from sfb_dfm_utils import initial_salinity

                usgs_init_salt=initial_salinity.samples_from_usgs(run_start,field='salinity')
                usgs_init_temp=initial_salinity.samples_from_usgs(run_start,field='temperature')

                cc_salt = initial_salinity.samples_to_cells(usgs_init_salt,g)
                cc_temp = initial_salinity.samples_to_cells(usgs_init_temp,g)
                salt_extrap_field=field.XYZField(X=cc_salt[:,:2], F=cc_salt[:,2])
                temp_extrap_field=field.XYZField(X=cc_temp[:,:2], F=cc_temp[:,2])
                salt_extrap_field.build_index()
                temp_extrap_field.build_index()
                missing_val=-999
            else:
                missing_val=32

            # Scan the coastal model files, find one close to our start date
            for fn in coastal_files:
                snap=xr.open_dataset(fn)
                if snap.time.ndim>0:
                    snap=snap.isel(time=0)
                if snap.time>run_start:
                    print("Will use %s for initial condition"%fn)
                    break
                snap.close()

            for ic_fn,map_fn in zip(ic_fns,map_fns):
                ic_map=ca_roms.set_ic_from_map_output(snap,
                                                      map_file=map_fn,
                                                      mdu=mdu,
                                                      output_fn=None, # ic_fn,
                                                      missing=missing_val)
                if extrap_bay:
                    # Any missing data should get filled in with 2D data from cc_salt
                    # This is way harder than it ought to be because in this case xarray
                    # is getting in the way a lot.
                    all_xy=np.c_[ ic_map.FlowElem_xcc.values,
                                  ic_map.FlowElem_ycc.values ]

                    if mdu['physics','Salinity']=="1":
                        # Salinity:
                        #  fill missing values in ic_map.sa1 with the 2D extrapolated data
                        salt_fill_2d=salt_extrap_field.interpolate(all_xy,interpolation='nearest') # 2-3 seconds
                        salt_fill_3d=xr.DataArray(salt_fill_2d,dims=['nFlowElem'])
                        _,salt_fill_3dx=xr.broadcast(ic_map.sa1,salt_fill_3d)
                        sa1=ic_map.sa1
                        new_sa1=sa1.where(sa1!=missing_val,other=salt_fill_3dx)
                        ic_map['sa1']=new_sa1
                    if mdu['physics','Temperature']=="1":
                        # Temperature:
                        #  fill missing values in ic_map.tem1 with the 2D extrapolated data
                        temp_fill_2d=temp_extrap_field.interpolate(all_xy,interpolation='nearest') # 2-3 seconds
                        temp_fill_3d=xr.DataArray(temp_fill_2d,dims=['nFlowElem'])
                        _,temp_fill_3dx=xr.broadcast(ic_map.tem1,temp_fill_3d)
                        tem1=ic_map.tem1
                        new_tem1=tem1.where(tem1!=missing_val,other=temp_fill_3dx)
                        ic_map['tem1']=new_tem1

                ic_map.to_netcdf(ic_fn,format='NETCDF3_64BIT')
        else:
            ic_map=xr.open_dataset(ic_fns[0]) # for timestamping

        mdu['restart','RestartFile']='initial_conditions_map.nc'
        # Had some issues when this timestamp exactly lined up with the reference date.
        # adding 1 minute works around that, with a minor warning that these don't match
        # exactly
        restart_time=utils.to_datetime(ic_map.time.values[0] + np.timedelta64(60,'s')).strftime('%Y%m%d%H%M')
        mdu['restart','RestartDateTime']=restart_time
else:
    # don't set 3D IC:
    mdu['restart','RestartFile']=""
    mdu['restart','RestartDateTime']=""


##

mdu_fn=os.path.join(run_base_dir,run_name+".mdu")
mdu.write(mdu_fn)

##

partition_mdu(mdu_fn)

## 
if set_3d_ic and nprocs>1:
    # The partition script doesn't scatter initial conditions, though
    mdu_fns=[mdu_fn.replace('.mdu','_%04d.mdu'%n)
             for n in range(nprocs)]
    for sub_mdu_fn,ic_fn in zip(mdu_fns,ic_fns):
        sub_mdu=dio.MDUFile(sub_mdu_fn)
        sub_mdu['restart','RestartFile']=os.path.basename(ic_fn)
        sub_mdu.write(sub_mdu_fn)

##

dflowfm(mdu_fn)

# Getting there, but it fails on no data associated with unnamed.. ?
