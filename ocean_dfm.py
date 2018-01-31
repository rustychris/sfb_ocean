"""
Driver script for coastal-ocean scale DFM runs, initially
to support microplastics project
"""
import subprocess
import copy
import os
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

#dfm_bin_dir="/opt/software/delft/dfm/r52184-opt/bin"
dfm_bin_dir="/opt/software/delft/dfm/r53925-opt/bin"

utm2ll=proj_utils.mapper('EPSG:26910','WGS84')

## 

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
run_name="short_test_13"

include_fresh=True
# layers='sigma'
layers='z'
# Seems that it runs okay without set_3d_ic, but hangs up with it.
set_3d_ic=True

extrap_bay=True # for 3D initial condition whether to extrapolated data inside the Bay.

##

run_base_dir=os.path.join('runs',run_name)
os.path.exists(run_base_dir) or os.makedirs(run_base_dir)

mdu=dio.MDUFile('template.mdu')
mdu.set_filename(os.path.join(run_base_dir,run_name+".mdu"))

mdu['geometry','Kmx']=20
if layers=='sigma':
    mdu['geometry','SigmaGrowthFactor']=1

run_start=ref_date=np.datetime64('2017-07-10')
run_stop=np.datetime64('2017-07-30')

mdu.set_time_range(start=run_start,
                   stop=run_stop,
                   ref_date=ref_date)


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
else:
    mdu['geometry','Layertype']=1 # sigma
    if 1: # 
        mdu['geometry','StretchType']=1 # user defined 
        # These must sum exactly to 100.
        mdu['geometry','StretchCoef']="8 8 7 7 6 6 6 6 5 5 5 5 5 5 5 5 2 2 1 1"
    else:
        mdu['geometry','StretchType']=0 # uniform
    
    

old_bc_fn = os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])
## 

from sfb_dfm_utils import ca_roms, coamps

# Get the ROMS inputs:
ca_roms_files = ca_roms.fetch_ca_roms(run_start,run_stop)

##
if 0: # rectangular subset
    ugrid_file='derived/matched_grid_v00.nc'

    if not os.path.exists(ugrid_file):
        g=ca_roms.extract_roms_subgrid()
        ca_roms.add_coastal_bathy(g)
        g.write_ugrid(ugrid_file)
    else:
        g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)

elif 0: # ragged edge
    ugrid_file='derived/matched_grid_v01.nc'
    
    if not os.path.exists(ugrid_file):
        six.moves.reload_module(ca_roms)
        poly=wkb2shp.shp2geom('grid-poly-v00.shp')[0]['geom']
        g=ca_roms.extract_roms_subgrid_poly(poly)
        ca_roms.add_coastal_bathy(g)
        g.write_ugrid(ugrid_file)
    else:
        g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
        g_shp='derived/matched_grid_v01.shp'
        if not os.path.exists(g_shp):
            g.write_edges_shp(g_shp)
else: # Spliced grid generated in splice_grids.py
    ugrid_file='spliced_grids_01_bathy.nc'
    g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
        
## 

# Identify ocean boundary edges
# Limit the boundary edges to edges which have a real cell on the other
# side in the ROMS output

# This is picking up some extra points in the spliced grid.
# So limit the edges we consider to edges which are exactly from
# ROMS.  That should be safe.
if 'edge_src' in g.edges.dtype.names:
    candidates=np.nonzero(g.edges['edge_src']==2)[0] # ROMS edges
else:
    candidates=None # assume it's a ROMS-only grid, all edges are from ROMS.
ca_roms.annotate_grid_from_data(g,run_start,run_stop,candidate_edges=candidates)

boundary_edges=np.nonzero( g.edges['src_idx_out'][:,0] >= 0 )[0]

## 

# To get lat/lon info
src=xr.open_dataset(ca_roms_files[0])

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

##

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

# Pre-extract some fields from the ROMS data
# Might move this to ca_roms.py in the future
# This is pretty slow
extracted=[]
lat_da=xr.DataArray(g.edges['src_idx_out'][boundary_edges,0],dims='boundary')
lon_da=xr.DataArray(g.edges['src_idx_out'][boundary_edges,1],dims='boundary')

for ca_roms_file in ca_roms_files:
    print(ca_roms_file)
    ds=xr.open_dataset(ca_roms_file)
    ds.load()
    
    # timestamps appear to be wrong in the files, always
    # holding 2009-01-02.
    # use the title, which appears to be consistent with the filename
    # and the true time
    t=utils.to_dt64( datetime.datetime.strptime(ds.title,'CA-%Y%m%d%H') )
    ds.time.values[0]=t
    
    sub_ds=ds.isel(time=0).isel(lat=lat_da,lon=lon_da)
    extracted.append(sub_ds)
    ds.close()
    
roms_at_boundary=xr.concat(extracted,dim='time')

## ----------

os.path.exists(old_bc_fn) and os.unlink(old_bc_fn)

# for adding in MSL => NAVD88 correction.  Just dumb luck that it's 1.0
dfm_zeta_offset=1.0

# average tidal prediction across boundary at start of simulation
# plus above DC offset
zeta_ic = dfm_zeta_offset + np.interp(utils.to_dnum(run_start),
                                      utils.to_dnum(otps_water_level.time),
                                      otps_water_level.result.mean(dim='site'))
mdu['geometry','WaterLevIni'] = zeta_ic


def write_pli(src_name,j,suffix):
    seg=g.nodes['x'][ g.edges['nodes'][j] ]
    src_feat=(src_name,seg,[src_name+"_0001",src_name+"_0002"])
    feat_suffix=dio.add_suffix_to_feature(src_feat,suffix)
    dio.write_pli(os.path.join(run_base_dir,'%s%s.pli'%(src_name,suffix)),
                  [feat_suffix])
    return feat_suffix

def write_tim(da,suffix,feat_suffix):
    # Write the data:
    columns=['elapsed_minutes']
    if da.ndim==1: # yuck pandas.
        df=da.to_dataframe().reset_index()
        df['elapsed_minutes']=(df.time.values - ref_date)/np.timedelta64(60,'s')
        columns.append(da.name)
    else:
        # it's a bit gross, but coercing pandas into putting a second dimension
        # into separate columns is too annoying.
        df=pd.DataFrame()
        df['elapsed_minutes']=(da.time.values - ref_date)/np.timedelta64(60,'s')
        for idx in range(da.shape[1]):
            col_name='val%d'%idx
            df[col_name]=da.values[:,idx]
            columns.append(col_name)

    if len(feat_suffix)==3:
        node_names=feat_suffix[2]
    else:
        node_names=[""]*len(feat_suffix[1])

    for node_idx,node_name in enumerate(node_names):
        # if no node names are known, create the default name of <feature name>_0001
        if not node_name:
            node_name="%s%s_%04d"%(src_name,suffix,1+node_idx)

        tim_fn=os.path.join(run_base_dir,node_name+".tim")
        df.to_csv(tim_fn, sep=' ', index=False, header=False, columns=columns)

def write_t3d(da,suffix,feat_suffix,edge_depth,quantity='salinity'):
    """
    Write a 3D boundary condition for a feature from ROMS data
     - most of the time writing boundaries is here
    """
    # Luckily the ROMS output does not lose any surface cells - so we don't
    # have to worry about a surface cell in roms_at_boundary going nan.
    assert da.ndim==2
    
    # get the depth of the internal cell:
    valid_depths=np.all( np.isfinite( da.values ), axis=da.get_axis_num('time') )
    valid_depth_idxs=np.nonzero(valid_depths)[0]

    # Roms values count from the surface, positive down.
    # but DFM wants bottom-up.
    # limit to valid depths, and reverse the order at the same time
    roms_depth_slice=slice(valid_depth_idxs[-1],None,-1)

    # This should be the right numbers, but reverse order
    sigma = (-edge_depth - da.depth.values[roms_depth_slice]) / -edge_depth
    sigma_str=" ".join(["%.4f"%s for s in sigma])
    elapsed_minutes=(da.time.values - ref_date)/np.timedelta64(60,'s')
    
    ref_date_str=utils.to_datetime(ref_date).strftime('%Y-%m-%d %H:%M:%S')
    
    # assumes there are already node names
    node_names=feat_suffix[2]

    t3d_fns=[os.path.join(run_base_dir,node_name+".t3d")
             for node_idx,node_name in enumerate(node_names) ]

    assert da.dims[0]=='time' # for speed up of direct indexing
    
    # Write the first, then copy it to the second node
    with open(t3d_fns[0],'wt') as fp:
        fp.write("\n".join([
            "LAYER_TYPE=sigma",
            "LAYERS=%s"%sigma_str,
            "VECTORMAX=1", # default, but be explicit
            "quant=%s"%quantity,
            "quantity1=%s"%quantity,
            "# start of data",
            ""]))
        for ti,t in enumerate(elapsed_minutes):
            fp.write("TIME=%g minutes since %s\n"%(t,ref_date_str))
            #data=" ".join( ["%.3f"%v for v in da.isel(time=ti).values[roms_depth_slice]] )
            # Faster direct indexing:
            data=" ".join( ["%.3f"%v for v in da.values[ti,roms_depth_slice]] )
            fp.write(data)
            fp.write("\n")
            
    for t3d_fn in t3d_fns[1:]:
        shutil.copyfile(t3d_fns[0],t3d_fn)


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
        veloc_normal=g.edges['bc_norm_in'][j,0]*veloc_u + g.edges['bc_norm_in'][j,1]*veloc_v
        
    if 1: # ROMS:        
        if 1: # Add ROMS zeta to waterlevel
            roms_water_level=roms_at_boundary.zeta.isel(boundary=ji)
    
            # 36h cutoff with 6h data.
            # This will have some filtfilt trash at the end, probably okay
            # at the beginning
            roms_water_level.values[:] = filters.lowpass(roms_water_level.values,
                                                         cutoff=36.,order=4,dt=6)

            # As far as I know, ROMS zeta is relative to MSL
            roms_interp=np.interp( utils.to_dnum(water_level.time),
                                   utils.to_dnum(roms_water_level.time),
                                   roms_water_level.values )
            water_level.values += roms_interp
            
        if 1: # salinity, temperature
            if 1: # proper spatial variation:
                salinity_3d=roms_at_boundary.isel(boundary=ji).salt
                temperature_3d=roms_at_boundary.isel(boundary=ji).temp
                
                for zi in range(len(salinity_3d.depth)):
                    salinity_3d.values[:,zi] = filters.lowpass(salinity_3d.values[:,zi],
                                                               cutoff=36,order=4,dt=6)
                    temperature_3d.values[:,zi] = filters.lowpass(temperature_3d.values[:,zi],
                                                                  cutoff=36,order=4,dt=6)

            else: # spatially constant
                salinity_3d=roms_at_boundary.salt.mean(dim='boundary')
                for zi in range(len(salinity_3d.depth)):
                    salinity_3d.values[:,zi] = filters.lowpass(salinity_3d.values[:,zi],
                                                               cutoff=36,order=4,dt=6)
                
    # try to include velocity here, too.
    # page 124 of the user manual:
    # zeta = 2*zeta_b - sqrt(H/g)*u - zeta_0
    # zeta_0 is initial water level, aka zeta_ic
    # if zeta is what we want the water level to be,
    # and zeta_b is what we give to DFM, then
    # zeta_b=0.5*( zeta+zeta_0 + sqrt(H/g)*u)
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

    if 1: # Riemann only in shallow areas:
        if depth<-200:
            forcing_data.append( ('waterlevelbnd',water_level,'_ssh') )
        else:
            forcing_data.append( ('riemannbnd',riemann,'_rmn') )
            

    for quant,da,suffix in forcing_data:
        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=%s"%quant,
                   "FILENAME=%s%s.pli"%(src_name,suffix),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

        feat_suffix=write_pli(src_name,j,suffix)
        if da.ndim==1:
            write_tim(da,suffix,feat_suffix)
        elif da.ndim==2:
            write_t3d(da,suffix,feat_suffix,edge_depth[j],
                      quantity=quant.replace('bnd','') )
            
    if 1: # advected velocity is 0 in attempt for most stable
        quant='uxuyadvectionvelocitybnd'
        suffix='_uv'
        
        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=%s"%quant,
                   "FILENAME=%s%s.pli"%(src_name,suffix),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))
        feat_suffix=write_pli(src_name,j,suffix)
            
        times=np.array( [run_start-np.timedelta64(1,'D'),
                         run_stop+np.timedelta64(1,'D')] )
        da=xr.DataArray( np.zeros( (len(times),2) ),
                         dims=['time','two'],
                         coords={'time':times} )
        write_tim(da,suffix,feat_suffix)

## 

# Write spatially-variable horizontal eddy viscosity field
# with background value of 10, and 1000 near
# the boundary.

ca_roms.add_sponge_layer(mdu,run_base_dir,g,boundary_edges,
                         sponge_visc=1000,
                         background_visc=10,
                         sponge_L=25000,quantity='viscosity')

# For diffusivity, have to use smaller background number so that
# freshwater inflows are not driven by diffusion
ca_roms.add_sponge_layer(mdu,run_base_dir,g,boundary_edges,
                         sponge_visc=1000,
                         background_visc=0.001,
                         sponge_L=25000,quantity='diffusivity')

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

if include_fresh: # disable while using ROMS-only grid
    # ---------SF FRESH, POTW, DELTA
    # 
    # SF Bay Freshwater and POTW, copied from sfb_dfm_v2:
    # features which have manually set locations for this grid
    # Borrow files from sfb_dfm_v2 -- should switch to submodules
    sfb_dfm_v2_base_dir="../../sfb_dfm_v2"
    adjusted_pli_fn = os.path.join(sfb_dfm_v2_base_dir,'nudged_features.pli')

    if 1: # Transcribe to shapefile for debuggin/vis
        from shapely import geometry
        from stompy.spatial import wkb2shp
        adj_pli_feats=dio.read_pli(adjusted_pli_fn)
        names=[feat[0] for feat in adj_pli_feats]
        geoms=[geometry.Point(feat[1].mean(axis=0)) for feat in adj_pli_feats]
        wkb2shp.wkb2shp('derived/input_locations.shp',geoms,fields={'name':names},
                        overwrite=True)

    dredge_depth=-1

    # kludge - wind the clock back a bit:
    print("TOTAL KLUDGE ON FRESHWATER")
    from sfb_dfm_utils import sfbay_freshwater

    # This will pull freshwater data from 2012, where we already
    # have a separate run which kind of makes sense
    time_offset=np.datetime64('2012-01-01') - np.datetime64('2017-01-01') 

    sfbay_freshwater.add_sfbay_freshwater(run_base_dir,
                                          run_start,run_stop,ref_date,
                                          adjusted_pli_fn,
                                          freshwater_dir=os.path.join(sfb_dfm_v2_base_dir, 'sfbay_freshwater'),
                                          grid=g,
                                          dredge_depth=dredge_depth,
                                          old_bc_fn=old_bc_fn,
                                          all_flows_unit=False,
                                          time_offset=time_offset)
                     
##

if include_fresh:
    # POTW inputs:
    # The new-style boundary inputs file (FlowFM_bnd_new.ext) cannot represent
    # sources and sinks, so these come in via the old-style file.
    potw_dir=os.path.join(sfb_dfm_v2_base_dir,'sfbay_potw')
    from sfb_dfm_utils import sfbay_potw

    sfbay_potw.add_sfbay_potw(run_base_dir,
                              run_start,run_stop,ref_date,
                              potw_dir,
                              adjusted_pli_fn,
                              g,dredge_depth,
                              old_bc_fn,
                              all_flows_unit=False,
                              time_offset=time_offset)

    # Delta boundary conditions
    # may need help with inputs-static
    from sfb_dfm_utils import delta_inflow

    delta_inflow.add_delta_inflow(run_base_dir,
                                  run_start,run_stop,ref_date,
                                  static_dir=os.path.join(sfb_dfm_v2_base_dir,"inputs-static"),
                                  grid=g,dredge_depth=dredge_depth,
                                  old_bc_fn=old_bc_fn,
                                  all_flows_unit=False,
                                  time_offset=time_offset)


# ---------- END SF FRESH, POTW, DELTA

##

if 1:
    mdu['geometry','NetFile'] = os.path.basename(ugrid_file).replace('.nc','_net.nc')
    dfm_grid.write_dfm(g,os.path.join(run_base_dir,mdu['geometry','NetFile']),
                       overwrite=True)

# This step is pretty slow the first time around.
coamps.add_coamps_to_mdu(mdu,run_base_dir,g,use_existing=True)


##

# Setting a full 3D initial condition requires a partitioned
# run, so go ahead partition now:

nprocs=16


def dflowfm(mdu_fn,args=['--autostartstop']):
    cmd=[os.path.join(dfm_bin_dir,"dflowfm")] + args
    if mdu_fn is not None:
        cmd.append(os.path.basename(mdu_fn))

    if nprocs>1:
        cmd=["%s/mpiexec"%dfm_bin_dir,"-n","%d"%nprocs] + cmd

    # This is more backwards compatible than 
    # passing cwd to subprocess()
    pwd=os.getcwd()
    try:
        os.chdir(run_base_dir)
        res=subprocess.call(cmd)
    finally:
        os.chdir(pwd)
    return res


def partition_grid():
    if nprocs<=1:
        return

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

    # This will not pick up on the map output being older than the partitioned grid!
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
                missing_val=20

            snap=src.isel(time=[0])

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

                    # Salinity:
                    #  fill missing values in ic_map.sa1 with the 2D extrapolated data
                    salt_fill_2d=salt_extrap_field.interpolate(all_xy,interpolation='nearest') # 2-3 seconds
                    salt_fill_3d=xr.DataArray(salt_fill_2d,dims=['nFlowElem'])
                    _,salt_fill_3dx=xr.broadcast(ic_map.sa1,salt_fill_3d)
                    sa1=ic_map.sa1
                    new_sa1=sa1.where(sa1!=missing_val,other=salt_fill_3dx)
                    ic_map['sa1']=new_sa1

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

