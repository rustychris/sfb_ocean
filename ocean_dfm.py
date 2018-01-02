"""
Driver script for coastal-ocean scale DFM runs, initially
to support microplastics project
"""

import subprocess
import os
import shutil
import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy.spatial import proj_utils

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
run_name='short_test_06'

run_base_dir=os.path.join('runs',run_name)
os.path.exists(run_base_dir) or os.makedirs(run_base_dir)

mdu=dio.MDUFile('template.mdu')

mdu['geometry','Kmx']=20
mdu['geometry','SigmaGrowthFactor']=1 
mdu['geometry','StretchType']=1 # user defined 
# These must sum exactly to 100.
mdu['geometry','StretchCoef']="8 8 7 7 6 6 6 6 5 5 5 5 5 5 5 5 2 2 1 1"

run_start=ref_date=np.datetime64('2017-08-10')
#run_stop=np.datetime64('2017-09-10')
run_stop=np.datetime64('2017-08-20') # start shorter in 3D

mdu.set_time_range(start=run_start,
                   stop=run_stop,
                   ref_date=ref_date)

old_bc_fn = os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])
## 

from sfb_dfm_utils import ca_roms, coamps

# Get the ROMS inputs:
ca_roms_files = ca_roms.fetch_ca_roms(run_start,run_stop)

## 
ugrid_file='derived/matched_grid_v00.nc'

if not os.path.exists(ugrid_file):
    g=ca_roms.extract_roms_subgrid()
    ca_roms.add_coastal_bathy(g)
    g.write_ugrid(ugrid_file)
else:
    g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
    
## 

# Identify ocean boundary edges
# Limit the boundary edges to edges which have a real cell on the other
# side in the ROMS output
ca_roms.annotate_grid_from_data(g,run_start,run_stop)

boundary_edges=np.nonzero( g.edges['src_idx_out'][:,0] >= 0 )[0]

## 

# # Need a source dataset to get an idea of what's where
# src=xr.open_dataset('local/concat-ca_subCA_das.nc')
# # estimate layer thicknesses
# dz=utils.center_to_interval(src.depth.values)
# src['dz']=('depth',),dz

def roms_davg(val):
    dim=val.get_axis_num('depth')
    dz=utils.center_to_interval(val.depth.values)
    weighted= np.nansum( (val*dz).values, axis=dim )
    unit = np.sum( np.isfinite(val)*dz, axis=dim)
    return weighted / unit

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

##

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

def write_t3d(da,suffix,feat_suffix,edge_depth):
    """
    Write a 3D boundary condition for a feature from ROMS data
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
    sigma = (-depth - da.depth.values[roms_depth_slice]) / -depth
    sigma_str=" ".join(["%.4f"%s for s in sigma])
    elapsed_minutes=(da.time.values - ref_date)/np.timedelta64(60,'s')
    
    ref_date_str=utils.to_datetime(ref_date).strftime('%Y-%m-%d %H:%M:%S')
    
    # assumes there are already node names
    node_names=feat_suffix[2]

    for node_idx,node_name in enumerate(node_names):
        t3d_fn=os.path.join(run_base_dir,node_name+".t3d")
        with open(t3d_fn,'wt') as fp:
            fp.write("\n".join([
                "LAYER_TYPE=sigma",
                "LAYERS=%s"%sigma_str,
                "VECTORMAX=1", # default, but be explicit
                "quant=salinity",
                "quantity1=salinity",
                "# start of data",
                ""]))
            for ti,t in enumerate(elapsed_minutes):
                fp.write("TIME=%g minutes since %s\n"%(t,ref_date_str))
                data=" ".join( ["%.3f"%v for v in da.isel(time=ti).values[roms_depth_slice]] )
                fp.write(data)
                fp.write("\n")


for ji,j in enumerate(boundary_edges):
    if j in [99]:
        print("EDGE %d might be a bad apple.  Skip"%j)
        continue
    
    src_name='oce%05d'%j
    print(src_name)
    
    depth=g.edges['edge_depth'][j]

    if 1: # bring in OTPS harmonics:
        water_level=dfm_zeta_offset + otps_water_level.result.isel(site=ji)

        veloc_u=otps_u.result.isel(site=ji)
        veloc_v=otps_v.result.isel(site=ji)
        veloc_uv=xr.DataArray(np.c_[veloc_u.values,veloc_v.values],
                              coords=[('time',veloc_u.time),('comp',['e','n'])])
        veloc_uv.name='uv'

        # inward-positive
        veloc_normal=g.edges['bc_norm_in'][j,0]*veloc_u + g.edges['bc_norm_in'][j,1]*veloc_v

        # 
        riemann=water_level
        if 1: # try to include velocity here, too.
            # page 124 of the user manual:
            # zeta = 2*zeta_b - sqrt(H/g)*u - zeta_0
            # zeta_0 is initial water level, aka zeta_ic
            # if zeta is what we want the water level to be,
            # and zeta_b is what we give to DFM, then
            # zeta_b=0.5*( zeta+zeta_0 + sqrt(H/g)*u)
            
            riemann=0.5*(riemann + zeta_ic + np.sqrt(np.abs(depth)/9.8)*veloc_normal)
        
    if 1: # ROMS:
        if 0: 
            water_level=src.zeta.isel(lat=lat_idx_out,lon=lon_idx_out)
    
            # 36h cutoff with 6h data.
            # This will have some filtfilt trash at the end, probably okay
            # at the beginning
            water_level.values[:] = filters.lowpass(water_level.values,
                                                    cutoff=36.,order=4,dt=6)
        if 1: # salinity
            # HERE - need to extract this from the individual files
            salinity_3d=roms_at_boundary.isel(boundary=ji).salt
            salinity=roms_davg(salinity_3d)
            
            salinity.values[:] = filters.lowpass(salinity.values,
                                                 cutoff=36.,order=4,dt=6)
            salinity.name='salinity'
        
    assert np.all( np.isfinite(water_level.values) )

    forcing_data=[]

    #('waterlevelbnd',water_level,'_ssh'),
    #('riemannbnd',riemann,'_rmn'),
    #('salinitybnd',salinity,'_salt'),
    #('uxuyadvectionvelocitybnd',veloc_uv,'_uv'),
    #('velocitybnd',veloc_normal,'_vel'),
    #('temperaturebnd',water_temp,'_temp')
    
    if int(mdu['physics','Salinity']):
        # forcing_data.append( ('salinitybnd',salinity,'_salt') )
        forcing_data.append( ('salinitybnd',salinity_3d,'_salt') )

    if 1: # riemann only
        # This works pretty well, good agreement at Point Reyes
        forcing_data.append( ('riemannbnd',riemann,'_rmn') )
    if 0:  # water level only
        forcing_data.append( ('waterlevelbnd',water_level,'_ssh') )

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
            write_t3d(da,suffix,feat_suffix,g.edges['edge_depth'][j])
            

## 

if 1:
    mdu['geometry','NetFile'] = 'matched_grid_v00_net.nc'
    dfm_grid.write_dfm(g,os.path.join(run_base_dir,mdu['geometry','NetFile']),
                       overwrite=True)
## 

# This step is pretty slow
coamps.add_coamps_to_mdu(mdu,run_base_dir,g,use_existing=True)

##

mdu['restart','RestartFile']='../../initial_conditions_map.nc'
mdu['restart','RestartDateTime'] ="201708100001"

##

mdu_fn=os.path.join(run_base_dir,run_name+".mdu")
mdu.write(mdu_fn)

##

# Run it.
subprocess.call([os.path.join(dfm_bin_dir,"dflowfm"),
                 "--autostartstop",
                 os.path.basename(mdu_fn)],
                cwd=run_base_dir)



##


# What would a vertical profile look like?
snap=src # snapshot in time, full domain

lat_slice=slice( g.cells['lati'].min(),g.cells['lati'].max()+1 )
lon_slice=slice( g.cells['loni'].min(),g.cells['loni'].max()+1 )

sub_salt=src.salt.isel(lat=lat_slice,lon=lon_slice)

bSalt,bDepth=xr.broadcast(sub_salt,sub_salt.depth)

all_salt=bSalt.values.ravel()
all_depth=-bDepth.values.ravel()
valid=np.isfinite(all_salt)
all_salt=all_salt[valid]
all_depth=all_depth[valid]

plt.figure(21).clf()
fig,ax=plt.subplots(num=21)
ax.plot(all_salt,all_depth,'k.')

##

# Using a restart file.
# Ran into issues using a proper restart file, but works okay to read in
# a map file.
# So read one step of a map file


def set_ic_from_map_output(map_file,output_fn='initial_conditions_map.nc'):
    """
    copy the strucutre of the map_file at one time step, but overwrite
    fields with ROMS snapshot data
    """
    map_in=xr.open_dataset(map_file)

    map_out=map_in.isel(time=[0])
    # there are some name clashes -- drop any coordinates attributes

    ## 
    for dv in map_out.data_vars:
        if 'coordinates' in map_out[dv].attrs:
            del map_out[dv].attrs['coordinates']


    ##
    # Overwrite salinity with data from ROMS:

    # DFM reorders the cells, so read it back in.
    g_map=dfm_grid.DFMGrid(map_out)
    g_map_cc=g_map.cells_centroid()
    g_map_ll=utm2ll(g_map_cc)

    ## 
    dlon=np.median(np.diff(snap.lon.values))
    dlat=np.median(np.diff(snap.lat.values))
    for c in g_map.valid_cell_iter():
        if c%1000==0:
            print("%d/%d"%(c,g_map.Ncells()))
        loni=utils.nearest(snap.lon.values%360,g_map_ll[c,0]%360)
        lati=utils.nearest(snap.lat.values,g_map_ll[c,1])
        lon_err=(snap.lon.values[loni]%360 - g_map_ll[c,0]%360.0)/dlon
        lat_err=(snap.lat.values[lati] - g_map_ll[c,1])/dlat

        if abs(lon_err) > 1 or abs(lat_err)>1:
            print("skip cell %d"%c)
            continue

        roms_salt=snap.salt.isel(lat=lati,lon=loni,time=0).values[::-1]
        roms_z=-snap.depth.values[::-1]
        valid=np.isfinite(roms_salt)
        roms_salt=roms_salt[valid]
        roms_z=roms_z[valid]

        bl=map_out.FlowElem_bl.isel(nFlowElem=c).values # positive up from the z datum
        wd=map_out.waterdepth.isel(nFlowElem=c,time=0).values
        sigma=map_out.LayCoord_cc.values
        dfm_z=bl + wd*sigma

        dfm_salt=np.interp(dfm_z,roms_z,roms_salt)
        sel_time=xr.DataArray([0],dims=['time'])
        sel_cell=xr.DataArray([c],dims=['nFlowElem'])
        map_out.sa1[sel_time,sel_cell]=dfm_salt


    # Does the map timestamp have to match what's in the mdu?
    # Yes.
    map_out.to_netcdf(output_fn,format='NETCDF3_64BIT')
    return map_out

map_file='runs/short_test_06/DFM_OUTPUT_short_test_06/short_test_06_map.nc'
map_out=set_ic_from_map_output(map_file,'initial_conditions_map.nc')

# before, when choosing a later time step, it worked okay to encode the datetime
# in the mdu as YYYYMMDDHHMM
# but when using the first timestep, that failed until I requested instead
# 201708100001
# which DFM complains about but seems to use without any trouble.
