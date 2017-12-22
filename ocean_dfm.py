"""
Driver script for coastal-ocean scale DFM runs, initially
to support microplastics project
"""

import subprocess
import os
import shutil

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

dfm_bin_dir="/opt/software/delft/dfm/r52184-opt/bin"

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
run_name='short_test_05'

run_base_dir=os.path.join('runs',run_name)
os.path.exists(run_base_dir) or os.makedirs(run_base_dir)

mdu=dio.MDUFile('template.mdu')

# Set the start to src's start


run_start=ref_date=np.datetime64('2017-08-10')
run_stop=np.datetime64('2017-09-10')

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
# 
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

for ji,j in enumerate(boundary_edges):
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
        forcing_data.append( ('salinitybnd',salinity,'_salt') )

    if 0: # alternating water level and velocity
        if ji%2==0:
            forcing_data.append( ('waterlevelbnd',water_level,'_ssh') )
        else:
            forcing_data.append( ('velocitybnd',veloc_normal,'_vel') )

    if 1: # riemann only
        # This works pretty well, good agreement at Point Reyes
        forcing_data.append( ('riemannbnd',riemann,'_rmn') )

    if 0:  # water level only
        forcing_data.append( ('waterlevelbnd',water_level,'_ssh') )

    if 0: # mostly waterlevel, and some neumann in problem areas
        # This was disastrous on the southern edge.
        norm_in=g.edges['bc_norm_in'][j]
        math_angle=np.arctan2(*list(norm_in[::-1]))*180/np.pi % 360.0
        if (math_angle>250) and (math_angle<290):
            forcing_data.append( ('neumannbnd',0.0*water_level,'_dssh') )
        else:
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

        seg=g.nodes['x'][ g.edges['nodes'][j] ]
        src_feat=(src_name,seg,[src_name+"_0001",src_name+"_0002"])
        
        feat_suffix=dio.add_suffix_to_feature(src_feat,suffix)
        
        dio.write_pli(os.path.join(run_base_dir,'%s%s.pli'%(src_name,suffix)),
                      [feat_suffix])

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

## 

if 1:
    mdu['geometry','NetFile'] = 'matched_grid_v00_net.nc'
    dfm_grid.write_dfm(g,os.path.join(run_base_dir,mdu['geometry','NetFile']),
                       overwrite=True)
## 

coamps.add_coamps_to_mdu(mdu,run_base_dir,g,use_existing=True)


##

mdu_fn=os.path.join(run_base_dir,run_name+".mdu")
mdu.write(mdu_fn)

##

# Run it.
subprocess.call([os.path.join(dfm_bin_dir,"dflowfm"),
                 "--autostartstop",
                 os.path.basename(mdu_fn)],
                cwd=run_base_dir)

