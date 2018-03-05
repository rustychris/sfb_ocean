"""
See how well the OTPS fluxes close.

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
# short_21: now OTPS velocities, no freesurface BCsn
run_name="short_21"

include_fresh=False # or True
layers='z' # or 'sigma'
grid='ragged_coast' # 'rectangle_coast' 'ragged_full', ...
nprocs=1 # 16
mdu['physics','Temperature']=1
mdu['physics','Salinity']=1
use_wind=True

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
run_stop=np.datetime64('2017-07-15')

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
U_harmonics = Otps.extract_HC( boundary_out_ll, quant='u')
V_harmonics = Otps.extract_HC( boundary_out_ll, quant='v')

pad=np.timedelta64(2,'D')
otps_times=np.arange(run_start-pad, run_stop+pad,
                     np.timedelta64(600,'s'))
otps_water_level=otps.reconstruct(z_harmonics,otps_times)
otps_u=otps.reconstruct(u_harmonics,otps_times)
otps_v=otps.reconstruct(v_harmonics,otps_times)
otps_U=otps.reconstruct(U_harmonics,otps_times)
otps_V=otps.reconstruct(V_harmonics,otps_times)

# convert cm/s to m/s
otps_u.result[:] *= 0.01 
otps_v.result[:] *= 0.01

##

if 'edge_depth' in g.edges.dtype.names:
    edge_depth=g.edges['edge_depth']
else:
    edge_depth=g.nodes['depth'][ g.edges['nodes'] ].mean(axis=1)
                                                         
## 

if 0:
    sign='regular'
    map_ds=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_21_norm_orig/DFM_OUTPUT_short_21/short_21_map.nc')
else:
    sign='reverse'
    map_ds=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_21/DFM_OUTPUT_short_21/short_21_map.nc')
map_time_i=200

# 348
otps_time_i=np.searchsorted( otps_times, map_ds.time.isel(time=map_time_i) )

time_err=otps_times[otps_time_i] - map_ds.time.isel(time=map_time_i)
print("Time mismatch between output and OTPS is %.3fh"%( time_err/np.timedelta64(3600,'s')))

boundary_fluxes=np.zeros(len(boundary_edges),'f8')

for ji,j in enumerate(boundary_edges):
    src_name='oce%05d'%j
    # print(src_name)
    
    depth=edge_depth[j]

    if 1: # bring in OTPS harmonics:
        water_level=dfm_zeta_offset + otps_water_level.result.isel(site=ji)

        veloc_u=otps_u.result.isel(site=ji)
        veloc_v=otps_v.result.isel(site=ji)
        veloc_uv=xr.DataArray(np.c_[veloc_u.values,veloc_v.values],
                              coords=[('time',veloc_u.time),('comp',['e','n'])])
        veloc_uv.name='uv'

        # inward-positive
        # testing for reversed:
        veloc_normal=(g.edges['bc_norm_in'][j,0]*veloc_u + g.edges['bc_norm_in'][j,1]*veloc_v)
        if sign=='reverse':
            veloc_normal*=-1

        boundary_fluxes[ji]=veloc_normal.isel(time=otps_time_i)


# 


unorms=map_ds.unorm.isel(time=map_time_i,laydim=-1) # nFlowLink
FlowLink_from=map_ds.FlowLink.values[:,0] - 1 # make 0-based
FlowLink_to  =map_ds.FlowLink.values[:,1] - 1 

elt_x = map_ds.FlowElem_xzw.values
elt_y = map_ds.FlowElem_yzw.values
elt_xy=np.c_[elt_x,elt_y]

# All 'to' indices are real elements.
# from indices include boundaries

from_xy=np.zeros( (len(FlowLink_from),2), 'f8')
to_xy  =np.zeros( (len(FlowLink_from),2), 'f8')
is_bc=FlowLink_from >= len(elt_xy)
from_xy[ ~is_bc ] = elt_xy[ FlowLink_from[~is_bc] ]

to_xy[ :,: ] = elt_xy[ FlowLink_to ]
from_xy[ is_bc ]= to_xy[is_bc] 

link_xy=0.5*(from_xy+to_xy)
link_xy[ is_bc ] += (np.random.random( (is_bc.sum(),2)) - 0.5)*1000


link_norms=utils.to_unit( to_xy - from_xy )
link_norms[ np.isnan(link_norms) ] = 0

# Not a robust way to discern normals to go with boundary condition unorm values.
# can at least scatter plot them.

# 

if sign=='reverse':
    num=1
else:
    num=2

plt.figure(num).clf()
fig,ax=plt.subplots(num=num)
fig.suptitle('%s BCs'%sign)

g.plot_edges(ax=ax,lw=0.5,color='k',zorder=-1)

boundary_edge_xy=g.edges_center()[boundary_edges]


ax.quiver( boundary_edge_xy[:,0],boundary_edge_xy[:,1],
           boundary_fluxes*g.edges['bc_norm_in'][boundary_edges,0],
           boundary_fluxes*g.edges['bc_norm_in'][boundary_edges,1])
scat=ax.scatter( link_xy[:,0],link_xy[:,1],30,unorms,cmap='seismic')
scat.set_clim([-0.25,0.25])

plt.colorbar(scat)


ax.quiver( link_xy[:,0],link_xy[:,1],
           unorms * link_norms[:,0], unorms*link_norms[:,1],
           color='g')

ax.axis( (432345.21392444102,
          463446.01820448751,
          4263579.5847409964,
          4298346.4066975173) )

## 

# At first glance the BC velocities are reverse to what is in the output.
# Hard to be sure without knowing normals

# Plotting unorms * normals for the FlowLinks we can (non-BC), 
# seems that positive unorm is north and east

# the majority of the bc edges seem consistent with positive=in, and
# matching up with imposed OTPS velocities, but not all.


## 

# e.g. 
# an edge with an inward normal pointing south is at 
# This place looks like the forcing and the output are opposite.
if 0:
    southward=np.array( [421590.92978347285, 4270014.9450852871] )
    southward_element=np.array([421607.6316436067, 4268211.144190832])
else:
    southward=np.array( [447855.82588489528, 4286429.771834890])
    southward_element=np.array( [447812.06420383614, 4284723.066273585])


plt.figure(1).axes[0].text( southward[0],southward[1],'Southward')
plt.figure(1).axes[0].text( southward_element[0],southward_element[1],'Southward Elt')

# In the reversed case, it has OTPS velocities coming in,
# but the two blue dots show that dfm thinks flow is out.
# how do they look in time?

ji=np.argmin( utils.dist(boundary_edge_xy - southward ) )
j=boundary_edges[ji]

plt.figure(1).axes[0].text( boundary_edge_xy[ji,0],
                            boundary_edge_xy[ji,1],
                            'Southward pick')

veloc_u=otps_u.result.isel(site=ji)
veloc_v=otps_v.result.isel(site=ji)
veloc_uv=xr.DataArray(np.c_[veloc_u.values,veloc_v.values],
                      coords=[('time',veloc_u.time),('comp',['e','n'])])
veloc_uv.name='uv'

# inward-positive
# testing for reversed:
veloc_normal=(g.edges['bc_norm_in'][j,0]*veloc_u + g.edges['bc_norm_in'][j,1]*veloc_v)
if sign=='reverse':
    veloc_normal*=-1

# What link is this?
link_i = np.argsort( utils.dist( link_xy - southward_element) )[:2]
for li in link_i:
    plt.figure(1).axes[0].text( link_xy[li,0],
                                link_xy[li,1],
                                'Southward Link')


plt.figure(3).clf()

fig3,ax3=plt.subplots(num=3)
# These applied velocities ...
ax3.plot( utils.to_dnum(otps_times), veloc_normal,'r-' )
# are opposite these fluxes coming out
for li in link_i:
    ax3.plot( utils.to_dnum(map_ds.time),
              map_ds.unorm.isel(nFlowLink=li) )

# sure enough, at this point, these unorm values are opposite
# Is that true for other places?
# Here, they don't look like the same data, though.
# where 'here' is from the second set of coordinates up there.

