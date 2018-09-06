# Revisiting 2018-06-28

# Check the BC fluxes, namely to get a sense of whether velocities
# are being included as expected.
#
# Run `short_22` has exactly one boundary with forcing, `ji=10`, which
# is on the northern side of the grid, N-S exchange, on a ragged corner,
# not far from the NW corner of the grid.
#
# The unorm values match with the expected forcing on this edge.
#
# The next question is whether the observed lack of match before was from
# having multiple velocities defined on a single cell, or some other cause.

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

utm2ll=proj_utils.mapper('EPSG:26910','WGS84')
ll2utm=proj_utils.mapper('WGS84','EPSG:26910')

mdu=dio.MDUFile('template.mdu')

# short_21: now OTPS velocities, no freesurface BCs
# short_22: a single velocity BC
# short_23: each cell gets at most one velocity BC
# run_name="short_21"

# include_fresh=False # or True
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


# In[99]:


run_base_dir=os.path.join('runs',run_name)
os.path.exists(run_base_dir) or os.makedirs(run_base_dir)

mdu.set_filename(os.path.join(run_base_dir,run_name+".mdu"))

run_start=ref_date=np.datetime64('2017-07-01')
run_stop=np.datetime64('2017-07-15')

mdu.set_time_range(start=run_start,
                   stop=run_stop,
                   ref_date=ref_date)

old_bc_fn = os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])

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


# In[100]:


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
# otps_U=otps.reconstruct(U_harmonics,otps_times)
# otps_V=otps.reconstruct(V_harmonics,otps_times)

# convert cm/s to m/s
otps_u.result[:] *= 0.01 
otps_v.result[:] *= 0.01


## 

if 'edge_depth' in g.edges.dtype.names:
    edge_depth=g.edges['edge_depth']
else:
    edge_depth=g.nodes['depth'][ g.edges['nodes'] ].mean(axis=1)

sign='regular'

# looks like I blew away ds22.
# map_ds21=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_21/DFM_OUTPUT_short_21/short_21_map.nc')
# map_ds22=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_22/DFM_OUTPUT_short_22/short_22_map.nc')
# map_ds23=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_23/DFM_OUTPUT_short_23/short_23_map.nc')
# map_ds24=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_24/DFM_OUTPUT_short_24/short_24_map.nc')

map_ds25=xr.open_dataset('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_25/DFM_OUTPUT_short_25/short_25_map.nc')

## 


def calc_veloc_normal(ji):
    j=boundary_edges[ji]
    veloc_u=otps_u.result.isel(site=ji)
    veloc_v=otps_v.result.isel(site=ji)
    veloc_uv=xr.DataArray(np.c_[veloc_u.values,veloc_v.values],
                          coords=[('time',veloc_u.time),('comp',['e','n'])])
    veloc_uv.name='uv'

    # inward-positive
    veloc_normal=(g.edges['bc_norm_in'][j,0]*veloc_u + g.edges['bc_norm_in'][j,1]*veloc_v)
    if sign=='reverse':
        veloc_normal*=-1
    return veloc_normal

# 
# # in short_22, ji==10 is the only bc link
# ji=10 # np.argmin( utils.dist(boundary_edge_xy - southward ) )
# j=boundary_edges[ji]
# 
# fig,ax=plt.subplots(figsize=(15,10))
# 
# g.plot_edges(ax=ax,lw=0.5,color='k',zorder=-1)
# 
# boundary_edge_xy=g.edges_center()[boundary_edges]
# 
# ax.text( boundary_edge_xy[ji,0],
#          boundary_edge_xy[ji,1],
#          'ji=%d'%ji)
# 
# x0,y0=boundary_edge_xy[ji,:]
# zoom=[x0-10e3,x0+10e3,y0-10e3,y0+10e3]
# ax.axis(zoom)
# 
# # What link is this?
# # if chosen based on coordinate:
# # link_i = np.argsort( utils.dist( link_xy - southward_element) )[:2]
# link_i = np.argsort( utils.dist( link_xy - boundary_edge_xy[ji]))[:2]
# for li in link_i:
#     ax.text( link_xy[li,0],
#              link_xy[li,1],
#              'Link %d'%li)
#     
# # based on the plot, then it's clear which is the boundary link,
# # since it plots in the middle of the cell.
# link=7312
# 

##


# Difficult to do this graphically, and hard to match the links up
# 1:1.  However, it would be enough to just get the right flux into
# each cell, and that should be unambiguous.

# select the file to test:
if 0:
    map_ds=map_ds22
    def test_boundary(ji):
        return ji==10
if 0:
    map_ds=map_ds21
    def test_boundary(ji):
        return True
if 0:
    map_ds=map_ds23
    def test_boundary(ji):
        return True

if 0:
    map_ds=map_ds24
    def test_boundary(ji):
        return True

if 1:
    map_ds=map_ds25
    def test_boundary(ji):
        return True
    
map_time_i=len(map_ds.time)-1


## 

unorms=map_ds.unorm.isel(time=map_time_i,laydim=-1) # nFlowLink
FlowLink_from=map_ds.FlowLink.values[:,0] - 1 # make 0-based
FlowLink_to  =map_ds.FlowLink.values[:,1] - 1 

elt_xy = np.c_[ map_ds.FlowElem_xzw.values, map_ds.FlowElem_yzw.values]

# All 'to' indices are real elements.
# from indices include boundaries

to_xy = elt_xy[ FlowLink_to ]
from_xy=to_xy.copy()
is_bc=FlowLink_from >= len(elt_xy)
from_xy[ ~is_bc ] = elt_xy[ FlowLink_from[~is_bc] ]

link_xy=np.c_[ map_ds.FlowLink_xu.values,
               map_ds.FlowLink_yu.values ]

link_norms= utils.to_unit( to_xy - link_xy)

e2c=g.edge_to_cells(recalc=True)

bc_elems=np.unique(FlowLink_to[is_bc])

##

# 1. Find all cells which have BC exchanges, based on map_ds

# 2. Loop over those, and collect both bc links, and boundary_edges
#    involved with that cell.

# in ds21:
# bc_elems[0]: scaling maybe by -2?
# bc_elems[1]: scaling by 1.3 ?
# in ds23:
# bc_elems[0]: looks great, and has two OTPS edges
# bc_elems[1]: scaled by -0.5?  but not exactly.  
# bc_elems[2]: minor error, generally good.
# bc_elems[3]: this only has 1 otps edge, but unorm is way off.

# in ds24, where 0.001 should be the first boundary cell, 0.002 the second, etc.
# bc_elems[3]: should have a flow of 0.0040, but really 0.001033
# bc_elems[0]: good match to 0.001
bc_elem=bc_elems[4]

print("Internal element:",bc_elem)

plt.figure(3).clf()
fig3,ax3=plt.subplots(figsize=(10,5),num=3)

match_bc_links=np.nonzero( (FlowLink_to==bc_elem)&is_bc )[0]
# display basic info on those links
influx_data=0
assert len(match_bc_links)==1,"Not critical just reminder"

for bc_link in match_bc_links:
    to_elt=FlowLink_to[bc_link]
    from_elt=FlowLink_from[bc_link]
    print("  link %d is BC  %d(%d)=>%d"%(bc_link,from_elt,
                                       len(map_ds.nFlowElem)-1-from_elt,to_elt))
    #print("      internal cell ctr: %s"%(elt_xy[to_elt]))
    this_link=map_ds.unorm.isel(nFlowLink=bc_link)
    # These are reliably all about the same
    #ax3.plot( utils.to_dnum(this_link.time),this_link.values,
    #         label='bc_link=%d'%bc_link)
    influx_data=influx_data + this_link
    print("  flow is %.5f"%(this_link.isel(time=-1).mean()))


# now on the input side:
# first, select the cell based on center, just to be sure

cell_i = g.select_cells_nearest(elt_xy[bc_elem])
cell_xy = g.cells_centroid([cell_i])[0]
delta=utils.dist(cell_xy - elt_xy[to_elt])
print("  match element %d to grid cell %d with dist %g"%(bc_elem,cell_i,delta))
assert delta<1000.0 # so far <<1.0

# load the time series from file:
element_tim=np.loadtxt('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_23/oce%05d_un_0001.tim'%cell_i)
element_tim_dn=utils.to_dnum(mdu.time_range()[0])+element_tim[:,0]*(60./86400)

# go from that cell index to the boundary edges:
bc_jis,sides=np.nonzero( e2c[boundary_edges] == cell_i )
bc_js=boundary_edges[bc_jis]
assert np.all(sides==0) # not really necessary, but weird if it fails

print("  Which participates in boundary edges j=%s"%( ",".join(["%d"%j for j in bc_js])))

active=np.array([test_boundary(ji) for ji in bc_jis])
print("  of which the active are j=%s"%( ",".join(["%d"%j for j in bc_js[active]])))

veloc_normal_sum=0.0
for ji in bc_jis[active]:
    print("  including OTPS flow for ji=%d  j=%d"%(ji,boundary_edges[ji]))
    veloc_normal=calc_veloc_normal(ji)
    #ax3.plot( utils.to_dnum(veloc_normal.time),veloc_normal, 'b-',
    #        label='OTPS ji=%d'%ji)
    veloc_normal_sum = veloc_normal_sum + veloc_normal

influx_mean=influx_data.mean(dim='laydim')


# Plot the sums
ax3.plot( utils.to_dnum(veloc_normal_sum.time), veloc_normal_sum,'r-',zorder=-0.5,lw=2.5,
          label="Sum of OTPS")

ax3.plot( utils.to_dnum(influx_data.time),influx_mean.values,'g-',
          label="mean_z of sum of unorms")

ax3.plot( element_tim_dn, element_tim[:,1],'b-',label='TIM data')
ax3.axis( xmin=736510., xmax=736514.)
ax3.legend()


# In[131]:


# So OTPS and unorm do not match for bc_elems[3] == 4
# That's link 7315, the boundary element -4 (aka 3750),
# to the internal element 4.
# matches cell 4 in the grid, and edge j=14.

# What does this look like in the forcing file?
# as it should, this edge matches well with oce00004_un.pli,
# which are now labeled by element.

# The tim file matches the OTPS data.  So there must be a disconnect
# in how this is routed through DFM, and/or how unorm is output.

# The edge coordinate is ...
g.edges_center()[14]


# In[141]:

# Are pli's overwriting each other? Nope, those all look legit.
import glob
import stompy.model.delft.io as dio

pli_fns=glob.glob('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_23/oce0*_un.pli')
segs=[dio.read_pli(pli_fn)[0][1] for pli_fn in pli_fns]


## 

# As a test of the validity of unorm, at least for internal edges, compare unorms
# for an internal cell to the cell-center velocity.

plt.figure(10).clf()
fig,ax=plt.subplots(num=10)

g.plot_edges(ax=ax,color='k',lw=0.5)

# deep
#  pnt=np.array([436012.54295528703, 4128316.0099351429])
#  zoom=(429210.81260892103, 442143.73724100599, 4123090.9662331804, 4133040.4347458566)
# shelf:
# pnt=np.array([510590, 4168156.])
# zoom=(493809.81802497664, 525650.46758594736, 4156005.7799958498, 4180501.2092388305)

# SW corner:
zoom=(474508.07510420476, 511999.90329425375, 4009899.1682620877, 4038742.1233299724)

#elem_sel=np.argmin( utils.dist( pnt - elt_xy ) )
#cell_sel=g.select_cells_nearest(pnt) # Matches.
#g.plot_cells(mask=[cell_sel],ax=ax)

ax.axis(zoom)

ti=len(map_ds.time)-1
lay_i=15

# unorms=map_ds.unorm.isel(time=ti,laydim=2)
elt_xy=np.c_[map_ds.FlowElem_xzw,
             map_ds.FlowElem_yzw]
elem_select= utils.within_2d(elt_xy,zoom)

scale=7e-5
cquiv = ax.quiver( elt_xy[elem_select,0],elt_xy[elem_select,1],
                   map_ds.ucx.isel(time=ti,laydim=lay_i).values[elem_select],
                   map_ds.ucy.isel(time=ti,laydim=lay_i).values[elem_select],
                   angles='xy',scale_units='xy',scale=scale)

unorms=map_ds.unorm.isel(time=ti,laydim=lay_i).values

link_xy=np.c_[ map_ds.FlowLink_xu.values,
               map_ds.FlowLink_yu.values ]

link_sel=utils.within_2d(link_xy,zoom)

equiv= ax.quiver( link_xy[link_sel,0],link_xy[link_sel,1],
                  link_norms[link_sel,0] * unorms[link_sel],
                  link_norms[link_sel,1] * unorms[link_sel],
                  angles='xy',scale_units='xy',scale=scale,
                  color='0.5')


for li in np.nonzero(link_sel)[0]:
    txt="%.4f"%unorms[li]

    li_to_ji= utils.dist( link_xy[li] - g.edges_center()[boundary_edges] )
    bc_ji=np.argmin(li_to_ji)
    if li_to_ji[bc_ji]<1000:
        print("Found a match from link to boundary within %.2f"%li_to_ji[bc_ji])
        
        #expected=(bc_ji+1)*0.001
        bc_i=np.nonzero( bc_elems==FlowLink_to[li] )[0][0]
        expected=(bc_i+1)*0.001
        txt+="\nexpect %.4f"%expected

    ax.text( link_xy[li,0],link_xy[li,1],txt,size=8)

##

# does BndLink help here?  BndLink just means boundary of the grid,
# not whether it is an open/closed boundary.
# it's 364 entries, ranging from 1 to 7676.
# that's the range of nNetLink (length 7676)


##

# Disregard unorm on boundary, and compare to normal velocity of
# first cell in

bc_elem=bc_elems[0]

print("Internal element:",bc_elem)

plt.figure(3).clf()
fig3,ax3=plt.subplots(figsize=(10,5),num=3)

match_bc_links=np.nonzero( (FlowLink_to==bc_elem)&is_bc )[0]
# display basic info on those links
influx_data=0
assert len(match_bc_links)==1,"Not critical just reminder"

for bc_link in match_bc_links:
    to_elt=FlowLink_to[bc_link]
    from_elt=FlowLink_from[bc_link]
    print("  link %d is BC  %d(%d)=>%d"%(bc_link,from_elt,
                                       len(map_ds.nFlowElem)-1-from_elt,to_elt))
    #print("      internal cell ctr: %s"%(elt_xy[to_elt]))
    this_link=map_ds.unorm.isel(nFlowLink=bc_link)
    # These are reliably all about the same
    #ax3.plot( utils.to_dnum(this_link.time),this_link.values,
    #         label='bc_link=%d'%bc_link)
    influx_data=influx_data + this_link
    print("  flow is %.5f"%(this_link.isel(time=-1).mean()))


# now on the input side:
# first, select the cell based on center, just to be sure

cell_i = g.select_cells_nearest(elt_xy[bc_elem])
cell_xy = g.cells_centroid([cell_i])[0]
delta=utils.dist(cell_xy - elt_xy[to_elt])
print("  match element %d to grid cell %d with dist %g"%(bc_elem,cell_i,delta))
assert delta<1000.0 # so far <<1.0

# load the time series from file:
element_tim=np.loadtxt('/hpcvol1/rusty/dfm/sfb_ocean/runs/short_23/oce%05d_un_0001.tim'%cell_i)
element_tim_dn=utils.to_dnum(mdu.time_range()[0])+element_tim[:,0]*(60./86400)

# go from that cell index to the boundary edges:
bc_jis,sides=np.nonzero( e2c[boundary_edges] == cell_i )
bc_js=boundary_edges[bc_jis]
assert np.all(sides==0) # not really necessary, but weird if it fails

print("  Which participates in boundary edges j=%s"%( ",".join(["%d"%j for j in bc_js])))

active=np.array([test_boundary(ji) for ji in bc_jis])
print("  of which the active are j=%s"%( ",".join(["%d"%j for j in bc_js[active]])))

veloc_normal_sum=0.0
for ji in bc_jis[active]:
    print("  including OTPS flow for ji=%d  j=%d"%(ji,boundary_edges[ji]))
    veloc_normal=calc_veloc_normal(ji)
    #ax3.plot( utils.to_dnum(veloc_normal.time),veloc_normal, 'b-',
    #        label='OTPS ji=%d'%ji)
    veloc_normal_sum = veloc_normal_sum + veloc_normal

influx_mean=influx_data.mean(dim='laydim')


# Plot the sums
ax3.plot( utils.to_dnum(veloc_normal_sum.time), veloc_normal_sum,'r-',zorder=-0.5,lw=2.5,
          label="Sum of OTPS")

ax3.plot( utils.to_dnum(influx_data.time),influx_mean.values,'g-',
          label="mean_z of sum of unorms")

ax3.plot( element_tim_dn, element_tim[:,1],'b-',label='TIM data')
ax3.axis( xmin=736510., xmax=736514.)
ax3.legend()

