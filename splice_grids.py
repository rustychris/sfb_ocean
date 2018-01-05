import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry

from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import field
from stompy.plot import plot_wkb
from stompy import utils
##

g_roms=unstructured_grid.UnstructuredGrid.from_ugrid('derived/matched_grid_v01.nc')

g_sfb=dfm_grid.DFMGrid('../../sfb_dfm_v2/sfei_v19_net.nc')

## 
g_roms_poly=g_roms.boundary_polygon()
g_sfb_poly =g_sfb.boundary_polygon()

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g_roms.plot_edges(ax=ax,color='b',lw=0.8)
g_sfb.plot_edges(ax=ax,color='g',lw=0.8)

plot_wkb.plot_polygon(g_roms_poly,facecolor='b',alpha=0.3)
plot_wkb.plot_polygon(g_sfb_poly,facecolor='g',alpha=0.3)

##

# Clear out a buffer region 
sfb_dx_min=200
roms_dx_typ=3500
# cheating a bit - ought to be more like 10xn
buff_size=(roms_dx_typ - sfb_dx_min)*6

sfb_buff=g_sfb_poly.buffer(buff_size)

## 
overlap_cells=g_roms.select_cells_intersecting(sfb_buff)

for c in np.nonzero(overlap_cells)[0]:
    g_roms.delete_cell(c)
    
g_roms.renumber_cells()
g_roms.make_edges_from_cells()
g_roms.delete_orphan_nodes()
g_roms.renumber()

## 

# Find the points where the shorelines should join

g_roms_new_poly=g_roms.boundary_polygon()
##

new_boundary=g_roms_new_poly.buffer(50).boundary
crossings=new_boundary.intersection( g_roms_poly.boundary )
roms_tie_points=np.array(crossings)

## 
matching_points=[]

for roms_tie_point in roms_tie_points:
    node=g_sfb.select_nodes_nearest(roms_tie_point)
    matching_points.append( g_sfb.nodes['x'][node] )
matching_points=np.array(matching_points)
    
##

ax.plot(roms_tie_points[:,0],
        roms_tie_points[:,1],
        'bo')
ax.plot(matching_points[:,0],
        matching_points[:,1],
        'go')

##

g_merge=g_sfb.copy()
g_merge.add_grid(g_roms)

##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
g_merge.plot_edges(ax=ax,color='k',lw=0.8)


##

# Get a proper shoreline for joining them.
from stompy.spatial import wkb2shp

land_shp=wkb2shp.shp2geom('/opt/data/GIS/shorelines/noaa_med_res/land_polygon-utm10.shp')
land_geom=land_shp['geom'][0] # first is the mainland
# plot_wkb.plot_wkb(land_geom,ax=ax)

##

# Too fragile to do this automatically
# Find the points where the shoreline intersects each of the two grids

def splice_shore(roms_pnt,dfm_pnt):
    # May have to do some smoothing on the shoreline before this step
    # Need to extract a linestring from the shoreline which joins
    # those two points in the grid.
    shore_string=np.array(land_geom.exterior)

    roms_best=np.argmin( utils.dist( shore_string,roms_pnt))
    dfm_best =np.argmin( utils.dist( shore_string,dfm_pnt))
    shore_substring=shore_string[ min(roms_best,dfm_best):max(roms_best,dfm_best)+1]

    # Do this before adding the new nodes
    roms_node=g_merge.select_nodes_nearest(roms_pnt)
    dfm_node=g_merge.select_nodes_nearest(dfm_pnt)

    new_nodes=[g_merge.add_node(x=x)
               for x in shore_substring]
    new_edges=[g_merge.add_edge(nodes=[a,b])
               for a,b in zip(new_nodes[:-1],new_nodes[1:])]

    # careful of the order!
    if roms_best<dfm_best:
        g_merge.add_edge(nodes=[new_nodes[0],roms_node])
        g_merge.add_edge(nodes=[new_nodes[-1],dfm_node])
    else:
        g_merge.add_edge(nodes=[new_nodes[-1],roms_node])
        g_merge.add_edge(nodes=[new_nodes[0],dfm_node])
    

##

# Add a connecting bit of shoreline

if 1: # splice in real shoreline between the two grids        
    # Hand-picked these point on the north side
    # Give the existing grids a little bit of space, so pick a point just
    # outside them
    splice_shore(roms_pnt=(499033., 4239247.),
                 dfm_pnt =(499079., 4206506.))

    # Southern side:
    splice_shore(roms_pnt=(552462, 4135376),
                 dfm_pnt=(542700., 4152880.))
        
if 0: # OLD - join them with a straight line:
    for roms_pnt,sfb_pnt in zip( roms_tie_points,
                                 matching_points):
        nA=g_merge.select_nodes_nearest(roms_pnt)
        nB=g_merge.select_nodes_nearest(sfb_pnt)
        g_merge.add_edge(nodes=[nA,nB])

##


# work around issues with pulling polygons from incomplete grid
from stompy.spatial import join_features
ls=g_merge.boundary_linestrings()
ls_polys=[geometry.Polygon(l)
          for l in ls]

biggest=np.argmax( [p.area for p in ls_polys])
outer_boundary=ls_polys[biggest]
no_mans_land=outer_boundary.difference(g_roms_new_poly).difference(geometry.Polygon(g_sfb_poly.exterior))

seed_point=no_mans_land.representative_point() # [500870., 4170787.]
seed_point=np.array(seed_point)
# 5000 ought to be plenty of nodes to get around this loop
nodes=g_merge.enclosing_nodestring(seed_point,5000)

## 

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

g_merge.plot_edges(ax=ax,color='k',lw=0.8)

xy_shore=g_merge.nodes['x'][nodes]


plot_wkb.plot_wkb(no_mans_land,facecolor='r',alpha=0.3)
ax.plot(xy_shore[:,0],
        xy_shore[:,1],
        'r-')

##

# Construct a scale based on existing spacing
# But only do this for edges that are part of one of the original grids
seg_starts=xy_shore
seg_ends=xy_shore[ (np.arange(len(xy_shore))+1)%(len(xy_shore)) ]

sample_xy=0.5*(seg_starts + seg_ends)
sample_scale=utils.dist(seg_starts - seg_ends)

apollo=field.PyApolloniusField(X=sample_xy,F=sample_scale)

## 

# Prepare that shoreline for grid generation.
from stompy.grid import front, cgal_line_walk, shadow_cdt
from stompy.spatial import field, robust_predicates

six.moves.reload_module(robust_predicates)
six.moves.reload_module(cgal_line_walk)
six.moves.reload_module(shadow_cdt)
six.moves.reload_module(front)



grid_to_pave=unstructured_grid.UnstructuredGrid(max_sides=6)

AT=front.AdvancingTriangles(grid=grid_to_pave)

AT.add_curve(xy_shore)
# This should be safe about not resampling existing edges
AT.scale=field.ConstantField(50000)

AT.initialize_boundaries()

AT.grid.nodes['fixed'][:]=AT.RIGID
AT.grid.edges['fixed'][:]=AT.RIGID

# But the two stitch edges are not rigid:
for roms_pnt,sfb_pnt in zip( roms_tie_points,
                             matching_points):
    nA=AT.grid.select_nodes_nearest(roms_pnt)
    nB=AT.grid.select_nodes_nearest(sfb_pnt)
    j=AT.grid.nodes_to_edge(nA,nB)
    AT.grid.edges['fixed'][j]=0

AT.scale=apollo    

# What is the current status of who will or won't resample these things?

# plt.figure(10).clf()
# fig,ax=plt.subplots(num=10)
# AT.grid.plot_edges(values=AT.grid.edges['fixed'],cmap='RdYlGn')
# AT.grid.plot_nodes(values=AT.grid.nodes['fixed'],cmap='jet',
#                    labeler=lambda i,r: str(i))

##
AT.loop()
AT.grid.renumber()

# Comes out at 3607 cells.  Not too bad.

##

## 
#for _ in range(10):
if 1:
    AT.loop(100)
    ax.cla()
    AT.grid.plot_edges(lw=0.6)

##
ax.cla()
AT.grid.plot_edges(lw=0.6)
g_sfb.plot_edges(lw=0.6,color='orange')
g_roms.plot_edges(lw=0.6,color='green')

##
ax.xaxis.set_visible(0)
ax.yaxis.set_visible(0)
fig.savefig('spliced-grids.pdf')
fig.savefig('spliced-grids.png',dpi=200)

##

# Not quite there -
g_complete=g_sfb.copy()

# annotate which grid various elements came from
for gr,src in [(g_complete,1),
               (g_roms,2),
               (AT.grid,3)]:
    gr.add_node_field('src',src*np.ones(gr.Nnodes()),on_exists='overwrite')
    
    gr.add_edge_field('src',src*np.ones(gr.Nedges()),on_exists='overwrite')
    try: # hack, remove after reloading unstructured_grid
        gr.add_cell_field('src',src*np.ones(gr.Ncells())) # ,on_exists='overwrite')
    except ValueError:
        gr.cells['src']=src

## 
g_complete.add_grid(g_roms)
g_complete.add_grid(AT.grid)

# Surprisingly, this works!
g_complete.merge_duplicate_nodes()

g_complete.renumber()

##

plt.clf()
g_complete.plot_edges(lw=0.4,values=g_complete.edges['src'])
g_complete.plot_cells(values=g_complete.cells['src'])
g_complete.plot_nodes(values=g_complete.nodes['src'])

##

g_complete.write_ugrid('spliced_grids_00.nc')

##

# Add bathy -- mostly copy from existing grids
splice_bathy=np.zeros(g_complete.Nnodes(),'f8')

# Copy bathy from SFB and ROMS grids:
for g_src,src in [ (g_sfb,1), (g_roms,2) ]:
    for n in np.nonzero(g_complete.nodes['src']==src)[0]:
        n_src=g_src.select_nodes_nearest( g_complete.nodes['x'][n] )
        splice_bathy[n]=g_src.nodes['depth'][n_src]

##

# Stitch gets bathy from NGDC:
from sfb_dfm_utils import ca_roms
dem=ca_roms.coastal_dem()

sel_nodes=np.nonzero(g_complete.nodes['src']==3)[0]

sel_node_depth=dem( g_complete.nodes['x'][sel_nodes] )
splice_bathy[sel_nodes] = sel_node_depth

##

g_complete.add_node_field('depth',splice_bathy)

##

plt.clf()
# g_complete.plot_edges(lw=0.4,color='k')
ncoll=g_complete.plot_nodes(values=splice_bathy)
ncoll.set_clim([-40,0])

##

plt.clf()
g_complete.contourf_node_values( np.sqrt( (-g_complete.nodes['depth']).clip(0,np.inf)) ,
                                 30,
                                 cmap='jet',extend='both')
