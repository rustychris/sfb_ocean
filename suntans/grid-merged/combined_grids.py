from functools import reduce
import numpy as np 
from stompy import utils
from stompy.grid import front, cgal_line_walk, shadow_cdt
from stompy.spatial import field, robust_predicates

from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
from stompy.spatial import wkb2shp, proj_utils
from shapely import geometry
from stompy.plot import plot_wkb

##

# Load the Bay grid, with bathy
#g_bay=unstructured_grid.UnstructuredGrid.from_ugrid('../grid-sfbay/sfbay-grid-20190301b-bathy.nc')
# moving forward, use this cropped one:
g_bay=unstructured_grid.UnstructuredGrid.from_ugrid('cropped-sfbay-20190301b.nc')

##

# Generate a cartesian for the ocean portion.

# shoreline..
states=wkb2shp.shp2geom("/home/rusty/svn/research/my_papers/gross-recruitment/figures/map/st99_d00.shp")
ca=states[ states['NAME']=='California' ]

## project to UTM

ll2utm=proj_utils.mapper('WGS84','EPSG:26910')

p_geoms=[]
for geom in ca['geom']:
    assert geom.type=='Polygon'

    p_ext=ll2utm(np.array(geom.exterior))
    p_ints=[ ll2utm(np.array(r)) for r in geom.interiors]
    p_geoms.append( geometry.Polygon(p_ext,p_ints))

##

g_cart=unstructured_grid.UnstructuredGrid()

L=290e3
W=160e3
dx=dy=2.0e3

g_cart.add_rectilinear([-W,-L/2],[0,L/2],
                       int(1+W/dx),int(1+L/dy))

# Rotate CCW and translate 0,0 of that grid to hit ~GG
g_cart.nodes['x']=utils.rot( 30*np.pi/180., g_cart.nodes['x']) + np.array([560e3,4191e3])
cart_poly=g_cart.boundary_polygon()

##

# Clear out a buffer region 
sfb_dx_min=200
cart_dx_typ=dx

# This ends up being a bit overkill, since it buffers
# from parts of the bay grid which are not touching
# the ocean grid.  Still, the effect is kind of
# nice, giving some higher near-shore resolution, so
# don't fix it at this point.
buff_size=(cart_dx_typ - sfb_dx_min)*10

g_sfb_poly=g_bay.boundary_polygon()
sfb_buff=g_sfb_poly.buffer(buff_size)

g_cart_buff=g_cart.copy()
overlap_cells=g_cart_buff.select_cells_intersecting(sfb_buff)

for c in np.nonzero(overlap_cells)[0]:
    g_cart_buff.delete_cell(c)

g_cart_buff.delete_orphan_edges()
g_cart_buff.delete_orphan_nodes()
g_cart_buff.renumber()

## 

g_cart_trim=g_cart_buff.copy()

# Find nodes falling on land
mask=reduce(lambda a,b: a|b,
            [ g_cart_trim.select_nodes_intersecting(geo)
              for geo in p_geoms])

for n in np.nonzero(mask)[0]:
    g_cart_trim.delete_node_cascade(n)

g_cart_trim.delete_orphan_edges()
g_cart_trim.delete_orphan_nodes()
g_cart_trim.renumber()

##

# use the full cartesian grid to grab the relevant shoreline
# linestring

# Get a linestring shoreline
idx=np.argmax([geo.area for geo in ca['geom']])
ca_poly=p_geoms[idx] # yep
shoreline=ca_poly.boundary

##

# distance out from the respective grids where
# we pick up the shoreline
buff_poly=g_cart_buff.boundary_polygon()
buff_bdry=buff_poly.buffer(cart_dx_typ).boundary

outer_hits=shoreline.intersection(buff_bdry) # 6 points -- meh

bay_poly=g_sfb_poly.buffer(2*sfb_dx_min) # fudge factor as it's a min dx
bay_bdry=bay_poly.boundary 
inner_hits=shoreline.intersection(bay_bdry) # 78 !!

##

outer_proj=[shoreline.project(pnt) for pnt in outer_hits.geoms]
inner_proj=[shoreline.project(pnt) for pnt in inner_hits.geoms]

outer_pnts=np.array([np.array(shoreline.interpolate(s)) for s in outer_proj])
inner_pnts=np.array([np.array(shoreline.interpolate(s)) for s in inner_proj])

## put proj in order:
all_proj=outer_proj + inner_proj
all_sources=['outer']*len(outer_proj) + ['inner']*len(inner_proj)
order=np.argsort(all_proj)
order_source=[all_sources[i] for i in order]

cut_points=[]

g_combined=g_bay.copy()
g_combined.add_grid(g_cart_trim)

for i in range(len(order)-1):
    a=order[i]
    b=order[i+1]
    
    if all_sources[a]!=all_sources[b]:
        cut_points.append( np.array(shoreline.interpolate(all_proj[a])) )
        cut_points.append( np.array(shoreline.interpolate(all_proj[b])) )
        # probably there is a better way to do this, but not worrying about it
        # right now.
        seg=[shoreline.interpolate(x) for x in np.arange(all_proj[a],all_proj[b],100)]

        node_start=g_combined.select_nodes_nearest(seg[0])
        node_end  =g_combined.select_nodes_nearest(seg[-1])
        
        nodes=[g_combined.add_node(x=pnt) for pnt in seg]
        
        for j in range(len(seg)-1):
            g_combined.add_edge(nodes=[nodes[j],nodes[j+1]])

        g_combined.add_edge(nodes=[node_start,nodes[0]])
        g_combined.add_edge(nodes=[node_end,nodes[-1]])

##

def triangulate_hole(grid,seed_point,max_nodes=5000):
    # manually tell it where the region to be filled is.
    # 5000 ought to be plenty of nodes to get around this loop
    nodes=grid.enclosing_nodestring(seed_point,max_nodes)
    xy_shore=grid.nodes['x'][nodes]

    # Construct a scale based on existing spacing
    # But only do this for edges that are part of one of the original grids
    grid.edge_to_cells() # update edges['cells']
    sample_xy=[]
    sample_scale=[]
    ec=grid.edges_center()
    el=grid.edges_length()

    for na,nb in utils.circular_pairs(nodes):
        j=grid.nodes_to_edge([na,nb])
        if np.any( grid.edges['cells'][j] >= 0 ):
            sample_xy.append(ec[j])
            sample_scale.append(el[j])

    sample_xy=np.array(sample_xy)
    sample_scale=np.array(sample_scale)

    apollo=field.PyApolloniusField(X=sample_xy,F=sample_scale)

    # Prepare that shoreline for grid generation.

    grid_to_pave=unstructured_grid.UnstructuredGrid(max_sides=6)

    AT=front.AdvancingTriangles(grid=grid_to_pave)

    AT.add_curve(xy_shore)
    # This should be safe about not resampling existing edges
    AT.scale=field.ConstantField(50000)

    AT.initialize_boundaries()

    AT.grid.nodes['fixed'][:]=AT.RIGID
    AT.grid.edges['fixed'][:]=AT.RIGID

    # Old code compared nodes to original grids to figure out RIGID
    # more general, if it works, to see if a node participates in any cells.
    # At the same time, record original nodes which end up HINT, so they can
    # be removed later on.
    src_hints=[]
    for n in AT.grid.valid_node_iter():
        n_src=grid.select_nodes_nearest(AT.grid.nodes['x'][n])
        delta=utils.dist( grid.nodes['x'][n_src], AT.grid.nodes['x'][n] )
        assert delta<0.1 # should be 0.0

        if len(grid.node_to_cells(n_src))==0:
            # It should be a HINT
            AT.grid.nodes['fixed'][n]=AT.HINT
            src_hints.append(n_src)
            # And any edges it participates in should not be RIGID either.
            for j in AT.grid.node_to_edges(n):
                AT.grid.edges['fixed'][j]=AT.UNSET

    AT.scale=apollo
    
    if AT.loop():
        AT.grid.renumber()
    else:
        print("Grid generation failed")
        return AT # for debugging -- need to keep a handle on this to see what's up.

    for n in src_hints:
        grid.delete_node_cascade(n)
        
    grid.add_grid(AT.grid)

    # Surprisingly, this works!
    grid.merge_duplicate_nodes()

    grid.renumber()

    return grid
##

# with the original bay grid this is pretty good
#seed_point=np.array([500870., 4170787.])
# with the truncated bay grid, use this point:
seed_point=np.array([524296., 4174803.])
g_complete=triangulate_hole(g_combined,seed_point)

##

zoom=(527081., 530582, 4189842, 4192451)

plt.clf()
g_complete.plot_edges(lw=0.5)
g_complete.plot_cells(lw=0.5,zorder=-1,color='0.8')
#plt.axis(zoom)

## 

g_complete.write_ugrid('spliced_grids_01.nc',overwrite=True)

##

# Add bathy -- copy from existing Bay grid where possible, otherwise
# from NGDC

# Bay grid:
#  cells: 'depth'
#  edges: 'edge_depth'

from scipy.spatial import cKDTree as KDTree

g_complete.add_edge_field('edge_depth',np.nan*np.ones(g_complete.Nedges(),np.float64),
                          on_exists='overwrite')
g_complete.add_cell_field('depth',np.nan*np.ones(g_complete.Ncells(),np.float64),
                          on_exists='overwrite')


def copy_matching_fields(g_src,g_dest,eps=1.0,
                         cell_fields=['depth'],edge_fields=['edge_depth']):
    # match cells by centroid in case centers have been
    # adjusted
    src_cc=g_src.cells_centroid()
    src_cc_kdt=KDTree(data=src_cc)
    
    dest_cc=g_dest.cells_centroid()

    for c in utils.progress(range(g_dest.Ncells())):
        dist,src_cell=src_cc_kdt.query(dest_cc[c],distance_upper_bound=eps)
        if not np.isfinite(dist): continue
        for fld in cell_fields:
            g_dest.cells[fld][c]=g_src.cells[fld][src_cell]

    src_ec=g_src.edges_center()
    src_ec_kdt=KDTree(data=src_ec)
    
    dest_ec=g_dest.edges_center()
    
    for j in utils.progress(range(g_dest.Nedges())):
        dist,src_edge=src_ec_kdt.query(dest_ec[j],distance_upper_bound=eps)
        if not np.isfinite(dist): continue
        for fld in edge_fields:
            g_dest.edges[fld][j]=g_src.edges[fld][src_edge]
            
copy_matching_fields(g_src=g_bay,g_dest=g_complete,
                     cell_fields=['depth'],edge_fields=['edge_depth'])

##
utils.path("..")

import bathy

dem=bathy.dem()

sel_cells=np.nonzero(np.isnan(g_complete.cells['depth']))[0]
centr=g_complete.cells_centroid()

g_complete.cells['depth'][sel_cells]=dem(centr[sel_cells])

sel_edges=np.nonzero(np.isnan(g_complete.edges['edge_depth']))[0]

e2c=g_complete.edge_to_cells()

nc1=e2c[sel_edges,0].copy()
nc2=e2c[sel_edges,1].copy()
nc1[nc1<0]=nc2[nc1<0]
nc2[nc2<0]=nc1[nc2<0]
cell_depth=g_complete.cells['depth']
g_complete.edges['edge_depth'][sel_edges] = np.maximum( cell_depth[nc1],
                                                        cell_depth[nc2] )

##
import stompy.plot.cmap as scmap

cmap=scmap.load_gradient('hot_desaturated.cpt')

plt.figure(1).clf()
g_complete.plot_cells(values=g_complete.cells['depth'],clim=[-4000,10],cmap=cmap)
g_complete.plot_edges(values=g_complete.edges['edge_depth'],clim=[-4000,10],cmap=cmap)

plt.setp(plt.gca().collections,clim=[-100,10])

## 
g_complete.write_ugrid('spliced_grids_01_bathy.nc',overwrite=True)

