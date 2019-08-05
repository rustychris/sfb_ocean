from stompy.grid import unstructured_grid
from stompy import utils
import numpy as np
utils.path("..")
import bathy
from stompy.grid import depth_connectivity

## 
g_in='splice-merge-05-filled-edit42.nc'
g_out=g_in.replace('.nc','-bathy.nc')


assert g_in!=g_out
shallow_thresh=-1

g=unstructured_grid.UnstructuredGrid.from_ugrid(g_in)

dem=bathy.dem()

z_cell_mean=depth_connectivity.cell_mean_depth(g,dem)


## 

e2c=g.edge_to_cells().copy()
nc1=e2c[:,0]
nc2=e2c[:,1]
nc1[nc1<0]=nc2[nc1<0]
nc2[nc2<0]=nc1[nc2<0]

## 

# starting point for edges is shallower of the neighboring cells
z_edge=np.maximum(z_cell_mean[nc1],z_cell_mean[nc2])

# only worry about connectivity when the edge is starting above
# the threshold
shallow=(z_edge>shallow_thresh)

# centers='centroid' seemed to be losing a lot of connectivity.
z_edge_conn=depth_connectivity.edge_connection_depth(g,dem,
                                                     edge_mask=shallow,
                                                     centers='lowest')
valid=np.isfinite(z_edge_conn)

z_edge[valid]=z_edge_conn[valid]

## 

# edge-based is better at getting the unresolved channels connected
# leads to alligator teeth in some places.
# only use edge connectivity approach down to edge_thresh
z_cell_edgeminthresh=[ min(max(shallow_thresh,
                               z_edge[ g.cell_to_edges(c) ].min()),
                           z_cell_mean[c])
                       for c in range(g.Ncells()) ]

## 
g.add_cell_field('z_bed',np.asarray(z_cell_edgeminthresh),
                 on_exists='overwrite')

##

# raise Exception("Add code to set z0B where not already set")
rough='z0B'
if rough in g.edges.dtype.names:
    missing=g.edges[rough]==0
    g.edges[rough][missing]=0.002

## 
g.write_ugrid(g_out)


