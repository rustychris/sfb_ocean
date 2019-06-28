from stompy.grid import unstructured_grid
from stompy import utils
## 
g_in='splice-merge-05-filled-edit06.nc'
g_out=g_in.replace('.nc','-bathy.nc')
assert g_in!=g_out


g=unstructured_grid.UnstructuredGrid.from_ugrid(g_in)
##

utils.path("..")
import bathy

dem=bathy.dem()

##
z_cell_mean=depth_connectivity.cell_mean_depth(g,dem)


z_node=dem(g.nodes['x'])
z_cell=dem(g.cells_centroid())
z_cell_nodemin=[ min(z_node[ g.cell_to_nodes(c) ].min(),
                     z_cell[c])
                 for c in range(g.Ncells()) ]

##

z_cell_nodemiddle=[ min(0.5*(z_node[ g.cell_to_nodes(c) ].min() +
                             z_node[ g.cell_to_nodes(c) ].max()),
                        z_cell[c])
                 for c in range(g.Ncells()) ]

##

plt.figure(1).clf()
plt.title("min(min(nodes(cell),),cell)")

ccoll=g.plot_cells(values=z_cell_nodemin,cmap='jet')
plt.axis('equal')
ccoll.set_clim([-5,2])
plt.colorbar(ccoll)

##

# Edge connectivity
from stompy.grid import depth_connectivity

z_edge=z_node[g.edges['nodes']].mean(axis=1)
shallow=(z_edge>-1)

# centers='lowest' is too much bias.
edge_depths=depth_connectivity.edge_connection_depth(g,dem,
                                                     edge_mask=shallow,
                                                     centers='centroid')
valid=shallow & np.isfinite(edge_depths)

z_edge[valid] = np.minimum(z_edge[valid],edge_depths[valid])
assert np.all(np.isfinite(z_edge))

## 
z_cell_edgemin=[ min(z_edge[ g.cell_to_edges(c) ].min(),
                     z_cell[c])
                 for c in range(g.Ncells()) ]

##
plt.figure(2).clf()
plt.title("min(min(edges(cell),),cell)")

ccoll=g.plot_cells(values=z_cell_edgemin,cmap='jet')
plt.axis('equal')
ccoll.set_clim([-5,2])
plt.colorbar(ccoll)

##

# edge-based is better at getting the unresolved channels connected
# leads to alligator teeth in some places.
# only use edge connectivity approach down to edge_thresh
edge_thresh=-1
z_cell_edgeminthresh=[ min(max(edge_thresh,z_edge[ g.cell_to_edges(c) ].min()),
                           z_cell_mean[c])
                       for c in range(g.Ncells()) ]

# plt.figure(3).clf()
# plt.title("min(max(-1,min(edges(cell))),cell)")
# 
# ccoll=g.plot_cells(values=z_cell_edgeminthresh,cmap='jet')
# plt.axis('equal')
# ccoll.set_clim([-5,2])
# plt.colorbar(ccoll)


##




##
if 0:
    depth_fields=[ # ('cell',z_cell),
                   ('z_cell_nodemin',z_cell_nodemin),
                   ('z_cell_edgemin',z_cell_edgemin),
                   ('z_cell_mean',z_cell_mean),
                   ('z_cell_edgeminthresh',z_cell_edgeminthresh),
                   #('z_cell_nodemiddle',z_cell_nodemiddle),
    ]

    plt.figure(1).clf()
    fig,axs=plt.subplots(2,2,sharex=True,sharey=True,num=1)
    axs=axs.ravel()

    from matplotlib import colors
    norm=colors.Normalize(vmin=-5,vmax=2)
    ccolls=[]
    for ax,(label,depths) in zip(axs,depth_fields):
        ccoll=g.plot_cells(values=depths,cmap='jet',norm=norm,ax=ax)
        ax.xaxis.set_visible(0)
        ax.yaxis.set_visible(0)
        # ax.set_title(label)

    cax=fig.add_axes([0.95,.5,0.02,0.3])    
    plt.colorbar(ccoll,cax=cax)
    axs[0].axis('equal')
    fig.tight_layout()

##

# Evaluating which is best:
# 1. are small channels connected in Suisun and LSB?
# 2. avoid alligator teeth?
# 3. avoid splotches from node-min?


# Best seems to be
#  min( true mean of cells,
#       max(-1, min_{j of cell} edge_depth_connectivity(centers='centeroid') ) )
# 
