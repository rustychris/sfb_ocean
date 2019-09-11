import os
import numpy as np
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.transforms import Bbox

import stompy.plot.cmap as scmap
from stompy.plot import plot_utils

##

sfb_ocean="."
grid_ocean=unstructured_grid.UnstructuredGrid.from_ugrid(os.path.join(sfb_ocean,
                                                                      "grid-merged",
                                                                      "spliced_grids_01_bathy.nc"))
##

cmap=scmap.load_gradient('oc-sst.cpt',reverse=True)

fig=plt.figure(1)
fig.clf()
fig.set_size_inches((7,9),forward=True)
ax=fig.add_axes([0,0,1,1])
ax.xaxis.set_visible(0)
ax.yaxis.set_visible(0)


grid_ocean.plot_edges(ax=ax,lw=0.5,color='k',alpha=0.2)
ccoll=grid_ocean.plot_cells(ax=ax,values=(-grid_ocean.cells['depth']).clip(1,np.inf),
                            norm=colors.LogNorm(),
                            cmap=cmap)
cax=fig.add_axes([0.07,0.15,0.03,0.35])

cbar=plt.colorbar(ccoll,cax=cax,label="Depth (m)")
cax.invert_yaxis()
plot_utils.scalebar([0.07,0.025],L=100000,fractions=[0,0.25,0.5,1.0],
                    unit_factor=1e-3,label_txt=" km",
                    ax=ax,xy_transform=ax.transAxes,
                    dy=0.01)
fig.savefig("sfb_ocean_grid_bathy-panels.png",dpi=150)

##
# (1.0, 4177.424014636387)

normed=ccoll.norm( (-grid_ocean.cells['depth']).clip(1,np.inf) )

grid_ocean.write_cells_shp("../gis/merged_grid_bathy.shp",
                           extra_fields=[ ('depth',(-grid_ocean.cells['depth']).clip(1,np.inf)),
                                          ('elev',grid_ocean.cells['depth'].clip(-np.inf,-1)),
                                          ('logdepth',np.log10( (-grid_ocean.cells['depth']).clip(1,np.inf))),
                                          ('normed',normed)
                           ],
                           overwrite=True)
##

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

#extent = cax.get_window_extent()
extent=full_extent(cax,pad=0.05)
extent=extent.transformed(fig.dpi_scale_trans.inverted())
fig.savefig('sfb_ocean_grid_bathy-colorbar.png', bbox_inches=extent,dpi=600)

# Pad the saved area by 10% in the x-direction and 20% in the y-direction
#fig.savefig('ax2_figure_expanded.png', bbox_inches=extent.expanded(1.1, 1.2))
