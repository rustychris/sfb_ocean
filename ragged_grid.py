# Grid-specific details for ragged coastal-only grid.
# These modules should specify
#    grid: an unstuctured_grid with bathy on the nodes
#    bc_features: array ready to be fed to DFlowModel machinery
#      as if it were read in via wkb2shp
import os
from stompy.spatial import wkb2shp
# from sfb_dfm_utils import ca_roms
from stompy.grid import unstructured_grid

here=os.path.dirname(__file__)
ragged_dir=os.path.join(here,'ragged')
ugrid_file=os.path.join(ragged_dir,'grid_v01.nc')

# if not os.path.exists(ugrid_file):
#     poly=wkb2shp.shp2geom('grid-poly-v00.shp')[0]['geom']
#     g=ca_roms.extract_roms_subgrid_poly(poly)
#     ca_roms.add_coastal_bathy(g)
#     g.write_ugrid(ugrid_file)
# else:

grid=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)

derived=os.path.join(ragged_dir,'derived')
if not os.path.exists(derived):
    os.makedirs(derived)

g_shp=os.path.join(derived,'grid_v01-edges.shp')
if not os.path.exists(g_shp):
    grid.write_edges_shp(g_shp)

coastal_bc_coords=[ [450980., 4291405.], # northern
                    [595426., 4037083.] ] # southern

bc_features=wkb2shp.shp2geom(os.path.join(ragged_dir,'forcing-v01.shp'))

