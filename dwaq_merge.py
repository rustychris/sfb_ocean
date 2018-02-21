"""
Testing the python version of the coupling code, as ddcouplefm is having issues.
"""
import six
import xarray as xr
from stompy.model.delft import dfm_grid
import stompy.model.delft.io as dio
import stompy.model.delft.waq_scenario as waq
##

# Load a global grid, which will be used for defining aggregation
# (or lack thereof in this case).
# This one might work, though there is also the global grid which is
# written out by DFM, DFM_interpreted_idomain_spliced_grids_01_bathy_net.nc
g_global=dfm_grid.DFMGrid("runs/short_test_13/spliced_grids_01_bathy_net.nc")

##

six.moves.reload_module(waq)

# merge_only should make this a bit faster, but also requires the exact
# match of g_global
# sparse_layers should be False, as parts of dwaq don't like sparse layers.
# exch_z_area_constant: no need to force this to be true, and it makes writing
# areas 20x slower.

multi_hydro=waq.HydroMultiAggregator(run_prefix="short_test_13",
                                     path="runs/short_test_13",
                                     agg_shp=g_global,
                                     link_ownership="owner_of_min_elem",
                                     sparse_layers=False,
                                     exch_z_area_constant=False)

# Need a scenario for arcane reasons
scen=waq.Scenario(hydro=multi_hydro)

scen.base_path=os.path.join("runs/short_test_13/global")
scen.name="merged"

scen.write_hydro()

print("Really need reload hydro as HydroFiles, then run adjust_boundaries_for_conservation()")

##

hyd=multi_hydro
geom=hyd.get_geom()

g=dfm_grid.DFMGrid(geom)

##

bnd=multi_hydro.create_bnd()

##

dio.write_bnd(bnd,'test.bnd')

