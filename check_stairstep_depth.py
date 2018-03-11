"""
Does the map output include depth of truncated water columns?

 - NetNode_z does not have the quantized depths
 - FlowElem_zcc does not have the quantized depths
 - FlowElem_bl does not have the quantized depths
 - waterdepth is not quantized
 - DWAQ output, with volumes and planform area, *does* include stairstepping.

This is all from a run with mdu['numerics','Keepzlayeringatbed']=1
which I think produces stairstepped beds

"""
import xarray as xr
from stompy.grid import unstructured_grid
from stompy.plot import plot_utils
from stompy.model.delft import dfm_grid
import stompy.model.delft.waq_scenario as waq

##

map_ds=xr.open_dataset('runs/short_28/DFM_OUTPUT_short_28-tmp/short_28-tmp_map.nc')

##

g=dfm_grid.DFMGrid(map_ds)

##

plt.figure(1).clf()
g.plot_edges(lw=0.5,color='k')


ncoll=g.plot_nodes(values=map_ds.NetNode_z.values )
cbar=plot_utils.cbar(ncoll)

##

hydro=waq.HydroFiles('runs/short_28/DFM_DELWAQ_short_28/short_28.hyd')

hyd_g=hydro.grid()

# This file is truncated, and currently waq won't infer n_seg:
kmx=20
hydro._n_seg=hyd_g.Ncells() * kmx

vols=hydro.volumes(hydro.t_secs[1])

A=hydro.planform_areas()

seg_depth = vols/A.data


##

plt.figure(2).clf()
# plt.hist(map_ds.NetNode_z.values,bins=100)
# plt.hist(map_ds.FlowElem_zcc.values,bins=100)
# plt.hist(map_ds.FlowElem_bl.values,bins=100)
# plt.hist(map_ds.waterdepth.isel(time=0).values, bins=100)
plt.hist(seg_depth,bins=100)

##

# HERE
#   wind the DWAQ output back into per-element depths
#   Verify that (a) I can calculate the right z layer boundaries,
#           and (b) depths are truncated, not extended.
#   Add an option to ocean_dfm.py to truncate depths at those interfaces,
#   plus a bit of slop (assuming they are truncated, not extended)
elt_depths = seg_depth
hydro.seg_to_2d_element

##

runs/short_28/DFM_DELWAQ_short_28/
