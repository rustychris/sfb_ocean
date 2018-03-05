import xarray as xr
from stompy.model.delft import dfm_grid
from stompy.io.local import noaa_coops
from stompy.grid import unstructured_grid

## 

map_nc="/hpcvol1/rusty/dfm/sfb_ocean/runs/short_21_norm_orig/DFM_OUTPUT_short_21/short_21_map.nc"
map_rev_nc="/hpcvol1/rusty/dfm/sfb_ocean/runs/short_21/DFM_OUTPUT_short_21/short_21_map.nc"

map_ds=xr.open_dataset(map_nc)
map_rev_ds=xr.open_dataset(map_rev_nc)

## 
ptreyes=noaa_coops.coops_dataset('9415020',map_ds.time[0],map_ds.time[-1],
                                 ['water_level'],cache_dir='cache',days_per_request='M')


g=dfm_grid.DFMGrid(map_ds)
## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(ax=ax,color='k',lw=0.5)

## 

# middle of the domain
samp_point=np.array( [464675., 4146109.] )

samp_idx=g.select_cells_nearest(samp_point)

## 

eta=map_ds.s1.isel(nFlowElem=samp_idx)

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
ax.plot(eta.time,eta,label='Model normal')
ax.plot(map_rev_ds.time,map_rev_ds.s1.isel(nFlowElem=samp_idx),label='Model reverse')
ax.plot(ptreyes.time,ptreyes.water_level.isel(station=0),label='PtR')

ax.legend()
## 

