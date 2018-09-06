# 2018-06-28
# With 53925, compare model output as forced by OTPS fluxes
# with OTPS water surface.
import xarray as xr
from stompy.grid import unstructured_grid
from stompy.spatial import proj_utils
import matplotlib.pyplot as plt

##

map_fn="runs/ocean2_001/DFM_OUTPUT_flowfm/flowfm_map.nc"
ds=xr.open_dataset(map_fn)

##

map_kmx0_fn="runs/ocean2_001kmx0/DFM_OUTPUT_flowfm/flowfm_map.nc"
ds_kmx0=xr.open_dataset(map_kmx0_fn)

##

# Going back to r52184.
map_52184_fn="runs/ocean2_004/DFM_OUTPUT_flowfm/flowfm_map.nc"
ds_52184=xr.open_dataset(map_52184_fn)

##

to_ll=proj_utils.mapper('EPSG:26910','WGS84')

# A little off Point Reyes
xy=[504160, 4191352]
ll=to_ll(xy)

##

g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

##

from stompy.model.otps import read_otps
read_otps.OTPS_DATA='derived'

modfile=read_otps.model_path('wc')

h,u,v = read_otps.tide_pred(modfile,
                            lon=ll[:1],lat=ll[1:],time=ds.time.values)


##

# sanity check for grid geometry: good.
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(ax=ax,lw=0.4,color='k')
ccoll=g.plot_cells(values=ds.FlowElem_bl.values,ax=ax,cmap='jet')

ax.axis('equal')

##

# Find a good cell:
c=g.select_cells_nearest(xy) # 2004

# Get a tidal timeseries there:
c_s1=ds.s1.isel(nFlowElem=c)

##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
ax.plot(c_s1.time,c_s1,label='Map output off PR')
ax.plot(c_s1.time,h,label='OTPS wc')
ax.plot(ds_kmx0.time,
        ds_kmx0.s1.isel(nFlowElem=c),label='Map output kmx=0')
ax.plot(ds_52184.time,
        ds_52184.s1.isel(nFlowElem=c),label='Map output ocean004')

ax.legend()

##

# 
