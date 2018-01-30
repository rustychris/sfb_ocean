import os
import xarray as xr
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid

## 

run_name='short_test_12'

run_base_dir=os.path.join('runs',run_name)

procs=2

maps=[ os.path.join( run_base_dir,
                     "DFM_OUTPUT_%s"%run_name,
                     "%s_%04d_map.nc"%(run_name,proc) )
       for proc in range(procs)]

his_fn=os.path.join( run_base_dir,
                     "DFM_OUTPUT_%s"%run_name,
                     "%s_0000_his.nc"%(run_name) )

##

his=xr.open_dataset(his_fn)

##

ds0=xr.open_dataset(maps[0])

g0=dfm_grid.DFMGrid(ds0)

##
# StretchCoef       = 8 8 7 7 6 6 6 6 5 5 5 5 5 5 5 5 2 2 1 1 

salt=ds0.sa1.isel(time=1)

plt.figure(3).clf()
#for lay_i in range(len(salt.laydim)):
lay_i=13
salt_layer=salt.isel(laydim=lay_i).values

g0.plot_edges(color='k')
coll=g0.plot_cells(values=salt_layer,mask=np.isfinite(salt_layer),cmap='jet')
coll.set_clim([33.9,34.01])

##

# If I fake the z coord, will it be happy?
n_layer=len(ds0.LayCoord_cc)
ds0.LayCoord_cc.values[:]=np.arange(-n_layer,0) + 0.5
ds0.LayCoord_w.values[:]=np.arange(-n_layer,1) 

##

for v in ds0.data_vars:
    if 'coordinates' in ds0[v].attrs:
        print(v)
        del ds0[v].attrs['coordinates']
##         
ds0.to_netcdf('zlayer_rewrite.nc')
