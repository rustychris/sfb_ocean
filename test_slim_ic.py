"""
Try removing some of the fields in the initial condition to see if that will improve
the z-layer model starting up.

This led to figuring out that unorm in the restart files is bad news.
"""
import glob
import os
import xarray as xr

ic_maps=glob.glob(os.path.join('runs/short_test_12/initial_conditions_*_map.nc'))

ic_maps_out=[fn.replace('short_test_12','short_test_12_test')
             for fn in ic_maps]

## 


for i,ic_map_fn in enumerate(ic_maps):
    ic_map=xr.open_dataset(ic_maps[0])

    for field in [
            #'LayCoord_cc','LayCoord_w', # made no difference
            #'windx','windy','windxu','windyu',# no difference
            # 'ucx','ucy','ucz','ucxa','ucya',
            #'sa1','tem1','turkin1','vicwwu','tureps1',
            # 'Patm',  # no difference...
            # 's1','waterdepth','numlimdt',
            'unorm',
            # 'viu'
    ]:
        del ic_map[field]
    
    ic_map.to_netcdf(ic_maps_out[i],format='NETCDF3_64BIT')

