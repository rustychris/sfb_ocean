#  any hope of opendap helping us out?
from stompy.model.delft import dfm_grid
from stompy.grid import unstructured_grid
from stompy import utils
import xarray as xr
import matplotlib.pyplot as plt

from sfb_dfm_utils import hycom
six.moves.reload_module(hycom)

## 
lon_range=[-124.7, -121.7 ]
lat_range=[36.2, 38.85]
time_range=[np.datetime64('2017-06-01'),
            np.datetime64('2017-11-01')]

## 

hycom_files=hycom.fetch_range( lon_range=lon_range,lat_range=lat_range,time_range=time_range )

coastal_ds=xr.open_dataset(hycom_files[0])
                   
##  

g=unstructured_grid.UnstructuredGrid.from_ugrid('spliced_grids_01.nc')

g_ll=g.reproject('EPSG:26910','WGS84')

##

plt.figure(10).clf()
fig,ax=plt.subplots(num=10)

g_ll.plot_edges(color='k',lw=0.4,ax=ax)

ax.pcolormesh( coastal_ds.lon.values, coastal_ds.lat.values,
               coastal_ds.surf_el.values )

## 

# start and end of chunk of grid boundary,UTM, which is driven by
# coastal model.
coastal_bc_coords=[ [450980., 4291405.], # northern
                    [595426., 4037083.] ] # southern

## 

plt.figure(11).clf()
fig,ax=plt.subplots(num=11)

g.plot_edges(lw=0.5,color='k',ax=ax)

ax.plot( g.nodes['x'][boundary_nodes,0],
         g.nodes['x'][boundary_nodes,1],'g-',lw=1.5)

## 


def ll_to_lonilati(ll,coastal_ds):
    """
    Match lon/lat coordinates to a cell in the coastal_ds
    dataset.
    ll: [lon,lat]
    coastal_ds: xarray dataset, assumed to be lon/lat grid, rectilinear.
    """
    loni=utils.nearest(snap_lon%360,g_map_ll[c,0]%360)
    lati=utils.nearest(snap_lat,g_map_ll[c,1])
    err_km=utils.haversine([snap_lon[loni],snap_lat[lati]],
                           g_map_ll[c,:])
    


# HERE - instead of reinventing here, generalize the ROMS 
# code.  even if that means repeated BC sources.
    
# match contiguous runs of nodes/edges with sources
# of hycom/roms cells
probes=[]

for na,nb in zip( boundary_nodes[:-1],
                  boundary_nodes[1:] ):
    jab=g.nodes_to_edge([na,nb])
    vec_AB=utils.rot(-np.pi/2, g_ll.nodes['x'][nb] - g_ll.nodes['x'][na])
    probe_xy= g_ll.nodes['x'][ [na,nb] ].mean(axis=0) + 0.5*vec_AB
    probes.append(probe_xy)

    
