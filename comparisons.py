import os
import numpy as np
import xarray as xr
import datetime

from sfb_dfm_utils import ca_roms

from stompy import utils
from stompy.spatial import wkb2shp, proj_utils
from stompy.io.local import noaa_coops
import stompy.model.delft.io as dio

############ COMPARISONS ##################

run_name="short_test_18"
run_base_dir="runs/%s"%run_name
mdu_fn=os.path.join(run_base_dir,"%s.mdu"%run_name)

mdu=dio.MDUFile(mdu_fn)

##


utm2ll=proj_utils.mapper('EPSG:26910',"WGS84")

## 
# Comparisons across ROMS, NOAA tides, NOAA ADCPs, and this model

obs_pnts=wkb2shp.shp2geom("/opt/data/delft/sfb_dfm_v2/inputs-static/observation-points.shp")

##
def load_subroms():
    ca_roms_files = ca_roms.fetch_ca_roms(run_start,run_stop)

    sub_vars=['zeta','salt']

    ds=xr.open_dataset(ca_roms_files[0])

    lat_idxs=np.searchsorted(ds.lat,[dfm_ll[:,1].min(), dfm_ll[:,1].max()])
    lon_idxs=np.searchsorted(ds.lon%360,[dfm_ll[:,0].min()%360, dfm_ll[:,0].max()%360])
    lat_slice=slice(lat_idxs[0],1+lat_idxs[1])
    lon_slice=slice(lon_idxs[0],1+lon_idxs[1])

    sub_roms=[]

    for fn in ca_roms_files:
        print(fn)
        ds=xr.open_dataset(fn)
        for v in ds.data_vars:
            if v not in sub_vars:
                del ds[v]
        sub_ds=ds.isel(lat=lat_slice,lon=lon_slice)
        # Fix goofy ROMS timestamps
        sub_ds.time.values[0] = utils.to_dt64( datetime.datetime.strptime(sub_ds.title,"CA-%Y%m%d%H") )
        sub_roms.append(sub_ds)

    src=xr.concat(sub_roms,dim='time')
    return src
src=load_subroms()
## 
def roms_davg(val):
    dim=val.get_axis_num('depth')
    dz=utils.center_to_interval(val.depth.values)
    weighted= np.nansum( (val*dz).values, axis=dim )
    unit = np.sum( np.isfinite(val)*dz, axis=dim)
    return weighted / unit

# Compare tides at point reyes:

_,t_start,t_stop=mdu.time_range()

pr_tides_noaa_fn="noaa_9415020-%s_%s.nc"%(utils.to_datetime(t_start).strftime('%Y%m%d'),
                                          utils.to_datetime(t_stop).strftime('%Y%m%d'))
if not os.path.exists(pr_tides_noaa_fn):
    ds=noaa_coops.coops_dataset("9415020",t_start,t_stop,["water_level"])
    ds.to_netcdf(pr_tides_noaa_fn)
    ds.close()
pr_tides_noaa=xr.open_dataset(pr_tides_noaa_fn)

feat=obs_pnts[ np.nonzero( obs_pnts['name']=='NOAA_PointReyes' )[0][0] ]
xy=np.array(feat['geom'])
ll=utm2ll(xy)




ll_idx=[np.searchsorted( src.lon%360, ll[0]%360 ),
        np.searchsorted( src.lat, ll[1] ) ]
##

#dfm_map=xr.open_dataset(os.path.join(run_base_dir,
#                                     "DFM_OUTPUT_%s"%run_name,
#                                     "%s_map.nc"%run_name))
dfm_map=xr.open_dataset('short_test_06_map.nc')

dfm_map_g=dfm_grid.DFMGrid(dfm_map)

dfm_cc=dfm_map_g.cells_centroid()
dfm_ll=utm2ll(dfm_cc)

pr_cell_idx=dfm_map_g.select_cells_nearest(xy)

##

# Get some salinity data to compare to:

roms_off=1.2
dfm_off=0.0

salt_var='sa1' # mesh2d_sa1 in ugrid output
eta_var='s1' # mesh2d_s1 ...
cell_dim='nFlowElem' # or 'nmesh2d_face'

plt.figure(21).clf()
fig,(ax,ax_s)=plt.subplots(2,1,num=21,sharex=True)
ax.plot(utils.to_dnum(pr_tides_noaa.time),
        pr_tides_noaa.water_level.isel(station=0),
        label='NOAA')
ax.plot(utils.to_dnum(dfm_map.time),
        dfm_map[eta_var].isel(**{cell_dim:pr_cell_idx}) + dfm_off,
        label='DFM+%.1f'%dfm_off)
ax.plot(utils.to_dnum(src.time),
        src.zeta.isel(lon=ll_idx[0],lat=ll_idx[1])+ roms_off,
        'ro',label='ROMS+%.1f'%roms_off)
ax.set_ylabel('Water level (m NAVD88)')

if salt_var in dfm_map:
    ax_s.plot(utils.to_dnum(dfm_map.time),
              dfm_map[salt_var].isel(**{cell_dim:pr_cell_idx}),
              label='DFM')
    
ax_s.plot(utils.to_dnum(src.time),
          roms_davg(src.salt.isel(lon=ll_idx[0],lat=ll_idx[1])),
          'ro',label='ROMS')
ax_s.set_ylabel('Salinity')


ax.axis(xmin=utils.to_dnum(t_start),
        xmax=utils.to_dnum(t_stop))

ax.legend(fontsize=8)

##

g_map_ll=g_map.copy()
g_map_ll.nodes['x'] = utm2ll( g_map.nodes['x'] )
g_map_ll.nodes['x'][:,0] %= 360.

##


plt.figure(22).clf()
fig,axs=plt.subplots(1,2,num=22,sharex=True,sharey=True)

## 
for i in [190]: # range(0,len(dfm_map.time),18):
    # Compare surface salinity over time:
    compare_time=dfm_map.time[i]
    # dfm_tidx=np.searchsorted( dfm_map.time
    dfm_salt=dfm_map.sa1.sel(time=compare_time,laydim=-1)

    roms_tidx=np.searchsorted(utils.to_dnum(src.time.values),
                              utils.to_dnum(compare_time))
    roms_salt=src.salt.isel(time=roms_tidx,depth=0)

    axs[0].cla()
    axs[1].cla()
    
    axs[0].axis('equal')

    ccoll=g_map_ll.plot_cells(ax=axs[1],values=dfm_salt,lw=0.5)
    ccoll.set_edgecolor('face')

    rcoll=axs[0].pcolormesh( src.lon, src.lat, roms_salt )

    ccoll.set_clim( rcoll.get_clim() )
    plt.pause(0.05)

##

lat_idxs=g.edges['src_idx_out'][:,0]
lon_idxs=g.edges['src_idx_out'][:,1]

lat_idxs=lat_idxs[ lat_idxs>= 0 ]
lon_idxs=lon_idxs[ lon_idxs>= 0 ]
lat_slice=slice(lat_idxs.min(),lat_idxs.max())
lon_slice=slice(lon_idxs.min(),lon_idxs.max())

# Compare map of velocity snapshots
plt.figure(22).clf()
fig,ax=plt.subplots(num=22)

for t_idx in range(30):
    slice_u=src_davg( src['u'].isel(time=t_idx,lat=lat_slice,lon=lon_slice) )
    slice_v=src_davg( src['v'].isel(time=t_idx,lat=lat_slice,lon=lon_slice) )
    Lon,Lat=np.meshgrid(src.lon.values[lon_slice],
                        src.lat.values[lat_slice])

    ax.cla()
    qset=ax.quiver(Lon.ravel(),
                   Lat.ravel(),
                   slice_u.values.ravel(),
                   slice_v.values.ravel() )

    ax.quiverkey(qset,0.8,0.8,0.5,coordinates='axes',label='0.5 m/s')
    plt.pause(0.05)
