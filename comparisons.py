
############ COMPARISONS ##################

# Comparisons across ROMS, NOAA tides, NOAA ADCPs, and this model

obs_pnts=wkb2shp.shp2geom("/opt/data/delft/sfb_dfm_v2/inputs-static/observation-points.shp")

## 

# Compare tides at point reyes:
from stompy.io.local import noaa_coops

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

dfm_map=xr.open_dataset(os.path.join(run_base_dir,
                                     "DFM_OUTPUT_%s"%run_name,
                                     "%s_map.nc"%run_name))

dfm_map_g=dfm_grid.DFMGrid(dfm_map)

pr_cell_idx=dfm_map_g.select_cells_nearest(xy)

##

roms_off=1.2
dfm_off=0.0

plt.figure(21).clf()
fig,(ax,ax_s)=plt.subplots(2,1,num=21,sharex=True)
ax.plot(utils.to_dnum(pr_tides_noaa.time),
        pr_tides_noaa.water_level.isel(station=0),
        label='NOAA')
ax.plot(utils.to_dnum(dfm_map.time),
        dfm_map.mesh2d_s1.isel(nmesh2d_face=pr_cell_idx) + dfm_off,
        label='DFM+%.1f'%dfm_off)
ax.plot(utils.to_dnum(src.time),
        src.zeta.isel(lon=ll_idx[0],lat=ll_idx[1])+ roms_off,
        'ro',label='ROMS+%.1f'%roms_off)
ax.set_ylabel('Water level (m NAVD88)')

if 'mesh2d_sa1' in dfm_map:
    ax_s.plot(utils.to_dnum(dfm_map.time),
              dfm_map.mesh2d_sa1.isel(nmesh2d_face=pr_cell_idx),
              label='DFM')
ax_s.plot(utils.to_dnum(src.time),
          src_davg(src.salt.isel(lon=ll_idx[0],lat=ll_idx[1])),
          'ro',label='ROMS')
ax_s.set_ylabel('Salinity')


ax.axis(xmin=utils.to_dnum(t_start),
        xmax=utils.to_dnum(t_stop))

ax.legend(fontsize=8)

##

if 0:
    # Plot up the boundary conditions:
    plt.figure(21).clf()
    fig,ax=plt.subplots(num=21)

    for t_idx in range(len(otps_u.time)):
        ax.cla()
        qset=ax.quiver( boundary_out_ll[:,0],
                        boundary_out_ll[:,1],
                        otps_u.result.isel(time=t_idx),
                        otps_v.result.isel(time=t_idx) )
        ax.quiverkey(qset,0.8,0.8,0.1,coordinates='axes',label='0.1 m/s')
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
