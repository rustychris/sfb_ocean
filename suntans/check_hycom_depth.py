import xarray as xr

ds=xr.open_dataset('/home/rusty/cache/hycom/depth_GLBa0.08_09.nc')

hycom_lon_range=[-124.7, -121.7 ]
hycom_lat_range=[36.2, 38.85]

##

# plot bathy just as points -
sel=((ds.Latitude.values>=hycom_lat_range[0]) &
     (ds.Latitude.values<=hycom_lat_range[1]) &
     (ds.Longitude.values>=hycom_lon_range[0]) &
     (ds.Longitude.values<=hycom_lon_range[1]) )

##
plt.figure(2).clf()
scat=plt.scatter( ds.Longitude.values[sel], ds.Latitude.values[sel],
                  30,ds.bathymetry.isel(MT=0).values[sel],cmap='jet' )

# this is positive down, meters. farallons are blank.
