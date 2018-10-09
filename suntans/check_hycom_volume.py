# Try to understand how close the hycom velocity field is to
# obeying continuity
# first cut answer is that the data online starts 6/01/2017,
# but the downloader did not catch that, and would return that
# date's data for requested earlier days.
import numpy as np
from stompy import utils
utils.path("..")
import six

from sfb_dfm_utils import hycom
import xarray as xr

##
six.moves.reload_module(hycom)

run_start=np.datetime64("2017-06-01")
run_stop =np.datetime64("2017-06-04")

hycom_lon_range=[-124.7, -121.7 ]
hycom_lat_range=[36.2, 38.85]
coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]
coastal_files=hycom.fetch_range(hycom_lon_range,hycom_lat_range,coastal_time_range)

##

ds=xr.open_dataset(coastal_files[4])

##

extents=[ds.lon[0], ds.lon[-1], ds.lat[0],ds.lat[-1] ]

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

img=plt.imshow( ds.water_u.isel(depth=0).values, extent=extents,
                origin='bottom')
ax.set_aspect( 1./np.cos(ds.lat.mean()*np.pi/180.))

##

# define a transect,
from stompy.plot import plot_utils
from stompy.spatial import linestring_utils, proj_utils

tran=np.array([[-121.87058419,   36.35840629],
               [-124.44609969,   36.35840629],
               [-124.4505556 ,   38.6026    ],
               [-123.35885785,   38.61673666]])


tran_samp=linestring_utils.upsample_linearring(tran,0.005,closed_ring=0)
tran_mid=0.5*(tran_samp[:-1]+tran_samp[1:])
# even though these are degrees, can get close for normals this way,
# esp. with lat correction
tran_diff=(tran_samp[1:]-tran_samp[:-1])
tran_inward=np.c_[tran_diff[:,1],-tran_diff[:,0]*np.cos(tran_mid[:,1]*np.pi/180.)]
mags=utils.mag(tran_inward)
tran_inward/=mags[:,None]

# tran_samp_xy=proj_utils.mapper('WGS84','EPSG:26910')

tran_lon_i=utils.nearest( ds.lon.values, tran_mid[:,0] )
tran_lat_i=utils.nearest( ds.lat.values, tran_mid[:,1] )

# 1000: km=>m
dseg=1000*utils.haversine( tran_samp[:-1], tran_samp[1:])

##

# looks like the first 11 files are identical?
total_Qs=[]

for fn in coastal_files:
    ds=xr.open_dataset(fn)

    # convert u to transport U
    u=ds.water_u.values.copy()
    valid=np.isfinite(u)
    u[~valid]=0.0
    v=ds.water_v.values.copy()
    v[~valid]=0.0

    # ds.depth starts at 0, and there are no velocities at
    # the deep end of it.  so extend the finite difference
    # on the end.
    dz=np.diff(ds.depth.values)
    dz=np.concatenate( [dz,dz[-1:]] )

    U=(u*dz[:,None,None]).sum(axis=0)
    V=(v*dz[:,None,None]).sum(axis=0)

    tran_U=U[tran_lat_i,tran_lon_i]
    tran_V=V[tran_lat_i,tran_lon_i]

    tran_Q=dseg*(tran_U*tran_inward[:,0] + tran_V*tran_inward[:,1])

    total_Q=tran_Q.sum()
    total_Qs.append(total_Q)

##

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

img=plt.imshow( ds.water_u.isel(depth=0).values, extent=extents,
                origin='bottom')
ax.set_aspect( 1./np.cos(ds.lat.mean()*np.pi/180.))

