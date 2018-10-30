# Try to understand how close the hycom velocity field is to
# obeying continuity
# first cut answer is that the data online starts 6/01/2017,
# but the downloader did not catch that, and would return that
# date's data for requested earlier days.
import logging
logging.basicConfig(level=logging.DEBUG)
import xarray.backends.common as xb_common
import matplotlib.pyplot as plt

import numpy as np
from stompy import utils
from stompy.plot import plot_utils
from stompy.grid import unstructured_grid
from stompy.spatial import linestring_utils, proj_utils
from stompy.spatial import field

utils.path("..")
import six

from sfb_dfm_utils import hycom
import xarray as xr

##
six.moves.reload_module(hycom)

run_start=np.datetime64("2017-06-15")
run_stop =np.datetime64("2017-09-20")

hycom_lon_range=[-124.7, -121.7 ]
hycom_lat_range=[36.2, 38.85]
coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]

coastal_files=hycom.fetch_range(hycom_lon_range,hycom_lat_range,coastal_time_range)
# coastal_files=['cache/2017060600--124.70_-121.70_36.20_38.85.nc']

##
if 0: # pre-fetching these elsewhere
    from stompy.io.local import hycom
    cache_dir='cache'
    coastal_files=hycom.fetch_range(hycom_lon_range, hycom_lat_range, coastal_time_range,
                                    cache_dir=cache_dir)

##

# Load the grid to get reasonable approximation to its boundaries
g=unstructured_grid.UnstructuredGrid.read_suntans("runs/sun006",dialect='hybrid')

##

hy_ds=xr.open_dataset(coastal_files[0])

if 'time' in hy_ds.water_u.dims:
    hy_ds=hy_ds.isel(time=0)

# reproject the hycom data

_,Lon,Lat=xr.broadcast(hy_ds.water_u.isel(depth=0),hy_ds.lon,hy_ds.lat)

ll2utm=proj_utils.mapper('WGS84','EPSG:26910')
utm2ll=proj_utils.mapper('EPSG:26910','WGS84')

xy=ll2utm(Lon.values,Lat.values)
hy_ds['x']=Lon.dims,xy[...,0]
hy_ds['y']=Lon.dims,xy[...,1]

##

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

plot_utils.pad_pcolormesh(hy_ds.x,hy_ds.y,hy_ds.water_u.isel(depth=0))
ax.axis('equal')

ecoll=g.plot_edges(color='k',lw=0.5)

##
# bring in hycom bathy:
# load the bathy, and amend the
# given a hycom dataset ds on input, add bathy field, taken from the
# global grid:
fn=coastal_files[0]
sub_ds=xr.open_dataset(fn)

hy_bathy=xr.open_dataset('/home/rusty/cache/hycom/depth_GLBa0.08_09.nc')
lat_min=sub_ds.lat.values.min()
lat_max=sub_ds.lat.values.max()
lon_min=sub_ds.lon.values.min()
lon_max=sub_ds.lon.values.max()

sel=((hy_bathy.Latitude.values>=lat_min) &
     (hy_bathy.Latitude.values<=lat_max) &
     (hy_bathy.Longitude.values>=lon_min) &
     (hy_bathy.Longitude.values<=lon_max))

bathy_xyz=np.c_[ hy_bathy.Longitude.values[sel],
                 hy_bathy.Latitude.values[sel],
                 hy_bathy.bathymetry.isel(MT=0).values[sel] ]
bathy_xyz[:,:2]=ll2utm(bathy_xyz[:,:2])

bathy=field.XYZField(X=bathy_xyz[:,:2],F=bathy_xyz[:,2])

##

# counterclockwise along border, north to south.
nodes_bc=g.select_nodes_boundary_segment([[457184, 4289741],
                                          [592899, 4037040]])

# g.plot_nodes(mask=nodes_bc,ax=ax) # good

##

# rotation matrix to take true east/north vector, rotate to local UTM easting/northing
# centered on given utm x,y
def vec_ll2utm(utm_xy):
    ll0=utm2ll(utm_xy)

    # go a bit more brute force:
    dlat=dlon=0.0001
    ll_east=[ll0[0]+dlon,ll0[1]]
    ll_north=[ll0[0],ll0[1]+dlat]
    utm_east=ll2utm(ll_east) - utm_xy
    utm_north=ll2utm(ll_north) - utm_xy

    utm_to_true_distance_n=1000*utils.haversine(ll0,ll_north) / utils.mag(utm_north)
    utm_to_true_distance_e=1000*utils.haversine(ll0,ll_east) / utils.mag(utm_east)
    print("utm_to_true_distance e=%.5f n=%.5f"%(utm_to_true_distance_e,utm_to_true_distance_n))

    east_norm=utils.to_unit(utm_east)
    # a matrix which left-multiples a true east/north vector, to
    # get an easting/northing vector.  have not yet applied distance
    # correction
    rot=np.array([ [east_norm[0], -east_norm[1]],  #  u_east
                   [east_norm[1], east_norm[0]]])  #  v_north

    # e.g. if 100m in utm is 101m in reality, then a east/north vector gets rotated
    # to align with utm, but then we scale it up, too.
    # temporarily disable to see magnitude of its effect.
    # rot*=utm_to_true_distance
    # gets tricky... top row rot[0,:] yields a utm easting component, which will
    # later be multipled by a northing distance of the flux face.  northing distances
    # should be adjusted to true with utm_to_distance_n.
    # but that made the results slightly worse...
    # it should be multiplication, but trying division...
    rot[0,:] /= utm_to_true_distance_n
    rot[1,:] /= utm_to_true_distance_e
    
    # good news is that angle error in x is same as in y, about 1 degree for this point.
    #print("East is %.3f deg, vs 0"%(np.arctan2(utm_east[1],utm_east[0])*180/np.pi) )
    #print("North is %.3f deg, vs 90"%(np.arctan2(utm_north[1],utm_north[0])*180/np.pi) )
    return rot

##
# find the hycom cell which best matches each edge
# domain.
N_bc_edge=len(nodes_bc)-1
edge_to_hycom=np.zeros( (N_bc_edge,2), np.int32)
edge_hycom_bathy=np.zeros( N_bc_edge, np.float64 )
edge_inward_normal=np.zeros( (N_bc_edge,2), np.float64)
edge_lengths=np.zeros( N_bc_edge, np.float64)
edge_centers=0.5*(g.nodes['x'][nodes_bc[:-1]] + g.nodes['x'][nodes_bc[1:]])

vec_transforms=[None]*N_bc_edge

for i in range(N_bc_edge):
    na,nb=g.nodes['x'][ nodes_bc[i:i+2] ]
    delta=nb-na
    edge_inward_normal[i]=[-delta[1],delta[0]]
    edge_lengths[i]=utils.mag(delta)

    vec_transforms[i]=vec_ll2utm( 0.5*(na+nb) )

    hyc_dists=utils.dist( 0.5*(na+nb), xy )
    row,col=np.nonzero( hyc_dists==hyc_dists.min() )
    edge_to_hycom[i]=[row[0],col[0]]
    edge_hycom_bathy[i]=bathy( xy[row[0],col[0]])
edge_inward_normal /= utils.mag(edge_inward_normal)[:,None]

# those check out.
# ax.scatter( xy[edge_to_hycom[:,0],edge_to_hycom[:,1],0],
#             xy[edge_to_hycom[:,0],edge_to_hycom[:,1],1],
#             40,edge_hycom_bathy,cmap='jet')
# 
# ax.quiver( edge_centers[:,0],edge_centers[:,1],
#            edge_inward_normal[:,0],edge_inward_normal[:,1] )

##

def calc_edge_Q(fn,edge_to_hycom=edge_to_hycom,edge_lengths=edge_lengths,
                edge_inward_normal=edge_inward_normal,
                edge_hycom_bathy=edge_hycom_bathy,
                average_bottom_velocity=True,
                time_idx=0):
    ds=xr.open_dataset(fn)
    if 'time' in ds.dims:
        ds=ds.isel(time=time_idx)

    result=xr.Dataset()
    result['time']=(),ds.time

    u=ds.water_u.values[:, edge_to_hycom[:,0], edge_to_hycom[:,1] ].copy()
    v=ds.water_v.values[:, edge_to_hycom[:,0], edge_to_hycom[:,1] ].copy()
    u_bed=ds.water_u_bottom.values[edge_to_hycom[:,0], edge_to_hycom[:,1] ]
    v_bed=ds.water_v_bottom.values[edge_to_hycom[:,0], edge_to_hycom[:,1] ]

    valid=np.isfinite(u)
    u[~valid]=0.0
    v[~valid]=0.0

    if average_bottom_velocity:
        for j in range(len(edge_to_hycom)):
            k_bed=np.nonzero(valid[:,j])[0][-1]
            u[k_bed,j]=0.5*(u_bed[j] + u[k_bed,j])
            v[k_bed,j]=0.5*(v_bed[j] + v[k_bed,j])

    if edge_hycom_bathy is not None:
        # okay - seems that u is valid only down to the
        # last depth above the bed.  so bottom velocity
        # is what we should apply at edge_hycom_bathy,
        # then integrate the whole thing.
        U=np.zeros(len(edge_to_hycom),np.float64)
        V=np.zeros(len(edge_to_hycom),np.float64)
        for j in range(len(edge_to_hycom)):
            k_bed=1+np.nonzero(valid[:,j])[0][-1]
            u[k_bed,j]=u_bed[j]
            v[k_bed,j]=v_bed[j]
            my_depths=ds.depth.values.clip(0,edge_hycom_bathy[j])
            if 0: # heavy handed! and doesn't make much difference.
                my_depths[k_bed]=edge_hycom_bathy[j]

            # Maybe not quite getting the right selection of bathy points here.
            # about 80% are correct though.
            if 0:
                if my_depths[k_bed]!=edge_hycom_bathy[j]:
                    print("Depth mismatch: bed depth %.1f  bathy %.1f"%(my_depths[k_bed],edge_hycom_bathy[j]))
                else:
                    print("Depth    match: bed depth %.1f  bathy %.1f"%(my_depths[k_bed],edge_hycom_bathy[j]))

            # transform the vectors here:
            
            UV=np.array( [np.trapz(u[:k_bed+1,j],my_depths[:k_bed+1]),
                          np.trapz(v[:k_bed+1,j],my_depths[:k_bed+1])] )
            UV=np.dot(vec_transforms[j],UV)
            U[j]=UV[0]
            V[j]=UV[1]
    else:
        # ds.depth starts at 0, and there are no velocities at
        # the deep end of it.  so extend the finite difference
        # on the end.
        dz=np.diff(ds.depth.values)
        dz=np.concatenate( [dz,dz[-1:]] )
        dz=dz[:,None]

        U=(u*dz).sum(axis=0)
        V=(v*dz).sum(axis=0)

    # U is east, which may not be exactly the same as easting in the UTM grid.
    # rotate U,V to be parallel to the UTM coordinates, rather than cartesian
    # east/north
    # may also need to adjust for distortion of edge_lengths
    edge_Q=edge_lengths*(U*edge_inward_normal[:,0] + V*edge_inward_normal[:,1])
    result['edge_Q']=('edge',),edge_Q

    ds.close()

    return result


# extract transect from one file:
# looks like the first 11 files are identical?
edge_dss=[]
for fn in coastal_files:
    print(fn)
    #edge_ds=calc_edge_Q(fn,edge_hycom_bathy=None)
    edge_ds=calc_edge_Q(fn)
    edge_ds['total_Q']=(),edge_ds.edge_Q.sum()
    edge_dss.append(edge_ds)

Q_series=xr.concat(edge_dss,dim='time')

##

# How does that compare to tides at Point Reyes?
from stompy.io.local import noaa_coops
cache_dir='cache'
os.path.exists(cache_dir) or os.makedirs(cache_dir)

ds=noaa_coops.coops_dataset_product(9415020,"water_level",
                                    coastal_time_range[0],coastal_time_range[1],
                                    days_per_request='M',cache_dir=cache_dir)

##
# This is not convincing me that the variation in net flux is aliasing the
# tides.
plt.figure(6).clf()
fig,axs=plt.subplots(3,1,num=6,sharex=True)
axs[0].plot(Q_series.time,Q_series.total_Q,label="net Q")
axs[0].legend()
axs[1].plot(ds.time,ds.water_level.isel(station=0),label="PR tides")
axs[1].plot(Q_series.time,Q_series.total_Q*1e-6+1.5,'ro',label="~ net Q")
axs[1].legend()

# integrate and normalize to SSH
A_total=g.cells_area().sum()
axs[2].plot(Q_series.time,np.cumsum(Q_series.total_Q)*86400/A_total,
            label="cumul. $\Delta \eta$")
axs[2].set_ylabel('m')
axs[2].legend()


##
fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

img=plt.imshow( ds.water_u.isel(depth=0).values, extent=extents,
                origin='bottom')
ax.set_aspect( 1./np.cos(ds.lat.mean()*np.pi/180.))

