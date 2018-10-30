from stompy.model.suntans import sun_driver

import six
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy.spatial import proj_utils, field, wkb2shp
from stompy.model.delft import dfm_grid
import stompy.model.delft.io as dio
from stompy.model import otps
from stompy import utils, filters

import logging as log

utils.path("../")
#from sfb_dfm_utils import hycom
from stompy.io.local import hycom
cache_dir='cache'

from stompy.grid import unstructured_grid

from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv
##
six.moves.reload_module(dfm)
six.moves.reload_module(dfm_grid)
six.moves.reload_module(drv)

##

read_otps.OTPS_DATA='../derived'

use_temp=False
use_salt=True
ocean_method='hycom'

# sun001: 25 layers, 1.05 stretch, stairstep
# sun002: 1.08 stretch, no stairstep, no mpi
# sun003: zero salt/temp, and then static HYCOM
# sun004: longer. tidal fluxes, then tidal velocity
# sun005: hycom flows
# sun006: hycom flows, 1 month run
run_dir='runs/sun007'
run_start=np.datetime64("2017-06-15")
run_stop =np.datetime64("2017-09-10")

model=drv.SuntansModel()
model.projection="EPSG:26910"
model.num_procs=1
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.load_template("sun-template.dat")

model.set_run_dir(run_dir,mode='pristine')
model.config['Nkmax']=25
# would like to relax this ASAP
model.config['stairstep']=0

dt_secs=60
model.config['dt']=dt_secs
# quarter-hour map output:
model.config['ntout']=int(15*60/dt_secs)
# daily restart file
model.config['ntoutStore']=int(86400/dt_secs)
model.config['mergeArrays']=0
model.config['rstretch']=1.08

# these were scaled down by 1e-3 for debugging
if use_temp:
    model.config['gamma']=0.00021
else:
    model.config['gamma']=0.0

if use_salt:
    model.config['beta']=0.00077
else:
    model.config['beta']=0.0

model.run_start=run_start
model.run_stop=run_stop

# dest_grid="../derived/matched_grid_v01.nc"
dest_grid="../ragged/grid_v01.nc"
assert os.path.exists(dest_grid),"Grid %s not found"%dest_grid
model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid(dest_grid))

model.add_gazetteer("linear_features.shp")

#--

# HYCOM
class HycomMultiVelocityBC(drv.MultiBC):
    """
    Special handling of multiple hycom boundary segments to
    enforce specific net flux requirements.
    Otherwise small errors, including quantization and discretization,
    lead to a net flux.
    """
    # according to website, hycom runs for download are non-tidal, so
    # don't worry about filtering
    # Data is only daily, so go a bit longer than a usual tidal filter
    lp_hours=0
    pad=np.timedelta64(4,'D')

    def __init__(self,ll_box=None,**kw):
        self.ll_box=ll_box
        self.data_files=None
        super(HycomMultiVelocityBC,self).__init__(self.VelocityProfileBC,**kw)

    class VelocityProfileBC(drv.VelocityBC):
        _dataset=None # supplied by factory
        def dataset(self):
            return self._dataset

    # def factory(self,model,**sub_kw):
    #     """
    #     kludgey -- originally cls was a class attribute, but for hycom
    #     multibc we have a stronger tie between the individual BCs,
    #     so this is changed to a method on the MultiBC instance.
    #     """
    #     print("got the call to cls")
    #     # sub_kw should include geom, name, grid_edge, grid_cell
    #     raise Exception('Here is where to pull out a water column')

    def enumerate_sub_bcs(self):
        if self.ll_box is None:
            # grid=self.model.grid ...
            raise Exception("Not implemented: auto-calculating ll_box")
        self.populate_files()
        super(HycomMultiVelocityBC,self).enumerate_sub_bcs()
    
        # may decide that right after the stock enumerate_sub_bcs is the right
        # place to adjust fluxes...
        self.populate_velocity()

    def populate_files(self):
        data_start=self.model.run_start-self.pad
        data_stop =self.model.run_stop+self.pad
        self.data_files=hycom.fetch_range(self.ll_box[:2],self.ll_box[2:],
                                          [data_start,data_stop],
                                          cache_dir=cache_dir)

    def init_bathy(self):
        """
        populate self.bathy, an XYZField in native coordinates, with
        values as hycom's positive down bathymetry.
        """
        # TODO: download hycom bathy on demand.
        self.hy_bathy=xr.open_dataset( os.path.join(cache_dir,'depth_GLBa0.08_09.nc') )
        lon_min,lon_max,lat_min,lat_max=self.ll_box

        sel=((hy_bathy.Latitude.values>=lat_min) &
             (hy_bathy.Latitude.values<=lat_max) &
             (hy_bathy.Longitude.values>=lon_min) &
             (hy_bathy.Longitude.values<=lon_max))

        bathy_xyz=np.c_[ hy_bathy.Longitude.values[sel],
                         hy_bathy.Latitude.values[sel],
                         hy_bathy.bathymetry.isel(MT=0).values[sel] ]
        bathy_xyz[:,:2]=ll2utm(bathy_xyz[:,:2])

        self.bathy=field.XYZField(X=bathy_xyz[:,:2],F=bathy_xyz[:,2])

    def populate_velocity(self):
        """ Do the actual work of iterating over sub-edges and hycom files,
        interpolating in the vertical, projecting as needed, and adjust the overall
        fluxes
        """
        # The net inward flux in m3/s over the whole BC that we will adjust to.
        target_Q=np.zeros(len(self.data_files)) # assumes one time step per file

        # Get spatial information about hycom files
        hy_ds0=xr.open_dataset(self.data_files[0])
        if 'time' in hy_ds0.water_u.dims:
            hy_ds0=hy_ds0.isel(time=0)
        # makes sure lon,lat are compatible with water velocity
        _,Lon,Lat=xr.broadcast(hy_ds0.water_u.isel(depth=0),hy_ds0.lon,hy_ds0.lat)
        hy_xy=self.model.ll_to_native(Lon.values,Lat.values)

        self.init_bathy()

        # Initialize per-edge details
        self.model.grid._edge_depth=self.model.grid.edges['edge_depth']
        layers=self.model.layer_data(with_offset=True)

        for i,sub_bc in enumerate(self.sub_bcs):
            sub_bc.inward_normal=sub_bc.get_inward_normal()
            sub_bc.edge_length=sub_bc.geom.length
            sub_bc.edge_center=np.array(sub_bc.geom.centroid)

            # skip the transforms...
            hyc_dists=utils.dist( sub_bc.edge_center, hy_xy )
            row,col=np.nonzero( hyc_dists==hyc_dists.min() )
            row=row[0] ; col=col[0]
            sub_bc.hy_row_col=(row,col) # tuple, so it can be used directly in []

            # initialize the datasets
            sub_bc._dataset=sub_ds=xr.Dataset()
            # assumes that from each file we use only one timestep
            sub_ds['time']=('time',), np.ones(len(self.data_files),'M8[m]')
            # getting tricky here - do more work here rather than trying to push ad hoc interface
            # into the model class
            # velocity components in UTM x/y coordinate system
            sub_ds['u']=('time','layer'), np.zeros((sub_ds.dims['time'],layers.dims['Nk']),
                                                   np.float64)
            sub_ds['v']=('time','layer'), np.zeros((sub_ds.dims['time'],layers.dims['Nk']),
                                                   np.float64)
            # depth-integrated transport on suntans layers, in m2/s
            sub_ds['Uint']=('time',), np.nan*np.ones(sub_ds.dims['time'],np.float64)
            sub_ds['Vint']=('time',), np.nan*np.ones(sub_ds.dims['time'],np.float64)
            # project transport to edge normal * edge_length to get m3/s
            sub_ds['Q_in']=('time',), np.nan*np.ones(sub_ds.dims['time'],np.float64)

            sub_bc.edge_depth=edge_depth=self.model.grid.edge_depths()[sub_bc.grid_edge] # positive up

            # First, establish the geometry on the suntans side, in terms of z_interface values
            # for all wet layers.  below-bed layers have zero vertical span.  positive up, but
            # shift back to real, non-offset, vertical coordinate
            sun_z_interface=(-self.model.z_offset)+layers.z_interface.values.clip(edge_depth,np.inf)
            sub_bc.sun_z_interfaces=sun_z_interface
            # And the pointwise data from hycom:
            hy_layers=hy_ds0.depth.values.copy()
            sub_bc.hy_valid=valid=np.isfinite(hy_ds0.water_u.isel(lat=row,lon=col).values)
            hycom_depths=hy_ds0.depth.values[valid]
            # possible that hy_bed_depth is not quite correct, and hycom has data
            # even deeper.  in that case just pad out the depth a bit so there
            # is at least a place to put the bed velocity.
            if len(hycom_depths)!=0:
                sub_bc.hy_bed_depth=max(hycom_depths[-1]+1.0,bathy(hy_xy[sub_bc.hy_row_col]))
                sub_bc.hycom_depths=np.concatenate( [hycom_depths, [sub_bc.hy_bed_depth]])
            else:
                # edge is dry in HYCOM -- be careful to check and skip below.
                sub_bc.hycom_depths=hycom_depths
                sub_bc._dataset['u'].values[:]=0.0
                sub_bc._dataset['v'].values[:]=0.0
                sub_bc._dataset['Uint'].values[:]=0.0
                sub_bc._dataset['Vint'].values[:]=0.0

        # Populate the velocity data, outer loop is over hycom files, since
        # that's most expensive
        for ti,fn in enumerate(self.data_files):
            hy_ds=xr.open_dataset(fn)
            if 'time' in hy_ds.dims:
                # again, assuming that we only care about the first time step in each file
                hy_ds=hy_ds.isel(time=0)
            print(hy_ds.time.values)

            water_u=hy_ds.water_u.values
            water_v=hy_ds.water_v.values
            water_u_bottom=hy_ds.water_u_bottom.values
            water_v_bottom=hy_ds.water_v_bottom.values

            for i,sub_bc in enumerate(self.sub_bcs):
                hy_depths=sub_bc.hycom_depths
                sub_bc._dataset.time.values[ti]=hy_ds.time.values
                if len(hy_depths)==0:
                    continue # already zero'd out above.
                row,col=sub_bc.hy_row_col
                z_sel=sub_bc.hy_valid

                sun_dz=np.diff(-sub_bc.sun_z_interfaces)
                sun_valid=sun_dz>0
                for water_vel,water_vel_bottom,sun_var,trans_var in [ (water_u,water_u_bottom,'u','Uint'),
                                                                      (water_v,water_v_bottom,'v','Vint') ]:
                    sub_water_vel=np.concatenate([ water_vel[z_sel,row,col],
                                                   water_vel_bottom[None,row,col] ])

                    # integrate -- there isn't a super clean way to do this that I see.
                    # but averaging each interval is probably good enough, just loses some vertical
                    # accuracy.
                    interval_mean_vel=0.5*(sub_water_vel[:-1]+sub_water_vel[1:])
                    veldz=np.concatenate( ([0],np.cumsum(np.diff(hy_depths)*interval_mean_vel)) )
                    sun_veldz=np.interp(-sub_bc.sun_z_interfaces, hy_depths, veldz)
                    sun_d_veldz=np.diff(sun_veldz)

                    sub_bc._dataset[sun_var].values[ti,sun_valid]=sun_d_veldz[sun_valid]/sun_dz[sun_valid]
                    # might as well calculate flux while we are here
                    # explicit flux:
                    # sub_bc._dataset[trans_var].values[ti]=(sub_bc._dataset[sun_var]*sun_dz).sum()
                    # but we've already done the integration
                    sub_bc._dataset[trans_var].values[ti]=sun_veldz[-1]
            hy_ds.close() # free up netcdf resources

        # project transport onto edges to get fluxes
        total_Q=0.0
        total_flux_A=0.0
        for i,sub_bc in enumerate(self.sub_bcs):
            Q_in=sub_bc.edge_length*(sub_bc.inward_normal[0]*sub_bc._dataset['Uint'].values +
                                     sub_bc.inward_normal[1]*sub_bc._dataset['Vint'].values)
            sub_bc._dataset['Q_in'].values[:]=Q_in
            total_Q=total_Q+Q_in
            # edge_depth here reflects the expected water column depth.  it is the bed elevation, with
            # the z_offset removed (I hope), under the assumption that a typical eta is close to 0.0,
            # but may be offset as much as -10.
            total_flux_A+=sub_bc.edge_length*sub_bc.edge_depth

        Q_error=total_Q-target_Q
        vel_error=Q_error/total_flux_A
        print("Velocity error: %.6f -- %.6f m/s"%(vel_error.min(),vel_error.max()))

        # And apply the adjustment:
        # note that Q_in, Uint, Vint are no longer valid
        for i,sub_bc in enumerate(self.sub_bcs):
            sub_bc._dataset['u'].values[:,:] -= vel_error[:,None]*sub_bc.inward_normal[0]
            sub_bc._dataset['v'].values[:,:] -= vel_error[:,None]*sub_bc.inward_normal[1]

# spatially varying
hycom_ll_box=[-124.7, -121.7, 36.2, 38.85]

if ocean_method=='eta':
    ocean_bc=drv.MultiBC(drv.OTPSStageBC,name='Ocean',otps_model='wc')
elif ocean_method=='flux':
    ocean_bc=drv.MultiBC(drv.OTPSFlowBC,name='Ocean',otps_model='wc')
elif ocean_method=='velocity':
    ocean_bc=drv.MultiBC(drv.OTPSVelocityBC,name='Ocean',otps_model='wc')
elif ocean_method=='hycom':
    # explicity give bounds to make sure we always download the same
    # subset.
    ocean_bc=HycomMultiVelocityBC(ll_box=hycom_ll_box,
                                  name='Ocean')

    # leftover -- clean out once new code is working
    #coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
    #coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]
    #coastal_files=hycom.fetch_range(hycom_lon_range,hycom_lat_range,coastal_time_range)

model.add_bcs(ocean_bc)

## 
model.write()

##--

# Initial condition:
# map each cell to a hycom
fns=hycom.fetch_range(hycom_ll_box[:2],hycom_ll_box[2:],
                               [model.run_start,model.run_start+np.timedelta64(1,'D')],
                               cache_dir=cache_dir)
hycom_ic_fn=fns[0]

hycom_ds=xr.open_dataset(hycom_ic_fn)
if 'time' in hycom_ds.dims:
    hycom_ds=hycom_ds.isel(time=0)
cc=model.grid.cells_center()
cc_ll=model.native_to_ll(cc)

dlat=np.median(np.diff(hycom_ds.lat.values))
dlon=np.median(np.diff(hycom_ds.lon.values))
lat_i = utils.nearest(hycom_ds.lat.values,cc_ll[:,1],max_dx=1.2*dlat)
lon_i = utils.nearest(hycom_ds.lon.values,cc_ll[:,0],max_dx=1.2*dlon)

# make this positive:down to match hycom and make the interpolation
sun_z = -model.ic_ds.z_r.values

default_s=33.4 # would be nice to pull a nominal shallow value from HYCOM
assert ('time', 'Nk', 'Nc') == model.ic_ds.salt.dims,"Workaround is fragile"

for c in range(model.grid.Ncells()):
    sun_s=default_s
    if lat_i[c]<0 or lon_i[c]<0:
        print("Cell %d does not overlap HYCOM grid"%c)
    else:
        # top to bottom, depth positive:down
        s_profile=hycom_ds.salinity.isel(lon=lon_i[c],lat=lat_i[c])
        s_profile=s_profile.values
        valid=np.isfinite(s_profile)
        if not np.any(valid):
            print("Cell %d is dry in HYCOM grid"%c)
        else:
            # could add bottom salinity if we really cared.
            sun_s=np.interp( sun_z,
                             hycom_ds.depth.values[valid], s_profile[valid] )
            # if Nk wasn't broken:
            #model.ic_ds.salt.isel(time=0,Nc=c).values[:]=sun_s
    model.ic_ds.salt.values[0,:,c]=sun_s

model.write_ic_ds()

#----

model.copy_ic_to_bc('salt','S')
model.write_bc_ds()

#---
model.partition()

model.run_simulation()

# This actually appears to run.
# it has no freesurface forcing, a little scary.
# And it runs!  quite nicely.
##


