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
from sfb_dfm_utils import hycom
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
run_dir='runs/sun006'
run_start=np.datetime64("2017-06-15")
run_stop =np.datetime64("2017-07-15")

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

# these are scaled down by 1e-3 for debugging
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
#   get the data files
# don't choose these dynamically because hycom data are cached based
# on the region and we want to re-use the cache as much as possible.
hycom_lon_range=[-124.7, -121.7 ]
hycom_lat_range=[36.2, 38.85]
coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]
coastal_files=hycom.fetch_range(hycom_lon_range,hycom_lat_range,coastal_time_range)

# For starters, set IC and BC to a static hycom field.
hycom_ds=xr.open_dataset(coastal_files[0])


class HycomVelocityBC(drv.VelocityBC):
    """
    A velocityBC which takes its data from HYCOM output
    """
    # Data is only daily, so go a bit longer than a usual tidal filter
    lp_hours=3*24.0
    hycom_files=None # caller to pass a list of filenames

    def dataset(self):
        model=self.model
        # Extract time series from the datasets
        xy=np.array(self.geom.coords)
        L=utils.dist(xy[0],xy[-1])
        xy_mid=xy.mean(axis=0)
        ll=model.native_to_ll(xy_mid)

        ds=xr.open_dataset(self.hycom_files[0])

        hyc_depth=ds.depth.values # top to bottom, positive:down

        lat_i=utils.nearest(ds.lat.values,ll[1])
        lon_values=ds.lon.values
        lon_i=utils.nearest((lon_values-lon_values[0])%360., (ll[0]-lon_values[0])%360. )

        dst=xr.Dataset()

        # Scan to get times first
        times=[]
        for fn in self.hycom_files:
            # grab time from the filename -- timestamp in the file does not appear
            # to be reliable.
            t=utils.to_dt64(datetime.datetime.strptime(os.path.basename(fn).split('-')[0],
                                                       '%Y%m%d%H'))
            times.append(t)
        times=np.array(times)
        dst['time']=('time',),times

        # make this positive:down to match hycom and make the interpolation
        layers=model.layer_data()
        sun_z=-layers.z_mid.values

        Nk=int(model.config['Nkmax'])
        u=np.zeros( (len(times), Nk), np.float64)
        v=np.zeros( (len(times), Nk), np.float64)

        for ti,fn in enumerate(self.hycom_files):
            # grab time from the filename -- timestamp in the file does not appear
            # to be reliable.
            # actually that was a bug in the download code. should be okay now.
            ds=xr.open_dataset(fn)
            hyc_u=ds['water_u'].isel(lat=lat_i,lon=lon_i).values
            hyc_v=ds['water_v'].isel(lat=lat_i,lon=lon_i).values

            valid=np.isfinite(hyc_u)

            if not np.any(valid):
                ds.close()
                log.warning("BC point (%.1f,%.1f) is dry in HYCOM grid"%(xy_mid[0],xy_mid[1]))
                continue # will leave as 0

            # could add bottom values if we really cared.
            sun_u=np.interp(sun_z, hyc_depth[valid], hyc_u[valid])
            sun_v=np.interp(sun_z, hyc_depth[valid], hyc_v[valid])
            u[ti,:]=sun_u
            v[ti,:]=sun_v
            ds.close() # avoid too many open files

        dt_h=np.median(np.diff(times)) / np.timedelta64(3600,'s')
        for zi in range(len(sun_z)):
            u[:,zi] = filters.lowpass(u[:,zi],cutoff=self.lp_hours,order=4,dt=dt_h)
            v[:,zi] = filters.lowpass(v[:,zi],cutoff=self.lp_hours,order=4,dt=dt_h)
        dst['u']=('time','Nk'),u
        dst['v']=('time','Nk'),v
        dst['depth']=('Nk',),sun_z
        dst.depth.attrs['positive']='down'

        # calculate related quantities to make debugging easier
        j=self.model.grid.select_edges_nearest( xy_mid )
        grid_n=self.get_inward_normal(j)
        dz=-np.diff(layers.z_interface)
        dst['U']=('time',),(u*dz).sum(axis=1)
        dst['V']=('time',),(v*dz).sum(axis=1)
        dst['Q']=('time',),L*(dst.U*grid_n[0]+dst.V*grid_n[1])
        dst['unorm']=('time','Nk'), grid_n[0]*u + grid_n[1]*v

        return dst

##

# spatially varying
if ocean_method=='eta':
    ocean_bc=drv.MultiBC(drv.OTPSStageBC,name='Ocean',otps_model='wc')
elif ocean_method=='flux':
    ocean_bc=drv.MultiBC(drv.OTPSFlowBC,name='Ocean',otps_model='wc')
elif ocean_method=='velocity':
    ocean_bc=drv.MultiBC(drv.OTPSVelocityBC,name='Ocean',otps_model='wc')
elif ocean_method=='hycom':
    ocean_bc=drv.MultiBC(HycomVelocityBC,name='Ocean',hycom_files=coastal_files)

model.add_bcs(ocean_bc)

model.write()

#--

# How do those flows line up?

all_Q=0
for bc in utils.progress(model.bcs[0].sub_bcs):
    all_Q=all_Q + bc.dataset()['Q'].values

t=bc.dataset()['time']

#--
total_area=model.grid.cells_area().sum()

# these are on the order of 1e-5, shakes out to 7m/day.
# removing the repeated days makes it a bit better, though
# it still loses 1.5m over 200 steps
# output is every 15 steps, which is 15 minutes.
# 5 day run.
# should be 480 steps of output. not sure why I only see about 295
# steps in the time series plots.
# but at at the
d_eta_dt = all_Q / total_area

#--
#
# map each cell to a hycom
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


