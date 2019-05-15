from stompy.io.local import hycom
import numpy as np
import xarray as xr
from stompy import utils

# Then come back and match up subset with hycom
def set_ic_from_hycom(model,hycom_ll_box,cache_dir,default_s=None,default_T=None):
    """
    Update model.ic_ds with salinity and temperature from hycom.
    hycom_ll_box is like [-124.9, -121.7, 35.9, 39.0], and is specified here
    to make best use of cached data (so slightly mis-alignment between models doesn't require
    refetching all of the hycom files).
    where to save/find cached hycom files
    default_s, default_T: when a grid point does not intersect valid hycom data, what
     value to use.  leave as None in order to leave the value in ic_ds alone.

    In the past used: 
      default_s=33.4
      default_T=10.0
    """
    fns=hycom.fetch_range(hycom_ll_box[:2],hycom_ll_box[2:],
                          [model.run_start,model.run_start+np.timedelta64(1,'D')],
                          cache_dir=cache_dir)
    hycom_ic_fn=fns[0]

    hycom_ds=xr.open_dataset(hycom_ic_fn)
    if 'time' in hycom_ds.dims:
        hycom_ds=hycom_ds.isel(time=0)
    cc=model.grid.cells_center()
    cc_ll=model.native_to_ll(cc)

    # Careful - some experiments (such as 92.8) have lon in [0,360],
    # while later ones have lon in [-180,180]
    # this forces all to be [-180,180]
    hycom_ds.lon.values[:] = (hycom_ds.lon.values+180)%360.0 - 180.0
    
    dlat=np.median(np.diff(hycom_ds.lat.values))
    dlon=np.median(np.diff(hycom_ds.lon.values))
    lat_i = utils.nearest(hycom_ds.lat.values,cc_ll[:,1],max_dx=1.2*dlat)
    lon_i = utils.nearest(hycom_ds.lon.values,cc_ll[:,0],max_dx=1.2*dlon)

    # make this positive:down to match hycom and make the interpolation
    sun_z = -model.ic_ds.z_r.values

    assert ('time', 'Nk', 'Nc') == model.ic_ds.salt.dims,"Workaround is fragile"

    for scal,hy_var,sun_var,default in [
            ('s','salinity','salt',default_s),
            ('T','water_temp','temp',default_T) ]:
        if scal=='s' and float(model.config['beta'])==0.0:
            continue
        if scal=='T' and float(model.config['gamma'])==0.0:
            continue

        for c in utils.progress(range(model.grid.Ncells()),msg="HYCOM initial condition %s %%s"%scal):
            sun_val=default

            if lat_i[c]<0 or lon_i[c]<0:
                print("Cell %d does not overlap HYCOM grid"%c)
            else:
                # top to bottom, depth positive:down
                val_profile=hycom_ds[hy_var].isel(lon=lon_i[c],lat=lat_i[c]).values
                valid=np.isfinite(val_profile)
                if not np.any(valid):
                    # print("Cell %d is dry in HYCOM grid"%c)
                    pass
                else:
                    # could add bottom salinity if we really cared.
                    sun_val=np.interp( sun_z,
                                       hycom_ds.depth.values[valid], val_profile[valid] )
            if sun_val is not None:
                model.ic_ds[sun_var].values[0,:,c]=sun_val
