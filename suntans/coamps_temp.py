import numpy as np
from stompy.io.local import coamps

from shapely import geometry
from stompy.spatial import field
import logging

def add_coamps_fields(model,cache_dir,fields=[('air_temp','Tair'),
                                              ('rltv_hum','RH')],
                      mask_field=None):
    """
    Uses existing times, assumed already hourly, in model.met_ds.

    fields: list of tuples, (coamps_name, suntans_name), e.g.
       [ ('air_temp','Tair'), ... ]
    to grab air_temp from coamps, and save to the Tair field of met_ds.

    mask: stompy.spatial.Field, evaluates to nonzero for areas that should 
    be masked out.
    """
    period_start=model.met_ds.nt.values[0]
    period_stop=model.met_ds.nt.values[-1]

    coamps_fields=[fld[0] for fld in fields]
    src_ds=coamps.coamps_dataset(model.grid.bounds(),
                                 period_start,period_stop,
                                 cache_dir=cache_dir,
                                 fields=coamps_fields)

    # only include points which are within 3km of the model domain
    g_poly=geometry.Polygon(model.grid.boundary_polygon().exterior).buffer(3000)

    fld=field.SimpleGrid(extents=[src_ds.x.values[0],
                                  src_ds.x.values[-1],
                                  src_ds.y.values[0],
                                  src_ds.y.values[-1]],
                         F=src_ds[coamps_fields[0]].isel(time=0).values)
    logging.info("coamps temp: gridded resolution: %.2f %.2f"%(fld.dx,fld.dy))
    # RH 2019-08-01: some nan values sneak through the other masks. try to
    # masking out any pixels that are nan on this initial time slice.

    time_slc=(src_ds.time.values>=period_start) & (src_ds.time.values<=period_stop)
    src_sub_ds=src_ds.isel(time=time_slc)

    # make sure these are aligned
    # somehow there are some 1 second errors in the met times.
    # allow slop up to 10 minutes
    assert len( model.met_ds.nt.values) == len(src_sub_ds.time.values),"Mismatch in length of met_ds time and coamps ds time"
    assert np.all( np.abs( model.met_ds.nt.values - src_sub_ds.time.values) < np.timedelta64(600,'s') )

    X,Y=np.meshgrid(src_ds.x.values,src_ds.y.values)
    XY=np.stack((X,Y), axis=-1)
    base_valid=fld.polygon_mask(g_poly)
    if mask_field is not None:
        # additionally mask by a field passed in
        base_valid = base_valid & (mask_field(XY)==0.0)
            
    met_ds=model.met_ds
    for coamps_name,sun_name in fields:
        all_values=src_sub_ds[coamps_name].values
        # additionally restrict to pixels where this variable is finite
        # over all time steps
        valid=base_valid & np.all(np.isfinite(all_values),axis=0)
        values=src_sub_ds[coamps_name].values[:,valid]

        # moved mask code inside the loop in order to catch all nan values
        # even if they are finite at the start, and nan later.
        xcoords=X[valid]
        ycoords=Y[valid]

        for v in [sun_name,'x_'+sun_name,'y_'+sun_name,'z_'+sun_name]:
            if v in met_ds: del met_ds[v]
        met_ds['x_'+sun_name]=("N"+sun_name),xcoords
        met_ds['y_'+sun_name]=("N"+sun_name),ycoords
        met_ds['z_'+sun_name]=("N"+sun_name),10.0*np.ones_like(xcoords)

        # DBG:
        #if not np.all(np.isfinite(values)):
        #    import pdb # each time step, 519 values are nan.
        #    pdb.set_trace()
        assert np.all(np.isfinite(values))
        met_ds[sun_name]=(('nt','N'+sun_name),values)
