# Develop code to pull gridded Bay (SFEI) wind and 
# COAMPS wind, trim to grid and add to a met dataset.
from stompy.grid import unstructured_grid
from stompy.spatial import field
from stompy import utils
import logging
import os
import glob
import numpy as np
import six
import xarray as xr
from shapely import geometry
from shapely.ops import cascaded_union

from stompy.io.local import coamps

wind_dir=os.path.join(os.path.dirname(__file__),"..","wind")

def blended_dataset(period_start,period_stop):
    import glob

    files=glob.glob(os.path.join(wind_dir,"wind_natneighbor_*.nc"))
    files.sort()

    hits=[] # datasets overlapping the requested period
    for fn in files:
        ds=xr.open_dataset(fn)
        if (ds.time.values[0]<=period_stop) and (ds.time.values[-1]>=period_start):
            hits.append(ds)
        else:
            ds.close()
    blended_ds=xr.concat(hits,dim='time')
    # in one case there is a duplicate at the end of one file and start
    # of another.
    monotone=np.r_[True, blended_ds.time.values[1:]>blended_ds.time.values[:-1]]
    blended_ds=blended_ds.isel(time=monotone)
    return blended_ds

def add_wind_preblended(model,cache_dir,pad=np.timedelta64(3*3600,'s')):
    """
    model: A HydroModel instance
    cache_dir: path for caching wind data
    pad: how far before/after the simulation the wind dataset should extend
    
    Add wind data from pre-blended netcdf output
    """
    g=model.grid
    
    period_start=model.run_start - pad
    period_stop=model.run_stop + pad

    # note that this is a bit different than the sfei data
    #  - already in UTC
    #  - larger footprint, coarser grid
    #  - natural neighbors run on the observed + COAMPS (thinned).
    blended_ds=blended_dataset(period_start,period_stop)
    
    # Not ready for other years
    assert blended_ds.time.values[0] <= period_start,"FIX: pre-blended wind only set up for some of 2017"
    assert blended_ds.time.values[-1] >= period_stop,"FIX: pre-blended wind only set up for some of 2017"

    # buffer out the model domain a bit to get a generous footprint
    g_poly=geometry.Polygon(g.boundary_polygon().exterior).buffer(3000)

    # For each of the sources, which points will be included?
    # Add a mask variable for each dataset
    exclude_poly=None
    for src_name,src_ds in [('BLENDED',blended_ds)]:
        fld=field.SimpleGrid(extents=[src_ds.x.values[0],
                                      src_ds.x.values[-1],
                                      src_ds.y.values[0],
                                      src_ds.y.values[-1]],
                             F=src_ds.wind_u.isel(time=0).values)
        logging.info("%s: gridded resolution: %.2f %.2f"%(src_name,fld.dx,fld.dy))
        mask=fld.polygon_mask(g_poly)
        logging.info("%s: %d of %d samples fall within grid"%(src_name,mask.sum(),mask.size))
        if exclude_poly is not None:
            omit=fld.polygon_mask(exclude_poly)
            mask=mask&(~omit)
            logging.info("%s: %d of %d samples fall within exclusion poly, will use %d"%(src_name,omit.sum(),omit.size,mask.sum()))
            
        src_ds['mask']=src_ds.wind_u.dims[1:], mask

    #  Trim to the same period
    time_slc=(blended_ds.time.values>=period_start) & (blended_ds.time.values<=period_stop)
    blended_sub_ds=blended_ds.isel(time=time_slc)

    times=blended_sub_ds.time.values

    # Now we start to break down the interface with model, as wind is not really
    # ready to go.
    
    met_ds=model.zero_met(times=times)

    srcs=[blended_sub_ds]
    src_counts=[src.mask.values.sum() for src in srcs]
    n_points=np.sum(src_counts)

    xcoords=[]
    ycoords=[]
    for src in srcs:
        X,Y=np.meshgrid(src.x.values,src.y.values)
        xcoords.append(X[src.mask.values])
        ycoords.append(Y[src.mask.values])
    xcoords=np.concatenate(xcoords)
    ycoords=np.concatenate(ycoords)
        
    # Replace placeholder coordinates for wind variables.
    for name in ['Uwind','Vwind']:
        del met_ds["x_"+name]
        del met_ds["y_"+name]
        del met_ds["z_"+name]
        del met_ds[name]
        
        met_ds["x_"+name]=("N"+name),xcoords
        met_ds["y_"+name]=("N"+name,),ycoords
        met_ds["z_"+name]=("N"+name,),10.0*np.ones_like(xcoords)

    Uwind_t=[]
    Vwind_t=[]
    for ti in utils.progress(range(len(times)),msg="Compiling wind: %s"):
        Uwind=[]
        Vwind=[]
        for src in srcs:
            Uwind.append(src.wind_u.isel(time=ti).values[ src.mask ])
            Vwind.append(src.wind_v.isel(time=ti).values[ src.mask ])
        Uwind=np.concatenate(Uwind)
        Vwind=np.concatenate(Vwind)
        Uwind_t.append(Uwind)
        Vwind_t.append(Vwind)
    met_ds['Uwind']=('nt',"NUwind"),np.stack(Uwind_t)
    met_ds['Vwind']=('nt',"NVwind"),np.stack(Vwind_t)

    logging.info("New Met Dataset:")
    logging.info(str(met_ds))
    model.met_ds=met_ds
    if int(model.config['metmodel']) not in [4,5]:
        logging.warning("While adding wind, noticed metmodel %s"%(model.config['metmodel']))


def add_wind_coamps_sfei(model,cache_dir,pad=np.timedelta64(3*3600,'s'),
                         coamps_buffer=30e3,
                         air_temp=False):
    """
    model: A HydroModel instance
    cache_dir: path for caching wind data
    pad: how far before/after the simulation the wind dataset should extend
    
    Combine SFEI interpolated winds and COAMPS winds.
    coamps_buffer: coamps samples within this distance of SFEI data are omitted.

    This method does not work so well with SUNTANS.  The available interpolation
    methods (inverse distance and kriging) do not deal well with having two 
    distinct, densely sampled datasets with a gap in between.

    air_temp: if 'coamps', fill in air temperature samples from coamps data.
    """
    g=model.grid
    
    period_start=model.run_start - pad
    period_stop=model.run_stop + pad

    fields=['wnd_utru','wnd_vtru','pres_msl']
    if air_temp=='coamps':
        # may add sol_rad at some point...
        fields += ['air_temp','rltv_hum']
    coamps_ds=coamps.coamps_dataset(g.bounds(),
                                    period_start,period_stop,
                                    cache_dir=cache_dir,
                                    fields=fields)

    sfei_ds=xr.open_dataset('wind_natneighbor_WY2017.nc')
    # SFEI data is PST
    logging.info(sfei_ds.time.values[0])
    sfei_ds.time.values[:] += np.timedelta64(8*3600,'s')
    logging.info(sfei_ds.time.values[0]) # just to be sure it took.
    sfei_ds.time.attrs['timezone']='UTC'
    # Not ready for other years
    assert sfei_ds.time.values[0] <= period_start,"FIX: SFEI wind only setup for 2017"
    assert sfei_ds.time.values[-1] >= period_stop,"FIX: SFEI wind only setup for 2017"

    # buffer out the model domain a bit to get a generous footprint
    g_poly=geometry.Polygon(g.boundary_polygon().exterior).buffer(3000)

    # For each of the sources, which points will be included?
    # Add a mask variable for each dataset
    exclude_poly=None
    for src_name,src_ds in [('SFEI',sfei_ds),
                            ('COAMPS',coamps_ds)]:
        fld=field.SimpleGrid(extents=[src_ds.x.values[0],
                                      src_ds.x.values[-1],
                                      src_ds.y.values[0],
                                      src_ds.y.values[-1]],
                             F=src_ds.wind_u.isel(time=0).values)
        logging.info("%s: gridded resolution: %.2f %.2f"%(src_name,fld.dx,fld.dy))
        mask=fld.polygon_mask(g_poly)
        logging.info("%s: %d of %d samples fall within grid"%(src_name,mask.sum(),mask.size))
        if exclude_poly is not None:
            omit=fld.polygon_mask(exclude_poly)
            mask=mask&(~omit)
            logging.info("%s: %d of %d samples fall within exclusion poly, will use %d"%(src_name,omit.sum(),omit.size,mask.sum()))
            
        src_ds['mask']=src_ds.wind_u.dims[1:], mask

        # Add these points to the exclusion polygon for successive sources
        X,Y=fld.XY()
        xys=np.c_[X[mask], Y[mask]]
        pnts=[geometry.Point(xy[0],xy[1]) for xy in xys]
        poly=cascaded_union( [p.buffer(coamps_buffer) for p in pnts] )
        if exclude_poly is None:
            exclude_poly=poly
        else:
            exclude_poly=exclude_poly.union(poly)

    #  Trim to the same period
    # SFEI
    time_slc=(sfei_ds.time.values>=period_start) & (sfei_ds.time.values<=period_stop)
    sfei_sub_ds=sfei_ds.isel(time=time_slc)

    # COAMPS
    time_slc=(coamps_ds.time.values>=period_start) & (coamps_ds.time.values<=period_stop)
    coamps_sub_ds=coamps_ds.isel(time=time_slc)

    # make sure that worked:
    assert np.all( sfei_sub_ds.time.values == coamps_sub_ds.time.values )

    times=sfei_sub_ds.time.values

    # Now we start to break down the interface with model, as wind is not really
    # ready to go.
    
    met_ds=model.zero_met(times=times)

    srcs=[sfei_sub_ds,coamps_sub_ds]
    src_counts=[src.mask.values.sum() for src in srcs]
    n_points=np.sum(src_counts)

    xcoords=[]
    ycoords=[]
    for src in srcs:
        X,Y=np.meshgrid(src.x.values,src.y.values)
        xcoords.append(X[src.mask.values])
        ycoords.append(Y[src.mask.values])
    xcoords=np.concatenate(xcoords)
    ycoords=np.concatenate(ycoords)
        
    # Replace placeholder coordinates for wind variables.
    for name in ['Uwind','Vwind']:
        del met_ds["x_"+name]
        del met_ds["y_"+name]
        del met_ds["z_"+name]
        del met_ds[name]
        
        met_ds["x_"+name]=("N"+name),xcoords
        met_ds["y_"+name]=("N"+name,),ycoords
        met_ds["z_"+name]=("N"+name,),10.0*np.ones_like(xcoords)

    Uwind_t=[]
    Vwind_t=[]
    for ti in utils.progress(range(len(times)),msg="Compiling wind: %s"):
        Uwind=[]
        Vwind=[]
        for src in srcs:
            Uwind.append(src.wind_u.isel(time=ti).values[ src.mask ])
            Vwind.append(src.wind_v.isel(time=ti).values[ src.mask ])
        Uwind=np.concatenate(Uwind)
        Vwind=np.concatenate(Vwind)
        Uwind_t.append(Uwind)
        Vwind_t.append(Vwind)
    met_ds['Uwind']=('nt',"NUwind"),np.stack(Uwind_t)
    met_ds['Vwind']=('nt',"NVwind"),np.stack(Vwind_t)

    logging.info("New Met Dataset:")
    logging.info(str(met_ds))
    model.met_ds=met_ds
    if int(model.config['metmodel']) not in [0,4]:
        logging.warning("Adding wind, will override metmodel %s => %d"%(model.config['metmodel'],
                                                                        4))
    model.config['metmodel']=4 # wind only
    
    
