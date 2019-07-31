# Original file:
# generate_amu_amv_4SFB.py, by Allie King, downloaded from Google Drive
#   2019-03-25.

# import sys, os, and add "Library" directory to python path so python can find 
# wind_library.py
import sys, os

# specify base directory (path to SFEI_Wind folder)
basedir = './SFEI_Wind'

# import the rest of the packages
import numpy as np                      
import pandas as pd                     
import datetime as dt   
import matplotlib.pylab as plt
from stompy import utils
import xarray as xr
from scipy.interpolate import griddata            
import shapefile as shp                 # pip install pyshp          
from PIL import Image                   # pip install pillow
import wind_library as wlib
from stompy.io.local import coamps
from stompy import utils,memoize
cache_dir='../suntans/cache'

try:
    import Ngl as ngl                   
except Exception as e:
    print('')
    print(e)
    print('')
    print('It is ok to run this script without the pyngl package, but you cannot use')
    print('natural neighbor interpolation without pyngl installed. Unfortunately pyngl')
    print('causes problems with the latest version of conda, so it is best to install it')
    print('in a separate environement. To create an environment called pyngl_env with') 
    print('pyngl installed:')
    print('    conda create --name pyngl_env --channel conda-forge pyngl')
    print('Then to run this script you need to activate the environment:')
    print('   conda activate pyngl_env')
    print('and install the other packages that aren\'t part of pynio/pyngl:')
    print('   pip install pandas')
    print('   pip install datetime')
    print('   pip install matplotlib')
    print('   pip install scipy')
    print('   pip install pyshp')
    print('   pip install pillow')
    print('   pip install netCDF4')
    print('and if you use ipython, make sure to also install ipython in pyngl_env or it')
    print('will automatically launch in the base environment without telling you:')
    print('   conda install ipython')
    print('')

# Adapted for suntans use and coastal domain by Rusty Holleman


def process_period(start_time_string,end_time_string,outfileprefix,
                   force=False):
    nc_fn=outfileprefix+".nc"
    print("Processing %s to %s output to %s"%(start_time_string,end_time_string,nc_fn))
    
    if (not force) and os.path.exists(nc_fn):
        print("File %s exists - skipping"%nc_fn)
        return
    
    # pick interpolation method (natural neighbor or linear is recommended):
    #   'nearest' = nearest neighbor
    #   'linear' = linear
    #   'cubic' = cubic spline
    #   'natural' = natural neighbor
    interp_method = 'natural'

    # specify comment string (one line only, i.e., no '\n') for *.amu/*.amv files
    commentstring = 'Prepared by Allie King, SFEI, times are in PST (ignore the +00:00), adapted by Rusty Holleman'

    # specify properties of the wind grid -- this one was used for CASCaDE and sfb_dfm
    bounds=[340000, 610000, 3980000, 4294000]
    dx = 1500.
    dy = 1500.

    #--------------------------------------------------------------------------------------#
    # Main Program
    #--------------------------------------------------------------------------------------#

    n_cols = int(round(1+(bounds[1]-bounds[0])/dx))
    n_rows = int(round(1+(bounds[3]-bounds[2])/dy))
    x_llcorner = bounds[0]
    y_llcorner = bounds[2]

    start_date=np.datetime64(start_time_string)
    end_date=np.datetime64(end_time_string)

    # specify directory containing the compiled wind observation data and station 
    # coordinates (SFB_hourly_U10_2011.csv, SFB_hourly_V10_2011.csv, etc...)
    windobspath = os.path.join(basedir,'Compiled_Hourly_10m_Winds/data')

    # convert start and end time to datetime object
    start_dt=utils.to_datetime(start_date)
    end_dt  =utils.to_datetime(end_date)

    # create a meshgrid corresponding to the CASCaDE wind grid
    x_urcorner = x_llcorner + dx*(n_cols-1)
    y_urcorner = y_llcorner + dy*(n_rows-1)
    x = np.linspace(x_llcorner, x_urcorner, n_cols)
    # RH: orient y the usual way, not the arcinfo/dfm wind way (i.e. remove flipud)
    y = np.linspace(y_llcorner, y_urcorner, n_rows)
    xg, yg = np.meshgrid(x, y)

    # read the observed wind data
    tz_offset=dt.timedelta(hours=8)
    try:
        # start_time,end_time are in UTC, so remove the offset when requesting data
        # from wlib which expects PST
        time_days, station_names, U10_obs = wlib.read_10m_wind_data_from_csv(os.path.join(windobspath,'SFB_hourly_U10_'), start_dt-tz_offset, end_dt-tz_offset)
        time_days, station_names, V10_obs = wlib.read_10m_wind_data_from_csv(os.path.join(windobspath,'SFB_hourly_V10_'), start_dt-tz_offset, end_dt-tz_offset)
    except FileNotFoundError:
        print("Okay - probably beyond the SFEI data")
        U10_obs=V10_obs=None

    if U10_obs is not None:    
        # note that time_days is just decimal days after start, so it doesn't need to be adjusted for timezone.
        # read the coordinates of the wind observation stations
        df = pd.read_csv(os.path.join(windobspath,'station_coordinates.txt'))
        station_names_check = df['Station Organization-Name'].values
        x_obs = df['x (m - UTM Zone 10N)'].values
        y_obs = df['y (m - UTM Zone 10N)'].values
        Nstations = len(df)
        for snum in range(Nstations):
            if not station_names[snum]==station_names_check[snum]:
                raise('station names in station_coordinates.txt must match headers in SFB_hourly_U10_YEAR.csv and SFB_hourly_V10_YEAR.csv files')
    else:
        x_obs=np.zeros(0,np.float64)
        y_obs=np.zeros(0,np.float64)
        Nstations=0

        # Fabricate time_days
        all_times=[]
        t=start_dt
        interval=dt.timedelta(hours=1)
        while t<=end_dt:
            all_times.append(t)
            t=t+interval
        all_dt64=np.array([utils.to_dt64(t) for t in all_times])
        time_days=(all_dt64-all_dt64[0])/np.timedelta64(1,'s') / 86400.

    # zip the x, y coordinates for use in the griddata interpolation
    points = np.column_stack((x_obs,y_obs))

    # loop through all times, at each time step find all the non-nan data, and
    # interpolate it onto the model grid, then compile the data from all times 
    # into a dimension-3 matrix. keep track of which stations were non-nan ('good')
    # at each time step in the matrix igood
    coamps_ds=None # handled on demand below
    coamps_xy=None # ditto
    # drops COAMPS data points within buffer dist of a good observation
    buffer_dist=30e3

    for it in range(len(time_days)):
        if it%10==0:
            print("%d/%d steps"%(it,len(time_days)))

        #-- augment with COAMPS output
        target_time=start_date+np.timedelta64(int(time_days[it]*86400),'s')
        if (coamps_ds is None) or (target_time>coamps_ds.time.values[-1]):
            coamps_ds=coamps.coamps_dataset(bounds,
                                            target_time,target_time+np.timedelta64(1,'D'),
                                            cache_dir=cache_dir,
                                            fields=['wnd_utru','wnd_vtru'])
            # reduce dataset size -- out in the ocean really don't need too many points
            coamps_ds=coamps_ds.isel(x=slice(None,None,2),y=slice(None,None,2))

            coamps_X,coamps_Y=np.meshgrid(coamps_ds.x.values,coamps_ds.y.values)
            coamps_xy=np.c_[ coamps_X.ravel(), coamps_Y.ravel() ]
            print("COAMPS shape: ",coamps_X.shape) 

            # seems that the coamps dataset is not entirely consistent in its shape?
            # not sure what's going on, but best to redefine this each time to be
            # sure.
            @memoize.memoize()
            def mask_near_point(xy):
                dists=utils.dist(xy,coamps_xy)
                return (dists>buffer_dist)

        coamps_time_idx=utils.nearest(coamps_ds.time,target_time)
        coamps_sub=coamps_ds.isel(time=coamps_time_idx)

        # Which coamps points are far enough from good observations.  there are
        # also some time where coamps data is missing
        # mask=np.ones(len(coamps_xy),np.bool8)
        mask=np.isfinite(coamps_sub.wind_u.values.ravel())

        # find all non-nan data at this time step
        if U10_obs is not None:
            igood = np.logical_and(~np.isnan(U10_obs[it,:]), ~np.isnan(V10_obs[it,:]))
            obs_xy=np.c_[x_obs[igood], y_obs[igood]]

            for xy in obs_xy:
                mask=mask&mask_near_point(xy)

            input_xy=np.concatenate( [obs_xy,coamps_xy[mask]] )
            input_U=np.concatenate( [U10_obs[it,igood], coamps_sub.wind_u.values.ravel()[mask]])
            input_V=np.concatenate( [V10_obs[it,igood], coamps_sub.wind_v.values.ravel()[mask]])
        else:
            # No SFEI data --
            input_xy=coamps_xy[mask]
            input_U=coamps_sub.wind_u.values.ravel()[mask]
            input_V=coamps_sub.wind_v.values.ravel()[mask]

        if np.any(np.isnan(input_U)) or np.any(np.isnan(input_V)):
            import pdb
            pdb.set_trace()

        Ngood=len(input_xy)

        # set the interpolation method to be used in this time step: interp_method_1.
        # ideally, this would just be the user-defined interpolation method: 
        # interp_method. however, if we do not have enough non-nan data to use the 
        # user-defined method this time step, temporarily revert to the nearest 
        # neighbor method
        if interp_method == 'natural' or interp_method == 'linear' or interp_method =='cubic':
            if Ngood>=4:
                interp_method_1 = interp_method
            else:
                interp_method_1 = 'nearest'
        elif interp_method == 'nearest':
            interp_method_1 = 'nearest'

        # if natural neighbor method, interpolate using the pyngl package
        if interp_method_1=='natural':
            U10g = np.transpose(ngl.natgrid(input_xy[:,0],input_xy[:,1],input_U,xg[0,:],yg[:,0]))
            V10g = np.transpose(ngl.natgrid(input_xy[:,0],input_xy[:,1],input_V,xg[0,:],yg[:,0]))

        # for other interpolation methods use the scipy package
        else:
            U10g = griddata(input_xy, input_U, (xg, yg), method=interp_method_1)
            V10g = griddata(input_xy, input_V, (xg, yg), method=interp_method_1)

            # since griddata interpolation fills all data outside range with nan, use 
            # the nearest neighbor method to extrapolate
            U10g_nn = griddata(input_xy, input_U, (xg, yg), method='nearest')
            V10g_nn = griddata(input_xy, input_V, (xg, yg), method='nearest')
            ind = np.isnan(U10g)
            U10g[ind] = U10g_nn[ind]
            ind = np.isnan(V10g)
            V10g[ind] = V10g_nn[ind]

        # compile results together over time
        # igood_all not updated for COAMPS, omit here.
        if it==0:
            U10g_all = np.expand_dims(U10g,axis=0)
            V10g_all = np.expand_dims(V10g,axis=0)
            # igood_all = np.expand_dims(igood,axis=0)
        else:    
            U10g_all = np.append(U10g_all, np.expand_dims(U10g,axis=0), axis=0)
            V10g_all = np.append(V10g_all, np.expand_dims(V10g,axis=0), axis=0)
            # igood_all = np.append(igood_all, np.expand_dims(igood,axis=0), axis=0)

    ##

    # Write netcdf:
    ds=xr.Dataset()

    ds['time']=('time',), start_date + (time_days*86400).astype(np.int32)*np.timedelta64(1,'s')
    ds['x']=('x',),x
    ds['y']=('y',),y
    ds['wind_u']=('time','y','x'), U10g_all
    ds['wind_v']=('time','y','x'), V10g_all

    os.path.exists(nc_fn) and os.unlink(nc_fn)
    ds.to_netcdf(nc_fn)

if __name__=='__main__':
    #--------------------------------------------------------------------------------------#
    # User Input
    #--------------------------------------------------------------------------------------#
    # specify start and end times in format yyyy-mm-dd HH:MM
    if 0:
        start_time_string = '2017-06-01 00:00'
        end_time_string = '2017-07-01 00:00'
        # specify filename prefilx for *.amu/*.amv files
        outfileprefix = 'wind_natneighbor_201706'
        process_period(start_time_string,end_time_string,outfileprefix)
    if 1:
        dates=pd.date_range("2017-06-01 00:00", "2018-07-01 00:00", freq='MS').to_pydatetime()
        
        for start,end in zip(dates[:-1], dates[1:]):
            process_period(start.strftime('%Y-%m-%d %H:%M'),
                           end.strftime('%Y-%m-%d %H:%M'),
                           'wind_natneighbor_%s'%start.strftime('%Y%m'))

