# Original file:
# generate_amu_amv_4SFB.py, by Allie King, downloaded from Google Drive
#   2019-03-25.

# Adapted for suntans use and coastal domain by Rusty Holleman

#--------------------------------------------------------------------------------------#
# User Input
#--------------------------------------------------------------------------------------#
# specify start and end times in format yyyy-mm-dd HH:MM
start_time_string = '2017-05-01 00:00'
end_time_string = '2017-06-01 00:00'

# specify filename prefilx for *.amu/*.amv files
outfileprefix = 'wind_natneighbor_201705'

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
# specify base directory (path to SFEI_Wind folder)
basedir = '/opt/local/cws/hydro/SFEstuary/SOURCE_DATA/Wind/SFEI_Wind'


#--------------------------------------------------------------------------------------#
# Load packages
#--------------------------------------------------------------------------------------#

# import sys, os, and add "Library" directory to python path so python can find 
# wind_library.py
import sys, os
sys.path.append(os.path.join(basedir, 'Library'))

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

# specify the directory containing maps of the bay area -- this will be used to 
# plot the wind observation stations contributing to the wind input field
mapdir = os.path.join(basedir,'Maps')

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
# start_time,end_time are in UTC, so remove the offset when requesting data
# from wlib which expects PST
time_days, station_names, U10_obs = wlib.read_10m_wind_data_from_csv(os.path.join(windobspath,'SFB_hourly_U10_'), start_dt-tz_offset, end_dt-tz_offset)
time_days, station_names, V10_obs = wlib.read_10m_wind_data_from_csv(os.path.join(windobspath,'SFB_hourly_V10_'), start_dt-tz_offset, end_dt-tz_offset)

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

# zip the x, y coordinates for use in the griddata interpolation
points = np.column_stack((x_obs,y_obs)) 

# loop through all times, at each time step find all the non-nan data, and
# interpolate it onto the model grid, then compile the data from all times 
# into a dimension-3 matrix. keep track of which stations were non-nan ('good')
# at each time step in the matrix igood
from stompy.io.local import coamps
from stompy import utils,memoize

coamps_ds=None # handled on demand below
coamps_xy=None # ditto
cache_dir='cache'
# drops COAMPS data points within buffer dist of a good observation
buffer_dist=30e3


for it in range(len(time_days)):
    if it%10==0:
        print("%d/%d steps"%(it,len(time_days)))
    
    # find all non-nan data at this time step
    igood = np.logical_and(~np.isnan(U10_obs[it,:]), ~np.isnan(V10_obs[it,:]))

    #-- augment with COAMPS output
    target_time=start_date+np.timedelta64(int(time_days[it]*86400),'s')
    if (coamps_ds is None) or (target_time>coamps_ds.time.values[-1]):
        coamps_ds=coamps.coamps_press_windxy_dataset(bounds,
                                                     target_time,target_time+np.timedelta64(1,'D'),
                                                     cache_dir=cache_dir)
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
    
    obs_xy=np.c_[x_obs[igood], y_obs[igood]]

    # Which coamps points are far enough from good observations.  there are
    # also some time where coamps data is missing
    # mask=np.ones(len(coamps_xy),np.bool8)
    mask=np.isfinite(coamps_sub.wind_u.values.ravel())

    for xy in obs_xy:
        mask=mask&mask_near_point(xy)

    input_xy=np.concatenate( [obs_xy,coamps_xy[mask]] )
    input_U=np.concatenate( [U10_obs[it,igood], coamps_sub.wind_u.values.ravel()[mask]])
    input_V=np.concatenate( [V10_obs[it,igood], coamps_sub.wind_v.values.ravel()[mask]])

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
    if it==0:
        U10g_all = np.expand_dims(U10g,axis=0)
        V10g_all = np.expand_dims(V10g,axis=0)
        igood_all = np.expand_dims(igood,axis=0)
    else:    
        U10g_all = np.append(U10g_all, np.expand_dims(U10g,axis=0), axis=0)
        V10g_all = np.append(V10g_all, np.expand_dims(V10g,axis=0), axis=0)
        igood_all = np.append(igood_all, np.expand_dims(igood,axis=0), axis=0)


##

# Write netcdf:
ds=xr.Dataset()

ds['time']=('time',), start_date + (time_days*86400).astype(np.int32)*np.timedelta64(1,'s')
ds['x']=('x',),x
ds['y']=('y',),y
ds['wind_u']=('time','y','x'), U10g_all
ds['wind_v']=('time','y','x'), V10g_all

nc_fn=outfileprefix+".nc"
os.path.exists(nc_fn) and os.unlink(nc_fn)
ds.to_netcdf(nc_fn)

# # write amu/amv files
# wlib.write_winds_to_amu_amv(start_time, time_days, xg, yg, U10g_all, V10g_all,
#                             commentstring=commentstring, outfileprefix=outfileprefix,
#                             grid_unit='m', wind_unit='m s-1')
# 
# #--------------------------------------------------------------------------------------#
# # Create a plot showing which stations are reporting and at what percent
# #--------------------------------------------------------------------------------------#
# 
# # compute percent reporting
# percent_reporting = np.sum(igood_all, axis=0)/len(time_days)*100
# Nstations = len(percent_reporting)
# 
# # load digital elevation map (DEM) of the Bay Area and note its extent (extent is in the text file of 
# # the same name). set empty cells to lowest elevation (-11m, occurs in the delta
# zg_dem = Image.open(os.path.join(mapdir,'DEM_90m_Bay_Area.tif'))
# zg_dem = np.array(zg_dem).astype(float)
# zg_dem[zg_dem==-32768] = np.nan
# zg_dem[np.isnan(zg_dem)] = np.nanmin(zg_dem)
# extent_dem = np.array([434835, 676755, 4081065, 4318485])
# 
# # load the bay/delta shorline shape file
# sf = shp.Reader(os.path.join(mapdir,'shoreline_boundary.shp'))
# 
# # plot the DEM and shoreline data
# plt.figure(figsize=(12,6.5))
# plt.imshow(zg_dem,cmap='Greens',extent=extent_dem/1000)   
# for shape in sf.shapeRecords():
#     x = np.array([i[0] for i in shape.shape.points[:]])
#     y = np.array([i[1] for i in shape.shape.points[:]])
#     plt.plot(x/1000,y/1000,'k',linewidth=0.5)
# 
# # add the wind stations
# ms = ['o','v','^','<','>','s','p','P','h','H','D']*6
# cs = ['red','darkorange','darkturquoise','steelblue','navy']*11
# for snum in range(Nstations):
#     if snum==0:
#         labelstr = station_names[snum] + ': %0.0f%%*' % percent_reporting[snum]
#     else:
#         labelstr = station_names[snum] + ': %0.0f%%' % percent_reporting[snum]
#     if percent_reporting[snum]==0:
#         ms1 = '.'
#         mfc1 = cs[snum]
#     else:
#         ms1 = ms[snum]
#         mfc1 = 'none'
#     plt.plot(x_obs[snum]/1000,y_obs[snum]/1000,ms1,
#                                                label=labelstr,
#                                                color=cs[snum],
#                                                markeredgewidth=2,
#                                                markersize=10,
#                                                markerfacecolor=mfc1)
# 
# # add the wind input box
# plt.plot([x_llcorner/1000.,x_urcorner/1000.,x_urcorner/1000.,x_llcorner/1000.,x_llcorner/1000.],
#          [y_llcorner/1000.,y_llcorner/1000.,y_urcorner/1000.,y_urcorner/1000.,y_llcorner/1000.],'k',
#          label='DFlow wind\ninput domain')
#          
# # add legend and labels and save map as figure
# plt.legend(ncol=2, bbox_to_anchor=(1.1, 0.0, 0.1, 1.0), loc='center left')
# plt.xlabel('x (km, UTM Zone 10N) / * % in legend indicate % total time period station data is available')
# plt.ylabel('y (km, UTM Zone 10N)')
# plt.axis('equal')
# plt.tight_layout()
# plt.xlim((470,660))
# plt.ylim((4125,4300))
# plt.savefig(outfileprefix + '_percent_reporting.png')
# 
