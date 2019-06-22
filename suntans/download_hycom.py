# Try to understand how close the hycom velocity field is to
# obeying continuity
# first cut answer is that the data online starts 6/01/2017,
# but the downloader did not catch that, and would return that
# date's data for requested earlier days.
import logging
logging.basicConfig(level=logging.DEBUG)
import xarray.backends.common as xb_common

import numpy as np
from stompy import utils
utils.path("..")
import six

import xarray as xr
from stompy.io.local import hycom

##

six.moves.reload_module(utils)
six.moves.reload_module(hycom)

##
run_start=np.datetime64("2017-06-15")
run_stop =np.datetime64("2018-06-10")

hycom_lon_range=[-124.9, -121.7 ]
hycom_lat_range=[35.9, 39.0]
coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]

##
cache_dir='cache'
coastal_files=hycom.fetch_range(hycom_lon_range, hycom_lat_range, coastal_time_range,
                                cache_dir=cache_dir)


##
# 
# # try accessing opendap directly:
#   this ultimately gave a DAP failure.  maybe the timeouts are not as generous.
#   will stick with the original NCSS method for now, and just try to handle failures
#   more gracefully.

# ds=xr.open_dataset("https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9",
#                    decode_times=False,decode_cf=False)
# 
# ## 
# # hycom download is failing
# 
# # in the browser I get an error:
# #  Illegal Range for dimension 0: last requested 1269 > max 1254
# # the opendap ds shows 1270 as size of time dimension.
# 
# 
# # INFO:root:http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.9
# #   INFO:root:[('var', 'salinity_bottom'), ('var', 'surf_el'), ('var', 'water_temp_bottom'), ('var', 'water_u_bottom'), ('var', 'water_v_bottom'), ('var', 'salinity'), ('var', 'water_temp'), ('var', 'water_u'), ('var', 'water_v'), ('north', '39.000'), ('south', '35.900'), ('west', '-124.900'), ('east', '-121.700'), ('time_start', '2017-10-19T00:00:00Z'), ('time_end', '2017-10-20T00:00:00Z'), ('timeStride', '1'), ('disableProjSubset', 'on'), ('horizStride', '1'), ('vertCoord', ''), ('LatLon', 'true'), ('accept', 'netcdf4')]
# 
# # Traceback (most recent call last):
# #   File "./merged_sun.py", line 289, in <module>
# #     model.write()
# #   File "/home/rusty/src/stompy/stompy/model/suntans/sun_driver.py", line 745, in write
# #     self.write_forcing()
# #   File "/home/rusty/src/stompy/stompy/model/suntans/sun_driver.py", line 866, in write_forcing
# #     super(SuntansModel,self).write_forcing()
# #   File "/home/rusty/src/stompy/stompy/model/delft/dflow_model.py", line 1152, in write_forcing
# #     self.write_bc(bc)
# #   File "/home/rusty/src/stompy/stompy/model/suntans/sun_driver.py", line 1236, in write_bc
# #     super(SuntansModel,self).write_bc(bc)
# #   File "/home/rusty/src/stompy/stompy/model/delft/dflow_model.py", line 1156, in write_bc
# #     bc.enumerate_sub_bcs()
# #   File "/home/rusty/src/stompy/stompy/model/delft/dflow_model.py", line 1753, in enumerate_sub_bcs
# #     self.populate_files()
# #   File "/home/rusty/src/stompy/stompy/model/delft/dflow_model.py", line 1762, in populate_files
# #     cache_dir=self.cache_dir)
# #   File "/home/rusty/src/stompy/stompy/io/local/hycom.py", line 44, in fetch_range
# #     fetch_one_day(t,cache_name,lon_range,lat_range)
# #   File "/home/rusty/src/stompy/stompy/io/local/hycom.py", line 101, in fetch_one_day
# #     log=logging,params=params,timeout=1800)
# #   File "/home/rusty/src/stompy/stompy/utils.py", line 2022, in download_url
# #     r.raise_for_status()
# #   File "/opt/anaconda3/lib/python3.7/site-packages/requests/models.py", line 940, in raise_for_status
# #     raise HTTPError(http_error_msg, response=self)
# # requests.exceptions.HTTPError: 500 Server Error: 500 for url: http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.9?var=salinity_bottom&var=surf_el&var=water_temp_bottom&var=water_u_bottom&var=water_v_bottom&var=salinity&var=water_temp&var=water_u&var=water_v&north=39.000&south=35.900&west=-124.900&east=-121.700&time_start=2017-10-19T00%3A00%3A00Z&time_end=2017-10-20T00%3A00%3A00Z&timeStride=1&disableProjSubset=on&horizStride=1&vertCoord=&LatLon=true&accept=netcdf4
# #
# 
# ds.time.load()
# del ds['tau'] # related to analysis.
# dsd=xr.decode_cf(ds)
# 
# ##
# 
# period_start=np.datetime64("2017-10-15T00:00:00")
# period_stop =np.datetime64("2017-10-16T00:00:00")
# 
# time_slice=np.searchsorted(dsd.time.values,[period_start,period_stop])
# time_slice=slice(time_slice[0],time_slice[1]+1) # inclusive of period_stop.
# 
# ##                            
# dsd.lat.load()
# dsd.lon.load()
# 
# # verified that these match up with existing pulls.
# lat_slice=slice( np.searchsorted(dsd.lat.values, hycom_lat_range[0]),
#                  np.searchsorted(dsd.lat.values, hycom_lat_range[1])+1 )
# lon_slice=slice( np.searchsorted(dsd.lon.values % 360., np.array(hycom_lon_range[0])%360),
#                  np.searchsorted(dsd.lon.values % 360., np.array(hycom_lon_range[1])%360) + 1)
# 
# ## 
# 
# # sample=xr.open_dataset('cache/2017061800--124.90_-121.70_35.90_39.00.nc')
# 
# dsd_sel=dsd.isel(time=time_slice,lat=lat_slice,lon=lon_slice)
# 
# # using an ncss request, this often takes 10 minutes.
# # via opendap??  just gets a DAP failure after a minute.
# dsd_sel.to_netcdf("opendap-slice.nc")
# 
