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
run_stop =np.datetime64("2018-01-31")

hycom_lon_range=[-124.7, -121.7 ]
hycom_lat_range=[36.2, 38.85]
coastal_pad=np.timedelta64(10,'D') # lots of padding to avoid ringing from butterworth
coastal_time_range=[run_start-coastal_pad,run_stop+coastal_pad]

##
cache_dir='cache'
coastal_files=hycom.fetch_range(hycom_lon_range, hycom_lat_range, coastal_time_range,
                                cache_dir=cache_dir)


