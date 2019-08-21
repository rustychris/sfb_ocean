"""
Script to pre-fetch COAMPS data
"""
import sys, os

import numpy as np                      
import datetime as dt   
from stompy import utils
import xarray as xr
          
from stompy.io.local import coamps
from stompy import utils,memoize

import sfei_wind
cache_dir=sfei_wind.cache_dir

os.path.exists(cache_dir) or os.makedirs(cache_dir)

coamps.fetch_coamps_wind(np.datetime64("2017-06-01"),
                         np.datetime64("2018-09-15"),
                         cache_dir=cache_dir,
                         fields=['wnd_utru','wnd_vtru','grnd_sea_temp'])

