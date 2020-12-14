"""
Pre-fetch (and mark missing as needed) USGS streamflow data for the whole period.
"""
from stompy.io.local import usgs_nwis
import six
six.moves.reload_module(usgs_nwis)
import numpy as np
import os
from stompy.spatial import wkb2shp

grid_dir="grid-merge-suisun"
flow_locations_shp=os.path.join(grid_dir,'watershed_inflow_locations.shp')
flow_features=wkb2shp.shp2geom(flow_locations_shp)

all_gages=np.unique( np.concatenate( [gages.split('|') for gages in flow_features['gages']] ) )

cache_dir='cache'
usgs_gage_cache=os.path.join(cache_dir, 'usgs','streamflow')

flows_ds=usgs_nwis.nwis_dataset_collection(all_gages,
                                           start_date=np.datetime64("2017-05-25"),
                                           end_date=np.datetime64("2018-07-15"),
                                           products=[60], # streamflow
                                           days_per_request='M', # monthly chunks
                                           frequency='daily', # time resolution of the data
                                           cache_dir=usgs_gage_cache,
                                           cache_no_data=True)
