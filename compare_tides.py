import os
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from stompy.model import otps

from sfb_dfm_utils import ca_roms

from stompy import utils
from stompy.spatial import wkb2shp, proj_utils
from stompy.io.local import noaa_coops
import stompy.model.delft.io as dio

############ COMPARISONS ##################

obs_pnts=wkb2shp.shp2geom("/opt/data/delft/sfb_dfm_v2/inputs-static/observation-points.shp")
feat=obs_pnts[ np.nonzero( obs_pnts['name']=='NOAA_PointReyes' )[0][0] ]
pr_xy=np.array(feat['geom'])

##

def pr_dfm_s1(run_name):
    run_base_dir="runs/%s"%run_name
    mdu_fn=os.path.join(run_base_dir,"%s.mdu"%run_name)
    mdu=dio.MDUFile(mdu_fn)
    dfm_map=xr.open_dataset( 'runs/%s/DFM_OUTPUT_%s/%s_map.nc'%(run_name,run_name,run_name) )
    dfm_map_g=dfm_grid.DFMGrid(dfm_map)
    dfm_cc=dfm_map_g.cells_centroid()
    pr_cell_idx=dfm_map_g.select_cells_nearest(xy)
    return dfm_map.s1.isel(nFlowElem=pr_cell_idx)

dfm_short27=pr_dfm_s1('short_27') # probably misconfigured to have velo only shallower than 1000m deep.
# dfm_short26=pr_dfm_s1('short_26') # corrected by water level forcing.
# dfm_short25=pr_dfm_s1('short_25') # test run, not tidal.

t_start=dfm_short27.time.values[0]
t_stop=dfm_short27.time.values[-1]

##

pr_tides_noaa=noaa_coops.coops_dataset("9415020",t_start,t_stop,["water_level"],
                                       cache_dir='cache',days_per_request='M')
pr_tides_noaa=pr_tides_noaa.isel(station=0)

##

# And OTPS comparison:
# May move more of this to sfb_dfm_utils in the future
Otps=otps.otps_model.OTPS('/home/rusty/src/otps/OTPS2', # Locations of the OTPS software
                          '/opt/data/otps') # location of the data

pr_ll=[ pr_tides_noaa.lon.values[0],
        pr_tides_noaa.lat.values[0] ]
z_harmonics = Otps.extract_HC( [ pr_ll ] )

otps_times=np.arange(run_start, run_stop,np.timedelta64(600,'s'))

otps_water_level=otps.reconstruct(z_harmonics,otps_times)

##
dfm_short28=pr_dfm_s1('short_28') # includes velo down to 100m deep.

plt.figure(10).clf()
fig,ax=plt.subplots(num=10)
ax.plot(dfm_short27.time,dfm_short27,label='DFM27 near PR')
ax.plot(dfm_short28.time,dfm_short28,label='DFM28 near PR')

ax.plot(pr_tides_noaa.time,
        pr_tides_noaa.water_level,
        label='PR NOAA')
ax.plot(otps_water_level.time,
        1.0+otps_water_level.result.isel(site=0),
        label='OTPS@PR')
ax.legend()

