import xarray as xr
from stompy import utils
##

ds=xr.open_dataset('sfbay_freshwater/outputs/sfbay_freshwater.nc')

## 
stn_idx=np.nonzero(ds.station.values=='EBAYCc2')[0][0]

##

plt.figure(10).clf()
fig,ax=plt.subplots(num=10)

ax.plot(utils.to_dnum(ds.time),ds.flow_cms.isel(station=stn_idx),label='EBAYCc2 m3/s')

ax.xaxis.axis_date()
