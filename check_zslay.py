import xarray as xr
import stompy.model.delft.io as dio
##

ds=xr.open_dataset('runs/short_test_12_test/matched_grid_v01_net.nc')

mdu=dio.MDUFile('runs/short_test_12_test/short_test_12-tmp.mdu')
##


# This agrees with both NetNode_z.min(), and FlowElem_zcc.max()
# and FlowElem_bl.min().  All -4155 ish.

##

##


zslay=np.array( [-4155.000,
                 -3315.789,
                 -2644.420,
                 -2107.325,
                 -1677.649,
                 -1333.908,
                 -1058.916,
                 -838.922,
                 -662.926,
                 -522.130,
                 -409.493,
                 -319.383,
                 -247.296,
                 -189.626,
                 -143.490,
                 -106.581,
                 -77.054,
                 -53.432,
                 -34.535,
                 -19.417,
                 -7.322] )


plt.figure(1).clf()
    
plt.plot(k,zslay,'b-o')

##

        
zslay=z_layers(mdu)
##

# Another investigation - how to control size of the near-bed zlayers:

ds=xr.open_dataset('runs/short_test_13/DFM_OUTPUT_short_test_13/short_test_13_0002_map.nc')

zws=ds.FlowElem_zw.isel(time=0,nFlowElem=73).values
zws_val=zws[zws<1e35]

for k in range(len(zws_val)):
    if k>0:
        delta="%10.3f"%(zws[k]-zws[k-1])
    else:
        delta=""
    print(" k=%3d  zws=%10.3f  delta=%s"%(k,zws[k],delta))
# ds.close()

# first number maybe declares how thick the surface cell is?  or the elevation of the
# first interface below the waterlevini, as a fraction of waterlevini to zmin.

##

# What can be assumed about FlowElem_zw?
#  - does the number of layers ever change?
zcount=(ds.FlowElem_zw<1e30).sum(dim='wdim')
zcount_change=np.abs(np.diff(zcount.values,axis=0)).max()
# - at least with the current setup, it never changes.
# does this match with no surface cells being nan?
# seems to match up.
# So no info on what happens when a surface cell dries up.

# Any metadata to help out here?
# FlowElem_zw has basically no metadata (units, and a long_name)

