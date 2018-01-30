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
