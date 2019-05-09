# How best to set up PTM runs
# test things out with a small test domain
#   test the idea of running hydro, then using the BC netcdf
#   file to configure PTM runs.

import os
from stompy.grid import unstructured_grid
import numpy as np
from stompy.model.suntans import sun_driver

import xarray as xr

sun_driver.SuntansModel.sun_bin_dir="/home/rusty/src/suntans/main"
sun_driver.SuntansModel.mpi_bin_dir="/usr/bin"

g=unstructured_grid.UnstructuredGrid(max_sides=4)
g.add_rectilinear([0,0],[500,500],50,50)

# wavy bed
half_wave=150
cc=g.cells_center()
cell_depth=-6 + np.cos(cc[:,0]*np.pi/half_wave) * np.cos(cc[:,1]*np.pi/half_wave)
g.add_cell_field('depth',cell_depth)

model=sun_driver.SuntansModel()
model.load_template('template.dat')
model.set_grid(g)
model.run_start=np.datetime64("2018-01-01 00:00")
model.run_stop =np.datetime64("2018-01-05 00:00")

dt=np.timedelta64(600,'s')
times=np.arange(model.run_start-dt,model.run_stop+2*dt,dt)
secs=(times-times[0])/np.timedelta64(1,'s')
# unlike the original point source case, I don't want to
# get bogged down stress testing wetting and drying right now.
# so bump up that eta.
eta_values=-3 + 0.75*np.cos(secs*np.pi/7200)
eta_da=xr.DataArray(eta_values,coords=[ ('time',times) ])
eta_bc=sun_driver.StageBC(name="eta_bc",geom=np.array([ [0,0],
                                                        [0,500]]),
                          z=eta_da)

model.add_bcs(eta_bc)
model.add_bcs( [sun_driver.ScalarBC(parent=eta_bc,scalar="S",value=1),
                sun_driver.ScalarBC(parent=eta_bc,scalar="T",value=1)] )

# point source that is typically dry
hill_source=sun_driver.SourceSinkBC(name='inflow',geom=np.array([150,150]),
                                    z=-10,Q=1.0)

# on a saddle
saddle_source=sun_driver.SourceSinkBC(name='inflow',geom=np.array([220,225]),
                                      z=-10,Q=1.0)

model.add_bcs(hill_source)
model.add_bcs(saddle_source)
model.add_bcs( [sun_driver.ScalarBC(parent=hill_source,scalar="S",value=1),
                sun_driver.ScalarBC(parent=hill_source,scalar="T",value=0.5)] )
model.add_bcs( [sun_driver.ScalarBC(parent=saddle_source,scalar="S",value=1),
                sun_driver.ScalarBC(parent=saddle_source,scalar="T",value=0.5)] )

# and a flow BC
Q_bc=sun_driver.FlowBC(name='river',
                       geom=np.array([[500,500], [500,450]]),
                       Q=10.0)
model.add_bcs(Q_bc)
model.add_bcs( [sun_driver.ScalarBC(parent=Q_bc,scalar="S",value=1),
                sun_driver.ScalarBC(parent=Q_bc,scalar="T",value=2)] )

model.set_run_dir('rundata', mode='pristine')
model.projection='EPSG:26910'
model.num_procs=4
model.config['dt']=5.0
model.config['ntout']=int(1800/float(model.config['dt']))
model.config['Cmax']=30
model.config['Nkmax']=10
model.config['stairstep']=0
model.config['mergeArrays']=1
# PTM output
model.config['ntaverage']=int(1800/float(model.config['dt']))
model.config['calcaverage']=1
model.config['averageNetcdfFile']="average.nc"

model.write()

model.ic_ds.eta.values[:]=eta_da.values[1]
model.ic_ds.salt.values[:]=1.0
model.ic_ds.temp.values[:]=1.0
model.write_ic_ds()

model.partition()
model.sun_verbose_flag='-v'
model.run_simulation()

##
from stompy import utils
utils.path('/home/rusty/src')

from soda.dataio.suntans import sunpy
six.moves.reload_module(sunpy)
from soda.dataio.ugrid import suntans2untrim
six.moves.reload_module(suntans2untrim)

for idx,avg_file in enumerate(model.avg_outputs()):
    out_file=os.path.join(model.run_dir,"ptm_hydro_%04d.nc"%idx)
    suntans2untrim.suntans2untrim(avg_file, out_file, None, None)

## 

