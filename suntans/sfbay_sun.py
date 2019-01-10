"""
Port SF Bay DFM to suntans, for testing ahead of merging with
ocean domain.
"""
from stompy.model.suntans import sun_driver

import six
import os
import datetime

import numpy as np
import xarray as xr
from stompy.spatial import proj_utils, field, wkb2shp
from stompy.model.delft import dfm_grid
import stompy.model.delft.io as dio
from stompy.model import otps
from stompy import utils, filters

import logging as log

from stompy.io.local import hycom
cache_dir='cache'

from stompy.grid import unstructured_grid

from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv
##
six.moves.reload_module(dfm)
six.moves.reload_module(drv)

use_temp=True
use_salt=True

# bay003: add perimeter freshwater sources
# bay004: debugging missing san_jose
run_dir='/opt/sfb_ocean/suntans/runs/bay004'

model=drv.SuntansModel()
model.run_start=np.datetime64("2017-06-15")
model.run_stop=np.datetime64("2017-06-18")

model.projection="EPSG:26910"
model.num_procs=4
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.use_edge_depths=True
model.load_template("sun-template.dat")

model.set_run_dir(run_dir,mode='askclobber')
model.config['Nkmax']=1
model.config['stairstep']=0

dt_secs=5.0
model.config['dt']=dt_secs
model.config['ntout']=int(15*60/dt_secs) # quarter-hour map output
model.config['ntoutStore']=int(86400/dt_secs) # # daily restart file
model.config['mergeArrays']=0
model.config['rstretch']=1.1
model.config['Cmax']=30.0 # volumetric is a better test, this is more a backup.
# esp. with edge depths, seems better to use z0B so that very shallow
# edges can have the right drag.
model.config['CdB']=0
model.config['z0B']=0.001
model.z_offset=-5
model.dredge_depth=-2

if use_temp:
    model.config['gamma']=0.00021
else:
    model.config['gamma']=0.0

if use_salt:
    model.config['beta']=0.00077
else:
    model.config['beta']=0.0

dest_grid="grid-sfbay/sfei_v20_net.nc"
assert os.path.exists(dest_grid),"Grid %s not found"%dest_grid

g=unstructured_grid.UnstructuredGrid.read_dfm(dest_grid)

# for now, just cell depth from the node depths already in there
g.add_cell_field('depth',g.interp_node_to_cell(g.nodes['depth']))
g.delete_node_field('depth')
if 1: # edges take shallower cell
    de=np.zeros(g.Nedges(),np.float64)
    e2c=g.edge_to_cells()
    c1=e2c[:,0].copy() ; c2=e2c[:,1].copy()
    c1[c1<0]=c2[c1<0]
    c2[c2<0]=c1[c2<0]
    de=np.maximum(g.cells['depth'][c1],g.cells['depth'][c2])
if 1: # add levee elevations
    # load levee data:
    levee_fn='grid-sfbay/SBlevees_tdk.pli'
    levees=dio.read_pli(levee_fn)
    levee_de=dio.pli_to_grid_edges(g,levees)
    missing=np.isnan(levee_de)
    levee_de[missing]=de[missing]
    # levees only raise edges
    de=np.maximum(de,levee_de)

g.add_edge_field('edge_depth',de,on_exists='overwrite')

model.set_grid(g)

model.add_gazetteer("grid-sfbay/linear_features.shp")
model.add_gazetteer("grid-sfbay/point_features.shp")

ocean_salt_bc=drv.ScalarBC(name='ocean',scalar='salinity',value=34)
ocean_temp_bc=drv.ScalarBC(name='ocean',scalar='temperature',value=10)
ocean_bc=drv.NOAAStageBC(name='ocean',station=9415020,cache_dir=cache_dir,
                         filters=[dfm.Lowpass(cutoff_hours=1.5)])
model.add_bcs([ocean_bc,ocean_salt_bc,ocean_temp_bc]) 

#

# Delta inflow
# SacRiver, SJRiver

sac_bc=dfm.NwisFlowBC(name='SacRiver',station=11455420,cache_dir=cache_dir,
                      filters=[dfm.Lowpass(cutoff_hours=3)])
sj_bc =dfm.NwisFlowBC(name='SJRiver',station=11337190,cache_dir=cache_dir,
                      filters=[dfm.Lowpass(cutoff_hours=3)])

sac_salt_bc=drv.ScalarBC(name='SacRiver',scalar='salinity',value=0.0)
sj_salt_bc =drv.ScalarBC(name='SJRiver',scalar='salinity',value=0.0)
sac_temp_bc=drv.ScalarBC(name='SacRiver',scalar='temperature',value=20.0)
sj_temp_bc =drv.ScalarBC(name='SJRiver',scalar='temperature',value=20.0)

model.add_bcs([sac_bc,sj_bc,sac_salt_bc,sj_salt_bc,sac_temp_bc,sj_temp_bc])

#
if 1: # disable if no internet
    # USGS gauged creeks
    for station,name in [ (11172175, "COYOTE"),
                          (11169025, "SCLARAVCc"), # Alviso Sl / Guad river
                          (11180700,"UALAMEDA"), # Alameda flood control
                          (11458000,"NAPA") ]:
        Q_bc=dfm.NwisFlowBC(name=name,station=station,cache_dir=cache_dir)
        salt_bc=drv.ScalarBC(name=name,scalar='salinity',value=0.0)
        temp_bc=drv.ScalarBC(name=name,scalar='temperature',value=20.0)

        model.add_bcs([Q_bc,salt_bc,temp_bc])
else:
    print("Disabling USGS gauged inputs")
#

# WWTP discharging into sloughs
potw_dir="../sfbay_potw"
potw_ds=xr.open_dataset( os.path.join(potw_dir,"outputs","sfbay_delta_potw.nc"))

# the gazetteer uses the same names for potws as the source data
# omits some of the smaller sources, and this does not include any
# benthic discharges
for potw_name in ['sunnyvale','san_jose','palo_alto',
                  'lg','sonoma_valley','petaluma','cccsd','fs','ddsd',
                  'ebda','ebmud','sf_southeast']:
    Q_da=potw_ds.flow.sel(site=potw_name)
    # Have to seek back in time to find a year that has data for the
    # whole run
    offset=np.timedelta64(0,'D')
    while model.run_stop > Q_da.time.values[-1]+offset:
        offset+=np.timedelta64(365,'D')
    if offset:
        print("Offset for POTW %s is %s"%(potw_name,offset))

    # use the geometry to decide whether this is a flow BC or a point source
    hits=model.match_gazetteer(name=potw_name)
    if hits[0]['geom'].type=='LineString':
        print("%s: flow bc"%potw_name)
        Q_bc=drv.FlowBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)])
    else:
        print("%s: source bc"%potw_name)
        Q_bc=drv.SourceSinkBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)])
        
    salt_bc=drv.ScalarBC(parent=Q_bc,scalar='salinity',value=0.0)
    temp_bc=drv.ScalarBC(parent=Q_bc,scalar='temperature',value=20.0)
    model.add_bcs([Q_bc,salt_bc,temp_bc])
    
model.write()

## 
# while developing, initialize everywhere to 34ppt, so we can see rivers
# coming in
model.ic_ds.salt.values[:]=34
model.ic_ds.temp.values[:]=10
model.write_ic_ds()

model.partition()

# model.run_simulation()

##
import matplotlib.pyplot as plt
plt.figure(1).clf()
ecoll=model.grid.plot_edges(values=model.grid.edges['edge_depth'],clip=zoom,cmap='jet')
plt.axis('equal')
plt.colorbar(ecoll)
