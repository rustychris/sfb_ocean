"""
Reimplementation of ocean_dfm.py, using more recent DFlowModel code.
"""
import six
from stompy.spatial import interpXYZ
from stompy.model.otps import read_otps
import stompy.model.delft.dflow_model as dfm
six.moves.reload_module(interpXYZ)
six.moves.reload_module(read_otps)
six.moves.reload_module(dfm)

class SFOceanModel(dfm.DFlowModel):
    dfm_bin_dir="/home/rusty/src/dfm/r53925-opt/bin"
    num_procs=1
    z_datum='NAVD88'
    projection='EPSG:26910'

    def bc_factory(self,params):
        if params['name']=='Ocean':
            # apply the same boundary to all of those edges:
            # return dfm.OTPSStageBC(self,otps_model='wc',**params)
            return dfm.MultiBC(self,dfm.OTPSStageBC,otps_model='wc',**params)
        else:
            raise Exception("Unrecognized %s"%str(params))

    def config_z_layers(self):
        mdu=self.mdu
        mdu['geometry','Kmx']=20
        mdu['geometry','Layertype']=2 # z layers
        mdu['geometry','StretchType']=2 # exponential
        # surface percentage, ignored, bottom percentage
        # this should move the first interface down to below -1m for max depth 1000
        mdu['geometry','StretchCoef']="0.002 0.02 0.7"
        mdu['numerics','Zwsbtol'] = 0.0 # that's the default anyway...
        # This is the safer of the options, but gives a stairstepped bed.
        mdu['numerics','Keepzlayeringatbed']=1
        # This helps with reconstructing the z-layer geometry, better than
        # trying to duplicate dflowfm layer code.
        mdu['output','FullGridOutput']    = 1

model=SFOceanModel()
model.set_cache_dir('cache')
model.load_mdu('template.mdu')

model.set_run_dir('runs/ocean2_001')

model.run_start=np.datetime64('2017-07-01')
model.run_stop =np.datetime64('2017-08-01')

import ragged_grid as grid_mod

model.config_z_layers()
model.set_grid(grid_mod.grid)
model.add_bcs_from_features(grid_mod.bc_features)

model.write()
model.partition()
model.run_model()

# Return to HYCOM config -- get otis tides and 3D going first.
