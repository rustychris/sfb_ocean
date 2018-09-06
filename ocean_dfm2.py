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

read_otps.OTPS_DATA='derived'

class SFOceanModel(dfm.DFlowModel):
    num_procs=1
    z_datum='NAVD88'
    projection='EPSG:26910'

    def bc_factory(self,params):
        if params['name']=='Ocean':
            # spatially constant water level
            # return dfm.OTPSStageBC(self,otps_model='wc',**params)
            # spatially varying water level:
            # return dfm.MultiBC(self,dfm.OTPSStageBC,otps_model='wc',**params)
            # spatially varying transport
            # return dfm.MultiBC(self,dfm.OTPSVelocityBC,otps_model='wc',**params)
            # spatially varying transport with potential for depth-varying:
            return dfm.MultiBC(self,dfm.OTPSVelocity3DBC,otps_model='wc',**params)
        else:
            raise Exception("Unrecognized %s"%str(params))

    def config_2d(self):
        self.mdu['geometry','Kmx']=0

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

# This version does not handle 3D velocity BCs
# model.dfm_bin_dir="/home/rusty/src/dfm/r53925-opt/bin"
# This version has a patch which might help with 3D BCs
# It did not ultimately work very well.
# model.dfm_bin_dir="/home/rusty/src/dfm/r53925-zbndu-dbg/bin"
# This version *should* be what I was using on hpc
model.dfm_bin_dir="/home/rusty/src/dfm/r52184-dbg/bin"

model.set_cache_dir('cache')
model.load_mdu('template.mdu')

# ocean2_001: first 3D, but then revert to 2D.  Good tidal shape, but amplitude too small.
#    OTPSVelocityBC was dividing by the edge length, but that isn't right.
# ocean2_002: don't divide by length in OTPSVelocityBC
#    data for this run probably got corrupted.
# ocean2_003: back to 3D
# ocean2_004: 3D, using 52184.
model.set_run_dir('runs/ocean2_004',mode='clean')

model.run_start=np.datetime64('2017-07-01')
model.run_stop =np.datetime64('2017-08-01')

import ragged_grid as grid_mod
six.moves.reload_module(grid_mod)

model.config_z_layers()
#model.config_2d()

model.set_grid(grid_mod.grid)
model.add_bcs_from_features(grid_mod.bc_features)

model.write()
model.partition()
model.run_model()


# This is running, and with r52184, doing pretty well.
# Forcing is applied as 2D, but it's a 3D run.

# Next: extrude the forcing to 3D.
#  so far so good - sigma extends a bit beyond [0,1]
# running, looks fine.
# But when I try to add some vertical structure, it appears
# that everything just goes a bit negative when looking at
# ucx.
# this persists through the tidal cycle.
# what if I swap the sign?
# There were only a few water columns, near shore, which showed
# and vertical structure.  maybe because they were closed cells?
# swapping sign does swap everything visible.
# sample vertical at 10 layers

# starting to think that this is a fundamental limitation of
# DFM in this revision.

# in furu():
#   there is a call to update_vertical_profiles() - not sure what that does
#   that is called instead of a bunch of 2D stuff at the top.  that might
#   rule out pumps as an approach
# crap - furu doesn't support 3D velocity BC.  it makes a call to update_vertical_profiles,
#   unclear exactly what that does.
#   it optionally applies a log profile to the boundary.
# update_vertical_profiles seems to be about the turbulence closure.
# seems that ru is the destination for velocity BCs.
# what about getustbcfuhi()?  that's for ustar.

# hmm - what was the deal with bastardizing sources and sinks?
# Try a manual run with sources and sinks.
# Try a BC around 422248, 4.12232e+06, that's middle of the western ocean BC

# That might work.  But it's a gross hack.  Maybe better to see if recent
# svn versions are any better at 3D.

