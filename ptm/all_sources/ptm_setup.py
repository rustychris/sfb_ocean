import os
import glob
from stompy.model.fish_ptm import ptm_tools, ptm_config
from stompy.model.suntans import sun_driver

import numpy as np

## 
model=sun_driver.SuntansModel.load("/opt2/sfb_ocean/suntans/runs/merge_009-20170601")
model.load_bc_ds()

##

class Config(ptm_config.PtmConfig):
    def add_behaviors(self):
        self.lines+=["""\
BEHAVIOR INFORMATION
   NBEHAVIOR_PROFILES = 0
   NBEHAVIORS = 2

   -- behavior 1 ---
     BEHAVIOR_SET = 'down20000'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = 'down_20000.inp'
  
   -- behavior 2 ---
     BEHAVIOR_SET = 'up20000'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = 'up_20000.inp'

"""]

cfg=Config()
cfg.end_time=model.run_stop - np.timedelta64(3600,'s')
cfg.rel_time=model.run_start+np.timedelta64(3600,'s')
cfg.run_dir="ptm_20000"

# add releases

# flow BCs:

for seg_idx in range(len(model.bc_ds.Nseg)):
    flow_name=model.bc_ds.seg_name.values[seg_idx]
    print("Adding segment flow %s"%flow_name)

    segp=model.bc_ds.segp.values[seg_idx]
    type2s=np.nonzero( model.bc_ds.segedgep.values==segp)[0]
    edges=model.bc_ds.edgep.values[type2s]
    flow_xy=model.grid.edges_center()[edges]

    # flow-based at boundary edge
    release=["""\
   RELEASE_DISTRIBUTION_SET = '{name}' 
   MIN_BED_ELEVATION_METERS = -99.
   MAX_BED_ELEVATION_METERS =  99. 
   HORIZONTAL_DISTRIBUTION = 'side'
   SIDE_IDENTIFICATION_METHOD = 'list'
   NSIDES = {nsides}""".format(name=flow_name,nsides=len(flow_xy))]
    
    for xy in flow_xy:
        release+= ["   XCENTER = %.6f"%xy[0],
                   "   YCENTER = %.6f"%xy[1]]
    release+=["""\
   NPARTICLE_ASSIGNMENT = 'flow'
   DENSITY_SPECIFICATION_METHOD = 'constant'
   PARTICLE_DENSITY = {flow_particle_density}
   PARTICLE_DISTRIBUTION = 'random'
   ZMIN_NON_DIM = 0.0
   ZMAX_NON_DIM = 1.0
    VERT_SPACING = 'uniform'""".format(flow_particle_density=0.1)]
    
    cfg.releases.append(release)


# Add data for a single point release
for Npoint in range(len(model.bc_ds.Npoint)):
    region={}
    region['pnt_cell']=model.bc_ds.point_cell.isel(Npoint=Npoint)
    # avoid exact floating point comparisons - buffer
    # that polygon out a bit
    region['poly']=model.grid.cell_polygon(region['pnt_cell']).buffer(1.0,4)
    region['poly_name']="src%03d"%Npoint
    region['poly_fn']=region['poly_name']+".pol"
    
    pnts=np.array(region['poly'].exterior)

    with open(os.path.join(cfg.run_dir,region['poly_fn']),'wt') as fp:
        fp.write(" POLYGON_NAME\n")
        fp.write(" %s\n"%region['poly_name'])
        fp.write(" NUMBER_OF_VERTICES\n")
        fp.write(" %d\n"%len(pnts))
        fp.write(" EASTING   NORTHING (free format)\n")
        for pnt in pnts:
            fp.write(" %.6f %.6f\n"%(pnt[0],pnt[1]))

    region_lines=["""\
     REGION = '{name}'
     REGION_POLYGON_FILE = '{filename}'
    """.format(name=region['poly_name'],filename=region['poly_fn'])
                 ]
    
    cfg.regions.append(region_lines)


# for the moment, all point releases get the same distribution set
pnt_release=[
        """\
     RELEASE_DISTRIBUTION_SET = 'POINT' 
     MIN_BED_ELEVATION_METERS = -99.
     MAX_BED_ELEVATION_METERS =  99. 
     HORIZONTAL_DISTRIBUTION = 'region'
     DISTRIBUTION_IN_REGION = 'cell'
       CELL_RELEASE_TIMING = 'independent'
       PARTICLE_NUMBER_CALCULATION_BASIS = 'volume'
       -- aim high. was 10000.  made larger while debugging 
       VOLUME_PER_PARTICLE_CUBIC_METERS = 1.
     ZMIN_NON_DIM = 0.0
     ZMAX_NON_DIM = 0.05
     VERT_SPACING = 'uniform'
          """]
cfg.releases.append(pnt_release)

behaviors=['up20000','down20000','none']

# For each of the flow inputs, add up, down, neutral
for seg_idx in range(len(model.bc_ds.Nseg)):
    flow_name=model.bc_ds.seg_name.values[seg_idx]
    for behavior in behaviors:
        group=["""\
     GROUP = '{flow_name}_{behavior}'
     RELEASE_DISTRIBUTION_SET = '{flow_name}'
     RELEASE_TIMING_SET = 'flowbased'
     PARTICLE_TYPE = 'none'
     BEHAVIOR_SET = '{behavior}'
     OUTPUT_SET = '15min_output'
     OUTPUT_FILE_BASE = '{flow_name}_{behavior}'
        """.format(flow_name=flow_name,behavior=behavior)]
        cfg.groups.append(group)

# And for each point input:
for Npoint in range(len(model.bc_ds.Npoint)):
    for behavior in behaviors:
        group=["""\
     GROUP = '{point_name}_{behavior}'
     RELEASE_DISTRIBUTION_SET = 'POINT'
     REGION = '{region_name}'
     RELEASE_TIMING_SET = 'interval'
     PARTICLE_TYPE = 'none'
     BEHAVIOR_SET = '{behavior}'
     OUTPUT_SET = '15min_output'
     OUTPUT_FILE_BASE = '{point_name}_{behavior}'
        """.format(point_name="src%03d"%Npoint,
                   region_name="src%03d"%Npoint,
                   behavior=behavior)]
        cfg.groups.append(group)
        
cfg.clean()
cfg.write()


##       
if 0:            
    print("Running PTM")
    if 'LD_LIBRARY_PATH' in os.environ:
        del os.environ['LD_LIBRARY_PATH']
    pwd=os.getcwd()
    try:
        os.chdir(cfg.run_dir)
        subprocess.run(["/home/rusty/src/fish_ptm/PTM/FISH_PTM.exe"])
    finally:
        os.chdir(pwd)

