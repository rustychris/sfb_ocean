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
cfg.rel_time=model.run_start+np.timedelta64(3600,'s')
# 1 day while testing:
# cfg.end_time=cfg.rel_time + np.timedelta64(86400,'s')
cfg.end_time=model.run_stop - np.timedelta64(3600,'s')

#cfg.run_dir="ptm_20000" # all sources, 3 behaviors, 24 hours
cfg.run_dir="ebmud_2months"

# add releases

# flow BCs:
flow_particle_density={}
# These are scaled to get roughly 500/day, at least in dry weather.
flow_particle_density['SacRiver']=0.0001
flow_particle_density['SJRiver']=0.0001
flow_particle_density['COYOTE']=0.25
flow_particle_density['NAPA']=0.17
flow_particle_density['SCLARAVCc']=0.12
flow_particle_density['UALAMEDA']=1.0
flow_particle_density['cccsd']=0.1
flow_particle_density['ddsd']=0.5
flow_particle_density['fs']=0.33
flow_particle_density['lg']=1.0
flow_particle_density['palo_alto']=0.17
flow_particle_density['petaluma']=1.
flow_particle_density['san_jose']=0.04
flow_particle_density['sonoma_valley']=1.0
flow_particle_density['sunnyvale']=0.38

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
    VERT_SPACING = 'uniform'""".format(flow_particle_density=flow_particle_density[flow_name])]
    
    cfg.releases.append(release)


# Add data for a single point release
cc=model.grid.cells_center()
for Npoint in range(len(model.bc_ds.Npoint)):
    # regions are harder to control the particle count for.
    # try a degenerate line.  region code in past git revision.
    
    cell=model.bc_ds.point_cell.isel(Npoint=Npoint)
    name="src%03d"%Npoint
    pnt=cc[cell]

    pnt_release=[
            """\
         RELEASE_DISTRIBUTION_SET = '{name}' 
         MIN_BED_ELEVATION_METERS = -99.
         MAX_BED_ELEVATION_METERS =  99. 
         HORIZONTAL_DISTRIBUTION = 'line'
            XSTART = {x0:.5f}
            YSTART = {y0:.5f}
            XEND   = {x1:.5f}
            YEND   = {y1:.5f}
            NPARTICLE_ASSIGNMENT = 'specify'
              TIME_VARIABLE_RELEASE = 'false'
              NPARTICLES_PER_RELEASE_INTERVAL = 20
              -- average number of particles per water column
              AVERAGE_NPARTICLES_IN_VERTICAL = 10
              -- method of setting the number of particles released in water column
              DISTRIBUTION_AMONG_WATER_COLUMNS = 'uniform'
         ZMIN_NON_DIM = 0.0
         ZMAX_NON_DIM = 0.05
         VERT_SPACING = 'uniform'
""".format(name=name,
           x0=pnt[0],y0=pnt[1],x1=pnt[0],y1=pnt[1])
    ]
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
     OUTPUT_SET = '30min_output'
     OUTPUT_FILE_BASE = '{flow_name}_{behavior}'
        """.format(flow_name=flow_name,behavior=behavior)]
        cfg.groups.append(group)

# And for each point input:
for Npoint in range(len(model.bc_ds.Npoint)):
    name="src%03d"%Npoint

    for behavior in behaviors:
        group=["""\
     GROUP = '{point_name}_{behavior}'
     RELEASE_DISTRIBUTION_SET = '{point_name}'
     RELEASE_TIMING_SET = 'interval'
     PARTICLE_TYPE = 'none'
     BEHAVIOR_SET = '{behavior}'
     OUTPUT_SET = '30min_output'
     OUTPUT_FILE_BASE = '{point_name}_{behavior}'
        """.format(point_name="src%03d"%Npoint,
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

