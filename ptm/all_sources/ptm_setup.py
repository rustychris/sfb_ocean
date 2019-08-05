import os
import glob
import six
from stompy.model.fish_ptm import ptm_tools, ptm_config
from stompy.model.suntans import sun_driver
six.moves.reload_module(ptm_config)
import numpy as np

## 
model=sun_driver.SuntansModel.load("/opt2/sfb_ocean/suntans/runs/merge_016-201706")
model.load_bc_ds()

##

class Config(ptm_config.PtmConfig):
    # positive: up, negative: down
    #rising_speeds_mps=[0.06,0.02,0.006,0.002,0.0006,
    #                   0,
    #                   -0.06,-0.02,-0.006,-0.002,-0.0006]
    # slim that down some
    rising_speeds_mps=[0.002,0,-0.002]
    #rising_speeds_mps=[0.06,0.006,0.0006,
    #                   0,
    #                   -0.06,-0.006,-0.0006]
    @property
    def behavior_names(self):
        names=[]
        for b_idx,w_mps in enumerate(self.rising_speeds_mps):
            if w_mps>0: # rising
                name="up%d"%(1e6*w_mps)
            elif w_mps<0:
                name="down%d"%(-1e6*w_mps)
            else:
                name="none"
            names.append(name)
        return names
        
    def add_behaviors(self):
        self.lines+=["""\
BEHAVIOR INFORMATION
   NBEHAVIOR_PROFILES = 0
   NBEHAVIORS = {nbehaviors}
""".format(nbehaviors=len([x for x in self.rising_speeds_mps if x!=0]))
        ]
        
        for b_idx,(name,w_mps) in enumerate(zip(self.behavior_names,self.rising_speeds_mps)):
            if w_mps==0:
                continue # no need to write 0
            if w_mps>0: # rising
                dist_option="depth"
            else: #  w_mps<0
                dist_option="height_above_bed"
            
            fname=name+".inp"
            self.lines+=[
"""   -- behavior {idx} ---
     BEHAVIOR_SET = '{name}'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = '{fname}'
""".format(idx=b_idx+1,name=name,fname=fname)]

            with open(os.path.join(self.run_dir,fname),'wt') as fp:
                fp.write("""\
 -- {w_mps} m/s
 -- NUMBER_OF_LAYERS = <integer>
 -- DISTANCE_OPTION = {{'depth', 'elevation' or 'height_above_bed'}}
 NLAYERS = 2
 DISTANCE_OPTION = '{dist_option}'
                          LAYER_1         LAYER 2
YYYY-MM-DD HH:MM:SS      DISTANCE SPEED  DISTANCE   SPEED
1990-01-01 00:00:00        0.500   0.000   0.6      {w_mps:.6f}
1990-01-01 01:00:00        0.500   0.000   0.6      {w_mps:.6f}
2030-01-01 00:00:00        0.500   0.000   0.6      {w_mps:.6f}
2030-01-01 01:00:00        0.500   0.000   0.6      {w_mps:.6f}"""
                         .format(dist_option=dist_option,w_mps=w_mps) )
            
    def add_output_sets(self):
        self.lines+=["""\
OUTPUT INFORMATION 
   NOUTPUT_SETS = 1

   OUTPUT_SET = '60min_output'
   FLAG_LOG_LOGICAL = 'true'
   BINARY_OUTPUT_INTERVAL_HOURS = 1.0
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 24
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 24.0
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 24.
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 24.
   STATE_OUTPUT_INTERVAL_HOURS = 1.0
"""]
        

cfg=Config()

#cfg.rel_time=model.run_start+5*24*np.timedelta64(1,'h')
cfg.rel_time=np.datetime64("2017-06-05")
# 1 day while testing:
cfg.end_time=cfg.rel_time + np.timedelta64(86400,'s')
# cfg.end_time=np.datetime64("2017-12-30")
# cfg.end_time = model.run_stop - np.timedelta64(3600,'s')

# cfg.run_dir="ptm_20000" # all sources, 3 behaviors, 24 hours
# cfg.run_dir='ebmud_all_w' # actually ebda, and the full slate of w_s, for june 2017.
# cfg.run_dir="napa_select_w"
# cfg.run_dir='ebda_most_w_dec' # parallel run starting in June, most behaviors, ebda release.

cfg.run_dir='compare_sediment' # 3 behaviors at each source, 100 particles/hour

particles_per_interval=100

# add releases
for seg_idx in range(len(model.bc_ds.Nseg)):
    flow_name=model.bc_ds.seg_name.values[seg_idx]
    print("Adding segment flow %s"%flow_name)

    segp=model.bc_ds.segp.values[seg_idx]
    type2s=np.nonzero( model.bc_ds.segedgep.values==segp)[0]
    edges=model.bc_ds.edgep.values[type2s]
    flow_xy=model.grid.edges_center()[edges]

    release=[f"""\
   RELEASE_DISTRIBUTION_SET = '{flow_name}' 
   MIN_BED_ELEVATION_METERS = -99.
   MAX_BED_ELEVATION_METERS =  99. 
   HORIZONTAL_DISTRIBUTION = 'side'
   SIDE_IDENTIFICATION_METHOD = 'list'
   NSIDES = {len(flow_xy)}"""]
    
    for xy in flow_xy:
        release+= ["   XCENTER = %.6f"%xy[0],
                   "   YCENTER = %.6f"%xy[1]]
    release+=[f"""\
   NPARTICLE_ASSIGNMENT = 'specify'
    NPARTICLES_PER_RELEASE_INTERVAL = {particles_per_interval}
    AVERAGE_NPARTICLES_IN_VERTICAL=10
    DISTRIBUTION_AMONG_WATER_COLUMNS = 'depth_weighted'
   ZMIN_NON_DIM = 0.0
   ZMAX_NON_DIM = 1.0
   VERT_SPACING = 'uniform'"""]
    
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
        f"""\
         RELEASE_DISTRIBUTION_SET = '{name}' 
         MIN_BED_ELEVATION_METERS = -99.
         MAX_BED_ELEVATION_METERS =  99. 
         HORIZONTAL_DISTRIBUTION = 'line'
            XSTART = {pnt[0]:.5f}
            YSTART = {pnt[1]:.5f}
            XEND   = {pnt[0]:.5f}
            YEND   = {pnt[1]:.5f}
            NPARTICLE_ASSIGNMENT = 'specify'
              TIME_VARIABLE_RELEASE = 'false'
              NPARTICLES_PER_RELEASE_INTERVAL = {particles_per_interval}
              -- average number of particles per water column
              AVERAGE_NPARTICLES_IN_VERTICAL = 10
              -- method of setting the number of particles released in water column
              DISTRIBUTION_AMONG_WATER_COLUMNS = 'uniform'
         ZMIN_NON_DIM = 0.0
         ZMAX_NON_DIM = 0.05
         VERT_SPACING = 'uniform'
""" ]
    cfg.releases.append(pnt_release)

# full menu:
enable_sources=[
    'SacRiver',
    'SJRiver',
    'COYOTE', 
    'SCLARAVCc', 
    'UALAMEDA', 
    'NAPA', 
    'sunnyvale', 
    'san_jose', 
    'palo_alto', 
    'lg', 
    'sonoma_valley', 
    'petaluma', 
    'cccsd', 
    'fs', 
    'ddsd',
    'src000', # EBDA
    'src001', # EBMUD
    'src002'  # SFPUC
]

# enable_sources=['src000','NAPA']
    
# For each of the flow inputs, add up, down, neutral
for behavior in cfg.behavior_names:
    for seg_idx in range(len(model.bc_ds.Nseg)):
        flow_name=model.bc_ds.seg_name.values[seg_idx]
        if (enable_sources is not None) and (flow_name not in enable_sources):
            continue
        group=["""\
     GROUP = '{flow_name}_{behavior}'
     RELEASE_DISTRIBUTION_SET = '{flow_name}'
     RELEASE_TIMING_SET = 'interval'
     PARTICLE_TYPE = 'none'
     BEHAVIOR_SET = '{behavior}'
     OUTPUT_SET = '60min_output'
     OUTPUT_FILE_BASE = '{flow_name}_{behavior}'
        """.format(flow_name=flow_name,behavior=behavior)]
        cfg.groups.append(group)

    # And for each point input:
    for Npoint in range(len(model.bc_ds.Npoint)):
        point_name="src%03d"%Npoint
        if (enable_sources is not None) and (point_name not in enable_sources):
            continue

        group=["""\
     GROUP = '{point_name}_{behavior}'
     RELEASE_DISTRIBUTION_SET = '{point_name}'
     RELEASE_TIMING_SET = 'interval'
     PARTICLE_TYPE = 'none'
     BEHAVIOR_SET = '{behavior}'
     OUTPUT_SET = '60min_output'
     OUTPUT_FILE_BASE = '{point_name}_{behavior}'
        """.format(point_name=point_name,
                   behavior=behavior)]
        cfg.groups.append(group)
        
cfg.clean()
cfg.write()

##

# try to generate the hydro part
# assumes that all average files have been converted, and prefixed
# with ptm_
models=model.chain_restarts()
lines=[]
file_count=0
for mod in models:
    for avg in mod.avg_outputs():
        ptm_avg=os.path.join(os.path.dirname(avg),"ptm_"+os.path.basename(avg))
        assert os.path.exists(ptm_avg),ptm_avg
        file_count+=1
        lines+=[" -- INPUT FILE %d --"%file_count,
                "  HYDRO_FILE_PATH = '%s/'"%os.path.dirname(ptm_avg),
                "  FILENAME = '%s'"%os.path.basename(ptm_avg),
                "  GRD_NAME = '%s'"%os.path.basename(ptm_avg),
                ""]
        

with open(os.path.join(cfg.run_dir,"FISH_PTM_hydrodynamics.inp"),'wt') as fp:
    fp.write(" NUM_FILES = %d\n"%file_count)
    fp.write("\n".join(lines))

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

