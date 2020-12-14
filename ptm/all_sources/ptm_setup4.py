"""
Adjustments to behaviors to better mimic manta thickness.
"""
import os
import glob
import subprocess
import logging as log

import six
from stompy.model.fish_ptm import ptm_tools, ptm_config
from stompy.model.suntans import sun_driver
from stompy import utils
import numpy as np

## 


class Config(ptm_config.PtmConfig):
    model=None # a HydroModel instance, chainable back
    model_dirs=None # if set, this will be used to get the list of models instead of
    # chaining restarts
    model_dir=None # if set, and model is not set, this will be loaded and used to
    # set model. useful for multiprocessing
    
    # positive: up, negative: down
    #rising_speeds_mps=[0.06,0.02,0.006,0.002,0.0006,
    #                   0,
    #                   -0.06,-0.02,-0.006,-0.002,-0.0006]
    # slim that down some
    # rising_speeds_mps=[0.002,0,-0.002]

    # too many groups to do them all at once
    #rising_speeds_mps=[0.05,0.005,0.0005,
    #                   0,
    #                   -0.05,-0.005,-0.0005]

    rising_speeds_mps=None # filled in below
    # 5 is for pretty small runs.
    particles_per_interval=20
    rel_times=None # release time
    rel_duration=None # how long release go for after rel_time
    group_duration=None # how long the group runs for, or end_time if unspecified
    end_time=None # end of PTM run
    sources=None # list of source names to include

    def __init__(self,*a,**k):
        super(Config,self).__init__(*a,**k)

        if self.model is None and self.model_dir is not None:
            self.model=sun_driver.SuntansModel.load(self.model_dir)
            self.model.load_bc_ds()

        # gets populated as the sources are defined
        self.all_source_names=[]
        
        # make everything a sequence
        def to_seq(x):
            if isinstance(x,np.ndarray) or isinstance(x,list):
                return x
            else:
                return [x]
            
        self.rel_times=to_seq(self.rel_times)
        self.rising_speeds_mps=to_seq(self.rising_speeds_mps)
        self.sources=to_seq(self.sources)

        if self.end_time is None:
            group_ends=[rel+self.group_duration
                        for rel in self.rel_times]
            self.end_time=np.max(group_ends)
            print(f"Overall end time deduced to be {self.end_time}")

    def rising_speed_to_name(self,w_mps):
        if w_mps>0: # rising
            name="up%d"%(1e6*w_mps)
        elif w_mps<0:
            name="down%d"%(-1e6*w_mps)
        else:
            name="none"
        return name
    @property
    def behavior_names(self):
        return [self.rising_speed_to_name(w_mps)
                for w_mps in self.rising_speeds_mps]
        
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

            # 2019-10-12: change buffer region from 0.5/0.6
            #  to 0.2/0.3
            with open(os.path.join(self.run_dir,fname),'wt') as fp:
                fp.write("""\
 -- {w_mps} m/s
 -- NUMBER_OF_LAYERS = <integer>
 -- DISTANCE_OPTION = {{'depth', 'elevation' or 'height_above_bed'}}
 NLAYERS = 2
 DISTANCE_OPTION = '{dist_option}'
                          LAYER_1         LAYER 2
YYYY-MM-DD HH:MM:SS      DISTANCE SPEED  DISTANCE   SPEED
1990-01-01 00:00:00        0.050   0.000   0.075      {w_mps:.6f}
1990-01-01 01:00:00        0.050   0.000   0.075      {w_mps:.6f}
2030-01-01 00:00:00        0.050   0.000   0.075      {w_mps:.6f}
2030-01-01 01:00:00        0.050   0.000   0.075      {w_mps:.6f}"""
                         .format(dist_option=dist_option,w_mps=w_mps) )
            
    def add_output_sets(self):
        # when doing sediment particles, may want that state output,
        # but otherwise could probably set to 'none'
        # note that to turn state output on, need not just the interval
        # but then also something about number of variables output and
        # the list of variables
        self.lines+=["""\
OUTPUT INFORMATION 
   NOUTPUT_SETS = 1

   OUTPUT_SET = '60min_output'
   FLAG_LOG_LOGICAL = 'false'
   BINARY_OUTPUT_INTERVAL_HOURS = 1.0
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 'none'
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 'none'
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 'none'
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 'none'
   STATE_OUTPUT_INTERVAL_HOURS = 'none'
"""]

    def set_release_timing(self):
        self.release_timing_names=["rel"+utils.to_datetime(rel_time).strftime('%Y%m%d')
                                   for rel_time in self.rel_times]
        
    def add_release_timing(self):
        self.lines+=[f"""\
RELEASE TIMING INFORMATION
  NRELEASE_TIMING_SETS = {len(self.rel_times)}"""]
        def dfmt(t):
            return utils.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")

        for idx,(rel_name,rel_time) in enumerate(zip(self.release_timing_names,self.rel_times)):
            if self.group_duration is not None:
                inactive=dfmt(rel_time+self.group_duration)
            else:
                inactive='none'
            self.lines+=[f"""
    -- release timing set {idx+1} ---        
     RELEASE_TIMING_SET = '{rel_name}'
     INITIAL_RELEASE_TIME = '{dfmt(rel_time)}'
     RELEASE_TIMING = 'interval'
       NINTERVALS = {int(self.rel_duration/np.timedelta64(3600,'s'))}
       RELEASE_INTERVAL_HOURS = 1.0
            INACTIVATION_TIME = '{inactive}'"""
          ]
        
    def method_text(self):
        # bumping up the horizontal diffusivity to 0.5 m2/s
        return """\
 MAX_HORIZONTAL_ADVECTION_SUBSTEPS = 10
 MAX_HORIZONTAL_DIFFUSION_SUBSTEPS = 10
 GRID_TYPE = 'unstructured'
 ADVECTION_METHOD = 'streamline'
   NORMAL_VELOCITY_GRADIENT = 'constant'
 VERT_COORD_TYPE = 'z-level'
 HORIZONTAL_DIFFUSION_METHOD = 'constant'
   CONSTANT_HORIZONTAL_EDDY_DIFFUSIVITY = 0.5
 VERTICAL_ADVECTION_METHOD = 'streamline'
 MIN_VERTICAL_EDDY_DIFFUSIVITY = 0.00001
 MAX_VERTICAL_EDDY_DIFFUSIVITY = 0.10000
 MAX_VERTICAL_DIFFUSION_SUBSTEPS = 100
 MIN_VERTICAL_DIFFUSION_TIME_STEP = 1.0
 RANDOM_NUMBER_DISTRIBUTION = 'normal'
 SPECIFY_RANDOM_SEED = 'true'
   SPECIFIED_RANDOM_SEED = 1
 REMOVE_DEAD_PARTICLES = 'false'
 SUBGRID_BATHY = 'false'
"""

    def set_releases(self):           
        # add releases
        self.set_flow_releases()
        self.set_point_releases()
    def set_flow_releases(self):
        # Creates release distribution sets for all flow BCs.
        # note that they aren't necessarily enabled, they're
        # just setup
        for seg_idx in range(len(self.model.bc_ds.Nseg)):
            flow_name=self.model.bc_ds.seg_name.values[seg_idx]
            if self.sources is not None and flow_name not in self.sources:
                continue
            print("Adding segment flow %s"%flow_name)

            segp=self.model.bc_ds.segp.values[seg_idx]
            type2s=np.nonzero( self.model.bc_ds.segedgep.values==segp)[0]
            edges=self.model.bc_ds.edgep.values[type2s]
            flow_xy=self.model.grid.edges_center()[edges]

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
             TIME_VARIABLE_RELEASE = 'false'
             NPARTICLES_PER_RELEASE_INTERVAL = {self.particles_per_interval}
             AVERAGE_NPARTICLES_IN_VERTICAL = 1
             DISTRIBUTION_AMONG_WATER_COLUMNS = 'depth_weighted'
           ZMIN_NON_DIM = 0.0
           ZMAX_NON_DIM = 1.0
           VERT_SPACING = 'uniform'"""]

            self.releases.append(release)
    def set_point_releases(self):
        # Add data for a single point release
        cc=self.model.grid.cells_center()
        for Npoint in range(len(self.model.bc_ds.Npoint)):
            # regions are harder to control the particle count for.
            # try a degenerate line.  region code in past git revision.

            cell=self.model.bc_ds.point_cell.isel(Npoint=Npoint)
            name="src%03d"%Npoint
            pnt=cc[cell]

            if self.sources is not None and name not in self.sources:
                continue

            pnt_release=[
                f"""\
                 RELEASE_DISTRIBUTION_SET = '{name}' 
                 MIN_BED_ELEVATION_METERS = -99.
                 MAX_BED_ELEVATION_METERS =  99. 
                 HORIZONTAL_DISTRIBUTION = 'line'
                    XSTART = {pnt[0]:.6f}
                    YSTART = {pnt[1]:.6f}
                    XEND   = {pnt[0]:.6f}
                    YEND   = {pnt[1]:.6f}
                    NPARTICLE_ASSIGNMENT = 'specify'
                      TIME_VARIABLE_RELEASE = 'false'
                      NPARTICLES_PER_RELEASE_INTERVAL = {self.particles_per_interval}
                      -- average number of particles per water column
                      AVERAGE_NPARTICLES_IN_VERTICAL = {self.particles_per_interval}
                      -- method of setting the number of particles released in water column
                      DISTRIBUTION_AMONG_WATER_COLUMNS = 'uniform'
                 ZMIN_NON_DIM = 0.3
                 ZMAX_NON_DIM = 0.7
                 VERT_SPACING = 'uniform'
        """ ]
            self.releases.append(pnt_release)
            
    def set_groups(self):
        # For each of the flow inputs, add up, down, neutral

        for behavior in self.behavior_names:
            for source in self.sources:
                for rel_name in self.release_timing_names:
                    group=[f"""\
                 GROUP = '{source}_{behavior}_{rel_name}'
                 RELEASE_DISTRIBUTION_SET = '{source}'
                 RELEASE_TIMING_SET = '{rel_name}'
                 PARTICLE_TYPE = 'none'
                 BEHAVIOR_SET = '{behavior}'
                 OUTPUT_SET = '60min_output'
                 OUTPUT_FILE_BASE = '{source}_{behavior}_{rel_name}'
                    """]
                    self.groups.append(group)

    def write_hydro(self):
        # try to generate the hydro part
        # assumes that all average files have been converted, and prefixed
        # with ptm_

        if self.model_dirs is None: # with full set of files, easy to chain
            models=self.model.chain_restarts()
        else:
            models=[sun_driver.SuntansModel.load(run_dir)
                    for run_dir in self.model_dirs]
            for mod in models:
                assert mod is not None
        
        lines=[]
        file_count=0
        for mod in models:
            if 0: # Usual way, assuming the regular average output is present
                ptm_avgs=[ os.path.join(os.path.dirname(avg),"ptm_"+os.path.basename(avg))
                           for avg in mod.avg_outputs() ]
            else: # with rsync output only have minimal files
                ptm_avgs=glob.glob( os.path.join(mod.run_dir,'ptm_average*'))
                ptm_avgs.sort()
                
            for ptm_avg in ptm_avgs:
                assert os.path.exists(ptm_avg),ptm_avg
                file_count+=1
                lines+=[" -- INPUT FILE %d --"%file_count,
                        "  HYDRO_FILE_PATH = '%s/'"%os.path.dirname(ptm_avg),
                        "  FILENAME = '%s'"%os.path.basename(ptm_avg),
                        "  GRD_NAME = '%s'"%os.path.basename(ptm_avg),
                        ""]

        with open(os.path.join(self.run_dir,"FISH_PTM_hydrodynamics.inp"),'wt') as fp:
            fp.write(" NUM_FILES = %d\n"%file_count)
            fp.write("\n".join(lines))

    fish_ptm_exe="/home/rusty/src/fish_ptm/PTM/FISH_PTM.exe"
    fish_ptm_clear_ldpath=False
    def execute(self):
        log.info("Running PTM")
        if self.fish_ptm_clear_ldpath and ('LD_LIBRARY_PATH' in os.environ):
            del os.environ['LD_LIBRARY_PATH']
        pwd=os.getcwd()
        try:
            os.chdir(self.run_dir)
            subprocess.run([self.fish_ptm_exe])
        finally:
            os.chdir(pwd)

