import os
import glob
from stompy.model.fish_ptm import ptm_tools
from stompy.model.suntans import sun_driver

model=sun_driver.SuntansModel.load("rundata")
model.load_bc_ds()

##
class PtmConfig(object):
    run_dir=None    
    end_time=None
    def __init__(self):
        self.regions=[]
        self.releases=[]
        self.groups=[]
        
    def text(self):
        self.lines=[]
        self.add_global()
        self.add_transects()
        self.add_regions()
        self.add_release_distribution_sets()
        self.add_release_timing()
        self.add_behaviors()
        self.add_output_sets()
        self.add_groups()
        
        return "\n".join(self.lines)
    
    def add_global(self):
        self.lines +=[
        """\
GLOBAL INFORMATION
   END_TIME = '{0.end_time_str}'
   RESTART_DIR = 'none'
   TIME_STEP_SECONDS = 180.
 
   -- deactivation logicals ---
   REACHED_OPEN_BOUNDARY = 'true'
   REACHED_FLOW_BOUNDARY = 'true'
   ENTRAINED_BY_VOLUME_SINK = 'false'
   CROSSED_LINE = 'false'
   DEPOSITED_ON_BED = 'false'
   CONSOLIDATED_ON_BED = 'false'
 
   -- kill logicals ---
   REACHED_OPEN_BOUNDARY = 'true'
   REACHED_FLOW_BOUNDARY = 'false'
   ENTRAINED_BY_VOLUME_SINK = 'false'
   CROSSED_LINE = 'false'
   DEPOSITED_ON_BED = 'false'
   CONSOLIDATED_ON_BED = 'false'
 
   -- line information --- 
   NLINES = 0
        """.format(self)]
    @property
    def end_time_str(self):
        return utils.to_datetime(self.end_time).strftime("%Y-%m-%d %H:%M:%S")
    @property
    def rel_time_str(self):
        return utils.to_datetime(self.rel_time).strftime("%Y-%m-%d %H:%M:%S")

    def add_transects(self):
        self.lines+=["""\
TRANSECT INFORMATION -- applies to tidal surfing
   NTRANSECTS = 0
"""]

    def add_regions(self):
        self.lines.append("REGION INFORMATION")
        self.lines.append( "   NREGIONS = {nregions}".format(nregions=len(self.regions)) )
        for i,r in enumerate(self.regions):
            self.lines.append("   --- region %d ---"%i)
            self.lines += r
    def add_release_distribution_sets(self):
        self.lines.append("""RELEASE DISTRIBUTION INFORMATION
   NRELEASE_DISTRIBUTION_SETS = {num_releases}
""".format(num_releases=len(self.releases)))
    
        for i,rel in enumerate(self.releases):
            self.lines.append("   -- release distribution set %d ---"%i)
            self.lines+=rel

    def add_release_timing(self):
        self.lines+=["""\
RELEASE TIMING INFORMATION
   NRELEASE_TIMING_SETS = 3
   -- release timing set 1 ---        
     RELEASE_TIMING_SET = 'once'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'single'
     INACTIVATION_TIME = 'none'
   -- release timing set 2 ---        
     RELEASE_TIMING_SET = 'flowbased'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'interval'
          NINTERVALS = 1000000
          RELEASE_INTERVAL_HOURS = 1.0
          INACTIVATION_TIME = 'none'
   -- release timing set 3 ---        
     RELEASE_TIMING_SET = 'interval'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'interval'
       NINTERVALS = 100000
       RELEASE_INTERVAL_HOURS = 1.0
     INACTIVATION_TIME = 'none'""".format(rel_time_str=self.rel_time_str)
          ]
    def add_behaviors(self):
        self.lines+=["""\
BEHAVIOR INFORMATION
   NBEHAVIOR_PROFILES = 0
   NBEHAVIORS = 2

   -- behavior 1 ---
     BEHAVIOR_SET = 'down5000'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = 'down_5mm_per_s.inp'
  
   -- behavior 2 ---
     BEHAVIOR_SET = 'up5000'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = 'up_5mm_per_s.inp'

"""]


    def add_output_sets(self):
        self.lines+=["""\
OUTPUT INFORMATION 
   NOUTPUT_SETS = 2

   -- output set 1 ---
   OUTPUT_SET = '6min_output'
   FLAG_LOG_LOGICAL = 'true'
   BINARY_OUTPUT_INTERVAL_HOURS = 0.10
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 0.50
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 1.00
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 0.50
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 0.50
   STATE_OUTPUT_INTERVAL_HOURS = 0.10
   NUMBER_OF_VARIABLES_OUTPUT = 6
     VARIABLE_OUTPUT = 'velocity'
     VARIABLE_OUTPUT = 'salinity'
     VARIABLE_OUTPUT = 'layer'
     VARIABLE_OUTPUT = 'water_depth'
     VARIABLE_OUTPUT = 'water_level'
     VARIABLE_OUTPUT = 'bed_elevation'

     NODATA_VALUE = -999.0

   -- output set 2 ---
   OUTPUT_SET = '15min_output'
   FLAG_LOG_LOGICAL = 'true'
   BINARY_OUTPUT_INTERVAL_HOURS = 0.25
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 0.50
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 1.00
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 0.50
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 0.50
   STATE_OUTPUT_INTERVAL_HOURS = 'none'
"""]


    def add_groups(self):
        self.lines.append("""
PARTICLE GROUP INFORMATION 
   NGROUPS = {num_groups}
""".format(num_groups=len(self.groups)))
        for i,group in enumerate(self.groups):
            self.lines += ["   --- group %d ---"%i]
            self.lines += group

    def clean(self):
        print("Cleaning")
        for patt in ["*.out",
                     "*.log",
                     "*release_log",
                     "*.idx"]:
            for fn in glob.glob(os.path.join(self.run_dir,patt)):
                os.unlink(fn)

    def write(self):
        with open(os.path.join(self.run_dir,"FISH_PTM.inp"),'wt') as fp:
            fp.write(cfg.text())
    
            
cfg=PtmConfig()
cfg.end_time=model.run_stop - np.timedelta64(3600,'s')
cfg.rel_time=model.run_start+np.timedelta64(3600,'s')
cfg.run_dir="ptm_auto"

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
    region['poly']=model.grid.cell_polygon(pnt_cell).buffer(1.0,4)
    region['poly_name']="src%03d"%Npoint
    region['poly_fn']=region['poly_name']+".pol"
    
    pnts=np.array(poly.exterior)

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

# For each of the flow inputs, add up, down, neutral
for seg_idx in range(len(model.bc_ds.Nseg)):
    flow_name=model.bc_ds.seg_name.values[seg_idx]
    for behavior in ['up5000','down5000','none']:
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
    for behavior in ['up5000','down5000','none']:
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
            
print("Running PTM")
if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']
pwd=os.getcwd()
try:
    os.chdir(cfg.run_dir)
    subprocess.run(["/home/rusty/src/fish_ptm/PTM/FISH_PTM.exe"])
finally:
    os.chdir(pwd)


##

# Release at a specified interval is fine, but there isn't an
# obvious way to change the count over time

# these bed releases are all outfalls, with constant-ish flows
# so maybe that's okay.

ptmbin=ptm_tools.PtmBin(os.path.join(ptm_run,"UP5mm_bin.out"))

plt.figure(1).clf()
ax=plt.gca()
model.grid.plot_edges(color='k',ax=ax,alpha=0.25,lw=0.7)
ax.axis('equal')

# 34 timesteps, no particles.

for idx in range(ptmbin.count_timesteps()):
    ax.lines=[]
    dnum,data=ptmbin.read_timestep(idx)
    print(len(data))
    ax.plot(data['x'][:,0],
            data['x'][:,1],
            'g.',ms=2,alpha=0.5)
    plt.draw()
    plt.pause(0.1)

##

# ptm_hydro=xr.open_dataset('rundata/ptm_hydro_0000.nc')
