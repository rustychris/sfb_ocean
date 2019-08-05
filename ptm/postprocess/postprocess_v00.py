"""
Postprocess PTM output with constant particles/release, normalize particle weight
by flow.

The goal is to generate concentration fields on the computational grid reflecting
each source x settling velocity combination.

"""
import xarray as xr
from stompy import utils, memoize
from stompy.model.fish_ptm import ptm_tools
from stompy.grid import unstructured_grid

import os
import logging as log
log.basicConfig(level=log.INFO)

import re
import matplotlib.pyplot as plt

##

# Extract the relevant flow data from the BC files.
class PtmRun(object):
    run_dir=None
    def __init__(self,**kw):
        utils.set_keywords(self,kw)

    def hydrodynamics_inp(self):
        return self.parse_sql(os.path.join(self.run_dir,'FISH_PTM_hydrodynamics.inp'))

    @memoize.imemoize()
    def hydro_models(self):
        hydro=self.hydrodynamics_inp()
        all_paths=[tok[1] for tok in hydro if tok[0]=='HYDRO_FILE_PATH']
        paths=[]
        for p in all_paths:
            if paths and paths[-1]==p: continue
            paths.append(p)
        # Actually it's useful to load the model files to get the true model duration
        # as the BC files have extra fake data.

        return [sun_driver.SuntansModel.load(p)
                for p in paths]
        #return [ os.path.join(p,'Estuary_BC.nc') for p in paths]

    def bc_ds(self):
        """ Extract the relevant parts of the BC data, return as a single dataset
        """
        compiled_fn=os.path.join(self.run_dir,'bc_extracted_v3.nc')
        
        if not os.path.exists(compiled_fn):
            dss=[]

            for model in self.hydro_models():
                # only care about point sources, and river inflows (i.e. ignore
                # ocean flux BCs, and any freesurface BCs
                model.load_bc_ds()
                ds=model.bc_ds.copy()
                ti_start,ti_stop=np.searchsorted(ds.time.values,[model.run_start,model.run_stop])
                ds=ds.isel(Nt=slice(ti_start,ti_stop+1))

                for extra in ['T','S','h','boundary_h','boundary_w','boundary_T',
                              'boundary_u','boundary_v','z',
                              'point_S','point_T','cellp','xv','yv','uc','vc','wc']:
                    if extra in ds: del ds[extra]

                type2_sel= ds['boundary_S'].isel(Nk=0,Nt=0)==0.0
                ds=ds.isel(Ntype2=type2_sel)
                del ds['boundary_S']
                dss.append(ds)

            trim_dss=[]
            for ds in dss:
                if not trim_dss:
                    trim_dss.append(ds)
                    continue
                else:
                    t_sel=ds.time.values>trim_dss[-1].time.values[-1]
                    if t_sel.sum():
                        trim_dss.append(ds.isel(Nt=t_sel))
                    else:
                        log.warning("BC dataset had no useful times?")

            ds=xr.concat(trim_dss,dim='Nt',data_vars='different')
            # somehow there is some 1e-9 difference between xe 
            for v in ['xe','ye']:
                if 'Nt' not in ds[v].dims: continue
                # make sure it's not a terrible issue
                assert ds[v].std(dim='Nt').max()<1.0
                ds[v]=ds[v].isel(Nt=0)
                
            ds.to_netcdf(compiled_fn)
            for model in self.hydro_models():
                model.bc_ds.close()
                model.bc_ds=None

        ds=xr.open_dataset(compiled_fn)
        return ds

    def parse_sql(self,fn):
        """
        semi-parse a text file that has -- comments, 
        key = value lines
        and other lines are returned as is (stripping whitespace), in
        a list of length 1
        """
        with open(fn, 'rt') as fp:
            def tok():
                for line in fp:
                    line=line.split('--')[0] # drop comments
                    if '=' not in line:
                        yield [line.strip()]
                    else:
                        k,v=line.split('=')
                        k=k.strip()
                        v=v.strip()
                        if v[0]==v[-1]=="'":
                            v=v[1:-1]
                        yield k,v
            tokens=list(tok())
        return tokens
    def open_binfile(self,group):
        return ptm_tools.PtmBin(os.path.join(self.run_dir,group+'_bin.out'))

    def groups(self):
        """
        list of all the group with bin output for this run
        """
        all_bins=glob.glob(os.path.join(self.run_dir,'*_bin.out'))
        return [os.path.basename(b).replace('_bin.out','') for b in all_bins]

    def group_to_src_behavior(self,group):
        m=re.match('(.*)_([^_]*)',group)
        return m.group(1),m.group(2)
    
    def get_Qdata_for_group(self,group):
        """
        group: name of PTM group
        returns a time series DataArray for the
        respective source's flow rate.
        """
        src_name,behavior_name=self.group_to_src_behavior(group)

        bc_ds=self.bc_ds()
        
        # get a flow time series for this specific group
        try:
            seg_i=list(bc_ds.seg_name.values).index(seg_name)
            Q_time_series=bc_ds.set_coords('time')['boundary_Q'].isel(Nseg=seg_i)
        except ValueError:
            # point sources are labeled as srcNNN
            pnt_i=int(src_name.replace('src',''))
            Q_time_series=bc_ds.set_coords('time')['point_Q'].isel(Npoint=pnt_i)
    def get_Qfunc_for_group(self,group):
        """
        thin wrapper to handle time interpolation, 
        """
        Q_time_series=self.get_Qdata_for_group(group)
        
        def Q_for_t(t,Q_time_series=Q_time_series):
            ti=np.searchsorted(Q_time_series.time.values,t)
            return Q_time_series.values[ti]
        return Q_for_t
    
    
##

# Ultimately the interface is probably something along the lines of
ptm_runs=[ PtmRun(run_dir="../all_sources/all_source_select_w_const") ]

# May not be the right grid -- would be better to copy in a grid from
# one of the original ptm hydro paths
grid=unstructured_grid.UnstructuredGrid.from_ugrid("../../suntans/grid-merged/spliced_grids_01_bathy.nc")

##

# information that will be filled in by scan_group()
base_ret_dtype=[ ('x',np.float64,3), # location
                 ('group',object), # string name of group,
                 ('source',object),  # string name of source
                 ('behavior',object), # string name of behavior
                 ('part_id',np.int32), # id from fish-ptm within group
                 ('rel_time','M8[s]'), # time of release
                 ('obs_time','M8[s]'), # time of observation
                 # when this particle was released, how many others
                 # were released within the same group, per hour.
                 ('grp_rel_per_hour',np.float64)]

def scan_group(self,group,time_range,z_range=None,grid=None,
               max_age=np.timedelta64(30,'D'),spinup=None,
               extra_fields=[]):
    if spinup is None: spinup=max_age

    ret_dtype=base_ret_dtype+extra_fields
    
    src_name,behavior_name=self.group_to_src_behavior(group)

    bf=self.open_binfile(group)

    nsteps=bf.count_timesteps()
    t0,_=bf.read_timestep(0)
    t0=utils.to_dt64(t0)

    # Array mapping particle index to a mass.
    # here mass is for a unit concentration

    # data to store for each particle -- this is an intermediate
    # data structure, not what gets returned.  it is indexed by
    # particle id, while the return value is indexed by
    # particle-sample, potentially including each particle multiple times.

    calc_dtype=[ ('rel_time','M8[s]'),
                 ('grp_rel_per_hour',np.float64)]
    
    # mass=np.nan*np.ones(1000) # will be expanded as needed
    particles=np.zeros(1000,calc_dtype)

    # could calculate the number of particles released in an interval,
    # or just know that it's 5.
    count_per_release=5
    release_interval_s=3600 # and how often are particles released

    # accumulate per-time step observations, to be concatenated at the
    # end and returned
    ret_particles=[]
    
    for ti in range(nsteps):
        t,parts=bf.read_timestep(ti)
        t=utils.to_dt64(t)
        if t>=time_range[1]:
            log.info("Read beyond the time range. Done with this group")
            break
        
        max_part_id=parts['id'].max()
        while max_part_id+1>len(particles):
            # double the size
            new=np.zeros(len(particles),calc_dtype)
            new['grp_rel_per_hour']=np.nan # mark uninitialized
            particles=np.concatenate([particles,new])
            log.info(f"Doubled size of mass lookup to {len(particles)}")

        # any particles with nan grp_rel_per_hour are assumed new
        # missing now has indices into parts['id']
        missing=np.nonzero(np.isnan(particles['grp_rel_per_hour'][parts['id']]))[0]
        # 1g/m3 * m3/s / (particles/release) * (seconds/release)
        # 1g/particle
        # mass[parts['id'][missing]]=Q / count_per_release * release_interval_s

        new_ids=parts['id'][missing] # index into particles for new particles this step
        # this could be inferred.  gets trickier with SJ, Sac and DDSD. FIX.
        particles['grp_rel_per_hour'][new_ids]=5.0 
        particles['rel_time'][new_ids]=t
        
        if (t-t0>=spinup) and (t>=time_range[0]):
            # at this point only filter on age.
            age=(t-particles['rel_time'][parts['id']])
            sel=age < max_age
            # this is where we could further filter on where the particles
            # are, further narrowing the sel array
            assert z_range is None,"Have not implemented z-range yet"
            
            ret=np.zeros(len(parts[sel]),ret_dtype)
            ret['x']=parts['x'][sel]
            ret['group']=group
            ret['source']=src_name
            ret['behavior']=behavior_name
            ret['part_id']=parts['id'][sel]
            ret['rel_time']=particles['rel_time'][parts['id'][sel]]
            ret['obs_time']=t
            ret['grp_rel_per_hour']=particles['grp_rel_per_hour'][parts['id'][sel]]
            ret_particles.append(ret)
        else:
            # already done the bookkeeping, but too early to actually output
            # particles
            pass

    return np.concatenate(ret_particles)
               
def query_runs(ptm_runs,group_patt,time_range,z_range=None,grid=None,
               max_age=np.timedelta64(30,'D'),
               spinup=None):
    """
    ptm_runs: List of PtmRun instancse
    groups: regular expression matching the group names of interest
    time_range: pair of datetime64s defining time period to include
    z_range: come back to this, but it will be a way to filter on 
     vertical position.
    max_age: ignore particles older than this
    spinup: don't return particles within this interval of the
     start of the run, defaults to max_age.
    """
    if spinup is None:
        spinup=max_age

    # compile an array of particles that are
    #  (i) observed within the time_range, limited to output steps
    #      at least spinup into that specific run
    #  (ii) not older than max_age
    # and report each particle's
    #  (a) mass
    #  (b) age

    # tracers are assumed to be specific to a behavior, but
    # aggregated across runs and sources.
    # mass is normalized to unit concentration from each source
    
    # note that to normalize mass, we also need to know how many
    # particles were released in this group, including particles not
    # observed at this time.
    
    # fist step is to scan each matching group of each run, and
    # populate everything but mass
    all_part_obs=[]
    
    for run_idx,run in enumerate(ptm_runs):
        for group in run.groups():
            if re.match(group_patt,group) is None:
                continue
            part_obs=scan_group(run,group,time_range=time_range,z_range=z_range,
                                grid=grid, extra_fields=[('run_idx',np.int32),
                                                         ('mass',np.float64)])
            part_obs['run_idx']=run_idx
            all_part_obs.append(part_obs)
    return np.concatenate(all_part_obs)
        
part_obs=query_runs(ptm_runs,
                    group_patt='.*_up2000',
                    time_range=[np.datetime64("2017-07-20 00:00"),
                                np.datetime64("2017-07-20 01:00")],
                    z_range=None, # not ready
                    grid=grid)

##
# that returned 4.8M points.  with output for a single time step
# via time_range, that becomes 41k.
# 18 groups.
# 30 days in, 5 particles/hour/group
# 30 days*24 hr/day * 5 particles/grp/hr *18 grps
# but time range includes 5 days,


## 
conc=particle_to_density(particles,grid)

grid.plot_cells(values=conc)


## 

plt.figure(1).clf()
# plt.plot(part_obs['x'][:,0],part_obs['x'][:,1],'g.',ms=2)
age_secs=(part_obs['obs_time']-part_obs['rel_time'])/np.timedelta64(1,'s')
scat=plt.scatter(part_obs['x'][:,0],part_obs['x'][:,1],10,age_secs/86400.,
                 cmap='jet')
grid.plot_edges(lw=0.5,color='k',zorder=-1)
plt.axis('equal')
plt.colorbar(scat,label='Age (days)')

##
