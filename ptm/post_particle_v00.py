# Local Variables:
# python-shell-interpreter: "/opt/anaconda3/bin/ipython"
# python-shell-interpreter-args: "--simple-prompt --matplotlib=agg"
# End:
import glob
import os
import numpy as np
import xarray as xr
from stompy import utils
from stompy.model.suntans import sun_driver
from stompy.model.fish_ptm import ptm_tools

##

# hydro model
hydro=sun_driver.SuntansModel.load("/opt2/sfb_ocean/suntans/runs/merge_009-20171201/suntans.dat")

## 

# PTM output for a single source, and w:
# Try out SacRiver, rising 2mm/s
pb=ptm_tools.PtmBin("all_sources/all_source_select_w/SacRiver_up2000_bin.out")

##

nsteps=pb.count_timesteps()

ptm_start,_=pb.read_timestep(0)
ptm_end,_  =pb.read_timestep(nsteps-2)

print(f"Range of PTM output: {ptm_start} -- {ptm_end}")

##

# First, weight the particles by the adjusted inflow and concentration in
# that inflow.

def compile_model_inflow(hydro,inflow_name,t_start=None):
    """
    for a series of models runs ending at hydro,
    """
    inflows=[]

    if t_start is not None:
        t_start=utils.to_dt64(ptm_start)
    else:
        t_start=0 # read everything
        
    for mod in hydro.chain_restarts(count=t_start):
        mod.load_bc_ds()
        bc_ds=mod.bc_ds
        bc_ds=bc_ds.set_coords(['time','z','xv','yv','cellp','edgep','xe','ye',
                                'seg_name','segedgep','segp','point_cell','point_layer'])
        seg_idxs=np.nonzero(bc_ds.seg_name.values==inflow_name)[0]
        if len(seg_idxs):
            seg_idx=seg_idxs[0]
            Q=bc_ds.boundary_Q.isel(Nseg=seg_idx)
            Q.load()
        elif inflow_name.startswith("src"):
            src_i=int(inflow_name[3:])
            Q=bc_ds.point_Q.isel(Npoint=src_i)            
        else:
            raise Exception("Not ready for point sources")
        Q.name='Q' # so that dataset variable is 'Q'
        
        # bc data includes extra points in time for the interpolation.  drop them.
        Qsel=(Q.time.values>=mod.run_start) & (Q.time.values<=mod.run_stop)
        if len(inflows):
            Qsel=Qsel & (Q.time.values>inflows[-1].time.values[-1])
        Q=Q.isel(Nt=Qsel)
        inflows.append( Q )
        bc_ds.close()

    inflow=xr.concat(inflows,dim='Nt').to_dataset()
    inflow['dnum']=('Nt'),utils.to_dnum(inflow.time)
    return inflow

##

def get_particle_mass(pb,inflow,rel_stride=2):
    """
    pb: a PtmBin object.
    inflow: dataset with Q, dnum
    rel_stride: how many ptm outputs go by for each release period.
    returns a {particle id: particle mass}
    """
    ptm_out_dt_s=pb.dt_seconds()
    nsteps=pb.count_timesteps()

    # Associate weights with particle ids
    # At this stage, we just weight based on unit concentration
    # in the flow.
    part_mass={} # particle id => 'mass'
    for step in utils.progress(range(0,nsteps-1,rel_stride)):
        t,parts=pb.read_timestep(step)
        dnum=utils.to_dnum(t)
        new_ids=[ p['id'] for p in parts if p['id'] not in part_mass]
        # kludge - the quarter hour offset here gives more constant
        # particle mass.  It's close enough, so I'm not going to worry
        # about replicating the integration further.
        Qnow=np.interp(dnum+ptm_out_dt_s*0.5/86400,
                       inflow.dnum.values,inflow.Q)
        if len(new_ids)==0:
            if Qnow>0:
                print(f"WARNING: {t.strftime('%Y-%m-%d %H:%M')}: {len(new_ids):6d} new particles, Q={Qnow:9.2f} m3/s")
            continue
        
        mass_per_particle=max(0,Qnow*ptm_out_dt_s / len(new_ids))
        if step%20==0:
            print(f"{t.strftime('%Y-%m-%d %H:%M')}: {len(new_ids):6d} new particles, Q={Qnow:9.2f} m3/s, mass/part {mass_per_particle:9.2f}")
        for i in new_ids:
            part_mass[i]=mass_per_particle

    if len(part_mass)==0:
        # like petaluma, has no flow in this period.
        return np.nan*np.ones(1)
    # And convert to array for faster indexing.
    max_id=np.max(list(part_mass.keys()))
    print("max_id: ",max_id)
    # leave unset values nan to detect issues later.
    part_mass_a=np.nan*np.zeros(max_id+1,np.float64)
    for k in part_mass:
        part_mass_a[k]=part_mass[k]
    return part_mass_a

##

# That step is reasonably fast -- maybe a month in 10 seconds.
# I think I want to process all of a certain w_s at once.

ptm_run_dirs=["all_sources/all_source_select_w"]

# As a starting point,
#   - process a single vertical velocity
w_part=2000 # as a velocity, so positive up.
wsname="up2000"
#   - all sources
bin_files=[ fn 
            for ptm_run_dir in ptm_run_dirs
            for fn in glob.glob(os.path.join(ptm_run_dir,f'*{wsname}_bin.out')) ]
# for each bin file, get the name of the source
src_names=[ os.path.basename(fn).split('_'+wsname)[0]
            for fn in bin_files ]
# constant in time concentrations:
def source_concentration(name,w_part):
    potws=['san_jose', 'cccsd', 'petaluma', 'ddsd', 'sonoma_valley',
           'palo_alto', 'lg', 'sunnyvale', 'fs', 'src000', 'src002', 'src001']
    rivers=[ 'UALAMEDA', 'COYOTE', 'NAPA', 'SCLARAVCc']
    delta=['SJRiver', 'SacRiver', ]

    # THESE NUMBERS ARE FICTIONAL
    if name in potws:
        if w_part<0: # sinkers
            return 1.0 # particles/m3
        else:
            return 10.0 # particles/m3
    elif name in rivers:
        if w_part<0: # sinkers
            return 100.0
        else:
            return 20.0 # passive, floater
    elif name in delta:
        return 1.0 # uniform, low.

    assert False


src_concs=[source_concentration(src_name,w_part)
           for src_name in src_names]
# Load PTM output
pbs=[ptm_tools.PtmBin(fn) for fn in bin_files]

# And the correpsonding inflows:
inflows=[compile_model_inflow(hydro,src_name,ptm_start)
         for src_name in src_names]

##
# profiling
six.moves.reload_module(ptm_tools)
idx=0
# Thats still pretty slow.
pb=ptm_tools.PtmBin(bin_files[idx])
#get_particle_mass(pb,inflows[idx])

# for pb,inflow in zip(pbs,inflows)
##
# Get particle->mass mapping for each run
src_particle_mass=[ get_particle_mass(pb,inflow)
                    for pb,inflow in zip(pbs,inflows) ]

##

# Now we can step through the PTM timesteps of these files and make some maps.
# For starters, just sum up particles regardless of elevation.
# that will have to get adjusted to separately consider sampling at the surface
# or bed.

grid=hydro.grid

step=3000

date0=None
cell_mass=np.zeros(grid.Ncells(),np.float64)

for pb,mass in zip(pbs,src_particle_mass):
    print(pb.fn)
    d,parts=pb.read_timestep(step)
    if date0 is None:
        date0=d
    else:
        assert d==date0,"PTM outputs are not aligned in time"

    part_weights=mass[ parts['id'] ]
    for part in parts:
        c=grid.select_cells_nearest(part['x'][:2],inside=True)
        cell_mass[c] += mass[part['id']]

