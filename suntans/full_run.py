# External driver script
import os, sys
import numpy as np
import logging as log
log.basicConfig(level=log.INFO)
import glob
import tempfile
import datetime
import subprocess
import pandas as pd
import shutil
from stompy.model.suntans import sun_driver
from stompy import utils
import local_config

# 018: set of runs from EC2
# 019: same config, locally run.
# 020: time-varying data-based evaporation, and full set of freshwater inflows.
# 021: fix delta inflow to include Threemile Slough, Dutch Slough
# 022: Updated suntans code, running on Farm.
prefix="runs/merged_022_"

series_start=np.datetime64('2017-06-10')
series_end=np.datetime64('2018-07-01')
# 10 days and a 12h overlap
interval=np.timedelta64(10*24 + 12, 'h')

existing_runs=glob.glob(prefix+"*")
existing_runs.sort()

# NB: if an existing run is found it will be used regardless
# of whether it falls within or before the series start/end.
# ideally this would ignore runs that are too early (end
# "much" before series_start), or that come from a chain that
# starts too late.
# also assumes that the directories that match prefix will sort
# chronologically

previous_run_dir=None
for run_dir in existing_runs[::-1]:
    log.info(f"Checking runs {run_dir} for completion")
    if sun_driver.SuntansModel.run_completed(run_dir):
        previous_run_dir=run_dir
        log.info(f"{run_dir} is complete and will be the starting point")
        break

if previous_run_dir is not None:
    previous_model=sun_driver.SuntansModel.load(previous_run_dir)
    run_start=previous_model.restartable_time()
else:
    run_start=series_start

# Where MPI scripts go
def script_dir(): # make sure script destination exists, and return it
    s=prefix+"scripts"
    if not os.path.exists(s):
        os.makedirs(s)
    return s

# These only affect the first time through the loop.
# after that.
dryrun=False # invoke dry run.  For the moment the dry run portion finished, so leave it there...
wetrun=True # invoke wet

while run_start < series_end:
    # truncate run_start to integer days 
    run_day_start=utils.floor_dt64(run_start,dt=np.timedelta64(86400,'s'))
    run_stop=run_day_start+interval

    a=utils.to_datetime(run_start)
    b=utils.to_datetime(run_stop)
    run_dir=prefix+a.strftime('%Y%m%d')

    phases=[]
    if dryrun:
        phases.append('dry')
    if wetrun:
        phases.append('wet')
    if not (dryrun or wetrun):
        # unless something specific with wet/dry phases was requested, do a normal
        # combined run (setup and execute)
        phases.append('drywet')
    
    fmt='%Y-%m-%dT%H:%M'    

    for phase in phases:
        cmd=f"python merged_sun.py -d {run_dir}"
        
        if phase=='dry':
            cmd=cmd+" -n"

        if phase!='wet': # Define the run parameters
            if previous_run_dir is None:
                cmd=cmd+ f" -s {a.strftime(fmt)}"
            else:
                cmd=cmd+ f" -r {previous_run_dir}"
            cmd=cmd+f" -e {b.strftime(fmt)}"

            # And make sure that we remove an existing stale run and avoid
            # clobbering existing completed run.
            if os.path.exists(run_dir):
                if sun_driver.SuntansModel.run_completed(run_dir):
                    log.error("%s has already run, but we were about to clobber it"%run_dir)
                    sys.exit(1)
                else:
                    log.warning("Stale run in %s will be removed"%run_dir)
                    shutil.rmtree(run_dir)
        else:
            assert os.path.exists(run_dir)

        if phase!='dry': # Check MPI status
            assert local_config.slurm_jobid() is not None

            n_tasks_global=local_config.slurm_ntasks_global()
            if n_tasks_global!=local_config.num_procs:
                print("In SLURM task, but ntasks(%d) != local_config num_procs(%d)"%( n_tasks_global,
                                                                                      local_config.num_procs),
                      flush=True)
                raise Exception("Mismatch in number of processes")
        if phase=='wet':
            cmd=cmd+" -w"
            
        log.info(f"Running {phase} phase: {cmd}")
        proc=subprocess.run(cmd,shell=True,check=True)
        
    if not sun_driver.SuntansModel.run_completed(run_dir):
        log.error("That run failed -- will bail out")
        break

    # my stop is your start
    run_start=run_stop
    previous_run_dir=run_dir

    # subsequent times through the loop should do both phases
    dryrun=True
    wetrun=True 
