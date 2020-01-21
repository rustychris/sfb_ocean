# External driver script
import os, sys
import numpy as np
import logging as log
log.basicConfig(level=log.INFO)
import glob
import datetime
import subprocess
import pandas as pd
import shutil
from stompy.model.suntans import sun_driver
from stompy import utils

# 018: set of runs from EC2
# 019: same config, locally run.
# 020: time-varying data-based evaporation, and full set of freshwater inflows.
# 021: fix delta inflow to include Threemile Slough, Dutch Slough
prefix="/opt2/sfb_ocean/suntans/runs/merged_021_"

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
    
while run_start < series_end:
    # truncate run_start to integer days 
    run_day_start=utils.floor_dt64(run_start,dt=np.timedelta64(86400,'s'))
    run_stop=run_day_start+interval

    a=utils.to_datetime(run_start)
    b=utils.to_datetime(run_stop)
    run_dir=prefix+a.strftime('%Y%m%d')

    if os.path.exists(run_dir):
        if sun_driver.SuntansModel.run_completed(run_dir):
            log.error("%s has already run, but we were about to clobber it"%run_dir)
            sys.exit(1)
        else:
            log.warning("Stale run in %s will be removed"%run_dir)
            shutil.rmtree(run_dir)
    fmt='%Y-%m-%dT%H:%M'
    cmd=f"python merged_sun.py -d {run_dir}"
    if previous_run_dir is None:
        cmd=cmd+ f" -s {a.strftime(fmt)}"
    else:
        cmd=cmd+ f" -r {previous_run_dir}"
    cmd=cmd+f" -e {b.strftime(fmt)}"
    log.info("Running: %s"%cmd)

    # will raise exception on child failure.
    proc=subprocess.run(cmd,shell=True,check=True)

    if not sun_driver.SuntansModel.run_completed(run_dir):
        log.error("That run failed -- will bail out")
        break

    # my stop is your start
    run_start=run_stop
    previous_run_dir=run_dir
