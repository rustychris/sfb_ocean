"""
Execute batches of PTM runs
"""
import os
import sys
import glob
import numpy as np

import ptm_setup
from stompy.model.suntans import sun_driver
from stompy import utils

ptm_avgs=glob.glob("/opt2/sfb_ocean/suntans/runs/merged_018_*/ptm_average.nc_0000.nc")
ptm_avgs.sort()

model_dirs=[os.path.dirname(fn) for fn in ptm_avgs]

model=sun_driver.SuntansModel.load(model_dirs[-1])
model.load_bc_ds()

models_end_time=model.run_stop

models_start_time=sun_driver.SuntansModel.load(model_dirs[0]).run_start

print(f"{len(model_dirs)} model directories, spanning {models_start_time} to {models_end_time}")
##

# Each run covers one release time and one settling velocity
# these are odds then evens, to get the seasonal picture sooner.
rel_times=[ np.datetime64("2017-06-15"),
            np.datetime64("2017-07-15"),
            np.datetime64("2017-09-15"),
            np.datetime64("2017-11-15"),
            np.datetime64("2018-01-15"),
            np.datetime64("2018-03-15"),
            np.datetime64("2018-05-15"),
            
            np.datetime64("2017-08-15"),
            np.datetime64("2017-10-15"),
            np.datetime64("2017-12-15"),
            np.datetime64("2018-02-15"),
            np.datetime64("2018-04-15"),
            ]
rising_speeds=[0.0005,-0.0005,0.005,-0.005,0.05,-0.05,0.0]

sources=[
    'SacRiver',
    'SJRiver',
    'COYOTE', 
    'SCLARAVCc', 
    'UALAMEDA', 
    'NAPA', 
    'sunnyvale', 
    'san_jose', 
    'palo_alto', 
    #    'lg', 
    #    'sonoma_valley', 
    'petaluma', 
    'cccsd', 
    'fs', 
    #    'ddsd',
    'src000', # EBDA
    'src001', # EBMUD
    'src002'  # SFPUC
]

for rel_time in rel_times:
    end_time=rel_time + np.timedelta64(60,'D')
    if rel_time < models_start_time:
        raise Exception("Set of model data starts after release time")
    if end_time > models_end_time:
        print("Set of model data ends before end_time")
        continue # don't exit, in case the rel_times are not chronological

    rel_str=utils.to_datetime(rel_time).strftime('%Y%m%d')
    
    for rising_speed in rising_speeds:
        run_dir=f"/opt2/sfb_ocean/ptm/all_source/{rel_str}/w{rising_speed}"
        if os.path.exists(run_dir):
            # This will probably need to get a better test, for when a run
            # failed.
            print(f"Directory exists {run_dir}, will skip")
            continue
        
        cfg=ptm_setup.Config(rel_time=rel_time,
                             end_time=end_time,
                             run_dir=run_dir,
                             model=model,
                             rising_speeds_mps=[rising_speed],
                             sources=sources)

        cfg.set_releases()
        cfg.set_groups()
        cfg.clean()
        cfg.write()
        cfg.write_hydro()
        cfg.execute()

## 
