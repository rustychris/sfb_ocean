# External driver script
import os
import datetime
import subprocess
import pandas as pd
import shutil
from stompy.model.suntans import sun_driver

periods=list(pd.date_range(start="2017-06-01",end="2018-06-01",freq='MS').to_pydatetime())
periods[0]=datetime.datetime(2017,6,10)

run_dir=None

for a,b in zip(periods[:-1],periods[1:]):
    b=b.replace(hour=12) # get some overlap
    previous_run_dir=run_dir
    run_dir="/shared2/src/sfb_ocean/suntans/runs/merged_018_%s"%(a.strftime('%Y%m'))

    if os.path.exists(run_dir):
        if sun_driver.SuntansModel.run_completed(run_dir):
            print("%s has already run"%run_dir)
            continue
        else:
            print("Stale run in %s will be removed"%run_dir)
            shutil.rmtree(run_dir)
    fmt='%Y-%m-%dT%H:%M'
    cmd=f"python merged_sun.py -d {run_dir}"
    if previous_run_dir is None:
        cmd=cmd+ f" -s {a.strftime(fmt)}"
    else:
        cmd=cmd+ f" -r {previous_run_dir}"
    cmd=cmd+f" -e {b.strftime(fmt)}"
    print("Running: %s"%cmd)

    # will raise exception on child failure.
    proc=subprocess.run(cmd,shell=True,check=True)

    if not sun_driver.SuntansModel.run_completed(run_dir):
        print("That run failed -- will bail out")
        break
    
