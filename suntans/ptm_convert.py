"""
Given a list of run directories, convert average output for
ptm use in directories where this does not exist and where
the run is completed.
"""
import os
import sys

import logging as log
log.basicConfig(level=log.INFO)

import xarray as xr
from stompy.model.suntans import sun_driver
from soda.dataio.ugrid import suntans2untrim

##
if __name__=="__main__":
    # Inputs
    import argparse

    parser=argparse.ArgumentParser(description='Batch convert SUNTANS output to UnTRIM/ugrid-ish.')

    parser.add_argument("run_dirs",help="One or more run directories",nargs='+')
    parser.add_argument("-v", "--verbose",help="Increase verbosity",default=1,action='count')

    args=parser.parse_args()

    for run_dir in args.run_dirs:
        log.info(f"Checking {run_dir}")
        if not sun_driver.SuntansModel.run_completed(run_dir):
            log.info("  not complete -- skip")
            continue
        model=sun_driver.SuntansModel.load(run_dir)

        for avg in model.avg_outputs():
            log.info(f"  checking {avg}")
            outfile=os.path.join( os.path.dirname(avg),
                                  "ptm_"+os.path.basename(avg))
            assert avg!=outfile,"Yikes - that is bad"
            if os.path.exists(outfile):
                log.info(f"  output {outfile} exists -- move on")
                continue
            log.info(f"  processing {avg} => {outfile}")
            suntans2untrim.suntans2untrim(avg,outfile, None, None)


