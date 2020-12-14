"""
Execute batches of PTM runs
batch_run3.py: run all of each source in one go,
  and divide evenly and completely.
batch_run4.py: Re-run, and limit the surface/bed distance even more, to 0.095m
to match the manta trawl calculations.
"""

import os
import time
import sys
import glob
import numpy as np

import ptm_setup5 as ptm_setup
from stompy.model.suntans import sun_driver
from stompy.model.fish_ptm import ptm_config
from stompy import utils

# goal is to have 10-day analysis windows,
# and each window accounts for particles up to 60 days old.
# the first manta data to compare to 2017-08-21.
# rel_times=np.arange(
#     np.datetime64("2017-06-20"),
#     np.datetime64("2018-04-20"),
#     np.timedelta64(10*86400,'s'))

# Be explicit so we can avoid any extra work.
rel_times=np.array([
    # These are needed to get to 20 days
    np.datetime64('2017-07-30','s'),
    np.datetime64('2017-08-09','s'),
    np.datetime64('2017-08-19','s'),
    np.datetime64('2017-08-29','s'),
    np.datetime64('2017-09-08','s'),
    np.datetime64('2017-10-08','s'),
    np.datetime64('2017-10-18','s'),
    np.datetime64('2017-10-28','s'),
    np.datetime64('2017-12-17','s'),
    np.datetime64('2017-12-27','s'),
    np.datetime64('2018-02-05','s'),
    np.datetime64('2018-02-15','s'),
    np.datetime64('2018-02-25','s'),
    np.datetime64('2018-03-07','s'),
    np.datetime64('2018-03-17','s'),

    # These are needed to get to 40 days
    np.datetime64('2017-07-10','s'),
    np.datetime64('2017-07-20','s'),
    np.datetime64('2017-09-18','s'),
    np.datetime64('2017-09-28','s'),
    np.datetime64('2017-11-27','s'),
    np.datetime64('2017-12-07','s'),
    np.datetime64('2018-01-16','s'),
    np.datetime64('2018-01-26','s'),

    # These are needed to get to 60 days
    np.datetime64('2017-06-20','s'), 
    np.datetime64('2017-06-30','s'),
    np.datetime64('2017-11-07','s'),
    np.datetime64('2017-11-17','s'),
    np.datetime64('2018-01-06','s'),

    # These are never needed for manta comparisons
    np.datetime64('2018-03-27','s'),
    np.datetime64('2018-04-06','s'),
    np.datetime64('2018-04-16','s')
])


rising_speeds=[0.0005,-0.0005,0.005,-0.005,0.05,-0.05,0.0]

# prioritize the sources
sources=[
    # Stormwater sources by mean flow in March, 2018:
    'Napa_River',        # : 4.044
    'Guadalupe_Riv',     # : 2.242
    'Alameda_Creek',     # : 0.103 # That number seems off
    'Coyote_Creek_',     # : 1.317
    'Pacheco_Creek',     # : 1.949
    'Sonoma_Creek',      # : 1.857
    'Guadalupe_Slo',     # : 0.719
    'San_Lorenzo_C',     # : 0.658
    'Montezuma_Slo',     # : 0.492
    'Petaluma_Rive',     # : 0.454

    # The WWTPs
    "sunnyvale",
    "san_jose",
    "palo_alto",
    "cccsd",
    "fs",
    # and the potw point source discharges
    'src000', # EBDA
    'src001', # EBMUD
    'src002',  # SFPUC

    # Introduce a few more stormwater sources down to a cutoff of 0.2
    'Suisun_Slough',     # : 1.720
    'Montezuma_Slo_1ser', # : 0.705
    'unnamed08',         # : 0.701
    'Old_Alameda_C',     # : 0.452
    'unnamed07',         # : 0.439
    
    'San_Leandro_C',     # : 0.364
    'San_Pablo_Cre',     # : 0.344
    'San_Francisqu',     # : 0.335
    
    'Steinberger_S',     # : 0.267
    'Stevens_Creek',     # : 0.260
    'Glen_Echo_Cre',     # : 0.244
    'Matadero_and_',     # : 0.224
    'Sulphur_Sprin',     # : 0.215

    # insert the Delta sources in the middle here
    # 'SacRiver',          # : 306.644
    #'SJRiver',           # : -27.774

    # These had been excluded in the past.
    'Arroyo_del_Ha',     # : 0.188
    'Redwood_Creek',     # : 0.180 This was included in the past
    #'Islais_Creek',      # : 0.173
    'Pinole_Creek',      # : 0.172
    'Hastings_Slou',     # : 0.170
    'Meeker_Slough',     # : 0.167
    'Permanente_Cr',     # : 0.152
    'Novato_Creek',      # : 0.147
    'Colma_Creek',       # : 0.144
    'Estudillo_Can',     # : 0.130
    #'sonoma_valley',     # : 0.121
    'Rodeo_Creek',       # : 0.121
    'Strawberry_Cr',     # : 0.119
    #'Black_Point',       # : 0.100
    'Seal_Creek',        # : 0.094
    'Southampton_B',     # : 0.093
    'Corte_Madera_',     # : 0.085
    'Visitacion',        # : 0.084
    'Temescal_Cree',     # : 0.082 This was included in the past
    'Highline_Cana',     # : 0.073
    # close to Temescal    'unnamed05',         # : 0.071
    # close to Estudillo   'unnamed04',         # : 0.066
    # close to Meeker 'Cerrito_Creek',     # : 0.062
    'Coyote_Point',      # : 0.053
    # close to Arroyo_del_Ha 'unnamed13',         # : 0.052
    # close to Pinole_Creek 'Refugio_Creek',     # : 0.051
]

sources_per_chunk=13
source_chunks=[ sources[i:i+sources_per_chunk]
                for i in range(0,len(sources),sources_per_chunk)]

other_sources=[
    # # WWTP that were not sampled, not large.
    # "lg",
    # "sonoma_valley",
    # "petaluma",
    # "ddsd",
    
    'Foster_City',       # : 0.046
    'unnamed14',         # : 0.046
    'unnamed06',         # : 0.045
    'Garrity_Creek',     # : 0.045
    'Miller_Creek',      # : 0.044
    'San_Mateo_Cre',     # : 0.040
    'San_Bruno_Cre',     # : 0.035
    'Castro_Creek',      # : 0.034
    'Gallinas_Cree',     # : 0.029
    'San_Rafael_Cr',     # : 0.028
    'Arroyo_Corte_',     # : 0.024
    'unnamed11',         # : 0.023
    'Oyster_Point',      # : 0.023
    'unnamed12',         # : 0.021
    'unnamed09',         # : 0.020
    'Easton_Creek',      # : 0.016
    'unnamed00',         # : 0.014
    'unnamed02',         # : 0.013
    'Coyote_Creek__1ser', # : 0.012
    'Mills_Creek',       # : 0.011
    'Rheem_Creek',       # : 0.010
    'unnamed01',         # : 0.009
    'unnamed03',         # : 0.007
    'unnamed10',         # : 0.006
]


def make_call(args):
    # need to adapt Config to allow model_dir
    cfg=ptm_setup.Config(**args)

    cfg.set_releases()
    cfg.set_release_timing()
    
    cfg.set_groups()
    run_dir=args['run_dir']
    t=time.time()
    if os.path.exists(run_dir):
        # This will probably need to get a better test, for when a run
        # failed.
        print(f"Second check - directory exists {run_dir}, will skip")
        return
    # I'm getting a surprising number of crashes where it gets past
    # the above check, past clean, and in cfg.write() it somehow fails.
    print(f"Preemptively creating run directory {run_dir}")
    try:
        os.makedirs(run_dir)
    except FileExistsError:
        now=time.time()
        print(f"Very strange. {now-t} seconds ago tested for {run_dir} and it did not exist")
        print(f"Then makedirs failed, and now does it exist? {os.path.exists(run_dir)}")
        print("Bailing on this one, but it's weird.")
        return

    cfg.clean()
    cfg.write()
    cfg.write_hydro()
    # os.environ['OMP_NUM_THREADS']="8" # set by slurm batch script
    cfg.execute()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check existing PTM runs and run remaining.')

    parser.add_argument('-n', "--dry-run", help="Report on what would be done then exit",
                        action='store_true')
    parser.add_argument("-v", "--verbose", help="Increase reporting",
                        action='store_true')
    args=parser.parse_args()

    ptm_avgs=glob.glob("/home/rustyh/src/sfb_ocean/suntans/runs/merged_022_*/ptm_average.nc_0000.nc")
    ptm_avgs.sort()

    model_dirs=[os.path.dirname(fn) for fn in ptm_avgs]

    model=sun_driver.SuntansModel.load(model_dirs[-1])
    model.load_bc_ds()

    models_end_time=model.run_stop

    models_start_time=sun_driver.SuntansModel.load(model_dirs[0]).run_start

    print(f"{len(model_dirs)} model directories, spanning {models_start_time} to {models_end_time}")

    for rel_time in rel_times:
        end_time=rel_time + np.timedelta64(30,'D')
        assert rel_time >= models_start_time,"Set of model data starts after release time"
        assert end_time < models_end_time,"Set of model data ends before end_time"

    # list of dicts for individual PTM calls.
    calls=[]

    incompletes=[]

    # with batch_run5, each run is a chunk of sources
    for chunk_idx,chunk_sources in enumerate(source_chunks):
        chunk_name="chunk%02d"%chunk_idx
        for rel_time in rel_times:
            date_name=utils.to_datetime(rel_time).strftime('%Y%m%d')
            # 021b => v21 hydro, suffix to clarify hydro, and 'b' says we're doing per-source
            # 021c => longer simulations
            # 021d => fixed (hopefully) ptm average output
            # 022a => farm hydro
            run_dir=f"/home/rustyh/src/sfb_ocean/ptm/all_source_022a/{chunk_name}/{date_name}"
            if os.path.exists(run_dir):
                pc=ptm_config.PtmConfig.load(run_dir)
                if not pc.is_complete():
                    incompletes.append(run_dir)
                print(f"Directory exists {run_dir}, will skip")
                continue

            call=dict(model_dir=model_dirs[-1],
                      rel_times=[rel_time],
                      rel_duration=np.timedelta64(10,'D'),
                      end_time=rel_time+np.timedelta64(70,'D'),
                      run_dir=run_dir,
                      model=model,
                      rising_speeds_mps=rising_speeds,
                      particles_per_interval=10,
                      sources=chunk_sources)
            calls.append(call)

    print(f"{len(calls)} total PTM calls queued")
    if args.verbose:
        for kwargs in calls:
            print(f"  {kwargs['run_dir']}")
        print()
            
    if incompletes:
        # this will pick up on ongoing runs, too.
        print("Runs that are present but appear incomplete.  Fix code or remove directory")
        for r in incompletes:
            print(f"  {r}")
        if not args.dry_run:
            print("Waiting 10 seconds")
            time.sleep(10)

    if args.dry_run:
        print("Dry run: exiting")
        sys.exit(0)
        
    # each run is pretty chunky - try just one run at a time
    for kwargs in calls:
        make_call(kwargs)
