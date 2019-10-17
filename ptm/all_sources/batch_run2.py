"""
Execute batches of PTM runs
batch_run2.py: adapt to new runs, and go to multiprocessing
"""
import os
import sys
import glob
import numpy as np

import ptm_setup
from stompy.model.suntans import sun_driver
from stompy import utils

# Each run covers one release time and one settling velocity
# these are odds then evens, to get the seasonal picture sooner.
# the most important ones are
# 2017-08-30 -- 2017-09-14, 15 days old.
# 2018-03-02 -- 2018-03-17, 15 days old
# 2017-08-28 -- 2017-09-11, 44 days old
# 2018-02-28 -- 2018-03-14, 44 days old.
# the relevant releases are then
#  2017-08-15
#  2018-02-15
#  2017-07-15
#  2018-01-15

rel_times=[ #np.datetime64("2017-06-15"),
            np.datetime64("2017-07-15"),
            #np.datetime64("2017-09-15"),
            # np.datetime64("2017-11-15"),
            np.datetime64("2018-01-15"),
            # np.datetime64("2018-03-15"),
            #np.datetime64("2017-10-15"),
            #np.datetime64("2018-05-15"),
            np.datetime64("2017-08-15"),
            #np.datetime64("2018-04-15"),
            #np.datetime64("2017-12-15"),
            np.datetime64("2018-02-15"),
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
    # 'petaluma', 
    'cccsd', 
    'fs', 
    #    'ddsd',
    'src000', # EBDA
    'src001', # EBMUD
    'src002'  # SFPUC
]


sources=[
    "SacRiver",
    "SJRiver",
    "Coyote_Creek_",
    "Guadalupe_Riv",
    "San_Francisqu",
    "Redwood_Creek",
    "Steinberger_S",
    "Foster_City",
    "Seal_Creek",
    "San_Mateo_Cre",
    "Coyote_Point",
    "Easton_Creek",
    "Mills_Creek",
    "Highline_Cana",
    "San_Bruno_Cre",
    "Colma_Creek",
    "Oyster_Point",
    "Visitacion",
    "Islais_Creek",
    "Black_Point",
    "unnamed02",
    "Coyote_Creek__1ser",
    "Arroyo_Corte_",
    "unnamed00",
    "unnamed01",
    "Corte_Madera_",
    "San_Rafael_Cr",
    "unnamed03",
    "Gallinas_Cree",
    "Miller_Creek",
    "Novato_Creek",
    "Petaluma_Rive",
    "Sonoma_Creek",
    "Napa_River",
    "Southampton_B",
    "Sulphur_Sprin",
    "unnamed08",
    "Suisun_Slough",
    "Montezuma_Slo",
    "Montezuma_Slo_1ser",
    "unnamed07",
    "Hastings_Slou",
    "Pacheco_Creek",
    "unnamed13",
    "Arroyo_del_Ha",
    "unnamed14",
    "unnamed09",
    "unnamed10",
    "unnamed11",
    "unnamed12",
    "Rodeo_Creek",
    "Refugio_Creek",
    "Pinole_Creek",
    "Garrity_Creek",
    "San_Pablo_Cre",
    "Meeker_Slough",
    "Cerrito_Creek",
    "Strawberry_Cr",
    "Temescal_Cree",
    "unnamed05",
    "Glen_Echo_Cre",
    "unnamed06",
    "San_Leandro_C",
    "unnamed04",
    "Estudillo_Can",
    "San_Lorenzo_C",
    "Old_Alameda_C",
    "Alameda_Creek",
    "Matadero_and_",
    "Permanente_Cr",
    "Stevens_Creek",
    "Guadalupe_Slo",
    "Rheem_Creek",
    "Castro_Creek",
    "sunnyvale",
    "san_jose",
    "palo_alto",
    "lg",
    "sonoma_valley",
    "petaluma",
    "cccsd",
    "fs",
    "ddsd",
    # and the potw point source discharges
    'src000', # EBDA
    'src001', # EBMUD
    'src002'  # SFPUC
    ]


def make_call(args):
    # need to adapt Config to allow model_dir
    cfg=ptm_setup.Config(**args)

    cfg.set_releases()
    cfg.set_groups()
    run_dir=args['run_dir']
    if os.path.exists(run_dir):
        # This will probably need to get a better test, for when a run
        # failed.
        print(f"Second check - directory exists {run_dir}, will skip")
        return

    cfg.clean()
    cfg.write()
    cfg.write_hydro()
    cfg.execute()


if __name__=='__main__':
    #ptm_avgs=glob.glob("/opt2/sfb_ocean/suntans/runs/merged_018_*/ptm_average.nc_0000.nc")
    # 2019-10-12: switch to new runs with more freshwater inputs.
    ptm_avgs=glob.glob("/opt2/sfb_ocean/suntans/runs/merged_020_*/ptm_average.nc_0000.nc")
    ptm_avgs.sort()

    model_dirs=[os.path.dirname(fn) for fn in ptm_avgs]

    model=sun_driver.SuntansModel.load(model_dirs[-1])
    model.load_bc_ds()

    models_end_time=model.run_stop

    models_start_time=sun_driver.SuntansModel.load(model_dirs[0]).run_start

    print(f"{len(model_dirs)} model directories, spanning {models_start_time} to {models_end_time}")

    # list of dicts for individual PTM calls.
    calls=[]
    
    for rel_time in rel_times:
        end_time=rel_time + np.timedelta64(60,'D')
        if rel_time < models_start_time:
            raise Exception("Set of model data starts after release time")
        if end_time > models_end_time:
            print("Set of model data ends before end_time")
            continue # don't exit, in case the rel_times are not chronological

        rel_str=utils.to_datetime(rel_time).strftime('%Y%m%d')

        for rising_speed in rising_speeds:
            # 020 suffix to clarify hydro
            run_dir=f"/opt2/sfb_ocean/ptm/all_source_020/{rel_str}/w{rising_speed}"
            if os.path.exists(run_dir):
                # This will probably need to get a better test, for when a run
                # failed.
                print(f"Directory exists {run_dir}, will skip")
                continue

            call=dict(model_dir=model_dirs[-1],
                      rel_time=rel_time,
                      end_time=end_time,
                      run_dir=run_dir,
                      model=model,
                      rising_speeds_mps=[rising_speed],
                      sources=sources)
            calls.append(call)

    print(f"{len(calls)} total PTM calls queued")

    # each run is pretty chunky - try just one run at a time
    for kwargs in calls:
        make_call(kwargs)
