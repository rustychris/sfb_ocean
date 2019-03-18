import numpy as np
import xarray as xr

import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv

def add_delta_bcs(model,cache_dir):
    # Delta inflow
    # SacRiver, SJRiver
    sac_bc=dfm.NwisFlowBC(name='SacRiver',station=11455420,cache_dir=cache_dir,
                          filters=[dfm.Lowpass(cutoff_hours=3)])
    sj_bc =dfm.NwisFlowBC(name='SJRiver',station=11337190,cache_dir=cache_dir,
                          filters=[dfm.Lowpass(cutoff_hours=3)])

    sac_salt_bc=drv.ScalarBC(name='SacRiver',scalar='salinity',value=0.0)
    sj_salt_bc =drv.ScalarBC(name='SJRiver',scalar='salinity',value=0.0)
    sac_temp_bc=drv.ScalarBC(name='SacRiver',scalar='temperature',value=20.0)
    sj_temp_bc =drv.ScalarBC(name='SJRiver',scalar='temperature',value=20.0)

    model.add_bcs([sac_bc,sj_bc,sac_salt_bc,sj_salt_bc,sac_temp_bc,sj_temp_bc])

def add_usgs_stream_bcs(model,cache_dir):
    # USGS gauged creeks
    for station,name in [ (11172175, "COYOTE"),
                          (11169025, "SCLARAVCc"), # Alviso Sl / Guad river
                          (11180700,"UALAMEDA"), # Alameda flood control
                          (11458000,"NAPA") ]:
        Q_bc=dfm.NwisFlowBC(name=name,station=station,cache_dir=cache_dir)
        salt_bc=drv.ScalarBC(name=name,scalar='salinity',value=0.0)
        temp_bc=drv.ScalarBC(name=name,scalar='temperature',value=20.0)

        model.add_bcs([Q_bc,salt_bc,temp_bc])

def add_potw_bcs(model,cache_dir):
    # WWTP discharging into sloughs
    potw_dir="../sfbay_potw"
    potw_ds=xr.open_dataset( os.path.join(potw_dir,"outputs","sfbay_delta_potw.nc"))

    # the gazetteer uses the same names for potws as the source data
    # omits some of the smaller sources, and this does not include any
    # benthic discharges
    for potw_name in ['sunnyvale','san_jose','palo_alto',
                      'lg','sonoma_valley','petaluma','cccsd','fs','ddsd',
                      'ebda','ebmud','sf_southeast']:
        # This has variously worked and not worked with strings vs bytes.
        # Brute force and try both.
        try:
            Q_da=potw_ds.flow.sel(site=potw_name)
        except KeyError:
            Q_da=potw_ds.flow.sel(site=potw_name.encode())

        # Have to seek back in time to find a year that has data for the
        # whole run
        offset=np.timedelta64(0,'D')
        while model.run_stop > Q_da.time.values[-1]+offset:
            offset+=np.timedelta64(365,'D')
        if offset:
            print("Offset for POTW %s is %s"%(potw_name,offset))

        # use the geometry to decide whether this is a flow BC or a point source
        hits=model.match_gazetteer(name=potw_name)
        if hits[0]['geom'].type=='LineString':
            print("%s: flow bc"%potw_name)
            Q_bc=drv.FlowBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)])
        else:
            print("%s: source bc"%potw_name)
            Q_bc=drv.SourceSinkBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)])

        salt_bc=drv.ScalarBC(parent=Q_bc,scalar='salinity',value=0.0)
        temp_bc=drv.ScalarBC(parent=Q_bc,scalar='temperature',value=20.0)
        model.add_bcs([Q_bc,salt_bc,temp_bc])
        
