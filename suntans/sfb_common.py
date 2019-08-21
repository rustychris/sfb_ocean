import numpy as np
import xarray as xr
import os
import logging as log

from stompy.io.local import usgs_nwis
import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as drv


def add_delta_bcs(model,cache_dir):
    # Delta inflow
    # SacRiver, SJRiver
    sac_bc=dfm.NwisFlowBC(name='SacRiver',station=11455420,cache_dir=cache_dir,
                          filters=[dfm.Lowpass(cutoff_hours=3)],
                          dredge_depth=model.dredge_depth)
    sj_bc =dfm.NwisFlowBC(name='SJRiver',station=11337190,cache_dir=cache_dir,
                          filters=[dfm.Lowpass(cutoff_hours=3)],
                          dredge_depth=model.dredge_depth)

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
        Q_bc=dfm.NwisFlowBC(name=name,station=station,cache_dir=cache_dir,
                            dredge_depth=model.dredge_depth)
        salt_bc=drv.ScalarBC(name=name,scalar='salinity',value=0.0)
        temp_bc=drv.ScalarBC(name=name,scalar='temperature',value=20.0)

        model.add_bcs([Q_bc,salt_bc,temp_bc])

def add_potw_bcs(model,cache_dir,temperature=20.0):
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
            log.info("Offset for POTW %s is %s"%(potw_name,offset))

        # use the geometry to decide whether this is a flow BC or a point source
        hits=model.match_gazetteer(name=potw_name)
        if hits[0]['geom'].type=='LineString':
            log.info("%s: flow bc"%potw_name)
            Q_bc=drv.FlowBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)],
                            dredge_depth=model.dredge_depth)
        else:
            log.info("%s: source bc"%potw_name)
            Q_bc=drv.SourceSinkBC(name=potw_name,Q=Q_da,filters=[dfm.Lag(-offset)],
                                  dredge_depth=model.dredge_depth)

        salt_bc=drv.ScalarBC(parent=Q_bc,scalar='salinity',value=0.0)
        temp_bc=drv.ScalarBC(parent=Q_bc,scalar='temperature',value=temperature)
        model.add_bcs([Q_bc,salt_bc,temp_bc])
        
def add_scaled_streamflow(model,
                          flow_locations_shp,
                          usgs_inventory_shp,
                          cache_dir,
                          time_offset=np.timedelta64(0)):
    """
    Add freshwater flows from a combination of gaged and ungaged 
    watersheds, with simple scaling between them.
    This is the approach that was used for SUNTANS runs, was replaced
    by BAHM for sfbay_dfm_v2, but is useful for periods outside 
    existing BAHM runs.

    model: instance of HydroModel, with grid set, simulation period set.

    flow_locations_shp: A shapefile with linestring giving each input
    location, 
    fields: 
      gages: A '|' separate listed of USGS gage numbers from which flow data
        will be pulled.
      area_sq_mi: watershed area for this flow.  This area is compared to the
        area in USGS inventory, in order to establish a scaling factor.
      amplify: an additional adjustment to scaling factor.

    usgs_inventory_shp: path to shapefile containing watershed area information
    
    time_offset: pull freshwater flows from this timedelta off from the
    specified.  I.e. if your run is really 2016, but you want 2015 flows,
    specify np.timedelta64(-365,'D').

    Flows are given 0 salinity and 20degC temperature.
    """
    DAY=np.timedelta64(86400,'s') # useful for adjusting times
    FT3_to_M3=0.028316847
    
    run_start=model.run_start
    run_stop=model.run_stop

    run_start = run_start + time_offset
    run_stop = run_stop + time_offset

    flow_features=wkb2shp.shp2geom(flow_locations_shp)

    # First need the observations --
    # get a list of all the gages that are referenced:
    all_gages=np.unique( np.concatenate( [gages.split('|') for gages in flow_features['gages']] ) )

    usgs_gage_cache=os.path.join(cache_dir, 'usgs','streamflow')
    
    flows_ds=usgs_nwis.nwis_dataset_collection(all_gages,
                                               start_date=run_start-5*DAY,
                                               end_date=run_stop+5*DAY,
                                               products=[60], # streamflow
                                               days_per_request='M', # monthly chunks
                                               frequency='daily', # time resolution of the data
                                               cache_dir=usgs_gage_cache)

    usgs_inventory=wkb2shp.shp2geom(usgs_inventory_shp_fn)
    station_to_area=dict( [ ("%d"%site, area)
                            for site,area
                            in zip(usgs_inventory['site_no'],
                                   usgs_inventory['drain_area']) ] )

    unique_names={}
    
    for feat_i,feat in enumerate(flow_features):
        gages=feat['gages'].split('|')
        sub_flows=flows_ds.sel(site=gages)

        featA=feat['area_sq_mi']
        gage_areas=np.array(  [ float(station_to_area[g] or 'nan')
                                for g in gages ] )

        # assume the variable name here, and that dims are [site,time],
        # and units start as cfs.

        # Weighted average of reference gages based on watershed area, and
        # data availability
        # total flow from all reference gages
        site_axis=0
        assert sub_flows['stream_flow_mean_daily'].dims[site_axis]=='site'

        # flow summed over gages at each time, in m3/s
        ref_cms=np.nansum( sub_flows['stream_flow_mean_daily'].values,axis=site_axis ) * FT3_to_M3
        # area summed over gages with valid data at each time step
        ref_area=np.sum( np.isfinite(sub_flows['stream_flow_mean_daily'].values) * gage_areas[:,None],
                         axis=site_axis )
        
        # avoid division by zero for steps missing all flows
        feat_cms=featA * ref_cms
        feat_cms[ref_area>0] /= ref_area[ ref_area>0 ]
        feat_cms[ref_area==0.0] = np.nan

        stn_ds=xr.Dataset()
        stn_ds['time']=flows_ds.time
        missing=np.isnan(feat_cms)
        if np.all(missing):
            raise Exception("Composite from gages %s has no data for period %s - %s"%(gages,
                                                                                      stn_ds.time.values[0],
                                                                                      stn_ds.time.values[-1]))
        if np.any(missing):
            missing_frac=np.sum(missing)/len(missing)
            logging.warning("Composite from gages %s has missing data (%.1f%%) in period %s - %s"%(gages,
                                                                                                   100*missing_frac,
                                                                                                   stn_ds.time.values[0],
                                                                                                   stn_ds.time.values[-1]))
            # Fill by period average
            feat_cms[missing]=np.mean(feat_cms[~missing])
        stn_ds['flow_cms']=('time',),feat_cms

        # sanitize and trim the feature name
        # the 13 character limit is for DFM/DWAQ.  maybe could be longer?
        src_name=feat['name'].replace(' ','_').replace(',','_')[:13]
        if src_name in unique_names:
            serial=1
            while True:
                test_name="%s_%dser"%(src_name,serial)
                if test_name not in unique_names:
                    break
                serial+=1
            logging.warning("Source name %s duplicate - will use %s"%(src_name,test_name))
            src_name=test_name

        unique_names[src_name]=src_name

        flow_bc=model.FlowBC(name=src_name,
                             geom=feat['geom'],
                             Q=stn_ds.flow_cms)
        salt_bc=model.ScalarBC(parent=flow_bc,salinity=0.0)
        temp_bc=model.ScalarBC(parent=flow_bc,temperature=20.0)
        
        model.add_bcs([flow_bc,salt_bc,temp_bc])
