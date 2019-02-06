from stompy.spatial import interp_4d
from stompy.plot import plot_utils
from stompy import utils
import sfbay_sun
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 
from stompy.io.local import usgs_sfbay

# Now we have data with dimensions of station,time,depth
# and want to extrapolate onto the grid with x,y,depth for a specific time
def calc_z_gradient(grp):
    z=grp.values[:,0]
    scal=grp.values[:,1]
    if len(z>1):
        m=np.polyfit(z,scal,1)[0]
    else:
        m=0

        
    scal_name=grp.columns[1]
    ret={scal_name+"_grad":m,
         scal_name+"_surf":scal[np.argmin(z)]}
    return pd.Series(ret)
def time_interp(grp,target):
    ret={}
    if len(grp)==1:
        for col in grp.columns[2:]: # should be the gradient and surface values
            ret[col]=grp[col].values[0]
    else:
        dnums=utils.to_dnum(grp['Date'].values)
        tgt_dnum=utils.to_dnum(target)
        for col in grp.columns[2:]: # should be the gradient and surface values
            ret[col]=np.interp(tgt_dnum,dnums,grp[col])
    return pd.Series(ret)

def set_ic_from_usgs_sfbay(model,
                           search_pad=np.timedelta64(20,'D'),
                           scalar='salt',
                           usgs_scalar='Salinity',
                           clip=None,
                           ocean_surf=None,ocean_grad=None,
                           ocean_xy=[534000,4181000],
                           cache_dir=None):
    df=usgs_sfbay.query_usgs_sfbay(period_start=model.run_start-search_pad,
                                   period_end=model.run_start+search_pad,
                                   cache_dir=cache_dir)
    xy=model.ll_to_native( df.loc[:, ['longitude','latitude'] ].values )
    df['x']=xy[:,0]
    df['y']=xy[:,1]

    # make the scalar name generic to simplify code below
    df.rename({usgs_scalar:'scalar'},axis=1,inplace=True)

    # Condense vertical information into a surface value and a linear vertical gradient
    df_by_profile=df.groupby(['Date','Station Number'])[ 'Depth','scalar'].apply( calc_z_gradient )

    # Condense multiple cruises by linearly interpolated that profiles
    df_time_interp=df_by_profile.reset_index().groupby('Station Number').apply( time_interp, target=model.run_start)

    # Get the x/y data back in there
    station_locs=df.groupby('Station Number')['x','y'].first()
    scal_data=pd.merge(df_time_interp,station_locs,left_index=True,right_index=True)

    # Okay - ready for spatial extrapolation

    # This doesn't do so well around ggate, where it doesn't really figure out
    # which sample should dominate.  So force an ocean salinity:
    closest_station=np.argmin(utils.dist(ocean_xy, scal_data[ ['x','y']].values ))

    # default to the value nearest the ggate, but allow caller to override
    if ocean_surf is None:
        ocean_surf=scal_data['scalar_surf'].iloc[closest_station]
    if ocean_grad is None:
        ocean_grad=scal_data['scalar_grad'].iloc[closest_station]

    scal_data_ocean=scal_data.append( {'scalar_surf':ocean_surf,
                                       'scalar_grad':ocean_grad,
                                       'x':ocean_xy[0], # point outside Golden Gate.
                                       'y':ocean_xy[1]},
                                      ignore_index=True )
    scal_surf_2d=interp_4d.weighted_grid_extrapolation(model.grid,scal_data_ocean,
                                                       value_col='scalar_surf',
                                                       alpha=1e-5,
                                                       weight_col=None)
    scal_grad_2d=interp_4d.weighted_grid_extrapolation(model.grid,scal_data_ocean,
                                                       value_col='scalar_grad',
                                                       alpha=1e-5,
                                                       weight_col=None)

    # Set salinity in the ic file:
    # salt has dimensions [time,layer,cell], use isel and transpose
    # to be sure
    depth_Nk=np.cumsum(model.ic_ds.dz.values) # positive down
    ic_values=scal_surf_2d[None,:] + scal_grad_2d[None,:]*depth_Nk[:,None]
    if clip is not None:
        ic_values=ic_values.clip(clip)
    model.ic_ds[scalar].isel(time=0).transpose('Nk','Nc').values[...]=ic_values


##
# Testing
if __name__=='__main__': 
    cache_dir="cache"
    os.path.exists(cache_dir) or os.makedirs(cache_dir)

    import six
    six.moves.reload_module(sfbay_sun)

    model=sfbay_sun.model

    model.write()

    set_ic_from_usgs_sfbay(model,
                           scalar='salt',
                           usgs_scalar='Salinity',
                           ocean_surf=34.0,ocean_grad=0.0,
                           clip=[0,34],
                           cache_dir=cache_dir)
    set_ic_from_usgs_sfbay(model,
                           scalar='temp',
                           usgs_scalar='Temperature',
                           ocean_grad=0.0,
                           clip=[5,30],
                           cache_dir=cache_dir)

    model.write_ic_ds()
    model.partition()
    model.sun_verbose_flag="-v"
    model.run_simulation()
