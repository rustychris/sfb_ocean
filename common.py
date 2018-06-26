import os
import shutil

import numpy as np 
import pandas as pd
import stompy.model.delft.io as dio

from stompy import utils

# utility functions which are not yet general enough to go 
# into sfb_dfm_utils, but cluttering ocean_dfm.py
def write_pli(g,run_base_dir,src_name,j,suffix):
    seg=g.nodes['x'][ g.edges['nodes'][j] ]
    src_feat=(src_name,seg,[src_name+"_0001",src_name+"_0002"])
    feat_suffix=dio.add_suffix_to_feature(src_feat,suffix)
    dio.write_pli(os.path.join(run_base_dir,'%s%s.pli'%(src_name,suffix)),
                  [feat_suffix])
    return feat_suffix

def write_tim(da,suffix,feat_suffix,mdu):
    """
    da: xr.DataArray with timeseries data
    suffix: suffix to add the
    """
    # Write the data:
    run_base_dir=mdu.base_path

    ref_date,t_start,t_stop = mdu.time_range()

    columns=['elapsed_minutes']
    if da.ndim==1: # yuck pandas.
        df=da.to_dataframe().reset_index()
        df['elapsed_minutes']=(df.time.values - ref_date)/np.timedelta64(60,'s')
        columns.append(da.name)
    else:
        # it's a bit gross, but coercing pandas into putting a second dimension
        # into separate columns is too annoying.
        df=pd.DataFrame()
        df['elapsed_minutes']=(da.time.values - ref_date)/np.timedelta64(60,'s')
        for idx in range(da.shape[1]):
            col_name='val%d'%idx
            df[col_name]=da.values[:,idx]
            columns.append(col_name)

    if len(feat_suffix)==3:
        node_names=feat_suffix[2]
    else:
        node_names=[""]*len(feat_suffix[1])

    for node_idx,node_name in enumerate(node_names):
        # if no node names are known, create the default name of <feature name>_0001
        if not node_name:
            node_name="%s%s_%04d"%(src_name,suffix,1+node_idx)

        tim_fn=os.path.join(run_base_dir,node_name+".tim")
        df.to_csv(tim_fn, sep=' ', index=False, header=False, columns=columns)
        # remaining nodes should default to the same value as the first.
        # break

def write_t3d(da,suffix,feat_suffix,edge_depth,quantity,mdu):
    """
    Write a 3D boundary condition for a feature from a vertical profile (likely
       ROMS or HYCOM data)
     - most of the time writing boundaries is here
     - DFM details for rev52184:
         the LAYERS line is silently truncated to 100 characters.
         LAYER_TYPE=z assumes a coordinate of 0 at the bed, positive up
    """
    run_base_dir=mdu.base_path
    ref_date,t_start,t_stop = mdu.time_range()

    # Luckily the ROMS output does not lose any surface cells - so we don't
    # have to worry about a surface cell in roms_at_boundary going nan.
    assert da.ndim in [2,3]
    
    # get the depth of the internal cell:
    valid_depths=np.all( np.isfinite( da.values ), axis=da.get_axis_num('time') )

    valid_depths = valid_depths & (-da.depth.values > edge_depth)
    # if this becomes a problem, may have to fill in backup value?  
    assert valid_depths[0],"No valid layers in Coastal model data"

    valid_depth_idxs=np.nonzero(valid_depths)[0]

    # ROMS values count from the surface, positive down.
    # but DFM wants bottom-up.
    # limit to valid depths, and reverse the order at the same time

    da_sub=da.isel(depth=valid_depth_idxs[::-1])

    max_line_length=100 # limitation in DFM on the whole LAYERS line
    # 7 is '_2.4567'
    # -1 for minor bit of safety
    max_layers=(max_line_length-len("LAYERS=")) // 7 - 1

    # This should be the right numbers, but reverse order
    sigma = (-edge_depth - da_sub.depth.values) / -edge_depth

    # Force it to span the full water column
    sigma[0]=min(0.0,sigma[0])
    sigma[-1]=max(1.0,sigma[-1])

    if len(sigma)>max_layers:
        remapper=lambda y: np.interp(np.linspace(0,1,max_layers),
                                     np.linspace(0,1,len(sigma)),y)
        # Just because the use of remapper below is not compatible
        # with vector quantities at this time.
        assert da_sub.ndim-1 == 1
    else:
        remapper=lambda y: y

    sigma_str=" ".join(["%.4f"%s for s in remapper(sigma)])

    # This line is truncated at 100 characters in DFM r52184.
    layer_line="LAYERS=%s"%sigma_str
    assert len(layer_line)<max_line_length

    elapsed_minutes=(da_sub.time.values - ref_date)/np.timedelta64(60,'s')

    ref_date_str=utils.to_datetime(ref_date).strftime('%Y-%m-%d %H:%M:%S')
    
    # assumes there are already node names
    node_names=feat_suffix[2]

    t3d_fns=[os.path.join(run_base_dir,node_name+".t3d")
             for node_idx,node_name in enumerate(node_names) ]

    assert da_sub.dims[0]=='time' # for speed up of direct indexing

    # Write the first, then copy it to the second node
    with open(t3d_fns[0],'wt') as fp:
        fp.write("\n".join([
            "LAYER_TYPE=sigma",
            layer_line,
            "VECTORMAX=%d"%(da_sub.ndim-1), # default, but be explicit
            "quant=%s"%quantity,
            "quantity1=%s"%quantity, # why is this here?
            "# start of data",
            ""]))
        for ti,t in enumerate(elapsed_minutes):
            fp.write("TIME=%g minutes since %s\n"%(t,ref_date_str))
            # Faster direct indexing:
            # The ravel will interleave components - unclear if that's correct.
            data=" ".join( ["%.3f"%v for v in remapper(da_sub.values[ti,:].ravel()) ] )
            fp.write(data)
            fp.write("\n")
            
    for t3d_fn in t3d_fns[1:]:
        shutil.copyfile(t3d_fns[0],t3d_fn)

