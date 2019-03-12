import os
from stompy.spatial import field
from stompy import memoize

base=os.path.dirname(__file__)
data_root=os.path.join(base,"../../../data/")

@memoize.memoize()
def dem():
    srcs=[os.path.join(data_root,"bathy_interp/master2017/tiles_2m_20171024/merged_2m.tif"),
          os.path.join(data_root,"bathy_dwr/gtiff/dem_bay_delta*.tif"),
          os.path.join(data_root,"ncei/coastal_relief-farallones_clip-utm-merge.tif")]
    mrf=field.MultiRasterField(srcs)
    return mrf
    
