import numpy as np
from stompy.grid import unstructured_grid, exact_delaunay
from stompy import utils
from stompy.spatial import wkb2shp, linestring_utils
import matplotlib.pyplot as plt
from shapely import geometry
from stompy.plot import plot_wkb
from stompy.spatial import constrained_delaunay, field

import six

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(constrained_delaunay)
six.moves.reload_module(linestring_utils)

##

centerlines=wkb2shp.shp2geom('centerlines.shp')
bounds =wkb2shp.shp2geom('quad_bounds.shp')


lin_scale=constrained_delaunay.ConstrainedXYZField.read_shps(["quad_scale.shp"],value_field='scale')
tel_scale=field.ApolloniusField.read_shps(['quad_tele_scale.shp'],value_field='scale')
scale = field.BinopField( lin_scale, np.minimum, tel_scale)

def process_quad_patch(name,M=None):

    center_idx=np.nonzero(centerlines['name']==name)[0][0]
    centerline=centerlines['geom'][center_idx]
    if M is None:
        M=centerlines['rows'][center_idx]
    bound     =bounds['geom'][ bounds['name']==name ][0]


    center=linestring_utils.resample_linearring( np.array(centerline), scale, closed_ring=0)

    g=unstructured_grid.UnstructuredGrid(max_sides=6)

    # Smooth the exterior

    ext_points=np.array(bound.exterior)
    ext_points=linestring_utils.resample_linearring(ext_points,scale,closed_ring=True)
    from stompy import filters
    ext_points[:,0]= filters.lowpass_fir(ext_points[:,0],3)
    ext_points[:,1]= filters.lowpass_fir(ext_points[:,1],3)
    smooth_bound=geometry.Polygon(ext_points)

    L=smooth_bound.exterior.length
    def profile(x,s,perp):
        probe_left=geometry.LineString([x,x+L*perp])
        probe_right=geometry.LineString([x,x-L*perp])

        left_cross=smooth_bound.exterior.intersection(probe_left)
        right_cross=smooth_bound.exterior.intersection(probe_right)

        assert left_cross.type=='Point',"Fix this for multiple intersections"
        assert right_cross.type=='Point',"Fix this for multiple intersections"

        pnt_left=np.array(left_cross)
        pnt_right=np.array(right_cross)
        d_left=utils.dist(x,pnt_left)
        d_right=utils.dist(x,pnt_right)

        return np.interp(np.linspace(-1,1,M),
                         [-1,0,1],[-d_right,0,d_left])

    g.add_rectilinear_on_line(center,profile)
    g.renumber()
    return g


grids=[process_quad_patch(name)
       for name in [ 'suisun_main','grizzly','cutoff_south',
                     'montezuma','cutoff_north','short_cut',
                     'port_chicago','griz_spur']]


g=grids[0].copy()
for other in grids[1:]:
    g.add_grid(other)
    
# 

fig=plt.figure(1)
fig.clf()
fig,axs=plt.subplots(2,1,sharex=True,sharey=True,num=1)

ax=axs[0]
for centerline in centerlines['geom']:
    plot_wkb.plot_wkb(centerline,color='r',ax=ax)
for bound in bounds['geom']:
    plot_wkb.plot_wkb(bound,color='0.8',zorder=-2,ax=ax)
g.plot_edges(color='k',ax=ax)
ax.axis('equal')

##

from stompy.grid import orthogonalize

tweaker=orthogonalize.Tweaker(g)

n_iters=20

angle_thresh_deg=1.0
angles=g.angle_errors()
angles[np.isnan(angles)]=0.0
bad_angles=np.abs(angles*180/np.pi)>angle_thresh_deg
print(f"{bad_angles.sum()} bad angles, for threshold of {angle_thresh_deg} degrees")

e2c=g.edge_to_cells()
cells=np.unique( e2c[bad_angles] )

##

for it in range(n_iters):
    for c in cells:
        tweaker.nudge_cell_orthogonal(c)
    angles=g.angle_errors()
    angles[np.isnan(angles)]=0.0
    bad_angles=np.abs(angles*180/np.pi)>angle_thresh_deg
    print(f"{bad_angles.sum()} bad angles, for threshold of {angle_thresh_deg} degrees")
    if bad_angles.sum()==0:
        break
##
ax=axs[1]
for centerline in centerlines['geom']:
    plot_wkb.plot_wkb(centerline,color='r',ax=ax)
for bound in bounds['geom']:
    plot_wkb.plot_wkb(bound,color='0.8',zorder=-2,ax=ax)
g.plot_edges(color='k',ax=ax)
ax.axis('equal')


##


g.renumber()
out_fn='suisun_main-v05.nc'
os.path.exists(out_fn) and os.unlink(out_fn)
g.write_ugrid(out_fn)
