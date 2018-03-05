import xarray as xr
from stompy.model.delft import dfm_grid

## 

ds=xr.open_dataset('runs/medium_16/DFM_OUTPUT_medium_16/medium_16_0002_map.nc')

g=dfm_grid.DFMGrid(ds)
## 

# That's 18hours
# This is only for FlowLinks...
# -1 should be surface
unorm=ds.unorm.isel(time=36,laydim=-1)

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

scat=ax.scatter( ds.FlowLink_xu.values,ds.FlowLink_yu.values,
                 30,unorm.values,
                 cmap='seismic')
scat.set_clim([-1,1])


g.plot_edges(color='k',lw=0.2)
ax.axis('equal')

## 

# examine one particular edge
j=3699 # g.select_edges_nearest( plt.ginput()[0] )
j_xy=g.edges_center()[j]

# 7331
flow_link=np.argmin( (j_xy[0]-ds.FlowLink_xu.values)**2 + 
                     (j_xy[1]-ds.FlowLink_yu.values)**2  )

## 

wc_unorm=ds.unorm.isel(nFlowLink=flow_link)

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

img=ax.imshow( wc_unorm.T, interpolation='none')
plt.colorbar(img)

# This shows a ramp up over the first 3 hours,
# to a mostly barotropic current, maxing out at
# unorm=-0.2 m/s
# with a surface layer velocity of maybe +0.03 m/s on top of the
# barotropic
# For this edge, into the domain is eastward.  Not sure of the 
# sign convention here.
# Also 0.2m/s seems high for a 3463m deep water column.

cell_inside=g.edges[j]['cells'].max() # 2335
cell_depth=ds.FlowElem_zcc[cell_inside]

# the midpoint of the edge is:
#  [  474463.84012483,  4035008.79871998]

## 

# I would like to see the actual input file for this point

g_splice=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_file)
# This is the index used for naming the BC feature
j_splice=g_splice.select_edges_nearest( g.edges_center()[j] ) # 97746

# edge_src: 2.0
# cell_src: 2.0
cell_inside_splice=g_splice.edge_to_cells(j_splice).max() # 51710

# velocity data:
# the layers are written like they are 0=bed, 1=surface, starts 
# at bed, finest layers near surface.
# fine.
# data is written alternating u/v
# first tuple is -0.612, 0.521, 
# last tuple is -0.152, 0.196

# seems a bit strange that the deep part would have such a large velocity.

# What is coming from the OTPS side?
ji=np.nonzero( boundary_edges==j_splice )[0][0] # 155
#  0.0134 m/s
veloc_u_at_t=otps_u.result.isel(site=ji).sel(time=unorm.time)
# -0.02160
veloc_v_at_t=otps_v.result.isel(site=ji).sel(time=unorm.time)

## 
roms_at_boundary.isel(boundary=ji)


# index 7
roms_ti=np.searchsorted( roms_at_boundary.time.values,
                         unorm.time.values )
# instantaneous values here max out at 0.11
# u velocity maxes out at 0.03 m/s
#roms_u.isel(time=roms_ti)
#roms_v.isel(time=roms_ti)

## 
roms_u=roms_at_boundary.isel(boundary=ji).u
roms_v=roms_at_boundary.isel(boundary=ji).v

for zi in range(len(roms_u.depth)):
    roms_u.values[:,zi] = filters.lowpass(roms_u.values[:,zi],
                                          cutoff=36,order=4,dt=6)
    roms_v.values[:,zi] = filters.lowpass(roms_v.values[:,zi],
                                          cutoff=36,order=4,dt=6)
roms_uv=xr.DataArray( np.array([roms_u.values,roms_v.values]).transpose(1,2,0),
                      coords=[('time',roms_u.time),
                              ('depth',roms_u.depth),
                              ('comp',['e','n'])])

## 
# that yields roms_uv.sel(comp='e').isel(time=roms_ti)
# -0.025 at the surface, to 0.020 at the bed.
# but oddly, this roms output only goes up to 6 layers,
# which would be depth=50m.
from stompy.spatial import proj_utils
ll2utm=proj_utils.mapper('WGS84','EPSG:26910')

# len(boundary_edges)
# Out[158]: 215

# matches len(roms_at_boundary.boundary)
# 


# roms_u,v are coming from the right location
# p ll2utm( np.array( [roms_u.lon.values,roms_u.lat.values] ) ) => 473119.29832762,  4035012.86038378
# but the magnitudes are pretty high: 
# p roms_u.isel(time=7)
# array([-0.174131, -0.22905 , -0.244616, -0.162827, -0.152734, -0.16416 ,
#       -0.174424, -0.173242, -0.174055, -0.175324, -0.203597, -0.233165,
#       -0.25147 , -0.444882])
# That's surface down to the bed.
# some of that is ringing, looks like it needs a good 8 days to get past the ringing.

# made the pad much larger ...
# np.searchsorted( roms_u.time, run_start ) ==>  44 
# about 10 days in.
# but roms_u is still really large:
# array([-0.079227, -0.12812 , -0.146209, -0.08452 , -0.113202, -0.2065  ,
#        -0.288812, -0.339871, -0.417922, -0.443749, -0.46324 , -0.555395,
#        -0.612671, -0.884557])
# that's down to 1000m.
# This is before filtering.  the ROMS data at 1000m has a big event around 
#  2017-06-30.  Doesn't look that realistic: A spike of -0.9 m/s to 0.35 m/s
# at 1000m over 2 days.
# the deep currents are similar to the surface, but delayed 2--3 days.
# and the deep currents, even after filtering, peak at -0.8 m/s.
# 
# with the OTPS data, this ends up as:
# array([-0.220855, -0.268542, -0.283811, -0.218441, -0.209989, -0.204597,
#        -0.199946, -0.221937, -0.260027, -0.28805 , -0.339126, -0.403871,
#        -0.4981  , -0.801727])

# which is ultimately written to the t3d as:
# -0.802 0.279   -0.498 0.319   -0.404 0.321   -0.339 0.312   -0.288 0.308 
# -0.260 0.315   -0.222 0.322   -0.200 0.315   -0.205 0.316   -0.210 0.319
# -0.218 0.339   -0.284 0.292   -0.269 0.228   -0.221 0.161
# 
# look more closely at roms for     time     datetime64[ns] 2017-07-01T03:00:00
# 

## 

ds=xr.open_dataset('/opt/data/delft/cache/ca_roms/ca_subCA_das_2017070103.nc')

lat=36.46
lon=236.70

lat_i=np.searchsorted(ds.lat.values,36.45994386)
lon_i=np.searchsorted(ds.lon.values,236.6926819)

ds.isel(time=0,lat=lat_i,lon=lon_i).u

#array([-0.079227, -0.12812 , -0.146209, -0.08452 , -0.113202, -0.2065  ,
#        -0.288812, -0.339871, -0.417922, -0.443749, -0.46324 , -0.555395,
#        -0.612671, -0.884557])

# That just seems crazy.

## 

# Can I get a second opinion from HYCOM?
if 0:
    # opendap: takes forever to actually load the data.
    hycom=xr.open_dataset('http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7',
                          decode_times=False)
else:
    hycom=xr.open_dataset('/opt/data/hycom/snaps/hycom_GLBv0.08_577_2017070112_t000.nc',
                          decode_times=False)

del hycom['tau']
hycom=xr.decode_cf(hycom)

## 

lat=36.46
lon=236.70

lat_i=np.searchsorted(hycom.lat.values,lat)
lon_i=np.searchsorted(hycom.lon.values,lon-360.0)

## 

# Can I get the whole 4 months for this one spot?

hycom_wc=hycom.isel(lat=lat_i,lon=lon_i,time=0)

## 

# 40 depth, 976 times, 4 3D variables, 5 2D variables.
# 8 * 976*( 40*4 + 5 ) => ~1.5MB

plt.figure(4).clf()
fig,axs=plt.subplots(1,3,num=4,sharey=True)


z=-hycom_wc.depth

axs[0].plot( hycom_wc.water_u, z,label='u')
axs[0].plot( hycom_wc.water_v, z,label='v')
axs[1].plot( hycom_wc.water_temp, z,label='temp')
axs[2].plot( hycom_wc.salinity, z,label='salt')

for ax in axs:
    ax.legend()
