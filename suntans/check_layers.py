import xarray as xr
import matplotlib.pyplot as plt
import stompy.grid.unstructured_grid as ugrid
import numpy as np
##

ds=xr.open_dataset("/opt/sfb_ocean/suntans/runs/bay005/Estuary_SUNTANS.nc.nc.3")
g=ugrid.UnstructuredGrid.from_ugrid(ds)

## 
# zoom=(553549.7903616173, 568638.9767007448, 4156324.2293039323, 4173345.0000325167)
zoom=(578832.5198390371, 582952.382985177, 4149651.22715982, 4152186.982364899)

fig=plt.figure(1)
fig.clf()
fig,axs=plt.subplots(2,1,num=1,sharex=True,sharey=True)

t=64
Nk=4
dzz=ds.dzz.isel(time=t,Nk=Nk).values
salt=ds.salt.isel(time=t,Nk=Nk).values

g.plot_edges(color='k',lw=0.5,ax=axs[0])
ccoll=g.plot_cells(values=dzz,mask=np.isfinite(dzz),cmap='jet',ax=axs[0],
                   clim=[0,0.5])
#ccoll=g.plot_cells(values=np.arange(g.Ncells()),ax=axs[0],cmap='jet')
plt.colorbar(ccoll,ax=axs[0])

ccoll2=g.plot_cells(values=salt,mask=np.isfinite(salt),cmap='jet',ax=axs[1])
plt.colorbar(ccoll2,ax=axs[1])

axs[0].axis('equal')
axs[0].axis(zoom)

# over 44--47, Nk=3 is retreating to the south --
# in the cells that have Nk=3, salinity is almost 0.
# so the problem is not just a visit artifact.

# So take a closer look at cell 345, proc 0.
# cell 345 wets into k=3 at line 747095 of output.
# Why are there two calls to UpdateDZ?
# once phys.c:3515, in UPredictor, and once in boundaries.c.  Why would it do that??
# That could be really bad.  not sure yet.

# time step 46, ntout=180, so timestep 8280.
# currently at 3860.

# at t=40, seems that Nk=3 is totally dry,
# but t=41, the southern end starts to wet, but
# only the border triangles get an updated salinity.

# UpdateScalars called like this for salinity.  This is after UpdateDZ.
# 
#           UpdateScalars(grid,phys,prop,phys->wnew,phys->s,phys->boundary_s,phys->Cn_R,
#                      prop->kappa_s,prop->kappa_sH,phys->kappa_tv,prop->theta,
#                      phys->uold,phys->wtmp,NULL,NULL,0,0,comm,myproc,1,prop->TVDsalt,1);

# The problem is when ctop < ctopold.
# but scalars.c:412 should take care of this.a

# try again, logging c=12028 this time.
# also note that the cells that do not have the problem (like 345, and
# other clumps) are the lowest index cells.

# Fixed that, but there are some places where shallow cells get bad scalar.
# trying to log one of those - proc 3, cell 6653.

# This maybe right at n=11520
# ctop goes from 5 (dry) to 4 (wet)
# Hmm -- I may have made a misstep in totally drying cells out.
# when a cell dries, I think that currently all layers are
# set to to 0. it is completely omitted from any scalar updates
# until it wets, at which point it starts from zero?.

# Make a test case.
# tried a very minor fix, but needs more.  probably need to dissect the UpdateScalars
# code and do at least a little bit of work even when the cell is 'dry'.
