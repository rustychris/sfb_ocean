"""
Experiment with projection methods 

v01: Try 3D
v00: Basic 2D projection was working
"""
from scipy import sparse
from scipy.sparse import linalg

import xarray as xr

from sfb_dfm_utils import ca_roms
from stompy.grid import unstructured_grid
from stompy import utils
from stompy.plot import plot_utils


##

def edge_normal_to_cell_vector(g,unorm):
    edge_normals=g.edges_normals()
    cell_vecs=np.zeros( (g.Ncells(),2), 'f8')
    for c in range(g.Ncells()):
        # What about a little linear system:
        js=g.cell_to_edges(c)
        A=edge_normals[js]
        b=unorm[js]
        Ainv=np.linalg.pinv(A)
        cell_vecs[c,:]=Ainv.dot(b)
    return cell_vecs

def cell_vector_to_edge_normal(g,cell_u,cell_v,boundary='zero'):
    """
    g: grid
    cell_u: [g.Ncells()] u component of vector
    cell_v: ...
    boundary: "zero" edge normals on the boundary get 0.
     "inside" take value from valid neighbor
    """
    edge_u=np.zeros(g.Nedges(),'f8')
    edge_v=np.zeros(g.Nedges(),'f8')
    if boundary=='zero':
        valid=np.all(g.edges['cells']>=0,axis=1) # interior edges
        edge_u[valid]= 0.5*(cell_u[g.edges['cells'][valid,0]] +
                            cell_u[g.edges['cells'][valid,1]])
        edge_v[valid]= 0.5*(cell_v[g.edges['cells'][valid,0]] +
                            cell_v[g.edges['cells'][valid,1]])
    else:
        c0=g.edges['cells'][:,0].copy()
        c1=g.edges['cells'][:,1].copy()
        c0[ c0<0 ] = c1[ c0<0 ]
        c1[ c1<0 ] = c0[ c1<0 ]
        edge_u[:]=0.5*(cell_u[c0] + cell_u[c1])
        edge_v[:]=0.5*(cell_v[c0] + cell_v[c1])
    edge_uv=np.c_[edge_u,edge_v]

    edge_norms=g.edges_normals()
    return (edge_uv*edge_norms).sum(axis=1)

##

g=unstructured_grid.UnstructuredGrid.from_ugrid('derived/matched_grid_v01.nc')

##
# Start with evenly spaced sigma layers
class Un3D(object):
    g=None
    n_layer=10
    HOR=1
    VER=2

    def __init__(self,g):
        self.init_from_grid(g)
    def init_from_grid(self,g):
        self.g=g
        self.n_layer=10
        self.d_sigmas=(1./self.n_layer) * np.ones(self.n_layer)
        
        # 3700 cells in the horizontal
        # change to WAQ-ish nomenclature.  elements, links are 2D, segments,exchanges are 3D
        self.element_depth=self.g.cells['cell_depth'] # m, positive up, relative to NAVD88

        self.Nelement=self.g.Ncells()
        self.Nsegment=self.Nelement*self.n_layer
        self.Nlink=self.g.Nedges() # includes closed edges

        #          horiz
        self.Nexchange_hor=self.n_layer*self.Nlink
        self.Nexchange_ver=(self.n_layer-1)*self.Nelement
        self.Nexchange=self.Nexchange_hor + self.Nexchange_ver

        self.element_area=self.g.cells_area()

        self.init_segments()
        self.init_exchanges()

    def init_segments(self):
        self.meta_segment=np.zeros(self.Nsegment,
                                   [('elt','i4')])
        elts=np.arange(self.Nelement)
        
        self.meta_segment['elt']=np.tile(elts,self.n_layer)
    
    def init_exchanges(self):
        # Build metadata for exchanges:

        self.meta_exchange=np.zeros(self.Nexchange,
                                    [('src_seg','i4'),
                                     ('dst_seg','i4'),
                                     ('orient','i4'),
                                     # length scale constant
                                     ('L0','f8'),
                                     # length scale as factor * depth
                                     ('Lfac','f8')])

        self.meta_exchange['orient'][:self.Nexchange_hor]=self.HOR
        self.meta_exchange['orient'][self.Nexchange_hor:]=self.VER
        e2c=self.g.edge_to_cells()

        centers=g.cells_center() # can use circumcenters in testing at least
        
        # Fill in the horizontal exchanges:
        for j in range(self.Nlink):
            c0,c1 = e2c[j]
            for k in range(self.n_layer):
                j3d=self.exch3d_hor(j,k)
                self.meta_exchange['src_seg'][j3d] = self.seg3d(c0,k)
                self.meta_exchange['dst_seg'][j3d] = self.seg3d(c1,k)
                self.meta_exchange['L0'][j3d] = utils.dist(centers[c0] - centers[c1])
                self.meta_exchange['Lfac'][j3d]=0.0
        # Fill in the vertical exchanges:
        for c in range(self.Nelement):
            for k in range(self.n_layer-1):
                j3d=self.exch3d_ver(c,k)
                self.meta_exchange['src_seg'][j3d] = self.seg3d(c,k)
                self.meta_exchange['dst_seg'][j3d] = self.seg3d(c,k+1)
                self.meta_exchange['L0'][j3d] = 0.0
                self.meta_exchange['Lfac'][j3d] = 0.5*(self.d_sigmas[k] + self.d_sigmas[k+1])
                
        
    # Convert to linear indices
    def seg3d(self,elt,k):
        if elt<0:
            return elt
        else:
            return elt+k*self.Nelement
    def exch3d_hor(self,link,k):
        return k*self.Nlink + link
    def exch3d_ver(self,elt,k0):
        return self.Nexchange_hor + k0*self.Nelement + elt

    def volumes(self,eta):
        """
        return linear array of volumes for all segments
        """
        element_V=(eta-self.element_depth) * self.element_area
        # for sigma layers, can do this in a dense 2D array
        # element should be the fastest changing index
        segment_V_2=np.tile(element_V,[self.n_layer,1]) * self.d_sigmas[:,None]
        return segment_V_2.ravel()

    # -- OPERATORS --
    def exchange_to_link(self):
        """ 
        create a sparse matrix which, left-multiplied to J ~ 3D fluxes, [Nexchanges]
        returns J2 ~ 2D fluxes, [Nlinks]
        """
        ij=[]
        values=[]
        for j in range(self.g.Nedges()):
            for k in range(self.n_layer):
                j3d=self.exch3d_hor(j,k)
                ij.append( (j,j3d) )
                values.append( 1 )
        ijs=np.array(ij,dtype=np.int32)
        data=np.array(values,dtype=np.float64)
        Jproject=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(self.Nlink,self.Nexchange) )
        return Jproject

    def segment_to_element(self):
        """ 
        create a sparse matrix which, left-multiplied to V ~ 3D volumes, [Nsegment]
        returns V2 ~ 2D volumes, [Nelement]
        """
        ij=[]
        values=[]
        for elt in range(self.Nelement):
            for k in range(self.n_layer):
                seg3d=self.seg3d(elt,k)
                ij.append( (elt,seg3d) )
                values.append( 1 )
        ijs=np.array(ij,dtype=np.int32)
        data=np.array(values,dtype=np.float64)
        Vproject=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(self.Nelement,self.Nsegment) )
        return Vproject
    
    def div2d_full_op(self):
        g=self.g
        N=g.Ncells()
        M=g.Nedges()

        # construct the matrix from a sequence of indices and values
        # Divergence operator T is (N,M)
        # V1-V0 = dt * T.J

        ij=[]
        values=[] # successive value for the same i.j will be summed

        for j in range(g.Nedges()):
            e = g.edges[j]
            ic1,ic2 = e['cells']

            if ic1<0 or ic2<0:
                continue # boundary edge
            # Assumes the sign convention of UnstructuredGrid.edges_normals(),
            # which is positive towards c2
            ij.append( (ic1,j) )
            values.append(-1)
            ij.append( (ic2,j) )
            values.append(1)

        ijs=np.array(ij,dtype=np.int32)
        data=np.array(values,dtype=np.float64)
        T=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(N,M) )
        return T

    def div_full_op(self):
        ij=[]
        values=[]
        for j3d in range(self.Nexchange):
            src=self.meta_exchange['src_seg'][j3d]
            dst=self.meta_exchange['dst_seg'][j3d]
            if src<0 or dst<0:
                continue # boundary edge
            ij.append( (src,j3d) )
            values.append(-1)
            ij.append( (dst,j3d) )
            values.append(1)

        ijs=np.array(ij,dtype=np.int32)
        data=np.array(values,dtype=np.float64)
        T=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(self.Nsegment,self.Nexchange) )
        return T

    def grad2d_full_op(self):
        """ gradient operator in 2D as a sparse array 
        """
        g=self.g
        N=g.Ncells()
        M=g.Nedges()

        # construct the matrix from a sequence of indices and values
        # Gradient operator G is (M,N)
        # F = G.P

        g.edge_to_cells()

        centers=g.cells_center() # can use circumcenters in testing at least

        ij=[]
        values=[] # successive value for the same i.j will be summed

        for j in range(g.Nedges()):
            e = g.edges[j]
            ic1,ic2 = e['cells']

            if ic1<0 or ic2<0:
                continue # boundary edge

            L=utils.dist( centers[ic1] - centers[ic2] )

            # Assumes the sign convention of UnstructuredGrid.edges_normals(),
            # which is positive towards c2
            # 
            ij.append( (j,ic1) )
            values.append(-1/L)
            ij.append( (j,ic2) )
            values.append(1/L)

        ijs=np.array(ij,dtype=np.int32)
        data=np.array(values,dtype=np.float64)
        G=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(M,N) )
        return G

    def grad_full_op(self,eta=0.0):

        centers=self.g.cells_center() # can use circumcenters in testing at least

        ij=[]
        values=[] # successive value for the same i.j will be summed

        waterdepth=eta - self.element_depth
        
        for j3d in range(self.Nexchange):
            src=self.meta_exchange['src_seg'][j3d]
            dst=self.meta_exchange['dst_seg'][j3d]
            if src<0 or dst<0:
                continue # boundary edge

            if self.meta_exchange['orient'][j3d] == self.HOR:
                L=self.meta_exchange['L0'][j3d]
            else: # Vertical
                elt=self.meta_segment['elt'][src]
                L=self.meta_exchange['Lfac'][j3d] * waterdepth[elt]

            # Assumes the sign convention of UnstructuredGrid.edges_normals(),
            # which is positive towards c2
            #
            if L<=0.0:
                pdb.set_trace()
            ij.append( (j3d,src) )
            values.append(-1./L)
            ij.append( (j3d,dst) )
            values.append(1./L)

        ijs=np.array(ij,dtype=np.int32)
        data=np.array(values,dtype=np.float64)
        G=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(self.Nexchange,self.Nsegment) )

        return G
        
            
model=Un3D(g)

##

T=model.div_full_op()
G=model.grad_full_op()
L=T.dot(G) # the Laplacian : NxN

##

# 2D versions of those:
T2=model.div2d_full_op()
G2=model.grad2d_full_op()
L2=T2.dot(G2)

##

if 0: # grab a snapshot:
    # Pull in one layer of ROMS currents, just to have a pattern to deal with
    roms_ds=xr.open_dataset("../../cache/ca_roms/ca_subCA_das_2017091103.nc")

    roms_u=roms_ds.isel(time=0).u.transpose('lat','lon','depth').values
    roms_v=roms_ds.isel(time=0).v.transpose('lat','lon','depth').values
    roms_zeta=roms_ds.isel(time=0).zeta.transpose('lat','lon').values

else: # average a bunch of them:
    roms_files = ca_roms.fetch_ca_roms(np.datetime64("2017-08-01"),
                                       np.datetime64("2017-08-10"))
    roms_u=0
    roms_v=0
    roms_zeta0=0
    roms_zeta1=0

    # Three windows, one for the start of the timestep, one for the
    # middle, one for the end.
    # Each roms file is 6 hours, so 
    win=np.hanning(len(roms_files))
    win = win / win.sum()

    win0=np.zeros(len(roms_files))
    win1=np.zeros(len(roms_files))
    win0[:-1]=np.hanning(len(roms_files)-1)
    win1[1:]=np.hanning(len(roms_files)-1)
    win0=win0/win0.sum()
    win1=win1/win1.sum()

    for i,roms_file in enumerate(roms_files):
        print(roms_file)
        ds=xr.open_dataset(roms_file)

        u=ds.isel(time=0).u.transpose('lat','lon','depth').values
        v=ds.isel(time=0).v.transpose('lat','lon','depth').values
        zeta=ds.isel(time=0).zeta.transpose('lat','lon').values
        roms_u=roms_u + win[i]*u
        roms_v=roms_v + win[i]*v
        roms_zeta0=roms_zeta0 + win0[i]*zeta
        roms_zeta1=roms_zeta1 + win1[i]*zeta
        ds.close()
    dt=6*3600.
##

#eta0=0.0*np.ones(model.Nelement,'f8')
#eta1=eta0.copy()

# for adding in MSL => NAVD88 correction.  Just dumb luck that it's 1.0
dfm_zeta_offset=1.0

eta0=dfm_zeta_offset + roms_zeta0[model.g.cells['lati'],model.g.cells['loni']]
eta1=dfm_zeta_offset + roms_zeta1[model.g.cells['lati'],model.g.cells['loni']]

V0=model.volumes(eta0)
V1=model.volumes(eta1)

J0=0.0*np.ones(model.Nexchange,'f8')

# edge normal velocities:
u0=np.zeros(model.Nexchange,'f8')
# areas for exchanges:
flux_A=np.zeros(model.Nexchange,'f8')

roms_depth=roms_ds.depth.values

# order is surface down to the bed
# sign convention is -1 = bed, 0=water surface
model.sigma_centers=0-(np.cumsum(model.d_sigmas) - 0.5*model.d_sigmas )

# fill in horizontal velocities, calculate flux areas at the same time
e2c=model.g.edge_to_cells()
boundary='inside'
edge_normals=model.g.edges_normals()
eta=eta0
water_depth=eta-model.element_depth
edge_lengths=model.g.edges_length()

for j in range(model.Nlink):
    c0=e2c[j][0]
    c1=e2c[j][1]
    if boundary=='inside':
        if c0<0:
            c0=c1
        elif c1<0:
            c1=c0
    elif c0<0 or c1<0:
        continue # boundary, no flown
    
    uc0=roms_u[model.g.cells['lati'][c0],model.g.cells['loni'][c0],:]
    vc0=roms_v[model.g.cells['lati'][c0],model.g.cells['loni'][c0],:]
    uc1=roms_u[model.g.cells['lati'][c1],model.g.cells['loni'][c1],:]
    vc1=roms_v[model.g.cells['lati'][c1],model.g.cells['loni'][c1],:]

    normal=edge_normals[j]
    roms_unorms=0.5*(uc0+uc1)*normal[0] + 0.5*(vc0+vc1)*normal[1]

    valid=np.isfinite(roms_unorms)

    # water column depth at the edge:
    j_depth=0.5*(water_depth[c0]+water_depth[c1])
    # sounding down to center of edge face  - positive down, starting at the surface!
    j3d_depths=(-j_depth)*model.sigma_centers

    unorms=np.interp(j3d_depths,roms_depth[valid],roms_unorms[valid])

    for k in range(model.n_layer):
        j3d=model.exch3d_hor(j,k)
        u0[j3d]=unorms[k]
        flux_A[j3d]=edge_lengths[j] * (j_depth*model.d_sigmas[k])


# fill in vertical velocities: We don't have any!
# and areas are easy
for c in range(model.Nelement):
    for k in range(model.n_layer-1):
        j3d=model.exch3d_ver(c,k)
        flux_A[j3d]=model.element_area[c]
## 

# Conversion to 2D:
#  would like matrices to go from 3D cell centered to 2D cell centered,
#  i.e. just sum across water columns
#  and 3D flux-face centered to 2D edge-centered
#  so sum on edges
if 1: # 2D version
    Jproject=model.exchange_to_link()
    flux_A2=Jproject.dot(flux_A)

    J0=flux_A*u0

    J0_2=Jproject.dot(J0)
    Vproject=model.segment_to_element()
    V0_2=Vproject.dot(V0)
    V1_2=Vproject.dot(V1)

    if 0:# Dev: absorb the 2D divergence entirely into the
        # freesurface / volume
        depth_error=dt*div_J0_2 / model.element_area
        V1_2+=depth_error*model.element_area

    div_J0_2=T2.dot(J0_2)

    rhs=(V1_2-V0_2)/dt-T2.dot(J0_2)
    P2=linalg.spsolve(L2,rhs)
    J2corr=G2.dot(P2)
    corr_unorm2=J2corr/flux_A2
    corr_ucell=edge_normal_to_cell_vector(model.g,corr_unorm2)
    orig_ucell=edge_normal_to_cell_vector(model.g,Jproject.dot(u0))

##

if 0: # 3D version
    # ideally, this holds
    # the change in volume is equal to the divergence of the fluxes integrated
    # for one time step.
    # But since we have some errors, the additional corrector term is necessary
    # V1-V0 = dt *T.J0 + L.P
    # and we solve for P:
    # L.P = V1-V0-dt*T.J0
    # This seems to work, though P has entries ~ 1e9
    J0=flux_A*u0

    rhs=(V1-V0)/dt-T.dot(J0)
    # maybe 15 seconds?
    P=linalg.spsolve(L.tocsr(),rhs)
    Jcorr=G.dot(P)
    corr_unorm=Jcorr/flux_A

    # How did we do?

    div_Jnew=T.dot(Jcorr+J0) # This is very small.  Success!
    # just plot surface for the moment
    corr_ucell=edge_normal_to_cell_vector(g,corr_unorm[:model.Nlink])

##

# Need a method for going from edge normal to cell-centered vectors
# doesn't have to be exact, just for visualization

plt.figure(1).clf()
fig,(ax,ax2)=plt.subplots(1,2,sharex=True,sharey=True,num=1)

cc=model.g.cells_center()
#qset=ax.quiver(cc[:,0],cc[:,1],orig_ucell[:,0],orig_ucell[:,1])
qset2=ax.quiver(cc[:,0],cc[:,1],corr_ucell[:,0],corr_ucell[:,1],color='m')
#qset3=ax.quiver(cc[:,0],cc[:,1],
#                orig_ucell[:,0]+corr_ucell[:,0],
#                orig_ucell[:,1]+corr_ucell[:,1],color='k')

#ec=g.edges_center()
#norms=g.edges_normals()
# qset4=ax.quiver(ec[:,0],ec[:,1],norms[:,0]*edge_unorm,norms[:,1]*edge_unorm,
#                 units='xy',angles='xy',scale_units='xy',scale=2e-4,)
# qset5=ax.quiver(ec[:,0]+800,ec[:,1],norms[:,0]*(edge_unorm+corr_unorm),norms[:,1]*(edge_unorm+corr_unorm),
#                 color='b',alpha=0.5,
#                 units='xy',angles='xy',scale_units='xy',scale=2e-4)


# ecoll=g.plot_edges(values=J0)
# ccoll=g.plot_cells(values=div_J0,zorder=-1)
# ccoll=g.plot_cells(values=P[:model.Nelement],zorder=-1)
#ccoll=g.plot_cells(values=P2,zorder=-1,ax=ax,cmap='jet')
div_J0_2=T2.dot(J0_2)
depth_error=dt*div_J0_2 / self.element_area
ccoll=g.plot_cells(values=depth_error,zorder=-1,ax=ax,cmap='seismic')
ccoll.set_clim([-500,500])

water_depth=V0_2/self.element_area


ccoll_depth=g.plot_cells(values=water_depth,zorder=-1,ax=ax2,cmap='jet')

plot_utils.cbar(ccoll,ax=ax,label='Depth error/incr.')
plot_utils.cbar(ccoll_depth,ax=ax2,label='water column depth')

ax.axis('equal')

# Started with rms depth error of 796!?
print("Depth error rms: %.3f"%utils.rms(depth_error))

##

# Smoothing the depth error:
# Should depth or volume get smoothed?  Probably volume, so that using
# a
# Probably to 

