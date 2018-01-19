"""
Experiment with projection methods 

"""

from sfb_dfm_utils import ca_roms
from stompy.grid import unstructured_grid

from scipy import sparse
from scipy.sparse import linalg

##
# divergence operator:
def div_full_op(g):
    N=g.Ncells()
    M=g.Nedges()

    # construct the matrix from a sequence of indices and values
    # Divergence operator T is (N,M)
    # V1-V0 = dt * T.J

    g.edge_to_cells()

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

# gradient operator:
def grad_full_op(g):
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


def cell_vector_to_edge_normal(g,cell_u,cell_v):
    edge_u=np.zeros(g.Nedges(),'f8')
    edge_v=np.zeros(g.Nedges(),'f8')
    valid=np.all(g.edges['cells']>=0,axis=1) # interior edges
    edge_u[valid]= 0.5*(cell_u[g.edges['cells'][valid,0]] +
                        cell_u[g.edges['cells'][valid,1]])
    edge_v[valid]= 0.5*(cell_v[g.edges['cells'][valid,0]] +
                        cell_v[g.edges['cells'][valid,1]])
    edge_uv=np.c_[edge_u,edge_v]

    edge_norms=g.edges_normals()
    return (edge_uv*edge_norms).sum(axis=1)

##

g=unstructured_grid.UnstructuredGrid.from_ugrid('derived/matched_grid_v01.nc')

##

# 3700 cells.

#cell_depth=g.cells['cell_depth'] # m, positive up, relative to NAVD88
cell_depth=-50*np.ones(g.Ncells())

eta0=0.0*np.ones(g.Ncells(),'f8')
eta1=eta0.copy()

J0=0.0*np.ones(g.Nedges(),'f8')

V0=(eta0-cell_depth)
V1=(eta1-cell_depth)

##

T=div_full_op(g)
G=grad_full_op(g)
L=T.dot(G) # the Laplacian : NxN

##

# Pull in one layer of ROMS currents, just to have a pattern to deal with
roms_ds=xr.open_dataset("../../cache/ca_roms/ca_subCA_das_2017091103.nc")

##

cell_u=roms_ds.isel(time=0,depth=0).u.values[ g.cells['lati'], g.cells['loni']]
cell_v=roms_ds.isel(time=0,depth=0).v.values[ g.cells['lati'], g.cells['loni']]

edge_unorm=cell_vector_to_edge_normal(g,cell_u,cell_v)

# flux face areas to go with each flux
# For the moment, constant depths here.
flux_A=g.edges_length() * 50.0
J0=flux_A*edge_unorm

##

# What does the divergence look like?
# Looks good!
div_J0=T.dot(J0)

##

dt=1800 # seconds

# ideally, this holds
# the change in volume is equal to the divergence of the fluxes integrated
# for one time step.
# But since we have some errors, the additional corrector term is necessary
# V1-V0 = dt *T.J0 + L.P
# and we solve for P:
# L.P = V1-V0-dt*T.J0
# This seems to work, though P has entries ~ 1e9

rhs=(V1-V0)/dt-T.dot(J0)

P=linalg.spsolve(L.tocsr(),rhs)

Jcorr=G.dot(P)
corr_unorm=Jcorr/flux_A

##

# How did we do?
new_unorm=edge_unorm+corr_unorm

# L.dot(P) - rhs is very small.
# L.dot(P) - (T.dot(G.dot(P))) is very small, so L is working.
div_Jnew=T.dot(Jcorr+J0) # This is very small.  Success.

##

# Need a method for going from edge normal to cell-centered vectors
# doesn't have to be exact, just for visualization

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
    
corr_ucell=edge_normal_to_cell_vector(g,corr_unorm)


##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
cc=g.cells_center()
qset=ax.quiver(cc[:,0],cc[:,1],cell_u,cell_v)
#qset2=ax.quiver(cc[:,0],cc[:,1],corr_ucell[:,0],corr_ucell[:,1],color='red')
qset3=ax.quiver(cc[:,0],cc[:,1],cell_u+corr_ucell[:,0],cell_v+corr_ucell[:,1],color='red')

#ec=g.edges_center()
#norms=g.edges_normals()
#qset4=ax.quiver(ec[:,0],ec[:,1],norms[:,0]*edge_unorm,norms[:,1]*edge_unorm)
#qset5=ax.quiver(ec[:,0],ec[:,1],norms[:,0]*(edge_unorm+corr_unorm),norms[:,1]*(edge_unorm+corr_unorm))


# ecoll=g.plot_edges(values=J0)
#ccoll=g.plot_cells(values=div_J0,zorder=-1)
ccoll=g.plot_cells(values=P,zorder=-1)

ax.axis('equal')
