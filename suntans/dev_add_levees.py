"""
Development of code for adding edge depths to a grid ala
DFM fixed weirs.

"""
from stompy.plot import plot_wkb
from stompy.grid import unstructured_grid
import stompy.model.delft.io as dio

##

dest_grid="grid-sfbay/sfei_v20_net.nc"
g=unstructured_grid.UnstructuredGrid.read_dfm(dest_grid)

# standard stuff
# - interpolate pre-existing node depths to cells
g.add_cell_field('depth',g.interp_node_to_cell(g.nodes['depth']))
g.delete_node_field('depth')
# - set baseline edge depths as shallower or neighboring cells

##
de=np.zeros(g.Nedges(),np.float64)

c1=e2c[:,0].copy() ; c2=e2c[:,1].copy()
c1[c1<0]=c2[c1<0]
c2[c2<0]=c1[c2<0]
de=np.maximum(g.cells['depth'][c1],g.cells['depth'][c2])
g.add_edge_field('edge_depth',de,on_exists='overwrite')


## 
    
# load levee data:
levee_fn='grid-sfbay/SBlevees_tdk.pli'

levees=dio.read_pli(levee_fn)

##

from shapely import geometry
def pli_to_grid_edges(g,levees):
    """
    g: UnstructuredGrid
    levees: polylines in the format returned by stompy.model.delft.io.read_pli,
    i.e. a list of features
    [ 
      [ 'name', 
        [ [x,y,z,...],...], 
        ['node0',...]
      ], ... 
    ]

    returns an array of length g.Nedges(), with z values from those features
    mapped onto edges. when multiple z values map to the same grid edge, the 
    minimum value is used.
    grid edges which do not have a levee edge get nan.
    """
    poly=g.boundary_polygon()

    # The dual additionally allows picking out edges 
    gd=g.create_dual(center='centroid',create_cells=False,remove_disconnected=False,
                     remove_1d=False)

    levee_de=np.nan*np.zeros_like(g.edges['edge_depth'])

    for levee in utils.progress(levees,msg="Levees: %s"):
        # levee: [name, Nx{x,y,z,l,r}, {labels}]
        xyz=levee[1][:,:3]
        # having shapely check the intersection is 100x
        # faster than using select_cells_nearest(inside=True)
        ls=geometry.LineString(xyz[:,:2])
        if not poly.intersects(ls): continue

        # clip the edges to get some speedup
        xxyy=[xyz[:,0].min(),
              xyz[:,0].max(),
              xyz[:,1].min(),
              xyz[:,1].max()]
        edge_mask=gd.edge_clip_mask(xxyy,ends=True)

        # edges that make up the snapped line
        gde=gd.select_edges_intersecting(ls,mask=edge_mask)
        gde=np.nonzero(gde)[0]
        if len(gde)==0:
            continue
        # map the dual edge indexes back to the original grid
        ge=gd.edges['dual_edge'][gde]

        print("Got a live one!")

        # check for closed ring:
        closed=np.all( xyz[-1,:2]==xyz[0,:2] )
        dists=utils.dist_along(xyz[:,:2])

        for j in ge:
            n1,n2=g.edges['nodes'][j]
            l1=np.argmin( utils.dist(g.nodes['x'][n1] - xyz[:,:2] ) )
            l2=np.argmin( utils.dist(g.nodes['x'][n2] - xyz[:,:2] ) )
            if l1>l2:
                l1,l2=l2,l1
            zs=xyz[l1:l2+1,2]

            if closed:
                # allow for possibility that the levee is a closed ring
                # and this grid edge actually straddles the end,
                dist_fwd=dists[l2]-dists[l1]
                dist_rev=dists[-1] - dist_fwd
                if dist_rev<dist_fwd:
                    print("wraparound")
                    zs=np.concatenate( [xyz[l2:,2],
                                        xyz[:l1,2]] )

            z=zs.min() 
            levee_de[j]=z
    return levee_de


##
if 0:
    zoom=(585813.9075730171, 586152.9743682485, 4146296.412810114, 4146567.6036336995)

    plt.figure(1).clf()
    ecoll=g.plot_edges(values=levee_de,mask=np.isfinite(levee_de),lw=2.5)

    plot_wkb.plot_wkb(poly,fc="0.8",zorder=-3)

    plt.axis('equal')
    plt.axis(zoom)



