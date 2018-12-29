# Trying to understand if there is an issue with edges and salt BCs.
# 
# 3.328025e+03 2.688285e+03 -9.999457e-01 -1.042420e-02 5.9005078408e+05 4.0354404154e+06 14 14 3652 -1 1 -1 2 3825 3826 0 7476
# 2.687779e+03 3.328023e+03 1.027224e-02 -9.999472e-01 5.8868961961e+05 4.0370905325e+06 12 14 3652 3653 2 0 0 3826 3808 0 7477

##

from stompy.grid import unstructured_grid
import six
six.moves.reload_module(unstructured_grid)

g=unstructured_grid.UnstructuredGrid.read_suntans('/opt/sfb_ocean/suntans/runs/bay003_single',
                                                  subdomain=0)

## 

dj=np.loadtxt("/opt/sfb_ocean/suntans/runs/bay003_single/depths.dat-edge")

##

plt.figure(1).clf()
ecoll=g.plot_edges(values=dj[:,2],lw=2.,cmap='jet')
plt.colorbar(ecoll)

# zoom=(581351.361597184, 601055.5837273861, 4026195.258026521, 4045502.791969259)
# g.plot_edges(clip=zoom,
#              labeler=lambda i,r: str(r['Nke']) )
# g.plot_cells(clip=zoom,
#              labeler=lambda i,r: str(r['dv']) )

