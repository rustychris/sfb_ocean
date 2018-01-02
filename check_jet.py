# Try to understand what causes one particular boundary cell to go crazy?

evil_pnt=[355383, 4.11931e+06]
# z=1.8474

evil_c=g.select_cells_nearest(evil_pnt)

##

evil_edges=g.cell_to_edges(evil_c)
bad_edge = np.nonzero( g.edges['cells'][evil_edges] < 0 )[0]
bad_j=evil_edges[bad_edge[0]] # 99

##

bad_pli=dio.read_pli('runs/short_test_06/oce00099_rmn.pli')

##

# Is there anything different about the timeseries
# nope, they nicely line next to each other
g.plot_edges( clip=ax.axis(), labeler=lambda i,r: str(i))

j_nbr_n=102
j_nbr_s=96

##

tim99 =np.loadtxt('/opt/data/delft/sfb_ocean/sfb_ocean/runs/short_test_06/oce00099_rmn_0001.tim')
tim102=np.loadtxt('/opt/data/delft/sfb_ocean/sfb_ocean/runs/short_test_06/oce00102_rmn_0001.tim')
tim96=np.loadtxt('/opt/data/delft/sfb_ocean/sfb_ocean/runs/short_test_06/oce00096_rmn_0001.tim')

plt.figure(25).clf()
plt.plot( tim99[:,0], tim99[:,1],label='99')
plt.plot(tim102[:,0],tim102[:,1],label='102')
plt.plot( tim96[:,0], tim96[:,1],label='96')
plt.legend()

##


tim99 =np.loadtxt('/opt/data/delft/sfb_ocean/sfb_ocean/runs/short_test_06/oce00099_rmn_0001.tim')
tim102=np.loadtxt('/opt/data/delft/sfb_ocean/sfb_ocean/runs/short_test_06/oce00102_rmn_0001.tim')
tim96=np.loadtxt('/opt/data/delft/sfb_ocean/sfb_ocean/runs/short_test_06/oce00096_rmn_0001.tim')

## 



plt.figure(24).clf()
ax=plt.gca()

g.plot_edges(ax=ax)

g.plot_cells(mask=[evil_c],ax=ax)

segs=bad_pli[0][1]

ax.plot(segs[:,0],segs[:,1],'r-')
##
