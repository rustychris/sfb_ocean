from stompy.grid import unstructured_grid

##

from stompy.grid import triangulate_hole
six.moves.reload_module(triangulate_hole)

##
#g=unstructured_grid.UnstructuredGrid.from_ugrid('suisun_main-v05-edit29.nc')
fn_in='splice-merge-05.nc'
g=unstructured_grid.UnstructuredGrid.from_ugrid(fn_in)

## 
fills=[ ('Grizzly',[585405., 4219599.]),
        ("Port Chicago",[583217, 4212566.]),
        ("Pittsburg",[589946, 4211895.]),
        ("S of Ryer",[582176, 4214095.]),
        ("SE of Ryer",[587000, 4213895.]),
        ("Honker", [593774., 4214007]),
        ("Ryer",[583000, 4216227]),
        ("Chipps",[599851., 4212025]),
        # ("RyerEast",[587869, 4214920])
]

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g.plot_edges(ax=ax,lw=0.5,color='k')
for name,seed in fills:
    ax.text(seed[0],seed[1],name)
    ax.plot( [seed[0]],[seed[1]],'go')
    

## 
gnew=g
for name,seed in fills:
    print(name)
    seed=np.array(seed)
    res=triangulate_hole.triangulate_hole(gnew,seed,max_nodes=50000)
    if isinstance(res,front.AdvancingTriangles):
        ax.cla()
        g.plot_edges(color='0.5',lw=0.5,ax=ax)
        AT=res
        AT.grid.plot_edges(color='k',lw=0.75,ax=ax)
        AT.choose_site().plot()
        break
    gnew=res
else:
    plt.figure(1).clf()
    fig,ax=plt.subplots(1,1,num=1)
    gnew.plot_edges(ax=ax,lw=0.5,color='k')
    for name,seed in fills:
        ax.text(seed[0],seed[1],name)
        ax.plot( [seed[0]],[seed[1]],'go')

##

gnew.renumber()
gnew.write_ugrid(fn_in.replace('.nc','-filled.nc'))
