
## 
nproc=16

X=[]
Y=[]
nu=[]

for proc in range(nproc):
    ds=xr.open_dataset('runs/medium_09/DFM_OUTPUT_medium_09/medium_09_%04d_map.nc'%proc)
    X.append(ds.FlowLink_xu.values)
    Y.append(ds.FlowLink_yu.values)
    viu=ds.viu.isel(time=70,laydim=0).values
    nu.append(viu)

##

x=np.concatenate(X)
y=np.concatenate(Y)
all_nu=np.concatenate(nu)

##

plt.figure(1).clf()
scat=plt.scatter( x,y,30,all_nu )
plt.axis('equal')
