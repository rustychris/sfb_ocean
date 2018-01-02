matrix_b=np.loadtxt('runs/short_test_06/matrix.m',skiprows=44120,comments=';')

# it's diagonal
assert np.all( matrix_b[:,0] == matrix_b[:,1] )

##

cell_vals=np.ones(g.Ncells(),'f8')*np.nan

cell_idxs=matrix_b[:,0].astype('i4') - 1
cell_vals[cell_idxs] = matrix_b[:,2]

present=np.zeros(g.Ncells(),'b1')
present[cell_idxs]=True # fortran numbering!

valid=np.isfinite(cell_vals)
                 
##

plt.clf()
g.plot_cells(mask=~present,color='0.7') # not even part of the matrix
g.plot_cells(mask=valid,values=cell_vals) # part of the matrix and have finite value
g.plot_cells(color='r',mask=present & (~valid) ) # the troublemakers


##

# that shows that about half the cells are specified, in a roughly checkerboard
# pattern.
# and the NaN's arise from the lower left corner, which I

# How about a run with 1 second time steps, outputting a map every second?
# but I can't see matrix.m that fast.

# 
