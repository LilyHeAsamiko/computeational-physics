"""
FYS-4096 Computational physics 

1. Add code to function 'largest_eig'
- use the power method to obtain 
  the largest eigenvalue and the 
  corresponding eigenvector of the
  provided matrix

2. Compare the results with scipy's eigs
- this is provided, but you should use
  that to validating your power method
  implementation

"""


from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps


def largest_eigs(A,tol=1e-12):
    """
    Simple power method code needed here
    """
#    n = len(A)
    n = A.shape[0]
    x = np.random.rand(n, 1)
    x = x/np.linalg.norm(x)
    x_old =1.0*x
    eig_vector = A*x
    eig_value = mean(eig_vector)
    i = 1 

    while np.linalg.norm(A*eig_vector-eig_value*eig_vector,2)/n >= tol:
        eig_vector = A**i*eig_vector/np.linalg.norm(A**i*eig_vector,2) 
        eig_value = mean(eig_vector)
        eig_vector_old = 1.0*eig_vector
        i += 1

#        eig_vector = A**(i-1)*x/np.abs(A**(i-1)*x)
        print('eigen value: -%f with iteration: %d '% (eig_value, i-1))
        print('difference %f' % (np.linalg.norm(A*eig_vector-eig_value*eig_vector,2)/n))
    return eig_value, eig_vector

def main():
    grid = np.linspace(-5,5,100)
    grid_size = grid.shape[0]
    dx = grid[1]-grid[0]
    dx2 = dx*dx
    
    # make test matrix
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size) - 1.0/(abs(grid)+1.),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    
    # use scipy to calculate the largest eigenvalue
    # and corresponding vector
    eigs, evecs = sla.eigsh(H0, k=1, which='LA')
    
    # use your power method to calculate the same
    l,vec=largest_eigs(H0)
    
    # see how they compare
    print('largest_eig estimate: ', l)
    print('scipy eigsh estimate: ', eigs)
    
    psi0=evecs[:,0]
    norm_const=simps(abs(psi0)**2,x=grid)
    psi0=psi0/norm_const
    
    psi0_=vec[:]
    norm_const=simps(abs(psi0_)**2,x=grid)
    psi0_=psi0_/norm_const
    
    plt.plot(grid,abs(psi0)**2,label='scipy eig. vector squared')
    plt.plot(grid,abs(psi0_)**2,'r--',label='largest_eig vector squared')
    legend(loc=0)
    show()


if __name__=="__main__":
    main()
