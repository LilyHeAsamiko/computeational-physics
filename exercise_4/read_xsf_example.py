from numpy import *
from spline_class import *
import numpy.linalg as LA
from num_calculus import *
from scipy.integrate import simps
import matplotlib.pyplot as plt

def read_example_xsf_density(filename):
    lattice=[]
    density=[]
    grid=[]
    shift=[]
    i=0
    start_reading = False
    with open(filename, 'r') as f:
        for line in f:
            if "END_DATAGRID_3D" in line:
                start_reading = False
            if start_reading and i==1:
                grid=array(line.split(),dtype=int)
            if start_reading and i==2:
                shift.append(array(line.split(),dtype=float))
            if start_reading and i==3:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i==4:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i==5:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i>5:            
                density.extend(array(line.split(),dtype=float))
            if start_reading and i>0:
                i=i+1
            if "DATAGRID_3D_UNKNOWN" in line:
                start_reading = True
                i=1
    
    rho=zeros((grid[0],grid[1],grid[2]))
    ii=0
    for k in range(grid[2]):
        for j in range(grid[1]):        
            for i in range(grid[0]):
                rho[i,j,k]=density[ii]
                ii+=1

    # convert density to 1/Angstrom**3 from 1/Bohr**3
    a0=0.52917721067
    a03=a0*a0*a0
    rho/=a03
    return rho, array(lattice), grid, shift

def prob2(rho, lattice, grid):
#    x = grid
    A = lattice
    r = A*1.0
    aplha = LA.inv(A)*r
#    B = 2*np.pi*np.eye(3)/A
    print('reciprocal lattice:')
#    print(B)    
    print(A)
    V = LA.det(A)
    dV = 1.0*V
#wh
#    for i in range(3):
#        dV[i,:] = A[i]/(grid[i]-1)
#    spl1d=spline(x=V,f=dV,dims=1)
    print('number of electrons in the simulation cell')
    print('--------------------------') 
    N =sum(sum(sum(rho[0:-1,0:-1,0:1])))
    dN = sum(sum(sum(rho[0:-1,0:-1,0:1])))*dV
#    figure()
#    plot(dN, spl1d.eval1d(dN))
    print('electron numbers:%d' % N)
    
 
def prob3(rho, lattice, grid, shift, r0, r1): 
    inv_cell = array(LA.inv(transpose(lattice)))
    x = linspace(0,1.,grid[0])
    y = linspace(0,1.,grid[1])
    z = linspace(0,1.,grid[2])
    
    int_dx = simps(rho, x=x, axis=0)
    int_dxdy = simps(int_dx, x=y, axis=0)
    int_dxdydz = simps(int_dxdy, x=z)
    print(' Simpson int = ', int_dxdydz*abs(LA.det(lattice)))
    
    spl3d = spline(x=x, y=y, z=z, f=rho, dims=3)
    
    t = np.linspace(0, 1, 500)
    f = zeros(shape=shape(t))
    for i in range(len(t)):
        xx = inv_cell.dot(r0+t[i]*(r1-r0))
        f[i] = spl3d.eval3d(xx[0],xx[1],xx[2])
    figure()
    plot(t,f)
    plt.savefig('electron_density_%f.pdf'% r0[0],dpi=200)

    
    
def main():
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    print(grid)
    prob2(rho, lattice, grid)
    r0 = array([0.1, 0.1, 2.8528])
    r1 = array([4.45, 4.45, 2.8528])
    prob3(rho, lattice, grid, shift, r0, r1)
            

    
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    prob2(rho, lattice, grid)    
    r0 = array([-1.4466, 1.3073, 3.2115])
    r1 = array([1.4361, 3.1883, 1.3542])
    prob3(rho, lattice, grid, shift, r0, r1)
    r0 = array([2.9996, 2.1733, 2.1462])
    r1 = array([8.7516, 2.1733, 2.1462])
    prob3(rho, lattice, grid, shift, r0, r1)    
    
if __name__=="__main__":
    main()



