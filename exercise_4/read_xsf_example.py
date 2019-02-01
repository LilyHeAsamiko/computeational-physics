from numpy import *
from spline_class import *
import numpy.linalg as LA

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
    B = 2*np.pi*np.eye(3)/A
    print('reciprocal lattice:')
    print(B)
    N = sum(sum(sum(rho[0:-1,0:-1,0:1])))
    print('electron numbers:%d' % N)
    print('--------------------------')
    V = np.abs(LA.det(A))
    print('number of electrons in the simulation cell')
    
def prob3():
    line = (linspace(0.1,4.45,500),linspace(0.1,4.45,500),linspace(0.8528,0.8528,500))    
    dlattice = 0.0*lattice
    for i in range(3):
        dlattice[i,:] = lattice[i,:]/(grid[i]-1)
    spl1d=spline(x=x,f=lattice,dims=1)

    figure()
    # function
    plot(xx,spl1d.eval1d(xx))
    title('function')

def main():
    filename1 = 'dft_chargedensity1.xsf'
    filename2 = 'dft_chargedensity2.xsf'
    rho1, lattice1, grid1, shift1 = read_example_xsf_density(filename1)
    rho2, lattice2, grid2, shift2 = read_example_xsf_density(filename2)
    prob2(rho1, lattice1, grid1)
    prob2(rho2, lattice2, grid2)
    
if __name__=="__main__":
    main()



