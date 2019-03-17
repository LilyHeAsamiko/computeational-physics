#! /usr/bin/env python3

from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os

def nHFi(orbitals, spin, ix, ix2, i):
    n = zeros(len(orbitals),)
    for j in range(len(orbitals)):
        if spin[i] == spin[j]:
            n[j] = spin[i]*spin[j]*orbitals[j][ix2].conj()*orbitals[i][ix2]*orbitals[j][ix]/orbitals[i][ix];
    return sum(n)

def x_potentials(orbitals, x, spin, i):
        orbitals = np.array(orbitals)
        Vx=0.0*orbitals[0]
        dx =x[1]-x[0]
        for ix in range(len(x)):
            r = x[ix]
            for ix2 in range(len(x)):
                rp = x[ix2]
                Vx[ix]+=nHFi(orbitals, spin, ix, ix2, i)*ee_potential(r-rp)
        return -Vx*dx    

def hartree_potential(ns,x):
    """ 
    Hartree potential using 'Rieman' integration 
    - assuming linear grid
    """
    Vhartree=0.0*ns
    dx=(x[1]-x[0])
    for ix in range(len(x)):
        r = x[ix]
        for ix2 in range(len(x)):
            rp = x[ix2]
            Vhartree[ix]+=ns[ix2]*ee_potential(r-rp)
    return Vhartree*dx

def ee_potential(x):
    global ee_coef
    """ 1D electron-electron interaction """
    return ee_coef[0]/sqrt(x**2+ee_coef[1])

def ext_potential(x,m=1.0,omega=1.0):
    """ 1D harmonic quantum dot """
    return 0.5*m*omega**2*x**2

def density(psis):
    ns=zeros((len(psis[0]),))
    for i in range(len(psis)):
        ns+=abs(psis[i])**2
    return ns
    
def initialize_density(x,dx,normalization=1):
    rho=exp(-x**2)
    A=sum(rho[:-1])*dx
    return normalization/A*rho

def check_convergence(Vold,Vnew,threshold):
    difference_ = amax(abs(Vold-Vnew))
    print('  Convergence check:', difference_)
    converged=False
    if difference_ <threshold:
        converged=True
    return converged

def diagonal_energy(T,orbitals,dx):
    """ 
    Calculate diagonal energy
    (using Rieman sum)
    """
    Tt=sp.csr_matrix(T)
    E_diag=0.0
    for i in range(len(orbitals)):
        evec=orbitals[i]
        E_diag+=dot(evec.conj().T,Tt.dot(evec))
    return E_diag*dx

def offdiag_potential_energy(orbitals,x):
    """ 
    Calculate off-diagonal energy
    (using Rieman sum)
    """
    U = 0.0
    dx=x[1]-x[0]
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            for i1 in range(len(x)):
                for j1 in range(len(x)):
                    U+=abs(orbitals[i][i1])**2*abs(orbitals[j][j1])**2*ee_potential(x[i1]-x[j1])
    return U*dx**2

def save_ns_in_ascii(ns,filename):
    s=shape(ns)
    f=open(filename+'.txt','w')
    for ix in range(s[0]):
        f.write('{0:12.8f}\n'.format(ns[ix]))
    f.close()
    f=open(filename+'_shape.txt','w')
    f.write('{0:5}'.format(s[0]))
    f.close()
    
def load_ns_from_ascii(filename):
    f=open(filename+'_shape.txt','r')
    for line in f:
        s=array(line.split(),dtype=int)
    f.close()
    ns=zeros((s[0],))
    d=loadtxt(filename+'.txt')
    k=0
    for ix in range(s[0]):
        ns[ix]=d[k]
        k+=1
    return ns

def save_data_to_hdf5_file(fname,orbitals,density,N_e,occ,grid,ee_coefs):
    return

def calculate_SIC(orbitals,x):
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC

def calculate_SIC2(orbitals, x, spin, i):
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(x_potentials(orbitals,x, spin, i))
    return V_SIC
            
def normalize_orbital(evec,dx):
    return evec/sqrt((sum(abs(evec)**2)*dx))

def kinetic_hamiltonian(x):
    grid_size = x.shape[0]
    dx = x[1] - x[0]
    dx2 = dx**2
    
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    return H0

def main():
    global ee_coef
    # e-e potential parameters [strenght, smoothness]
    ee_coef = [1.0, 1.0]
    # number of electrons
    N_e = 4
    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    occ = [0,1,2,3]  
    #spin up is 0.5, down is -0.5
    spin = [0,1,0,1]
    # grid
    x=linspace(-4,4,120)
    # threshold
    threshold=1.0e-4
    # mixing value
    mix=0.2
    # maximum number of iterations
    maxiters = 100

    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    # READ in density / orbitals / etc.
    if os.path.isfile('density.txt'):
        ns=load_ns_from_ascii('density')
    else:
        ns=initialize_density(x,dx,N_e)

    print('Density integral', sum(ns[:-1])*dx)
    print(' -- should be close to', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)

    VSIC=[]
    VSIC2 = []
    for i in range(N_e):
        VSIC.append(ns*0.0)

    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('Calculating orbitals for electron', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR')
            eigs=real(eigs)
            evecs=real(evecs)
            print('eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],dx)
            orbitals.append(evecs[:,occ[i]])

        Veff_old = Veff
        ns=density(orbitals)
        Vhartree=hartree_potential(ns,x)
        VSIC=calculate_SIC(orbitals,x)
        VSIC2=calculate_SIC2(orbitals,x, spin, i)
        Veff_new=sp.diags(Vext+Vhartree,0)
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            Veff=(1.0-mix)*Veff_new+mix*Veff_old
            H = T+Veff
        
#    Vx = x_potentials(orbitals2,x, spin, i)
#    Veff2 = sp.diags(Vext+Vhartree+Vx,0)
#    H2 = T+Veff2
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals2=[]
        for i in range(N_e):
            print('Calculating orbitals for electrons again', i+1)
            eigs2, evecs2 = sla.eigs(H+sp.diags(VSIC2[i],0), k=N_e, which='SR')
            eigs=real(eigs2)
            evecs=real(evecs2)
            print('eigenvalues2', eigs2)
            evecs2[:,occ[i]]=normalize_orbital(evecs2[:,occ[i]],dx)
            orbitals2.append(evecs[:,occ[i]])
        
        Vx = x_potentials(orbitals2,x, spin, i)
        Veff2 = sp.diags(Vext+Vhartree+Vx,0)
        Veff_old2 = Veff2
        ns2=density(orbitals2)
        Vhartree2=hartree_potential(ns2,x)
        VSIC2=calculate_SIC(orbitals2,x)
        Veff_new2=sp.diags(Vext+Vhartree2+Vx,0)
        if check_convergence(Veff_old2,Veff_new2,threshold):
            break
        else:
            Veff2=(1.0-mix)*Veff_new2+mix*Veff_old2
            H = T+Veff2

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin = diagonal_energy(T,orbitals,dx)
    E_pot = diagonal_energy(sp.diags(Vext,0),orbitals,dx) + off
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', sum(ns[:-1])*dx)

    print('\n\n')
    off2 = offdiag_potential_energy(orbitals2,x)
    E_kin2 = diagonal_energy(T,orbitals2,dx)
    E_pot2 = diagonal_energy(sp.diags(Vext,0),orbitals2,dx) + off2
    E_tot2 = E_kin2 + E_pot2
    print('Total energy     ', E_tot2)
    print('Kinetic energy   ', E_kin2)
    print('Potential energy ', E_pot2) 
    print('\nDensity integral ', sum(ns2[:-1])*dx)

    # WRITE OUT density / orbitals / energetics / etc.
    #save_ns_in_ascii(ns,'density')
    
    plot(x,abs(ns),'r', x,abs(ns2),'b')
    savefig('integeral.pdf',dpi=200)
    show()

if __name__=="__main__":
    main()
