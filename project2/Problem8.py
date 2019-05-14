# -*- coding: utf-8 -*-
"""
Created on Sun May  5 08:02:21 2019

@author: user
"""
from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os

def hartree_potential(ns,x,y):
    """ 
    Hartree potential using 'Rieman' integration 
    - assuming linear grid
    """
    Vhartree=dot([0.0,0.0],ns)
    dx=(x[1]-x[0])
    dy=(y[1]-y[0])
    for i in range(len(x)):
            for ij in range(len(x)):
                r = [x[i],y[ij]]
                for j in range(len(x)):
                    for ji in range(len(x)):
                        rp = [x[j],y[ji]]
                        Vhartree[i]+=dot(ns[i],ee_potential(r-rp))
    return dot(Vhartree,[dx,dy])

def ee_potential(x,y):
    global ee_coef
    """ 2D electron-electron interaction """
    return ee_coef[0]/sqrt([x,y]**2+ee_coef[1])

def ext_potential(x,y,m=1.0,omega=1.0):
    """ 2D harmonic quantum dot """
    return 0.5*m*omega**2*(x**2+y**2)

def density(psis):
    ns=zeros((len(psis[0]),len(psis[0])))
    for i in range(len(psis)):
        ns+=abs((psis[i])**2)
    return ns
    
def initialize_density(x,y,dx,dy,normalization=1):
    rho=exp(-(x**2+y**2))
    A=dot(sum(rho[:-1,:-1]),[dx,dy])
    return normalization/dot(A,rho)

def check_convergence(Vold,Vnew,threshold):
    difference_ = amax(abs(Vold-Vnew))
    print('  Convergence check:', difference_)
    converged=False
    if difference_ <threshold:
        converged=True
    return converged

def diagonal_energy(T,orbitals,dx,dy):
    """ 
    Calculate diagonal energy
    (using Rieman sum)
    """
    Tt=sp.csr_matrix(T)
    E_diag=[0.0,0.0]
    for i in range(len(orbitals)):
        evec=orbitals[i]
        E_diag+=dot(evec.conj().T,Tt.dot(evec))
    return dot(E_diag,[dx,dy])

def offdiag_potential_energy(orbitals,x,y):
    """ 
    Calculate off-diagonal energy
    (using Rieman sum)
    """
    U = [0.0,0.0]
    dx=x[1]-x[0]
    dy=y[1]-y[0]    
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            for i1 in range(len(x)):
                for j1 in range(len(x)):
                    U+=dot(dot(abs(orbitals[i][i1,i1])**2,abs(orbitals[j][j1,j1])**2),ee_potential([x[i1]-x[j1],y[i1]-y[j1]]))
    return dot(U,[dx,dy]**2)

def offdiag_potential_energy_exchange(orbitals,x,y,spin):
    """
    Calculate off-diagonal energy
    (using Rieman sum)
    """
    U = [0.0,0.0]
    dx=x[1]-x[0]
    dy=y[1]-y[0]    
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            for i1 in range(len(x)):
                for j1 in range(len(x)):
                    if(spin[i] == spin[j]):
                        U+= dot(dot(np.conjugate(dot(orbitals[i][i1,i1])),np.conjugate(dot(orbitals[j][j1,j1],orbitals[i][j1,j1])),orbitals[j][i1,i1]),ee_potential([x[i1]-x[j1],y[i1]-y[j1]]))
    return U*dx**2


def save_ns_in_ascii(ns,filename):
    s=shape(ns)
    f=open(filename+'.txt','w')
    for i in range(s[0]):
            for j in range(s[0]):
                f.write('{0:12.8f}\n'.format(ns[i,j]))
    f.close()
    f=open(filename+'_shape.txt','w')
    f.write('{0:5}'.format(s[0]))
    f.close()
    
def load_ns_from_ascii(filename):
    f=open(filename+'_shape.txt','r')
    for line in f:
        s=array(line.split(),dtype=int)
    f.close()
    ns=zeros((s[0],s[0]))
    d=loadtxt(filename+'.txt')
    k=0
    for i in range(s[0]):
        for j in range(s[0]):
            ns[i,j]=d[k,k]
            k+=1
    return ns

def save_data_to_hdf5_file(fname,orbitals,density,N_e,occ,grid,ee_coefs):
    return

def calculate_SIC(orbitals,x):
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC



def normalize_orbital(evec,dx,dy):
    return evec/sqrt(dot*(sum(abs(evec)**2),[dx,dy]))

def kinetic_hamiltonian(x, y):
    grid_size = [x.shape[0],y.shape[0]]
    dx = x[1] - x[0]
    dy = y[1] - y[0]    
    dx2 = dx**2
    dy2 = dy**2
    
    H0 = sp.diags(
        [
            -0.5 / [dx2 * np.ones(grid_size - 1),dy2 * np.ones(grid_size - 1)],
            1.0 / [dx2 * np.ones(grid_size),dy2 * np.ones(grid_size - 1)],
            -0.5 / dx2 * np.ones(grid_size - 1),dy2 * np.ones(grid_size - 1)]
        [1.0 / [dx2 * np.ones(grid_size - 1),dy2 * np.ones(grid_size - 1)],
            -0.5  / [dx2 * np.ones(grid_size),dy2 * np.ones(grid_size - 1)],
          1.0 / dx2 * np.ones(grid_size - 1),dy2 * np.ones(grid_size - 1)],
        [-0.5 / [dx2 * np.ones(grid_size - 1),dy2 * np.ones(grid_size - 1)],
            1.0 / [dx2 * np.ones(grid_size),dy2 * np.ones(grid_size - 1)],
            -0.5 /[ dx2 * np.ones(grid_size - 1),dy2 * np.ones(grid_size - 1)]],
        [-1, 0, 1],[0,-1,0],[1, 0, -1])
    return H0

def exchange_potential(x,y,orbitals,ns,spin,i):
    Vhartree =dot([0.0,0.0] ,ns)
    dx = (x[1] - x[0])
    dy = (y[1] - y[0])
    for i in range(len(x)):
        r = [x[i],y[i]]
        for j in range(len(x)):
            rp = [x[j],y[j]]
            Vhartree[i,j] += hartee_density(orbitals,i,j,spin,i) * ee_potential(r - rp)
    return Vhartree * [dx,dy]

def hartee_density(orbital1,xi ,yi,spin,i):
    sum = 0
    for n in range(size(spin)):
        if (spin[i] == spin[j]):
            sum += np.conjugate(dot(dot(orbital1[j][x[xi],y[xi]]),orbital1[i][x[xj],y[xj]]),orbital1[j][x[xi],y[xi]])/orbital1[i][x[xj],y[xj]]
    return sum

def calculate_SIC_exchange(orbitals,x,y,ns,spin):
    V_SIC = []
    for j in range(len(orbitals)):
        V_SIC.append(exchange_potential([x,y],orbitals,ns,spin,j))
    return V_SIC

def main():
    global ee_coef
    # e-e potential parameters [strenght, smoothness]
    ee_coef = [1.0, 1.0]
    # number of electrons
    N_e = 6
    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    occ = [0,0,1,1,2,2]
    spin = [0,1,0,1,0,1]

    #occ = [0,0,1,1]
    #spin = [0,1,0,1]

    # grid
    X=linspace(-4,4,120)
    Y=linspace(-4,4,120)
    [x,y] = meshgrid(X,Y)
    # threshold
    threshold=1.0e-4
    # mixing value
    mix=0.2
    # maximum number of iterations
    maxiters = 100



    dx = X[1]-X[0]
    dy = Y[1]-Y[0]    
    T = kinetic_hamiltonian(x, y)
    Vext = ext_potential(x, y)

    # READ in density / orbitals / etc.
    if os.path.isfile('density.txt'):
        ns=load_ns_from_ascii('density')
    else:
        ns=initialize_density(x,y,dx,dy,N_e)

    print('Density integral        ', sum(dot(ns[:-1,:-1]),[dx,dy]))
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x,y)
    VSIC=[]
    for i in range(N_e):
        VSIC.append(dot(ns,[0.0,0,0]))
    Veff=sp.diags(Vext+Vhartree,[0,0]) #add here v_i^X
    H=T+Veff
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],[0,0]), k=N_e, which='SR')
            eigs=real(eigs)
            evecs=real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],dx)
            orbitals.append(evecs[:,occ[i]])
        Veff_old = Veff
        ns=density(orbitals)
        Vhartree=hartree_potential(ns,x,y)
        VSIC=calculate_SIC(orbitals,x,y)
        Veff_new=sp.diags(Vext+Vhartree,[0,0])
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            Veff=([1.0,1.0]-mix)*Veff_new+mix*Veff_old
            H = T+Veff

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x,y)
    E_kin = diagonal_energy(T,orbitals,dx,dy)
    E_pot = diagonal_energy(sp.diags(Vext,[0,0]),orbitals,dx,dy) + off
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', sum(dot(ns[:-1,:-1]),[dx,dy]))

    # WRITE OUT density / orbitals / energetics / etc.
    #save_ns_in_ascii(ns,'density')
    ns2=initialize_density(x,y,dx,dy,N_e)
    print('\nCalculating initial state')
    Vhartree = hartree_potential(ns2, x,y)
    VSIC = []
    for i in range(N_e):
        VSIC.append(ns2 * [0.0,0.0])
    Veff = sp.diags(Vext + Vhartree, [0,0])  # add here v_i^X
    H = T + Veff
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals = []
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i + 1)
            eigs, evecs = sla.eigs(H + sp.diags(VSIC[i], 0), k=N_e, which='SR')
            eigs = real(eigs)
            evecs = real(evecs)
            print('    eigenvalues', eigs)
            evecs[:, occ[i]] = normalize_orbital(evecs[:, occ[i]], dx)
            orbitals.append(evecs[:, occ[i]])
        Veff_old = Veff
        ns2 = density(orbitals)
        Vhartree = hartree_potential(ns2, x,y)
        VSIC = calculate_SIC_exchange(orbitals, x,y,ns2,spin)
        Veff_new = sp.diags(Vext + Vhartree, 0)
        if check_convergence(Veff_old, Veff_new, threshold):
            break
        else:
            Veff = ([1.0,1.0] - mix) * Veff_new + mix * Veff_old
            H = T + Veff

    print('\n\n')

    off2 = offdiag_potential_energy_exchange(orbitals, x,y,spin)
    off = offdiag_potential_energy(orbitals,x,y)
    E_kin2 = diagonal_energy(T, orbitals, dx,dy)
    E_pot2 = diagonal_energy(sp.diags(Vext, 0), orbitals, dx,dy) + off - off2
    E_tot2 = E_kin2 + E_pot2
    print('Total energy     ', E_tot2)
    print('Kinetic energy   ', E_kin2)
    print('Potential energy ', E_pot2)
    print('\nDensity integral ', sum(dot(ns2[:-1,:-1]) ,[dx,dy]))









    contourf([x,y], abs(ns))
    contourf([x,y],abs(ns2))
    xlabel(r'$x$ (a.u.)')
    ylabel(r'$n(x)$ (1/a.u.)')
    title('N-electron density for N={0}'.format(N_e))
    text([0,0], [0.25,0.25], r'$E_{{tot}}= {:.4f}$'.format(E_tot))
    text([0,0], [0.32,0.32], r'$E_{{kin}}= {:.4f}$'.format(E_kin))
    text([0,0], [0.39,0.39],r'$E_{{pot}}= {:.4f}$'.format(E_pot))
    text([0,0], [0.46,0.46], r'$E_{{tot fock}}= {:.4f}$'.format(E_tot2))
    text([0,0], [0.53,0.53], r'$E_{{kin fock}}= {:.4f}$'.format(E_kin2))
    text([0,0], [0.60,0.6], r'$E_{{pot fock}}= {:.4f}$'.format(E_pot2))
    legend()
    show()

if __name__=="__main__":
    main()
