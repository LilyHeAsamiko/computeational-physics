# -*- coding: utf-8 -*-
"""
Created on Fri May  3 05:41:16 2019

@author: user
"""
from numpy import *
import numpy as np
from matplotlib.pyplot import * 
from scipy.integrate import simps
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os


def hartree_potential(N_e, ns, x):
    """
    Hartree potential using 'Riemann' integration
    - assuming linear grid
    So, interaction between electrons by integrating electron density.
    """
    Vhartree = 0.0 * ns
    dx = x[1] - x[0]
    for ix in range(len(x)):
        r = x[ix]
        for ix2 in range(len(x)):
            rp = x[ix2]
            Vhartree[ix] += ns[ix2] * Coulumb_potential(N_e, x)
    return Vhartree * dx

def ee_potential(x):
    global ee_coef
    """ 1D electron-electron interaction """
    V_ee = ee_coef[0] / sqrt(x ** 2 + ee_coef[1])
    return V_ee

def Coulumb_potential(N_e, x):
    """
    Exchange potential of Hartree-Fock using Columb
    """
    U = 0.0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            U += -N_e**2 / sqrt((x[i] - x[j])**2+1)
    return U 

def exchange_potential(N_e, orbitals,spin,i,x):
    """ 
    Exchange potential of Hartree-Fock using 'Rieman' 
    integration 
    - assuming linear grid
    """
    Vix=0.0*orbitals[0]
    dx=(x[1]-x[0])
    for ix in range(len(x)):
        r = x[ix]
        for ix2 in range(len(x)):
            rp = x[ix2]
            Vix[ix]+=n_iHF(orbitals,spin,i,ix,ix2)* Coulumb_potential(N_e, x)
    return -Vix*dx

def ext_potential(x, m=1.0, omega=1.0):
    """ 1D harmonic quantum dot """
    return 0.5 * m * omega ** 2 * x ** 2


def density(orbitals):
    """ Calculates the complete electron density from column vectors of orbital data."""
    ns = zeros((len(orbitals[0]),))
    for i in range(len(orbitals)):
        ns += abs(orbitals[i]) ** 2
    return ns


def initialize_density(x, dx, normalization=1):
    rho = exp(-x ** 2)
    A = sum(rho[:-1]) * dx
    return normalization / A * rho


def check_convergence(Vold, Vnew, threshold):
    difference_ = amax(abs(Vold - Vnew))
    print('  Convergence check:', difference_)
    converged = False
    if difference_ < threshold:
        converged = True
    return converged


def diagonal_energy(T, orbitals, dx):
    """
    Calculate diagonal energy
    (using Rieman sum)
    """
    Tt = sp.csr_matrix(T)
    E_diag = 0.0
    for i in range(len(orbitals)):
        evec = orbitals[i]
        E_diag += dot(evec.conj().T, Tt.dot(evec))
    return E_diag * dx


def offdiag_potential_energy(orbitals, x):
    """
    Calculate off-diagonal energy
    (using Rieman sum)
    """
    U = 0.0
    dx = x[1] - x[0]
    for i in range(len(orbitals) - 1):
        for j in range(i + 1, len(orbitals)):
            for i1 in range(len(x)):
                for j1 in range(len(x)):
                    U += abs(orbitals[i][i1]) ** 2 * abs(orbitals[j][j1]) ** 2 * ee_potential(x[i1] - x[j1])
    return U * dx ** 2


def exchange_potential_energy_HF(orbitals, x, spins):
    """
    Calculate off-diagonal energy
    (using Rieman sum)
    """
    U = 0.0
    dx = x[1] - x[0]
    for i in range(len(orbitals) - 1):
        for j in range(i + 1, len(orbitals)):
            for i1 in range(len(x)):
                for j1 in range(len(x)):
                    U += abs(orbitals[i][i1]) ** 2 * abs(orbitals[j][j1]) ** 2 * ee_potential(x[i1] - x[j1])
    return - U * dx ** 2


def save_data_to_hdf5_file(fname, orbitals, density, N_e, occ, grid, ee_coefs, Etot, Ekin, Epot):
    f = h5py.File(fname+'.hdf5', "w")
    gset = f.create_dataset("grid", data=grid, dtype='f')
    gset.attrs["info"] = '1D grid'

    oset = f.create_dataset("orbitals", shape=(len(grid), N_e), dtype='f')
    oset.attrs["info"] = '1D orbitals as (len(grid),N_electrons)'
    for i in range(len(orbitals)):
        oset[:, i] = orbitals[i]

    dset = f.create_dataset("density", data=density, dtype='f')
    dset.attrs["info"] = 'Electron densities on grid.'

    qset = f.create_dataset("N_e", data=N_e, dtype='i')
    qset.attrs["info"] = 'Number of electrons.'

    wset = f.create_dataset("occ", data=occ, dtype='i')
    wset.attrs["info"] = 'List of energy states occupied by corresponding electrons.'

    eset = f.create_dataset("ee_coefs", data=ee_coefs, dtype='f')
    eset.attrs["info"] = 'Coefficients for electron electron Coulombic interaction.'

    etset = f.create_dataset("Etot", data=Etot, dtype='f')
    etset.attrs["info"] = 'Total energy.'

    ekset = f.create_dataset("Ekin", data=Ekin, dtype='f')
    ekset.attrs["info"] = 'Kinetic energy.'

    epset = f.create_dataset("Epot", data=Epot, dtype='f')
    epset.attrs["info"] = 'Potential energy.'

    f.close()


def load_data_from_hdf5_file(fname):
    f = h5py.File(fname+ '.hdf5', "r")
    grid = array(f["grid"])
    orbs = array(f["orbitals"])
    orbitals = []
    for i in range(len(orbs[0, :])):
        orbitals.append(orbs[:, i])
    density = array(f["density"])
    N_e = array(f["N_e"])
    occ = array(f["occ"])
    ee_coefs = array(f["ee_coefs"])

    f.close()
    return orbitals, density, N_e, occ, grid, ee_coefs


def calculate_SIC(N_e, orbitals, x):
    """ Calculates the self interaction correction for all orbital vectors in list of orbitals."""
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(N_e, abs(orbitals[i]) ** 2, x))
    return V_SIC

def calculate_Vx(orbitals,x,spins):
    """ Calculates the self interaction correction for all orbital vectors in list of orbitals."""
    V_x = []
    for i in range(len(orbitals)):
        V_x.append(potential_x(orbitals[i], x, orbitals, spins[i], spins))
    return V_x

def density_HF(orbital, orbitals, spin, spins, ix, ix2):
    """ Calculates the Hartree-Fock electron density used for calculating
    correction potential v^x"""
    n = 0
    for i in range(len(orbitals)):
        if spins[i] == spin:
            n += conj(orbitals[i][ix2])*orbital[ix2]*orbitals[i][ix]/orbital[ix]
    return n


def potential_x(i_orbital, x, orbitals, i_spin, spins):
    """ Hartree-Fock potential. """
    dx = x[1] - x[0]
    V_i = zeros(shape(i_orbital))
    for ix in range(len(x)):
        r = x[ix]
        for ix2 in range(len(x)):
            rp = x[ix2]
            n_i = density_HF(i_orbital, orbitals, i_spin, spins, ix, ix2)
            V_i[ix] += -n_i * ee_potential(r-rp)
    return V_i*dx


def normalize_orbital(single_orb, dx):
    """ Normalize the orbital probability density to one """
    N_const = simps(abs(single_orb) ** 2, dx=dx)
    norm_orb = single_orb / np.sqrt(N_const)
    return norm_orb


def kinetic_hamiltonian(x):
    grid_size = x.shape[0]
    dx = x[1] - x[0]
    dx2 = dx ** 2

    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    return H0

def main():

    
    #cubic simulation cell with L= 10
    L = 10
    nmax = 1000
    lam = 0.5
    d = 1
    b = 0.001 #1/(1.38e-23*4)
    x = linspace(-L / 2, L / 2)
    R, Rp = meshgrid(x, x)

    rho0 = (4 * pi * lam * b) ** (-d) * exp(-(R - Rp) ** 2 / (4 * lam * b))
    f1 = plt.figure
    plt.contourf(R, Rp, rho0)
    title('Gaussian, beta = {}'.format(b))

    # 1
    n = arange(0, nmax)
    k = 2 * pi * n / L
    E = lam * k ** 2

    rho = zeros(R.shape)
    for i in range(len(x)):
        for j in range(len(x)):
            rho[i, j] = sum(L ** (-d / 2) * 2 * cos(-b * E * (R[i, j] - Rp[i, j])))

    ax = figure()
    contourf(R, Rp, rho)
    title(' n_max = {}, beta = {}'.format(nmax, b))

    # --- Setting up the system etc. ---
    global ee_coef
    # e-e potential parameters [strenght, smoothness]=[a,b]
    ee_coef = [1.0, 1.0]
    # number of electrons
    N_e = 5
    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    #B
    occ = [0,0,1,1,2] 
    spins = [1,0,1,0,1]  # This could be inferred from occ but this is way easier
    
    # grid
    x = np.linspace(-4, 4, 120)
    # threshold
    threshold = 1.0e-4
    # mixing value
    mix = 0.2 # alpha
    # maximum number of iterations
    maxiters = 100
    HF_run = False

    dx = x[1]-x[0]
    # kinetic_hamiltonian
    T = kinetic_hamiltonian(x)
#    Vext = ext_potential(x)
    Vext = Coulumb_potential(N_e, x)

    # READ in density / orbitals / etc.
    if os.path.isfile('density.txt'):
        ns=load_ns_from_ascii('density')
    else:
        ns=initialize_density(x,dx,N_e)

    print('Density integral        ', sum(ns[:-1])*dx)
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    #calculate hartree potential
    Vhartree=hartree_potential(N_e,ns,x)
    VSIC=[]
    for i in range(N_e):
        VSIC.append(ns*0.0)
        
    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff
    #calculate hartree energy with hartree fock
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR')
            eigs=real(eigs)
            evecs=real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],dx)
            orbitals.append(evecs[:,occ[i]])
        Veff_old = Veff
        ns=density(orbitals)
        Vhartree=hartree_potential(N_e,ns,x)
        if (HF_run):
            """Hartree-Fock"""
            VSIC=calculate_Vx(orbitals,spin,x)
        else:
            """Hartree"""
            VSIC=calculate_SIC(N_e, orbitals, x)
        Veff_new=sp.diags(Vext+Vhartree,0)
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            Veff=(1.0-mix)*Veff_new+mix*Veff_old
            H = T+Veff

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin = diagonal_energy(T,orbitals,dx)
    E_pot = diagonal_energy(sp.diags(Vext,0),orbitals,dx) + off
    if (HF_run):
        E_pot+=exchange_potential_energy_HF(orbitals,spin,x)
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', sum(ns[:-1])*dx)

    plot(x, abs(ns))
    xlabel(r'$x$ (a.u.)')
    ylabel(r'$n(x)$ (1/a.u.)')
    plt.title('B-electron density for N={5}'.format(N_e))
    plt.savefig('B.pdf',dpi=200)    
    grid(True)    
    text(-2, 0,
         "Total energy    {:.4f} \nKinetic energy    {:.4f} \nPotential energy  {:.4f}".format(E_tot, E_kin, E_pot),
         fontsize=12)
    show()

    N_e = 4
    occ = [0,0,1,1] 
    spins = [1,0,1,0]  # This could be inferred from occ but this is way easier
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs_1, evecs_1 = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR')
            eigs_1=real(eigs_1)
            evecs_1=real(evecs_1)
            print('    eigenvalues', eigs_1)
            evecs_1[:,occ[i]]=normalize_orbital(evecs_1[:,occ[i]],dx)
            orbitals_1.append(evecs_1[:,occ[i]])
        Veff_old_1 = Veff_1
        ns_1=density(orbitals_1)
        Vhartree_1=hartree_potential(N_e,ns_1,x)
        if (HF_run):
            """Hartree-Fock"""
            VSIC_1=calculate_vix(orbitals_1,spin,x)
        else:
            """Hartree"""
            VSIC_1=calculate_SIC(orbitals_1,x)
        Veff_new_1=sp.diags(Vext_1+Vhartree_1,0)
        if check_convergence(Veff_old_1,Veff_new_1,threshold_1):
            break
        else:
            Veff_1=(1.0-mix)*Veff_new_1+mix*Veff_old_1
            H_1 = T_1+Veff_1

    print('\n\n')
    off_1 = offdiag_potential_energy(orbitals_1,x)
    E_kin_1 = diagonal_energy(T_1,orbitals_1,dx)
    E_pot_1 = diagonal_energy(sp.diags(Vext,0),orbitals_1,dx) + off_1
    if (HF_run):
        E_pot_1+=exchange_potential_energy_HF(N_e, orbitals_1,spin,x)
    E_tot_1 = E_kin_1 + E_pot_1
    print('Total energy     ', E_tot_1)
    print('Kinetic energy   ', E_kin_1)
    print('Potential energy ', E_pot_1) 
    print('\nDensity integral ', sum(ns[:-1])*dx)
    print('Ionization energy    ', E_tot_1-E_tot)

if __name__ == "__main__":
    main()
