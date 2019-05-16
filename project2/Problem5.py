# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:39:08 2019

@author: LilyHeAsamiko
"""

from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf

#initiate the walker
class Walker:
    def __init__(self,*args,**kwargs):
        self.spin = kwargs['spin']
        self.nearest_neighbors = kwargs['nn']
        self.sys_dim = kwargs['dim']
        self.coords = kwargs['coords']

    def w_copy(self):
        return Walker(spin=self.spin.copy(),
                      nn=self.nearest_neighbors.copy(),
                      dim=self.sys_dim,
                      coords=self.coords.copy())
    
# calculate the energy
def Energy(Walkers):
    E = 0.0
    J = 4.0
    for i in range(len(Walkers)):
        for k in range(len(Walkers[i].nearest_neighbors)):
            j = Walkers[i].nearest_neighbors[k]
            E += -J*Walkers[i].spin*Walkers[j].spin
    return E/2

# calculate the size energy with J1 and J2 
def site_Energy(Walkers,Walker):
    E = 0.0
    J1 = 4.0
    J2 = 1.0
    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E += -J1*Walker.spin*Walkers[j].spin-J2*Walker.spin*Walkers[j].spin
    return E

# calculate the magnetization
def magnetization(Walkers):
    m = 0.0
    for i in range(len(Walkers)):
        m += Walkers[i].spin
    return m

# ising model
def ising(Nblocks,Niters,Walkers,beta):
    #length of the Walkers
    M = len(Walkers)
    #initialize the Energy
    Eb = zeros((Nblocks,))
    #initialize the square of the Energy
    Eb2 = zeros((Nblocks,))
    #initialize the magnetization
    mb = zeros((Nblocks,))
    #initialize the square of the magnetization
    mb2 = zeros((Nblocks,))
    #initialize the count
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    #observables interval
    obs_interval = 5 #H,He,Li,Be,B
    for i in range(Nblocks):
        #count
        EbCount = 0
        #iter in Niters steps
        for j in range(Niters):
            site = int(random.rand()*M)
            # old spin
            s_old = 1.0*Walkers[site].spin
            # old energy
            E_old = site_Energy(Walkers,Walkers[site])
            #choose the spin 
            if random.rand()>0.5:
                s_new = 0.5
            else:
                s_new = -0.5
            # update into new spin    
            Walkers[site].spin = 1.0*s_new
            # update into new energy
            E_new = site_Energy(Walkers,Walkers[site])
            # difference of E 
            deltaE = E_new-E_old
            q_s_sp = exp(-beta*deltaE)
            A_s_sp = min(1.0,q_s_sp)
            # configuration 
            if (A_s_sp > random.rand()):
                Accept[i] += 1.0
            else:
                Walkers[site].spin=1.0*s_old
            AccCount[i] += 1

            if j % obs_interval == 0:
                E_tot = Energy(Walkers)
                Eb[i] += E_tot
                Eb2[i] += E_tot**2
                mag = magnetization(Walkers)
                mb[i] += mag
                mb2[i] += mag**2
                EbCount += 1
        # calculate the energy and accuracy    
        Eb[i] /= EbCount
        Eb2[i] /= EbCount
        mb[i] /= EbCount
        mb2[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, Eb2, mb, mb2, Accept


def main():
    Walkers=[]
   
    dim = 2
#    grid_side = 10
    grid_sides = [4,8,16,32] 
    Cv = zeros((4,1))
    E_tot = zeros((4,1))
    Mag = zeros((4,1))
    Chi = zeros((4,1))
    M = zeros((4,1))
    for k in range(0,4):
        grid_side = grid_sides[k]
        grid_size = grid_side**dim
        
        # Ising model nearest neighbors only
        mapping = np.zeros((grid_side,grid_side),dtype=int)
        inv_map = []
        ii = 0
        for i in range(grid_side):
            for j in range(grid_side):
                mapping[i,j]=ii
                inv_map.append([i,j])
                ii += 1
     
        for i in range(grid_side):
            for j in range(grid_side):
                # nearest neibors
                j1=mapping[i,(j-1) % grid_side]
                j2=mapping[i,(j+1) % grid_side]
                i1=mapping[(i-1) % grid_side,j]
                i2=mapping[(i+1) % grid_side,j]
                # next nearest neibors 
                j3=mapping[i,(j-2) % grid_side]
                j4=mapping[i,(j+2) % grid_side]
                i3=mapping[(i-2) % grid_side,j]
                i4=mapping[(i+2) % grid_side,j]
                Walkers.append(Walker(spin=0.5,
                                      #nn = [j1,j2,i1,i2],
                                      nn=[j1,j2,i1,i2,j3,j4,i3,i4],
                                      dim=dim,
                                      coords = [i,j]))
        
        Nblocks = 100
        Niters = 1000
        #tamperature from 0.5 to 6
#        T = np.linspace(0.5,8,60)
        Edata = []
        Edata2 = []
        Mdata = []
        Mdata2 = []

#        beta = zeros((len(T),1))
        eq = 20
#        for i in range(len(T)):
        T = 2.27 
        beta = 1.0/T
        
        Walkers, Eb, Eb2, mb, mb2, Acc = ising(Nblocks,Niters,Walkers,beta)
            
        E_and_err = array([mean(Eb[eq:]), std(Eb[eq:])/sqrt(len(Eb[eq:]))])
        E2_and_err = array([mean(Eb2[eq:]), std(Eb2[eq:])/sqrt(len(Eb2[eq:]))])
        Edata.append(E_and_err)
        Edata2.append(E2_and_err)
        M_and_err = array([mean(mb[eq:]), std(mb[eq:])/sqrt(len(mb[eq:]))])
        M2_and_err = array([mean(mb2[eq:]), std(mb2[eq:])/sqrt(len(mb2[eq:]))])
        Mdata.append(M_and_err)
        Mdata2.append(M2_and_err)
        
        Edata=array(Edata)
        Edata2=array(Edata2)
        Mdata=array(Mdata)
        Mdata2=array(Mdata2)

        # results plot(observables: ENERGY, HEAT CAPACITY, MAGNETIZATION, SUSCEPTIBILITY)        
        cv = (Edata2[:,0]-Edata[:,0]**2)/T**2/grid_size
        e_tot = Edata[:,0]/grid_size
        mag = Mdata[:,0]/grid_size
        chi = (Mdata2[:,0]-Mdata[:,0]**2)/T**2/grid_size
        m = len(Walkers) 
    
        print('Cv:',cv)
        print('E_tot:',e_tot)
        print('Mag:',mag)
        print('chi:',chi)
        print('M:',m)
        Cv[k] = cv  
        E_tot[k] = e_tot
        Mag[k] = mag
        Chi[k] = chi
        M[k] = m
        # results plot(observables: ENERGY, HEAT CAPACITY, MAGNETIZATION, SUSCEPTIBILITY)
        #   figure()   
        #    Cv = (Edata2[:,0]-Edata[:,0]**2)/T**2/grid_size
        #    plot(T,Cv,'-o')
        #    plot([2.27, 2.27],[0.0, 1.03*amax(Cv)],'k--')
        #    ylabel('Heat Capacity with grid_size: %s' %(grid_size))
        #    xlabel('Temperature')
        #    savefig('Heat Capacity with Temperature second(grid_size_10).pdf', mpi = 200)
        #    # Notice: T_c_exact = 2.27
        #
        #    figure()
        #    errorbar(T,Edata[:,0]/grid_size,Edata[:,1]/grid_size)
        #    ylabel('Energy with Temperature second(grid_size_10)')
        #    xlabel('Temperature')
        #    savefig('Energy with Temperature second(grid_size_10).pdf', mpi = 200)
        #        
        #    figure()
        #    errorbar(T,Mdata[:,0]/grid_size,Mdata[:,1]/grid_size)
        #    ylabel('Magnetization with Temperature second(grid_size_10)')
        #    xlabel('Temperature')
        #    savefig('Magnetization with Temperature second(grid_size_10).pdf', mpi = 200)
        #    figure()
        #    chi = (Mdata2[:,0]-Mdata[:,0]**2)/T**2/grid_size
        #    plot(T,chi,'-o')
        #    plot([2.27, 2.27],[0.0, 1.03*amax(chi)],'k--')
        #    ylabel('Susceptibility with Temperature second(grid_size_10)')
        #    xlabel('Temperature')
        #    savefig('Susceptibility with Temperature second(grid_size_10).pdf', mpi = 200) 
    
    figure()
    plot(grid_sides,Cv,'-o')
    ylabel('Heat Capacity with grid_size')
    xlabel('grid size')
    savefig('Heat Capacity with grid_size.pdf', mpi = 200)
            # Notice: T_c_exact = 2.27
        
    figure()
    plot(grid_sides,E_tot,'-o')
    ylabel('Energy with grid_size')
    xlabel('grid size')
    savefig('Temperature with grid_size.pdf', mpi = 200)
                
    figure()
    plot(grid_sides,Mag,'-o')
    ylabel('Magnetization with grid_size')
    xlabel('grid size')    
#    savefig('Magnetization with grid_size.pdf', mpi = 200)            figure()
    savefig('Magnetization with grid_size.pdf', mpi = 200) 
                
    figure()
    plot(grid_sides,Chi,'-o')
    ylabel('Susceptibility with grid_size')
    xlabel('grid size')
    savefig('Susceptibility with grid_size.pdf', mpi = 200)          
    
    #calculate transition tamperature

    figure()
    Tc = 1.0/(3.16681*10**(-6)*M)
#    print('The transition tamperature with grid_size: %s' %(grid_size))
    plot(grid_sides,Tc,'-o')
    ylabel('The transition tamperature with grid_size' )
    xlabel('grid size')
    savefig('The transition tamperature.pdf', mpi = 200)
    show()
        
"""""""""""""""""""""""
For spin models, we have a finite d-dimensional lattice of  sites.
 But only get a true phase transition (i.e., divergence) when  .
 For a finite system, get rounded peaks rather than divergences.
 The peaks narrow and increase in height as L is increased, 
 and the location of the peak shifts slightly.
 
"""""""""""""""""""""""

if __name__=="__main__":
    main()