"""
Simple Monte Carlo for Ising model

Related to course FYS-4096 Computational Physics

Problem 1:
- Make the code to work, that is, include code to where it reads "# ADD"
- Comment the parts with "# ADD" and make any additional comments you 
  think could be useful for yourself later.
- Follow the assignment from ex11.pdf.

Problem 2:
- Add observables: heat capacity, magnetization, magnetic susceptibility
- Follow the assignment from ex11.pdf.

Problem 3:
- Look at the temperature effects and locate the phase transition temperature.
- Follow the assignment from ex11.pdf.

"""


from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf

def save_data_to_hdf5_file(fname,Walkers, Eb, Acc, Magb, C, chi):
    return

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
    
def Mag(Walkers):
    Mag = 0.0
    for  i in range(len(Walkers)):      
        Mag += Walkers[i].spin
    return Mag
    
def Energy(Walkers):
    E = 0.0
    J = 4.0 # given in units of k_B
    # ADD calculation of energy
    M = len(Walkers)
    for  i in range(len(Walkers)):      
        E += site_Energy(Walkers,Walkers[i])
    return E/2 

def site_Energy(Walkers,Walker):
    E = 0.0
    J = 4.0 # given in units of k_B(exchange constant)
    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E += -J*Walker.spin*Walkers[j].spin
    return E

def ising(Nblocks,Niters,Walkers,beta):
    M = len(Walkers)
    Eb = zeros((Nblocks,))
    Ec = zeros((Nblocks,))
    Magb = zeros((Nblocks,))
    Magc = zeros((Nblocks,))
    C = zeros((Nblocks,))
    chi = zeros((Nblocks,))
    s_old = zeros((Nblocks,))
    s_2 = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))

    obs_interval = 5
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            site = int(random.rand()*M)

            s_old = 1.0*Walkers[site].spin
 
            E_old = site_Energy(Walkers,Walkers[site])

            # ADD selection of new spin to variable s_new
            s_new = random.choice([0.5,-0.5])

            Walkers[site].spin = 1.0*s_new

            E_new = site_Energy(Walkers,Walkers[site])

            # ADD Metropolis Monte Carlo
            q_R_Rp = exp(-beta*(E_new-E_old)) #calculate the density
            A_RtoRp = min(1.0,q_R_Rp)
            if (A_RtoRp > random.rand()):
                Accept[i] += 1.0 #count correct terms
            else:
                Walkers[site].spin=1.0*s_old

            AccCount[i] += 1

            if j % obs_interval == 0:
                E_tot = Energy(Walkers)/M # energy per spin
                Mag_tot = Mag(Walkers)/M
                Eb[i] += E_tot
                Magc[i] += Mag_tot**2
                Ec[i] += E_tot**2
                Magb += Mag_tot
#                s_old += 1.0*Walkers[i].spin
#                s_2 += s_old**2
                EbCount += 1
        
        kb = 1.38*10**(-23)
        Eb[i] /= EbCount
        Ec[i] /= EbCount
        Magb[i] /= EbCount
        Magc[i] /= EbCount
        
        T=3
        C[i] = (Ec[i]-Eb[i]**2)/(kb*T**2)
        chi[i] = (Magc[i]-Magb[i]**2)/(kb*T)
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))

#        T = 1.0/(3.16681*10**(-6)*len(Walkers))
      
        print('    Capacity = '+str(C[i]))
        print('    Magnetization '+str(Magb[i]))
        print('    susceptibility'+str(chi[i]))
#        figure()
#        plot(range(EbCount),C/EbCount,range(EbCount),Mag/EbCount,range(EbCount),sus/EbCount)
        
    return Walkers, Eb, Accept, Magb, C, chi

def main():
    Walkers=[]

    dim = 2 # dimension of the system
    grid_side = 10
    grid_size = grid_side**dim
    
    # Ising model nearest neighbors only
    mapping = zeros((grid_side,grid_side),dtype=int) # mapping
    inv_map = [] # inverse mapping
    ii = 0
    for i in range(grid_side):
        for j in range(grid_side):
            mapping[i,j]=ii
            inv_map.append([i,j])
            ii += 1
 

    # get the 4 nearest neighbors of the walkers which are at the grids: 
    #[9,1,90,10],[0,2,90,10]...[19,11,0,2],[10,12,1,3]... avoiding boundary problem
    for i in range(grid_side):
        for j in range(grid_side):
            j1=mapping[i,(j-1) % grid_side]
            j2=mapping[i,(j+1) % grid_side]
            i1=mapping[(i-1) % grid_side,j]
            i2=mapping[(i+1) % grid_side,j]
            Walkers.append(Walker(spin=0.5,
                                  nn=[j1,j2,i1,i2],
                                  dim=dim,
                                  coords = [i,j]))
 
    
    Nblocks = 200 # grid size
    Niters = 1000
    eq = 20 # equilibration "time" boundary
    T = 3.0 #tamperature
    beta = 1.0/T
    """
    Notice: Energy is measured in units of k_B, which is why
            beta = 1/T instead of 1/(k_B T)
    """
    Walkers, Eb, Acc, Magb, C, chi= ising(Nblocks,Niters,Walkers,beta)

    figure()
    plot(Eb)
    title('Metropolis Monte Carlo Ising model')
    savefig('Metropolis Monte Carlo Ising model', dpi = 200)
    Eb = Eb[eq:]
    print('Ising total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(std(Eb)/mean(Eb)))) 

    figure()
    plot(Magb)
    title('Metropolis Monte Carlo Ising model Magnetizaton')
    savefig('Metropolis Monte Carlo Ising model Magnetizaton', dpi = 200)
    Magb = Magb[eq:]
    print('Ising total Magnetization: {0:.5f} +/- {1:0.5f}'.format(mean(Magb), std(Magb)/sqrt(len(Magb))))
    print('Variance to Magnetization ratio: {0:.5f}'.format(abs(std(Magb)/mean(Magb)))) 

    figure()
    plot(C)
    title('Metropolis Monte Carlo Ising model Capacity')
    savefig('Metropolis Monte Carlo Ising model Capacity', dpi = 200)
    C = C[eq:]
    print('Ising total Capacity: {0:.5f} +/- {1:0.5f}'.format(mean(C), std(C)/sqrt(len(C))))
    print('Variance to Capacity ratio: {0:.5f}'.format(abs(std(C)/mean(C)))) 

    figure()
    plot(chi)
    title('Metropolis Monte Carlo Ising model Susceptibility')
    savefig('Metropolis Monte Carlo Ising model Susceptibility', dpi = 200)
    chi = chi[eq:]
    print('Ising total Susceptibilty: {0:.5f} +/- {1:0.5f}'.format(mean(chi), std(chi)/sqrt(len(chi))))
    print('Variance to Susceptiblity ratio: {0:.5f}'.format(abs(std(chi)/mean(chi)))) 
    
    save_data_to_hdf5_file(fname,Walkers, Eb, Acc, Magb, C, chi)
    f = open('observable quantities.txt','w+')
    f.write(Walkers, Eb, Acc, Magb, C, chi)
    f.close()

if __name__=="__main__":
    main()
        
