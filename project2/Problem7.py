# -*- coding: utf-8 -*-
"""
Created on Sun May  5 04:45:23 2019

@author: user
"""
from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf
#from matplotlib.ticker import ScalarFormatter

#initiate the walker
class Walker:
    def __init__(self,*args,**kwargs):
        for k, v in kwargs.items():
            self.Ne = kwargs['Ne']
            self.Re = kwargs['Re']
            self.spins = kwargs['spins']
            self.Nn = kwargs['Nn']
            self.Rn = kwargs['Rn']
            self.Zn = kwargs['Zn']
            self.tau = kwargs['tau']
            self.sys_dim = kwargs['dim']
            self.epsilon =  kwargs['epsilon']
            self.sig = kwargs['sig']
# K
def kinetic_action(r1,r2,tau,lambda1):
    return sum((r1-r2)**2)/lambda1/tau/4
# V
def potential_action(Walkers,time_slice1,time_slice2,tau):
    return 0.5*tau*(potential(Walkers[time_slice1])+potential(Walkers[time_slice2]))
# Lennard-Jones
def lennard_jones_potential(r, epsilon, sig):
    return 4 * epsilon * ((sig/r)**12 - (sig/r)**6)
#pimc
def pimc(Nblocks,Niters,Walkers):
    M = len(Walkers)
    for i in [0,2,4,6]:            
        tau1 = 1.0*Walkers[i].tau
    lambda1 = 921+1  # Mass of hydrogen: proton + electron
    for i in [1,3,5,7]:            
        tau2 = 1.0*Walkers[i].tau
    lambda2 =  1450  # Mass of helium: proton He

    Eb = zeros((Nblocks,))
    dst = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    sigma2 = (lambda1*tau1+lambda2*tau2)/2#sigma 2 for mixing H_He
    sigma = sqrt(sigma2)#sigma  for mixing H_He

    obs_interval = 8
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            if j % 2 == 0:            
                Ne = Walkers[i % obs_interval].Ne*1
                sys_dim = 1*Walkers[i % obs_interval].sys_dim
                tau = 1.0*Walkers[i % obs_interval].tau
  # hydrogen: proton + electron
            else:            
                Ne = Walkers[i % obs_interval].Ne*1
                sys_dim = 1*Walkers[i % obs_interval].sys_dim
                tau = 1.0*Walkers[i % obs_interval].tau
  # Helium: proton He    
                        
            time_slice0 = int(random.rand()*M)
            time_slice1 = int((time_slice0+1)%M)
            time_slice2 = int((time_slice1+1)%M)
            ptcl_index = int(random.rand()*Ne)

            if (size(Walkers[time_slice0].Re) == 2): 
                r0 = Walkers[time_slice0].Re[ptcl_index]
            else:
                r0 = Walkers[time_slice0].Re
            if (size(Walkers[time_slice1].Re) == 2): 
                r1 = Walkers[time_slice1].Re[ptcl_index]
            else:
                r1 = Walkers[time_slice1].Re
            if (size(Walkers[time_slice2].Re) == 2): 
                r2 = Walkers[time_slice2].Re[ptcl_index]
            else:
                r2 = Walkers[time_slice2].Re
                
            KineticActionOld = kinetic_action(r0,r1,tau,lambda1) +\
                kinetic_action(r1,r2,tau,lambda1)
            PotentialActionOld = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            # bisection sampling
            r02_ave = (r0+r2)/2
            log_S_Rp_R = -sum((r1 - r02_ave)**2)/2/sigma2             
            Rp = r02_ave + random.randn(sys_dim)*sigma
            log_S_R_Rp = -sum((Rp - r02_ave)**2)/2/sigma2
            if(size(Walkers[time_slice1].Re) == 1):
                Walkers[time_slice1].Re = 1.0*Rp
            elif(size(Walkers[time_slice1].Re) == 2): 
                Walkers[time_slice1].Re[ptcl_index] = 1.0*Rp
            
            KineticActionNew = kinetic_action(r0,Rp,tau,lambda1) +\
                kinetic_action(Rp,r2,tau,lambda1)
            PotentialActionNew = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            deltaK = KineticActionNew-KineticActionOld
            deltaU = PotentialActionNew-PotentialActionOld
            #print('delta K', deltaK)
            #print('delta logS', log_S_R_Rp-log_S_Rp_R)
            #print('exp(dS-dK)', exp(log_S_Rp_R-log_S_R_Rp-deltaK))
            #print('deltaU', deltaU)
            q_R_Rp = exp(log_S_Rp_R-log_S_R_Rp-deltaK-deltaU)
            A_RtoRp = min(1.0,q_R_Rp)
            if (A_RtoRp > random.rand()):
                Accept[i] += 1.0
            else:
                if (size(Walkers[time_slice1].Re) == 2): 
                    Walkers[time_slice1].Re[ptcl_index]=1.0*r1
                else:
                    Walkers[time_slice1].Re=1.0*r1
            AccCount[i] += 1
            Exp_Ek = zeros((Niters))
            Eb = zeros((Niters))
            Eb2 = zeros((Niters))
            mag = zeros((Niters,2))
            mb = zeros((Niters,2))
            mb2 = zeros((Niters,2))
            EbCount = 0
            if j % obs_interval == 0:
                E_kin, E_pot = Energy(Walkers)
                Exp_Ek[i] = expect_energy(Walkers)
                dst[i] += distance(Walkers)
                #print(E_kin,E_pot)
                Eb[i] += E_kin + E_pot
                Eb2[i] += Eb[i]**2
                mag = magnetization(Walkers)
                mb[i,:] += mag
                mb2[i,:] += mag**2
                EbCount += 1
                print('E_k:', E_kin)
                print('E_pot:', E_pot)
                print('E_tot:', Eb)
                print('Exp_Ek:', Exp_Ek)
                print('dst:', dst)
                print('mag:',mb)                
            #exit()
            
        Eb[i] /= EbCount
        dst[i] /= EbCount
        Exp_Ek[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('Exp_E   = {0:.5f}'.format(Exp_Ek[i]))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))

    return Walkers, Eb, Eb2, Exp_Ek, Accept, dst, mb, mb2, EbCount 

#calculate the energy
def Energy(Walkers):
    M = len(Walkers)
    d = 1.0*Walkers[0].sys_dim
    tau1 = Walkers[0].tau
    tau2 = Walkers[1].tau    
    lambda1 =922
    lambda2 =1450
    U = 0.0
    K = 0.0
    for i in range(M):
        U += potential(Walkers[i])
        if (Walkers[i].Ne == 1 & Walkers[i].Nn ==1):
            if (i<M-1): K += d/2/Walkers[i].tau-sum((Walkers[i].Re-Walkers[i+1].Rn)**2)/4/lambda1/tau**2 
            else: K += d/2/Walkers[i].tau-sum((Walkers[i].Re-Walkers[0].Rn)**2)/4/lambda1/tau**2
        elif (Walkers[i].Nn == 2 & Walkers[i+1].Nn ==2):    
            for j in range(Walkers[i].Nn*1):
                if (i<M-1):
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[i+1].Rn[j])**2)/4/lambda1/tau**2
                else:
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[0].Rn)**2)/4/lambda1/tau**2
        elif (Walkers[i].Nn == 2 & Walkers[i+1].Nn ==1):
            for j in range(Walkers[i].Nn*1):
                if (i<M-1):
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[i+1].Rn)**2)/4/lambda1/tau**2
                else:
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[0].Rn)**2)/4/lambda1/tau**2
            
    return K/M,U/M
        
    

def potential(walker):
    V = 0.0
    r_cut = 1.0e-12
    # e-ion  
    if (walker.Ne == 1): a = walker.Re
    elif (walker.Nn == 1): b = walker.Rn
    elif(walker.Ne == 2 & walker.Nn == 2):       
        for i in range(walker.Ne*1):
            for j in range(walker.Nn*1):
                a = walker.Re[i]
                b = walker.Rn[j]
                r = sqrt(sum((a-b)**2))
                epsilon = walker.epsilon
                sig = walker.sig 
                V += lennard_jones_potential(r, epsilon, sig)    
    return V
# calculate the intra-distance
def distance(Walkers):
    d = 0.0
    for walker in Walkers:
#        assert(walker.Ne == 2)
        if (walker.Ne == 2):
            d += sqrt(sum((walker.Re[0]-walker.Re[1])**2))
    return d / len(Walkers)


# calculate the magnetization
def magnetization(Walkers):
    m = 0.0
    for i in range(len(Walkers)):
        m += Walkers[i].spins
    return m

def external_potential(Walker):
    V = 0.0
    
    for i in range(Walker.Ne):
        V += 0.5*sum(Walker.Re[i]**2)

    """
    r_cut = 1.0e-12
    # e-Ion
    sigma = 0.05
    for i in range(Walker.Ne):
        for j in range(Walker.Nn):
            r = max(r_cut,sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2)))
            #V -= Walker.Zn[j]/max(r_cut,r)
            V -= Walker.Zn[j]*erf(r/sqrt(2.0*sigma))/r

    # Ion-Ion
    for i in range(Walker.Nn-1):
        for j in range(i+1,Walker.Nn):
            r = sqrt(sum((Walker.Rn[i]-Walker.Rn[j])**2))
            V += Walker.Zn[i]*Walker.Zn[j]/max(r_cut,r)
    """
    
    return V
# calculate the expectation of the energy
def expect_energy(Walker):
    M = len(Walkers)
    d = 1.0*Walkers[0].sys_dim
    tau1 = Walkers[0].tau
    tau2 = Walkers[1].tau    
    lambda1 =  922
    lambda2 =  1450
    K = 0.0
    for i in range(M):
        if (Walkers[i].Ne == 1 & Walkers[i].Nn ==1):
            if (i<M-1): K += d/2/Walkers[i].tau-sum((Walkers[i].Re-Walkers[i+1].Rn)**2)/4/lambda1/tau**2 
            else: K += d/2/Walkers[i].tau-sum((Walkers[i].Re-Walkers[0].Rn)**2)/4/lambda1/tau**2
        elif (Walkers[i].Nn == 2 & Walkers[i+1].Nn ==2):    
            for j in range(Walkers[i].Nn*1):
                if (i<M-1):
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[i+1].Rn[j])**2)/4/lambda1/tau**2
                else:
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[0].Rn)**2)/4/lambda1/tau**2
        elif (Walkers[i].Nn == 2 & Walkers[i+1].Nn ==1):
            for j in range(Walkers[i].Nn*1):
                if (i<M-1):
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[i+1].Rn)**2)/4/lambda1/tau**2
                else:
                    K += d/2/Walkers[i].tau-sum((Walkers[i].Rn[j]-Walkers[0].Rn)**2)/4/lambda1/tau**2 
    return K/M
    """
    r_cut = 1.0e-12
    # e-Ion
    sigma = 0.05
    for i in range(Walker.Ne):
        for j in range(Walker.Nn):
            r = max(r_cut,sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2)))
            #V -= Walker.Zn[j]/max(r_cut,r)
            V -= Walker.Zn[j]*erf(r/sqrt(2.0*sigma))/r

    # Ion-Ion
    for i in range(Walker.Nn-1):
        for j in range(i+1,Walker.Nn):
            r = sqrt(sum((Walker.Rn[i]-Walker.Rn[j])**2))
            V += Walker.Zn[i]*Walker.Zn[j]/max(r_cut,r)
    """




def main():

    """
    # For H2
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0,0]),array([-0.5,0,0])],
                          spins=[0,1],
                          Nn=2,
                          Rn=[array([-0.7,0,0]),array([0.7,0,0])],
                          Zn=[1.0,1.0],
                          tau = 0.5,
                          dim=3))
    """

    # M*tau = 1/(k*T)
    # tau = 1/(T*k*M)
    Walkers= []

    T = 300.15
    Ms = 2**np.arange(6)  # [1,2,4,8...]
    taus = 1 / (T*Ms)

    # initialize the walkers
    for i in [0,2,4,6]:
        walker = Walkers.append(Walker(Ne=2*1,
                        Re=array([0.5,-0.5]),
                        epsilon=sqrt(0.1745*3.2*10**(-5)),
                        sig=(4.73+1.25)/2,
                        spins=array([0,1]),
                        Nn=2, 
                        Rn=array([0.7,-0.7]), 
                        Zn=array([1.0,1.0]), 
                        tau = 0.1,
                        dim=1))
    
    for i in [1,3,5,7]:
        walker = Walkers.append(Walker(Ne=1*1,
                        Re=array(0.3),
                        epsilon=sqrt(0.1745*3.2*10**(-5)),
                        sig=(4.73+1.25)/2,
                        spins=array([1,0]),
                        Nn=1, 
                        Rn=array(0.1), 
                        Zn=2.0, 
                        tau = 0.1,
                        dim=1))


    Nblocks = 200
    Niters = 1000
    
    Edata = []
    Edata2 = []
    Mdata = []
    Mdata2 = []
    #pimc
    Walkers, Eb, Eb2, Exp_Ek, Accept, dst, mb, mb2, EbCount  = pimc(Nblocks,Niters,Walkers)

    eq = 15            
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
    
    figure()
    Cv = (Edata2[:,0]-Edata[:,0]**2)/T**2/M
    plot(T,Cv,'-o')
    plot([2.27, 2.27],[0.0, 1.03*amax(Cv)],'k--')
    ylabel('Heat Capacity per spin H_He')
    xlabel('Temperature')
    savefig('Heat Capacity per spinH_He.pdf', mpi = 200)
    # Notice: T_c_exact = 2.27
    # results plot(observables: ENERGY, HEAT CAPACITY, MAGNETIZATION, SUSCEPTIBILITY)
    figure()
    errorbar(T,Edata[:,0]/M,Edata[:,1]/M)
    ylabel('Energy per spin H_He' )
    xlabel('Temperature')
    savefig('Energy per spin H_He.pdf' , mpi = 200)

    figure()
    errorbar(T,Mdata[:,0]/M,Mdata[:,1]/M)
    ylabel('Magnetization per spin H_He' )
    xlabel('Temperature')
    savefig('Magnetization per spin H_He.pdf', mpi = 200)
    
    figure()
    chi = (Mdata2[:,0]-Mdata[:,0]**2)/T**2/M
    plot(T,chi,'-o')
    plot([2.27, 2.27],[0.0, 1.03*amax(chi)],'k--')
    ylabel('Susceptibility per spin H_He')
    xlabel('Temperature')
    savefig('Susceptibility per spin H_He.pdf', mpi = 200)
    show()
    
    figure()    
    plt.plot(Exp_Ek)
    xlabel('Nblocks')
    ylabel('Expectation of Energy H_He')
    title('Expectation of Energy')
    savefig('Expectation per spin H_He.pdf', mpi = 200)

    figure()    
    plt.plot(dst)
    xlabel('Nblocks')
    ylabel('Averaged distance H_He')
    title('Averaged distance')
    savefig('Averaged distance per spin H_He.pdf', mpi = 200)    
    show()

if __name__=="__main__":
    main()

