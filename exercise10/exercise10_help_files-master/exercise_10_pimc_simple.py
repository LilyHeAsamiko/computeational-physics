from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf

class Walker:
    def __init__(self,*args,**kwargs):
        self.Ne = kwargs['Ne']
        self.Re = kwargs['Re']
        self.spins = kwargs['spins']
        self.Nn = kwargs['Nn']
        self.Rn = kwargs['Rn']
        self.Zn = kwargs['Zn']
        self.tau = kwargs['tau']
        self.sys_dim = kwargs['dim']

    def w_copy(self):
        return Walker(Ne=self.Ne,
                      Re=self.Re.copy(),
                      spins=self.spins.copy(),
                      Nn=self.Nn,
                      Rn=self.Rn.copy(),
                      Zn=self.Zn,
                      tau=self.tau,
                      dim=self.sys_dim)
    

def kinetic_action(r1,r2,tau,lambda1):
    #calculate the kinetic part
    return sum((r1-r2)**2)/lambda1/tau/4
	
def factor(n):
	fac = 1
	for i in range(1,n+1):
		fac = fac*i 
	return fac

def potential_action(Walkers,time_slice1,time_slice2,tau):
    #calculate the potential part(primitive approximation)
    return 0.5*tau*(potential(Walkers[time_slice1]) \
                    +potential(Walkers[time_slice2]))

def bisection(r0, r1, r2, sigma, sigma2, tau, lambda1, Walkers, time_slice0, time_slice1, time_slice2, sys_dim, ptcl_index):
    # bisection sampling
    r02_ave = (r0+r2)/2 
    log_S_Rp_R = -sum((r1-r02_ave)**2)/2/sigma2  # sum of the density
    Rp = r02_ave + random.randn(sys_dim)*sigma # calculate R'
    log_S_R_Rp = -sum((Rp - r02_ave)**2)/2/sigma2 
    Walkers[time_slice1].Re[ptcl_index] = 1.0*Rp
    KineticActionNew = kinetic_action(r0,Rp,tau,lambda1) +\
        kinetic_action(Rp,r2,tau,lambda1) #iterate the kinetic energy
    PotentialActionNew = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)
    return log_S_Rp_R, log_S_R_Rp, KineticActionNew, PotentialActionNew
    
def pimc(Nblocks,Niters,Walkers):
    M = len(Walkers) #path length
    Ne = Walkers[0].Ne*1 #Initiate the e numbers
    sys_dim = 1*Walkers[0].sys_dim #dimension 
    tau = 1.0*Walkers[0].tau #initiate the tau which is beta/M
    lambda1 = 0.5 #set the lambda
    Eb = zeros((Nblocks,)) #create the space for energy
    Accept=zeros((Nblocks,)) #create the space for accuracy
    AccCount=zeros((Nblocks,)) #create the total neumber of terms in accuracy
    sigma2 = lambda1*tau #to adjust the distribution's spread
    sigma = sqrt(sigma2) #to adjust the distribution's spread

    obs_interval = 5
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            time_slice0 = int(random.rand()*M)
            time_slice1 = (time_slice0+1)%M
            time_slice2 = (time_slice1+1)%M
            ptcl_index = int(random.rand()*Ne)

            r0 = Walkers[time_slice0].Re[ptcl_index]
            r1 = 1.0*Walkers[time_slice1].Re[ptcl_index]
            r2 = Walkers[time_slice2].Re[ptcl_index]
 
            KineticActionOld = kinetic_action(r0,r1,tau,lambda1) +\
                kinetic_action(r1,r2,tau,lambda1)
            PotentialActionOld = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)
            log_S_Rp_R, log_S_R_Rp, KineticActionNew, PotentialActionNew = bisection(r0, r1, r2, sigma, sigma2, tau, lambda1, Walkers, time_slice0, time_slice1, time_slice2, sys_dim, ptcl_index)
            #iterate the potential energy
            deltaK = KineticActionNew-KineticActionOld #difference of kinetic energy
            deltaU = PotentialActionNew-PotentialActionOld #difference potential energy
            #print('delta K', deltaK)
            #print('delta logS', log_S_R_Rp-log_S_Rp_R)
            #print('exp(dS-dK)', exp(log_S_Rp_R-log_S_R_Rp-deltaK))
            #print('deltaU', deltaU)
            q_R_Rp = exp(log_S_Rp_R-log_S_R_Rp-deltaK-deltaU) #calculate the density
            A_RtoRp = min(1.0,q_R_Rp)
            if (A_RtoRp > random.rand()):
                Accept[i] += 1.0 #count correct terms
            else:
                Walkers[time_slice1].Re[ptcl_index]=1.0*r1
            AccCount[i] += 1
            if j % obs_interval == 0:
                E_kin, E_pot = Energy(Walkers)
                #print(E_kin,E_pot)
                Eb[i] += E_kin + E_pot
                EbCount += 1
            #exit()
            
        Eb[i] /= EbCount #calculate the energy
        Accept[i] /= AccCount[i] #calculate the acceptance
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))
        
        
    return Walkers, Eb, Accept


def Energy(Walkers): 
    M = len(Walkers)
    d = 1.0*Walkers[0].sys_dim
    tau = Walkers[0].tau
    lambda1 = 0.5
    U = 0.0
    K = 0.0
    for i in range(M):
        U += potential(Walkers[i])
        for j in range(Walkers[i].Ne):
            if (i<M-1):
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[i+1].Re[j])**2)/4/lambda1/tau**2
            else:
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[0].Re[j])**2)/4/lambda1/tau**2    
    return K/M,U/M
        
    

def potential(Walker):
    V = 0.0
    r_cut = 1.0e-12
    # e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
            V += 1.0/max(r_cut,r)

    Vext = external_potential(Walker)
    
    return V+Vext

#def potential(Walker):
#    V = 0.0
#    r_cut = 1.0e-4
    # e-e
#    for i in range(Walker.Ne-1):
#        for j in range(i+1,Walker.Ne):
#            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
#            V += 1.0/max(r_cut,r)

    # e-Ion
#    for i in range(Walker.Ne):
#        for j in range(Walker.Nn):
#            r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
#            V -= Walker.Zn[j]/max(r_cut,r)

    # Ion-Ion
#    for i in range(Walker.Nn-1):
#        for j in range(i+1,Walker.Nn):
#            r = sqrt(sum((Walker.Rn[i]-Walker.Rn[j])**2))
#            V += 1.0/max(r_cut,r)

	# e-Ion 
#    sigma = 0.1
#    r_cut = 7.4e-10 #r=1.4*a0
#    for i in range(Walker.Ne):
#        for j in range(i+1,Walker.Nn):
#            r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
#            rr = max(r_cut,r)
#            V += Walker.Zn[j]*erf(r/sqrt(2*sigma))/rr
    
#    return -V


#harmonic
def external_potential(Walker):
    V = 0.0
    for i in range(Walker.Ne):
        V += 0.5*sum(Walker.Re[i]**2)
        
    return V

def main():
    Walkers=[]

"""
    # For H2
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0,0]),array([-0.5,0,0])],
                          spins=[0,1],
                          Nn=2,
                          Rn=[array([-0.7,0,0]),array([0.7,0,0])],
                          Zn=[1.0,1.0],
                          tau = 0.1,
                          dim=3))

 """
    # For 2D quantum dot
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0]),array([-0.5,0])],
                          spins=[0,1],
                          Nn=2, # not used
                          Rn=[array([-0.7,0]),array([0.7,0])], # not used
                          Zn=[1.0,1.0], # not used
                          tau = 0.25,

    
    M=100
    for i in range(M-1):
         Walkers.append(Walkers[i].w_copy())
    Nblocks = 200
    Niters = 100
    
    Walkers, Eb, Acc = pimc(Nblocks,Niters,Walkers)

    plot(Eb)
#    title ('cumulant approximation')
    title ('e-e harmonic quantum dot')
    Eb = Eb[20:]
    N = len(Eb)
    print('PIMC total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(std(Eb)/mean(Eb)))) 
    show()
    savefig('e-e harmonic quantum dot.pdf',dpi = 200)
    #
    n = np.array(linspace(1, N, N))
    Eb_mean = zeros(N,)
    Eb_std = zeros(N,)
    err_l = zeros(N,)
    err_u = zeros(N,)
    for i in range(N):
        a = int(n[i])
        Eb_mean[i] = mean(Eb[0:a])
        Eb_std[i] =std(Eb[0:a])
    figure(),
    plot(linspace(1,N,N), Eb,'b-', linspace(1,N,N), (mean(Eb)-std(Eb))*ones(N,), 'k--',linspace(1,N,N),  mean(Eb)*ones(N,), 'r-',linspace(1,N,N), (mean(Eb)+std(Eb))*ones(N,), 'k--')
    legend(['Energy','Standard Deviation','Total Energy'])
    title('Energy')
    savefig('e-e_harmonic_Energy.pdf',dpi = 200)
    figure(),
    errorbar(linspace(1,N,N),Eb_mean, yerr=Eb_std)
    title('Mean Energy ')
    savefig('e-e_harmonic_Mean_Energy.pdf',dpi = 200)
    figure(),
    plot(Acc)
    title('Acceptance Ratio')
    savefig('e-e_harmonic_Acceptance.pdf',dpi = 200)
    
    #calculate the tamperature
    T = 1.0/(3.16681*10**(-6)*len(Walkers));
    print(T)
    
    

if __name__=="__main__":
    main()
        
