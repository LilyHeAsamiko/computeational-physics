from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf
from mpl_toolkits.mplot3d import Axes3D

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

def potential_action(Walkers,time_slice1,time_slice2,tau):
    #calculate the potential part(primitive approximation)
    return 0.5*tau*(potential(Walkers[time_slice1]) \
                    +potential(Walkers[time_slice2]))

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

            # bisection sampling
            r02_ave = (r0+r2)/2 #
            log_S_Rp_R = -sum((r1-r02_ave)**2)/2/sigma2  # sum of the density
            Rp = r02_ave + random.randn(sys_dim)*sigma # calculate R'
            log_S_R_Rp = -sum((Rp - r02_ave)**2)/2/sigma2 
            Walkers[time_slice1].Re[ptcl_index] = 1.0*Rp
            KineticActionNew = kinetic_action(r0,Rp,tau,lambda1) +\
                kinetic_action(Rp,r2,tau,lambda1) #iterate the kinetic energy
            PotentialActionNew = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)
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

def density(Walkers):
    M = len(Walkers)
    Rou = np.zeros((M,M))
    Lambda = 0.5
    L = 2
    d =1
    k = 2*pi/L
    kb = 1.38*10**(-23)
    T = 1.0/(3.16681*10**(-6)*len(Walkers));
    beta = 1/(kb*T)
    psi = L**(-d/2)*exp(-1j*k*Walkers[0].Re[0])
    print(psi)
    psi2 = L**(-d/2)*exp(1j*k*Walkers[1].Re[0])
    E = Lambda*k**2
    rou = psi*psi2*exp(-beta*E)
    print(rou)
    Rou[1,1] = mean(rou)

    for m in range(M):
        for n in range(Walkers[0].Ne):
            k = 2*pi*n/L
            if (m<M-1):
                psi = L**(-d/2)*exp(-1j*k*Walkers[m].Re[n])
                psi2 = L**(-d/2)*exp(1j*k*Walkers[m+1].Re[n])
            else:
                psi = L**(-d/2)*exp(-1j*k*Walkers[m].Re[n])
                psi2 = L**(-d/2)*exp(1j*k*Walkers[0].Re[n])                
            E = Lambda*k**2
            rou += psi*psi2*exp(-beta*E)
            Rou[m,n] = mean(rou)  
    return Rou


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

def external_potential(Walker):
    V = 0.0
    for i in range(Walker.Ne):
        V += 0.5*sum(Walker.Re[i]**2)
        
    return V

def main():
    """
    # For 2D quantum dot
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0]),array([-0.5,0])],
                          spins=[0,1],
                          Nn=2, # not used
                          Rn=[array([-0.7,0]),array([0.7,0])], # not used
                          Zn=[1.0,1.0], # not used
                          tau = 0.25,
                          dim=2))
"""   


    Walkers=[]
    # For H2
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0,0]),array([-0.5,0,0])],
                          spins=[0,1],
                          Nn=2,
                          Rn=[array([-0.7,0,0]),array([0.7,0,0])],
                          Zn=[1.0,1.0],
                          tau = 0.1,
                          dim=3))

    M=100
    for i in range(M-1):
         Walkers.append(Walkers[i].w_copy())
    Nblocks = 200
    Niters = 100
    
    Walkers, Eb, Acc = pimc(Nblocks,Niters,Walkers)

    plot(Eb)
    Eb = Eb[20:]
    print('PIMC total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(std(Eb)/mean(Eb)))) 
    show()
    #calculate the tamperature
    
    T = 1.0/(3.16681*10**(-6)*len(Walkers));
    print(T)
    
    Rou = density(Walkers)
    print(Rou)
    x = Rou[1,:]
    y = Rou[:,1]
    [X, Y] = meshgrid(x, y)
    figure()
    #set the axis for four subplot separately
    ax = subplot(111, projection='3d')

    # plot interpolation 2D with 11*11 grids
    ax.plot_wireframe(X,Y,Rou)
    savefig('density.pdf', dpi=200)    

if __name__=="__main__":
    main()
        
