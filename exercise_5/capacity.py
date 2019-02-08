# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 05:16:30 2019

@author: user
"""
#from Poisson import *
from mpl_toolkits.mplot3d import Axes3D
from electric_field import *

def Jacobi_1(L, a, N):
    T = np.zeros((L+1, L+1, N))
    T[2, :, :] = 0
    T[-2, :, :] = 0
    T[1, :, :] = 1
    T[-1, :, :] = -1
    for i in np.linspace(0, L+1, N):
        for j in np.linspace(0, L+1, N):
                for n in range(0, N-1):        
                    T[i*N, j*N, n] = a*(T[(i+1)*N,j*N, n] + T[(i-1)*N, j*N, n] + T[i*N, (j+1)*N, n] + T[i*N, (j-1)*N, n])
    return T 
   
if __name__=="__main__":
    a = 1
    L = 6
    w = 0.5 
    N = 100
    fig=plt.figure()
    ax1=fig(projection = '3d')
    T = Jacobi_1(L, a, N)
    D = poisson(T)
    Ex = simpson_nonuniform(x,D)
    Ey = simpson_nonuniform(y,D)
    ax1.quiver(Ex, Ey, D[0], D[1])  
    ax1.set_title('Electriv field') 
    show()
    
