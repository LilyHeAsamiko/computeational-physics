# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 03:59:49 2019

@author: user
"""
from num_calculus import * 
import matplotlib.pyplot as plt 
def poisson(T):
    eps0 = 1
    x = T(0,)
    dx = x[0, 0] - x[0, 1]  
    rou = eps* (eval_2nd_derivative(T, x, dx)-eval_2nd_derivative(T, y, dy))
    return rou

def Jacobi(L, a, N):
    T = np.zeros((L+1, L+1, N))
    T[0, :, :] = 0
    T[L, :, :] = 0
    T[:, L, :] = 0
    for i in np.linspace(0, L+1, N):
        for j in np.linspace(0, L+1, N):
                for n in range(0, N-1):        
                    T[i*N, j*N, n] = a*(T[(i+1)*N,j*N, n] + T[(i-1)*N, j*N, n] + T[i*N, (j+1)*N, n] + T[i*N, (j-1)*N, n])
    return T

def Gauss_Seidel(L, a, N):
    T = np.zeros((L+1, L+1, N))
    T[0, :, :] = 0
    T[L, :, :] = 0
    T[:, L, :] = 0             
    for i in np.linspace(0, L+1, N):
        for j in np.linspace(0, L+1, N):
                for n in range(0, N-1):       
                    T[i*N, j*N, n+1] = a*(T[(i+1)*N, j*N, n] + T[(i-1)*N, j*N, n+1] + T[i*N, (j+1)*N, n] + T[i*N, (j-1)*N, n+1])
    return T

def SOR(L, w, N):
    T = np.zeros((L+1, L+1, N))
    T[0, :, :] = 0
    T[L, :, :] = 0
    T[:, L, :] = 0               
    for i in np.linspace(0, L+1, N):
        for j in np.linspace(0, L+1, N):
                for n in range(0, N-1): 
                    T[i*N, j*N, n+1] = (1-w)*T[(i+1)*N, j*N, n] + w/4*(T[(i-1)*N, j*N, n] + T[i*N, (j+1)*N, n] + T[i*N, (j-1)*N, n])
    return T

if __name__=="__main__":
    a = 1
    L = 1
    w = 1.8 
    N = 100
    fig=plt.figure()
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    T = Jacobi(L, a, N)
    d = poisson(T)
    ax1.set_title('Jacobi')
    plot(d[0], d[1])
    T = Gauss_Seidel(L, a, N)
    d = poisson(T)
    plot(d[0], d[1])
    ax2.set_title('Gauss_Seidel')
    T = SOR(L, a, N)
    d = poisson(T)
    plot(d[0], d[1])
    ax3.set_title('SOR')
    show()
    


