# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:25:21 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from num_calculus import * 

#x = np.linspace(0, 1, 50)

def matrix(n, h):
    A = np.zeros((n+2,n+2))
    A[0,0]= 1
    A[101,101] = 1
    
    #b[0] = np.pi/h*(x[1]-2*x[0])*np.cos(np.pi*x[0])+1/h*[2*np.sin(np.pi*x[0])-np.sin(np.pi*x[1])]
    for i in range(1, n):
        for j in range(1, n):  
            if i == j: A[j, i] = 2/h
            elif i == j + 1 or i == j - 1: A[j, i] = -1/h
    return A

def sol(n):
    b = np.zeros((n+1,))
    b[1:n-1] = np.pi/h*(x[0:n-2]+x[2:n]-2*x[1:n-1])*(np.cos(np.pi*x[1:n-1]))+(2*np.sin(np.pi*x[1:n-1])-np.sin(np.pi*x[0:n-2])-np.sin(np.pi*x[2:n]))/h
    b = b[0:n]
    return b


#x1 = np.linspace(0, 1, 50)
#for j in range (0, n):
#    for i in range(1,n-1):
#        if (x1[j]< x[i] and x1[j]> x[i-1]):
#            u[j， i] = (x1[j]-x[i-1])/h
#        elif (x1[j] < x[i+1] and x1[j] > x[i]):
#            u[j, i] = (x[i+1]-x1[j])/h
#        else: u[j, i] = 0

def hat(x1, x, n, h):
    u = np.zeros((n,n))
    for i in range(0, n-1):
        for j in range(0, n-1):
            if (x1 <= x[j] and x1 >= x[j-1]):
                u[i, j] = (x1 - x[j-1])/h
            elif (x[j+1] >= x1 and x[j] <= x1):
                u[i, j] = (x[j+1] - x1)/h
            else: u[i, j] = 0
            return u
    
def matrix1(x, u, i, j, h):
    A = np.zeros((x.size, x.size))
    dh = 0.1*h
    x_1 = np.arange(x[0], x[-1], dh)
    for i in range(0,A.shape[0]-1):
        for j in range(0, A.shape[1]-1):
            def ui(u):
                return u[i,]
            def uj(u):
                d = A.shape[1]
                return np.transpose(np.transpose(u)[j,])
#                return u[0:d-1,j]
            def dui(x_1):
                return eval_derivative(ui, x_1, dh)   
            def duj(x_1):
                return eval_derivative(uj, x_1, dh)  
            def integral(dui, duj):
                return dui(x_1)*duj(x_1)
            A[i, j] = simpson_integration(x_1,integral(dui,duj))
    return A

if __name__=="__main__":
    h =0.01
    x = np.arange(0, 1+h, h)
    h = x[1] - x[0]
    n = np.shape(x)[0]
    A = matrix(n, h)
    b = sol(n) 
    a = b@np.linalg.inv(A[0:n,0:n])
    x1 = 2
    u = hat(x1, x, n, h)
    A_1 = matrix1(x, u, i, j, h)                                                                                                                                              
    a_1 = b@np.linalg.inv(A[0:n-1,0:n-1])
    plt.figure()
    plt.plot(0:n-1, a, 0:n-1, a_1)                       