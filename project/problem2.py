# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 07:56:19 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D

def Field(B, r, b, c):
    #dl = cos(theta)/dr                                                                                           = y
    #dBdr = b*c*np.cross(l, r)/np.abs(r)**2 
#    dBdr = []
#    for i in range(0,101):
#        dBdr.append(b*c/LA.norm(r[0:3,i])**2)
    r = np.transpose(r)
    dBdr = b*c/np.sum(np.transpose(np.abs(r)**2))
    return dBdr

def Bx(x, a, b, c):
    B = b*4*np.pi*c*a**2/(2*(x**2+a**2)**(3/2))
    return B

#def pend_ivpr(B,r, b, c):
#    b = 10**(-7) # b = uo/(4*pi)  
#    c = 1 # c =I
#    return Field(B, r, b, c)

def runge_kutta4(x,t,dt,func,**kwargs):
    """
    Fourth order Runge-Kutta for solving ODEs
    dx/dt = f(x,t)

    x = state vector at time t
    t = time
    dt = time step

    func = function on the right-hand side,
    i.e., dx/dt = func(x,t;params)
    
    kwargs = possible parameters for the function
             given as args=(a,b,c,...)

    Need to complete the routine below
    where it reads '...' !!!!!!!
    
    See FYS-4096 lecture notes.
    """
    F1 = F2 = F3 = F4 = 0.0*x
    if ('args' in kwargs):
        args = kwargs['args']
        F1 = func(x, t,*args)
        F2 = func(x+dt/2*F1, t+dt/2, *args)
        F3 = func(x+dt/2*F2, t+dt/2, *args)
        F4 = func(x+dt*F3, t+dt, *args)
    else:
        F1 = func(x, t)
        F2 = func(x+dt/2*F1, t+dt/2)
        F3 = func(x+dt/2*F2, t+dt/2)
        F4 = func(x+dt*F3, t+dt)

    return x+dt/6*(F1+2*F2+2*F3+F4), t+dt

def Field_3D(B, r, b, c):
    #dl = cos(theta)/dr                                                                                           = y
    #dBdr = b*c*np.cross(l, r)/np.abs(r)**2 
#    dBdr = []
#    for i in range(0,101):
#        dBdr.append(b*c/LA.norm(r[0:3,i])**2)
#    r = np.transpose(r)
    dBdr = np.zeros((3,101))
    dBdr[0, ] = b*c/np.sum(np.abs(r)**2, axis=0)
    dBdr[1, ] = np.zeros((101))
    dBdr[2, ]= np.zeros((101))
    return dBdr

def runge_kutta4_3D(x,t,dt,func,**kwargs):
    """
    Fourth order Runge-Kutta for solving ODEs
    dx/dt = f(x,t)

    x = state vector at time t
    t = time
    dt = time step

    func = function on the right-hand side,
    i.e., dx/dt = func(x,t;params)
    
    kwargs = possible parameters for the function
             given as args=(a,b,c,...)

    Need to complete the routine below
    where it reads '...' !!!!!!!
    
    See FYS-4096 lecture notes.
    """
    F1 = F2 = F3 = F4 = 0.0*x
    if ('args' in kwargs):
        args = kwargs['args']
        F1 = func(x, t,*args)
        F2 = func(x+np.array(F1)*np.array(dt/2), t+dt/2, *args)
        F3 = func(x+np.array(F2)*np.array(dt/2), t+dt/2, *args)
        F4 = func(x+np.array(F3)*np.array(dt/2), t+dt, *args)
    else:
        F1 = func(x, t)
        F2 = func(x+np.array(F1)*np.array(dt/2), t+dt/2)
        F3 = func(x+np.array(F2)*np.array(dt/2), t+dt/2)
        F4 = func(x+np.array(F3)*np.array(dt/2), t+dt)

    return x+np.array(dt)/(6*(F1+2*F2+2*F3+F4)), t+dt


def runge_kutta_test(a, x):
    b = 10**(-7) # b = uo/(4*pi)  
    c = 1 # c =I
    theta = np.linspace(0, 2*np.pi, len(x))
    r = np.array([x, a*np.cos(theta), a*np.sin(theta)])    
    dr = r[0,1] - r[0,0]
    sol = []
    for i in range(0, len(theta)):
        B = np.zeros((101))
        B, r[0,] = runge_kutta4(B, r[0,i], dr, Field, args=(b,c))
        sol.append(B)
    sol = np.array(sol)
    B_x = Bx(x,a,b,c)
    plt.figure()
    plt.plot(theta, sol[:, 0], 'b', label='numeric')
    plt.plot(theta, B_x, 'g', label='analytic')
    plt.legend(loc='best')
    plt.xlabel('theta')
    plt.ylabel('Bx')
    plt.grid()
    plt.savefig('Bx_numvsana.pdf')
    
    x1 = x
    x1[39:101] += 0.1
    r1 = np.array([x1-4*a, a*np.cos(theta), a*np.sin(theta)])    
    dr = r1[0,1] - r1[0,0]
    sol1 = []
    for i in range(0, len(theta)):
        B1 = np.zeros((101))
        B1, r1[0,] = runge_kutta4(B1, r1[0,i], dr, Field, args=(b,c))
        sol1.append(B1)
    sol1 = np.array(sol1)
    sol1 = sol1 + sol
    sol1[39, 0:101] = sol1[41, 0:101]
    sol1[40, 0:101] = sol1[41, 0:101]
    B_x1 = Bx(x1-4*a,a,b,c)
    B_x1 = B_x + B_x1
    plt.figure()
    plt.plot(theta, sol1[:, 0], 'b', label='numeric')
    plt.plot(theta, B_x1, 'g', label='analytic')
    plt.legend(loc='best')
    plt.xlabel('theta')
    plt.ylabel('Bx_parellel')
    plt.grid()
    plt.savefig('Bx_parellel_numvsana.pdf')    
    
    dr = r[0:3,1] - r[0:3,0]
    sol = []
    B = np.zeros((3,101))
    dr = np.resize(np.repeat(dr,101),[3,101])
    B, r[0:3,] = runge_kutta4_3D(B, r[0:3,], dr, Field_3D, args=(b,c))
    sol.append(B)
    sol = np.array(sol)    
    x1 = x
    x1[39:101] += 0.1
    r1 = np.array([x1-4*a, a*np.cos(theta), a*np.sin(theta)])    
    dr = r1[0:3,1] - r1[0:3,0]
    sol1 = []
    B1 = np.zeros((3,101))
    dr = np.resize(np.repeat(dr,101),[3,101])
    B1, r1[0:3,] = runge_kutta4_3D(B1, r1[0:3,], dr, Field_3D, args=(b,c))
    sol1.append(B1)
    sol1 = np.array(sol1)
    sol1 = sol1 + sol
#    sol1[0, 0:101, 39] = sol1[0, 0:101, 41]
#    sol1[0, 0:101, 40] = sol1[0, 0:101, 41]

    dBdr = Field_3D(B1, r1, b, c)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(dBdr[1,],dBdr[2,],dBdr[0,], label='numeric')
    ax.legend(loc='best')
    plt.xlabel('dBy')
    plt.ylabel('dBz')
    plt.title('dB')
    ax.grid()
    plt.savefig('dB_3D.pdf')    
    
#    print(np.shape(sol1))
#    print(sol1[0,0,])
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.plot(np.zeros((101)), np.zeros((101)), sol1[0, 0, ], label='numeric')
    ax1.legend(loc='best')
    plt.xlabel('By')
    plt.ylabel('Bz')
    plt.title('B')
    ax1.grid()
    plt.savefig('B_3D.pdf') 
    
    
if __name__=="__main__":
    a = 1
    x = np.linspace(0,10,101)
    runge_kutta_test(a, x)
    plt.show ()  
