# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:20:38 2019

@author: user
"""
import numpy as np
from runge_kutta import *

def Force(b, c, r0, v0, t):
    Fm, v = y
    dXdr = [r, b + np.cross(v, c)]
    X =[Fmx, Fmy, Fmz, vx, vy, vz]
    return X

def pend_ivp(r,y):
    b = 0.25
    c = 4.0
    return Force(y,r,b,c)

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
        F3 = func(x+dt/2*F3, t+dt/2)
        F4 = func(x+dt*F3, t+dt)

    return x+dt/6*(F1+2*F2+2*F3+F4), t+dt

def runge_kutta_test(ax):
    b = 0.25
    c = 4.0
    v0 = [0.1, 0.1, 0.1]   
    y0 = [[0.05, 0, 0]+np.cross(v0, [0, 4.0, 0]),[0.1, 0.1, 0.1]]
    t = np.linspace(0,5, 101)
    dt = t[1]-t[0]
    sol=[]
    x=1.0*np.array(y0)
    for i in range(len(t)):
        sol.append(x)
        x, tp = runge_kutta4(x,t[i],dt,Force,args=(b,c))
    sol=np.array(sol)
    ax.plot(t, sol[:, 0], 'b', label='v(t)')
    ax.plot(t, sol[:, 1], 'g', label='r(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()
    
if __name__=="__main__":
    ax = figure()
    runge_kutta_test(ax)
    ax.set_title('own Runge-Kutta 4')
    show ()  
    
