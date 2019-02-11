# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:51:54 2019

@author: user
"""
from numpy import *
from scipy.integrate import simps
from spline_class import *
from num_calculus import * 
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D

def diffun_electric_field(r_rod, r, eps0, Q, L):
    dx = r[0,0,1]-r[0,0,0]
#   return (x+y)/(4*np.pi*eps0)*Q/L/(x^2+y^2)**(3/2)
    dEx = Q*dx*(r[0,:,:]-r_rod[0,:,:])/(4*np.pi*eps0*L*LA.norm(r)**(3/2))
    dEy = Q*dx*(r[1,:,:]-r_rod[1,:,:])/(4*np.pi*eps0*L*LA.norm(r)**(3/2))
    return dEx, dEy

def uv_electric_field(r_rod, x_y, eps0, Q, L):
    u_mesh = np.meshgrid(r_rod[0,0], dEx[:,0])
    v_mesh = np.meshgrid(r_rod[0,0], dEy[:,0])
    return u_mesh, v_mesh

def fun_electric_field(r, eps0, Q, L):
#    r = np.array([L/2+d,0]);
    return Q/L/(4*np.pi*eps0)*(1/d-1/(d+L))


if __name__=="__main__":
    L = 10
    Q = 10
    eps0 = 8.85*10**(-12)
#   eps0 =1
    x = np.linspace(-L/2, L/2, 100)
    y1 = np.linspace(-L/2, 0, 50)
    y2 = np.linspace(0.1, 0.1+L/2, 50)
    y = np.concatenate([y1,y2])
    r_rod = [np.linspace(-L/2,L/2,100), np.zeros((100, 1))];
    r_rod = np.array(np.meshgrid(r_rod[0],r_rod[1]))
    r = np.array(np.meshgrid(x, y))
    dEx, dEy = diffun_electric_field(r_rod, r, eps0, Q, L)
    plt.quiver(r[0], r[1], dEx, dEy)
    u_mesh, v_mesh = uv_electric_field(r_rod, r, eps0, Q, L)
    plt.quiver(r_rod[0], r_rod[1], u_mesh[1], v_mesh[0])
    x = r[0,:,:]
    y = r[1,:,:]
    E = np.zeros((len(x),1))
#    for i in range(0, len(x)):
#        xy=meshgrid(x[i,:],y[:,i])
#        x = xy[0]
#        y = xy[1]
#        dx = float("{0:.2f}".format(x[i,1]-x[i,0]))
#        dy = float("{0:.2f}".format(y[1,i]-y[0,i]))
#        E[i]= simps(simps((x+y)/(4*np.pi*eps0)*Q/L/(x**2+y**2)**(3/2)
#    , dx = dx), dx = dy)
    
    for i in range(0,len(x)):
        E[i] = sum(dEy[0:i,0])
 
    d = r[0,0,:];
    E_ana = fun_electric_field((L/2+d, 0), eps0, Q, L)
    
    fig=plt.figure()
    plt.plot(E_ana, label="Analytical")
    plt.plot(E, label="Numerical")
    plt.legend()
    plt.title("Analytical vs. numerical (normed)")
    plt.show()