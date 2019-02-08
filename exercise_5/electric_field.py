# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:51:54 2019

@author: user
"""
from numpy import *
from spline_class import *
from num_calculus import * 
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D


def diffun_electric_field(r, eps0, Q, L):
    r_0 = [0, 0]
    x = r[0,1]
    dx = x[1]-x[0]
    r_rod = np.array([dx,0*dx]);
#   return (x+y)/(4*np.pi*eps0)*Q/L/(x^2+y^2)**(3/2)
    return Q*dx*r_rod/(4*np.pi*eps0*L*LA.norm(r)**2)

    
def fun_electric_field(r, eps0, Q, L):
    d = 1;
    r = np.array([L/2+d,0]);
    return r*Q/L/(4*np.pi*eps0)*(1/d-1/(d+L))

if __name__=="__main__":
    L = 10
    Q = 10
    eps0 = 8.85*10**(-12)
    x = np.linspace(-L/2, L/2, 100)
    y1 = np.linspace(-L/2, 0, 50)
    y2 = np.linspace(0.1, 0.1+L/2, 50)
    y = np.concatenate([y1,y2])
    r = np.array(np.meshgrid(x, y))
    F = diffun_electric_field(r, eps0, Q, L)
    spl2d = spline(x=x, y=y, f=F, dims=2)
    F = spl2d.eval2d(x,y)
    dEx = diffun_electric_field(x, eps0, Q, L)
    dEy = diffun_electric_field(y, eps0, Q, L)
    
    fig=figure()
    ax1=fig.add_subplot(121)
    ax1.pcolor(r,F)
    ax1.set_title('interpolated numeric')
    G = fun_electric_field(r, eps0, Q, L)
    spl2d = spline(x=x, y=y, f=G, dims=2)
    G = spl2d.eval2d(x,y)
    ax2=fig.add_subplot(122, projection='3d')
    ax2.quiver(x,y,dEx, dEy)
    ax2.set_title('interpolated analytic')    
    
    
    
