# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:51:54 2019

@author: user
"""
from numpy import *
from linear_interp import *

def diffun_electric_field(x, y, eps0, Q, L):
    return (x+y)/(4*np.pi*eps0)*Q/L/(x^2+y^2)^3/2

def fun_electric_field(x, y, eps0, Q, L):
    return (x+y)*Q/L/(4*np.pi*eps0)

if __name__=="__main__":
    L = 10
    Q = 10
    eps0 = 8.85*10**(-12)
    x = np.linspace(-L/2, L/2, 100)
    y1 = np.linspace(-L/2, 0, 100)
    y2 = np.linspace(0.1, 0.1+L/2, 100)
    [X, Y1] = np.meshgrid(x, y1)
    F = diffun_electric_field(X, Y1, eps0, Q, L)
    spl2d = spline(x=x, y=y1, f=F, dims=2)
    F = spl2d.eval2d(x,y1)
    
    [X, Y2] = np.meshgrid(x, y2)
    F = diffun_electric_field(X, Y1, eps0, Q, L)
    spl2d = spline(x=x, y=y1, f=F, dims=2)
    F = spl2d.eval2d(x,y1)
    
    fig=figure()
    ax1=fig.add_subplot(121)
    ax1.pcolor(X,Y,F)
    ax1.set_title('interpolated numeric')
    G = fun_electric_field(x, y, eps0, Q, L)
    spl2d = spline(x=x, y=y, f=G, dims=2)
    G = spl2d.eval2d(x,y)
    ax2=fig.add_subplot(122)
    ax2.pcolor(X,Y,G)
    ax2.set_title('interpolated analytic')    
    
    
    
