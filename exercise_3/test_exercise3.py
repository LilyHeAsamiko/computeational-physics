# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:28:41 2019

@author: he
"""

from scipy.integrate import simps
import numpy as np
from linear_interp import *
from spline_class import spline 
import matplotlib.pyplot as plt

def fun_Simpson2D(x, y):
    return (x+y)*np.exp(-(x**2+y**2))

def fun_interp2D(x):
    return np.sqrt(2)*x
    
def test_Simpson2D():
    x = np.linspace(0, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x,  y)
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    z = fun_Simpson2D(x, y)
    I = simps(simps(z, dx = dx), dx = dy)
    return I

def test_Interp2D(): 
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-2, 2, 40)
    [X, Y] = np.meshgrid(x, y)
    F = fun_Simpson2D(X,Y)
    spl2d=spline(x=x, y=y, f=F, dims=2)
    xp = np.linspace(0, 2.0/np.sqrt(2), 100)
    f = np.zeros((len(xp),))
    ff = np.zeros((len(xp),))
    for i in range(len(xp)):
        x0 = xp[i]
        y0 = np.sqrt(2)*xp[i]  	
        f[i] = spl2d.eval2d(x0, y0)
        ff[i]=fun_Simpson2D(x0, y0)
    plt.plot(xp, f, '-', xp, ff, 'o')
    plt.show() 
   
if __name__=="__main__":
    test_Simpson2D();
    test_Interp2D();