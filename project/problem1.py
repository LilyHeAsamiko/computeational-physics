# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 05:56:32 2019

@author: user
"""
import numpy as np
import matplotlib as plt
from scipy.integrate import simps

def test_f(x, y):
    return (x+y)*np.exp(-0.5*(x**2+y**2)**0.5)

def test_simpson_integeration2D(mesh, test_f):
    X = mesh[0]
    Y = mesh[1]
    F=test_f(X, Y)
    dx= X[1]-X[0]
    int_dx = simps(F,dx = dx,axis=0)
    dy = Y[1]-Y[0]
    int_dxdy = simps(int_dx, dx=dy, axis=0)
    print('2Dintegeral= ',int_dxdy)
    fig = plt.figure
    fig.Figure()
    plt.pyplot.contourf([X,Y],test_f)
    plt.pyplot.savefig('num_simps2D.pdf',dpi=200)
    return int_dxdy

if __name__=="__main__":
    x = np.linspace(0,1.1,101)
    m = np.zeros((np.shape(x)[0],np.shape(x)[0]))
    n = np.zeros((np.shape(x)[0],np.shape(x)[0]))
    for i in range(0,101):
        n[i,] = 2*np.min(x[i])*np.ones((101,))
        m[i,] = np.linspace(np.min(x[i])+i*1.1/100, 2*np.min(x[i])+1.1, 101)
    [X, Y] = [m, n]
    test_simpson_integeration2D([X, Y], test_f)