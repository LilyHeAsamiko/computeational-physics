# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 05:56:32 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
#from spline_class import spline
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D

def fun_Simpson2D(x, y):
    return (x+y)*np.exp(-0.5*(x**2+y**2)**0.5)

def test_Simpson2D():
    x = np.linspace(0,1.1,101)
    y = np.zeros((101,))
    I = np.zeros((101,))
    for i in range(len(x)):
        if x[i] <= 0.5:
            y[i] = 2*x[i]
        elif 0.5 < x[i] and x[i] <= 1.1:
            y[i] = 1
        [X, Y] = np.meshgrid(x,  y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        z = fun_Simpson2D(X, Y)
        I[i] = simps(simps(z, dx = dx), dx = dy)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    print('Integeral by Simpson:',I)   
    ax1.plot(x, I, '-')
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_wireframe(x, y, I)
    plt.pyplot.savefig('num_simps3D.pdf',dpi=200)
#    plt.plot_wire(x, z, I)
    plt.contourf(x, z, I)
    plt.show()
    return I

#def test_simpson_integeration2D(mesh, test_f):
#    X = mesh[0]
#    Y = mesh[1]
#    F=test_f(X, Y)
#    dx= X[1]-X[0]
#    int_dx = simps(F,dx = dx,axis=0)
#    dy = Y[1]-Y[0]
#    int_dxdy = simps(int_dx, dx=dy, axis=0)
#    print('2Dintegeral= ',int_dxdy)
#    fig = plt.figure
#    fig.Figure()
#    plt.pyplot.contourf([X,Y],test_f)
#    plt.pyplot.savefig('num_simps2D.pdf',dpi=200)
#    return int_dxdy#

if __name__=="__main__":
#    test_Simpson2D()
    x = np.linspace(0,1.1,22)
    y = np.linspace(0,1,22)
    [X, Y] = np.meshgrid(x, y)
    F = fun_Simpson2D(X, Y)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, F)
    plt.savefig('area_simps3D.pdf',dpi=200)
    
#    spl2d = spline(x=x, y=y, f =F, dims =2)
    xp = np.linspace(0, 0.8, 101)
    yp = np.zeros((len(xp),)) 
#    f = np.zeros((len(xp), len(xp)))
    ff = np.zeros((len(xp)))
    for i in range(1,len(xp)):
        if xp[i] <= 0.5:
            yp[i] = 2*xp[i]
        elif 0.5 < xp[i] and xp[i] <= 1.1:
            yp[i] = 1
        [Xp, Yp] = np.meshgrid(xp[0:i],  yp[0:i])
        dxp = xp[1] - xp[0]
        dyp = yp[1] - yp[0]
        f = fun_Simpson2D(Xp, Yp)
        ff[i] = simps(simps(f, dx = dxp),dx = dyp)
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(311)
    print('Integeral by Simpson:',ff[-2])   
    ax1.plot(xp, ff, '-')
    ax2 = fig2.add_subplot(312, projection='3d')
    ax2.plot_wireframe(Xp, Yp, f)
#    ax2.xlabel('X')
#    ax2.ylabel('Y')
#    ax2.title('X_Y area with a1 and a2')
    plt.savefig('num_simps3D.pdf',dpi=200)
#    plt.plot_wire(x, z, I)
    ax3 = fig2.add_subplot(313, projection='3d')
    ax3.contour(Xp, Yp, f)
    plt.savefig('num_simps3Dcontour.pdf',dpi=200)
    plt.show()
#    m = np.zeros((np.shape(x)[0],np.shape(x)[0]))
#    n = np.zeros((np.shape(x)[0],np.shape(x)[0]))
#    for i in range(0,101):
#        if x[i] <= 0.5:
#            n[i,] = 2*np.min(x[i])*np.ones((101,))
#        elif 0.5 < x[i] and x[i] <= 1.1:
#            n[i,] = np.ones((101,))
#        m[i,] = np.linspace(np.min(x[i])+i*1.1/100, 2*np.min(x[i])+1.1, 101)
#    [X, Y] = [m, n]
#    test_simpson_integeration2D([X, Y], test_f)