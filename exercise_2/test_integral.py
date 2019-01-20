# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 08:21:01 2019

@author: user
"""
import numpy as np
from num_calculus import simpson_integration, volume_integral 
import matplotlib as plt
from numpy import linalg as LA

np.seterr(divide='ignore', invalid='ignore')

def MSE(x):
    return np.sqrt(sum((x-np.mean(x))**2))

def test_f1(x):
    return np.exp(-2*x)*(x**2)

def der_f1(x):
    return -(2*x**2+2*x+1)*np.exp(-2*x)/4

def test_f2(x):
#    i = 0
#    for val in x:
#        if val == 0:
#            x[i] = 0.000001
    return np.sin(x)/x

def der_f2(x):
    return x-x**3/(3*(3*2*1))+x**5/(5*(5*4*3*2*1))

def test_f3(x):
    return np.exp(np.sin(x**3))

def der_f3(x):
    return -1.297359854043857*10**52-2.247093182904103j*10**52

def test_f4(x):
    return x*np.exp(-(x**2+y**2))

def test_f4_1(x):
    return x*np.exp(-(x**2))

def test_f4_2(y):
    return np.exp(-(y**2))

def der_f4_1(x):
    return -0.5*np.exp(-(x**2))

def der_f4_2(y):
    return 0.5*np.sqrt(np.pi)
 

def test_simpson_integeration1(x, test_f) :
    i = 2
    xx = []
    err = []
    I = []
    II = der_f1(np.max(x))-der_f1(np.min(x))
    while i< 52:
        xx.append(np.linspace(np.min(x),np.max(x), i))
        I.append(simpson_integration(xx[i-2], test_f))
        err.append(LA.norm(np.array(I[0:i])-II, 2))
        i += 1
    fig1 = plt.figure
    fig1.Figure()
    ax1 = plt.pyplot.subplot(121)
    ax2 = plt.pyplot.subplot(122)
    ax1.plot(np.array(xx[-2]), I, 'r', np.array(xx[-2]), II*np.ones((i-2,)), 'g')
    ax1.legend(['num_integral','ana_integral'])
    ax1.set_title('num and ana integeral of exp(-2*x)*(x**2)')
    plt.pyplot.savefig('num_and_ana_int_fun1.pdf',dpi=200)
    ax2.plot(np.array(xx[-2]), err)
    ax2.legend(['norm2'])
    ax2.set_title('norm2 of exp(-2*x)*(x**2)')
    plt.pyplot.savefig('num_and_ana_int_fun1_error.pdf',dpi=200)    
    return err, I

def test_simpson_integeration2(x, test_f) :
    i = 2
    xx = []
    err = []
    I = []
    II = der_f2(np.max(x))-der_f2(np.min(x))
    while i< 52:
        xx.append(np.linspace(np.min(x),np.max(x), i))
        I.append(simpson_integration(xx[i-2], test_f))
        err.append(LA.norm(np.array(I[0:i])-II, 2))
        i += 1
    fig2 = plt.figure
    fig2.Figure()
    ax21 = plt.pyplot.subplot(121)
    ax22 = plt.pyplot.subplot(122)
    ax21.plot(np.array(xx[-2]), I, 'r', np.array(xx[-2]), II*np.ones((i-2,)), 'g')
    ax21.legend(['num_integral','ana_integral'])
    ax21.set_title('num and ana integeral of sin(x)/x')
    plt.pyplot.savefig('num_and_ana_int_fun2.pdf',dpi=200)    
    ax22.plot(np.array(xx[-2]), err)
    ax22.legend(['norm2'])
    ax22.set_title('norm2 of integeral of sin(x)/x')
    plt.pyplot.savefig('num_and_ana_int_fun2_error.pdf',dpi=200)
    return err, I

def test_simpson_integeration3(x, test_f) :
    i = 2
    xx = []
    err = []
    I = []
  
    while i< 52:
        xx.append(np.linspace(np.min(x),np.max(x), i))
        I.append(simpson_integration(xx[i-2], test_f))
        err.append(LA.norm(np.array(I[0:i])-der_f3(0), 2))
        i += 1        
    fig3 = plt.figure
    fig3.Figure()
    ax31 = plt.pyplot.subplot(311)
    ax32 = plt.pyplot.subplot(312)
    ax33 = plt.pyplot.subplot(313)
    cx1 =[x1.real for x1 in I]
    cy1 =[y1.real for y1 in I]    
    ax31.plot(cx1, cy1)
    ax31.legend(['num_integral'])
    ax31.set_title('num integeral of exp(sin(x**3))')
    plt.pyplot.savefig('num_int_fun3.pdf',dpi=200)
    cnums = der_f3(0)*np.ones((i-2,))
    cx =[x.real for x in cnums]
    cy =[y.real for y in cnums]
    ax32.plot(cx, cy, 'o')
    ax32.legend(['ana_integral'])
    ax32.set_title('ana integeral of exp(sin(x**3))')
    plt.pyplot.savefig('ana_int_fun3.pdf',dpi=200)
    cx2 =[x2.real for x2 in err]
    cy2 =[y2.real for y2 in err]
    ax33.plot(cx2, cy2)
    ax33.legend(['norm2'])
    ax33.set_title('norm2 of integeral of exp(sin(x**3))')
    plt.pyplot.savefig('ana_num_int_fun3_error.pdf',dpi=200)
    return err, I


def test_simpson_integeration2D(x, y, test_f) :
    i = 2
    xx = []
    yy = []
    err = []
    I1 = []
    I2 = []
    I = []
    II = (der_f4_1(np.max(x))-der_f4_1(np.min(x)))*der_f4_2(y)
    while i< 52:
        xx.append(np.linspace(np.min(x),np.max(x), i))
        yy.append(np.linspace(np.min(y),np.max(y), i))
        I1.append(simpson_integration(xx[i-2], test_f4_1))
        I2.append(simpson_integration(xx[i-2], test_f4_2)) 
        I.append(I1[i-2]*I2[i-2])
        err.append(LA.norm(np.array(I[0:i])-II, 2))
        i += 1        
    fig4 = plt.figure
    fig4.Figure()
    ax41 = plt.pyplot.subplot(121)
    ax42 = plt.pyplot.subplot(122)
    ax41.plot(np.array(xx[-2]), I, 'r', np.array(xx[-2]), II*np.ones((i-2,)) , 'g')
    ax41.legend(['num_integral','ana_integral'])
    ax41.set_title('num and ana integeral of x*exp(-(x**2+y**2))')
    plt.pyplot.savefig('num_and_ana_int_fun4.pdf',dpi=200)    
    ax42.plot(np.array(xx[-2]), err)
    ax42.legend(['norm2'])
    ax42.set_title('norm2 of integeral of x*exp(-(x**2+y**2))')
    plt.pyplot.savefig('num_and_ana_int_fun4_error .pdf',dpi=200)
    return err, I

def test_volume_integeral(x, N):
    I = np.zeros(N,)
    II = np.zeros(N,)
    err = np.zeros(N,)
    for i in range(0, N):
        I[i][:] = volume_integral(x[i][:], N)
        R = np.abs(x[i][1]-x[i][2])
        II[i] = (1-(1+R)*np.exp(-x*R))/R
        err[i] = LA.norm(I[i]-II[i], 2)
    fig5 = plt.figure
    fig5.Figure()
    ax51 = plt.pyplot.subplot(121)
    ax52 = plt.pyplot.subplot(122)
    ax51.plot(I, II, 'o')
    ax51.set_title('num v.s. ana volumetric integral')
    plt.pyplot.savefig('volumetric_integral.pdf',dpi=200)    
    ax52.plot(np.linspace(1,5,5), err)
    ax52.set_title('norm2 of volumetric_integral')
    plt.pyplot.savefig('norm_of_volumetric_integral_error.pdf',dpi=200)
    return err, I
    
       
if __name__=="__main__":
    x = np.linspace(0, 1000000, 10)
    err1,I1 = test_simpson_integeration1(x, test_f1)
    x = np.linspace(0.000001, 1, 10)
    err2,I2 = test_simpson_integeration2(x, test_f2)
    x = np.linspace(0, 5, 10)
    err3,I3 = test_simpson_integeration3(x, test_f3)
    y = np.linspace(-2, 2, 10)
    x = np.linspace(0, 2, 10)
    err4,I4 = test_simpson_integeration2D(x, y, test_f4)
    N = 5
    x = [[0,0,0],[1,1,1],[1,2,3],[-1,-3,2],[11,-30,45]]
    err5,I5 = test_volume_integeral(x, N)

    