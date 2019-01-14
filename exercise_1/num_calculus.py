# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 07:17:06 2019

@author: user
"""
import numpy as np
def eval_derivative(fun, x, dx):
    #evaluate the first derivative
    df = (fun(x+dx) - fun(x-dx))/(2*dx)
    return df

def test_first_derivative(fun, x, dx):
    #test the first derivative
    der = eval_derivative(fun, x, dx);
    return der
    
def eval_2nd_derivative(fun, x, dx):
    ddf = (fun(x+dx) + fun(x-dx) -2*fun(x))/dx**2 
    return ddf

def test_second_derivative(fun, x, dx):
    #test the first derivative
    der = eval_2nd_derivative(fun, x, dx)
    return der

def eval_Riemann(fun, x, dx):
    #compartment of the trapezoid
    I = fun(x)*dx
    return I
    
def Riemann_sum(fun, x, dx, n):
    #evaluate the first derivative
    I = eval_Riemann(fun, x, dx)
    I = sum(I)/n
    print(I)
    return I

def eval_trapezoid(fun, x, dx):
    #compartment of the trapezoid
    I = 0.5*(fun(x) + fun(x+dx))*dx
    return I
    
def trapezoid_sum(fun, x, dx, n):
    #evaluate the first derivative
    I = eval_trapezoid(fun, x, dx)
    I = sum(I[0:-1])/n
    print(I)
    return I

def eval_Simpson(fun, x, dx, n):
    #compartment of the trapezoid
    if n % 2 == 0: 
        x = np.linspace(x[0],2*dx,n/2)
        I = dx/3*(fun(x[:]-dx) + 4*fun(x[:])+ fun(x[:]+dx))
        return I
    else:
        x = np.linspace(x[0],2*dx,(n-1)/2)
        I = dx/3*(fun(x[:]-dx) + 4*fun(x[:])+ fun(x[:]+dx))
        dI = dx/12*(-fun(x[:]+dx)+8*fun(x[:]+2*dx)+5*fun(x[:]+3*dx))
        return I+dI
    
def Simpson_sum(fun, x, dx, n):
    #evaluate the first derivative
    I = eval_Simpson(fun, x, dx, n)
    I = 3*sum(I)/(2*n)
    print(I)
    return I

def monte_carlo_integration(fun,xmin,xmax,blocks,iters): 
# implement the Monte Carlo integration
    block_values=np.zeros((blocks,))
    # initiate the dimensional vevtor V with blocks number of length.
    L=xmax-xmin
    #get the range L
    for block in range(blocks):
        for i in range(iters):
            # go through the whole V within i iterations, 
            x = xmin+np.random.rand()*L
            # set x walks radomly with L
            block_values[block]+=fun(x)
            # Let iterate shuthe block and x together 
        block_values[block]/=iters
        #iterate the V 
    I = L*np.mean(block_values)
    #calculate the mean value of V
    dI = L*np.std(block_values)/np.sqrt(blocks)
    #calculate the standard deviationof V
    return I,dI
