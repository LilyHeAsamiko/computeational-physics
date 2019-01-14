# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:11:39 2019

@author: user
"""
import numpy as np
from num_calculus import  test_first_derivative, test_second_derivative
from num_calculus import  Riemann_sum,  trapezoid_sum, Simpson_sum
from num_calculus import monte_carlo_integration

if __name__=="__main__":
        
    def fun(x):
        return  3*x**2 
    x = 0.8
    dx = 0.001
    
    der1 = test_first_derivative(fun, x, dx);
    print(der1)
    
    der2 = test_second_derivative(fun, x, dx);
    print(der2)    


    def fun2(x):
        return  np.sin(x)
    x0 = 0.9
    dx = np.pi/2
    n = 100    
    x2 = np.linspace(x0, dx, n)
    
    I_Riemann = Riemann_sum(fun2, x2, dx, n)
    
    I_trapezoid =  trapezoid_sum(fun2, x2, dx, n)
    
    I_Simpson = Simpson_sum(fun2, x2, dx, n)
    
    I, dI = monte_carlo_integration(fun2, 0.0, np.pi/2, 10, 100)
    print(I, '+/-', 2*dI)