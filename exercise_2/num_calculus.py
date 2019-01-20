"""
This module contains functions for numerical calculus:
- first and second derivatives
- 1D integrals: Riemann, trapezoid, Simpson, 
  and Monte Carlo with uniform random numbers
"""

import numpy as np
#from sympy.mpmath import quad
from scipy import integrate
import sympy as sym
from numpy import linalg as LA

def eval_derivative(function, x, dx ):
    """ 
    This calculates the first derivative with
    symmetric two point formula, which has O(h^2)
    accuracy. See, e.g., FYS-4096 lecture notes.
    """
    return (function(x+dx)-function(x-dx))/2/dx

def eval_2nd_derivative(function, x, dx):
    """ 
    This calculates the second derivative with
    O(h^2) accuracy. See, e.g., FYS-4096 lecture 
    notes.
    """
    return (function(x+dx)+function(x-dx)-2.*function(x))/dx**2

def riemann_sum(x,function):
    """ 
    Left Rieman sum for uniform grid. 
    See, e.g., FYS-4096 lecture notes.
    """
    dx=x[1]-x[0]
    f=function(x)
    return np.sum(f[0:-1])*dx

def trapezoid(x,function):
    """ 
    Trapezoid for uniform grid. 
    See, e.g., FYS-4096 lecture notes.
    """
    dx=x[1]-x[0]
    f=function(x)
    return (f[0]/2+np.sum(f[1:-1])+f[-1]/2)*dx

def simpson_integration(x,function):
    """ 
    Simpson rule for uniform grid 
    See, e.g., FYS-4096 lecture notes.
    """
    f=function(x)
    N = len(x)-1
    dx=x[1]-x[0]
    s0=s1=s2=0.
    for i in range(1,N,2):
        s0+=f[i]
        s1+=f[i-1]
        s2+=f[i+1]
    s=(s1+4.*s0+s2)/3
    if (N+1)%2 == 0:
        return dx*(s+(5.*f[N]+8.*f[N-1]-f[N-2])/12)
    else:
        return dx*s

def simpson_nonuniform(x,function):
    """ 
    Simpson rule for nonuniform grid 
    See, e.g., FYS-4096 lecture notes.
    """
    f = function(x)
    N = len(x)-1
    h = np.diff(x)
    s=0.
    for i in range(1,N,2):
        hph=h[i]+h[i-1]
        s+=f[i]*(h[i]**3+h[i-1]**3+3.*h[i]*h[i-1]*hph)/6/h[i]/h[i-1]
        s+=f[i-1]*(2.*h[i-1]**3-h[i]**3+3.*h[i]*h[i-1]**2)/6/h[i-1]/hph
        s+=f[i+1]*(2.*h[i]**3-h[i-1]**3+3.*h[i-1]*h[i]**2)/6/h[i]/hph
    if (N+1)%2 == 0:
        s+=f[N]*(2.*h[N-1]**2+3.*h[N-2]*h[N-1])/6/(h[N-2]+h[N-1])
        s+=f[N-1]*(h[N-1]**2+3.*h[N-1]*h[N-2])/6/h[N-2]
        s-=f[N-2]*h[N-1]**3/6/h[N-2]/(h[N-2]+h[N-1])
    return s

def monte_carlo_integration(fun,xmin,xmax,blocks,iters):
    """ 
    1D Monte Carlo integration with uniform random numbers
    in range [xmin,xmax]. As output one gets the value of 
    the integral and one sigma statistical error estimate,
    that is, ~68% reliability. Two sigma and three sigma
    estimates are with ~95% and ~99.7% reliability, 
    respectively. See, e.g., FYS-4096 lecture notes. 
    """
    block_values=np.zeros((blocks,))
    L=xmax-xmin
    for block in range(blocks):
        for i in range(iters):
            block_values[block] += fun(x)
        block_values[block]/=iters
    I = L*np.mean(block_values)
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I, dI 

def volume_integral(x, N):
#   mpmath.dps = N
   f = lambda r,ra,rb: np.exp(-2*(r-ra))/np.pi/np.abs(r-rb)
#   I = quad(f,[-np.inf, x[0]],[-np.inf, x[1]],[-np.inf, x[2]], method = 'gauss-legendre')
   I = integrate.nquad(f, [[-np.inf,x[0]],[-np.inf, x[1]],[-np.inf, x[2]]])
   return I 

def Jacobian(name, fun):
    Vars = sym.symbols(name)
    f = sym.sympify(fun)
#   J = sym.zeros(len(f), np.size(Vars))
#    for i, fi in enumerate(f):
#        for j, s in enumerate(Vars):
#            J[i,j] = sym.diff(fi, s)
#    return J
    J = np.zeros((len(fun), len(name)))
    for i, fi in enumerate(f):
        for j, rj in enumerate(Vars):
            J[i,j] = sym.diff(fi,rj)
    return J

def Newton_root_search(test, r0, max_steps, dim, eps1, eps2):
    """
    N-dimensional Newton method in root searching with given unlinear grid
    of r0, h and dimension N
    """
    r = {}
    name = []
    value = []
    r.update({'r_1': r0 })
    name = list(r.keys())
    value = list(r.values())
    fun = []
    i = 1
    while i <= dim:
        fun.append('2*r_'+str(i)+'+1')
#        name.append('r_%d'% i+1)
        if i < dim:
            name.append('r_'+str(i+1))
        i += 1        
    J = Jacobian(name, fun)
    i = 0
    while i < max_steps: 
        dr = np.diag(-test(np.array(value))[i]/J)
        if LA.norm(dr, 2) < eps2 :  break
        elif np.linalg.det(J) == 0:  break
        value.append(np.array(value[i]) + dr)
        if LA.norm(test(np.array(value)[i+1]), 1) < eps1 : break
        r.update({name[i]: value[i]})
        i += 1       
    print("root is %s" % r)
    return value[-1]        
       
def Secant_root_search(test, r0, dr, max_steps, eps):
    """
    1-dimensional Secant method in root searching with linear grid 
    """
    r = []
    r.append(r0)
    r.append(r0+dr)
    i = 1
    while i < max_steps:
        r.append((np.array(r)[i]-(np.array(r)[i] - np.array(r)[i-1])*test(np.array(r)[i]))/(test(np.array(r)[i])-test(np.array(r)[i-1])))
        dr = np.abs(r[i+1] - r[i])
        if  dr < eps or r[i+1] == np.NaN : break   
        i += 1
        print("root is %s" % r[-1])
    return r[-1]

def N_conjugate_gradient(test, r0, max_steps, dim, eps):
    """
    N-dimensional conjugate gradient method
    """
    r = {}
    name = []
    value = []
    r.update({'r_1': r0 })
    name = list(r.keys())
    value = list(r.values())
    fun = []
    i = 1
    while i <= dim:
        fun.append('2*r_'+str(i)+'+1')
#        name.append('r_%d'% i+1)
        if i < dim:
            name.append('r_'+str(i+1))
        i += 1        
    A = Jacobian(name, fun)
    e = A*np.array(value)[0].T
    p = e
    eo = e.T*e
    i = 1
    while i <  max_steps:
        Ap = A*p
        alpha = eo/(p.T*Ap)
        np.array(value)[0] = np.array(value)[0] + np.diag(alpha * p)
        e = e - alpha * Ap
        en = e.T*e
        if np.std(np.diag(en)) < eps: break
        p = e + en/eo * p
        eo = en
        i += 1
    print('root is %s:' % value)
    return value

def Steepest_gradient_descent(test, r0, max_steps, dim, eps):
    r = {}
    name = []
    value = []
    gamma = np.zeros(dim,)
    r.update({'r_1': r0 })
    name = list(r.keys())
    value = list(r.values())
    fun = []
    i = 1
    while i <= dim:
        fun.append('2*r_'+str(i)+'+1')
#        name.append('r_%d'% i+1)
        if i < dim:
            name.append('r_'+str(i+1))
        i += 1        
    i = 0
    while i < max_steps-1:
        g = trapezoid(np.array(value)[0],test)
        gamma[i] = 0.1/(np.abs(g)+1)
        value.append(np.array(value)[i] - gamma[i]*g) 
        dr = np.array(value)[i+1] - np.array(value)[i]
        if LA.norm(dr, 2) < eps :  break
        r.update({name[i+1]: value[i+1]})
        i += 1       
    print("root is %s" % r)
    return value[-1]   

""" Test routines for unit testing """
def test_first_derivative(tolerance=1.0e-3):
    """ Test routine for first derivative of f"""
    x = 0.8
    dx = 0.01
    df_estimate = eval_derivative(test_fun,x,dx)
    df_exact = test_fun_der(x)
    err = np.abs(df_estimate-df_exact)
    working = False
    if (err < tolerance):
        print('First derivative is OK')
        working = True
    else:
        print('First derivative is NOT ok!!')
    return working

def test_second_derivative(tolerance=1.0e-3):
    """ Test routine for first derivative of f"""
    x = 0.8
    dx = 0.01
    df_estimate = eval_2nd_derivative(test_fun,x,dx)
    df_exact = test_fun_der2(x)
    err = np.abs(df_estimate-df_exact)
    working = False
    if (err<tolerance):
        print('Second derivative is OK')
        working = True
    else:
        print('Second derivative is NOT ok!!')
    return working

def test_riemann_sum(tolerance=1.0e-2):
    """ Test routine for Riemann integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,100)
    int_estimate = riemann_sum(x,test_fun2)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Riemann integration is OK')
        working = True
    else:
        print('Riemann integration is NOT ok!!')
    return working

def test_trapezoid(tolerance=1.0e-4):
    """ Test routine for trapezoid integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,100)
    int_estimate = trapezoid(x,test_fun2)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Trapezoid integration is OK')
        working = True
    else:
        print('Trapezoid integration is NOT ok!!')
    return working

def test_simpson_integration(tolerance=1.0e-6):
    """ Test routine for uniform simpson integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,100)
    int_estimate = simpson_integration(x,test_fun2)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Uniform simpson integration is OK')
        working = True
    else:
        print('Uniform simpson integration is NOT ok!!')
    return working

def test_simpson_nonuniform(tolerance=1.0e-6):
    """ Test routine for nonuniform simpson integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,100)
    int_estimate = simpson_nonuniform(x,test_fun2)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Nonuniform simpson integration is OK')
        working = True
    else:
        print('Nonuniform simpson integration is NOT ok!!')
    return working

def test_monte_carlo_integration():
    """ 
    Test routine for monte carlo integration.
    Testing with 3*sigma error estimate, i.e., 99.7%
    similar integrations should be within this range.
    """
    a = 0
    b = np.pi/2
    blocks = 100
    iters = 1000
    int_est, err_est = monte_carlo_integration(test_fun2,a,b,blocks,iters)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_est-int_exact)
    working = False
    if (err<3.*err_est):
        print('Monte Carlo integration is OK')
        working = True
    else:
        print('Monte Carlo integration is NOT ok!!')
    return working
    

def test_Newton_root_search():
    N = 100
    x = np.zeros(N,)
    x[0] = 0.1
    h = np.log10(1e7+1)/(N-1) 
    for i in range(1, N):
        x[i] = x[0]*(np.exp(i*h))
    root = Newton_root_search(test = test_fun_Newton, r0 = x, max_steps = 100, dim = N, eps1 = 0.01, eps2 = 1e-5)
    err = LA.norm(test_fun_Newton(root),2)
    tol = 0.00000001
    working = False
    if err < tol:
        print('Newton Method with root: %s is OK' % root)
        working = True
    else:
        print('Newton Method is NOT ok!!')
    return working

def test_Secant_root_search():
    x = -2
    root = Secant_root_search(test = test_fun_Secant, r0 = x, dr = 0.001, max_steps = 1000, eps = 0.001)
    err = np.abs(test_fun_Secant(root))
    tol = 1e-5
    working = False
    if err < tol:
        print('Secant Method with root: %d is OK' % root)
        working = True
    else:
        print('Secant Method is NOT ok with err: %s !!' % err)
    return working

def test_N_conjugate_gradient():
    N = 100
    x = np.zeros(N,)
    x[0] = 0.1
    h = np.log10(1e7+1)/(N-1) 
    for i in range(1, N):
        x[i] = x[0]*(np.exp(i*h))
    root = N_conjugate_gradient(test = test_fun_Newton, r0 = x, max_steps = 100, dim = N, eps = 1e-5)
    err = LA.norm(test_fun_Newton(np.array(root)),2)
    tol = 1e-5
    working = False
    if err < tol:
        print('Conjugate Method with root: %d is OK' % root)
        working = True
    else:
        print('Conjugate Method is NOT ok with err: %s !!' % err)
    return working   

def test_Steepest_gradient_descent():
    N = 100
    x = np.zeros(N,)
    x[0] = 0.1
    h = np.log10(1e7+1)/(N-1) 
    for i in range(1, N):
        x[i] = x[0]*(np.exp(i*h))
    root = Steepest_gradient_descent(test = test_fun_Newton, r0 = x, max_steps = 100, dim = N, eps = 1e-5)
    err = LA.norm(test_fun_Newton(np.array(root)),2)
    tol = 1e-5
    working = False
    if err < tol:
        print('Steepest gradient descent Method with root: %d is OK' % root)
        working = True
    else:
        print('Steepest gradient descent Method is NOT ok with err: %s !!' % err)
    return working    
    
""" Analytical test function definitions """
def test_fun(x):
    """ This is the test function used in unit testing"""
    return np.exp(-x)

def test_fun_der(x):
    """ 
    This is the first derivative of the test 
    function used in unit testing.
    """
    return -np.exp(-x)

def test_fun_der2(x):
    """ 
    This is the second derivative of the test 
    function used in unit testing.
    """
    return np.exp(-x)

def test_fun2(x):
    """
    sin(x) in range [0,pi/2] is used for the integration tests.
    Should give 1 for the result.
    """
    return np.sin(x)

def test_fun2_int(a,b):
    """
    Integration of the test function (test_fun2).
    """
    return -np.cos(b)+np.cos(a)

def test_fun_Newton(x):
    """
    Root search of the test function with Newton method
    """
    return -x**2+1

def test_fun_Secant(x):
    """
    Root search of the test function with Newton method
    """
    return 2*x+1

""" Tests performed in main """
def main():
    """ Performing all the tests related to this module """
    test_first_derivative()
    test_second_derivative()
    test_riemann_sum()
    test_trapezoid()
    test_simpson_integration()
    test_simpson_nonuniform()
    test_monte_carlo_integration()
    test_Newton_root_search()
    test_Secant_root_search()
    test_N_conjugate_gradient()
    test_Steepest_gradient_descent()   

if __name__=="__main__":
    main()
