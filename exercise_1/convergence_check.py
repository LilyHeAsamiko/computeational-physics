# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:18:31 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from num_calculus import  Riemann_sum, trapezoid_sum, Simpson_sum

def fun(x):
        return  np.sin(x)
    
def fun2(x):
        return  np.exp(-x)    

I1_Riemann = []
I1_trapezoid = []
I1_Simpson = []
err11 = []
err12 = []
err13 = []
I2_Riemann = []
I2_trapezoid = []
I2_Simpson = []
err21 = []
err22 = []
err23 = []

x0 = 0
dx = np.pi/2
n = 1000
i = 0
x = np.linspace(x0, dx, n)
f = fun(x).reshape(1,1000);
g = fun2(x).reshape(1,1000);

    
while i< n:       

    I1_Riemann.append(Riemann_sum(fun, x, dx, n))
    I2_Riemann.append(Riemann_sum(fun2, x, dx, n))
    I1_trapezoid.append(trapezoid_sum(fun, x, dx, n))
    I2_trapezoid.append(trapezoid_sum(fun2, x, dx, n))
    I1_Simpson.append(Simpson_sum(fun, x, dx, n))
    I2_Simpson.append(Simpson_sum(fun2, x, dx, n))
    err11.append(np.abs((I1_Riemann[i]-np.cos(x[i])+np.cos(x[0]))))
    err21.append(np.abs((I2_Riemann[i]+np.exp(-x[i])-np.cos(-x[0]))))
    err12.append(np.abs(I1_trapezoid[i]-np.cos(x[i])+np.cos(x[0])))
    err22.append(np.abs((I2_trapezoid[i]+np.exp(-x[i])-np.cos(-x[0]))))  
    err13.append(np.abs(I1_Simpson[i]-np.cos(x[i])+np.cos(x[0])))   
    err23.append(np.abs((I2_Simpson[i]+np.exp(-x[i])-np.cos(-x[0]))))

    i += 1

fig1 = plt.figure() # - things to consider with figures, e.g., size # - fig = plt.figure(figsize=(width, height)) # - "golden ratio": width=height*(sqrt(5.0)+1.0)/2.0 
ax1 = fig1.add_subplot(211)
# plot and add label if legend desired 
x = x.reshape(1,1000)
plt.plot(x,f,'r',x,np.array(I1_Riemann).reshape(1,1000),'g',x,np.array(I1_trapezoid).reshape(1,1000),'b',x,np.array(I1_Simpson).reshape(1,1000),'k')
# plot legend 
plt.legend(['f(x)=sin(x)','Integral by Riemann','Integral by Trapezoid','Integral by Simpson'])
# set axes labels and limits 
ax1.set_xlabel(r'$x$') 
ax1.set_ylabel(r'$f(x)$') 
ax1.set_xlim(x.min(), x.max())
fig1.tight_layout(pad=1)
# save figure as pdf with 200dpi resolution 
plt.savefig('f(x).pdf',dpi=200) 
plt.show()

ax2 = fig1.add_subplot(212)
# plot and add label if legend desired 
plt.plot(x,np.array(err11).reshape(1,1000),x,np.array(err12).reshape(1,1000),x,np.array(err13).reshape(1,1000))
# plot legend 
plt.legend(['Convergence of Riemann', 'Convergence of Trapezoid', 'Convergence of Simpson'])
# set axes labels and limits 
ax2.set_xlabel(r'$x$') 
ax2.set_ylabel(r'$f(x)$') 
ax2.set_xlim(x.min(), x.max())
fig1.tight_layout(pad=1)
# save figure as pdf with 200dpi resolution 
plt.savefig('f(x)convergence.pdf',dpi=200) 
plt.show()

fig32 = plt.figure() # - things to consider with figures, e.g., size # - fig = plt.figure(figsize=(width, height)) # - "golden ratio": width=height*(sqrt(5.0)+1.0)/2.0 
ax3 = fig2.add_subplot(211)
# plot and add label if legend desired 
#x = x.reshape(1,1000)
plt.plot(x,g,'r',x,np.array(I2_Riemann).reshape(1,1000),'g',x,np.array(I2_trapezoid).reshape(1,1000),'b',x,np.array(I2_Simpson).reshape(1,1000),'k')
# plot legend 
plt.legend(['g(x)=exp(-x)','Integral by Riemann','Integral by Trapezoid','Integral by Simpson'])
# set axes labels and limits 
ax1.set_xlabel(r'$x$') 
ax1.set_ylabel(r'$g(x)$') 
ax1.set_xlim(x.min(), x.max())
fig2.tight_layout(pad=1)
# save figure as pdf with 200dpi resolution 
plt.savefig('g(x).pdf',dpi=200) 
plt.show()

ax4 = fig2.add_subplot(212)
# plot and add label if legend desired 
plt.plot(x,np.array(err21).reshape(1,1000),x,np.array(err22).reshape(1,1000),x,np.array(err23).reshape(1,1000))
# plot legend 
plt.legend(['Convergence of Riemann', 'Convergence of Trapezoid', 'Convergence of Simpson'])
# set axes labels and limits 
ax4.set_xlabel(r'$x$') 
ax4.set_ylabel(r'$g(x)$') 
ax4.set_xlim(x.min(), x.max())
fig2.tight_layout(pad=1)
# save figure as pdf with 200dpi resolution 
plt.savefig('g(x)convergence.pdf',dpi=200) 
plt.show()