# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 07:56:19 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps

def diff_field(dl, r, b, I):
    dB = b*I*np.cross(dl, r)/(LA.norm(r))**3
    return dB

def Bx(x, a, b, I):
    B = b*4*np.pi*I*a**2/(2*(x**2+a**2)**(3/2))
    return B

def field_yz_loop_x(O, a, b, I, P):
    x = np.repeat(O[0],50)
    y = np.linspace(O[1]-a, O[1]+a, 50)
#    (y-O[1])**2 + (z-O[2])**2= a**2
    z = -np.sqrt(a**2 - (y-O[1])**2)
    r =np.array([x, y, z])
    diff = np.repeat(P,50).reshape(3,50) - r
    n = r - np.repeat(O,50).reshape(3,50)
    dl = np.array([x, n[2, ]/a, -n[1, ]/a])
    dB = diff_field(np.transpose(dl), np.transpose(diff), b, I)
    B1 = sum(dB)
    
    r = np.array([x, y, -z])
    diff = np.repeat(P,50).reshape(3,50) - r
    n = r - np.repeat(O,50).reshape(3,50)
    dl = np.array([x, n[2, ]/a, -n[1, ]/a])
    dB = diff_field(np.transpose(dl), np.transpose(diff), b, I)
    B2 = sum(dB)
    
    B = B1 + B2
#    if rr[1] >= O[1]:
#        Q1_y = np.abs(O[1]-rr[1])*a/LA.norm(diff)+O[1]
#    else:
#        Q1_y = O[1] - np.abs(O[1]-rr[1])*a/LA.norm(diff)
#    if rr[2] >= O[2]:
#        Q1_z = np.abs(O[2]-r[2])*a/LA.norm(diff)+O[2]
#    else:
#        Q1_z = O[1] - np.abs(O[2]-rr[1])*a/LA.norm(diff)        
#    Q2_y = 2*O[1]-P1_y
#    Q2_z = 2*O[2]-P1_z
#    B = diff_field(, diff, b, I) + diff_field([0, -diff[2]/np.sqrt(2), diff[1]/np.sqrt(2)], diff, b, I)  
    return B

def field_loop_2(O, a, b, I, P):
    OO = np.array([O[0]+4*a, O[1], O[2]])
    B = field_yz_loop_x(O, a, b, I, P) + field_yz_loop_x(OO, a, b, I, P)
    return B
       
if __name__=="__main__":
    a = 1 # radius of the loop
    b = 10**-7 #b = uo/4pi
    I = 10
    O = np.array([0, 0, 0])
    x = np.linspace(O[0], 20, 10)
    B1 = Bx(x, a, b, I)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(B1,np.zeros((len(x))),np.zeros((len(x))))
    plt.savefig('B_x_3D.pdf',dpi=200)
    B = np.zeros((len(x)))
    for i in range(len(x)):
        P = np.array([x[i], O[1], O[2]])
        print(field_yz_loop_x(O, a, b, I, P))
        B[i] = field_yz_loop_x(O, a, b, I, P)[0]
    err = B-B1
    print('numerical and analysis difference of B along axis x:')
    print(err)
    y = np.linspace(-5, 5, 50)
    z = np.linspace(-5, 5, 50)
    [Y, Z] = np.meshgrid(y, z)
    B = np.zeros((len(y),len(z),3))
    O = np.array([0, 0, 0])
    for k in range(len(x)):
        for i in range(len(y)): # i on y axis
            for j in range(len(z)): # j on z axis
                P = np.array([x[k], i, j])
                B[i, j,] = field_loop_2(O, a, b, I, P)
    print('B_3D_y_z: ')
    print(B[3])
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211, projection='3d')
    ax1.plot_wireframe(B[1], B[2], B[3])
    ax2 = fig2.add_subplot(212, projection='3d')
    ax2.contour(B[1], B[2], B[3])    
    plt.savefig('B_yz_3D.pdf',dpi=200)
    plt.show()
            
