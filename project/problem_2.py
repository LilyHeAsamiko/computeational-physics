# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 07:56:19 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D

def diff_field(dl, r, b, I):
    dB = b*I*np.cross(dl, r)/(LA.norm(r))**3
    return dB

def Bx(x, a, b, I):
    B = b*4*np.pi*I*a**2/(2*(x**2+a**2)**(3/2))
    return B

def field_yz_loop_x(O, a, b, I, r):
    diff = np.array(r) - np.array(O)    
#    if r[1] >= O[1]:
#        P1_y = np.abs(O[1]-r[1])*a/LA.norm(diff)+O[1]
#    else:
#        P1_y = O[1] - np.abs(O[1]-r[1])*a/LA.norm(diff)
#    if r[2] >= O[2]:
#        P1_z = np.abs(O[2]-r[2])*a/LA.norm(diff)+O[2]
#    else:
#        P1_z = O[1] - np.abs(O[2]-r[1])*a/LA.norm(diff)        
#    P2_y = 2*O[1]-P1_y
#    P2_z = 2*O[2]-P1_z
    B = diff_field([0, diff[2]/np.sqrt(2), -diff[1]/np.sqrt(2)], diff, b, I) + diff_field([0, -diff[2]/np.sqrt(2), diff[1]/np.sqrt(2)], diff, b, I)  
    return B

def field_loop_2(O, a, b, I, r):
    OO = np.array([0, O[1], O[2]])
    rr = np.array([r[0]-O[0], r[1], r[2]])
    B_x = field_yz_loop_x(OO, a, b, I, rr) + field_yz_loop_x(OO + 4*a, a, b, I, rr)
    return B_x
       
if __name__=="__main__":
    a = 1 # radius of the loop
    b = 10**-7 #b = uo/4pi
    I = 10
    r = np.array([1, 1, 1])
    x = np.linspace(r[0], 20, 100)
    B1 = Bx(x, a, b, I)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(B1,np.zeros((100)),np.zeros((100)))
    plt.savefig('B_x_3D.pdf',dpi=200)
    O = np.array([1, 1, 1])
    B = np.zeros((len(x)))
    for i in range(len(x)):
        r = np.array([x[i], O[1], O[2]]) 
        print(field_yz_loop_x(O, a, b, I, r))
        B[i] = field_yz_loop_x(O, a, b, I, r)[0]
    err = B-B1
    print('numerical and analysis difference:')
    print(err)
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    B = np.zeros((len(x), len(y)))
    O = np.array([1, 2, 3])
    for i in range(len(x)): # i on y axis
        for j in range(len(y)): # j on z axis
            B[i, j] = field_loop_2(O, a, b, I, r)[0] 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    [X, Y] = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, B)
    plt.savefig('B_3D.pdf',dpi=200)
    plt.show()
            
