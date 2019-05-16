# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 07:56:19 2019

@author: LilyHeAsamiko
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps

# set O
class loop:
    def __init__(self,*args,**kwargs):
        self.O = kwargs['O']
        self.OO = kwargs['OO']
    #    self.sys_dim = kwargs['dim']
    #    self.coords = kwargs['coords']    
    def get_O(self,**kwargs):
        if self.O in kwargs.items():
            return 'O'
        #return 'kwargs[O]'
        if self.OO in kwargs.items():
            return 'OO'
# 3D Biot-Savart magnetic law
def diff_field(dl, r_hat, r, b, I):
    dB = b*I*np.cross(dl, r_hat)/(LA.norm(r))**2
# x,y,z direction vectors
    dBx = dB[:,0]
    dBy = dB[:,1]
    dBz = dB[:,2]

    return dB, dBx, dBy, dBz

# analytical solution of magnetic field on x field
def Bx(x, a, b, I):
    B_x = b*4*np.pi*I*a**2/(2*(x**2+a**2)**(3/2))
    return B_x

def field_yz_loop_x(O, a, b, I, P):
# electric field_x(O, a, b, I, P):
# electric circle coordinates: (y-O[1])**2 + (z-O[2])**2= a**2
#    xx = np.repeat(O[0],len(P)) # y_z plane
#    xx = np.repeat(O[0],len(P))
    rc = np.zeros((len(P)-1,3))
    rc[:,0] = np.repeat(O[0],len(P)-1)
    theta = np.linspace(0,2*np.pi,len(P)-1)
#    y = np.linspace(O[1]-a, O[1]+a, len(P))
    rc[:,1] = a*np.cos(theta)
    rc[:,2] = a*np.sin(theta)
#        z = -np.sqrt(a**2 - (y-O[1])**2) + O[2]
    # make the array of the magnetic field equation

    r_hat = np.zeros((len(rc)-1,3))
    diff = np.zeros((len(rc)-1,3))

    for i in range (0, len(rc)-1):
        # the segmentation of electric circuit
        dl = rc[i+1,:]-rc[i,:]
        # the r_hat is the normalized unit electric circuit
        r_hat[i,:] = rc[i,:]/LA.norm(rc)
        # the difference between the P and rc
        diff[i,:] =  rc[i,:] - P[i,:]
#        print(np.size(dl))
#        print(np.size(r_hat))
    dB, dBx, dBy, dBz = diff_field(dl, r_hat, np.transpose(diff[i,:]), b, I)

    # Sum of the dB
    B = np.zeros((len(rc)-1,3))
    Bx = np.zeros((len(rc)-1,1))
    By = np.zeros((len(rc)-1,1))
    Bz = np.zeros((len(rc)-1,1))
    for i in range (0,len(dBx)):
        B[i,] = sum(dB[0:i,]) 
        Bx[i,] = sum(dBx[0:i,]) 
        By[i,] = sum(dBy[0:i,])
        Bz[i,] = sum(dBz[0:i,])

    [Y, Z] = np.meshgrid(P[0:len(rc)-1,1], P[0:len(rc)-1,2])
    plt.quiver(Y,Z,By2,Bz2)
    plt.savefig('B_yz_ %s.pdf' % str(O) ,dpi=200)
       
    # array of the r
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
    return B, Bx, By, Bz

def field_loop_2(O, a, b, I, P):
    OO = np.array([O[0]+3*a, O[1], O[2]])
    #magnetic field on x direcion addited by two loops
    B1, Bx1, By1, Bz1 = field_yz_loop_x(O, a, b, I, P)
    B2, Bx2, By2, Bz2 = field_yz_loop_x(OO, a, b, I, P)
    #Addition of the magnetic field
    B_0 = B1+B2
    B_0x = Bx1+Bx2
    B_0y = By2+By2
    B_0z = Bz1+Bz2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    return B_0,B_0x,B_0y,B_0z
       
if __name__=="__main__":

    
    a = 1 # radius of the loop
    b = 10**-7 #b = uo/4pi
    I = 10
    O = np.array([1.5, 0, 0])
    x = np.linspace(O[0], 20, 10)
    y = np.linspace(O[1], 20, 10)
    z = np.linspace(O[2], 20, 10)    
    # analytical magnetic field on x direction
    B_x = Bx(x, a, b, I)
    fig1 = plt.figure()

    plt.plot(B_x)
    plt.xlabel('x direction')    
    plt.ylabel('B_x')
    plt.title('B_x')
    plt.savefig('B_x.pdf',dpi=200)
    # numertical magnetic field on x direction with one loop electric circuit
    P = np.zeros((len(x),3))
    B = np.zeros((len(x),3))    
    for i in range(len(x)):
#        P[i] = np.array([x[i], 0, 0]) special value on  y_z plane  
        P[i] = np.array([x[i], y[i], z[i]]) # special value on 
        print(field_yz_loop_x(O, a, b, I, P))
    B_0, B_0x, B_0y, B_0z = field_loop_2(O, a, b, I, P)

    #color_array
    c = np.zeros(shape=(len(B_0x), len(B_0y), len(B_0z), 4))
    c[:, :, :, 0] = u
    c[:, :, :, 1] = v
    c[:, :, :, 2] = w
    c[:, :, :, 3] = np.ones(shape=(len(B_0x), len(B_0y), len(B_0z)))
    c = np.abs(c)

    c2 = np.zeros(shape=(len(B_0x)*len(B_0y)*len(B_0z), 4))
    l = 0
    for i,j,k in np.ndindex((len(B_0x), len(B_0y), len(B_0z))):
        c2[l]=c[i,j,k]
        l+=1

    c3 = np.concatenate((c2, np.repeat(c2,2, axis=0)), axis=0)

    print('difference between B_x and B_3D_x:', B_x-B_0x)
    # 3D quiver figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(P[0:8,0],P[0:8,1],P[0:8,2],B_0x,B_0y,B_0z,pivot = 'middle', color = c3)
    plt.savefig('B_3D_quiver.pdf',dpi=200)
    print('B_3D_y_z:')
    print(B)
    # 3D wire, contour figure
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211, projection='3d')
    ax1.plot_wireframe(B_0x,B_0y, np.array([B_0z,B_0z]))
    plt.savefig('B_3D_framewire.pdf',dpi=200)
    ax2 = fig2.add_subplot(212, projection='3d')
    X,Y = np.meshgrid(B_0x,B_0y)
    ax2.contour(X, Y, np.array([B_0z,B_0z]))    
    plt.savefig('B_3D.pdf',dpi=200)
    plt.show()