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

# the segmentation of electric circuit
#r_hat is the unit 
def diff_field(dl, r_hat, r, b, I):
    dB = b*I*np.cross(dl, r_hat)/(LA.norm(r))**2
    dBx = np.zeros((len(r_hat),3))
    dBy = np.zeros((len(r_hat),3))
    dBz = np.zeros((len(r_hat),3))    
    for i in range(0, len(r_hat)-1):
        #r_hatx = unit r on x direction
        r_hatx = [r_hat[i+1,0]-r_hat[i,0], 0, 0]/LA.norm(r_hat)
        #r_haty = unit r on y direction  
        r_haty = [0, r_hat[i+1,1]-r_hat[i,1], 0]/LA.norm(r_hat)
        #r_hatz = unit r on z direction  
        r_hatz = [0, r_hat[i+1,2]-r_hat[i,2], 0]/LA.norm(r_hat)
        #magnetic field on x direction     
        dBx[i] = b*I*np.cross(dl, r_hatx)/(LA.norm(r))**2
        #magntic field on y direction    
        dBy[i] = b*I*np.cross(dl, r_haty)/(LA.norm(r))**2  
        #magntic field on z direction    
        dBz[i] = b*I*np.cross(dl, r_hatz)/(LA.norm(r))**2
    return dB, dBx, dBy, dBz

# analytical solution of magnetic field on x field
def Bx(x, a, b, I):
    B = b*4*np.pi*I*a**2/(2*(x**2+a**2)**(3/2))
    return B

# electric field loop on x direction
def field_yz_loop_x(O, a, b, I, P):
# electric circle coordinates: (y-O[1])**2 + (z-O[2])**2= a**2
    x = np.repeat(O[0],len(P))
    theta = np.linspace(0,2*np.pi,len(P))
#    y = np.linspace(O[1]-a, O[1]+a, len(P))
    for i in range(0, len(theta)):
        y = a*np.cos(theta)
        z = a*np.sin(theta)
#        z = -np.sqrt(a**2 - (y-O[1])**2) + O[2]
    # make the array of the magnetic field equation
    rc = np.array([x, y, z]).T
    r_hat = np.zeros((len(rc),3))
    diff = np.zeros((len(rc),3))
    dB = []
    dBy = []
    dBz = []

    for i in range (0, len(rc)-1):
        # the segmentation of electric circuit
        dl = rc[i+1,:]-rc[i,:]
        # the r_hat is the normalized unit electric circuit
        r_hat[i,:] = rc[i,:]/LA.norm(rc[i,:])
        # the difference between the P and rc
        diff[i,:] = P[i,:] - rc[i,:]
#        print(np.size(dl))
#        print(np.size(r_hat))
        dB, dBx, dBy, dBz = diff_field(dl, r_hat[i,:], np.transpose(diff[i,:]), b, I)
        dB.append(dB)
        dBx.append(dBx)
        dBy.append(dBy)
        dBz.append(dBz)
    By = np.zeros((len(dBx)))
    Bz = np.zeros((len(dBx)))    
    for i in range (0,len(dBx)):
        B[i] = sum(dB[i])        
        By[i] = sum(dBy[i])
        Bz[i] = sum(dBz[i])
    y = np.linspace(-5, 5, 10)
    z = np.linspace(-5, 5, 10)
    [Y, Z] = np.meshgrid(y, z)
    plt.quiver(Y,Z,By,Bz)
    plt.savefig('B_yz.pdf',dpi=200)
    #B1x = sum(dBx)
    # array of the r 
#    rc = np.array([x, y, -z]).T                                                                                                                                                                                                                                                                 
#    for i in range (0, len(rc)-1):
#        # the segmentation of electric circuit
#        dl = rc[i+1,:]-rc[i,:]
#        # the r_hat is the normalized unit electric circuit
#        r_hat = rc[i,:]/LA.norm(rc[i,:])
#        # the difference between the P and rc        
#        diff = np.repeat(P,50).reshape(50,3) - rc[i,:]
#        print(np.size(dl)) 
#        print(np.size(r_hat))
#        dB, dBx, dBy = diff_field(dl, r_hat, np.transpose(diff), b, I)
#        dB.append(dB)
#        dBx.append(dBx)
#        dBy.append(dBy)
        
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
    return B, By, Bz

def field_loop_2(O, a, b, I, P):
    OO = np.array([O[0]+3*a, O[1], O[2]])
    #magnetic field on x direcion addited by two loops
    Bx, By, Bz = field_yz_loop_x(O, a, b, I, P)
    Bx2, By2, Bz2 = field_yz_loop_x(OO, a, b, I, P)
    return Bx+Bx2, By+By2, Bz+Bz2
       
if __name__=="__main__":
    a = 1 # radius of the loop
    b = 10**-7 #b = uo/4pi
    I = 10
    O = np.array([1.5, 0, 0])
    x = np.linspace(O[0], 20, 10)
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
        P[i] = np.array([x[i], O[1], O[2]])
        print(field_yz_loop_x(O, a, b, I, P))
        B[i] = field_yz_loop_x(O, a, b, I, P)

    err = B_x-B
    # error of the B_x, B1x
    print('numerical and analysis difference of B along axis x:')
    print(err)
    y = np.linspace(-5, 5, 10)
    z = np.linspace(-5, 5, 10)
    [Y, Z] = np.meshgrid(y, z)
    plt.quiver(Y,Z,By,Bz)
    [X, Y, Z] = np.meshgrid(x, y, z)
    B = np.zeros((len(x),len(y),len(z)))
    O = np.array([0, 0, 0])
    for k in range(len(x)): # k on x axis
        for i in range(len(y)): # i on y axis
            for j in range(len(z)): # j on z axis
                P = np.array(x[k], y[i], z[j]) # any point P in the 3D space
                # numertical magnetic field on x direction with two loops of electric circuit
                Bx,By,Bz = field_loop_2(O, a, b, I, P)
    # quiver figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(X,Y,Z,B1x,By,Bz,color_array)
    x1 = np.repeat(O[0],len(P))
    theta = np.linspace(0,2*np.pi,len(P))
#    y = np.linspace(O[1]-a, O[1]+a, len(P))
    for i in range(0, len(theta)):
        y = a*np.cos(theta)
        z = a*np.sin(theta)
    plot3d(x1, y, z,'r')
    plot3d(x2, y, z, 'b')
    plt.savefig('B_3D_quiver.pdf',dpi=200)
    print('B_3D_y_z: ')
    print(B)
    # 
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211, projection='3d')
    ax1.plot_wireframe(B[0], B[1], B[2])
    ax2 = fig2.add_subplot(212, projection='3d')
    ax2.contour(B[0], B[1], B[2])    
    plt.savefig('B_3D.pdf',dpi=200)
    plt.show()
            
