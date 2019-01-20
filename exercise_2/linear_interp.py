"""
Linear interpolation in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA


"""
Add basis functions l1 and l2 here
"""

class linear_interp(object):
    
    def __init__(self,*args,**kwargs):
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
            self.hz=np.diff(self.z)
        else:
            print('Either dims is missing or specific dims is not available')

    def l1(self, t):
        return 1-t   
    def l2(self, t):
        return t
           
    def eval1d(self,x):
        # make sure that x is array
        if np.isscalar(x):
            x=np.array([x])
        N=len(self.x)-1
        f=np.zeros((len(x),))
        ii=0
        for val in x:
            # round and find the closest integer for i
            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)
            # when the axis reaches the maximum of the x, set the value of the
            # f as the last value for interpolation
            if i==N:
                f[ii]=self.f[i]
            # calculated according to the Hermite cubic splines
            else:
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*self.l1(t)+self.f[i+1]*self.l2(t)
            ii+=1
        return f

    def test1d(self, x, y, tol):
        plt.figure()
        plt.plot(y)
        plt.title('error norm1')
        plt.savefig('1D error.pdf',dpi=200)
        err = LA.norm(y,2)
        print('the norm 2 of the difference between interpolation and analytic function:')
        print(np.array(err))
        if err < tol:
            print('1D interpolate is OK')
        return err
    
    def eval2d(self,x,y):
        # make sure that x and y are arrays
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        # initiate the f and A
        f=np.zeros((len(x),len(y)))
        A=np.zeros((2,2))
        ii=0
        for valx in x:
        # round and find the closest integer for i
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            # when the axis reaches the maximum of the x,y, set the 
            # value of h as the last value for interpolation            
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                tx = (valx-self.x[i])/self.hx[i]
                ty = (valy-self.y[j])/self.hy[j]
                # calculate according to interpolation 2D
                ptx = np.array([self.l1(tx),self.l2(tx)])
                pty = np.array([self.l1(ty),self.l2(ty)])
                A[0,:]=np.array([self.f[i,j],self.f[i,j+1]])
                A[1,:]=np.array([self.f[i+1,j],self.f[i+1,j+1]])
                f[ii,jj]=np.dot(ptx,np.dot(A,pty))
                jj+=1
            ii+=1
        return f
    #end eval2d
    
    def test2d(self, x, y, z, tol):
        fig2 = plt.figure()
        ax21 = plt.subplot(121, projection='3d')
        ax22 = plt.subplot(122)        
        ax21.plot_wireframe(x, y, z)
        # the norm2 of the error
        plt.title('error norm2')
        plt.savefig('2D error.pdf',dpi=200)
        ax22.pcolor(x, y, z)
        # the colored norm2 of the error        
        plt.title('colored error')
        plt.savefig('colored_2D error.pdf',dpi=200)
        err = LA.norm(z,2)
        print('the norm 2 of the difference between interpolation and analytic function:')
        print(np.array(err))
        if err < tol:
            print('2D interpolate is OK')
        return err

    def eval3d(self,x,y,z):
        # round and find the closest integer for i
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        if np.isscalar(z):
            z=np.array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        # initiate f, A, B
        f=np.zeros((len(x),len(y),len(z)))
        A=np.zeros((2,2))
        B=np.zeros((2,2))
        ii=0
        for valx in x:
            # round and find the closest integer for i
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            # when the axis reaches the maximum of the x,y, set the 
            # value of h as the last value for interpolation            
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                for valz in z:
                    k=np.floor(np.where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1
                    tx = (valx-self.x[i])/self.hx[i]
                    ty = (valy-self.y[j])/self.hy[j]
                    tz = (valz-self.z[k])/self.hz[k]
                    # calculate according to interpolation 2D
                    ptx = np.array([self.l1(tx),self.l2(tx)])
                    pty = np.array([self.l1(ty),self.l2(ty)])
                    ptz = np.array([self.l1(tz),self.l2(tz)])
                    # calculate A and B according to interpolation 2D
                    B[0,:]=np.array([self.f[i,j,k],self.f[i,j,k+1]])
                    B[1,:]=np.array([self.f[i+1,j,k],self.f[i+1,j,k+1]])
                    A[:,0]=np.dot(B,ptz)
                    B[0,:]=np.array([self.f[i,j+1,k],self.f[i,j+1,k+1]])
                    B[1,:]=np.array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1]])
                    A[:,1]=np.dot(B,ptz)
                    f[ii,jj,kk]=np.dot(ptx,np.dot(A,pty))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
    
    def test3d(self, X, Y, Z, tol):
        fig3 = plt.figure()
        ax3 = plt.subplot(111)        
        ax3.pcolor(X, Y, Z[...,int(np.size(Z,2)/2)])
        # the colored norm2 of the error        
        plt.title('colored error')
        plt.savefig('colored_3D error.pdf',dpi=200)
        err = LA.norm(Z[...,int(np.size(Z,2)/2)],2)
        print('the norm 2 of the difference between interpolation and analytic function:')
        print(np.array(err))
        if err < tol:
            print('3D interpolate is OK')
        return err
# end class linear interp
    
def main():  
    fig1d = plt.figure()
    ax1d = plt.subplot(111)
    # 1d example
    x=np.linspace(0.,2.*np.pi,11)
    # test with sin
    def fun(x):
        return np.sin(x)
    y=fun(x)
    lin1d=linear_interp(x=x,f=y,dims=1)
    xx=np.linspace(0.,2.*np.pi,101)
    ax1d.plot(xx,lin1d.eval1d(xx))   
    ax1d.plot(x,y,'o',xx,np.sin(xx),'r--')
    ax1d.set_title('interpolation of f')
    plt.savefig('1D.pdf',dpi=200)
    tol = 0.5
    err1 = lin1d.test1d(xx, lin1d.eval1d(xx)-np.sin(xx), tol)
    
    # 2d example
    fig2d = plt.figure()
    #set the axis for four subplot separately
    ax2d = plt.subplot(221, projection='3d')
    ax2d2 = plt.subplot(222, projection='3d')
    ax2d3 = plt.subplot(223)
    ax2d4 = plt.subplot(224)

    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y)
    # test function is Z
    def fun2(X,Y):
        return X*np.exp(-1.0*(X*X+Y*Y))
    Z1 = fun2(X,Y)
    # plot interpolation 2D with 11*11 grids
    ax2d.plot_wireframe(X,Y,Z1)
    plt.title('interpolation 2D with 11*11 grids')
    plt.savefig('2D_11_11.pdf',dpi=200)
    # colored with z_value
    ax2d3.pcolor(X,Y,Z1)
    plt.title('color of interpolation 2D with 11*11 grids')
    plt.savefig('colored_2D_11_11.pdf',dpi=200)    
    #ax2d3.contourf(X,Y,Z)
    
    lin2d=linear_interp(x=x,y=y,f=Z1,dims=2)
    x=np.linspace(-2.0,2.0,51)
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y)
    Z2 = lin2d.eval2d(x,y)
    # plot interpolation 2D with 51*51 grids     
    ax2d2.plot_wireframe(X,Y,Z2)
    plt.title('interpolation 2D with 51*51 grids')
    plt.savefig('2D_51_51.pdf',dpi=200)
    # colored with z_value
    ax2d4.pcolor(X,Y,Z2)
    plt.title('color of interpolation 2D with 51*51 grids')
    plt.savefig('colored_2D_51_51.pdf',dpi=200)
    tol = 1    
    err2 = lin2d.test2d(X, Y, Z2-fun2(X,Y), tol)

    # 3d example
    x=np.linspace(0.0,3.0,11)
    y=np.linspace(0.0,3.0,11)
    z=np.linspace(0.0,3.0,11)
    X,Y,Z = np.meshgrid(x,y,z)
    def fun3(X,Y,Z):
        return (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    F1 = fun3(X,Y,Z)
    X,Y= np.meshgrid(x,y)
    fig3d=plt.figure()
    ax3d=plt.subplot(121)
    ax3d.pcolor(X,Y,F1[...,int(len(z)/2)])
    lin3d=linear_interp(x=x,y=y,z=z,f=F1,dims=3)
    
    x=np.linspace(0.0,3.0,50)
    y=np.linspace(0.0,3.0,50)
    z=np.linspace(0.0,3.0,50)
    X,Y,Z = np.meshgrid(x,y,z) 
    F1 = fun3(X,Y,Z)
    X,Y= np.meshgrid(x,y)
    F2 = lin3d.eval3d(x,y,z)
    ax3d2=plt.subplot(122)
    ax3d2.pcolor(X,Y,F2[...,int(len(z)/2)]) 
    tol = 0.1      
    err3 = lin3d.test3d(X, Y, F2-F1, tol)

    plt.show()
#end main
    
if __name__=="__main__":
    main()
