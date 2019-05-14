# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 05:56:32 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from numpy import *
# Create random data
import numpy as np
import h5py

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

#def init_1d_spline(x,f,h):
#    # now using complete boundary conditions
#    # with forward/backward derivative
#    # - natural boundary conditions commented
#    a=zeros((len(x),))
#    b=zeros((len(x),))
#    c=zeros((len(x),))
#    d=zeros((len(x),))
#    fx=zeros((len(x),))
#
#    # a[0]=1.0 # not needed
#    b[0]=1.0
#
#    # natural boundary conditions 
#    #c[0]=0.5
#    #d[0]=1.5*(f[1]-f[0])/(x[1]-x[0])
#
#    # complete boundary conditions
#    c[0]=0.0
#    d[0]=(f[1]-f[0])/(x[1]-x[0])
#    
#    for i in range(1,len(x)-1):
#        d[i]=6.0*(h[i]/h[i-1]-h[i-1]/h[i])*f[i]-6.0*h[i]/h[i-1]*f[i-1]+6.0*h[i-1]/h[i]*f[i+1]
#        a[i]=2.0*h[i]
#        b[i]=4.0*(h[i]+h[i-1])
#        c[i]=2.0*h[i-1]        
#    #end for
#
#    
#    b[-1]=1.0
#    #c[-1]=1.0 # not needed
#
#    # natural boundary conditions
#    #a[-1]=0.5
#    #d[-1]=1.5*(f[-1]-f[-2])/(x[-1]-x[-2])
#
#    # complete boundary conditions
#    a[-1]=0.0
#    d[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
#    
#    # solve tridiagonal eq. A*f=d
#    c[0]=c[0]/b[0]
#    d[0]=d[0]/b[0]
#    for i in range(1,len(x)-1):
#        temp=b[i]-c[i-1]*a[i]
#        c[i]=c[i]/temp
#        d[i]=(d[i]-d[i-1]*a[i])/temp
#    #end for
#        
#    fx[-1]=d[-1]
#    for i in range(len(x)-2,-1,-1):
#        fx[i]=d[i]-c[i]*fx[i+1]
#    #end for
#        
#    return fx
## end function init_1d_spline
#
#
#class spline(object):
#
#    def __init__(self,*args,**kwargs):
#        self.dims=kwargs['dims']
#        if (self.dims==1):
#            self.x=kwargs['x']
#            self.f=kwargs['f']
#            self.hx=diff(self.x)
#            self.fx=init_1d_spline(self.x,self.f,self.hx)
#        elif (self.dims==2):
#            self.x=kwargs['x']
#            self.y=kwargs['y']
#            self.f=kwargs['f']
#            self.hx=diff(self.x)
#            self.hy=diff(self.y)
#            self.fx=zeros(shape(self.f))
#            self.fy=zeros(shape(self.f))
#            self.fxy=zeros(shape(self.f))
#            for i in range(max([len(self.x),len(self.y)])):
#                if (i<len(self.y)):
#                    self.fx[:,i]=init_1d_spline(self.x,self.f[:,i],self.hx)
#                if (i<len(self.x)):
#                    self.fy[i,:]=init_1d_spline(self.y,self.f[i,:],self.hy)
#            #end for
#            for i in range(len(self.y)):
#                self.fxy[:,i]=init_1d_spline(self.x,self.fy[:,i],self.hx)
#        elif (self.dims==3):
#            self.x=kwargs['x']
#            self.y=kwargs['y']
#            self.z=kwargs['z']
#            self.f=kwargs['f']
#            self.hx=diff(self.x)
#            self.hy=diff(self.y)
#            self.hz=diff(self.z)
#            self.fx=zeros(shape(self.f))
#            self.fy=zeros(shape(self.f))
#            self.fz=zeros(shape(self.f))
#            self.fxy=zeros(shape(self.f))
#            self.fxz=zeros(shape(self.f))
#            self.fyz=zeros(shape(self.f))
#            self.fxyz=zeros(shape(self.f))
#            for i in range(max([len(self.x),len(self.y),len(self.z)])):
#                for j in range(max([len(self.x),len(self.y),len(self.z)])):
#                    if (i<len(self.y) and j<len(self.z)):
#                        self.fx[:,i,j]=init_1d_spline(self.x,self.f[:,i,j],self.hx)
#                    if (i<len(self.x) and j<len(self.z)):
#                        self.fy[i,:,j]=init_1d_spline(self.y,self.f[i,:,j],self.hy)
#                    if (i<len(self.x) and j<len(self.y)):
#                        self.fz[i,j,:]=init_1d_spline(self.z,self.f[i,j,:],self.hz)
#            #end for
#            for i in range(max([len(self.x),len(self.y),len(self.z)])):
#                for j in range(max([len(self.x),len(self.y),len(self.z)])):
#                    if (i<len(self.y) and j<len(self.z)):
#                        self.fxy[:,i,j]=init_1d_spline(self.x,self.fy[:,i,j],self.hx)
#                    if (i<len(self.y) and j<len(self.z)):
#                        self.fxz[:,i,j]=init_1d_spline(self.x,self.fz[:,i,j],self.hx)
#                    if (i<len(self.x) and j<len(self.z)):
#                        self.fyz[i,:,j]=init_1d_spline(self.y,self.fz[i,:,j],self.hy)
#            #end for
#            for i in range(len(self.y)):
#                for j in range(len(self.z)):
#                    self.fxyz[:,i,j]=init_1d_spline(self.x,self.fyz[:,i,j],self.hx)
#
#            #end for
#        else:
#            print('Either dims is missing or specific dims is not available')
#        #end if
#
#    def p1(self, t):
#        return (1+2*t)*((t-1)**2)  
#    
#    def p2(self, t):
#        return t**2*(3-2*t)
#    def q1(self, t):
#        return t*(t-1)**2   
#    def q2(self, t):
#        return t**2*(t-1)
#            
#    def eval1d(self,x):
#        if isscalar(x):
#            # make sure that x is array
#            x=array([x])
#        N=len(self.x)-1
#        f=zeros((len(x),))
#        ii=0
#        for val in x:
#            # round and find the closest integer for i
#            i=floor(where(self.x<=val)[0][-1]).astype(int)
#            # when the axis reaches the maximum of the x, set the value of the
#            # f as the last value for interpolation
#            if i==N:
#                f[ii]=self.f[i]
#            # calculated according to the Hermite cubic splines
#            else:
#                t=(val-self.x[i])/self.hx[i]
#                f[ii]=self.f[i]*self.p1(t)+self.f[i+1]*self.p2(t)+self.hx[i]*(self.fx[i]*self.q1(t)+self.fx[i+1]*self.q2(t))
#            ii+=1
#
#        return f
#    #end eval1d
#
#    def eval2d(self,x,y):
#        # make sure that x and y are arrays
#        if isscalar(x):
#            x=array([x])
#        if isscalar(y):
#            y=array([y])
#        Nx=len(self.x)-1
#        Ny=len(self.y)-1
#        # initiate the f and A
#        f=zeros((len(x),len(y)))
#        A=zeros((4,4))
#        ii=0
#        for valx in x:
#        # round and find the closest integer for i
#            i=floor(where(self.x<=valx)[0][-1]).astype(int)
#            # when the axis reaches the maximum of the x,y, set the 
#            # value of h as the last value for interpolation            
#
#            if (i==Nx):
#                i-=1
#            jj=0
#            for valy in y:
#                j=floor(where(self.y<=valy)[0][-1]).astype(int)
#                if (j==Ny):
#                    j-=1
#                u = (valx-self.x[i])/self.hx[i]
#                v = (valy-self.y[j])/self.hy[j]
#                # calculate according to Hermite cubic splines 2D
#                pu = array([self.p1(u),self.p2(u),self.hx[i]*self.q1(u),self.hx[i]*self.q2(u)])
#                pv = array([self.p1(v),self.p2(v),self.hy[j]*self.q1(v),self.hy[j]*self.q2(v)])
#                A[0,:]=array([self.f[i,j],self.f[i,j+1],self.fy[i,j],self.fy[i,j+1]])
#                A[1,:]=array([self.f[i+1,j],self.f[i+1,j+1],self.fy[i+1,j],self.fy[i+1,j+1]])
#                A[2,:]=array([self.fx[i,j],self.fx[i,j+1],self.fxy[i,j],self.fxy[i,j+1]])
#                A[3,:]=array([self.fx[i+1,j],self.fx[i+1,j+1],self.fxy[i+1,j],self.fxy[i+1,j+1]])           
#                
#                f[ii,jj]=dot(pu,dot(A,pv))
#                jj+=1
#            ii+=1
#        return f
#    #end eval2d

    
#    def fun(x) as the line r = t:
def fun(r_0, r_1, t): 
    r = zeros((np.size(t,0),2))
    for i in range(0,len(t)):
        r[i,] = r_0+t[i]*(r_1-r_0)
    return r
    
def main(): 
    #   read data for interpolation     
    f1 = h5py.File('P:\computational physics\computational-physics-master\computational-physics-master\project2\warm_up_data.h5','r+');
    data = [];
    x = [];
    y = [];
    #  write into array    
    for d in f1["data"]:
        data.append(d.T);
    for xg in f1["x_grid"]:
        x.append(xg)
    for yg in f1["y_grid"]:
        y.append(yg) 
#        y = y[1:8]
    # initialize the 1d interpolation   
    interp1d = linear_interp(x=x,f=data[0], dims=1)    
    plt.plot(x,interp1d.eval1d(x),'r--',x,y[0:7],'o')
    plt.title('interpolation1D of data')
    plt.savefig('interp1D.pdf',dpi=200)
    # initialize the 2d interpolation  
    interp2d = linear_interp(x=x,y=y,f=data, dims=2)  
    X,Y = np.meshgrid(x,y)
    plt.pcolor(X,Y,interp2d.eval2d(x,y))
    plt.title('interpolation2D of data')
    plt.savefig('interp2D.pdf',dpi=200)

#  data interpolation
#    x=np.linspace(-1.5,1.5,100)
#    y=np.linspace(-1.5,1.5,100)
    xx = np.linspace(x[0], x[-1], 100)
#    xx = linspace(data[0], 0.1, 100)
    plt.plot(xx,spl1d.eval1d(xx),'r--',x,y,'o')
    plt.title('interpolation1D of fun')
    plt.savefig('interp1D_eval.pdf',dpi=200)
    #calculate the difference between interpolation and analytic function
    err1 = spl1d.test1d(x, spl1d.eval1d(x)-data, tol)
    print('the difference between interpolation and analytic function:')
    print(np.array(err1))
    if err1 < tol:
        print('1D interpolate is OK')
# interpolation on y = x  
    t = linspace(0, 1, 100)
# 1D
    r_0 = -1.5
    r_1 = 1.5
    # initialize the interpolation on y = x 1D 
    spl1d2=spline(x=t,f=fun(r_0, r_1, t),dims=1)
    xx = linspace(0, 1.6/3, 100)
#   xx = linspace(data[0], 0.1, 100)
    plt.plot(xx,spl1d2.eval1d(xx),'r--',xx,fun(r_0, r_1,xx),'o')    
    plt.title('interpolation of fun')
    plt.savefig('1D_interpolation.pdf',dpi=200)
    err2 = spl1d2.test1d(xx, spl1d2.eval1d(xx)-fun(r_0, r_1,xx), tol)
    print('the difference between interpolation and analytic function:')
    print(np.array(err2))
    if err2 < tol:
        print('2D interpolate is OK')
    t = linspace(0, 1, 100)

# 2D 
    r_0 = array([-1.5, -1.5])
    r_1 = array([1.5, 1.5])
    r = fun(r_0, r_1, t)
    xx = linspace(0,1.6/3, 100)
    yy = linspace(0,1.6/3, 100)
    # initialize the interpolation on y =x  2D
    spl2d2=spl1d2.test2d(x=xx,y=yy ,f = fun(r_0, r_1, [xx,yy]), dims=2)   
    
# 2d plot 
    X,Y = np.meshgrid(xx,yy)
    plt.pcolor(X,Y,data,30)
    plt.title('color of interpolation 2D with 11*11 grids')
    plt.savefig('colored_2D_11_11.pdf',dpi=200) 
    plt.savefig('2D.pdf',dpi=200)
    err3 = spl2d2.test2d(xx, yy, spl2d2.eval2d(xx,yy)-fun(r_0, r_1, [xx,yy]), tol)
    if err3 < tol:
        print('2D interpolate is OK')
       
    plt.contourf(X,Y,F,30)
    plt.title('contourf')
    plt.savefig('contouf.pdf',dpi=200) 
    plt.savefig('contorf 2D.pdf',dpi=200)
    
    # Write data to HDF5
    data_file = h5py.File('warm_up_interpolated.h5', 'w')
    data_file.create_dataset('spline', data = [spl1d.eval1d(x), spl2d.eval2d(x,y), spl1d.eval1d(xx), spl2d.eval2d(xx,yy)])
    data_file.close()
    
if __name__=="__main__":
    main()
