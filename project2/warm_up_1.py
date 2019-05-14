"""
Cubic hermite splines in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2-4 assignments.

By Ilkka Kylanpaa on January 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from numpy import *
# Create random data
import numpy as np
import h5py

def p1(t):
    return (1.+2.*t)*(t-1.)**2
def p2(t):
    return t**2*(3.-2.*t)
def q1(t):
    return t*(t-1.)**2
def q2(t):
    return t**2*(t-1.)

def init_1d_spline(x,f,h):
    # now using complete boundary conditions
    # - natural boundary conditions commented
    a=zeros((len(x),))
    b=zeros((len(x),))
    c=zeros((len(x),))
    d=zeros((len(x),))
    fx=zeros((len(x),))

    # a[0]=1.0 # not needed
    b[0]=1.0

    # natural boundary conditions 
    #c[0]=0.5
    #d[0]=1.5*(f[1]-f[0])/(x[1]-x[0])

    # complete boundary conditions
    c[0]=0.0
    d[0]=(f[1]-f[0])/(x[1]-x[0])
    
    for i in range(1,len(x)-1):
        d[i]=6.0*(h[i]/h[i-1]-h[i-1]/h[i])*f[i]-6.0*h[i]/h[i-1]*f[i-1]+6.0*h[i-1]/h[i]*f[i+1]
        a[i]=2.0*h[i]
        b[i]=4.0*(h[i]+h[i-1])
        c[i]=2.0*h[i-1]        
    #end for

    
    b[-1]=1.0
    #c[-1]=1.0 # not needed

    # natural boundary conditions
    #a[-1]=0.5
    #d[-1]=1.5*(f[-1]-f[-2])/(x[-1]-x[-2])

    # complete boundary conditions
    a[-1]=0.0
    d[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    
    # solve tridiagonal eq. A*f=d
    c[0]=c[0]/b[0]
    d[0]=d[0]/b[0]
    for i in range(1,len(x)-1):
        temp=b[i]-c[i-1]*a[i]
        c[i]=c[i]/temp
        d[i]=(d[i]-d[i-1]*a[i])/temp
    #end for
        
    fx[-1]=d[-1]
    for i in range(len(x)-2,-1,-1):
        fx[i]=d[i]-c[i]*fx[i+1]
    #end for
        
    return fx


class spline(object):

    def __init__(self,*args,**kwargs):
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.fx=init_1d_spline(self.x,self.f,self.hx)
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.hy=diff(self.y)
            self.fx=zeros(shape(self.f))
            self.fy=zeros(shape(self.f))
            self.fxy=zeros(shape(self.f))
            for i in range(max([len(self.x),len(self.y)])):
                if (i<len(self.y)):
                    self.fx[:,i]=init_1d_spline(self.x,self.f[:,i],self.hx)
                if (i<len(self.x)):
                    self.fy[i,:]=init_1d_spline(self.y,self.f[i,:],self.hy)
            #end for
            for i in range(len(self.y)):
                self.fxy[:,i]=init_1d_spline(self.x,self.fy[:,i],self.hx)
            #end for
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.hy=diff(self.y)
            self.hz=diff(self.z)
            self.fx=zeros(shape(self.f))
            self.fy=zeros(shape(self.f))
            self.fz=zeros(shape(self.f))
            self.fxy=zeros(shape(self.f))
            self.fxz=zeros(shape(self.f))
            self.fyz=zeros(shape(self.f))
            self.fxyz=zeros(shape(self.f))
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fx[:,i,j]=init_1d_spline(self.x,self.f[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fy[i,:,j]=init_1d_spline(self.y,self.f[i,:,j],self.hy)
                    if (i<len(self.x) and j<len(self.y)):
                        self.fz[i,j,:]=init_1d_spline(self.z,self.f[i,j,:],self.hz)
            #end for
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxy[:,i,j]=init_1d_spline(self.x,self.fy[:,i,j],self.hx)
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxz[:,i,j]=init_1d_spline(self.x,self.fz[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fyz[i,:,j]=init_1d_spline(self.y,self.fz[i,:,j],self.hy)
            #end for
            for i in range(len(self.y)):
                for j in range(len(self.z)):
                    self.fxyz[:,i,j]=init_1d_spline(self.x,self.fyz[:,i,j],self.hx)
            #end for
        else:
            print('Either dims is missing or specific dims is not available')
        #end if
            
    def eval1d(self,x):
        if isscalar(x):
            x=array([x])
        N=len(self.x)-1
        f=zeros((len(x),))
        ii=0
        for val in x:
            i=floor(where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            else:
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*p1(t)+self.f[i+1]*p2(t)+self.hx[i]*(self.fx[i]*q1(t)+self.fx[i+1]*q2(t))
            ii+=1

        return f
    #end eval1d

    def eval2d(self,x,y):
        if isscalar(x):
            x=array([x])
        if isscalar(y):
            y=array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        f=zeros((len(x),len(y)))
        A=zeros((4,4))
        ii=0
        for valx in x:
            i=floor(where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=floor(where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                u = (valx-self.x[i])/self.hx[i]
                v = (valy-self.y[j])/self.hy[j]
                pu = array([p1(u),p2(u),self.hx[i]*q1(u),self.hx[i]*q2(u)])
                pv = array([p1(v),p2(v),self.hy[j]*q1(v),self.hy[j]*q2(v)])
                A[0,:]=array([self.f[i,j],self.f[i,j+1],self.fy[i,j],self.fy[i,j+1]])
                A[1,:]=array([self.f[i+1,j],self.f[i+1,j+1],self.fy[i+1,j],self.fy[i+1,j+1]])
                A[2,:]=array([self.fx[i,j],self.fx[i,j+1],self.fxy[i,j],self.fxy[i,j+1]])
                A[3,:]=array([self.fx[i+1,j],self.fx[i+1,j+1],self.fxy[i+1,j],self.fxy[i+1,j+1]])           
                
                f[ii,jj]=dot(pu,dot(A,pv))
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        if isscalar(x):
            x=array([x])
        if isscalar(y):
            y=array([y])
        if isscalar(z):
            z=array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=zeros((len(x),len(y),len(z)))
        A=zeros((4,4))
        B=zeros((4,4))
        ii=0
        for valx in x:
            i=floor(where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=floor(where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                for valz in z:
                    k=floor(where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1
                    u = (valx-self.x[i])/self.hx[i]
                    v = (valy-self.y[j])/self.hy[j]
                    t = (valz-self.z[k])/self.hz[k]
                    pu = array([p1(u),p2(u),self.hx[i]*q1(u),self.hx[i]*q2(u)])
                    pv = array([p1(v),p2(v),self.hy[j]*q1(v),self.hy[j]*q2(v)])
                    pt = array([p1(t),p2(t),self.hz[k]*q1(t),self.hz[k]*q2(t)])
                    B[0,:]=array([self.f[i,j,k],self.f[i,j,k+1],self.fz[i,j,k],self.fz[i,j,k+1]])
                    B[1,:]=array([self.f[i+1,j,k],self.f[i+1,j,k+1],self.fz[i+1,j,k],self.fz[i+1,j,k+1]])
                    B[2,:]=array([self.fx[i,j,k],self.fx[i,j,k+1],self.fxz[i,j,k],self.fxz[i,j,k+1]])
                    B[3,:]=array([self.fx[i+1,j,k],self.fx[i+1,j,k+1],self.fxz[i+1,j,k],self.fxz[i+1,j,k+1]])
                    A[:,0]=dot(B,pt)
                    B[0,:]=array([self.f[i,j+1,k],self.f[i,j+1,k+1],self.fz[i,j+1,k],self.fz[i,j+1,k+1]])
                    B[1,:]=array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1],self.fz[i+1,j+1,k],self.fz[i+1,j+1,k+1]])
                    B[2,:]=array([self.fx[i,j+1,k],self.fx[i,j+1,k+1],self.fxz[i,j+1,k],self.fxz[i,j+1,k+1]])
                    B[3,:]=array([self.fx[i+1,j+1,k],self.fx[i+1,j+1,k+1],self.fxz[i+1,j+1,k],self.fxz[i+1,j+1,k+1]])
                    A[:,1]=dot(B,pt)

                    B[0,:]=array([self.fy[i,j,k],self.fy[i,j,k+1],self.fyz[i,j,k],self.fyz[i,j,k+1]])
                    B[1,:]=array([self.fy[i+1,j,k],self.fy[i+1,j,k+1],self.fyz[i+1,j,k],self.fyz[i+1,j,k+1]])
                    B[2,:]=array([self.fxy[i,j,k],self.fxy[i,j,k+1],self.fxyz[i,j,k],self.fxyz[i,j,k+1]])
                    B[3,:]=array([self.fxy[i+1,j,k],self.fxy[i+1,j,k+1],self.fxyz[i+1,j,k],self.fxyz[i+1,j,k+1]])
                    A[:,2]=dot(B,pt)
                    B[0,:]=array([self.fy[i,j+1,k],self.fy[i,j+1,k+1],self.fyz[i,j+1,k],self.fyz[i,j+1,k+1]])
                    B[1,:]=array([self.fy[i+1,j+1,k],self.fy[i+1,j+1,k+1],self.fyz[i+1,j+1,k],self.fyz[i+1,j+1,k+1]])
                    B[2,:]=array([self.fxy[i,j+1,k],self.fxy[i,j+1,k+1],self.fxyz[i,j+1,k],self.fxyz[i,j+1,k+1]])
                    B[3,:]=array([self.fxy[i+1,j+1,k],self.fxy[i+1,j+1,k+1],self.fxyz[i+1,j+1,k],self.fxyz[i+1,j+1,k+1]])
                    A[:,3]=dot(B,pt)
                
                    f[ii,jj,kk]=dot(pu,dot(A,pv))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
#end class spline

#    def fun(x) as the line r = t:
def fun1(r_0, r_1, t): 
    r = zeros((np.size(t,0),1))
    for i in range(0,len(t)):
        r[i] = r_0+t[i]*(r_1-r_0)
    return r    
    
#def fun2(r_0, r_1, t): 
#    r = zeros((np.size(t,0),2))
#    for i in range(0,len(t)):
#        r[i,] = r_0+dot(t[i,],(r_1-r_0))
#    return r
    
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
    spl1d = spline(x=x,f=mean(data,1), dims=1)    
    fig = plt.figure()
    plt.plot(x,spl1d.eval1d(x),'r--',x,y[1:8],'o')
    plt.title('interpolation1D of data')
    plt.savefig('interp1D.pdf',dpi=200)
    # initialize the 2d interpolation  
    spl2d = spline(x=x,y=y,f=array(data), dims=2)  
    X,Y = np.meshgrid(x,y)
    fig = plt.figure()
    plt.pcolor(X,Y,spl2d.eval2d(x,y).T)
    plt.title('interpolation2D of data')
    plt.savefig('interp2D.pdf',dpi=200)
    tol = 0.1
    # contour
    t = linspace(0, 1, 100)
    r_0 = -1.5
    r_1 = 1.5
    x = fun1(r_0, r_1, t)
    fig = plt.figure()
    X,Y = meshgrid(x,y)
    # contour plot    
    fig = plt.figure()
    plt.contourf(X,Y,spl2d.eval2d(x,y).T,30)
    plt.title('contourfinterp')
    plt.savefig('contouf.pdf',dpi=200) 
    plt.savefig('contorf 2D.pdf',dpi=200)
#  data interpolation
#    x=np.linspace(-1.5,1.5,100)
#    y=np.linspace(-1.5,1.5,100)
    xx = np.linspace(x[0], x[-1], 100)
#    xx = linspace(data[0], 0.1, 100)
    fig = plt.figure()
    plt.plot(xx,spl1d.eval1d(xx),'r--',x,y[1:8],'o')
    plt.title('interpolation1D of fun')
    plt.savefig('interp1D_eval.pdf',dpi=200)
    #calculate the difference between interpolation and analytic function
    err1 =  spl1d.eval1d(x)-mean(data,1)
    print('the difference between interpolation and analytic function:')
    print(np.array(err1))
    #check the error
    if LA.norm(err1) < tol:
        print('1D interpolate is OK')
    fig = plt.figure()
    ax1 = plt.subplot(111)   
    ax1.plot(x, err1)
    # plot the error
    plt.ylabel('error')
    plt.xlabel('x')
    plt.title('error')
    plt.savefig('interp1D_error.pdf',dpi=200)
    
# interpolation on y = x  
    t = linspace(-1, 5, 100)
# 1D
    # initialize the interpolation on y = x 1D 
    spl1d2=spline(x=t,f=fun1(r_0, r_1, t),dims=1)
    xx = linspace(data[0][0], 1.6/3, 100) #line from -1.5 to 0.1
#   xx = linspace(data[0], 0.1, 100)
    fig = plt.figure()
    plt.plot(xx,spl1d2.eval1d(xx),'r--',xx,fun1(r_0, r_1,xx),'o')    
    plt.title('interpolation of fun')
    plt.savefig('1D_interpolation.pdf',dpi=200)
    err2 =  spl1d.eval1d(xx)-fun1(r_0, r_1,xx)
    print('the difference between interpolation and analytic function:')
    print(np.array(err2))
    #check the error
    if LA.norm(err2) < tol:
        print('1D interpolate is OK')
    fig = plt.figure()
    ax2 = plt.subplot(111)   
    ax2.plot(err2)
    # plotthe error
    plt.ylabel('error')
    plt.xlabel('xx')
    plt.title('error')
    plt.savefig('interp1D_test_error.pdf',dpi=200)
        
# 2D [0.1 0.1]
    r_0 = -1.5
    r_1 = 1.5
    test2d = spl2d.eval2d(0.1,0.1)
    print('test2D interpolation value:', test2d)

#    xx = linspace(data[0][0], 1.6/3, 100) #line from -1.5 to 0.1
#    yy = linspace(data[0][0], 1.6/3, 100) #line from -1.5 to 0.1
##    # initialize the interpolation on y =x  2D
#    f = fun2(r_0, r_1, array([xx,yy]).T)
#    spl2d=spline(x =xx, y=yy ,f = f, dims=2)
    X,Y = meshgrid(x,y)
#   err3 = spl2d.eval2d(linspace(-1.5,0.1,100),linspace(-1.5,0.1,100))-fun1(r_0, r_1, linspace(-1.5,0.1,100))
    xx = linspace(0, 1.6/3, 100) #line from -1.5 to 0.1
    yy = linspace(0, 1.6/3, 100) #line from -1.5 to 0.1
    err3 = spl2d.eval2d(xx,yy)-fun1(r_0, r_1, xx)
    #check the error
    if LA.norm(err3) < tol:
        print('2D interpolate is OK')
    print('the difference between interpolation and analytic function:')
    print(np.array(err3))
    fig = plt.figure()
    ax3 = plt.subplot(111) 
    #plot error
    ax3.plot(err3)
    plt.ylabel('error')
    plt.xlabel('2d')
    plt.title('error')
    plt.savefig('interp2D_test_error.pdf',dpi=200)
    
    # Write data to HDF5
    data_file = h5py.File('warm_up_interpolated.h5', 'w')
    data_file.create_dataset('spl_1d_x', data = spl1d.eval1d(x))
    data_file.create_dataset('spl_1d_xx', data = spl1d.eval1d(xx))
    data_file.create_dataset('spl_2d_xy', data = spl2d.eval2d(0.1,0.1))
    data_file.create_dataset('spl_2d_xx_yy', data = spl2d.eval2d(xx,yy))
    data_file.close()
#end main
    
if __name__=="__main__":
    main()
