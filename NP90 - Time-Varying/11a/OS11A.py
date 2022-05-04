import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

def f(x):
	return x/(1+x**2)

def g(x):
	return x**3

def Train():
    n=1
    end=100000
    endval=np.arange(-1*n,end)

    #u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)

    r = np.sin(2*np.pi*endval/25) + np.sin(2*np.pi*endval/10)

    yp=np.empty((end+n,1),dtype='float')
    yp[0]=0

    ym=np.empty((end+n,1),dtype='float')
    ym[0]=0
    yphat=np.empty((end+n,1),dtype='float')
    yphat[0]=0

    bias=1
    L1=18


    W1_f=np.random.normal(0,0.1,(L1,1+1))
    W1_g=np.random.normal(0,0.1,(L1,1+1))
    W1_h=np.random.normal(0,0.1,(L1,1+1))
    #W2=np.random.normal(0,0.1,(L2,L1+1))




    W2_f=np.zeros([1,L1+1])
    W2_g=np.zeros([1,L1+1])
    W2_h=np.zeros([1,L1+1])

    J_f=0
    J_g=0
    J_h=0

    lam_f=1
    lam_g=1
    lam_h=1

    P0_f=(1/lam_f)*np.identity(L1+1)
    P0_g=(1/lam_g)*np.identity(L1+1)
    P0_h=(1/lam_h)*np.identity(L1+1)
	

    epochs=1
    flag=1
    for a in range(epochs):
        for i in range(n,end+n):
        # for i in range(n,100):
            
            ym[i] = 0.6*ym[i-1] + r[i-1]

            S_f=yp[i-1]
        
            x_f=np.array([bias,S_f[0]])
            v0k_f=np.array([bias,S_f[0]])
            # A1_f=np.matmul(W1_f,x_f)
            v1kbar_f=tan_h(np.matmul(W1_f,v0k_f))

            v1k_f=np.insert(v1kbar_f,0,[bias])

            B1_f=np.matmul(W2_f,v1k_f)
            C_f=B1_f

            S_h = -C_f + 0.6*yp[i-1] + r[i-1]
        
            x_h=np.array([bias,S_h[0]])
            v0k_h=np.array([bias,S_h[0]])
            # A1_h=np.matmul(W1_h,x_h)
            v1kbar_h=tan_h(np.matmul(W1_h,v0k_h))

            v1k_h=np.insert(v1kbar_h,0,[bias])

            B1_h=np.matmul(W2_h,v1k_h)
            C_h=B1_h

            S_g=C_h
            x_g=np.array([bias,S_g[0]])
            v0k_g=np.array([bias,S_g[0]])
            # A1_g=np.matmul(W1_g,x_g)
            v1kbar_g=tan_h(np.matmul(W1_g,v0k_g))

            v1k_g=np.insert(v1kbar_g,0,[bias])

            B1_g=np.matmul(W2_g,v1k_g)
            C_g=B1_g
            
            yp[i] = f(yp[i-1]) + g(C_h)
            
            # ipdb.set_trace()

            yphat[i]=C_f + C_g

            e_f=(yp[i]-yphat[i])
            e_g=(yp[i]-yphat[i])
            
            # e_h=-(C_g-(-C_f + 0.6*yp[i-1]+r[i-1]))
            e_h=-(C_g-(r[i-1]))
            #del2=e*dtan_h(B1)
            v1k_f=v1k_f.reshape(len(v1k_f),1)
            v1k_T_f=v1k_f.reshape(1,len(v1k_f))
            Num_f=np.matmul(P0_f,np.matmul(np.matmul(v1k_f,v1k_T_f),P0_f))	
            Den_f=1+np.matmul(v1k_T_f,np.matmul(P0_f,v1k_f))
            P0_f=P0_f-(Num_f/Den_f)
            W2_f=W2_f+e_f*np.matmul(v1k_T_f,P0_f)
            J_f=J_f+(e_f*e_f)

            
            #del2=e*dtan_h(B1)
            v1k_g=v1k_g.reshape(len(v1k_g),1)
            v1k_T_g=v1k_g.reshape(1,len(v1k_g))
            Num_g=np.matmul(P0_g,np.matmul(np.matmul(v1k_g,v1k_T_g),P0_g))	
            Den_g=1+np.matmul(v1k_T_g,np.matmul(P0_g,v1k_g))
            P0_g=P0_g-(Num_g/Den_g)
            W2_g=W2_g+e_g*np.matmul(v1k_T_g,P0_g)
            J_g=J_g+(e_g*e_g)

            inp=np.array([bias,r[i-1]])
            inp=x_h.reshape(1,len(inp))
            v1k_h=v1k_h.reshape(len(v1k_h),1)
            v1k_T_h=v1k_h.reshape(1,len(v1k_h))
            Num_h=np.matmul(P0_h,np.matmul(np.matmul(v1k_h,v1k_T_h),P0_h))	
            Den_h=1+np.matmul(v1k_T_h,np.matmul(P0_h,v1k_h))
            P0_h=P0_h-(Num_h/Den_h)
            W2_h=W2_h+e_h*np.matmul(v1k_T_h,P0_h)
            J_h=J_h+(e_h*e_h)

            if(math.isnan(J_f) and flag):
                print(i)
                flag=0

    J_f=J_f/(end)
    J_g=J_g/(end)
    J_h=J_h/(end)
    print("Training Cost for OSLA 2 Layer N1 = ",J_f)
    print("Training Cost for OSLA 2 Layer N2 = ",J_g)
    print("Training Cost for OSLA 2 Layer N3 = ",J_h)

    print(np.max(yphat))
    lims=90000
    return yp
    # plt.figure()
    # plt.plot(endval,ym,color='red')
    # plt.plot(endval,yp,color='green')
    # plt.legend(["Model","Plant"])
    # plt.xlim([lims,lims+100])
    # plt.savefig("Ex11OSLA.png")


    # ustop=2.00
    # u=np.arange(-ustop,ustop+0.01,0.01)
    # #print(u)
    # uval=u/ustop
    # #print(uval)
    # gval=np.empty([len(uval),1])
    # gvalhat=np.empty([len(uval),1])
    # for i in range(len(uval)):
    # 	gval[i]=u[i]**3
    # 	S_g=u[i]
    # 	x_g=np.array([bias,S_g])
    # 	A1_g=np.matmul(W1_g,x_g)
    # 	A_g=tan_h(A1_g)
    # 	y_g=np.insert(A_g,0,[bias])
    # 	B1_g=np.matmul(W2_g,y_g)
    # 	B_g=tan_h(B1_g)
    # 	z_g=np.insert(B_g,0,[bias])
    # 	C1_g=np.matmul(W3_g,z_g)
    # 	C_g=C1_g
    # 	#print(C_g);
    # 	gvalhat[i]=C_g
    # print(np.sum(np.square(gval-gvalhat)))
    # print(np.max(gval))
    # print(np.max(gvalhat))
    # #plt.figure();
    # plt.plot(u,gval,color='red')
    # plt.plot(u,gvalhat,color='green')
    # plt.savefig("test.png")
    # #plt.show()
    # #print(gvalhat[0])
    # #print(gval[0])

    # return(W1_f,W2_f,W3_f,W1_g,W2_g,W3_g)

# W1_f,W2_f,W1_g,W2_g=Train()
if __name__ == '__main__':
	Train()
