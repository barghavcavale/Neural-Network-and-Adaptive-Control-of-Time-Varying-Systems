import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import json

def f(a,b,c):
    return (5*a*b)/(1+a**2 + b**2 + c**2)

def Train():

    n=3
    end=50
    
    endval=np.arange(-1*n,end)
    #u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)
    #print(u)

    yp=np.empty((end+n,1),dtype='float')
    r=np.empty((end+n,1),dtype='float')
    yp[0]=0
    yp[1]=0
    yp[2]=0
	
    for i in range(0,end+n):
        r[i] = np.sin(2*np.pi*(i)/25) #9a
        # r[i] = np.sin(2*np.pi*(i)/25)+np.sin(2*np.pi*(i)/10) #9b
        

    yphat=np.empty((end+n,1),dtype='float')
    ym=np.empty((end+n,1),dtype='float')
    uc=np.zeros((end+n,1),dtype='float')
    yphat[0]=0
    yphat[1]=0
    yphat[2]=0

    ym[0]=0
    ym[1]=0
    ym[2]=0

    bias=1
    L1=20
    L2=10
    eta=0.1

    W1=np.random.normal(0,0.1,(L1,3+1))
    W2=np.random.normal(0,0.1,(L2,L1+1))
    W3=np.zeros([1,L2+1])
    Jc=0
    Ji=0
    epochs=1
    Ti=1
    Tc=1

    for a in range(epochs):
        for i in range(n,end+n):

            ym[i]=0.32*ym[i-1]+0.64*ym[i-2]-0.5*ym[i-3]+r[i-1]

            S=yp[i-1]
            T=yp[i-2]
            U=yp[i-3]
            x=np.array([bias,S[0],T[0],U[0]])
            A1=np.matmul(W1,x)
            A=tan_h(A1)
            y=np.insert(A,0,[bias])
            B1=np.matmul(W2,y)
            B=tan_h(B1)
            z=np.insert(B,0,[bias])
            C1=np.matmul(W3,z)
            #C=tan_h(C1)
            C=C1

            if((i-2)%Tc==0):
                uc[i]=-C -1.1*uc[i-1] + 0.32*yp[i-1]+0.64*yp[i-2]-0.5*yp[i-3]+r[i-1]
            else:
                uc[i]=uc[i-1]
            yp[i]=f(yp[i-1],yp[i-2],yp[i-3])+uc[i]+1.1*uc[i-1]
            ec=ym[i]-yp[i]
            #del3=e*dtan_h(C1)

            Jc = Jc + ec*ec
            if((i-2)%Ti == 0):
                yphat[i]=C+uc[i]+1.1*uc[i-1]
                ei = yphat[i]-yp[i]
                del3=ei
                del2=np.matmul((np.transpose(W3[...,1:])),del3)*dtan_h(B1)
                del1=np.matmul((np.transpose(W2[...,1:])),del2)*dtan_h(A1)
                del2=del2.reshape(len(del2),1)
                del1=del1.reshape(len(del1),1)
                y=y.reshape(1,len(y))
                x=x.reshape(1,len(x))
                W3=W3-(eta*del3*np.transpose(z))
                W2=W2-(eta*np.matmul(del2,y))
                W1=W1-(eta*np.matmul(del1,x))
                Ji=Ji+ei*ei

    Ji=Ji*Tc/(end)
    Jc=Jc/(end)
    print("Training Cost for BPA 2 Layer Identification = ",Ji)
    print("Training Cost for BPA 2 Layer Control = ",Jc)
  
    start=0
    end=start+50
    # end=start+100
    plt.figure()
    plt.plot(endval[start:end],yp[start:end],color='red')
    plt.plot(endval[start:end],ym[start:end],color='green',linewidth=0.8)
    plt.legend(["Plant","Model"])
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.savefig("np90_9a_on_yp.png")


    start=0
    end=start+50
    plt.figure()
    plt.plot(endval[start:end],uc[start:end],color='red')
    plt.legend(["Control Input"])
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.savefig("np90_9a_on_u.png")
    plt.show()
    return(W1,W2,W3)

#W1,W2,W3=Train()

if __name__ == "__main__":
    W1,W2,W3=Train()

    weights = [W1.tolist(), W2.tolist(), W3.tolist()]
    #weight_json = json.dumps(weights)
    with open('weights_on.json', 'w') as outfile:
        json.dump(weights, outfile)

