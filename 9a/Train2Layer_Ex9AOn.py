import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import json
from vaf import vaf

def f(a,b,c):
    return (5*a*b)/(1+a**2 + b**2 + c**2)

def Train():

    n=3
    end=50
    
    endval=np.arange(-1*n,end)
    #u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)
    #print(u)

    yp=np.empty((end+n,1),dtype='float')
    yp_OSLA=np.zeros((end+n,1),dtype='float')
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
    uc_OSLA = np.zeros((end+n,1),dtype='float')
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

    L1=18
	#L2=10
    lam=0.000001

    W1_OSLA=np.random.normal(0,0.1,(L1,3+1))
    #W2=np.random.normal(0,0.1,(L2,L1+1))
    W2_OSLA=np.zeros([1,L1+1])
    P0=(1/lam)*np.identity(L1+1)
    Jc_OSLA=0
    Ji_OSLA=0
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

            S=yp[i-1]
            T=yp[i-2]
            U=yp[i-3]
            v0k=np.array([bias,S[0],T[0],U[0]])
            v1k_bar=tan_h(np.matmul(W1_OSLA,v0k))
            v1k=np.insert(v1k_bar,0,[bias])
            fin=np.matmul(W2_OSLA,v1k)

            if((i-2)%Tc==0):
                uc[i]=-C -1.1*uc[i-1] + 0.32*yp[i-1]+0.64*yp[i-2]-0.5*yp[i-3]+r[i-1]
                uc_OSLA[i]=-fin -1.1*uc_OSLA[i-1] + 0.32*yp[i-1]+0.64*yp[i-2]-0.5*yp[i-3]+r[i-1]
            else:
                uc[i]=uc[i-1]
                uc_OSLA=uc_OSLA[i-1]
            yp[i]=f(yp[i-1],yp[i-2],yp[i-3])+uc[i]+1.1*uc[i-1]
            yp_OSLA[i]=f(yp_OSLA[i-1],yp_OSLA[i-2],yp_OSLA[i-3])+uc_OSLA[i]+1.1*uc_OSLA[i-1]
            ec=ym[i]-yp[i]
            #del3=e*dtan_h(C1)
            Jc_OSLA+=(yp_OSLA[i]-ym[i])**2
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

                yphat[i]=fin+uc_OSLA[i]+0.8*uc_OSLA[i-1]
                e=yp_OSLA[i]-yphat[i]
                v1k=v1k.reshape(len(v1k),1)
                v1k_T=v1k.reshape(1,len(v1k))
                Num=np.matmul(P0,np.matmul(np.matmul(v1k,v1k_T),P0))	
                Den=1+np.matmul(v1k_T,np.matmul(P0,v1k))
                P0=P0-(Num/Den)
                W2_OSLA=W2_OSLA+e*np.matmul(v1k_T,P0)
                Ji_OSLA+=e*e
    Ji=Ji*Tc/(end)
    Jc=Jc/(end)
    Ji_OSLA=Ji_OSLA*Tc/(end)
    Jc_OSLA=Jc_OSLA/(end)
    print("Training Cost for BPA 2 Layer Identification = ",Ji)
    print("Training Cost for BPA 2 Layer Control = ",Jc)
    print("VAF BPA2",vaf(ym,yp))
    print("Training Cost for OSLA Identification = ",Ji)
    print("Training Cost for OSLA Control = ",Jc)
    print("VAF OSLA",vaf(ym,yp_OSLA))
    start=0
    end=start+50
    # end=start+100
    plt.figure()
    plt.plot(endval[start:end],yp[start:end],color='red')
    plt.plot(endval[start:end],yp_OSLA[start:end],color='black')
    plt.plot(endval[start:end],ym[start:end],color='green',linewidth=0.8)
    plt.legend(["BPA","OSLA","Model"])
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.title('Example 9a')
    plt.savefig("final_9a_on_yp.png")


    start=0
    end=start+50
    plt.figure()
    plt.plot(endval[start:end],uc[start:end],color='red')
    plt.legend(["Control Input"])
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.savefig("final_9a_on_u.png")
    plt.title('Example 9a Control Input')
    plt.show()
    return(W1,W2,W3)

#W1,W2,W3=Train()

if __name__ == "__main__":
    W1,W2,W3=Train()

    weights = [W1.tolist(), W2.tolist(), W3.tolist()]
    #weight_json = json.dumps(weights)
    with open('weights_on.json', 'w') as outfile:
        json.dump(weights, outfile)

