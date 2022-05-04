import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
import json
import Train2Layer_Ex8
import TrainOSLA_8A

n=3

mu=0.01

Train2Layer_Ex8.Train(mu)
TrainOSLA_8A.Train(mu)

end=10000
endval=np.arange(-1*n,end)

r=np.zeros((end+n,1),dtype='float')
yp=np.zeros((end+n,1),dtype='float')
yp3=np.empty((end+n,1),dtype='float')
ym=np.zeros((end+n,1),dtype='float')
uc=np.zeros((end+n,1),dtype='float')
uc3=np.zeros((end+n,1),dtype='float')

N=np.zeros((end+n,1),dtype='float')

with open("weights8a.json","r") as outfile:
    W1,W2,W3 = json.load(outfile)

with open("weights_osla.json","r") as outfile:
    W1_OSLA,W2_OSLA = json.load(outfile)

def f(a,b,c):
    return 5*a*b/(1 + a**2 + b**2 + c**2)

for i in range(3,end+n):
    
    # r[i-1]=np.sin(2*np.pi*(i-n)/25)  #Ex 8A
    r[i-1]=np.sin(2*np.pi*(i-n)/25) + np.sin(2*np.pi*(i-n)/10) #Ex 8B
          
    ym[i]=0.32*ym[i-1]+0.64*ym[i-2]-0.5*ym[i-3]+r[i-1]

    bias = 1
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

    # if(f[i]==0):
    #     print(yp[i-1]," ",yp[i-2]," ",yp[i-3]," ",f[i])
    #     break

    N[i-1] = C
    uc[i]=-C-0.8*uc[i-1]+0.32*yp[i-1]+0.64*yp[i-2]-0.5*yp[i-3]+r[i-1]
    #print(f[i-1], " ", uc[i-1])
    yp[i] = np.sin(mu*(i-1))*f(yp[i-1],yp[i-2],yp[i-3]) + uc[i-1] +0.8*uc[i-2]
   
    S=yp3[i-1]
    T=yp3[i-2]
    U=yp3[i-3]
    v0k=np.array([bias,S[0],T[0],U[0]])
    v1k_bar=tan_h(np.matmul(W1_OSLA,v0k))
    v1k=np.insert(v1k_bar,0,[bias])
    fin=np.matmul(W2_OSLA,v1k)

    uc3[i]=-fin-0.8*uc3[i-1]+0.32*yp3[i-1]+0.64*yp3[i-2]-0.5*yp3[i-3]+r[i-1]
    

    yp3[i] = np.sin(mu*(i-1))*f(yp3[i-1],yp3[i-2],yp3[i-3]) + uc3[i-1] +0.8*uc3[i-2]


val=ym[0]-yp[0]
plt.figure()
plt.plot(endval[9900:10000],yp[9900:10000],color='red')
# plt.plot(endval[9900:10000],yp3[9900:10000]-500,color='black')
plt.plot(endval[9900:10000],ym[9900:10000],color='blue')
plt.legend(["BPA","OSLA","Model"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.savefig("np90_8B_Offline_TV_2.png")
plt.show()

