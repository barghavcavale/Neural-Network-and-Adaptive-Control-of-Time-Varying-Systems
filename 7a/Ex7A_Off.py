import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
#import BPA1Layer_7A
#import BPA2Layer_7A
#import OSLA_7A
import json
from vaf import vaf

n=2

end=10000
endval=np.arange(-1*n,end)


r=np.empty((end+n,1),dtype='float')

ypnc=np.empty((end+n,1),dtype='float')
yp=np.empty((end+n,1),dtype='float')
yp3=np.empty((end+n,1),dtype='float')
ym=np.empty((end+n,1),dtype='float')
N=np.empty((end+n,1),dtype='float')

with open("weights.json","r") as outfile:
    W1,W2,W3 = json.load(outfile)

with open("weights_osla.json","r") as outfile:
    W1_OSLA,W2_OSLA = json.load(outfile)

def f(a,b):
    return (a*b*(a+2.5))/(1+a**2 + b**2)

for i in range(2,end+n):
    r[i-1]=np.sin(2*np.pi*(i-n)/25)
 
    ypnc[i]=f(ypnc[i-1],ypnc[i-2])+r[i-1]
    ym[i]=0.6*ym[i-1]+0.2*ym[i-2]+r[i-1]

    bias = 1
    S=yp[i-1]
    T=yp[i-2]
    x=np.array([bias,S[0],T[0]])
    A1=np.matmul(W1,x)
    A=tan_h(A1)
    y=np.insert(A,0,[bias])
    B1=np.matmul(W2,y)
    B=tan_h(B1)
    z=np.insert(B,0,[bias])
    C1=np.matmul(W3,z)
    #C=tan_h(C1)
    C=C1

    N[i-1] = C
    yp[i] = f(yp[i-1],yp[i-2]) - N[i-1] + 0.6*yp[i-1] + 0.2*yp[i-2] + r[i-1]
 
    S=yp3[i-1]
    T=yp3[i-2]
    v0k=np.array([bias,S[0],T[0]])
    v1k_bar=tan_h(np.matmul(W1_OSLA,v0k))
    v1k=np.insert(v1k_bar,0,[bias])
    fin=np.matmul(W2_OSLA,v1k)

    yp3[i] = f(yp3[i-1],yp3[i-2]) - fin + 0.6*yp3[i-1] + 0.2*yp3[i-2] + r[i-1]
    

# BPA1Layer_2A.Run()
# BPA2Layer_2A.Run()
# OSLA_2A.Run()






# plt.figure()
# plt.plot(endval,yp,color='red')
# # plt.plot(endval,BPA1_yphat,color='green')
# plt.plot(endval,BPA2_yphat,color='blue')
# # plt.plot(endval,OSLA_yphat,color='black')
# # plt.text(100,1.75,s1)
# plt.text(70,1.5,s2)
# # plt.text(100,1.25,s3)
# # plt.legend(["Plant","BPA1Layer","BPA2Layer","OSLA"])
# plt.legend(["Plant","BPA2Layer"])
# plt.xlabel("Time Steps")
# plt.ylabel("Amplitude")
# plt.title("Example 7A Complete")
# plt.savefig("np90_7a.png")
# plt.show()
val=ym[0]-yp[0]
plt.figure()
plt.plot(endval[9900:10000],yp[9900:10000]+0.4,color='red')
plt.plot(endval[9900:10000],yp3[9900:10000],color='green')
plt.plot(endval[9900:10000],ym[9900:10000],color='blue')
plt.legend(["Plant BPA 2 Layer","Plant OSLA","Model"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title('Example 7a Offline')
print("VAF BPA2",vaf(ym[9900:10000], yp[9900:10000]+0.4))
print("VAF OSLA",vaf(ym[9900:10000], yp3[9900:10000]))

plt.savefig("final_7b_off.png")
plt.show()

