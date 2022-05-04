import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
import BPA1Layer_4A
import BPA2Layer_4A
import OSLA_4A
from vaf import vaf

n=3

k=500
end=800
endval=np.arange(-1*n,end)

u=np.empty((end+n,1),dtype='float')
f=np.empty((end+n,1),dtype='float')

for i in range(0,end+n):
    if(i<(k+n)):
        u[i]=np.sin(2*np.pi*i/250)
    else:
        u[i]=0.8*np.sin(2*np.pi*i/250) + 0.2*np.sin(2*np.pi*i/25)

yp=np.empty((end+n,1),dtype='float')

yp[0]=0
yp[1]=0
yp[2]=0


for i in range(n,end+n):
    f[i-1]=(yp[i-1]*yp[i-2]*yp[i-3]*u[i-2]*(yp[i-3]-1)+u[i-1])/(1+yp[i-3]**2 + yp[i-2]**2)        
    yp[i]=f[i-1]


# BPA1Layer_4A.Run()
# BPA2Layer_4A.Run()
# OSLA_4A.Run()


OSLA_yphat=pickle.load(open("OSLA_4A_yphat.pickle","rb"))
OSLA_J=pickle.load(open("OSLA_4A_J.pickle","rb"))
BPA1_yphat=pickle.load(open("BPA1Layer_4A_yphat.pickle","rb"))
BPA1_J=pickle.load(open("BPA1Layer_4A_J.pickle","rb"))
BPA2_yphat=pickle.load(open("BPA2Layer_4A_yphat.pickle","rb"))
BPA2_J=pickle.load(open("BPA2Layer_4A_J.pickle","rb"))

# print("Cost Using BPA for 1 Hidden Layer = ",BPA1_J[0])

print("Cost Using BPA for 2 Hidden Layer = ",BPA2_J[0])
print("VAF BPA2 ",vaf(yp,BPA2_yphat))
print("Cost Using OSLA = ",OSLA_J[0])
print("VAF OSLA", vaf(yp, OSLA_yphat))

BPA1_J=round(BPA1_J[0],5)
BPA2_J=round(BPA2_J[0],5)
OSLA_J=round(OSLA_J[0],5)

# s1="BPA1 Error = "+str(BPA1_J)
# s2="BPA2 Error = "+str(BPA2_J)
# s3="OSLA Error = "+str(OSLA_J)

plt.figure()
plt.plot(endval,yp,color='red')
# plt.plot(endval,BPA1_yphat,color='green')
plt.plot(endval,BPA2_yphat,color='blue')
plt.plot(endval,OSLA_yphat,color='black')
# plt.text(800,0.25,s1)
# plt.text(800,0,s2)
# plt.text(800,-0.25,s3)
plt.legend(["Plant","BPA1Layer","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 4A Complete")
plt.savefig('final_4a.png')
plt.show()



