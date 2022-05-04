import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
import BPA2Layer_1A
import OSLA_1A
from vaf import vaf

n=2
k=500
kval=np.arange(-1*n,k)

end=1000
endval=np.arange(-1*n,end)

u=np.sin(2*np.pi*endval/250)
f=0.6*np.sin(np.pi*u)+0.3*np.sin(3*np.pi*u)+0.1*np.sin(5*np.pi*u)
yp=np.empty((end+2,1),dtype='float')
yp[0]=0
yp[1]=0
for i in range(2,end+2):
	yp[i]=0.3*yp[i-1]+0.6*yp[i-2]+f[i-1]

"""
BPA1Layer_1A.Run()
BPA2Layer_1A.Run()
OSLA_1A.Run()
"""

OSLA_yphat=pickle.load(open("OSLA_1A_yphat.pickle","rb"))
OSLA_J=pickle.load(open("OSLA_1A_J.pickle","rb"))
BPA1_yphat=pickle.load(open("BPA1Layer_yphat.pickle","rb"))
BPA1_J=pickle.load(open("BPA1Layer_J.pickle","rb"))
BPA2_yphat=pickle.load(open("BPA2Layer_yphat.pickle","rb"))
BPA2_J=pickle.load(open("BPA2Layer_J.pickle","rb"))


print("Cost Using BPA for 2 Hidden Layer = ",BPA2_J[0])
print("Cost Using OSLA = ",OSLA_J[0])


BPA2_J=round(BPA2_J[0],5)
OSLA_J=round(OSLA_J[0],5)

s2="BPA2 Error = "+str(BPA2_J)
s3="OSLA Error = "+str(OSLA_J)

plt.figure()
plt.plot(endval,yp,color='red')

plt.plot(endval,BPA2_yphat,color='blue')
print("VAF BPA2:", vaf(yp[0:500],BPA2_yphat[0:500]))
plt.plot(endval,OSLA_yphat,color='black')
print("VAF OSLA:", vaf(yp[0:500],OSLA_yphat[0:500]))

# plt.text(925,0.5,s2)
# plt.text(925,0.25,s3)
plt.legend(["Plant","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 1A Complete")
plt.savefig('final_1a_complete.png')


plt.figure()
plt.plot(kval,yp[0:k+n],color='red')
plt.plot(kval,BPA1_yphat[0:k+n],color='green')
plt.plot(kval,BPA2_yphat[0:k+n],color='blue')
plt.plot(kval,OSLA_yphat[0:k+n],color='black')

# plt.text(425,0.5,s2)
# plt.text(425,0.25,s3)
plt.legend(["Plant","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 1A Training")
plt.savefig('final_1a_training.png')

plt.show()



