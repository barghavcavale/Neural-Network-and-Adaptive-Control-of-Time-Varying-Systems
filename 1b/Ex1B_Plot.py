import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
import BPA1Layer_1B
import BPA2Layer_1B
import OSLA_1B
from vaf import vaf

n=2
k=250
kval=np.arange(-1*n,k)

end=500
endval=np.arange(-1*n,end)

u=np.empty((end+n,1),dtype='float')
f=np.empty((end+n,1),dtype='float')
yp=np.empty((end+n,1),dtype='float')
	

for i in range(2,end+n):
	if(i<k):
		u[i]=np.sin(2*np.pi*(i-n)/250)
	else:	
		u[i]=np.sin(2*np.pi*(i-n)/250)+np.sin(2*np.pi*(i-n)/25)
		u[i]=u[i]/2
	f[i]=np.power(u[i],3)+0.3*np.power(u[i],2)-0.4*u[i]
	yp[i]=0.3*yp[i-1]+0.6*yp[i-2]+f[i-1]

# BPA1Layer_1B.Run()
# BPA2Layer_1B.Run()
# OSLA_1B.Run()


OSLA_yphat=pickle.load(open("OSLA_1B_yphat.pickle","rb"))
OSLA_J=pickle.load(open("OSLA_1B_J.pickle","rb"))
BPA1_yphat=pickle.load(open("BPA1Layer_1B_yphat.pickle","rb"))
BPA1_J=pickle.load(open("BPA1Layer_1B_J.pickle","rb"))
BPA2_yphat=pickle.load(open("BPA2Layer_1B_yphat.pickle","rb"))
BPA2_J=pickle.load(open("BPA2Layer_1B_J.pickle","rb"))

# print("Cost Using BPA for 1 Hidden Layer = ",BPA1_J[0])
print("Cost Using BPA for 2 Hidden Layer = ",BPA2_J[0])
print("Cost Using OSLA = ",OSLA_J[0])

# BPA1_J=round(BPA1_J[0],5)
BPA2_J=round(BPA2_J[0],5)
OSLA_J=round(OSLA_J[0],5)

# s1="BPA1 Error = "+str(BPA1_J)
s2="BPA2 Error = "+str(BPA2_J)
s3="OSLA Error = "+str(OSLA_J)

plt.figure()
plt.plot(endval,yp,color='red')
# plt.plot(endval,BPA1_yphat,color='green')
plt.plot(endval,BPA2_yphat,color='blue')
print('VAF BPA2', vaf(yp,BPA2_yphat))
plt.plot(endval,OSLA_yphat,color='black')
print('VAF OSLA', vaf(yp, OSLA_yphat))
# plt.text(400,1.75,s1)
# plt.text(400,1.5,s2)
# plt.text(400,1.25,s3)
plt.legend(["Plant","BPA1Layer","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 1B Complete")
plt.savefig('final_1b.png')
plt.show()



