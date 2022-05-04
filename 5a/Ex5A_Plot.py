import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
import BPA1Layer_5A
import BPA2Layer_5A
import OSLA_5A
from vaf import vaf

n=1
end=100
endval=np.arange(-1*n,end)

u_N1=np.empty((end+n,1),dtype='float')
u_N2=np.empty((end+n,1),dtype='float')

#print(u)
f_N1=np.zeros([end+n,1])
f_N2=np.zeros([end+n,1])

yp_N1=np.empty((end+n,1),dtype='float')
yp_N2=np.empty((end+n,1),dtype='float')
yp_N1[0]=0
yp_N2[0]=0

for i in range(0,end+n):
    u_N1[i]=np.sin(2*np.pi*i/25)
    u_N2[i]=np.cos(2*np.pi*i/25)

for i in range(n,end+n):
    f_N1[i-1]=yp_N1[i-1]/(1+yp_N2[i-1]**2)
    f_N2[i-1]=(yp_N1[i-1]*yp_N2[i-1])/(1+yp_N2[i-1]**2)
    yp_N1[i]=f_N1[i-1]+u_N1[i-1]
    yp_N2[i]=f_N2[i-1]+u_N2[i-1]

# BPA1Layer_5A.Run()
# BPA2Layer_5A.Run()
# OSLA_5A.Run()


OSLA_yphat_N1=pickle.load(open("OSLA_5A_yphat_N1.pickle","rb"))
OSLA_J_N1=pickle.load(open("OSLA_5A_J_N1.pickle","rb"))
BPA1_yphat_N1=pickle.load(open("BPA1Layer_5A_yphat_N1.pickle","rb"))
BPA1_J_N1=pickle.load(open("BPA1Layer_5A_J_N1.pickle","rb"))
BPA2_yphat_N1=pickle.load(open("BPA2Layer_5A_yphat_N1.pickle","rb"))
BPA2_J_N1=pickle.load(open("BPA2Layer_5A_J_N1.pickle","rb"))

OSLA_yphat_N2=pickle.load(open("OSLA_5A_yphat_N2.pickle","rb"))
OSLA_J_N2=pickle.load(open("OSLA_5A_J_N2.pickle","rb"))
BPA1_yphat_N2=pickle.load(open("BPA1Layer_5A_yphat_N2.pickle","rb"))
BPA1_J_N2=pickle.load(open("BPA1Layer_5A_J_N2.pickle","rb"))
BPA2_yphat_N2=pickle.load(open("BPA2Layer_5A_yphat_N2.pickle","rb"))
BPA2_J_N2=pickle.load(open("BPA2Layer_5A_J_N2.pickle","rb"))


# print("Cost Using BPA for 1 Hidden Layer N1= ",BPA1_J_N1[0])
# print("Cost Using BPA for 1 Hidden Layer N2= ",BPA1_J_N2[0])
print("Cost Using BPA for 2 Hidden Layer N1= ",BPA2_J_N1[0])
print("VAF BPA2 N1",vaf(yp_N1, BPA2_yphat_N1))
print("Cost Using BPA for 2 Hidden Layer N2= ",BPA2_J_N2[0])
print("VAF BPA2 N2",vaf(yp_N2, BPA2_yphat_N2))
print("Cost Using OSLA N1= ",OSLA_J_N1[0])
print("VAF OSLA N1",vaf(yp_N1, OSLA_yphat_N1))
print("Cost Using OSLA N2= ",OSLA_J_N2[0])
print("VAF OSLA N2",vaf(yp_N2, OSLA_yphat_N2))

BPA1_J_N1=round(BPA1_J_N1[0],5)
BPA2_J_N1=round(BPA2_J_N1[0],5)
OSLA_J_N1=round(OSLA_J_N1[0],5)

# s1_N1="BPA1 N1 Error = "+str(BPA1_J_N1)
# s2_N1="BPA2 N1 Error = "+str(BPA2_J_N1)
# s3_N1="OSLA N1 Error = "+str(OSLA_J_N1)

BPA1_J_N2=round(BPA1_J_N2[0],5)
BPA2_J_N2=round(BPA2_J_N2[0],5)
OSLA_J_N2=round(OSLA_J_N2[0],5)

# s1_N2="BPA1 N2 Error = "+str(BPA1_J_N2)
# s2_N2="BPA2 N2 Error = "+str(BPA2_J_N2)
# s3_N2="OSLA N2 Error = "+str(OSLA_J_N2)

plt.figure()
# plt.subplot(1,2,1)
plt.plot(endval,yp_N1,color='red')
# plt.plot(endval,BPA1_yphat_N1,color='green')
plt.plot(endval,BPA2_yphat_N1,color='blue')
plt.plot(endval,OSLA_yphat_N1,color='black')
# plt.text(90,1.75,s1_N1)
# plt.text(90,1.5,s2_N1)
# plt.text(90,1.25,s3_N1)
plt.legend(["Plant","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 5A N1 Complete")
plt.savefig('final_5a_N1.png')

plt.figure()
# plt.subplot(1,2,2)
plt.plot(endval,yp_N2,color='red')
# plt.plot(endval,BPA1_yphat_N2,color='green')
plt.plot(endval,BPA2_yphat_N2,color='blue')
plt.plot(endval,OSLA_yphat_N2,color='black')
# plt.text(100,1.75,s1_N2)
# plt.text(100,1.5,s2_N2)
# plt.text(100,1.25,s3_N2)
plt.legend(["Plant","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 5A N2 Complete")
plt.savefig('final_5a_N2.png')

plt.show()



