import numpy as np
import pandas as pd
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import pickle
import BPA1Layer_6A
import BPA2Layer_6A
import OSLA_6A
from vaf import vaf

n=1

end=200
endval=np.arange(-1*n,end)


u=np.empty((end+n,1),dtype='float')
f=np.empty((end+n,1),dtype='float')

yp=np.empty((end+n,1),dtype='float')
yp[0]=0




for i in range(n,end+n):
    u[i-1]=np.sin(2*np.pi*(i-n)/25)
    f[i-1]=u[i-1]*(u[i-1]+0.5)*(u[i-1]-0.8)
    yp[i]=0.8*yp[i-1]+f[i-1]

# BPA1Layer_6A.Run()
# BPA2Layer_6A.Run()
# OSLA_6A.Run()


OSLA_yphat=pickle.load(open("OSLA_6A_yphat.pickle","rb"))
OSLA_J=pickle.load(open("OSLA_6A_J.pickle","rb"))
BPA1_yphat=pickle.load(open("BPA1Layer_6A_yphat.pickle","rb"))
BPA1_J=pickle.load(open("BPA1Layer_6A_J.pickle","rb"))
BPA2_yphat=pickle.load(open("BPA2Layer_6A_yphat.pickle","rb"))
BPA2_J=pickle.load(open("BPA2Layer_6A_J.pickle","rb"))

# print("Cost Using BPA for 1 Hidden Layer = ",BPA1_J[0])
print("Cost Using BPA for 2 Hidden Layer = ",BPA2_J[0])
print("VAF BPA2", vaf(yp,BPA2_yphat))
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
# print(len(endval),len(OSLA_yphat))
plt.plot(endval,OSLA_yphat,color='black')
# plt.text(200,0,s1)
# plt.text(200,-0.2,s2)
# plt.text(200,-0.4,s3)
plt.legend(["Plant","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 6A Identification Complete")
plt.savefig('final_6a_id.png')

t_t = np.arange(-1,1,0.01)
f_t = np.zeros([len(t_t)])

for i in range(0,len(t_t)):
    f_t[i]=(t_t[i]-0.8)*t_t[i]*(t_t[i]+0.5)

BPA1Layer_6A_J_U=pickle.load(open("BPA1Layer_6A_J_U.pickle","rb"))
BPA1Layer_6A_NT=pickle.load(open("BPA1Layer_6A_NT.pickle","rb"))
BPA2Layer_6A_J_U=pickle.load(open("BPA2Layer_6A_J_U.pickle","rb"))
BPA2Layer_6A_NT=pickle.load(open("BPA2Layer_6A_NT.pickle","rb"))
OSLA_6A_J_U=pickle.load(open("OSLA_6A_J_U.pickle","rb"))
OSLA_6A_NT=pickle.load(open("OSLA_6A_NT.pickle","rb"))

# print("Cost for F Using BPA for 1 Hidden Layer = ",BPA1Layer_6A_J_U)
print("Cost for F Using BPA for 2 Hidden Layer = ",BPA2Layer_6A_J_U)
print("VAF BPA f",vaf(f_t, BPA2Layer_6A_NT))
print("Cost for FUsing OSLA = ",OSLA_6A_J_U)
print("VAF OSLA f",vaf(f_t, OSLA_6A_NT))

BPA1_JU=round(BPA1Layer_6A_J_U,5)
BPA2_JU=round(BPA2Layer_6A_J_U,5)
OSLA_JU=round(OSLA_6A_J_U,5)

# s1="BPA1 Error = "+str(BPA1_JU)
s2="BPA2 Error = "+str(BPA2_JU)
s3="OSLA Error = "+str(OSLA_JU)

plt.figure()
plt.plot(t_t,f_t,color='red')
# plt.plot(t_t,BPA1Layer_6A_NT,color='green')
plt.plot(t_t,BPA2Layer_6A_NT,color='blue')
plt.plot(t_t,OSLA_6A_NT,color='black')
# plt.text(0.7,-0.4,s1)
# plt.text(0.7,-0.45,s2)
# plt.text(0.7,-0.5,s3)
plt.legend(["Plant","BPA2Layer","OSLA"])
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")
plt.title("Example 6A Function Complete")
plt.savefig('final_6a_complete.png')
plt.show()


