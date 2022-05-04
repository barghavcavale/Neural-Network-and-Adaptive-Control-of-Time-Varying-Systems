import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

T=1
step_size = 0.001
t_val=np.arange(0,T,step_size)

w = 2*np.pi
alpha = 1.2

A11 = w*( -1*np.ones((1,len(t_val))) + alpha*np.cos(w*t_val)**2)
A11 = np.reshape(A11,(len(t_val),1))
A12 = w*(1*np.ones((1,len(t_val))) - alpha*np.cos(w*t_val)*np.sin(w*t_val))
A12 = np.reshape(A11,(len(t_val),1))
A21 = w*( -1*np.ones((1,len(t_val))) - alpha*np.cos(w*t_val)*np.sin(w*t_val))
A21 = np.reshape(A11,(len(t_val),1))
A22 = w*( -1*np.ones((1,len(t_val))) + alpha*np.sin(w*t_val)**2)
A22 = np.reshape(A11,(len(t_val),1))

B11 = -t_val +(alpha/2)*(t_val+0.5*np.sin(2*t_val))
B12 = +t_val -(alpha/2)*(-0.5*np.cos(2*t_val))
B21 = -t_val -(alpha/2)*(-0.5*np.cos(2*t_val))
B22 = -t_val +(alpha/2)*(t_val-0.5*np.sin(2*t_val))

# C = A*B
C11 = (A11*B11)+(A12*B21)
C12 = (A11*B12)+(A12*B22)
C21 = (A21*B11)+(A22*B21)
C22 = (A21*B12)+(A22*B22) 

# D = B*A
D11 = (B11*A11)+(B12*A21)
D12 = (B11*A12)+(B12*A22)
D21 = (B21*A11)+(B22*A21)
D22 = (B21*A12)+(B22*A22) 

# if(C11.all()==D11.all()):
#     print(1)
# else:
#     print(0)
# if(C12.all()==D12.all()):
#     print(1)
# else:
#     print(0)
# if(C21.all()==D21.all()):
#     print(1)
# else:
#     print(0)
# if(C22.all()==D22.all()):
#     print(1)
# else:
#     print(0)

T=20
step_size = 0.001
t_val=np.arange(0,T,step_size)

u1 = np.ones((len(t_val),1))
u2 = np.ones((len(t_val),1))
z1 = np.zeros((len(t_val),1))
z2 = np.zeros((len(t_val),1))

yp = np.zeros((len(t_val),1))
yp[0] = 1*z1[0] + 1*z2[0]
for i in range(0,len(t_val)-1):
    z1[i+1] = z1[i] + (w*(alpha-1)*z1[i] + 0 - np.sin(w*i)*u1[i])*step_size 
    z2[i+1] = z2[i] + (0 - w*z2[i] + np.cos(w*i)*u2[i])*step_size 
    yp[i+1] = np.cos(w*(i+1))*z1[i+1] + np.sin(w*(i+1))*z2[i+1]

plt.figure()
plt.plot(t_val,yp)
plt.title("Ideal Plant without Controller")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(["Plant"])

g1 = 3.7699
z1 = np.zeros((len(t_val),1))
z2 = np.zeros((len(t_val),1))
z1[0] = 0
z2[0] = 0
yp = np.zeros((len(t_val),1))
yp[0] = np.sin(w*0)*z1[0] + np.cos(w*0)*z2[0]

for i in range(0,len(t_val)-1):
    z1[i+1] = z1[i] + ((w*(alpha-1)+g1*np.sin(w*i)*(np.cos(w*i)-np.sin(w*i)))*z1[i] + 0 -np.sin(w*i)*u1[i])*step_size
    z2[i+1] = z2[i] + ((g1*np.cos(w*i)*(np.cos(w*i)-np.sin(w*i)))*z1[i] - w*z2[i] + np.cos(w*i)*u2[i])*step_size 
    
    yp[i+1] = np.cos(w*(i+1))*z1[i+1] + np.sin(w*(i+1))*z2[i+1]

plt.figure()
plt.plot(t_val,yp)
plt.title("Plant with Controller g1 = "+str(g1))
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(["Plant"])

z1 = np.zeros((len(t_val),1))
z2 = np.zeros((len(t_val),1))

yp = np.zeros((len(t_val),1))
yp[0] = np.sin(w*0)*z1[0] + np.cos(w*0)*z2[0]
for i in range(0,len(t_val)-1):
    z1[i+1] = z1[i] + ((w*(alpha-1)+g1*np.sin(w*i)*(np.cos(w*i)-np.sin(w*i)))*z1[i] + 0 - np.sin(w*i)*u1[i])*step_size
    z2[i+1] = z2[i] + ((g1*np.cos(w*i)*(np.cos(w*i)-np.sin(w*i)))*z1[i] - w*z2[i]+ np.cos(w*i)*u2[i])*step_size 
    
    yp[i+1] = np.cos(w*(i+1))*z1[i+1] + np.sin(w*(i+1))*z2[i+1]
    
plt.show()