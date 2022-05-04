import math
import numpy as np 
import control
from matplotlib import pyplot as plt 
import pandas as pd 
import sympy as sp

t_end=100 
dt=0.001
t=np.arange(0,t_end,dt)
t_size=t.shape[0]

r=10*np.sin(t)

mu=1

a1=-6*np.ones([t_size,1])
# a1=-4*np.cos(mu*t)
a2=2*np.sin(mu*t)

a1dot=np.zeros([t_size,1])
# a1dot=4*mu*np.sin(mu*t)
a2dot=2*mu*np.cos(mu*t)

a1doubledot=np.zeros([t_size,1])
# a1doubledot=4*mu*mu*np.cos(mu*t)
a2doubledot=-2*mu*mu*np.sin(mu*t)

# print(a1)
# print(a2)

u_st=np.zeros([t_size,1])

ym_st=np.zeros([t_size,1])
ymdot_st=np.zeros([t_size,1])
yp_st=np.zeros([t_size,1])
ypdot_st=np.zeros([t_size,1])

w1_st=np.zeros([t_size,1])
w2_st=np.zeros([t_size,1])

theta1_st=np.zeros([t_size,1])
theta2_st=np.zeros([t_size,1])
theta3_st=np.zeros([t_size,1])

e_st=np.zeros([t_size,1])
fin_e_st=np.zeros([t_size,1])

# Standard MRC Law
for i in range(0,t_size-1):

	theta1_st[i]=a1[i]-3
	theta2_st[i]=(a1[i]**2)-4*a1[i]-a1[i]*a2[i]+3*a2[i]+3
	theta3_st[i]=4*a1[i]+a2[i]-(a1[i]**2)-5

	u_st[i]=theta1_st[i]*w1_st[i]+theta2_st[i]*w2_st[i]+theta3_st[i]*yp_st[i]+r[i]

	ymdot_st[i+1]=ymdot_st[i]+(-3*ymdot_st[i]-2*ym_st[i]+r[i])*dt
	ym_st[i+1]=ym_st[i]+(ymdot_st[i])*dt

	ypdot_st[i+1]=ypdot_st[i]+(-a1[i]*ypdot_st[i]-a2[i]*yp_st[i]+u_st[i])*dt
	yp_st[i+1]=yp_st[i]+(ypdot_st[i])*dt
	
	e_st[i+1]=yp_st[i+1]-ym_st[i+1]
	fin_e_st[i+1]=np.log(1+abs(e_st[i+1]))
	w1_st[i+1]=w1_st[i]+(-w1_st[i]+u_st[i])*dt
	w2_st[i+1]=w2_st[i]+(-w2_st[i]+yp_st[i])*dt



u_new=np.zeros([t_size,1])

ym_new=np.zeros([t_size,1])
ymdot_new=np.zeros([t_size,1])
yp_new=np.zeros([t_size,1])
ypdot_new=np.zeros([t_size,1])

w1_new=np.zeros([t_size,1])
w2_new=np.zeros([t_size,1])

theta1_new=np.zeros([t_size,1])
theta2_new=np.zeros([t_size,1])
theta3_new=np.zeros([t_size,1])

e_new=np.zeros([t_size,1])

# Standard MRC Law
for i in range(0,t_size-1):

	theta1_new[i]=a1[i]-3
	theta2_new[i]=(a1[i]**2)-4*a1[i]-a1[i]*a2[i]+3*a2[i]+3-5*a1dot[i]+2*a1[i]*a1dot[i]-a1doubledot[i]
	theta3_new[i]=4*a1[i]+a2[i]-(a1[i]**2)-5+a1dot[i]

	u_new[i]=w1_new[i]+w2_new[i]+theta3_new[i]*yp_new[i]+r[i]

	ymdot_new[i+1]=ymdot_new[i]+(-3*ymdot_new[i]-2*ym_new[i]+r[i])*dt
	ym_new[i+1]=ym_new[i]+(ymdot_new[i])*dt

	ypdot_new[i+1]=ypdot_new[i]+(-a1[i]*ypdot_new[i]-a2[i]*yp_new[i]+u_new[i])*dt
	yp_new[i+1]=yp_new[i]+(ypdot_new[i])*dt
	
	e_new[i+1]=yp_new[i+1]-ym_new[i+1]
	
	w1_new[i+1]=w1_new[i]+(-w1_new[i]+theta1_new[i]*u_new[i])*dt
	w2_new[i+1]=w2_new[i]+(-w2_new[i]+theta2_new[i]*yp_new[i])*dt




plt.figure()
plt.plot(t,ym_st,color='red')
plt.plot(t,yp_st,color='green')
plt.legend(["Model","Plant"])
plt.title("Output of Standard MRC for Example 1 Mu = "+str(mu))
plt.xlabel("Time")
plt.ylabel("Amplitude")


plt.figure()
plt.plot(t,ym_new,color='red')
plt.plot(t,yp_new,color='green')
plt.legend(["Model","Plant"])
plt.title("Output of New MRC for Example 1 Mu = "+str(mu))
plt.xlabel("Time")
plt.ylabel("Amplitude")

if(mu==1):
	plt.figure()
	# plt.plot(t,e_st,color='red')
	plt.plot(t,e_new)
	# plt.legend(["Standard MRC","New MRC"])
	plt.ylim(-10,10)
	plt.title("New MRC Error for Example 1 Mu = "+str(mu))
	plt.xlabel("Time")
	plt.ylabel("e")

	plt.figure()
	# plt.plot(t,e_st,color='red')
	plt.plot(t,fin_e_st)
	# plt.legend(["Standard MRC","New MRC"])
	plt.ylim(0,40)
	plt.title("Standard MRC Error for Example 1 Mu = "+str(mu))
	plt.xlabel("Time")
	plt.ylabel("log(1+|e|)")

if(mu==0.1):
	plt.figure()
	plt.plot(t,e_st,color='red')
	plt.plot(t,e_new,color='green')
	plt.legend(["Standard MRC","New MRC"])
	plt.ylim(-10,10)
	plt.title("New MRC Error for Example 1 Mu = "+str(mu))
	plt.xlabel("Time")
	plt.ylabel("e")

plt.show()