import math
import numpy as np 
import control
from matplotlib import pyplot as plt 
import pandas as pd 
import sympy as sp



A1=-6
w=1   # Change w here
A2=2

if(w==1):	
	t_end=300
elif(w==0.1): 
	t_end=100

dt=0.001
t=np.arange(0,t_end,dt)
t_size=t.shape[0]

r=10*np.sin(t)

a1=A1*np.ones([t_size,1])
# a1=-4*np.cos(w*t)
a2=A2*np.sin(w*t)

a1dot=np.zeros([t_size,1])
# a1dot=4*w*np.sin(w*t)
a2dot=2*w*np.cos(w*t)

a1doubledot=np.zeros([t_size,1])
# a1doubledot=4*w*w*np.cos(w*t)
a2doubledot=-2*w*w*np.sin(w*t)

theta1_0_star=A1-3
theta2_0_star=(A1-4)*A1+3
theta2_1_star=(3-A1)*A2
theta3_0_star=(4-A1)*A1-5
theta3_1_star=A2

theta1_0_hat=np.zeros([t_size,1])
theta2_0_hat=np.zeros([t_size,1])
theta2_1_hat=np.zeros([t_size,1])
theta3_0_hat=np.zeros([t_size,1])
theta3_1_hat=np.zeros([t_size,1])

theta1_0_hat[0]=-9.6
theta2_0_hat[0]=72.96
theta3_0_hat[0]=-74.96
theta2_1_hat[0]=21.2
theta3_1_hat[0]=2.2

delta0=0.9
delta1=0.5
sigma0=0.1
M0=120

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

m=np.zeros([t_size,1])
arb_m_ini=0.1
m[0]=(delta1/delta0)+arb_m_ini

in1=np.zeros([t_size,1])
in2=np.zeros([t_size,1])
in3=np.zeros([t_size,1])
in4=np.zeros([t_size,1])
in5=np.zeros([t_size,1])

z1=np.zeros([t_size,1])
z1dot=np.zeros([t_size,1])

z2=np.zeros([t_size,1])
z2dot=np.zeros([t_size,1])

z3=np.zeros([t_size,1])
z3dot=np.zeros([t_size,1])

z4=np.zeros([t_size,1])
z4dot=np.zeros([t_size,1])

z5=np.zeros([t_size,1])
z5dot=np.zeros([t_size,1])


out1=np.zeros([t_size,1])
out1dot=np.zeros([t_size,1])
out1doubledot=np.zeros([t_size,1])

sine=np.sin(w*t)
IL1=np.exp(-t)
IL2=np.exp(-t)-np.exp(-2*t)


eps1=np.zeros([t_size,1])

p=np.zeros([t_size,1])

# plt.plot(t,r)
# plt.title("r")
# plt.figure()
# plt.plot(t,sine)
# plt.title("Sine")
# plt.show()

# Standard MRC Law
for i in range(0,t_size-1):


	theta1_st[i]=theta1_0_hat[i]
	theta2_st[i]=theta2_0_hat[i]+theta2_1_hat[i]*sine[i]
	theta3_st[i]=theta3_0_hat[i]+theta3_1_hat[i]*sine[i]
	
	u_st[i]=theta1_st[i]*w1_st[i]+theta2_st[i]*w2_st[i]+theta3_st[i]*yp_st[i]+r[i]


	ymdot_st[i+1]=ymdot_st[i]+(-3*ymdot_st[i]-2*ym_st[i]+r[i])*dt
	ym_st[i+1]=ym_st[i]+(ymdot_st[i])*dt

	ypdot_st[i+1]=ypdot_st[i]+(-a1[i]*ypdot_st[i]-a2[i]*yp_st[i]+u_st[i])*dt
	yp_st[i+1]=yp_st[i]+(ypdot_st[i])*dt
	
	e_st[i+1]=yp_st[i+1]-ym_st[i+1]
	fin_e_st[i+1]=np.log(1+abs(e_st[i+1]))
	
	w1_st[i+1]=w1_st[i]+(-w1_st[i]+u_st[i])*dt
	w2_st[i+1]=w2_st[i]+(-w2_st[i]+yp_st[i])*dt

	#Additional for MRAC
	#Define in and Z 

	in1[i+1]=in1[i]+(-in1[i]+u_st[i])*dt
	in2[i+1]=in2[i]+(-in2[i]+yp_st[i])*dt
	in3[i+1]=yp_st[i+1]
	in4[i+1]=(in4[i]+(-in4[i]+yp_st[i])*dt)
	in5[i+1]=(yp_st[i+1]*sine[i+1])

	z1dot[i+1]=z1dot[i]+(-3*z1dot[i]-2*z1[i]+in1[i])*dt
	z1[i+1]=z1[i]+(z1dot[i])*dt

	z2dot[i+1]=z2dot[i]+(-3*z2dot[i]-2*z2[i]+in2[i])*dt
	z2[i+1]=z2[i]+(z2dot[i])*dt

	z3dot[i+1]=z3dot[i]+(-3*z3dot[i]-2*z3[i]+in3[i])*dt
	z3[i+1]=z3[i]+(z3dot[i])*dt

	z4dot[i+1]=z4dot[i]+(-3*z4dot[i]-2*z4[i]+(in4[i]*sine[i]))*dt
	z4[i+1]=z4[i]+(z4dot[i])*dt

	z5dot[i+1]=z5dot[i]+(-3*z5dot[i]-2*z5[i]+in5[i])*dt
	z5[i+1]=z5[i]+(z5dot[i])*dt
	

	#THETA_Hat'*Z
	p[i]=(theta1_0_hat[i]*z1[i])+(theta2_0_hat[i]*z2[i])+(theta3_0_hat[i]*z3[i])+(theta2_1_hat[i]*z4[i])+(theta3_1_hat[i]*z5[i])

	#out1
	out1dot[i+1]=out1dot[i]+(-3*out1dot[i]-2*out1[i]+u_st[i])*dt
	out1[i+1]=out1[i]+(out1dot[i])*dt

	#epsilon1
	eps1[i]=yp_st[i]+p[i]-out1[i]

	#||THETA||
	norm_theta=math.sqrt((theta1_0_hat[i]**2)+(theta2_0_hat[i]**2)+(theta2_1_hat[i]**2)+(theta3_0_hat[i]**2)+(theta3_1_hat[i]**2))

	#sigma
	if(norm_theta<M0):
		sigma=0
	elif(norm_theta>=M0 and norm_theta<=(2*M0)):
		sigma=sigma0*((norm_theta/M0)-1)       #???? OR
		# sigma=sigma0*((norm_theta)/(M0-1))		 #????
	elif(norm_theta>(2*M0)):
		sigma=sigma0
	
	#Update THETA
	mval=m[i]**2

	# print(mval)
	theta1_0_hat[i+1]=theta1_0_hat[i]+(((-100)*(eps1[i]*z1[i])/(mval))-(sigma*theta1_0_hat[i]))*dt
	theta2_0_hat[i+1]=theta2_0_hat[i]+(((-100)*(eps1[i]*z2[i])/(mval))-(sigma*theta2_0_hat[i]))*dt
	theta3_0_hat[i+1]=theta3_0_hat[i]+(((-100)*(eps1[i]*z3[i])/(mval))-(sigma*theta3_0_hat[i]))*dt
	theta2_1_hat[i+1]=theta3_0_hat[i]+(((-100)*(eps1[i]*z4[i])/(mval))-(sigma*theta2_1_hat[i]))*dt
	theta3_1_hat[i+1]=theta3_1_hat[i]+(((-100)*(eps1[i]*z5[i])/(mval))-(sigma*theta3_1_hat[i]))*dt

	#Update m
	m[i+1]=m[i]+(-delta0*m[i]+delta1*(abs(u_st[i])+abs(yp_st[i])+1))*dt
	# if(math.isnan(mval)):
	# 	print("BLOWN UP",t[i])
	# 	break

# plt.figure()
# plt.plot(t,ym_st)
# plt.title("ym")
# plt.show()		


# for i in range(t_size):
# 	print(i)
# 	print("z1 =",z1[i])
# 	print("z2 =",z2[i])
# 	print("z3 =",z3[i])
# 	print("z4 =",z4[i])
# 	print("z5 =",z5[i])
# 	print("out1 =",out1[i])
# 	print("eps1 =",eps1[i])
	# print("theta1_0_hat =",theta1_0_hat[i])
	# print("theta2_0_hat =",theta2_0_hat[i])
	# print("theta2_1_hat =",theta2_1_hat[i])
	# print("theta3_0_hat =",theta3_0_hat[i])
	# print("theta3_1_hat =",theta3_1_hat[i])
	# print("m =",m[i])
	# print("\n")

theta1_0_hat=np.zeros([t_size,1])
theta2_0_hat=np.zeros([t_size,1])
theta2_1_hat=np.zeros([t_size,1])
theta3_0_hat=np.zeros([t_size,1])
theta3_1_hat=np.zeros([t_size,1])

theta1_0_hat[0]=-9.6
theta2_0_hat[0]=72.96
theta3_0_hat[0]=-74.96
theta2_1_hat[0]=21.2
theta3_1_hat[0]=2.2

delta0=0.9
delta1=0.5
sigma0=0.1
M0=120

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
fin_e_new=np.zeros([t_size,1])

m=np.zeros([t_size,1])
arb_m_ini=0.1
m[0]=(delta1/delta0)+arb_m_ini

in1=np.zeros([t_size,1])
in2=np.zeros([t_size,1])
in3=np.zeros([t_size,1])
in4=np.zeros([t_size,1])
in5=np.zeros([t_size,1])

z1=np.zeros([t_size,1])
z1dot=np.zeros([t_size,1])

z2=np.zeros([t_size,1])
z2dot=np.zeros([t_size,1])

z3=np.zeros([t_size,1])
z3dot=np.zeros([t_size,1])

z4=np.zeros([t_size,1])
z4dot=np.zeros([t_size,1])

z5=np.zeros([t_size,1])
z5dot=np.zeros([t_size,1])


out1=np.zeros([t_size,1])
out1dot=np.zeros([t_size,1])
out1doubledot=np.zeros([t_size,1])

sine=np.sin(w*t)
IL1=np.exp(-t)
IL2=np.exp(-t)-np.exp(-2*t)

p=np.zeros([t_size,1])

eps1=np.zeros([t_size,1])

# Standard MRC Law
for i in range(0,t_size-1):


	theta1_new[i]=theta1_0_hat[i]
	theta2_new[i]=theta2_0_hat[i]+theta2_1_hat[i]*sine[i]
	theta3_new[i]=theta3_0_hat[i]+theta3_1_hat[i]*sine[i]
	
	u_new[i]=w1_new[i]+w2_new[i]+theta3_new[i]*yp_new[i]+r[i]


	ymdot_new[i+1]=ymdot_new[i]+(-3*ymdot_new[i]-2*ym_new[i]+r[i])*dt
	ym_new[i+1]=ym_new[i]+(ymdot_new[i])*dt

	ypdot_new[i+1]=ypdot_new[i]+(-a1[i]*ypdot_new[i]-a2[i]*yp_new[i]+u_new[i])*dt
	yp_new[i+1]=yp_new[i]+(ypdot_new[i])*dt
	
	e_new[i+1]=yp_new[i+1]-ym_new[i+1]
	fin_e_new[i+1]=np.log(1+abs(e_new[i+1]))
	
	w1_new[i+1]=w1_new[i]+(-w1_new[i]+theta1_new[i]*u_new[i])*dt
	w2_new[i+1]=w2_new[i]+(-w2_new[i]+theta2_new[i]*yp_new[i])*dt

	#Additional for MRAC
	#Define in and Z 

	in1[i+1]=in1[i]+(-in1[i]+u_new[i])*dt
	in2[i+1]=in2[i]+(-in2[i]+yp_new[i])*dt
	in3[i+1]=yp_new[i+1]
	in4[i+1]=in4[i]+(-in4[i]+(yp_new[i]*sine[i]))*dt
	in5[i+1]=(yp_new[i+1]*sine[i+1])

	z1dot[i+1]=z1dot[i]+(-3*z1dot[i]-2*z1[i]+in1[i])*dt
	z1[i+1]=z1[i]+(z1dot[i])*dt

	z2dot[i+1]=z2dot[i]+(-3*z2dot[i]-2*z2[i]+in2[i])*dt
	z2[i+1]=z2[i]+(z2dot[i])*dt

	z3dot[i+1]=z3dot[i]+(-3*z3dot[i]-2*z3[i]+in3[i])*dt
	z3[i+1]=z3[i]+(z3dot[i])*dt

	z4dot[i+1]=z4dot[i]+(-3*z4dot[i]-2*z4[i]+in4[i])*dt
	z4[i+1]=z4[i]+(z4dot[i])*dt

	z5dot[i+1]=z5dot[i]+(-3*z5dot[i]-2*z5[i]+in5[i])*dt
	z5[i+1]=z5[i]+(z5dot[i])*dt
	
	#THETA_Hat'*Z
	p[i]=(theta1_0_hat[i]*z1[i])+(theta2_0_hat[i]*z2[i])+(theta3_0_hat[i]*z3[i])+(theta2_1_hat[i]*z4[i])+(theta3_1_hat[i]*z5[i])

	#out1
	out1dot[i+1]=out1dot[i]+(-3*out1dot[i]-2*out1[i]+u_new[i])*dt
	out1[i+1]=out1[i]+(out1dot[i])*dt

	#epsilon1
	eps1[i]=yp_new[i]+p[i]-out1[i]

	#||THETA||
	norm_theta=math.sqrt((theta1_0_hat[i]**2)+(theta2_0_hat[i]**2)+(theta2_1_hat[i]**2)+(theta3_0_hat[i]**2)+(theta3_1_hat[i]**2))

	#sigma
	if(norm_theta<M0):
		sigma=0
	elif(norm_theta>=M0 and norm_theta<=(2*M0)):
		sigma=sigma0*((norm_theta/M0)-1)       #???? OR
		# sigma=sigma0*((norm_theta)/(M0-1))		 #????
	elif(norm_theta>(2*M0)):
		sigma=sigma0

	# print("Theta10",theta1_0_hat[i])
	# print("Theta20",theta2_0_hat[i])
	# print("Theta21",theta2_1_hat[i])
	# print("Theta30",theta2_0_hat[i])
	# print("Theta31",theta3_1_hat[i])
	# print("m",m[i])
	# print("eps1",eps1)
	
	#Update THETA
	mval=m[i]**2
	# print(mval)
	theta1_0_hat[i+1]=theta1_0_hat[i]+(((-100)*(eps1[i]*z1[i])/(mval))-(sigma*theta1_0_hat[i]))*dt
	theta2_0_hat[i+1]=theta2_0_hat[i]+(((-100)*(eps1[i]*z2[i])/(mval))-(sigma*theta2_0_hat[i]))*dt
	theta3_0_hat[i+1]=theta3_0_hat[i]+(((-100)*(eps1[i]*z3[i])/(mval))-(sigma*theta3_0_hat[i]))*dt
	theta2_1_hat[i+1]=theta2_1_hat[i]+(((-100)*(eps1[i]*z4[i])/(mval))-(sigma*theta2_1_hat[i]))*dt
	theta3_1_hat[i+1]=theta3_1_hat[i]+(((-100)*(eps1[i]*z5[i])/(mval))-(sigma*theta3_1_hat[i]))*dt

	#Update m
	m[i+1]=m[i]+(-delta0*m[i]+delta1*(abs(u_new[i])+abs(yp_new[i])+1))*dt

	if(math.isnan(mval)):
		print("BLOWN UP",t[i])
		break

# print(ym_st)
# print(u_st[1000])
# print(yp_st[1000])
plt.figure()
plt.plot(t,ym_st,color='red')
plt.plot(t,yp_st,color='green')
plt.legend(["Model","Plant"])
plt.title("Output of Standard MRAC for Example 2 w = "+str(w))
plt.xlabel("Time")
plt.ylabel("Amplitude")
# plt.ylim([-2-min(ym_st),2+max(ym_st)])


plt.figure()
plt.plot(t,ym_new,color='red')
plt.plot(t,yp_new,color='green')
plt.legend(["Model","Plant"])
plt.title("Output of New MRAC for Example 2 w = "+str(w))
plt.xlabel("Time")
plt.ylabel("Amplitude")

if(w==1):
	plt.figure()
	# plt.plot(t,e_st,color='red')
	plt.plot(t,e_new)
	# plt.legend(["Standard MRC","New MRC"])
	plt.ylim(-10,5)
	plt.xlim(0,100)
	plt.title("New MRAC Error for Example 2 w = "+str(w))
	plt.xlabel("Time")
	plt.ylabel("e")

	plt.figure()
	# plt.plot(t,e_st,color='red')
	plt.plot(t,fin_e_st)
	# plt.legend(["Standard MRC","New MRC"])
	# plt.ylim(0,20)
	plt.title("Standard MRAC Error for Example 2 w = "+str(w))
	plt.xlabel("Time")
	plt.ylabel("log(1+|e|)")

if(w==0.1):
	plt.figure()
	plt.plot(t,e_st,color='red')
	plt.plot(t,e_new,color='green')
	plt.legend(["Standard MRAC","New MRAC"])
	plt.ylim(-10,10)
	plt.title("New MRAC Error for Example 2 w = "+str(w))
	plt.xlabel("Time")
	plt.ylabel("e")

plt.show()
