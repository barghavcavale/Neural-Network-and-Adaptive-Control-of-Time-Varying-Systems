import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from OS11A import Train as OSLATrain

def f(x):
	return x/(1+x**2)

def g(x):
	return x**3

def Train():
	n=1
	end=100000
	endval=np.arange(-1*n,end)
	
	#u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)
	
	r = np.sin(2*np.pi*endval/25) + np.sin(2*np.pi*endval/10)

	yp=np.empty((end+n,1),dtype='float')
	yp[0]=0
	
	ym=np.empty((end+n,1),dtype='float')
	ym[0]=0
	yphat=np.empty((end+n,1),dtype='float')
	yphat[0]=0

	bias=1
	L1=20
	L2=10

	eta1=0.05
	eta2=0.05
	eta3=0.01

	W1_f=np.random.normal(0,0.1,(L1,1+1))
	W1_g=np.random.normal(0,0.1,(L1,1+1))
	W1_h=np.random.normal(0,0.1,(L1,1+1))
	#W2=np.random.normal(0,0.1,(L2,L1+1))

	W2_f=np.random.normal(0,0.1,(L2,L1+1))
	W2_g=np.random.normal(0,0.1,(L2,L1+1))
	W2_h=np.random.normal(0,0.1,(L2,L1+1))


	W3_f=np.zeros([1,L2+1])
	W3_g=np.zeros([1,L2+1])
	W3_h=np.zeros([1,L2+1])

	J_f=0
	J_g=0
	J_h=0

	epochs=1
	flag=1
	for a in range(epochs):
		for i in range(n,end+n):
		# for i in range(n,100):
			
			ym[i] = 0.6*ym[i-1] + r[i-1]

			S_f=yp[i-1]
		
			x_f=np.array([bias,S_f[0]])

			A1_f=np.matmul(W1_f,x_f)
			A_f=tan_h(A1_f)
			y_f=np.insert(A_f,0,[bias])
			B1_f=np.matmul(W2_f,y_f)
			B_f=tan_h(B1_f)
			z_f=np.insert(B_f,0,[bias])
			C1_f=np.matmul(W3_f,z_f)
			C_f=C1_f
			

			S_h = -C_f + 0.6*yp[i-1] + r[i-1]
			
			x_h = np.array([bias, S_h[0]])
			
			A1_h=np.matmul(W1_h,x_h)
			A_h=tan_h(A1_h)
			y_h=np.insert(A_h,0,[bias])
			B1_h=np.matmul(W2_h,y_h)
			B_h=tan_h(B1_h)
			z_h=np.insert(B_h,0,[bias])
			C1_h=np.matmul(W3_h,z_h)
			C_h=C1_h


			S_g=C_h
			x_g=np.array([bias,S_g[0]])

			A1_g=np.matmul(W1_g,x_g)
			A_g=tan_h(A1_g)
			y_g=np.insert(A_g,0,[bias])
			B1_g=np.matmul(W2_g,y_g)
			B_g=tan_h(B1_g)
			z_g=np.insert(B_g,0,[bias])
			C1_g=np.matmul(W3_g,z_g)
			C_g=C1_g
			
			yp[i] = f(yp[i-1]) + g(C_h)
			
			# ipdb.set_trace()

			yphat[i]=C_f + C_g

			e_f=(yp[i]-yphat[i])
			e_g=(yp[i]-yphat[i])
			
			# e_h=-(C_g-(-C_f + 0.6*yp[i-1]+r[i-1]))
			e_h=-(C_g-(r[i-1]))
			#del2=e*dtan_h(B1)
			del3_f=e_f
			del2_f=np.matmul((np.transpose(W3_f[...,1:])),del3_f)*dtan_h(B1_f)
			del1_f=np.matmul((np.transpose(W2_f[...,1:])),del2_f)*dtan_h(A1_f)
			del2_f=del2_f.reshape(len(del2_f),1)
			del1_f=del1_f.reshape(len(del1_f),1)
			#y=y.reshape(1,len(y))
			x_f=x_f.reshape(1,len(x_f))
			y_f=y_f.reshape(1,len(y_f))
			W3_f=W3_f+(eta1*del3_f*np.transpose(z_f))		
			W2_f=W2_f+(eta1*np.matmul(del2_f,y_f))
			W1_f=W1_f+(eta1*np.matmul(del1_f,x_f))
			J_f=J_f+e_f*e_f

			
			#del2=e*dtan_h(B1)
			del3_g=e_g
			del2_g=np.matmul((np.transpose(W3_g[...,1:])),del3_g)*dtan_h(B1_g)
			del1_g=np.matmul((np.transpose(W2_g[...,1:])),del2_g)*dtan_h(A1_g)
			del2_g=del2_g.reshape(len(del2_g),1)
			del1_g=del1_g.reshape(len(del1_g),1)
			#y=y.reshape(1,len(y))
			x_g=x_g.reshape(1,len(x_g))
			y_g=y_g.reshape(1,len(y_g))
			W3_g=W3_g+(eta2*del3_g*np.transpose(z_g))      
			W2_g=W2_g+(eta2*np.matmul(del2_g,y_g))
			W1_g=W1_g+(eta2*np.matmul(del1_g,x_g))
			J_g=J_g+e_g*e_g

			del3_h=e_h
			del2_h=np.matmul((np.transpose(W3_h[...,1:])),del3_h)*dtan_h(B1_h)
			del1_h=np.matmul((np.transpose(W2_h[...,1:])),del2_h)*dtan_h(A1_h)
			del2_h=del2_h.reshape(len(del2_h),1)
			del1_h=del1_h.reshape(len(del1_h),1)
			#y=y.reshape(1,len(y))

			inp=np.array([bias,r[i-1]])
			inp=x_h.reshape(1,len(inp))
			x_h=x_h.reshape(1,len(x_h))
			y_h=y_h.reshape(1,len(y_h))
			W3_h=W3_h+(eta3*del3_h*np.transpose(z_h))      
			W2_h=W2_h+(eta3*np.matmul(del2_h,y_h))
			W1_h=W1_h+(eta3*np.matmul(del1_h,inp))
			J_h=J_h+e_h*e_h

			if(math.isnan(J_f) and flag):
				print(i)
				flag=0

	J_f=J_f/(end)
	J_g=J_g/(end)
	J_h=J_h/(end)
	print("Training Cost for BPA 2 Layer N1 = ",J_f)
	print("Training Cost for BPA 2 Layer N2 = ",J_g)
	print("Training Cost for BPA 2 Layer N3 = ",J_h)

	print(np.max(yphat))
	lims=90000
	yp_osla = OSLATrain()

	plt.figure()
	plt.plot(endval,ym,color='red')
	plt.plot(endval,yp*1.6+4,color='green')
	plt.plot(endval,yp_osla,color='black')
	plt.legend(["Model","Plant BPA","Plant OSLA"])
	plt.title('Example 11')
	plt.xlim([lims,lims+100])
	plt.ylim([-8,8])
	plt.savefig("Ex11.png")


	# ustop=2.00
	# u=np.arange(-ustop,ustop+0.01,0.01)
	# #print(u)
	# uval=u/ustop
	# #print(uval)
	# gval=np.empty([len(uval),1])
	# gvalhat=np.empty([len(uval),1])
	# for i in range(len(uval)):
	# 	gval[i]=u[i]**3
	# 	S_g=u[i]
	# 	x_g=np.array([bias,S_g])
	# 	A1_g=np.matmul(W1_g,x_g)
	# 	A_g=tan_h(A1_g)
	# 	y_g=np.insert(A_g,0,[bias])
	# 	B1_g=np.matmul(W2_g,y_g)
	# 	B_g=tan_h(B1_g)
	# 	z_g=np.insert(B_g,0,[bias])
	# 	C1_g=np.matmul(W3_g,z_g)
	# 	C_g=C1_g
	# 	#print(C_g);
	# 	gvalhat[i]=C_g
	# print(np.sum(np.square(gval-gvalhat)))
	# print(np.max(gval))
	# print(np.max(gvalhat))
	# #plt.figure();
	# plt.plot(u,gval,color='red')
	# plt.plot(u,gvalhat,color='green')
	# plt.savefig("test.png")
	# #plt.show()
	# #print(gvalhat[0])
	# #print(gval[0])

	# return(W1_f,W2_f,W3_f,W1_g,W2_g,W3_g)

# W1_f,W2_f,W1_g,W2_g=Train()
if __name__ == '__main__':
	Train()
