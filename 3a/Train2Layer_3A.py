import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

def Train():
	n=1
	end=100000
	endval=np.arange(-1*n,end)
	u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)
	
	f=np.zeros([end+n,1])
	g=np.zeros([end+n,1])
	yp=np.empty((end+n,1),dtype='float')
	yp[0]=0
	
	
	
	for i in range(n,end+n):
		f[i-1]=yp[i-1]/(1+yp[i-1]**2)
		g[i-1]=u[i-1]**3
		yp[i]=f[i-1]+g[i-1]
	
	yp_factor = yp.max()
	
	
	fhat=np.zeros([end+n,1])
	ghat=np.zeros([end+n,1])
	yphat=np.empty((end+n,1),dtype='float')
	yphat[0]=0

	bias=1
	L1=20
	L2=10

	eta=0.05


	W1_f=np.random.normal(0,0.1,(L1,1+1))
	W1_g=np.random.normal(0,0.1,(L1,1+1))
	

	W2_f=np.random.normal(0,0.1,(L2,L1+1))
	W2_g=np.random.normal(0,0.1,(L2,L1+1))


	W3_f=np.zeros([1,L2+1])
	W3_g=np.zeros([1,L2+1])

	J_f=0
	J_g=0

	epochs=1
	flag=1
	for a in range(epochs):
		for i in range(n,end+n):
		# for i in range(n,100):

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
			fhat[i-1]=C_f
			
			S_g=u[i-1]
			x_g=np.array([bias,S_g[0]])

			A1_g=np.matmul(W1_g,x_g)
			A_g=tan_h(A1_g)
			y_g=np.insert(A_g,0,[bias])
			B1_g=np.matmul(W2_g,y_g)
			B_g=tan_h(B1_g)
			z_g=np.insert(B_g,0,[bias])
			C1_g=np.matmul(W3_g,z_g)
			C_g=C1_g
			ghat[i-1]=C_g

			# ipdb.set_trace()

			yphat[i]=fhat[i-1]+ghat[i-1]

			e_f=(yp[i]-yphat[i])
			e_g=(yp[i]-yphat[i])


			#del2=e*dtan_h(B1)
			del3_f=e_f
			del2_f=np.matmul((np.transpose(W3_f[...,1:])),del3_f)*dtan_h(B1_f)
			del1_f=np.matmul((np.transpose(W2_f[...,1:])),del2_f)*dtan_h(A1_f)
			del2_f=del2_f.reshape(len(del2_f),1)
			del1_f=del1_f.reshape(len(del1_f),1)
			#y=y.reshape(1,len(y))
			x_f=x_f.reshape(1,len(x_f))
			y_f=y_f.reshape(1,len(y_f))
			W3_f=W3_f+(eta*del3_f*np.transpose(z_f))		
			W2_f=W2_f+(eta*np.matmul(del2_f,y_f))
			W1_f=W1_f+(eta*np.matmul(del1_f,x_f))
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
			W3_g=W3_g+(eta*del3_g*np.transpose(z_g))      
			W2_g=W2_g+(eta*np.matmul(del2_g,y_g))
			W1_g=W1_g+(eta*np.matmul(del1_g,x_g))
			J_g=J_g+e_g*e_g
			if(math.isnan(J_f) and flag):
				print(i)
				flag=0

	J_f=J_f/(end)
	J_g=J_g/(end)
	print("Training Cost for BPA 2 Layer N1 = ",J_f)
	print("Training Cost for BPA 2 Layer N2 = ",J_g)
	print(fhat.max())
	print(ghat.max())
	ustop=2.00
	u=np.arange(-ustop,ustop+0.01,0.01)
	#print(u)
	uval=u/ustop
	#print(uval)
	gval=np.empty([len(uval),1])
	gvalhat=np.empty([len(uval),1])
	for i in range(len(uval)):
		gval[i]=u[i]**3
		S_g=u[i]
		x_g=np.array([bias,S_g])
		A1_g=np.matmul(W1_g,x_g)
		A_g=tan_h(A1_g)
		y_g=np.insert(A_g,0,[bias])
		B1_g=np.matmul(W2_g,y_g)
		B_g=tan_h(B1_g)
		z_g=np.insert(B_g,0,[bias])
		C1_g=np.matmul(W3_g,z_g)
		C_g=C1_g
		#print(C_g);
		gvalhat[i]=C_g
	print(np.sum(np.square(gval-gvalhat)))
	print(np.max(gval))
	print(np.max(gvalhat))
	#plt.figure();
	plt.plot(u,gval,color='red')
	plt.plot(u,gvalhat,color='green')
	plt.savefig("test.png")
	#plt.show()
	#print(gvalhat[0])
	#print(gval[0])

	return(W1_f,W2_f,W3_f,W1_g,W2_g,W3_g)

# W1_f,W2_f,W1_g,W2_g=Train()
if __name__ == '__main__':
	Train()
