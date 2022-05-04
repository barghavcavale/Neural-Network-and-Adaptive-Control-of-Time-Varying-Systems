import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
def Train():
	n=1
	end=100000
	endval=np.arange(-1*n,end)
	u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)
	
	#print(u)
	f=np.zeros([end+n,1])
	g=np.zeros([end+n,1])
	yp=np.empty((end+n,1),dtype='float')
	yp[0]=0
	
	
	for i in range(n,end+n):
		f[i-1]=yp[i-1]/(1+yp[i-1]**2)
		g[i-1]=u[i-1]**3
		yp[i]=f[i-1]+g[i-1]
	
	

	yphat=np.empty((end+n,1),dtype='float')
	yphat[0]=0


	bias=1
	L1=200
	eta=0.1


	W1_f=np.random.normal(0,0.1,(L1,1+1))
	W1_u=np.random.normal(0,0.1,(L1,1+1))
	#W2=np.random.normal(0,0.1,(L2,L1+1))
	W2_f=np.zeros([1,L1+1])
	W2_u=np.zeros([1,L1+1])

	J_f=0
	J_u=0

	epochs=1

	for a in range(epochs):
		for i in range(n,end+n):
			S_f=yp[i-1]
			x_f=np.array([bias,S_f[0]])
			A1_f=np.matmul(W1_f,x_f)
			A_f=tan_h(A1_f)
			y_f=np.insert(A_f,0,[bias])
			B1_f=np.matmul(W2_f,y_f)
			#B=tan_h(B1)
			B_f=B1_f
			"""
			z=np.insert(A,0,[bias])
			C1=np.matmul(W2,z)
			C=tan_h(C1)
			"""
			e_f=f[i-1]-B_f

			#del2=e*dtan_h(B1)
			del2_f=e_f
			#del2=np.matmul((np.transpose(W3[...,1:])),del3)*dtan_h(B1)
			del1_f=np.matmul((np.transpose(W2_f[...,1:])),del2_f)*dtan_h(A1_f)
			#del2=del2.reshape(len(del2),1)
			del1_f=del1_f.reshape(len(del1_f),1)
			#y=y.reshape(1,len(y))
			x_f=x_f.reshape(1,len(x_f))
			#W3=W3-(eta*del3*np.transpose(z))
			W2_f=W2_f-(eta*del2_f*np.transpose(y_f))
			W1_f=W1_f-(eta*np.matmul(del1_f,x_f))
			J_f=J_f+e_f*e_f





			
			S_u=u[i-1]
			x_u=np.array([bias,S_u[0]])
			A1_u=np.matmul(W1_u,x_u)
			A_u=tan_h(A1_u)
			y_u=np.insert(A_u,0,[bias])
			B1_u=np.matmul(W2_u,y_u)
			#B=tan_h(B1)
			B_u=B1_u
			"""
			z=np.insert(A,0,[bias])
			C1=np.matmul(W2,z)
			C=tan_h(C1)
			"""
			e_u=g[i-1]-B_u
			#del2=e*dtan_h(B1)
			del2_u=e_u
			#del2=np.matmul((np.transpose(W3[...,1:])),del3)*dtan_h(B1)
			del1_u=np.matmul((np.transpose(W2_u[...,1:])),del2_u)*dtan_h(A1_u)
			#del2=del2.reshape(len(del2),1)
			del1_u=del1_u.reshape(len(del1_u),1)
			#y=y.reshape(1,len(y))
			x_u=x_u.reshape(1,len(x_u))
			#W3=W3-(eta*del3*np.transpose(z))
			W2_u=W2_u-(eta*del2_u*np.transpose(y_u))
			W1_u=W1_u-(eta*np.matmul(del1_u,x_u))
			J_u=J_u+e_u*e_u
			yphat[i]=B_f+B_u

	J_f=J_f/(end)
	J_u=J_u/(end)
	print("Training Cost for BPA 1 Layer for f = ",J_f)
	print("Training Cost for BPA 1 Layer for u = ",J_u)
	return(W1_f,W2_f,W1_u,W2_u)


if __name__ == '__main__':
	Train()
