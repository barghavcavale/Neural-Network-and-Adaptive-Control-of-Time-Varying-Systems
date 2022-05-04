import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
import json

def Train():

	n=2
	end=50000
	endval=np.arange(-1*n,end)
	u=-2*np.ones((end+n,1),dtype='float')+4*np.random.rand(end+n,1)
	#print(u)
	f=np.zeros([end+n,1])
	yp=np.empty((end+n,1),dtype='float')
	yp[0]=0
	yp[1]=0
	
	for i in range(2,end+n):
		f[i-1]=yp[i-1]*yp[i-2]*(yp[i-1]+2.5)/(1+yp[i-1]**2 + yp[i-2]**2)
		yp[i]=f[i-1]+u[i-1]

	yphat=np.empty((end+n,1),dtype='float')
	yphat[0]=0
	yphat[1]=0

	bias=1
	L1=20
	L2=10
	eta=0.1

	W1=np.random.normal(0,0.1,(L1,2+1))
	W2=np.random.normal(0,0.1,(L2,L1+1))
	W3=np.zeros([1,L2+1])
	J=0
	epochs=1

	for a in range(epochs):
		for i in range(2,end+n):
			S=yp[i-1]
			T=yp[i-2]
			x=np.array([bias,S[0],T[0]])
			A1=np.matmul(W1,x)
			A=tan_h(A1)
			y=np.insert(A,0,[bias])
			B1=np.matmul(W2,y)
			B=tan_h(B1)
			z=np.insert(B,0,[bias])
			C1=np.matmul(W3,z)
			#C=tan_h(C1)
			C=C1
			yphat[i]=C + u[i-1]
			e=yphat[i]-yp[i]
			#del3=e*dtan_h(C1)
			del3=e
			del2=np.matmul((np.transpose(W3[...,1:])),del3)*dtan_h(B1)
			del1=np.matmul((np.transpose(W2[...,1:])),del2)*dtan_h(A1)
			del2=del2.reshape(len(del2),1)
			del1=del1.reshape(len(del1),1)
			y=y.reshape(1,len(y))
			x=x.reshape(1,len(x))
			W3=W3-(eta*del3*np.transpose(z))
			W2=W2-(eta*np.matmul(del2,y))
			W1=W1-(eta*np.matmul(del1,x))
			J=J+e*e
	
	J=J/(end)
	print("Training Cost for BPA 2 Layer = ",J)
    

	return(W1,W2,W3)


#W1,W2,W3=Train()

if __name__ == "__main__":
	W1,W2,W3=Train()

	weights = [W1.tolist(), W2.tolist(), W3.tolist()]
	#weight_json = json.dumps(weights)
	with open('weights.json', 'w') as outfile:
		json.dump(weights, outfile)

