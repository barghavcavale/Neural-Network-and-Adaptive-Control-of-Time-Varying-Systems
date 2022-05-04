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
	L1=18
	#L2=10
	lam=0.000000001

	W1=np.random.normal(0,0.1,(L1,2+1))
	#W2=np.random.normal(0,0.1,(L2,L1+1))
	W2=np.zeros([1,L1+1])
	P0=(1/lam)*np.identity(L1+1)
	J=0

	epochs=1

	for a in range(epochs):
		for i in range(2,end+n):
			S=yp[i-1]
			T=yp[i-2]
			v0k=np.array([bias,S[0],T[0]])
			v1k_bar=tan_h(np.matmul(W1,v0k))
			v1k=np.insert(v1k_bar,0,[bias])
			fin=np.matmul(W2,v1k)
			yphat[i]=fin+u[i-1]
			e=yp[i]-yphat[i]
			v1k=v1k.reshape(len(v1k),1)
			v1k_T=v1k.reshape(1,len(v1k))
			Num=np.matmul(P0,np.matmul(np.matmul(v1k,v1k_T),P0))	
			Den=1+np.matmul(v1k_T,np.matmul(P0,v1k))
			P0=P0-(Num/Den)
			W2=W2+e*np.matmul(v1k_T,P0)
			J=J+(e*e)
	print("Training Cost for OSLA = ",J/end)
	return(W1,W2)
#W1,W2=Train()

if __name__ == "__main__":
	W1,W2=Train()

	weights = [W1.tolist(), W2.tolist()]
	#weight_json = json.dumps(weights)
	with open('weights_osla.json', 'w') as outfile:
		json.dump(weights, outfile)

