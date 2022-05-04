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
	
	#print(u)


	yp_N1=np.empty((end+n,1),dtype='float')
	yp_N2=np.empty((end+n,1),dtype='float')
	yp_N1[0]=0
	yp_N2[0]=0
	
	for i in range(n,end+n):
		f_N1[i-1]=yp_N1[i-1]/(1+yp_N2[i-1]**2)
		f_N2[i-1]=(yp_N1[i-1]*yp_N2[i-1])/(1+yp_N2[i-1]**2)
		yp_N1[i]=f_N1[i-1]+u_N1[i-1]
		yp_N2[i]=f_N2[i-1]+u_N2[i-1]


	yphat_N1=np.empty((end+n,1),dtype='float')
	yphat_N2=np.empty((end+n,1),dtype='float')
	yphat_N1[0]=0
	yphat_N2[0]=0
	bias=1
	L1=18
	#L2=10
	lam_N1=0.0000001
	lam_N2=0.001
	W1_N1=np.random.normal(0,0.1,(L1,2+1))
	W2_N1=np.zeros([1,L1+1])
	W1_N2=np.random.normal(0,0.1,(L1,2+1))
	W2_N2=np.zeros([1,L1+1])

	P0_N1=(1/lam_N1)*np.identity(L1+1)
	P0_N2=(1/lam_N2)*np.identity(L1+1)
	J_N1=0
	J_N2=0
	epochs=1

	for a in range(epochs):
		for i in range(2,end+n):
			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			v0k_N1=np.array([bias,S_N1[0],S_N2[0]])
			v1k_bar_N1=tan_h(np.matmul(W1_N1,v0k_N1))
			v1k_N1=np.insert(v1k_bar_N1,0,[bias])
			fin_N1=np.matmul(W2_N1,v1k_N1)
			yphat_N1[i]=fin_N1+u_N1[i-1]
			e_N1=yp_N1[i]-yphat_N1[i]
			v1k_N1=v1k_N1.reshape(len(v1k_N1),1)
			v1k_T_N1=v1k_N1.reshape(1,len(v1k_N1))
			Num_N1=np.matmul(P0_N1,np.matmul(np.matmul(v1k_N1,v1k_T_N1),P0_N1))	
			Den_N1=1+np.matmul(v1k_T_N1,np.matmul(P0_N1,v1k_N1))
			P0_N1=P0_N1-(Num_N1/Den_N1)
			W2_N1=W2_N1+e_N1*np.matmul(v1k_T_N1,P0_N1)
			J_N1=J_N1+(e_N1*e_N1)


			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			v0k_N2=np.array([bias,S_N1[0],S_N2[0]])
			v1k_bar_N2=tan_h(np.matmul(W1_N2,v0k_N2))
			v1k_N2=np.insert(v1k_bar_N2,0,[bias])
			fin_N2=np.matmul(W2_N2,v1k_N2)
			yphat_N2[i]=fin_N2+u_N2[i-1]
			e_N2=yp_N2[i]-yphat_N2[i]
			v1k_N2=v1k_N2.reshape(len(v1k_N2),1)
			v1k_T_N2=v1k_N2.reshape(1,len(v1k_N2))
			Num_N2=np.matmul(P0_N2,np.matmul(np.matmul(v1k_N2,v1k_T_N2),P0_N2)) 
			Den_N2=1+np.matmul(v1k_T_N2,np.matmul(P0_N2,v1k_N2))
			P0_N2=P0_N2-(Num_N2/Den_N2)
			W2_N2=W2_N2+e_N2*np.matmul(v1k_T_N2,P0_N2)
			J_N2=J_N2+(e_N2*e_N2)

	print("Training Cost for OSLA N1 = ",J_N1/end)
	print("Training Cost for OSLA N2 = ",J_N2/end)
	return(W1_N1,W2_N1,W1_N2,W2_N2)

if __name__ == '__main__':
	Train()
