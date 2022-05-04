import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from TrainOSLA_7A import Train

def Run():
	W1,W2=Train()
	n=2

	end=100
	endval=np.arange(-1*n,end)
	
	u=np.empty((end+n,1),dtype='float')
	f=np.empty((end+n,1),dtype='float')

	yp=np.empty((end+n,1),dtype='float')
	yp[0]=0
	yp[1]=0
	

	yphat=np.empty((end+n,1),dtype='float')
	yphat[0]=0
	yphat[1]=0
	J=0
	bias=1
	epochs=1

	for a in range(epochs):
		for i in range(2,end+n):

			u[i-1]=np.sin(2*np.pi*(i-n)/25)

			f[i-1]=yp[i-1]*yp[i-2]*(yp[i-1]+2.5)/(1+yp[i-1]**2 + yp[i-2]**2)
			
			yp[i]=f[i-1]+u[i-1]
			S=yp[i-1]
			T=yp[i-2]

			v0k=np.array([bias,S[0],T[0]])
			v1k_bar=tan_h(np.matmul(W1,v0k))
			v1k=np.insert(v1k_bar,0,[bias])
			fin=np.matmul(W2,v1k)
			yphat[i]=fin+u[i-1]
			e=yp[i]-yphat[i]
			J=J+(e*e)	
	J=J/(end)
	
	pickle_out=open("OSLA_7A_yphat.pickle","wb")
	pickle.dump(yphat,pickle_out)
	pickle_out.close()
	pickle_out=open("OSLA_7A_J.pickle","wb")
	pickle.dump(J,pickle_out)
	pickle_out.close()

	# print("Testing Cost = ", J)
	# plt.plot(endval,yp,color='r')
	# plt.plot(endval,yphat,color='g')
	# plt.legend(["Plant","OSLA"])
	# plt.xlabel("Time")
	# plt.ylabel("Val")
	# plt.title("2 A")
	# plt.show()
	
	
if __name__ == "__main__":
	Run()