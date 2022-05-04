import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from activations import linear
from activations import dlinear
from matplotlib import pyplot as plt
def Run(mu):
	n=2
	k=500

	end=1000
	endval=np.arange(-1*n,end)
	
	
	u=np.sin(2*np.pi*endval/250)
	f=0.6*np.sin(mu*endval)*np.sin(np.pi*u)+0.3*np.sin(3*np.pi*u)+0.1*np.sin(5*np.pi*u)

	yp=np.empty((end+n,1),dtype='float')
	yp[0]=0
	yp[1]=0
	
	for i in range(2,end+n):
		yp[i]=0.3*yp[i-1]+0.6*yp[i-2]+f[i-1]

	yphat=np.empty((end+n,1),dtype='float')
	yphat[0]=0
	yphat[1]=0

	bias=1
	L1=20
	L2=10
	eta=0.25

	W1=np.random.normal(0,0.1,(L1,1+1))
	W2=np.random.normal(0,0.1,(L2,L1+1))
	W3=np.zeros([1,L2+1])
	J=0
	epochs=1

	for a in range(epochs):
		for i in range(2,end):
			if(i<(k+n)):
				S=u[i-1]
				x=np.array([bias,S])
				A1=np.matmul(W1,x)
				A=tan_h(A1)
				y=np.insert(A,0,[bias])
				B1=np.matmul(W2,y)
				B=tan_h(B1)
				z=np.insert(B,0,[bias])
				C1=np.matmul(W3,z)
				# C=tan_h(C1)
				C=linear(C1)
				yphat[i]=0.3*yp[i-1]+0.6*yp[i-2]+C
				e=yphat[i]-yp[i]
				# del3=e*dtan_h(C1)
				del3 = e*dlinear(C1)
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
			else:
				S=u[i]
				x=np.array([bias,S])
				A1=np.matmul(W1,x)
				A=tan_h(A1)
				y=np.insert(A,0,[bias])
				B1=np.matmul(W2,y)
				B=tan_h(B1)
				z=np.insert(B,0,[bias])
				C1=np.matmul(W3,z)
				# C=tan_h(C1)
				C= linear(C1)
				yphat[i]=0.3*yphat[i-1]+0.6*yphat[i-2]+C	

	J=J/(k)
	
	pickle_out=open("BPA2Layer_yphat.pickle","wb")
	pickle.dump(yphat,pickle_out)
	pickle_out.close()
	pickle_out=open("BPA2Layer_J.pickle","wb")
	pickle.dump(J,pickle_out)
	pickle_out.close()
	
	# print("Cost = ", J)
	# plt.plot(endval,yp,color='g')
	# plt.plot(endval,yphat,color='r')
	# plt.xlabel("Time")
	# plt.ylabel("Val")
	# plt.title("1 A Time-Varying 2 Layer BPA Mu = "+str(mu))
	# plt.show()
if __name__ == '__main__':
	mu=0.1
	Run(mu)




	

