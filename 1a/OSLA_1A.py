import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt


def Run():

	k=500;
	n=2;
	kval=np.arange(-1*n,k);
	"""
	kval1=np.arange(k);
	kval2=np.arange(1,k+1);
    """
	end=1000;
	endval=np.arange(-1*n,end);

	u=np.sin(2*np.pi*endval/250);
	f=0.6*np.sin(np.pi*u)+0.3*np.sin(3*np.pi*u)+0.1*np.sin(5*np.pi*u);
	yp=np.empty((end+n,1),dtype='float');
	yp[0]=0;
	yp[1]=0;
	for i in range(n,end):
		yp[i]=0.3*yp[i-1]+0.6*yp[i-2]+f[i-1];

	yphat=np.empty((end+n,1),dtype='float');
	yphat[0]=0;
	yphat[1]=0;
	bias=1;
	#Initialization
	L1=18;      #Change
	lam=0.00000001;    #Change

	W1=np.random.normal(0,0.1,(L1,1+1));  #Random
	W2=np.zeros((1,L1+1));     #Zero Matrix
	P0=(1/lam)*np.identity(L1+1);
	J=0;
	
	epochs=1;
	for a in range(epochs):
		J=0;
		for i in range(2,end+n):
			if(i<(k+n)):
				S=u[i-1];
				v0k=np.array([bias,S]);
				v1k_bar=tan_h(np.matmul(W1,v0k));
				v1k=np.insert(v1k_bar,0,[bias]);
				fin=np.matmul(W2,v1k);
				yphat[i]=0.3*yp[i-1]+0.6*yp[i-2]+fin;
				e=yp[i]-yphat[i];
				v1k=v1k.reshape(len(v1k),1);
				v1k_T=v1k.reshape(1,len(v1k));
				Num=np.matmul(P0,np.matmul(np.matmul(v1k,v1k_T),P0));	
				Den=1+np.matmul(v1k_T,np.matmul(P0,v1k));
				P0=P0-(Num/Den);
				W2=W2+e*np.matmul(v1k_T,P0);
				J=J+(e*e);
			else:
				S=u[i-1];
				v0k=np.array([bias,S]);
				v1k_bar=tan_h(np.matmul(W1,v0k));
				v1k=np.insert(v1k_bar,0,[bias]);
				fin=np.matmul(W2,v1k);
				yphat[i]=0.3*yphat[i-1]+0.6*yphat[i-2]+fin;
				e=yp[i]-yphat[i];
	J=J/(k);
	
	pickle_out=open("OSLA_1A_yphat.pickle","wb");
	pickle.dump(yphat,pickle_out);
	pickle_out.close();
	pickle_out=open("OSLA_1A_J.pickle","wb");
	pickle.dump(J,pickle_out);
	pickle_out.close();
	
	print("Cost = ", J);
	plt.plot(endval,yp,color='r');
	plt.plot(endval,yphat,color='g');
	plt.legend(["Plant","OSLA"]);
	plt.xlabel("Time");
	plt.ylabel("Val");
	plt.title("1 A");
	plt.show();

if __name__ == '__main__':	
	Run();