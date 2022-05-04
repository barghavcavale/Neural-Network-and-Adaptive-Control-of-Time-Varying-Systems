import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from Train1Layer_1B import Train 
 
def Run():
	W1,W2=Train();
	n=2;
	k=250;
	

	end=500;
	endval=np.arange(-1*n,end);
	
	u=np.empty((end+n,1),dtype='float');
	f=np.empty((end+n,1),dtype='float');

	yp=np.empty((end+n,1),dtype='float');
	yp[0]=0;
	yp[1]=0;
	

	yphat=np.empty((end+n,1),dtype='float');
	yphat[0]=0;
	yphat[1]=0;
	J=0;
	bias=1;
	epochs=1;

	for a in range(epochs):
		for i in range(2,end+n):
			if(i<k):
				u[i]=np.sin(2*np.pi*(i-n)/250)
			else:	
				u[i]=np.sin(2*np.pi*(i-n)/250)+np.sin(2*np.pi*(i-n)/25);
				u[i]=u[i]/2;
			f[i]=np.power(u[i],3)+0.3*np.power(u[i],2)-0.4*u[i];
			yp[i]=0.3*yp[i-1]+0.6*yp[i-2]+f[i-1];
			S=u[i-1];
			x=np.array([bias,S[0]]);
			A1=np.matmul(W1,x);
			A=tan_h(A1);
			y=np.insert(A,0,[bias]);
			B1=np.matmul(W2,y);
			#B=tan_h(B1);
			B=(B1);
			"""
			z=np.insert(B,0,[bias]);
			C1=np.matmul(W3,z);
			C=tan_h(C1);
			"""
			yphat[i]=0.3*yphat[i-1]+0.6*yphat[i-2]+B;
			e=(yp[i]-yphat[i]);	
			J=J+e*e;
	J=J/(end);

	
	pickle_out=open("BPA1Layer_1B_yphat.pickle","wb");
	pickle.dump(yphat,pickle_out);
	pickle_out.close();
	pickle_out=open("BPA1Layer_1B_J.pickle","wb");
	pickle.dump(J,pickle_out);
	pickle_out.close();
	"""
	
	print("Testing Cost = ", J);
	plt.plot(endval,yp,color='g');
	plt.plot(endval,yphat,color='r');
	plt.xlabel("Time");
	plt.ylabel("Val");
	plt.title("1 B");
	plt.show();
	"""
#Run();


	

