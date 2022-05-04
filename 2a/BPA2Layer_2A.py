import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from Train2Layer_2A import Train 
 
def Run():
	W1,W2,W3=Train();
	n=2;

	end=100;
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
			u[i-1]=np.sin(2*np.pi*(i-n)/25)

			f[i-1]=yp[i-1]*yp[i-2]*(yp[i-1]+2.5)/(1+yp[i-1]**2 + yp[i-2]**2)
			
			yp[i]=f[i-1]+u[i-1]
			S=yp[i-1]
			T=yp[i-2];
			x=np.array([bias,S[0],T[0]]);
			A1=np.matmul(W1,x);
			A=tan_h(A1);
			y=np.insert(A,0,[bias]);
			B1=np.matmul(W2,y);
			B=tan_h(B1);
			z=np.insert(B,0,[bias]);
			C1=np.matmul(W3,z);
			#C=tan_h(C1);
			C=C1;
			yphat[i]=C + u[i-1];
			e=(yp[i]-yphat[i]);	
			J=J+e*e;
	J=J/(end);

	
	pickle_out=open("BPA2Layer_2A_yphat.pickle","wb");
	pickle.dump(yphat,pickle_out);
	pickle_out.close();
	pickle_out=open("BPA2Layer_2A_J.pickle","wb");
	pickle.dump(J,pickle_out);
	pickle_out.close();
	
	
# 	print("Testing Cost = ", J);
# 	plt.plot(endval,yp,color='g');
# 	plt.plot(endval,yphat,color='r');
# 	plt.xlabel("Time");
# 	plt.ylabel("Val");
# 	plt.title("2 A");
# 	plt.show();

# Run();

	

