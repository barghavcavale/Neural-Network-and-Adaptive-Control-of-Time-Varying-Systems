import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from Train2Layer_4A import Train 
 
def Run():
	W1,W2,W3=Train();
	n=3;

	k=500
	end=800;
	endval=np.arange(-1*n,end);
	
	u=np.empty((end+n,1),dtype='float');
	f=np.empty((end+n,1),dtype='float');

	for i in range(0,end+n):
		if(i<(k+n)):
			u[i]=np.sin(2*np.pi*i/250)
		else:
			u[i]=0.8*np.sin(2*np.pi*i/250) + 0.2*np.sin(2*np.pi*i/25)


	yp=np.empty((end+n,1),dtype='float');
	yp[0]=0;
	yp[1]=0;
	yp[2]=0;

	yphat=np.empty((end+n,1),dtype='float');
	yphat[0]=0;
	yphat[1]=0;
	yphat[2]=0;
	J=0;
	bias=1;
	epochs=1;

	for a in range(epochs):
		for i in range(n,end+n):
			f[i-1]=(yp[i-1]*yp[i-2]*yp[i-3]*u[i-2]*(yp[i-3]-1)+u[i-1])/(1+yp[i-3]**2 + yp[i-2]**2)
			
			yp[i]=f[i-1]
			S=yp[i-1]
			T=yp[i-2]
			U=yp[i-2]
			V=u[i-1]
			W=u[i-2]
			x=np.array([bias,S[0],T[0],U[0],V[0],W[0]]);

			A1=np.matmul(W1,x);
			A=tan_h(A1);
			y=np.insert(A,0,[bias]);
			B1=np.matmul(W2,y);
			B=tan_h(B1);
			z=np.insert(B,0,[bias]);
			C1=np.matmul(W3,z);
			#C=tan_h(C1);
			C=C1;
			yphat[i]=C;
			e=(yp[i]-yphat[i]);	
			J=J+e*e;
	J=J/(end);

	
	pickle_out=open("BPA2Layer_4A_yphat.pickle","wb");
	pickle.dump(yphat,pickle_out);
	pickle_out.close();
	pickle_out=open("BPA2Layer_4A_J.pickle","wb");
	pickle.dump(J,pickle_out);
	pickle_out.close();
	
	
# 	print("Testing Cost = ", J);
# 	plt.plot(endval,yp,color='g');
# 	plt.plot(endval,yphat,color='r');
# 	plt.xlabel("Time");
# 	plt.ylabel("Val");
# 	plt.title("4 A");
# 	plt.show();

# Run();

	

