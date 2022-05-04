import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from Train1Layer_6A import Train 
 
def Run():
	W1,W2=Train();
	n=1;

	end=200;
	endval=np.arange(-1*n,end);
	

	u=np.empty((end+n,1),dtype='float');
	f=np.empty((end+n,1),dtype='float');

	yp=np.empty((end+n,1),dtype='float');
	yp[0]=0;
	

	yphat=np.empty((end+n,1),dtype='float');
	yphat[0]=0;
	
	J=0;
	bias=1;
	epochs=1;

	for a in range(epochs):
		for i in range(n,end+n):
			u[i-1]=np.sin(2*np.pi*(i-n)/25);
			f[i-1]=u[i-1]*(u[i-1]+0.5)*(u[i-1]-0.8);
			yp[i]=0.8*yp[i-1]+f[i-1];
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
			yphat[i]=0.8*yphat[i-1]+B;
			e=(yp[i]-yphat[i]);	
			J=J+e*e;
	J=J/(end);

	
	pickle_out=open("BPA1Layer_6A_yphat.pickle","wb");
	pickle.dump(yphat,pickle_out);
	pickle_out.close();
	pickle_out=open("BPA1Layer_6A_J.pickle","wb");
	pickle.dump(J,pickle_out);
	pickle_out.close();
	
	
	print("Testing Cost = ", J);
	plt.figure();
	plt.plot(endval,yp,color='g');
	plt.plot(endval,yphat,color='r');
	plt.xlabel("Time");
	plt.ylabel("Val");
	plt.title("6 A Identification");

	t_t = np.arange(-1,1,0.01)
	f_t = np.zeros([len(t_t)])
	N_t = np.zeros([len(t_t)])


	J_U=0;
	for i in range(0,len(t_t)):
		f_t[i]=(t_t[i]-0.8)*t_t[i]*(t_t[i]+0.5)
		S=t_t[i];
		x=np.array([bias,S]);
		A1=np.matmul(W1,x);
		A=tan_h(A1);
		y=np.insert(A,0,[bias]);
		B1=np.matmul(W2,y);
		#B=tan_h(B1);
		N_t[i]=(B1);
		e_u=f_t[i]-N_t[i];
		J_U+=e_u*e_u;
	J_U/=len(t_t);

	pickle_out=open("BPA1Layer_6A_J_U.pickle","wb");
	pickle.dump(J_U,pickle_out);
	pickle_out.close();
	pickle_out=open("BPA1Layer_6A_NT.pickle","wb");
	pickle.dump(N_t,pickle_out);
	pickle_out.close();


	print("Function Cost = ",J_U);
	plt.figure();
	plt.plot(t_t,f_t,color='g');
	plt.plot(t_t,N_t,color='r');
	plt.xlabel("Time");
	plt.ylabel("Val");
	plt.legend(["F","N(u)"]);
	plt.title("6 A Function");



	plt.show();

if __name__ == '__main__':
	Run();


	

