import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
def Train():
	n=1;
	end=100000;
	endval=np.arange(-1*n,end);
	u_N1=-1*np.ones((end+n,1),dtype='float')+2*np.random.rand(end+n,1);
	u_N2=-1*np.ones((end+n,1),dtype='float')+2*np.random.rand(end+n,1);
	#print(u);
	f_N1=np.zeros([end+n,1])
	f_N2=np.zeros([end+n,1])

	yp_N1=np.empty((end+n,1),dtype='float');
	yp_N2=np.empty((end+n,1),dtype='float');
	yp_N1[0]=0;
	yp_N2[0]=0;
	
	for i in range(n,end+n):
		f_N1[i-1]=yp_N1[i-1]/(1+yp_N2[i-1]**2)
		f_N2[i-1]=(yp_N1[i-1]*yp_N2[i-1])/(1+yp_N2[i-1]**2)
		yp_N1[i]=f_N1[i-1]+u_N1[i-1];
		yp_N2[i]=f_N2[i-1]+u_N2[i-1];


	yphat_N1=np.empty((end+n,1),dtype='float');
	yphat_N2=np.empty((end+n,1),dtype='float');
	yphat_N1[0]=0;
	yphat_N2[0]=0;

	bias=1;
	L1=20;
	eta=0.1;


	W1_N1=np.random.normal(0,0.1,(L1,2+1));
	W1_N2=np.random.normal(0,0.1,(L1,2+1));
	#W2=np.random.normal(0,0.1,(L2,L1+1));
	W2_N1=np.zeros([1,L1+1]);
	W2_N2=np.zeros([1,L1+1]);

	J_N1=0;
	J_N2=0;

	epochs=1;

	for a in range(epochs):
		for i in range(n,end+n):
			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			x_N1=np.array([bias,S_N1[0],S_N2[0]]);

			A1_N1=np.matmul(W1_N1,x_N1);
			A_N1=tan_h(A1_N1);
			y_N1=np.insert(A_N1,0,[bias]);
			B1_N1=np.matmul(W2_N1,y_N1);
			#B=tan_h(B1);
			B_N1=B1_N1;
			"""
			z=np.insert(A,0,[bias]);
			C1=np.matmul(W2,z);
			C=tan_h(C1);
			"""
			yphat_N1[i]=B_N1+u_N1[i-1];
			e_N1=yphat_N1[i]-yp_N1[i];
			#del2=e*dtan_h(B1);
			del2_N1=e_N1;
			#del2=np.matmul((np.transpose(W3[...,1:])),del3)*dtan_h(B1);
			del1_N1=np.matmul((np.transpose(W2_N1[...,1:])),del2_N1)*dtan_h(A1_N1);
			#del2=del2.reshape(len(del2),1);
			del1_N1=del1_N1.reshape(len(del1_N1),1);
			#y=y.reshape(1,len(y));
			x_N1=x_N1.reshape(1,len(x_N1));
			#W3=W3-(eta*del3*np.transpose(z));
			W2_N1=W2_N1-(eta*del2_N1*np.transpose(y_N1));
			W1_N1=W1_N1-(eta*np.matmul(del1_N1,x_N1));
			J_N1=J_N1+e_N1*e_N1;





			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			x_N2=np.array([bias,S_N1[0],S_N2[0]]);
			A1_N2=np.matmul(W1_N2,x_N2);
			A_N2=tan_h(A1_N2);
			y_N2=np.insert(A_N2,0,[bias]);
			B1_N2=np.matmul(W2_N2,y_N2);
			#B=tan_h(B1);
			B_N2=B1_N2;
			"""
			z=np.insert(A,0,[bias]);
			C1=np.matmul(W2,z);
			C=tan_h(C1);
			"""
			yphat_N2[i]=B_N2+u_N2[i-1];
			e_N2=yphat_N2[i]-yp_N2[i];
			#del2=e*dtan_h(B1);
			del2_N2=e_N2;
			#del2=np.matmul((np.transpose(W3[...,1:])),del3)*dtan_h(B1);
			del1_N2=np.matmul((np.transpose(W2_N2[...,1:])),del2_N2)*dtan_h(A1_N2);
			#del2=del2.reshape(len(del2),1);
			del1_N2=del1_N2.reshape(len(del1_N2),1);
			#y=y.reshape(1,len(y));
			x_N2=x_N2.reshape(1,len(x_N2));
			#W3=W3-(eta*del3*np.transpose(z));
			W2_N2=W2_N2-(eta*del2_N2*np.transpose(y_N2));
			W1_N2=W1_N2-(eta*np.matmul(del1_N2,x_N2));
			J_N2=J_N2+e_N2*e_N2;

	J_N1=J_N1/(end);
	J_N2=J_N2/(end);
	print("Training Cost for BPA 1 Layer N1 = ",J_N1)
	print("Training Cost for BPA 1 Layer N2 = ",J_N2)
	return(W1_N1,W2_N1,W1_N2,W2_N2);

# W1_N1,W2_N1,W1_N2,W2_N2=Train();
if __name__ == '__main__':
	Train()
