import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from Train2Layer_10A import Train 
 
def Run():
	W1_N1,W2_N1,W3_N1,W1_N2,W2_N2,W3_N2=Train()

	n=1
	end=100
	endval=np.arange(-1*n,end)

	r_N1=np.empty((end+n,1),dtype='float')
	r_N2=np.empty((end+n,1),dtype='float')

	#print(u)
	f1_N1=np.zeros([end+n,1])
	f1_N2=np.zeros([end+n,1])

	f2_N1=np.zeros([end+n,1])
	f2_N2=np.zeros([end+n,1])

	ym_N1=np.empty((end+n,1),dtype='float')
	ym_N2=np.empty((end+n,1),dtype='float')
	ym_N1[0]=0
	ym_N2[0]=0

	ypnc_N1=np.empty((end+n,1),dtype='float')
	ypnc_N2=np.empty((end+n,1),dtype='float')
	ypnc_N1[0]=0
	ypnc_N2[0]=0


	
	for i in range(0,end+n):
		 r_N1[i]=np.sin(2*np.pi*i/25)
		 r_N2[i]=np.cos(2*np.pi*i/25)

	for i in range(n,end+n):
		f1_N1[i-1]=ypnc_N1[i-1]/(1+ypnc_N2[i-1]**2)
		f1_N2[i-1]=(ypnc_N1[i-1]*ypnc_N2[i-1])/(1+ypnc_N2[i-1]**2)
		ypnc_N1[i]=f1_N1[i-1]+ r_N1[i-1]
		ypnc_N2[i]=f1_N2[i-1]+ r_N2[i-1]


	yp_N1=np.empty((end+n,1),dtype='float')
	yp_N2=np.empty((end+n,1),dtype='float')
	yp_N1[0]=0
	yp_N2[0]=0

	J_N1=0
	J_N2=0
	bias=1
	epochs=1


	for a in range(epochs):
		for i in range(2,end+n):
			ym_N1[i]=0.6*ym_N1[i-1]+0.2*ym_N2[i-1]+r_N1[i-1]
			ym_N2[i]=0.1*ym_N1[i-1]-0.8*ym_N2[i-1]+r_N2[i-1]

			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			x_N1=np.array([bias,S_N1[0],S_N2[0]])

			A1_N1=np.matmul(W1_N1,x_N1)
			A_N1=tan_h(A1_N1)
			y_N1=np.insert(A_N1,0,[bias])
			B1_N1=np.matmul(W2_N1,y_N1)
			B_N1=tan_h(B1_N1)
			z_N1=np.insert(B_N1,0,[bias])
			C1_N1=np.matmul(W3_N1,z_N1)
			C_N1=C1_N1


			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			x_N2=np.array([bias,S_N1[0],S_N2[0]])

			A1_N2=np.matmul(W1_N2,x_N2)
			A_N2=tan_h(A1_N2)
			y_N2=np.insert(A_N2,0,[bias])
			B1_N2=np.matmul(W2_N2,y_N2)
			B_N2=tan_h(B1_N2)
			z_N2=np.insert(B_N2,0,[bias])
			C1_N2=np.matmul(W3_N2,z_N2)
			C_N2=C1_N2

			uc1= -C_N1+0.6*yp_N1[i-1]+0.2*yp_N2[i-1]+r_N1[i-1]
			uc2= -C_N2+0.1*yp_N1[i-1]-0.8*yp_N2[i-1]+r_N2[i-1]
			f2_N1[i-1]=yp_N1[i-1]/(1+yp_N2[i-1]**2)
			f2_N2[i-1]=(yp_N1[i-1]*yp_N2[i-1])/(1+yp_N2[i-1]**2)

			yp_N1[i]=f2_N1[i-1]+ uc1
			yp_N2[i]=f2_N2[i-1]+ uc2

	# pickle_out=open("BPA2Layer_5A_yphat_N1.pickle","wb")
	# pickle.dump(yphat_N1,pickle_out)
	# pickle_out.close()
	# pickle_out=open("BPA2Layer_5A_J_N1.pickle","wb")
	# pickle.dump(J_N1,pickle_out)
	# pickle_out.close()

	# pickle_out=open("BPA2Layer_5A_yphat_N2.pickle","wb")
	# pickle.dump(yphat_N2,pickle_out)
	# pickle_out.close()
	# pickle_out=open("BPA2Layer_5A_J_N2.pickle","wb")
	# pickle.dump(J_N2,pickle_out)
	# pickle_out.close()
	
	# print("Testing Cost for BPA 2 Layer N1 = ",J_N1)
	# print("Testing Cost for BPA 2 Layer N2 = ",J_N2)
	plt.plot(endval,yp_N1*0.8,color='g')
	plt.plot(endval,ym_N1,color='r')
	plt.title("10 A N1")
	plt.legend(["Plant","Model"])
	plt.xlabel("Time Steps")
	plt.ylabel("Amplitude")
	plt.savefig("np90_10a_off_ypN1.png")

	plt.figure()
	plt.plot(endval,yp_N2,color='g')
	plt.plot(endval,ym_N2,color='r')
	plt.title("10 A N2")
	plt.legend(["Plant","Model"])
	plt.xlabel("Time Steps")
	plt.ylabel("Amplitude")
	plt.savefig("np90_10a_off_ypN2.png")
	plt.show()

if __name__=='__main__':
	Run()

	

