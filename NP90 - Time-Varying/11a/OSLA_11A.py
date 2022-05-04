import numpy as np
import pandas as pd
import pickle
from activations import tan_h
from activations import dtan_h
from matplotlib import pyplot as plt
from TrainOSLA_3A import Train

def Run():
	W1_N1,W2_N1,W1_N2,W2_N2=Train()

	n=1
	end=100
	endval=np.arange(-1*n,end)

	u_N1=np.empty((end+n,1),dtype='float')
	u_N2=np.empty((end+n,1),dtype='float')

	#print(u)
	f_N1=np.zeros([end+n,1])
	f_N2=np.zeros([end+n,1])

	yp_N1=np.empty((end+n,1),dtype='float')
	yp_N2=np.empty((end+n,1),dtype='float')
	yp_N1[0]=0
	yp_N2[0]=0
	
	for i in range(0,end+n):
		u_N1[i]=np.sin(2*np.pi*i/25)
		u_N2[i]=np.cos(2*np.pi*i/25)

	for i in range(n,end+n):
		f_N1[i-1]=yp_N1[i-1]/(1+yp_N2[i-1]**2)
		f_N2[i-1]=(yp_N1[i-1]*yp_N2[i-1])/(1+yp_N2[i-1]**2)
		yp_N1[i]=f_N1[i-1]+u_N1[i-1]
		yp_N2[i]=f_N2[i-1]+u_N2[i-1]


	yphat_N1=np.empty((end+n,1),dtype='float')
	yphat_N2=np.empty((end+n,1),dtype='float')
	yphat_N1[0]=0
	yphat_N2[0]=0

	J_N1=0
	J_N2=0
	bias=1
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
			J_N1=J_N1+(e_N1*e_N1)


			S_N1=yp_N1[i-1]
			S_N2=yp_N2[i-1]
			v0k_N2=np.array([bias,S_N1[0],S_N2[0]])
			v1k_bar_N2=tan_h(np.matmul(W1_N2,v0k_N2))
			v1k_N2=np.insert(v1k_bar_N2,0,[bias])
			fin_N2=np.matmul(W2_N2,v1k_N2)
			yphat_N2[i]=fin_N2+u_N2[i-1]
			e_N2=yp_N2[i]-yphat_N2[i]
			J_N2=J_N2+(e_N2*e_N2)

	J_N1=J_N1/end
	J_N2=J_N2/end
	
	pickle_out=open("OSLA_5A_yphat_N1.pickle","wb")
	pickle.dump(yphat_N1,pickle_out)
	pickle_out.close()
	pickle_out=open("OSLA_5A_J_N1.pickle","wb")
	pickle.dump(J_N1,pickle_out)
	pickle_out.close()

	pickle_out=open("OSLA_5A_yphat_N2.pickle","wb")
	pickle.dump(yphat_N2,pickle_out)
	pickle_out.close()
	pickle_out=open("OSLA_5A_J_N2.pickle","wb")
	pickle.dump(J_N2,pickle_out)
	pickle_out.close()
	
	print("Testing Cost for OSLA Layer N1 = ",J_N1)
	print("Testing Cost for OSLA Layer N2 = ",J_N2)
	# plt.plot(endval,yp_N1,color='g')
	# plt.plot(endval,yphat_N1,color='r')
	# plt.xlabel("Time")
	# plt.ylabel("Val")
	# plt.title("5 A N1")
	
	# plt.figure()
	# plt.plot(endval,yp_N2,color='g')
	# plt.plot(endval,yphat_N2,color='r')
	# plt.xlabel("Time")
	# plt.ylabel("Val")
	# plt.title("5 A N2")
	# plt.show()
	
if __name__=='__main__':
	Run()