import numpy as np
def tan_h(a):
	return(np.tanh(a));
def dtan_h(a):
	return(1-(np.tanh(a)*np.tanh(a)))
	
