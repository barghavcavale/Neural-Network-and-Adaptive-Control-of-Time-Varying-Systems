import numpy as np

def vaf(a,b):
    return 100*(1-(np.var(a-b))/(np.var(a)))