import scipy.io
import numpy as np 
a = np.array([[np.zeros((256,256)), np.zeros((256,257))]],dtype=object)
print(a.shape)
print(a[0,0].shape) ## 256x256