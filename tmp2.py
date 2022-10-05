import scipy.io
import numpy as np 
import sys
# a = np.array([[np.zeros((256,256)), np.zeros((256,257))]],dtype=object)
# print(a.shape)
# print(a[0,0].shape) ## 256x256

class A():
    def __init__(self):
        print(self.__class__.__name__)
        print(sys._getframe(0).f_code.co_name)
        print(f'aa')

a = A()