import pickle
import scipy.io
import numpy as np 
import imp 
import cv2 


def print_debug(variable, shape=True):
    if not shape:
        print(variable, '=', repr(eval(variable)))
    else:
        variable_shape = variable+ '.shape'
        print(variable, '=', repr(eval(variable)), ';', repr(eval(variable_shape)))
#####################
# INPUT
#####################
MyRGBD = imp.load_source('RGBD', 'lib/RGBD.py')

with open('./dumps/frame0_segment_output.pkl', 'rb') as f:
    RGBD = pickle.load(f)
Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
#####################
# PROCESS
#####################
RGBD[0].myPCA()
RGBD[0].BuildBB()
RGBD[0].getWarpingPlanes()

'''
The first image is process differently from the other since it does not have any previous value.
'''
# Stock all Local to global Transform
Tg = []
Tg.append(Id4)
# bp = 0 is the background (only black) No need to process it.
for bp in range(1,RGBD[0].bdyPart.shape[0]+1):
    # Get the tranform matrix from the local coordinates system to the global system
    Tglo = RGBD[0].TransfoBB[bp]
    Tg.append(Tglo.astype(np.float32))
#####################
# OUTPUT
#####################
with open('./dumps/frame0_transformation_Tg.pkl', 'wb') as f:
    pickle.dump(Tg, f)

with open('./dumps/frame0_transformation_RGBD.pkl', 'wb') as f:
    pickle.dump(RGBD, f)