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
# Read depth 
matfilename ='String4b'
mat = scipy.io.loadmat('../data/' + matfilename + '.mat')
lImages = mat['DepthImg'] 
pos2d = mat['Pos2D']
ColorImg = np.zeros((0))


Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
print('Pose ', Pose)
calib_file = open('../data/Calib.txt', 'r')
calib_data = calib_file.readlines()
Size = [int(calib_data[0]), int(calib_data[1])]
print('Size =', Size )
intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                            [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                            [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)

# print(intrinsic)
# [[357.324   0.    250.123]
#  [  0.    362.123 217.526]
#  [  0.      0.      1.   ]]
connectionMat = scipy.io.loadmat('../data/SkeletonConnectionMap.mat')
connection = connectionMat['SkeletonConnectionMap']
print_debug('connection')
fact = 690
RGBD = []
Index = 4

#####################
# PROCESS
#####################
for bp in range(15):
    # add an RGBD Object in the list
    RGBD.append(MyRGBD.RGBD('dummy/Depth.tiff', 'dummy/RGB.tiff', intrinsic, fact))
    # load data in the RGBD Object
    RGBD[bp].LoadMat(lImages,pos2d,connection, ColorImg)
    RGBD[bp].ReadFromMat(Index)
    # process depth image
    RGBD[bp].BilateralFilter(-1, 0.02, 3)
    # segmenting the body
    if bp == 0:
        RGBD[bp].RGBDSegmentation()
        RGBD[bp].depth_image *= (RGBD[bp].labels >0)
    else:
        RGBD[bp].depth_image *= (RGBD[0].labelList[bp] >0)
    # Compute vertex map and normal map
    RGBD[bp].Vmap_optimize()
    RGBD[bp].NMap_optimize()
    print(RGBD[bp].depth_image.shape)
    print(RGBD[bp].Vtx.shape)
    print(RGBD[bp].Nmls.shape)
    
#####################
# OUTPUT
#####################
import pickle
with open('./dumps/frame0_segment_output.pkl', 'wb') as f:
    pickle.dump(RGBD, f)