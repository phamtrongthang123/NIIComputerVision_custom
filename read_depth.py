import scipy.io
import numpy as np 
import MyRGBD
# Read depth 
matfilename ='String4b'
mat = scipy.io.loadmat('data/' + matfilename + '.mat')
lImages = mat['DepthImg'] 

ColorImg = np.zeros((0))

cut_out = lImages[0][0]
far = 5000
near = 3000
cut_out[cut_out>far] = 0 
cut_out[cut_out<near] = 0

Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)

calib_file = open('data/Calib.txt', 'r')
calib_data = calib_file.readlines()
Size = [int(calib_data[0]), int(calib_data[1])]
intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                            [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                            [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)

# print(intrinsic)
# [[357.324   0.    250.123]
#  [  0.    362.123 217.526]
#  [  0.      0.      1.   ]]
fact = 690
RGBD = []
Index = 4
# add an RGBD Object in the list
# the path is dummy, it only 
RGBD.append(MyRGBD.RGBD('/Depth.tiff', '/RGB.tiff', intrinsic, fact))
# load data in the RGBD Object
RGBD[0].LoadMat(cut_out)
RGBD[0].ReadFromMat(Index)
# process depth image
RGBD[0].BilateralFilter(-1, 0.02, 3)

# Compute vertex map and normal map
RGBD[0].Vmap_optimize()
RGBD[0].NMap_optimize()

# create the transform matrices that transform from local to global coordinate
RGBD[0].myPCA()
RGBD[0].BuildBB()
RGBD[0].getWarpingPlanes()