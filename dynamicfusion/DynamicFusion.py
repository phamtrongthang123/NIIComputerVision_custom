

# read input depth 
self.root = master
self.path = path
self.GPUManager = GPUManager
self.draw_bump = False # useless
self.draw_spline = False # useless

tk.Frame.__init__(self, master)
self.pack()

self.color_tag = 2
# Calibration matrix
calib_file = open(self.path + '/Calib.txt', 'r')
calib_data = calib_file.readlines()
self.Size = [int(calib_data[0]), int(calib_data[1])]
self.intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                            [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                            [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)

print self.intrinsic

fact = 690

TimeStart = time.time()

#load data
matfilename ='String_4b'
mat = scipy.io.loadmat(path + '/' + matfilename + '.mat')
lImages = mat['DepthImg']
self.pos2d = mat['Pos2D']

# use color image in Segmentation
if 'Color2DepthImg' in mat:
    ColorImg = mat['Color2DepthImg']
else:
    ColorImg = np.zeros((0))
ColorImg = np.zeros((0))
# initialization
self.connectionMat = scipy.io.loadmat(path + '/SkeletonConnectionMap.mat')
self.connection = self.connectionMat['SkeletonConnectionMap']
self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
T_Pose = []
PoseBP = np.zeros((15, 4, 4), dtype=np.float32)
Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)

# number of images in the sequence. Start and End
self.Index = 4
nunImg = 8
sImg = 1

# Former Depth Image (i.e: i)
self.RGBD = []



# Update deform W 


# fuse tsdf 
self.Size = Size
self.c_x = Size[0]/2
self.c_y = Size[1]/2
self.c_z = Size[2]/2
# resolution
#VoxSize = 0.005
self.dim_x = 1.0/VoxSize
self.dim_y = 1.0/VoxSize
self.dim_z = 1.0/VoxSize
# put dimensions and resolution in one vector
self.res = np.array([self.c_x, self.dim_x, self.c_y, self.dim_y, self.c_z, self.dim_z], dtype = np.float32)

self.GPUManager = GPUManager
self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                        hostbuf = np.array([self.Size[0], self.Size[1], self.Size[2]], dtype = np.int32))

# TSDF is V, Weight is W, instead of R^2 in paper, they split it into 2 matrix for each voxel xyz index
self.TSDF = np.zeros(self.Size, dtype=np.int16)
self.TSDFGPU =cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)
self.Weight = np.zeros(self.Size, dtype=np.int16)
self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)

self.Param = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                        hostbuf = self.res)
# what is this pose, just a simple diag(1) rank 4 
self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
self.DepthGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, Image.depth_image.nbytes)
self.Calib_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = Image.intrinsic)
self.Pose_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, self.Pose.nbytes)
cl.enqueue_write_buffer(self.GPUManager.queue, self.Pose_GPU, Tg)

# calculate the corner weight
self.planeF = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=planeF)
tempDQ = np.zeros((2,4), dtype=np.float32)
self.boneDQGPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, tempDQ.nbytes)
self.jointDQGPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, tempDQ.nbytes)

# for next frame