"""
7 August 2017
@author: Inoe ANDRE
All process within a body part
"""


import numpy as np
import math
from numpy import linalg as LA
import imp
import time
import sys 

PI = math.pi
RGBD = imp.load_source('RGBD', './lib/RGBD.py')
TrackManager = imp.load_source('TrackManager', './lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', './lib/TSDF.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
My_MC = imp.load_source('My_MarchingCube', './lib/My_MarchingCube_custom.py')
Stitcher = imp.load_source('Stitcher', './lib/Stitching.py')
General = imp.load_source('General', './lib/General.py')


class BodyParts():
    """
    Body parts
    """
    def __init__(self, RGBD,RGBD_BP, Tlg):
        """
        Init a body parts
        :param RGBD: Image having all the body
        :param RGBD_BP: Image containing just the body part
        :param Tlg: Transform local to global for the concerned body parts
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.RGBD = RGBD
        self.Tlg =Tlg
        self.RGBD_BP = RGBD_BP
        self.VoxSize = 0.005


    def Model3D_init(self,bp, jointDQ, GPUManager):
        """
        Create a 3D model of the body parts
        :param bp: number of the body part
        :param jointDQ: first frame bind matrix in dual quaternion type
        :return:  none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))

        # need to put copy transform amtrix in PoseBP for GPU
        PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        Id4DQ = np.array([[1., 0., 0., 0.], [0., 0., 0., 0.]], dtype = np.float32)

        # Compute the dimension of the body part to create the volume
        wider = self.VoxSize*5
        Xraw = int(round( (self.RGBD.BBsize[bp][0]+wider*2) / self.VoxSize)) + 1
        Yraw = int(round( (self.RGBD.BBsize[bp][1]+wider*2) / self.VoxSize)) + 1
        Zraw = int(round( (self.RGBD.BBsize[bp][2]+wider*2) / self.VoxSize)) + 1

        # Dimensions of body part volume
        X = max(Xraw, Zraw)
        Y = Yraw
        Z = max(Xraw, Zraw)
        # show result
        print "bp = %d, X= %d; Y= %d; Z= %d" % (bp, X, Y, Z)

        # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
        for i in range(4):
            for j in range(4):
                PoseBP[i][j] = self.Tlg[i][j]

        # TSDF Fusion of the body part
        TSDFManager = TSDFtk.TSDFManager((X, Y, Z), self.RGBD_BP, GPUManager, self.RGBD.planesF[bp], PoseBP, self.VoxSize)
        TSDFManager.FuseRGBD_GPU(self.RGBD_BP, Id4DQ, jointDQ)

        # Create Mesh
        self.MC = My_MC.My_MarchingCube(TSDFManager.Size, TSDFManager.res, 0.0)
        # Mesh rendering
        self.MC.runGPU(TSDFManager.TSDFGPU, GPUManager)
        start_time3 = time.time()

        # save with the number of the body part
        bpStr = str(bp)
        self.MC.SaveToPly("body0_" + bpStr + ".ply")
        elapsed_time = time.time() - start_time3
        print "SaveBPToPly: %f" % (elapsed_time)





