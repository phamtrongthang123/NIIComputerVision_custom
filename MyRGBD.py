"""
 File created by Diego Thomas the 16-11-2016
 improved by Inoe Andre from 02-2017

 Define functions to manipulate RGB-D data
"""
import cv2
import sys 
import numpy as np
from numpy import linalg as LA
import random
import imp
import time
import scipy.ndimage.measurements as spm
import pdb
from skimage import img_as_ubyte
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy
from skimage.draw import line_aa

segm = imp.load_source('segmentation', './code/lib/segmentation.py')
General = imp.load_source('General', './code/lib/General.py')



class RGBD():
    """
    Class to handle any processing on depth image and the image breed from the depth image
    """

    def __init__(self, depthname, colorname, intrinsic, fact):
        """
        Constructor
        :param depthname: path to a depth image
        :param colorname: path to a RGBD image
        :param intrinsic: matrix with calibration parameters
        :param fact: factor for converting pixel value to meter or conversely
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.depthname = depthname # useless
        self.colorname = colorname # useless
        self.intrinsic = intrinsic
        self.fact = fact

    def LoadMat(self, Images):
        """
        Load information in datasets into the RGBD object
        :param Images: List of depth images put in function of time
        :param Pos_2D: List of junctions position for each depth image
        :param BodyConnection: list of doublons that contains the number of pose that represent adjacent body parts
        :return:  none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.depth_img = Images
        # self.numbImages = len(self.depth_img.transpose()) # useless
        self.numbImages = 1
        self.Index = -1


    def ReadFromMat(self, idx = -1):
        """
        Read an RGB-D image from matrix (dataset)
        :param idx: number of the
        :return:
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        if (idx == -1):
            self.Index = self.Index + 1
        else:
            self.Index = idx

        depth_in = self.depth_img
        print "Input depth image is of size: " + str(depth_in.shape)
        size_depth = depth_in.shape
        self.Size = (size_depth[0], size_depth[1], 3)
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        self.depth_image_ori = depth_in
        self.depth_image = depth_in.astype(np.float32) / self.fact
        # self.skel = self.depth_image.copy() # useless

        # handle positions which are out of boundary
        # we don't need this, we alread cut near far
        # self.pos2d[0,idx][:,0] = (np.maximum(0, self.pos2d[0, idx][:,0]))
        # self.pos2d[0,idx][:,1] = (np.maximum(0, self.pos2d[0, idx][:,1]))
        # self.pos2d[0,idx][:,0] = (np.minimum(self.Size[1], self.pos2d[0, idx][:,0]))
        # self.pos2d[0,idx][:,1] = (np.minimum(self.Size[0], self.pos2d[0, idx][:,1]))

        # get kmeans of image
        # if self.hasColor:
        #     self.color_image = self.CImages[0][self.Index]

    #####################################################################
    ################### Map Conversion Functions #######################
    #####################################################################

    def Vmap(self):
        """
        Create the vertex image from the depth image and intrinsic matrice
        :return: none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.Vtx = np.zeros(self.Size, np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]): # column index (i.e. horizontal x axis)
                d = self.depth_image[i,j]
                if d > 0.0:
                    x = d*(j - self.intrinsic[0,2])/self.intrinsic[0,0]
                    y = d*(i - self.intrinsic[1,2])/self.intrinsic[1,1]
                    self.Vtx[i,j] = (x, y, d)


    def Vmap_optimize(self):
        """
        Create the vertex image from the depth image and intrinsic matrice
        :return: none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        #self.Vtx = np.zeros(self.Size, np.float32)
        #matrix containing depth value of all pixel
        d = self.depth_image[0:self.Size[0]][0:self.Size[1]]
        d_pos = d * (d > 0.0)
        # create matrix that contains index values
        x_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        y_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        # change the matrix so that the first row is on all rows for x respectively colunm for y.
        x_raw[0:-1,:] = ( np.arange(self.Size[1]) - self.intrinsic[0,2])/self.intrinsic[0,0]
        y_raw[:,0:-1] = np.tile( ( np.arange(self.Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()
        # multiply point by point d_pos and raw matrices
        x = d_pos * x_raw
        y = d_pos * y_raw
        self.Vtx = np.dstack((x, y,d))
        return self.Vtx

    def NMap(self):
        """
        Compute normal map
        :return: none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.Nmls = np.zeros(self.Size, np.float32)
        for i in range(1,self.Size[0]-1):
            for j in range(1, self.Size[1]-1):
                # normal for each direction
                nmle1 = General.normalized_cross_prod(self.Vtx[i+1, j]-self.Vtx[i, j], self.Vtx[i, j+1]-self.Vtx[i, j])
                nmle2 = General.normalized_cross_prod(self.Vtx[i, j+1]-self.Vtx[i, j], self.Vtx[i-1, j]-self.Vtx[i, j])
                nmle3 = General.normalized_cross_prod(self.Vtx[i-1, j]-self.Vtx[i, j], self.Vtx[i, j-1]-self.Vtx[i, j])
                nmle4 = General.normalized_cross_prod(self.Vtx[i, j-1]-self.Vtx[i, j], self.Vtx[i+1, j]-self.Vtx[i, j])
                nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
                # normalized
                if (LA.norm(nmle) > 0.0):
                    nmle = nmle/LA.norm(nmle)
                self.Nmls[i, j] = (nmle[0], nmle[1], nmle[2])

    def NMap_optimize(self):
        """
        Compute normal map, CPU optimize algo
        :return: none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.Nmls = np.zeros(self.Size, np.float32)
        # matrix of normales for each direction
        nmle1 = General.normalized_cross_prod_optimize(self.Vtx[2:self.Size[0],1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1,2:self.Size[1]] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle2 = General.normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[0:self.Size[0]-2,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle3 = General.normalized_cross_prod_optimize(self.Vtx[0:self.Size[0]-2,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle4 = General.normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[2:self.Size[0],1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
        # normalized
        norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
        norm_mat_nmle = General.in_mat_zero2one(norm_mat_nmle)
        #norm division
        nmle = General.division_by_norm(nmle,norm_mat_nmle)
        self.Nmls[1:self.Size[0]-1][:,1:self.Size[1]-1] = nmle
        return self.Nmls

    #############################################################################
    ################### Projection and transform Functions #######################
    #############################################################################

    def DrawMesh(self, rendering,Vtx,Nmls,Pose, s, color = 2) :
        """
        Project vertices and normales from a mesh in 2D images
        :param rendering : 2D image for overlay purpose or black image
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if color=0, put color in the image, if color=1, put boolean in the image
        :return: scene projected in 2D space
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        result = rendering#np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)#
        stack_pix = np.ones( (np.size(Vtx[ ::s,:],0)) , dtype = np.float32)
        stack_pt = np.ones( (np.size(Vtx[ ::s,:],0)) , dtype = np.float32)
        pix = np.zeros( (np.size(Vtx[ ::s,:],0),2) , dtype = np.float32)
        pix = np.stack((pix[:,0],pix[:,1],stack_pix),axis = 1)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        pt = np.dot(pt,Pose.T)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Pose[0:3,0:3].T)


        # projection in 2D space
        lpt = np.split(pt,4,axis=1)
        lpt[2] = General.in_mat_zero2one(lpt[2])
        pix[ :,0] = (lpt[0]/lpt[2]).reshape(np.size(Vtx[ ::s,:],0))
        pix[ :,1] = (lpt[1]/lpt[2]).reshape(np.size(Vtx[ ::s,:],0))
        pix = np.dot(pix,self.intrinsic.T)

        column_index = (np.round(pix[:,0])).astype(int)
        line_index = (np.round(pix[:,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < self.Size[1])
        cdt_line = (line_index > -1) * (line_index < self.Size[0])
        cdt = cdt_column*cdt_line
        line_index = line_index*cdt
        column_index = column_index*cdt
        if (color == 0):
            result[line_index[:], column_index[:]]= np.dstack((self.color_image[ line_index[:], column_index[:],2]*cdt, \
                                                                    self.color_image[ line_index[:], column_index[:],1]*cdt, \
                                                                    self.color_image[ line_index[:], column_index[:],0]*cdt) )
        elif (color == 1):
            result[line_index[:], column_index[:]]= 1.0
        else:
            result[line_index[:], column_index[:]]= np.dstack( ( (nmle[ ::s,0]+1.0)*(255./2.)*cdt, \
                                                                       ((nmle[ ::s,1]+1.0)*(255./2.))*cdt, \
                                                                       ((nmle[ ::s,2]+1.0)*(255./2.))*cdt ) ).astype(int)
        return result


##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        """
        Bilateral filtering the depth image
        see cv2 documentation
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.depth_image = (self.depth_image[:,:] > 0.0) * cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)


##################################################################
################### Segmentation Function #######################
##################################################################
    def RemoveBG(self,binaryImage):
        """
        Delete all the little group (connected component) unwanted from the binary image
        :param binaryImage: a binary image containing several connected component
        :return: A binary image containing only big connected component
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        labeled, n = spm.label(binaryImage)
        size = np.bincount(labeled.ravel())
        #do not consider the background
        size2 = np.delete(size,0)
        threshold = max(size2)-1
        keep_labels = size >= threshold
        # Make sure the background is left as 0/False
        keep_labels[0] = 0
        filtered_labeled = keep_labels[labeled]
        return filtered_labeled

    def BdyThresh(self):
        """
        Threshold the depth image in order to to get the whole body alone with the bounding box (BB)
        :return: The connected component that contain the body
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        #'''
        pos2D = self.CroppedPos
        max_value = 1
        self.CroppedBox = self.CroppedBox.astype(np.uint16)
        # Threshold according to detph of the body
        bdyVals = self.CroppedBox[pos2D[self.connection[:,0]-1,1]-1,pos2D[self.connection[:,0]-1,0]-1]
        #only keep vales different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini =  np.min(bdy)
        #print "mini: %u" % (mini)
        maxi = np.max(bdy)
        #print "max: %u" % (maxi)
        # double threshold according to the value of the depth
        bwmin = (self.CroppedBox > mini-0.01*max_value)
        bwmax = (self.CroppedBox < maxi+0.01*max_value)
        bw0 = bwmin*bwmax
        # Remove all stand alone object
        bw0 = ( self.RemoveBG(bw0)>0)
        '''
        #for MIT
        bw0 = (self.CroppedBox>0)
        #'''
        return bw0





#######################################################################
################### Bounding boxes Function #######################
##################################################################


    def SetTransfoMat3D(self,evecs,i):
        """
        Generate the transformation matrix
        :param evecs: eigen vectors
        :param i: number of the body part
        :return: none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        ctr = self.ctr3D[i]#self.coordsGbl[i][0]#[0.,0.,0.]#
        e1 = evecs[0]
        e2 = evecs[1]
        e3 = evecs[2]
        # axis of coordinates system
        e1b = np.array( [e1[0],e1[1],e1[2],0])
        e2b = np.array( [e2[0],e2[1],e2[2],0])
        e3b = np.array( [e3[0],e3[1],e3[2],0])
        #center of coordinates system
        origine = np.array( [ctr[0],ctr[1],ctr[2],1])
        # concatenate it in the right order.
        Transfo = np.stack( (e1b,e2b,e3b,origine),axis = 0 )
        self.TransfoBB.append(Transfo.transpose())
        #display
        #print "TransfoBB[%d]" %(i)
        #print self.TransfoBB[i]


    def bdyPts3D(self, mask):
        """
        create of cloud of point from part of the RGBD image
        :param mask: a matrix containing one only in the body parts indexes, 0 otherwise
        :return:  list of vertices = cloud of points
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        start_time2 = time.time()
        nbPts = sum(sum(mask))
        res = np.zeros((nbPts, 3), dtype = np.float32)
        k = 0
        for i in range(self.Size[0]):
            for j in range(self.Size[1]):
                if(mask[i,j]):
                    res[k] = self.Vtx[i,j]
                    k = k+1
        elapsed_time3 = time.time() - start_time2
        print "making pointcloud process time: %f" % (elapsed_time3)
        return res

    def bdyPts3D_optimize(self, mask):
        """
        create of cloud of point from part of the RGBD image
        :param mask: a matrix containing one only in the body parts indexes, 0 otherwise
        :return:  list of vertices = cloud of points
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        #start_time2 = time.time()
        nbPts = sum(sum(mask))

        # threshold with the mask
        x = self.Vtx[:,:,0]*mask
        y = self.Vtx[:,:,1]*mask
        z = self.Vtx[:,:,2]*mask

        #keep only value that are different from 0 in the list
        x_res = x[~(z==0)]
        y_res = y[~(z==0)]
        z_res = z[~(z==0)]

        #concatenate each axis
        res = np.dstack((x_res,y_res,z_res)).reshape(nbPts,3)

        #elapsed_time3 = time.time() - start_time2
        #print "making pointcloud process time: %f" % (elapsed_time3)

        return res

    def getSkeletonVtx(self):
        """
        calculate the skeleton in 3D
        :retrun: none
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        # get pos2D
        pos2D = self.pos2d[0,self.Index].astype(np.double)-1
        # initialize
        skedepth = np.zeros(25)

        # compute depth of each junction
        for i in range(21): # since 21~24 uesless
            if i==0 or i == 1 or i == 20:
                j=10
            elif i==2 or i==3:
                j=9
            elif i==4 or i==5:
                j=2
            elif i==6:
                j=1
            elif i==7 or i==21 or i==22:
                j=12
            elif i==8 or i==9:
                j=4
            elif i==10:
                j=3
            elif i==11 or i==23 or i==24:
                j=11
            elif i==12:
                j=7
            elif i==13 or i==14:
                j=8
            elif i==15:
                j=13
            elif i==16:
                j=5
            elif i==17 or i==18:
                j=6
            elif i==19:
                j=14

            depth = abs(np.amax(self.coordsGbl[j][:,2])-np.amin(self.coordsGbl[j][0,2]))/2
            depth = 0
            if self.labels[int(pos2D[i][1]), int(pos2D[i][0])]!=0:
                skedepth[i] = self.depth_image[int(pos2D[i][1]), int(pos2D[i][0])]+depth
            else:
                print "meet the pose " + str(i) + "==0 when getting junction"
                if self.labels[int(pos2D[i][1])+1, int(pos2D[i][0])]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1])+1, int(pos2D[i][0])]+depth
                elif self.labels[int(pos2D[i][1]), int(pos2D[i][0])+1]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1]), int(pos2D[i][0])+1]+depth
                elif self.labels[int(pos2D[i][1])-1, int(pos2D[i][0])]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1])-1, int(pos2D[i][0])]+depth
                elif self.labels[int(pos2D[i][1]), int(pos2D[i][0])-1]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1]), int(pos2D[i][0])-1]+depth
                else:
                    print "QAQQQQ"
                    #exit()

        #  project to 3D
        pos2D[:,0] = (pos2D[:,0]-self.intrinsic[0,2])/self.intrinsic[0,0]
        pos2D[:,1] = (pos2D[:,1]-self.intrinsic[1,2])/self.intrinsic[1,1]
        x = skedepth * pos2D[:,0]
        y = skedepth * pos2D[:,1]
        z = skedepth

        # give hand and foot't joint correct
        for i in [7,11,15,19]:
            x[i] = (x[i-1]-x[i-2])/4+x[i-1]
            y[i] = (y[i-1]-y[i-2])/4+y[i-1]
            z[i] = (z[i-1]-z[i-2])/4+z[i-1]

        return np.dstack((x,y,z)).astype(np.float32)

    def FindCoord3D(self,i):
        '''
        draw the bounding boxes in 3D for each part of the human body
        :param i : number of the body parts
        '''
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        # Adding a space so that the bounding boxes are wider
        VoxSize = 0.005
        wider = 5*VoxSize*0
        # extremes planes of the bodies
        minX = np.min(self.TVtxBB[i][:,0]) - wider
        maxX = np.max(self.TVtxBB[i][:,0]) + wider
        minY = np.min(self.TVtxBB[i][:,1]) - wider
        maxY = np.max(self.TVtxBB[i][:,1]) + wider
        minZ = np.min(self.TVtxBB[i][:,2]) - wider
        maxZ = np.max(self.TVtxBB[i][:,2]) + wider
        # extremes points of the bodies
        xymz = np.array([minX,minY,minZ])
        xYmz = np.array([minX,maxY,minZ])
        Xymz = np.array([maxX,minY,minZ])
        XYmz = np.array([maxX,maxY,minZ])
        xymZ = np.array([minX,minY,maxZ])
        xYmZ = np.array([minX,maxY,maxZ])
        XymZ = np.array([maxX,minY,maxZ])
        XYmZ = np.array([maxX,maxY,maxZ])

        # New coordinates and new images
        self.coordsL.append( np.array([xymz,xYmz,XYmz,Xymz,xymZ,xYmZ,XYmZ,XymZ]) )
        #print "coordsL[%d]" %(i)
        #print self.coordsL[i]

        # transform back
        self.coordsGbl.append( self.pca[i].inverse_transform(self.coordsL[i]))
        #print "coordsGbl[%d]" %(i)
        #print self.coordsGbl[i]

        # save the boundingboxes size
        self.BBsize.append([LA.norm(self.coordsGbl[i][3] - self.coordsGbl[i][0]), LA.norm(self.coordsGbl[i][1] - self.coordsGbl[i][0]), LA.norm(self.coordsGbl[i][4] - self.coordsGbl[i][0])])

\


    def GetProjPts2D(self, vects3D, Pose, s=1) :
        """
        Project a list of vertexes in the image RGBD
        :param vects3D: list of 3 elements vector
        :param Pose: Transformation matrix
        :param s: subsampling coefficient
        :return: transformed list of 3D vector
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        drawVects = []
        for i in range(len(vects3D)):
            pt[0] = vects3D[i][0]
            pt[1] = vects3D[i][1]
            pt[2] = vects3D[i][2]
            # transform list
            pt = np.dot(Pose, pt)
            pt /= pt[:,3].reshape((pt.shape[0], 1))
            #Project it in the 2D space
            if (pt[2] != 0.0):
                pix[0] = pt[0]/pt[2]
                pix[1] = pt[1]/pt[2]
                pix = np.dot(self.intrinsic, pix)
                column_index = pix[0].astype(np.int)
                line_index = pix[1].astype(np.int)
            else :
                column_index = 0
                line_index = 0
            #print "line,column index : (%d,%d)" %(line_index,column_index)
            drawVects.append(np.array([column_index,line_index]))
        return drawVects

    def GetProjPts2D_optimize(self, vects3D, Pose) :
        """
        Project a list of vertexes in the image RGBD. Optimize for CPU version.
        :param vects3D: list of 3 elements vector
        :param Pose: Transformation matrix
        :return: transformed list of 3D vector
        """
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        '''Project a list of vertexes in the image RGBD'''
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        pix = np.stack((pix for i in range(len(vects3D)) ))
        pt = np.stack((pt for i in range(len(vects3D)) ))
        pt[:,0:3] = vects3D
        # transform list
        pt = np.dot(pt,Pose.T)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        # Project it in the 2D space
        pt[:,2] = General.in_mat_zero2one(pt[:,2])
        pix[:,0] = pt[:,0]/pt[:,2]
        pix[:,1] = pt[:,1]/pt[:,2]
        pix = np.dot( pix,self.intrinsic.T)
        column_index = pix[:,0].astype(np.int)
        line_index = pix[:,1].astype(np.int)
        drawVects = np.array([column_index,line_index]).T
        return drawVects



    def GetNewSys(self, Pose,ctr2D,nbPix) :
        '''
        compute the coordinates of the points that will create the coordinates system
        '''
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        self.drawNewSys = []
        maxDepth = max(0.0001, np.max(self.Vtx[:,:,2]))

        for i in range(1,len(self.vects3D)):
            vect = np.dot(self.vects3D[i],Pose[0:3,0:3].T )
            vect /= vect[:,3].reshape((vect.shape[0], 1))
            newPt = np.zeros(vect.shape)
            for j in range(vect.shape[0]):
                newPt[j][0] = ctr2D[i][0]-nbPix*vect[j][0]
                newPt[j][1] = ctr2D[i][1]-nbPix*vect[j][1]
                newPt[j][2] = vect[j][2]-nbPix*vect[j][2]/maxDepth
            self.drawNewSys.append(newPt)



    def Cvt2RGBA(self,im_im):
        '''
        convert an RGB image in RGBA to put all zeros as transparent
        THIS FUNCTION IS NOT USED IN THE PROJECT
        '''
        print("Call {}::{}".format(self.__class__.__name__,sys._getframe(0).f_code.co_name))
        img = im_im.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        return img

