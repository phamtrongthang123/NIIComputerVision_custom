import pickle
import scipy.io
import numpy as np 
import imp 
import cv2 

#####################
# INPUT
#####################
Stitcher = imp.load_source('Stitcher', './lib/Stitching.py')
# need this because pickle can't dump C-related attribute
BdyPrt = imp.load_source('BodyParts', './lib/BodyParts_custom.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
GPUManager = GPU.GPUManager()
GPUManager.print_device_info()
GPUManager.load_kernels()
PoseBP = np.zeros((15, 4, 4), dtype=np.float32)
Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
Index = 4
with open('./dumps/frame0_transformation_Tg.pkl', 'rb') as f:
    Tg = pickle.load(f)

with open('./dumps/frame0_transformation_RGBD.pkl', 'rb') as f:
    RGBD= pickle.load(f)
#####################
# PROCESS
#####################
nb_verticesGlo = 0
nb_facesGlo = 0
# Number of body part (+1 since the counting starts from 1)
bpstart = 1
nbBdyPart = RGBD[0].bdyPart.shape[0]+1
#Initialize stitcher object. It stitches the body parts
StitchBdy = Stitcher.Stitch(nbBdyPart)
boneTrans, boneSubTrans = StitchBdy.GetVBonesTrans(RGBD[0].skeVtx[0], RGBD[0].skeVtx[0])
boneTr_all = boneTrans
boneSubTr_all = boneSubTrans
# Initialize Body parts
Parts = []
Parts.append(BdyPrt.BodyParts(RGBD[0],RGBD[0], Tg[0]))

BPVtx = []
BPVtx.append(np.array((0,0,0)))
BPNml = []
BPNml.append(np.array((0,0,0)))
# Creating mesh of each body part range(1,15)
for bp in range(bpstart,nbBdyPart):
    # get dual quaternion of bond's transformation
    boneMDQ, boneJDQ = StitchBdy.getJointInfo(bp, boneTrans, boneSubTrans)

    # create volume
    Parts.append(BdyPrt.BodyParts(RGBD[0], RGBD[bp], Tg[bp]))
    # Compute the 3D Model (TSDF + MC)
    # Saving self.MC.SaveToPly("body0_" + bpStr + ".ply") here
    Parts[bp].Model3D_init(bp, boneJDQ, GPUManager)
    # Update number of vertices and faces in the stitched mesh
    nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
    nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]

    #Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
    for i in range(4):
        for j in range(4):
            PoseBP[bp][i][j] = Tg[bp][i][j]
    # Concatenate all the body parts for stitching purpose
    BPVtx.append(StitchBdy.TransformVtx(Parts[bp].MC.Vertices, RGBD[0].coordsGbl[bp], RGBD[0].coordsGbl[bp], RGBD[0].BBTrans[bp], Id4, Id4, 0, PoseBP[bp], bp))
    BPNml.append(StitchBdy.TransformNmls(Parts[bp].MC.Normales, Parts[bp].MC.Vertices, RGBD[0].coordsGbl[bp], RGBD[0].coordsGbl[bp], RGBD[0].BBTrans[bp], Id4, Id4, 0, PoseBP[bp], bp))
    if bp == bpstart  :
        StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices, RGBD[0].coordsGbl[bp], RGBD[0].coordsGbl[bp], RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, RGBD[0].planesF[bp], PoseBP[bp], bp,1, RGBD[0])
        StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,Parts[bp].MC.Vertices, RGBD[0].coordsGbl[bp], RGBD[0].coordsGbl[bp], RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, RGBD[0].planesF[bp], PoseBP[bp], bp,1, RGBD[0])
        StitchBdy.StitchedFaces = Parts[bp].MC.Faces
    else:
        StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces, RGBD[0].coordsGbl[bp], RGBD[0].coordsGbl[bp], RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, RGBD[0].planesF[bp], PoseBP[bp], bp, RGBD[0])

Parts[1].MC.SaveToPlyExt("wholeBody"+str(Index)+".ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces)
#####################
# OUTPUT
#####################
with open('./dumps/frame0_stitcher_Parts.pkl', 'wb') as f:
    pickle.dump(Parts, f)

with open('./dumps/frame0_stitcher_StitchBdy.pkl', 'wb') as f:
    pickle.dump(StitchBdy, f)
with open('./dumps/frame0_stitcher_BPVtx.pkl', 'wb') as f:
    pickle.dump(BPVtx, f)

with open('./dumps/frame0_stitcher_BPNml.pkl', 'wb') as f:
    pickle.dump(BPNml, f)