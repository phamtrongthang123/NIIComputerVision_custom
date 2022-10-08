import scipy.io
import numpy as np 
import MyRGBD
import cv2 
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

Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)

# First it is transformation local global, idk what this is 
Tg = []
Tg.append(Id4)
# bp = 0 is the background (only black) No need to process it.
for bp in range(1,RGBD[0].bdyPart.shape[0]+1):
    # Get the tranform matrix from the local coordinates system to the global system
    Tglo = RGBD[0].TransfoBB[bp]
    Tg.append(Tglo.astype(np.float32))

# Then stitch the body 
# Sum of the number of vertices and faces of all body parts
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
Parts.append(BdyPrt.BodyParts(GPUManager,RGBD[0],RGBD[0], Tg[0]))
BPVtx = []
BPVtx.append(np.array((0,0,0)))
BPNml = []
BPNml.append(np.array((0,0,0)))
# Creating mesh of each body part
for bp in range(bpstart,nbBdyPart):
    # get dual quaternion of bond's transformation
    boneMDQ, boneJDQ = StitchBdy.getJointInfo(bp, boneTrans, boneSubTrans)

    # create volume
    Parts.append(BdyPrt.BodyParts(GPUManager, RGBD[0], RGBD[bp], Tg[bp]))
    # Compute the 3D Model (TSDF + MC)
    Parts[bp].Model3D_init(bp, boneJDQ)

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
    # save vertex in global of each body part
    # Parts[1].MC.SaveToPlyExt("GBody"+str(Index)+"_"+str(bp)+".ply",Parts[bp].MC.nb_vertices[0],Parts[bp].MC.nb_faces[0],StitchBdy.TransformVtx(Parts[bp].MC.Vertices,RGBD[0].coordsGbl[bp], RGBD[0].coordsGbl[bp], RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, RGBD[0].planesF[bp], PoseBP[bp], bp,1,RGBD[0]),Parts[bp].MC.Faces)

# save with the number of the body part
Parts[1].MC.SaveToPlyExt("wholeBody"+str(Index)+".ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces)


# projection in 2d space to draw the 3D model(meshes) (testing)
rendering =np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
bbrendering =np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
rendering_oneVtx = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
rendering = RGBD[0].DrawMesh(rendering,StitchBdy.StitchedVertices,StitchBdy.StitchedNormales,Id4, 1, color_tag)
cornernal =np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
# show segmentation result
img_label_temp = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
img_label_temp = DrawColors2D(RGBD[0], img_label_temp)
img_label = img_label_temp.copy()
img_label[:,:,0] = img_label_temp[:,:,2].copy()
img_label[:,:,1] = img_label_temp[:,:,1].copy()
img_label[:,:,2] = img_label_temp[:,:,0].copy()
for i in range(3):
    for j in range(3):
        img_label[pos2d[0,Index][:,1].astype(np.int16)+i-1, pos2d[0,Index][:,0].astype(np.int16)+j-1,0] = 255
        img_label[pos2d[0,Index][:,1].astype(np.int16)+i-1, pos2d[0,Index][:,0].astype(np.int16)+j-1,1] = 220
        img_label[pos2d[0,Index][:,1].astype(np.int16)+i-1, pos2d[0,Index][:,0].astype(np.int16)+j-1,2] = 240
img_label = img_label.astype(np.double)/255
# show depth map result
img_depthmap_temp =(RGBD[0].lImages[0][Index].astype(np.double)/7000*255).astype(np.uint8)
img_depthmap = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
img_depthmap[:,:,0] = img_depthmap_temp.copy()
img_depthmap[:,:,1] = img_depthmap_temp.copy()
img_depthmap[:,:,2] = img_depthmap_temp.copy()
img_skeleton = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
for i in range(3):
    for j in range(3):
        img_skeleton[pos2d[0,Index][:,1].astype(np.int16)+i-1, pos2d[0,Index][:,0].astype(np.int16)+j-1,0:3] = 1000
# draw Boundingboxes
for i in range(1,len(RGBD[0].coordsGbl)):
    # Get corners of OBB
    pt = RGBD[0].GetProjPts2D_optimize(RGBD[0].coordsGbl[i],Id4)
    pt[:,0] = np.maximum(0,np.minimum(pt[:,0], Size[1]-1))
    pt[:,1] = np.maximum(0,np.minimum(pt[:,1], Size[0]-1))
    # create point of the boxes
    bbrendering[pt[:,1], pt[:,0],0] = 100
    bbrendering[pt[:,1], pt[:,0],1] = 230
    bbrendering[pt[:,1], pt[:,0],2] = 230
    if i==10:
        for j in range(7):
            rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+1][1],pt[j+1][0])
            rr = np.maximum(0,np.minimum(rr, Size[0]-1))
            cc = np.maximum(0,np.minimum(cc, Size[1]-1))
            bbrendering[rr,cc, 0] = 100
            bbrendering[rr,cc, 1] = 230
            bbrendering[rr,cc, 2] = 250
            rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+9][1],pt[j+9][0])
            rr = np.maximum(0,np.minimum(rr, Size[0]-1))
            cc = np.maximum(0,np.minimum(cc, Size[1]-1))
            bbrendering[rr,cc, 0] = 100
            bbrendering[rr,cc, 1] = 230
            bbrendering[rr,cc, 2] = 250
            rr,cc,val = line_aa(pt[j+9][1],pt[j+9][0],pt[j+10][1],pt[j+10][0])
            rr = np.maximum(0,np.minimum(rr, Size[0]-1))
            cc = np.maximum(0,np.minimum(cc, Size[1]-1))
            bbrendering[rr,cc, 0] = 100
            bbrendering[rr,cc, 1] = 230
            bbrendering[rr,cc, 2] = 250
        rr,cc,val = line_aa(pt[0][1],pt[0][0],pt[8][1],pt[8][0])
        rr = np.maximum(0,np.minimum(rr, Size[0]-1))
        cc = np.maximum(0,np.minimum(cc, Size[1]-1))
        bbrendering[rr,cc, 0] = 100
        bbrendering[rr,cc, 1] = 230
        bbrendering[rr,cc, 2] = 250
        rr,cc,val = line_aa(pt[9][1],pt[9][0],pt[17][1],pt[17][0])
        rr = np.maximum(0,np.minimum(rr, Size[0]-1))
        cc = np.maximum(0,np.minimum(cc, Size[1]-1))
        bbrendering[rr,cc, 0] = 100
        bbrendering[rr,cc, 1] = 230
        bbrendering[rr,cc, 2] = 250
    else:
        # create lines of the boxes
        for j in range(3):
            rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+1][1],pt[j+1][0])
            rr = np.maximum(0,np.minimum(rr, Size[0]-1))
            cc = np.maximum(0,np.minimum(cc, Size[1]-1))
            bbrendering[rr,cc, 0] = 100
            bbrendering[rr,cc, 1] = 230
            bbrendering[rr,cc, 2] = 250
            rr,cc,val = line_aa(pt[j+4][1],pt[j+4][0],pt[j+5][1],pt[j+5][0])
            rr = np.maximum(0,np.minimum(rr, Size[0]-1))
            cc = np.maximum(0,np.minimum(cc, Size[1]-1))
            bbrendering[rr,cc, 0] = 100
            bbrendering[rr,cc, 1] = 230
            bbrendering[rr,cc, 2] = 250
            rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+4][1],pt[j+4][0])
            rr = np.maximum(0,np.minimum(rr, Size[0]-1))
            cc = np.maximum(0,np.minimum(cc, Size[1]-1))
            bbrendering[rr,cc, 0] = 100
            bbrendering[rr,cc, 1] = 230
            bbrendering[rr,cc, 2] = 250
        rr,cc,val = line_aa(pt[3][1],pt[3][0],pt[0][1],pt[0][0])
        rr = np.maximum(0,np.minimum(rr, Size[0]-1))
        cc = np.maximum(0,np.minimum(cc, Size[1]-1))
        bbrendering[rr,cc, 0] = 100
        bbrendering[rr,cc, 1] = 230
        bbrendering[rr,cc, 2] = 250
        rr,cc,val = line_aa(pt[7][1],pt[7][0],pt[4][1],pt[4][0])
        rr = np.maximum(0,np.minimum(rr, Size[0]-1))
        cc = np.maximum(0,np.minimum(cc, Size[1]-1))
        bbrendering[rr,cc, 0] = 100
        bbrendering[rr,cc, 1] = 230
        bbrendering[rr,cc, 2] = 250
        rr,cc,val = line_aa(pt[3][1],pt[3][0],pt[7][1],pt[7][0])
        rr = np.maximum(0,np.minimum(rr, Size[0]-1))
        cc = np.maximum(0,np.minimum(cc, Size[1]-1))
        bbrendering[rr,cc, 0] = 100
        bbrendering[rr,cc, 1] = 230
        bbrendering[rr,cc, 2] = 250
# mix
result_stack = np.concatenate((rendering*0.0020+img_depthmap*0.0020+rendering_oneVtx*0.0025, np.ones((Size[0],1,3), dtype = np.uint8), bbrendering*0.001+img_depthmap*0.0020+img_skeleton*0.001+cornernal*0.003), axis=1)
#result_stack = np.concatenate((result_stack, np.ones((Size[0],1,3), dtype = np.uint8)*255, img_label), axis=1)
print ("frame"+str(Index))
cv2.imshow("BB", result_stack)
cv2.waitKey(1)
if(Index<10):
        imgstr = '00'+str(Index)
elif(Index<100):
    imgstr = '0'+str(Index)
else:
    imgstr = str(Index)
cv2.imwrite('../boundingboxes/bb_'+ imgstr +'.png', bbrendering)
cv2.imwrite('../normal/nml_'+ imgstr +'.png', rendering)

#as prev RGBD
newRGBD = RGBD

# Then start again with new image, but init the tracker first
# initialize tracker for camera pose
Tracker = TrackManager.Tracker(0.001, 0.5, 1, [10])
formerIdx = Index
for bp in range(nbBdyPart+1):
    T_Pose.append(Id4)