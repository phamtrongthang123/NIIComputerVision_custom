#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:21:40 2017

@author: diegothomas
"""
Kernel_Test = """
__kernel void Test(__global float *TSDF) {

        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        int z = get_global_id(2); /*depth*/
        TSDF[x + 512*y + 512*512*z] = 1.0f;
        printf("Call KernelsOpenCL::Kernel_Test\\n");
}
"""
#__global float *prevTSDF, __global float *Weight
#__read_only image2d_t VMap
Kernel_FuseTSDF = """
__kernel void FuseTSDF(__global short int *TSDF,  __global float *Depth, __constant float *Param, __constant int *Dim,
                           __constant float *Pose,
                           __constant float *boneDQ,  __constant float *jointDQ, __constant float *planeF,
                           __constant float *calib, const int n_row, const int m_col, __global short int *Weight) {
        //const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

        const float nu = 0.05f;
        printf("Call KernelsOpenCL::Kernel_FuseTSDF\\n");


        float4 pt;
        float4 ctr;
        float4 pt_T;
        float4 ctr_T;
        int2 pix;

        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        pt.x = ((float)(x)-Param[0])/Param[1];
        pt.y = ((float)(y)-Param[2])/Param[3];
        float x_T =  Pose[0]*pt.x + Pose[1]*pt.y + Pose[3];
        float y_T =  Pose[4]*pt.x + Pose[5]*pt.y + Pose[7];
        float z_T =  Pose[8]*pt.x + Pose[9]*pt.y + Pose[11];
        float w_T =  Pose[12]*pt.x + Pose[13]*pt.y + Pose[15];

        float convVal = 32767.0f;
        int z ;
        for ( z = 0; z < Dim[2]; z++) { /*depth*/
            // On the GPU all pixel are stocked in a 1D array
            int idx = z + Dim[2]*y + Dim[2]*Dim[1]*x;

            // Transform voxel coordinates into 3D point coordinates
            // Param = [c_x, dim_x, c_y, dim_y, c_z, dim_z]
            pt.z = ((float)(z)-Param[4])/Param[5];

            // Transfom the voxel into the Image coordinate space
            //transform form local to global
            pt_T.x = x_T + Pose[2]*pt.z; //Pose is column major
            pt_T.y = y_T + Pose[6]*pt.z;
            pt_T.z = z_T + Pose[10]*pt.z;
            pt_T.w = w_T + Pose[14]*pt.z;
            //transform from first frame to current frame according interploation
            float weight, weightspara=0.02;
            weight = planeF[0]*pt_T.x+planeF[1]*pt_T.y+planeF[2]*pt_T.z+planeF[3];
            if(weight<=0){
                weight = 1;
            }else{
                weight = exp(-weight*weight/2/weightspara/weightspara);
            }
            float DQ[2][4];
            DQ[0][0] = (1.0-weight)*boneDQ[0*4+0];
            DQ[0][1] = (1.0-weight)*boneDQ[0*4+1];
            DQ[0][2] = (1.0-weight)*boneDQ[0*4+2];
            DQ[0][3] = (1.0-weight)*boneDQ[0*4+3];
            DQ[1][1] = (1.0-weight)*boneDQ[1*4+1];
            DQ[1][2] = (1.0-weight)*boneDQ[1*4+2];
            DQ[1][3] = (1.0-weight)*boneDQ[1*4+3];
            DQ[0][0] += (weight)*jointDQ[0*4+0];
            DQ[0][1] += (weight)*jointDQ[0*4+1];
            DQ[0][2] += (weight)*jointDQ[0*4+2];
            DQ[0][3] += (weight)*jointDQ[0*4+3];
            DQ[1][1] += (weight)*jointDQ[1*4+1];
            DQ[1][2] += (weight)*jointDQ[1*4+2];
            DQ[1][3] += (weight)*jointDQ[1*4+3];
            float mag = DQ[0][0]*DQ[0][0]+DQ[0][1]*DQ[0][1]+DQ[0][2]*DQ[0][2]+DQ[0][3]*DQ[0][3];
            mag = pow(mag, 0.5f);
            if(mag!=0){
                int d;
                DQ[0][0] = DQ[0][0]/mag;
                DQ[0][1] = DQ[0][1]/mag;
                DQ[0][2] = DQ[0][2]/mag;
                DQ[0][3] = DQ[0][3]/mag;
                DQ[1][1] = DQ[1][1]/mag;
                DQ[1][2] = DQ[1][2]/mag;
                DQ[1][3] = DQ[1][3]/mag;
            }else{
                DQ[0][0] = 0;
                DQ[0][1] = 0;
                DQ[0][2] = 0;
                DQ[0][3] = 0;
                DQ[1][1] = 0;
                DQ[1][2] = 0;
                DQ[1][3] = 0;
            }
            //get matrix from DQ
            float Tr[4][4];
            Tr[0][0] = DQ[0][0]*DQ[0][0]+DQ[0][1]*DQ[0][1]-DQ[0][2]*DQ[0][2]-DQ[0][3]*DQ[0][3];
            Tr[1][0] = 2.0*DQ[0][1]*DQ[0][2] + 2.0*DQ[0][0]*DQ[0][3];
            Tr[2][0] = 2.0*DQ[0][1]*DQ[0][3] - 2.0*DQ[0][0]*DQ[0][2];
            Tr[0][1] = 2.0*DQ[0][1]*DQ[0][2] - 2.0*DQ[0][0]*DQ[0][3];
            Tr[1][1] = DQ[0][0]*DQ[0][0]+DQ[0][2]*DQ[0][2]-DQ[0][1]*DQ[0][1]-DQ[0][3]*DQ[0][3];
            Tr[2][1] = 2.0*DQ[0][2]*DQ[0][3] + 2.0*DQ[0][0]*DQ[0][1];
            Tr[0][2] = 2.0*DQ[0][1]*DQ[0][3] + 2.0*DQ[0][0]*DQ[0][2];
            Tr[1][2] = 2.0*DQ[0][2]*DQ[0][3] - 2.0*DQ[0][0]*DQ[0][1];
            Tr[2][2] = DQ[0][0]*DQ[0][0]+DQ[0][3]*DQ[0][3]-DQ[0][1]*DQ[0][1]-DQ[0][2]*DQ[0][2];
            Tr[0][3] = -2.0*DQ[1][2]*DQ[0][3]+2.0*DQ[1][3]*DQ[0][2]+2.0*DQ[1][1]*DQ[0][0];
            Tr[1][3] = -2.0*DQ[1][3]*DQ[0][1]+2.0*DQ[1][1]*DQ[0][3]+2.0*DQ[1][2]*DQ[0][0];
            Tr[2][3] = -2.0*DQ[1][1]*DQ[0][2]+2.0*DQ[1][2]*DQ[0][1]+2.0*DQ[1][3]*DQ[0][0];
            Tr[3][0] = Tr[3][1] = Tr[3][2] = 0;
            Tr[3][3] = 1;
            pt = pt_T;
            pt.x =  Tr[0][0]*pt_T.x + Tr[0][1]*pt_T.y + Tr[0][2]*pt_T.z + Tr[0][3];
            pt.y =  Tr[1][0]*pt_T.x + Tr[1][1]*pt_T.y + Tr[1][2]*pt_T.z + Tr[1][3];
            pt.z =  Tr[2][0]*pt_T.x + Tr[2][1]*pt_T.y + Tr[2][2]*pt_T.z + Tr[2][3];
            pt.w =  Tr[3][0]*pt_T.x + Tr[3][1]*pt_T.y + Tr[3][2]*pt_T.z + Tr[3][3];
            pt_T.x = pt.x/pt.w;
            pt_T.y = pt.y/pt.w;
            pt_T.z = pt.z/pt.w;
            pt_T.w = pt.w/pt.w;

            // Project onto Image
            pix.x = convert_int(round((pt_T.x/fabs(pt_T.z))*calib[0] + calib[2]));
            pix.y = convert_int(round((pt_T.y/fabs(pt_T.z))*calib[4] + calib[5]));

            // Check if the pixel is in the frame
            if (pix.x < 0 || pix.x > m_col-1 || pix.y < 0 || pix.y > n_row-1){
                if (Weight[idx] == 0)
                    TSDF[idx] = (short int)(convVal);
                continue;
            }

            //Compute distance between project voxel and surface in the RGBD image
            float dist = -(pt_T.z - Depth[pix.x + m_col*pix.y])/nu;
            dist = min(1.0f, max(-1.0f, dist));
            if (Depth[pix.x + m_col*pix.y] == 0) {
                if (Weight[idx] == 0)
                    TSDF[idx] = (short int)(convVal);
                continue;
            }

            float w = 1.0f;
            if (dist<(float)(TSDF[idx])/convVal) w=0.1f;
            if (dist > 1.0f) dist = 1.0f;
            else dist = max(-1.0f, dist);


            // Running Average
            float prev_tsdf = (float)(TSDF[idx])/convVal;
            float prev_weight = (float)(Weight[idx]);

            TSDF[idx] =  (short int)(round(((prev_tsdf*prev_weight+dist*w)/(prev_weight+w))*convVal));
            Weight[idx] = min(1000, Weight[idx]+1);
         }

}
"""



