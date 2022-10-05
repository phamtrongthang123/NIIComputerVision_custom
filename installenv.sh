#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
rm -rf env_nii/
conda create --prefix env_nii/ python=3.7 -y
conda activate env_nii/
conda install -y -c kayarre plyfile
conda install -y -c salilab imp
conda install -y mayavi
# # install opencl in case we haven't
# sudo apt install ocl-icd-opencl-dev
# sudo apt install ocl-icd-libopencl1
pip install Mako
pip install opencv-python==4.2.0.32 numpy scikit-learn scikit-image Pillow scipy pandas pyquaternion
# pip install pyopencl==2017.2
# for cpu 
pip install pyopencl[pocl] 