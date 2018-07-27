#!/usr/bin/env bash


CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 \
	   -gencode arch=compute_70,code=sm_70 "

echo 'setup coco eval ...'
cd datasets/eval/PythonAPI
make
cd ../../../

echo 'setup NMS ...'
cd lib/nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../..
python build.py
cd ../

echo 'setup RoI pooling ...'
cd lib/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../roi_pooling/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py
cd ../../
