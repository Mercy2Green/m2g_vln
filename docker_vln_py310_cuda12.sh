#! /bin/bash

# export MATTERPORT_DATA_DIR=/media/m2g/Data/Datasets/dataset/v1/unzipped
# export MATTERPORT_SIMULATOR_DIR=/media/m2g/Data/Datasets/m2g_vln/Matterport3DSimulator_opencv4
# export BEVBERT_DIR=/media/m2g/Data/Datasets/m2g_vln/VLN-BEVBert


export GSA_DIR="/home/lg1/peteryu_workspace/m2g_vln/Grounded-Segment-Anything"
export TAG2TEXT_PATH="/home/lg1/peteryu_workspace/m2g_vln/recognize-anything"
export CG_FOLDER="/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs"
export VLN_HGT_DIR="/home/lg1/lujia/VLN_HGT"

export CG_HDF5_DIR=/data0/vln_datasets/preprocessed_data/finetune_cg_hdf5
export EDGE_HDF5_DIR=/data0/vln_datasets/preprocessed_data

export PLACEANDROOM_DIR=$VLN_HGT_DIR/PlacesAndRoom
export PRETRAINED_PATH=$VLN_HGT_DIR/pretrained/HGT/cg_frame

export TENSORBOARD_DIR=$VLN_HGT_DIR/data/logs/hgt_cg_frame/tensorboard_dirs
export CHECKPOINT_FOLDER=/data1/vln_data/finetune/checkpoint/hgt_cg_frame
export EVAL_CKPT_PATH_DIR=$VLN_HGT_DIR/data/logs/hgt_cg_frame/checkpoints
export RESULTS_DIR=$VLN_HGT_DIR/logs/hgt_cg_frame/eval_results
export CONNECTIVITY_DIR=/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity/

export NVIDIA_DRIVER_DIR=/home/lg1/peteryu_workspace/NVIDIA-Linux-x86_64-550.107.02

# docker run -ti --gpus 1 \
#    --mount type=bind,source=$GSA_DIR,target=/root/mount/Grounded-Segment-Anything \
#    --mount type=bind,source=$CG_FOLDER,target=/root/mount/concept-graphs \
#    --mount type=bind,source=$TAG2TEXT_PATH,target=/root/mount/recognize-anything \
#    --mount type=bind,source=$VLN_HGT_DIR,target=/root/mount/VLN-HGT \
#    --mount type=bind,source=$PLACEANDROOM_DIR,target=/root/mount/VLN-HGT/PlacesAndRoom \
#    --mount type=bind,source=$PRETRAINED_PATH,target=/root/mount/VLN-HGT/pretrained/HGT/cg_frame \
#    --mount type=bind,source=$TENSORBOARD_DIR,target=/root/mount/VLN-HGT/data/logs/hgt_cg_frame/tensorboard_dirs \
#    --mount type=bind,source=$CHECKPOINT_FOLDER,target=/data1/vln_data/finetune/checkpoint/hgt_cg_frame \
#    --mount type=bind,source=$EVAL_CKPT_PATH_DIR,target=/root/mount/VLN-HGT/data/logs/hgt_cg_frame/checkpoints \
#    --mount type=bind,source=$RESULTS_DIR,target=/root/mount/VLN-HGT/logs/hgt_cg_frame/eval_results \
#     vln:12.4.1-cudnn-devel-ubuntu20.04  

docker run -ti --gpus 1\
   --mount type=bind,source=$GSA_DIR,target=$GSA_DIR \
   --mount type=bind,source=$CG_FOLDER,target=$CG_FOLDER \
   --mount type=bind,source=$TAG2TEXT_PATH,target=$TAG2TEXT_PATH \
   --mount type=bind,source=$VLN_HGT_DIR,target=$VLN_HGT_DIR \
   --mount type=bind,source=$PLACEANDROOM_DIR,target=$PLACEANDROOM_DIR \
   --mount type=bind,source=$PRETRAINED_PATH,target=$PRETRAINED_PATH \
   --mount type=bind,source=$CHECKPOINT_FOLDER,target=$CHECKPOINT_FOLDER \
   --mount type=bind,source=$CG_HDF5_DIR,target=$CG_HDF5_DIR \
   --mount type=bind,source=$EDGE_HDF5_DIR,target=$EDGE_HDF5_DIR \
   --mount type=bind,source=$CONNECTIVITY_DIR,target=$CONNECTIVITY_DIR \
   --mount type=bind,source=$NVIDIA_DRIVER_DIR,target=$NVIDIA_DRIVER_DIR \
      vln:12.4.1-cudnn-devel-ubuntu20.04

# ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build

# # ENV GSA_PATH=/root/mount/Grounded-Segment-Anything
# ENV TAG2TEXT_PATH=/root/mount/recognize-anything
# ENV CG_FOLDER=/root/mount/concept-graphs

    # --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans \
    # --mount type=bind,source=$MATTERPORT_SIMULATOR_DIR,target=/root/mount/Matterport3DSimulator \
    # --mount type=bind,source=$BEVBERT_DIR,target=/root/mount/VLN-BEVBert \

# CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash train 2122