#! /bin/bash

export MATTERPORT_DATA_DIR=/data/vln_datasets/matterport_skybox/v1/unzipped

export MATTERPORT_SIMULATOR_DIR=/home/lg1/peteryu_workspace/m2g_vln/Matterport3DSimulator_opencv4

export BEVBERT_DIR=/home/lg1/peteryu_workspace/m2g_vln/VLN-BEVBert

# Habitat dataset for BEV-BERT

export HABITAT_DATA_DIR=/data/vln_datasets/mp3d/v1/tasks/mp3d

# local model path

export LOCAL_MODEL_DIR=/home/lg1/peteryu_workspace/model

# export MATTERPORT_SIMULATOR_DIR=/media/m2g/Data/Datasets/m2g_vln/Matterport3DSimulator
# export MATTERPORT_SIMULATOR_OPENCV4_DIR=/media/m2g/Data/Datasets/m2g_vln/Matterport3DSimulator_opencv4

docker run -ti --gpus '"device=6,7"' \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans \
    --mount type=bind,source=$MATTERPORT_SIMULATOR_DIR,target=/root/mount/Matterport3DSimulator \
    --mount type=bind,source=$BEVBERT_DIR,target=/root/mount/VLN-BEVBert \
    --mount type=bind,source=$HABITAT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/scene_datasets/mp3d \
    --mount type=bind,source=$LOCAL_MODEL_DIR,target=/root/mount/Model \
    mattersim:11.3.0-devel-ubuntu20.04

#    mattersim:9.2-devel-ubuntu18.04
# python precompute_features/grid_mp3d_clip.py --scan_dir=/root/mount/Matterport3DSimulator/data/v1/scans

# mattersim:11.8.0-cudnn8-devel-ubuntu20.04

# docker run --gpus all -it\
#     --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans \
#     --mount type=bind,source=$MATTERPORT_SIMULATOR_DIR,target=/root/mount/Matterport3DSimulator \
#     --mount type=bind,source=$BEVBERT_DIR,target=/root/mount/VLN-BEVBert \
#     mattersim:11.8.0-cudnn8-devel-ubuntu20.04